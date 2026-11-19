# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate the reviewed rule corpus without initializing external resources."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from quarry.config import QuarryConfig
from quarry.dsl.parser import DSLParser
from quarry.engine.compiler import compile_rule
from quarry.integrations.resources import requirements_from_artifact
from quarry.safety.validator import RuleValidator, validate_conversation
from quarry.scanner import DSLScanner

_MAX_FILE_BYTES = 2 * 1024 * 1024
_DOCS = {"README.md"}
_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STEM = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


@dataclass
class ValidationReport:
    """Counts deliberately distinguish compilation from evaluated fixtures."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    compiled_rules: int = 0
    passed_cases: int = 0
    unevaluated_cases: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


class _CorpusLoader(yaml.SafeLoader):
    """Accept plain YAML data, rejecting aliases and duplicate mapping keys."""

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._depth = 0

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.AliasEvent):
            raise ValueError("YAML aliases are not supported")
        if self._depth >= 50:
            raise ValueError("YAML nesting exceeds 50 levels")
        self._depth += 1
        try:
            return super().compose_node(parent, index)
        finally:
            self._depth -= 1

    def construct_mapping(self, node: Any, deep: bool = False) -> dict:
        pairs = self.construct_pairs(node, deep=deep)
        return _unique_mapping(pairs)


def _unique_mapping(pairs: list[tuple[Any, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key!r}")
        result[key] = value
    return result


def _relative_file(root: Path, base: Path, name: str) -> Path:
    """Allow only canonical relative paths contained by the requested directory."""
    relative = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or relative.is_absolute()
        or any(part in {".", ".."} for part in relative.parts)
        or relative.as_posix() != name
    ):
        raise ValueError(f"invalid relative path: {name!r}")
    path = base.joinpath(*relative.parts)
    for ancestor in (path, *path.parents):
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise ValueError(f"symlink is not allowed: {path.relative_to(root)}")
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"path escapes corpus: {name!r}")
    return path


def _read(root: Path, path: Path) -> str:
    path = _relative_file(root, root, path.relative_to(root).as_posix())
    with path.open("rb") as stream:
        data = stream.read(_MAX_FILE_BYTES + 1)
    if len(data) > _MAX_FILE_BYTES:
        raise ValueError(f"file exceeds {_MAX_FILE_BYTES} bytes")
    return data.decode("utf-8")


def _load(root: Path, path: Path) -> Any:
    text = _read(root, path)
    if path.suffix == ".json":
        return json.loads(text, object_pairs_hook=_unique_mapping)
    return yaml.load(text, Loader=_CorpusLoader)


def _inventory(root: Path, directory: Path, report: ValidationReport) -> list[Path]:
    files = []
    if directory.is_symlink():
        report.errors.append(f"{directory.relative_to(root)}: symlink is not allowed")
        return files
    if not directory.is_dir():
        report.errors.append(f"missing corpus directory: {directory.relative_to(root)}")
        return files
    for path in sorted(directory.rglob("*")):
        if "__pycache__" in path.parts or path.name == ".DS_Store":
            continue
        if path.is_symlink():
            report.errors.append(f"{path.relative_to(root)}: symlink is not allowed")
        elif path.is_file() and path.name not in _DOCS and path.suffix != ".pyc":
            files.append(path)
    return files


def _check_manifest(root: Path, packs: list[Path], cases: list[Path]) -> None:
    path = root / "rules/test-cases/corpus-manifest.json"
    actual = {
        "rules": {p.relative_to(root).as_posix() for p in packs if p.suffix == ".rule"},
        "resources": {p.relative_to(root).as_posix() for p in packs if p.suffix != ".rule"},
        "test_cases": {p.relative_to(root).as_posix() for p in cases if p != path},
    }
    manifest = _load(root, path)
    if not isinstance(manifest, dict) or set(manifest) != set(actual):
        raise ValueError("manifest must contain rules, resources, and test_cases lists")
    issues = []
    for category, discovered in actual.items():
        entries = manifest[category]
        if not isinstance(entries, list) or not all(isinstance(item, str) for item in entries):
            raise ValueError(f"manifest {category} must be a list of relative paths")
        for entry in entries:
            _relative_file(root, root, entry)
        if len(set(entries)) != len(entries):
            issues.append(f"duplicate inventory entries in {category}")
        issues.extend(
            f"not inventoried in {category}: {p}" for p in sorted(discovered - set(entries))
        )
        issues.extend(
            f"missing or misclassified in {category}: {p}"
            for p in sorted(set(entries) - discovered)
        )
    if issues:
        raise ValueError("; ".join(issues))


def _check_metadata(metadata: dict, schema: dict) -> None:
    """Apply the string-field contract in metadata-schema.yaml (no remote refs)."""
    for key in schema["required"]:
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"required metadata {key!r} must be a nonempty string")
    for key, value in metadata.items():
        if not isinstance(value, str):
            raise ValueError(f"metadata {key!r} must be a string")
        contract = schema.get("properties", {}).get(key, {})
        if len(value) < contract.get("minLength", 0):
            raise ValueError(f"metadata {key!r} is too short")
        if "enum" in contract and value not in contract["enum"]:
            raise ValueError(f"metadata {key!r} must be one of {contract['enum']}")
        if "pattern" in contract and re.fullmatch(contract["pattern"], value) is None:
            raise ValueError(f"metadata {key!r} has invalid format: {value!r}")


def _rule_identity(path: Path, packs_dir: Path, report: ValidationReport) -> str:
    parts = path.relative_to(packs_dir).parts
    valid = len(parts) == 4 and parts[0] == "core" or len(parts) == 5 and parts[0] == "community"
    if not valid or parts[-1] != f"{parts[-2]}.rule":
        raise ValueError(
            "expected core/<pack>/<stem>/<stem>.rule or community/<publisher>/<pack>/<stem>/<stem>.rule"
        )
    if any(not _SLUG.fullmatch(part) for part in parts[1:-2]) or not _STEM.fullmatch(parts[-2]):
        report.warnings.append(
            f"{path.name}: use kebab-case pack/publisher names and snake_case rule stems"
        )
    return "/".join(parts[:-1])


def _expectations(root: Path, directory: Path) -> tuple[list[tuple[Path, dict]], dict[str, str]]:
    data = _load(root, directory / "expectations.yaml")
    if not isinstance(data, dict) or not isinstance(data.get("cases"), dict):
        raise ValueError("expectations.yaml needs a cases mapping")
    expected = data["cases"]
    discovered = {
        path.relative_to(directory).as_posix()
        for label in ("positive", "negative")
        for path in (directory / label).glob("*.json")
    }
    for label in ("positive", "negative"):
        if not any(name.startswith(f"{label}/") for name in discovered):
            raise ValueError(f"at least one {label}/ JSON case is required")
    cases = []
    for name, expectation in expected.items():
        if not isinstance(name, str):
            raise ValueError("case paths must be strings")
        path = _relative_file(root, directory, name)
        if name not in discovered:
            raise ValueError(f"case is missing or outside positive/negative: {name}")
        if not isinstance(expectation, dict) or type(expectation.get("match")) is not bool:
            raise ValueError(f"{name}: expectation needs a boolean match")
        if expectation["match"] != name.startswith("positive/"):
            raise ValueError(f"{name}: match must agree with positive/negative directory")
        if "evidence" in expectation and type(expectation["evidence"]) is not bool:
            raise ValueError(f"{name}: evidence must be a boolean")
        cases.append((path, expectation))
    if set(expected) != discovered:
        raise ValueError(f"cases missing expectations: {sorted(discovered - set(expected))}")
    compile_options = data.get("compile", {})
    if not isinstance(compile_options, dict):
        raise ValueError("compile must be a mapping")
    schemas = compile_options.get("response_schemas", {})
    if not isinstance(schemas, dict) or not all(
        isinstance(alias, str) and isinstance(schema, (dict, str))
        for alias, schema in schemas.items()
    ):
        raise ValueError("compile.response_schemas must map model aliases to JSON schemas")
    return cases, {
        alias: json.dumps(schema) if isinstance(schema, dict) else schema
        for alias, schema in schemas.items()
    }


def _validate_rule(
    root: Path,
    path: Path,
    identity: str,
    schema: dict,
    names: set[str],
    compile_only: bool,
    report: ValidationReport,
) -> set[Path]:
    source = _read(root, path)
    rule = DSLParser().parse(source, source=str(path))
    _check_metadata(rule.metadata, schema)
    if rule.name in names:
        raise ValueError(f"duplicate declared rule name: {rule.name}")
    names.add(rule.name)
    if rule.name != identity:
        report.warnings.append(f"{identity}: recommended declared rule name is {identity!r}")
    deprecated = rule.metadata.get("status") == "deprecated"
    if deprecated:
        report.warnings.append(
            f"{identity}: deprecated; document its reason and last supported version in the contribution history"
        )
        if not rule.metadata.get("deprecation_reason"):
            report.warnings.append(f"{identity}: add deprecation_reason metadata")
    safety = RuleValidator().validate(rule)
    if safety.regex_errors or safety.semantic_errors:
        raise ValueError(f"rule validation: {safety.regex_errors + safety.semantic_errors}")
    report.warnings.extend(f"{identity}: {warning}" for warning in safety.warnings)
    directory = root / "rules/test-cases" / identity
    cases, schemas = _expectations(root, directory)
    consumed = {directory / "expectations.yaml", *(case[0] for case in cases)}
    compiled = compile_rule(rule, response_schemas=schemas)
    if not compiled.ok or compiled.artifact is None:
        errors = "; ".join(str(item.message) for item in compiled.diagnostics.errors)
        hint = (
            "; update or remove the unsupported rule and its cases together, recording the last supported commit in the contribution history"
            if deprecated
            else ""
        )
        raise ValueError(f"compile failed: {errors}{hint}")
    report.compiled_rules += 1
    artifact = compiled.artifact
    requirements = requirements_from_artifact(artifact)
    collect_plan = artifact.collect_plan
    external = (
        bool(rule.parameters)
        or bool(requirements["llm_models"] or requirements["text_resources"])
        or any(spec.predicate == "external" for spec in artifact.atom_specs.values())
        or collect_plan is not None
        and getattr(collect_plan, "has_external_work", True)
    )
    skip = compile_only or external
    scanner = None
    if skip:
        report.warnings.append(
            f"{identity}: compile-only; {len(cases)} cases not evaluated ({'requested' if compile_only else 'runtime resources or parameter bindings required'})"
        )
    else:
        # No contributed configuration, imports, plugins, providers, or text resources.
        scanner = DSLScanner(QuarryConfig(quarry_rule_texts=(source,), config_version=2))
    for case_path, expectation in cases:
        label = case_path.relative_to(root)
        try:
            payload = _load(root, case_path)
            validate_conversation(payload)
            if scanner is None:
                report.unevaluated_cases += 1
                continue
            result = scanner.scan_data(payload, enable_explain=True)
            if not result.is_complete:
                raise ValueError(f"incomplete scan: {result.completion_issues}")
            matched = rule.name in result.rule_matches
            if matched != expectation["match"]:
                raise ValueError(f"expected match={expectation['match']}, got {matched}")
            if expectation.get("evidence") and not result.detailed_matches:
                raise ValueError("expected match evidence is missing")
            report.passed_cases += 1
        except Exception as exc:
            report.errors.append(f"{label}: {exc}")
    return consumed


def validate_repository(repo_root: Path, *, compile_only: bool = False) -> ValidationReport:
    """Compile every rule and execute only self-contained local fixtures."""
    root = Path(repo_root).resolve()
    report = ValidationReport()
    packs_dir = root / "rules/rule-packs"
    cases_dir = root / "rules/test-cases"
    packs = _inventory(root, packs_dir, report)
    cases = _inventory(root, cases_dir, report)
    try:
        _check_manifest(root, packs, cases)
    except Exception as exc:
        report.errors.append(f"corpus-manifest.json: {exc}")
    try:
        schema = _load(root, packs_dir / "metadata-schema.yaml")
        if not isinstance(schema, dict) or not isinstance(schema.get("required"), list):
            raise ValueError("metadata schema must define required string fields")
    except Exception as exc:
        report.errors.append(f"metadata-schema.yaml: {exc}")
        return report
    names: set[str] = set()
    consumed = {cases_dir / "corpus-manifest.json"}
    rules = [path for path in packs if path.suffix == ".rule"]
    if not rules:
        report.errors.append("no rule files found in rules/rule-packs")
    for path in rules:
        identity = None
        try:
            identity = _rule_identity(path, packs_dir, report)
            consumed.update(
                _validate_rule(root, path, identity, schema, names, compile_only, report)
            )
        except Exception as exc:
            report.errors.append(f"{path.relative_to(root)}: {exc}")
            # Existing cases still belong to this failed rule; avoid orphan noise.
            if identity is not None:
                consumed.update((cases_dir / identity).rglob("*"))
    for orphan in sorted(set(cases) - consumed):
        report.errors.append(
            f"{orphan.relative_to(root)}: case file has no corresponding rule/expectation"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1], help="repository root"
    )
    parser.add_argument(
        "--compile-only", action="store_true", help="validate and compile without evaluating cases"
    )
    args = parser.parse_args(argv)
    report = validate_repository(args.root, compile_only=args.compile_only)
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    print(
        f"{'PASS' if report.ok else 'FAIL'}: {report.compiled_rules} rules compiled; {report.passed_cases} cases passed; {report.unevaluated_cases} cases not evaluated"
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
