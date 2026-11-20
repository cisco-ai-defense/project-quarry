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

"""Validate and compile rule packs without loading runtime providers."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from quarry.dsl.parser import DSLParser
from quarry.engine.compiler import compile_rule
from quarry.safety.validator import RuleValidator

_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_PACK_BYTES = 64 * 1024 * 1024
_PACK_PATH = PurePosixPath("rules/rule-packs")
_SCHEMA_PATH = _PACK_PATH / "metadata-schema.yaml"
_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STEM = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")
_PACK_DEPTH = {"core": 3, "community": 4}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    compiled_rules: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


class _MetadataLoader(yaml.SafeLoader):
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


def _read_text(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("expected a regular file")
    with path.open("rb") as stream:
        data = stream.read(_MAX_FILE_BYTES + 1)
    if len(data) > _MAX_FILE_BYTES:
        raise ValueError(f"file exceeds {_MAX_FILE_BYTES} bytes")
    if b"\x00" in data:
        raise ValueError("expected UTF-8 text")
    return data.decode("utf-8")


def _load_schema(path: Path) -> dict:
    schema = yaml.load(_read_text(path), Loader=_MetadataLoader)
    required = schema.get("required") if isinstance(schema, dict) else None
    if not isinstance(required, list) or not all(isinstance(key, str) for key in required):
        raise ValueError("metadata schema must define required string fields")
    return schema


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


def _validate_rule(
    path: Path,
    identity: str,
    schema: dict,
    names: set[str],
    report: ValidationReport,
) -> None:
    source = _read_text(path)
    rule = DSLParser().parse(source, source=str(path))
    _check_metadata(rule.metadata, schema)
    if rule.name in names:
        raise ValueError(f"duplicate declared rule name: {rule.name}")
    names.add(rule.name)
    if rule.name != identity:
        report.warnings.append(f"{identity}: recommended declared rule name is {identity!r}")
    safety = RuleValidator().validate(rule)
    if safety.regex_errors or safety.semantic_errors:
        raise ValueError(f"rule validation: {safety.regex_errors + safety.semantic_errors}")
    report.warnings.extend(f"{identity}: {warning}" for warning in safety.warnings)
    compiled = compile_rule(rule)
    if not compiled.ok or compiled.artifact is None:
        errors = "; ".join(str(item.message) for item in compiled.diagnostics.errors)
        raise ValueError(f"compile failed: {errors}")
    report.compiled_rules += 1


def _pack_files(root: Path, packs_dir: Path, report: ValidationReport) -> list[Path]:
    for directory in (packs_dir.parent, packs_dir):
        if directory.is_symlink() or not directory.is_dir():
            report.errors.append(f"{directory.relative_to(root)}: expected a non-symlink directory")
            return []
    files = []
    for path in sorted(packs_dir.rglob("*")):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            report.errors.append(f"{path.relative_to(root)}: expected a regular file or directory")
        elif path.is_file():
            files.append(path)
    return files


def _allow_text_file(path: Path, packs_dir: Path, rule_dirs: set[Path]) -> bool:
    parts = path.relative_to(packs_dir).parts
    depth = _PACK_DEPTH.get(parts[0], 0)
    if path.name == "README.md" and (len(parts) == 1 or len(parts) <= depth):
        return True
    if path.suffix not in {".md", ".txt"}:
        return False
    return path.parent in rule_dirs or any(
        parent.name == "resources" and parent.parent in rule_dirs for parent in path.parents
    )


def _check_text_files(
    root: Path, packs_dir: Path, files: list[Path], rule_dirs: set[Path], report: ValidationReport
) -> None:
    readmes = {directory / "README.md" for directory in rule_dirs}
    for path in sorted(readmes.difference(files)):
        report.errors.append(f"{path.relative_to(root)}: rule README is required")
    for path in files:
        if path.suffix == ".rule" or path == packs_dir / "metadata-schema.yaml":
            continue
        label = path.relative_to(root)
        if not _allow_text_file(path, packs_dir, rule_dirs):
            report.errors.append(
                f"{label}: only rule files, READMEs and text rule resources are supported"
            )
            continue
        try:
            source = _read_text(path).lstrip()
            if path in readmes:
                if source.startswith("<!--"):
                    _, _, source = source.partition("-->")
                if not source.strip():
                    raise ValueError("rule README must include an overview")
        except (OSError, ValueError) as exc:
            report.errors.append(f"{label}: {exc}")


def validate_repository(repo_root: Path) -> ValidationReport:
    """Check the on-disk pack layout, metadata and rule compilation."""
    root = Path(repo_root).resolve()
    report = ValidationReport()
    packs_dir = root / _PACK_PATH
    files = _pack_files(root, packs_dir, report)
    if report.errors:
        return report
    try:
        schema = _load_schema(packs_dir / "metadata-schema.yaml")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        report.errors.append(f"metadata-schema.yaml: {exc}")
        return report
    names: set[str] = set()
    rule_dirs: set[Path] = set()
    rules = [path for path in files if path.suffix == ".rule"]
    if not rules:
        report.errors.append("no rule files found in rules/rule-packs")
    for path in rules:
        try:
            identity = _rule_identity(path, packs_dir, report)
            rule_dirs.add(path.parent)
            _validate_rule(path, identity, schema, names, report)
        except Exception as exc:
            report.errors.append(f"{path.relative_to(root)}: {exc}")
    _check_text_files(root, packs_dir, files, rule_dirs, report)
    return report


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


def validate_proposed_packs(trusted_root: Path, proposed_commit: str) -> ValidationReport:
    """Validate a proposed Git tree as data using the checkout's trusted schema."""
    if re.fullmatch(r"[0-9a-f]{40}", proposed_commit) is None:
        raise ValueError("expected a full Git commit SHA")
    entries = _git(
        trusted_root, "ls-tree", "-rlz", "--full-tree", proposed_commit, "--", _PACK_PATH.as_posix()
    )

    with TemporaryDirectory(prefix="quarry-proposed-packs-") as directory:
        staged = Path(directory)
        (staged / _PACK_PATH).mkdir(parents=True)
        total_bytes = 0
        has_schema = False
        for entry in entries.split(b"\x00"):
            if not entry:
                continue
            header, raw_path = entry.split(b"\t", 1)
            mode, kind, oid, size = header.split()
            if mode != b"100644" or kind != b"blob":
                raise ValueError("rule pack entries must be regular non-executable files")
            name = raw_path.decode("utf-8")
            path = PurePosixPath(name)
            if (
                path == _PACK_PATH
                or not path.is_relative_to(_PACK_PATH)
                or ".." in path.parts
                or "\\" in name
                or ":" in name
                or path.as_posix() != name
            ):
                raise ValueError("invalid rule pack path")
            if path == _SCHEMA_PATH:
                has_schema = True
                continue
            length = int(size)
            total_bytes += length
            if length > _MAX_FILE_BYTES or total_bytes > _MAX_PACK_BYTES:
                raise ValueError("rule pack input exceeds the file or pack size limit")
            destination = staged / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(_git(trusted_root, "cat-file", "blob", oid.decode("ascii")))

        if not has_schema:
            raise ValueError("proposed rule packs must retain metadata-schema.yaml")
        (staged / _SCHEMA_PATH).write_text(
            _read_text(trusted_root / _SCHEMA_PATH), encoding="utf-8"
        )
        return validate_repository(staged)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1], help="repository root"
    )
    parser.add_argument(
        "--proposed-commit", help="validate only rule pack data from a fetched Git commit"
    )
    args = parser.parse_args(argv)
    if args.proposed_commit is not None:
        try:
            report = validate_proposed_packs(args.root, args.proposed_commit)
        except (OSError, ValueError, subprocess.CalledProcessError):
            print("FAIL: could not read the proposed rule pack files")
            return 1
        # Compiler diagnostics can contain submitted text. Keep it out of CI logs.
        print(
            f"{'PASS' if report.ok else 'FAIL'}: {report.compiled_rules} rules compiled, "
            f"{len(report.errors)} errors, {len(report.warnings)} warnings"
        )
        if not report.ok:
            print("Run the rule pack validator locally to see the diagnostics.")
        return 0 if report.ok else 1

    report = validate_repository(args.root)
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    print(f"{'PASS' if report.ok else 'FAIL'}: {report.compiled_rules} rules compiled")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
