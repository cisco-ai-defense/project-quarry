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

"""Repository contribution validation, including its offline execution boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.scanner import DSLScanner
from rules.validate import main, validate_repository

_MARKER = "QUARRY_FIXTURE_MATCH"
_REQUIRED_METADATA = {
    "description": "Detect an example fixture marker.",
    "author": "Quarry contributors",
    "version": "0.1.0",
    "status": "experimental",
    "requires_quarry": ">=0.1.0,<0.2.0",
}


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_yaml(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _manifest(repo: Path) -> None:
    packs = repo / "rules/rule-packs"
    cases = repo / "rules/test-cases"
    inventory: dict[str, list[str]] = {"rules": [], "resources": [], "test_cases": []}
    for path in sorted(packs.rglob("*")):
        if path.is_file() and path.name != "README.md":
            category = "rules" if path.suffix == ".rule" else "resources"
            inventory[category].append(path.relative_to(repo).as_posix())
    for path in sorted(cases.rglob("*")):
        if path.is_file() and path.name not in {"README.md", "corpus-manifest.json"}:
            inventory["test_cases"].append(path.relative_to(repo).as_posix())
    _write_json(cases / "corpus-manifest.json", inventory)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": list(_REQUIRED_METADATA),
        "properties": {key: {"type": "string", "minLength": 1} for key in _REQUIRED_METADATA},
        "additionalProperties": {"type": "string"},
    }
    _write_yaml(tmp_path / "rules/rule-packs/metadata-schema.yaml", schema)
    return tmp_path


def _add_rule(
    repo: Path,
    *,
    pack: str = "core/fixture-pack",
    stem: str = "indirect_injection_marker",
    name: str | None = None,
    condition: str | None = None,
    metadata: dict[str, str] | None = None,
    parameters: str = "",
) -> tuple[Path, Path]:
    values = _REQUIRED_METADATA | (metadata or {})
    rule_path = repo / "rules/rule-packs" / pack / stem / f"{stem}.rule"
    rule_path.parent.mkdir(parents=True, exist_ok=True)
    rule_path.write_text(
        f"rule {json.dumps(name or f'{pack}/{stem}')}\nmetadata:\n"
        + "".join(f"    {key} = {json.dumps(value)}\n" for key, value in values.items())
        + (f"parameters:\n{parameters}\n" if parameters else "")
        + "condition:\n    "
        + (condition or f'any msg in user_messages where msg.content contains "{_MARKER}"')
        + "\n",
        encoding="utf-8",
    )
    case_dir = repo / "rules/test-cases" / pack / stem
    _write_json(
        case_dir / "positive/matches.json",
        {"messages": [{"role": "user", "content": _MARKER}]},
    )
    _write_json(
        case_dir / "negative/does_not_match.json",
        {"messages": [{"role": "user", "content": "ordinary example request"}]},
    )
    _write_yaml(
        case_dir / "expectations.yaml",
        {
            "cases": {
                "positive/matches.json": {"match": True},
                "negative/does_not_match.json": {"match": False},
            }
        },
    )
    _manifest(repo)
    return rule_path, case_dir


def test_valid_core_pack_compiles_and_checks_positive_and_negative_cases(corpus: Path) -> None:
    _add_rule(corpus)

    report = validate_repository(corpus)

    assert report.ok, report.errors
    assert report.compiled_rules == 1
    assert report.passed_cases == 2
    assert report.unevaluated_cases == 0


@pytest.mark.parametrize(
    ("case", "content"),
    [("positive/matches.json", "ordinary request"), ("negative/does_not_match.json", _MARKER)],
)
def test_fixture_verdict_mismatch_fails(corpus: Path, case: str, content: str) -> None:
    _, cases = _add_rule(corpus)
    _write_json(cases / case, {"messages": [{"role": "user", "content": content}]})

    report = validate_repository(corpus)

    assert not report.ok
    assert report.passed_cases == 1
    assert any(Path(case).name in error for error in report.errors)


@pytest.mark.parametrize("payload", [{"messages": "not-a-message-list"}, {"tools": []}])
def test_invalid_input_cannot_pass_as_a_clean_negative(corpus: Path, payload: object) -> None:
    _, cases = _add_rule(corpus)
    _write_json(cases / "negative/does_not_match.json", payload)

    report = validate_repository(corpus)

    assert not report.ok
    assert report.passed_cases == 1
    assert any("does_not_match.json" in error for error in report.errors)


def test_incomplete_scans_cannot_pass_as_clean_fixture_verdicts(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_rule(corpus)
    monkeypatch.setattr(
        DSLScanner,
        "scan_data",
        lambda *args, **kwargs: SimpleNamespace(
            is_complete=False, completion_issues=["example evaluation timeout"], rule_matches=[]
        ),
    )

    report = validate_repository(corpus)

    assert not report.ok
    assert report.passed_cases == 0
    assert any("incomplete" in error.lower() for error in report.errors)


def test_positive_fixture_can_require_match_evidence(corpus: Path) -> None:
    _, cases = _add_rule(corpus)
    expectations = yaml.safe_load((cases / "expectations.yaml").read_text(encoding="utf-8"))
    expectations["cases"]["positive/matches.json"]["evidence"] = True
    _write_yaml(cases / "expectations.yaml", expectations)

    report = validate_repository(corpus)

    assert report.ok, report.errors
    assert report.passed_cases == 2


def test_manifest_must_include_new_resource(corpus: Path) -> None:
    rule, _ = _add_rule(corpus)
    (rule.parent / "prompt.txt").write_text("Example policy text.", encoding="utf-8")

    report = validate_repository(corpus)

    assert not report.ok
    assert any("prompt.txt" in error for error in report.errors)


def test_manifest_must_drop_removed_file(corpus: Path) -> None:
    rule, _ = _add_rule(corpus)
    resource = rule.parent / "prompt.txt"
    resource.write_text("Example policy text.", encoding="utf-8")
    _manifest(corpus)
    resource.unlink()

    report = validate_repository(corpus)

    assert not report.ok
    assert any("prompt.txt" in error for error in report.errors)


def test_rule_removal_must_remove_its_fixture_directory(corpus: Path) -> None:
    rule, _ = _add_rule(corpus)
    rule.unlink()
    _manifest(corpus)

    report = validate_repository(corpus)

    assert not report.ok
    assert any("indirect_injection_marker" in error for error in report.errors)


@pytest.mark.parametrize("unsafe_path", ["../outside.rule", "/tmp/outside.rule"])
def test_manifest_rejects_paths_outside_repository(corpus: Path, unsafe_path: str) -> None:
    _add_rule(corpus)
    path = corpus / "rules/test-cases/corpus-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["rules"].append(unsafe_path)
    _write_json(path, manifest)

    report = validate_repository(corpus)

    assert not report.ok
    assert any(unsafe_path in error for error in report.errors)


def test_symlink_to_rule_outside_repository_is_rejected(corpus: Path, tmp_path: Path) -> None:
    rule, _ = _add_rule(corpus)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.rule"
    outside.write_text(rule.read_text(encoding="utf-8"), encoding="utf-8")
    rule.unlink()
    rule.symlink_to(outside)

    report = validate_repository(corpus)

    assert not report.ok
    assert any(rule.name in error for error in report.errors)


def test_duplicate_declared_names_fail_across_pack_namespaces(corpus: Path) -> None:
    _add_rule(corpus, name="shared declared name")
    _add_rule(corpus, pack="core/another-pack", name="shared declared name")

    report = validate_repository(corpus)

    assert not report.ok
    assert any("duplicate" in error.lower() for error in report.errors)


def test_extra_string_metadata_and_optional_readmes_are_accepted(corpus: Path) -> None:
    rule, _ = _add_rule(corpus, metadata={"severity": "deployment-defined", "custom": "value"})
    (rule.parent / "README.md").write_text("Optional rule notes.\n", encoding="utf-8")

    report = validate_repository(corpus)

    assert report.ok, report.errors


def test_missing_required_metadata_is_reported_with_field_name(corpus: Path) -> None:
    rule, _ = _add_rule(corpus)
    rule.write_text(
        rule.read_text(encoding="utf-8").replace('    author = "Quarry contributors"\n', ""),
        encoding="utf-8",
    )

    report = validate_repository(corpus)

    assert not report.ok
    assert any("author" in error for error in report.errors)


def test_unsafe_yaml_tag_is_rejected_without_executing_constructor(corpus: Path) -> None:
    _, cases = _add_rule(corpus)
    sentinel = corpus / "unexpected-yaml-side-effect"
    (cases / "expectations.yaml").write_text(
        f"!!python/object/apply:os.mkdir [{json.dumps(str(sentinel))}]\n", encoding="utf-8"
    )

    report = validate_repository(corpus)

    assert not report.ok
    assert any("python/object" in error for error in report.errors)
    assert not sentinel.exists(), "Loading a contributor's YAML must never execute Python objects"


def test_expectation_path_cannot_escape_its_fixture_directory(corpus: Path) -> None:
    _, cases = _add_rule(corpus)
    _write_yaml(cases / "expectations.yaml", {"cases": {"../outside.json": {"match": False}}})

    report = validate_repository(corpus)

    assert not report.ok
    assert any("outside.json" in error for error in report.errors)


@pytest.mark.parametrize(
    "condition",
    [
        'model("unconfigured").score(messages, "policy") >= 0.8',
        'model("unconfigured").score(messages, text_ref("policy")) >= 0.8',
        "any msg in assistant_messages where off_topic(msg.content)",
        'any call in tool_calls where capability_tags("code_execution", call.name)',
    ],
    ids=["model", "model-text-resource", "topic-prober", "capability-classifier"],
)
def test_resource_dependent_rule_compiles_without_backend_or_fixture_execution(
    corpus: Path, monkeypatch: pytest.MonkeyPatch, condition: str
) -> None:
    _add_rule(corpus, condition=condition)

    def reject_external_execution(*args: object, **kwargs: object) -> None:
        pytest.fail(
            "Contribution validation must not construct a model backend or scan model inputs"
        )

    monkeypatch.setattr(ModelInferenceBackend, "__init__", reject_external_execution)
    monkeypatch.setattr(DSLScanner, "__init__", reject_external_execution)
    monkeypatch.setattr(DSLScanner, "scan_data", reject_external_execution)

    report = validate_repository(corpus)

    assert report.ok, report.errors
    assert report.compiled_rules == 1
    assert report.passed_cases == 0
    assert report.unevaluated_cases == 2
    assert report.warnings


def test_parameterized_policy_rule_compiles_without_unbound_fixture_execution(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_rule(
        corpus,
        parameters="    minimum: number",
        condition='model("unconfigured").score(messages, "policy") >= params.minimum',
    )

    def reject_scan(*args: object, **kwargs: object) -> None:
        pytest.fail("Unbound policy parameters must not be evaluated as fixture verdicts")

    monkeypatch.setattr(DSLScanner, "__init__", reject_scan)

    report = validate_repository(corpus)

    assert report.ok, report.errors
    assert report.compiled_rules == 1
    assert report.passed_cases == 0
    assert report.unevaluated_cases == 2


def test_typed_model_completion_uses_compile_schema_without_model_execution(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, cases = _add_rule(
        corpus, condition='model("verdict").complete(messages).json.label == "unsafe"'
    )
    expectations = yaml.safe_load((cases / "expectations.yaml").read_text(encoding="utf-8"))
    expectations["compile"] = {
        "response_schemas": {
            "verdict": {
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
                "additionalProperties": False,
            }
        }
    }
    _write_yaml(cases / "expectations.yaml", expectations)

    def reject_scan(*args: object, **kwargs: object) -> None:
        pytest.fail("A typed completion schema permits compilation without model execution")

    monkeypatch.setattr(ModelInferenceBackend, "__init__", reject_scan)
    monkeypatch.setattr(DSLScanner, "__init__", reject_scan)

    report = validate_repository(corpus)

    assert report.ok, report.errors
    assert report.compiled_rules == 1
    assert report.passed_cases == 0
    assert report.unevaluated_cases == 2


def test_compile_only_never_evaluates_even_local_fixtures(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_rule(corpus)

    def reject_scan(*args: object, **kwargs: object) -> None:
        pytest.fail("--compile-only must not run fixture conversations")

    monkeypatch.setattr(DSLScanner, "scan_data", reject_scan)

    report = validate_repository(corpus, compile_only=True)

    assert report.ok, report.errors
    assert report.compiled_rules == 1
    assert report.passed_cases == 0
    assert report.unevaluated_cases == 2


def test_supported_deprecated_rule_still_runs_its_regression_cases(corpus: Path) -> None:
    _add_rule(corpus, metadata={"status": "deprecated", "deprecation_reason": "Replaced behavior."})

    report = validate_repository(corpus)

    assert report.ok, report.errors
    assert report.passed_cases == 2
    assert any("deprecat" in warning.lower() for warning in report.warnings)


def test_incompatible_deprecated_rule_has_actionable_failure(corpus: Path) -> None:
    _add_rule(corpus, metadata={"status": "deprecated", "requires_quarry": ">=99.0.0"})

    report = validate_repository(corpus)

    assert not report.ok
    assert report.compiled_rules == 0
    assert any("update" in error.lower() or "remove" in error.lower() for error in report.errors)


def test_community_pack_validates_without_publisher_metadata(corpus: Path) -> None:
    _add_rule(corpus, pack="community/fixture-author/fixture-pack")

    report = validate_repository(corpus)

    assert report.ok, report.errors
    assert report.compiled_rules == 1
    assert report.passed_cases == 2


def test_cli_returns_failure_for_invalid_corpus_and_success_for_valid_corpus(corpus: Path) -> None:
    _, cases = _add_rule(corpus)
    assert main(["--root", str(corpus)]) == 0

    _write_json(cases / "negative/does_not_match.json", {"messages": "invalid"})

    assert main(["--root", str(corpus)]) != 0
