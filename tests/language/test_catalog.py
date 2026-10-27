# Copyright 2025 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""The generated Quarry catalog stays in step with the runtime language."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from quarry.dsl.parser import DSLParser
from quarry.engine.compiler import compile_rule
from quarry.language import get_language_spec
from quarry.language.generate import generate_language_spec, render_reference, write_language_spec
from quarry.safety.validator import RuleValidator
from quarry.types.dsl.model_calls import MODEL_OPERATION_TYPES
from quarry.types.exceptions import RuleParseError
from tests.conftest import rule_matches

REPO_ROOT = Path(__file__).parents[2]
LANGUAGE_DIR = REPO_ROOT / "quarry" / "language"
SPEC = get_language_spec()


def _compile_stage(rule_text: str) -> tuple[str, list[str]]:
    """Use the runtime parser, validator, and compiler without the optional editor plugin."""

    try:
        parsed = DSLParser().parse(rule_text)
    except RuleParseError as exc:
        return "parse", [str(exc)]
    validation = RuleValidator().validate(parsed)
    if not validation.is_valid:
        return "validate", [
            *(error.message for error in validation.semantic_errors),
            *(error.error for error in validation.regex_errors),
        ]
    compiled = compile_rule(parsed)
    if not compiled.ok:
        return "compile", [error.message for error in compiled.diagnostics.errors]
    return "complete", []


def test_generated_catalog_and_reference_match_the_annotated_implementation(tmp_path):
    assert generate_language_spec() == SPEC
    output = tmp_path / "spec.json"
    assert write_language_spec(output) == SPEC
    assert output.read_bytes() == (LANGUAGE_DIR / "spec.json").read_bytes()
    assert render_reference(SPEC) == (LANGUAGE_DIR / "reference.md").read_text()


def test_every_model_operation_is_published_as_a_model_method():
    published = {
        item["name"].removeprefix("model.")
        for item in SPEC["items"]
        if item["id"].startswith("compiler_form:model.")
    }
    assert published == set(MODEL_OPERATION_TYPES)


_RUNNABLE_CHECKS = [
    pytest.param(example["rule_path"], check, id=f"{example['id']}:{check['input_path']}")
    for example in SPEC["examples"]
    if not example["requires_backend"]
    for check in example["checks"]
]


@pytest.mark.parametrize(("rule_path", "check"), _RUNNABLE_CHECKS)
def test_a_published_example_check_produces_its_declared_verdict(rule_path, check):
    rule_text = (LANGUAGE_DIR / rule_path).read_text()
    conversation = json.loads((LANGUAGE_DIR / check["input_path"]).read_text())
    assert rule_matches(rule_text, conversation) is check["expected_match"]


def test_every_runnable_example_shows_a_match_and_a_clean_conversation():
    polarities = {
        example["id"]: {check["expected_match"] for check in example["checks"]}
        for example in SPEC["examples"]
        if not example["requires_backend"]
    }
    assert polarities == dict.fromkeys(polarities, {True, False})


@pytest.mark.parametrize(
    "rule_text",
    [
        pytest.param((LANGUAGE_DIR / example["rule_path"]).read_text(), id=example["id"])
        for example in SPEC["examples"]
        if example["requires_backend"]
    ],
)
def test_a_backend_example_compiles_with_the_runtime_language(rule_text):
    assert _compile_stage(rule_text) == ("complete", [])


@pytest.mark.parametrize(
    ("content", "matched"),
    [("please EXPLOIT it", True), ("a normal request", False)],
    ids=["folds-case", "stays-quiet"],
)
def test_the_documented_case_insensitive_alternation_folds_case(content, matched):
    rule = (
        'rule "Case-insensitive alternatives"\n'
        "strings:\n"
        '    $bad = "attack|exploit" nocase regex\n'
        "condition:\n"
        "    any m in user_messages where m.content contains $bad\n"
    )
    assert rule_matches(rule, {"messages": [{"role": "user", "content": content}]}) is matched


@pytest.mark.parametrize(
    "record",
    [pytest.param(record, id=record["source_form"]) for record in SPEC["rejected_spellings"]],
)
def test_a_rejected_spelling_fails_at_its_documented_stage(record):
    stage, diagnostics = _compile_stage(
        f'rule "Compatibility"\ncondition:\n    {record["source_form"]}\n'
    )
    # The omitted editor plugin named its function-resolution pass ``link``; the
    # runtime compiler performs that same check during compilation.
    expected_stage = "compile" if record["rejected_by"] == "link" else record["rejected_by"]
    assert stage == expected_stage, diagnostics
    assert diagnostics
    if record["rejected_call"] is not None:
        assert record["rejected_call"] in " ".join(diagnostics)


_LITERALS = {
    "string": '"x"',
    "integer": "1",
    "number": "0.5",
    "boolean": "true",
    "model_input": "conversation()",
}
_AS_CONDITION = {"boolean": "{}", "number": "{} > 0", "integer": "{} > 0", "string": '{} == "x"'}


def _call_rule(item: dict, violated: int | None = None) -> str:
    arguments = []
    for index, parameter in enumerate(item["parameters"]):
        wants_field = bool(parameter.get("field_only")) != (index == violated)
        arguments.append(
            "msg.content" if wants_field else _LITERALS[parameter["accepted_types"][0]]
        )
    call = f"{item['name']}({', '.join(arguments)})"
    condition = _AS_CONDITION.get(item["return_type"], "{}").format(call)
    return f'rule "R"\ncondition:\n    any msg in messages where {condition}\n'


_RESTRICTED_ARGUMENTS = [
    pytest.param(item, index, id=f"{item['name']}:{parameter['name']}")
    for item in SPEC["items"]
    if item["kind"] == "function" and not item.get("syntax")
    for index, parameter in enumerate(item["parameters"])
    if parameter.get("literal_only") or parameter.get("field_only")
]


@pytest.mark.parametrize(("item", "index"), _RESTRICTED_ARGUMENTS)
def test_a_published_argument_restriction_is_enforced(item, index):
    parameter = item["parameters"][index]
    expected = (
        f"argument '{parameter['name']}' must be written as a literal"
        if parameter.get("literal_only")
        else f"argument '{parameter['name']}' must be a conversation field"
    )
    accepted_stage, accepted_diagnostics = _compile_stage(_call_rule(item))
    rejected_stage, rejected_diagnostics = _compile_stage(_call_rule(item, violated=index))
    assert accepted_stage != "validate", accepted_diagnostics
    assert rejected_stage == "validate"
    assert expected in " ".join(rejected_diagnostics)


@pytest.mark.parametrize(
    "surface",
    [
        pytest.param(item, id=item["name"])
        for item in SPEC["items"]
        if item["kind"] == "surface" and item.get("fields")
    ],
)
def test_a_surface_accepts_exactly_its_documented_fields(surface):
    def compile_field(field: str) -> tuple[str, list[str]]:
        return _compile_stage(
            f'rule "R"\ncondition:\n    any x in {surface["name"]} where x.{field} == "x"\n'
        )

    assert [
        (field["name"], compile_field(field["name"])[0]) for field in surface["field_details"]
    ] == [(field, "complete") for field in surface["fields"]]
    stage, messages = compile_field("not_a_field")
    assert stage == "validate"
    assert f"Available fields: {', '.join(surface['fields'])}" in " ".join(messages)


def test_a_missing_catalog_fails_on_first_use_and_names_the_generator():
    script = textwrap.dedent(
        """
        from importlib import resources

        class MissingCatalog:
            def joinpath(self, _name):
                return self

            def read_text(self, **_kwargs):
                raise FileNotFoundError("spec.json")

        resources.files = lambda _package: MissingCatalog()
        from quarry.config import QuarryConfig
        from quarry.language import LanguageSpecMissingError
        from quarry.scanner import DSLScanner

        print("scanner imported")
        try:
            DSLScanner(QuarryConfig(quarry_rule_texts=('rule "R"\\ncondition:\\n    true\\n',)))
        except LanguageSpecMissingError as exc:
            print(str(exc))
        else:
            raise AssertionError("catalog access unexpectedly succeeded")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines()[0] == "scanner imported"
    assert "python -m scripts.generate_language_spec" in completed.stdout
