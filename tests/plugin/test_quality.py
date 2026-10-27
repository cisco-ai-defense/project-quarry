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

"""Shared quality checks and their plugin warning integration."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from quarry.engine.atoms import AtomDomain, AtomSpec
from quarry.language import get_language_spec
from quarry.plugin.service import QuarryPlugin
from quarry.safety.quality import _constant_count_value, _count_cases, _has_event_anchor
from quarry.safety.validator import RuleValidator, Severity, ValidationWarning


def _compile(condition: str) -> dict:
    return QuarryPlugin().compile(rule_text=f'rule "Quality"\ncondition:\n    {condition}\n')


def _quality_warnings(report: dict) -> list[dict]:
    return [
        warning
        for warning in report.get("warnings", [])
        if "tautology" in warning["message"]
        or "vacuous condition" in warning["message"]
        or "no specific event or message-content anchor" in warning["message"]
    ]


@pytest.mark.parametrize(
    ("op", "bound", "expected"),
    [
        (">=", 0, True),
        (">", -1, True),
        ("<=", -1, False),
        ("<", 0, False),
        ("=", -1, False),
        ("!=", -1, True),
        ("=", 0, None),
        (">", 0, None),
        ("<", 1, None),
        ("=", float("nan"), None),
    ],
)
def test_count_fact_only_proves_nonnegative_size(op, bound, expected):
    spec = AtomSpec(
        domain=AtomDomain.STATIC,
        selector="messages",
        value_expr_key="",
        predicate="surface_count",
        op=op,
        number=bound,
    )
    assert _constant_count_value(spec) is expected


@pytest.mark.parametrize("predicate", ["on_surface", "at_candidate"])
def test_structural_event_atoms_are_not_specific_anchors(predicate):
    spec = AtomSpec(
        domain=AtomDomain.EVENT,
        selector="messages",
        value_expr_key="",
        predicate=predicate,
    )
    artifact = SimpleNamespace(atom_specs={"atom": spec}, capture_plan=object())
    assert _has_event_anchor(artifact) is False


@pytest.mark.parametrize(
    ("condition", "severity", "fragment"),
    [
        ('len(messages) >= 0 or called_tool("anything")', "high", "tautology"),
        ("true", "high", "tautology"),
        ("false", "high", "vacuous condition"),
        ('called_tool("x") or not called_tool("x")', "high", "tautology"),
        ('called_tool("x") and not called_tool("x")', "high", "vacuous condition"),
        ("len(messages) < 0", "high", "vacuous condition"),
        ("len(messages) > 5 and len(messages) < 3", "high", "vacuous condition"),
        ("len(messages) < 3 or len(messages) >= 3", "high", "tautology"),
        ("len(messages) < 2.5 and len(messages) >= 3", "high", "vacuous condition"),
        ("len(messages) < 2.5 or len(messages) >= 3", "high", "tautology"),
        (
            'has_tool("x") and (len(messages) > 5 and len(messages) < 3)',
            "high",
            "vacuous condition",
        ),
        (
            'has_tool("x") or (len(messages) < 3 or len(messages) >= 3)',
            "high",
            "tautology",
        ),
        (
            'count_tool_calls("x") > 3 and count_tool_calls("x") < 2',
            "high",
            "vacuous condition",
        ),
        ('count_tool_calls("run_query") >= 0', "high", "tautology"),
        ('len(get_tool_responses("read_file")) < 0', "high", "vacuous condition"),
        ("len(messages) > 0", "medium", "no specific event or message-content anchor"),
        ("count(tool_calls) > 0", "medium", "no specific event or message-content anchor"),
        ('count_tool_calls("") > 0', "medium", "no specific event or message-content anchor"),
        ('has_tool("delete_file")', "medium", "no specific event or message-content anchor"),
        (
            'any tool in tools where tool.name == "delete_file"',
            "medium",
            "no specific event or message-content anchor",
        ),
        (
            'has_tool("x") or (has_tool("x") and user_said("secret"))',
            "medium",
            "no specific event or message-content anchor",
        ),
        (
            'has_tool("x") and (has_tool("x") or user_said("secret"))',
            "medium",
            "no specific event or message-content anchor",
        ),
        (
            "len(messages) > 5 and len(user_messages) < 3",
            "medium",
            "no specific event or message-content anchor",
        ),
        (
            "any msg in messages where true",
            "medium",
            "no specific event or message-content anchor",
        ),
        (
            "any call in tool_calls where true",
            "medium",
            "no specific event or message-content anchor",
        ),
    ],
)
def test_quality_warning_for_constant_or_unanchored_rule(condition, severity, fragment):
    report = _compile(condition)

    assert report["valid"] is True
    assert [
        (warning["severity"], fragment in warning["message"])
        for warning in _quality_warnings(report)
    ] == [(severity, True)]


@pytest.mark.parametrize(
    "condition",
    [
        'called_tool("read_file")',
        'user_said("secret")',
        'system_prompt contains "secret"',
        'called_tool("http_request") after(within: 3) called_tool("read_file")',
        'any msg in user_messages where msg.content contains "secret"',
        'count_tool_calls("run_query") >= 3',
        'len(get_tool_responses("read_file")) > 1',
        'any tool in tools where tool.description contains_any ["Use this tool first", "ALWAYS EXECUTE"]',
        "any tool in tools where tool.parameters.properties.hidden != null",
    ],
)
def test_anchored_rule_has_no_quality_warning(condition):
    report = _compile(condition)

    assert report["valid"] is True
    assert _quality_warnings(report) == []


@pytest.mark.parametrize(
    ("condition", "expected_matches"),
    [
        ('len(messages) >= 0 or called_tool("anything")', ["Quality"]),
        ("len(messages) < 3 or len(messages) >= 3", ["Quality"]),
        ("len(messages) > 5 and len(messages) < 3", []),
        ("false", []),
    ],
)
async def test_quality_warning_does_not_change_scan_verdict(condition, expected_matches):
    plugin = QuarryPlugin()
    rule = f'rule "Quality"\ncondition:\n    {condition}\n'
    report = plugin.compile(rule_text=rule)

    assert _quality_warnings(report)[0]["severity"] == "high"
    assert report["valid"] is True
    assert report["rule_complexities"] == {"Quality": report["estimated_complexity"]}
    result = await plugin.scan(
        rule_text=rule,
        conversation={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert result["completion_state"] == "complete"
    assert [match["rule_name"] for match in result["matches"]] == expected_matches
    assert _quality_warnings(result["compilation"]) == _quality_warnings(report)
    assert result["compilation"]["estimated_complexity"] == report["estimated_complexity"]
    assert result["compilation"]["rule_complexities"] == report["rule_complexities"]

    invalid = await plugin.scan(rule_text=rule, conversation={"messages": []})
    assert invalid["completion_state"] == "invalid"
    assert "matches" not in invalid


def test_count_case_expansion_is_bounded_and_does_not_invent_a_constant():
    condition = " or ".join(f'count_tool_calls("t{i}") == 0' for i in range(6))
    plugin = QuarryPlugin()
    rule = f'rule "Quality"\ncondition:\n    {condition}\n'
    report, scanner = plugin._prepare_rule(rule, None)

    assert report["valid"] is True
    assert scanner is not None
    artifact = next(iter(scanner.compiled_rules.values()))
    assert len(_count_cases(artifact)) <= 32
    assert _quality_warnings(report) == []


def test_nested_field_example_is_anchored():
    rule_text = (Path(__file__).parents[2] / "examples/rules/nested_field.rule").read_text()
    report = QuarryPlugin().compile(rule_text=rule_text)

    assert report["valid"] is True
    assert _quality_warnings(report) == []


@pytest.mark.parametrize("failure", ["raise", "malformed_warning"])
async def test_quality_analysis_failure_does_not_reject_a_valid_rule(monkeypatch, failure):
    def fail_quality_analysis(_rule, _artifact):
        if failure == "raise":
            raise RuntimeError("advisory analysis failed")
        return [object()]

    original_validate = RuleValidator.validate

    def validate_with_warning(self, rule):
        result = original_validate(self, rule)
        result.warnings.append(
            ValidationWarning(Severity.LOW, "Validator warning", "Review the rule")
        )
        return result

    monkeypatch.setattr("quarry.plugin.service.quality_warnings", fail_quality_analysis)
    monkeypatch.setattr(RuleValidator, "validate", validate_with_warning)
    plugin = QuarryPlugin()
    rule = 'rule "Quality"\ncondition: true\n'

    compilation = plugin.compile(rule_text=rule)
    assert compilation["valid"] is True
    assert [warning["message"] for warning in compilation["warnings"]] == ["Validator warning"]
    assert compilation["rule_complexities"] == {"Quality": compilation["estimated_complexity"]}

    scan = await plugin.scan(
        rule_text=rule,
        conversation={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert scan["completion_state"] == "complete"
    assert [match["rule_name"] for match in scan["matches"]] == ["Quality"]
    assert scan["compilation"]["warnings"] == compilation["warnings"]


@pytest.mark.parametrize(
    "example",
    [
        pytest.param(example, id=example["id"])
        for example in get_language_spec()["examples"]
        if not example["requires_backend"]
    ],
)
def test_packaged_runnable_examples_have_no_quality_warning(example):
    rule_text = (Path(__file__).parents[2] / "quarry/language" / example["rule_path"]).read_text()
    report = QuarryPlugin().compile(rule_text=rule_text)

    assert report["valid"] is True
    assert _quality_warnings(report) == []
