# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Cross-section until verdicts need supplied order, never declaration order."""

from copy import deepcopy

import pytest

from quarry import DSLScanner, compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.safety.limits import SafetyLimits

DEFS = {
    "audit.records": {"facts": {"kind": {"type": "string"}}},
    "memory.writes": {"facts": {"content": {"type": "string"}}},
}
CONDITION = (
    '(any a in audit.records where a.kind == "safe") '
    'until (any w in memory.writes where w.content == "release")'
)
DATA = {
    "audit": {"records": [{"id": "a", "kind": "deny"}]},
    "memory": {"writes": [{"id": "w", "content": "release"}]},
}
AUDIT = {"input": "audit.records", "id": "a"}
MEMORY = {"input": "memory.writes", "id": "w"}


def config(reverse=False, rules=()):
    return QuarryConfig(
        config_version=2,
        inputs=dict(reversed(list(DEFS.items()))) if reverse else DEFS,
        quarry_rule_texts=rules,
        safety_limits=SafetyLimits(rule_timeout=None),
    )


def policy(prepared, reverse=False, condition=CONDITION):
    cfg = config(reverse)
    result = compile_policy('rule "Flow" condition: ' + condition, runtime_config=cfg)
    if prepared:
        wire = result.to_bytes()
        result = load_compiled_policy(wire, runtime_config=cfg)
        assert result.to_bytes() == wire
    return result


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
@pytest.mark.parametrize("problem", ["absent", "unknown", "duplicate", "missing", "not-list"])
def test_cross_section_until_never_invents_an_unsupplied_or_faulty_clock(
    reverse, prepared, explain, problem
):
    data = deepcopy(DATA)
    if problem == "unknown":
        data["timeline"] = [AUDIT, {"input": "memory.writes", "id": "unknown"}]
    elif problem == "duplicate":
        data["timeline"] = [AUDIT, MEMORY, AUDIT]
    elif problem == "missing":
        data["timeline"] = [MEMORY]
    elif problem == "not-list":
        data["timeline"] = 4
    result = policy(prepared, reverse).scan(data, explain=explain)
    assert result.completion_state == "incomplete"
    assert result.rule_matches == [] and result.validation_error is None
    assert [failure.to_dict() for failure in result.failures] == [
        {"category": "invalid_input", "rule_name": "Flow"}
    ]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
@pytest.mark.parametrize("release_first", [False, True])
def test_cross_section_until_uses_the_explicit_clock_in_either_declaration_order(
    reverse, prepared, explain, release_first
):
    data = {**deepcopy(DATA), "timeline": [MEMORY, AUDIT] if release_first else [AUDIT, MEMORY]}
    result = policy(prepared, reverse).scan(data, explain=explain)
    assert result.is_complete
    assert result.rule_matches == (["Flow"] if release_first else [])


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
@pytest.mark.parametrize("kind", ["safe", "deny"])
def test_until_on_one_declared_section_keeps_its_known_item_order(prepared, explain, kind):
    condition = (
        '(any a in audit.records where a.kind == "safe") '
        'until (any b in audit.records where b.kind == "release")'
    )
    data = {"audit": {"records": [{"id": "a", "kind": kind}, {"id": "b", "kind": "release"}]}}
    result = policy(prepared, condition=condition).scan(data, explain=explain)
    assert result.is_complete and result.rule_matches == (["Flow"] if kind == "safe" else [])


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
@pytest.mark.parametrize("empty", ["audit", "memory"])
def test_no_cross_section_order_is_needed_without_an_ambient_or_release_item(
    prepared, explain, empty
):
    data = deepcopy(DATA)
    data[empty]["records" if empty == "audit" else "writes"] = []
    result = policy(prepared).scan(data, explain=explain)
    assert result.is_complete and result.rule_matches == (["Flow"] if empty == "audit" else [])


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
@pytest.mark.parametrize("positive", [False, True])
def test_a_folded_away_until_does_not_require_a_clock(prepared, explain, positive):
    condition = ("true or (" if positive else "false and (") + CONDITION + ")"
    result = policy(prepared, condition=condition).scan(deepcopy(DATA), explain=explain)
    assert result.is_complete and result.rule_matches == (["Flow"] if positive else [])


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
def test_an_until_clock_fault_retains_independent_rule_matches(reverse, explain):
    rules = (
        'rule "Flow" condition: ' + CONDITION,
        'rule "Constant" condition: true',
        'rule "Audit" condition: any a in audit.records where a.kind == "deny"',
    )
    result = DSLScanner(config(reverse, rules)).scan_data(deepcopy(DATA), enable_explain=explain)
    assert result.completion_state == "incomplete"
    assert set(result.rule_matches) == {"Constant", "Audit"}
    assert set(result.evaluation_errors) == {"Flow"}
