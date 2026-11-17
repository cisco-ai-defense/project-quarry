# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""A malformed clock fails only flows that still need that clock."""

from copy import deepcopy

import pytest

from quarry import DSLScanner, compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.safety.limits import SafetyLimits

INPUTS = {"memory.writes": {"facts": {"content": {"type": "string"}}}}
DATA = {
    "messages": [{"role": "user", "content": "go"}],
    "tools": [{"type": "function", "function": {"name": "q"}}],
    "memory": {"writes": [{"id": "w1", "content": "release"}]},
}
TIMELINE = [{"message_index": 0}, {"input": "memory.writes", "id": "w1"}]
SETTLED = '(any t in tools where t.name == "q") until (any w in memory.writes where w.content == "release")'
LIVE = '(any w in memory.writes where w.content == "release") after user_said("go")'


def _data(problem):
    data = deepcopy(DATA)
    data["timeline"] = deepcopy(TIMELINE)
    if problem == "unknown":
        data["timeline"][-1]["id"] = "unknown"
    elif problem == "duplicate":
        data["timeline"].append(dict(TIMELINE[-1]))
    elif problem == "missing":
        data["timeline"].pop()
    return data


@pytest.mark.parametrize("problem", ["valid", "unknown", "duplicate", "missing"])
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize(
    ("condition", "matches"),
    [
        (SETTLED, True),
        (
            '(any w in memory.writes where w.content == "release") after '
            '(any t in tools where t.name == "missing")',
            False,
        ),
    ],
)
def test_settled_flows_keep_their_verdict_with_a_bad_clock(
    problem, restored, explain, condition, matches
):
    config = QuarryConfig(
        config_version=2, inputs=INPUTS, safety_limits=SafetyLimits(rule_timeout=None)
    )
    policy = compile_policy(f'rule "Settled" condition: {condition}', runtime_config=config)
    if restored:
        wire = policy.to_bytes()
        policy = load_compiled_policy(wire, runtime_config=config)
        assert policy.to_bytes() == wire
    result = policy.scan(_data(problem), explain=explain)
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == (["Settled"] if matches else [])


@pytest.mark.parametrize("problem", ["unknown", "duplicate", "missing"])
def test_settled_flow_does_not_hide_a_live_flow_timeline_fault(problem):
    config = QuarryConfig(
        config_version=2,
        inputs=INPUTS,
        safety_limits=SafetyLimits(rule_timeout=None),
        quarry_rule_texts=(
            f'rule "Settled" condition: {SETTLED}',
            f'rule "Live" condition: {LIVE}',
            f'rule "Mixed" condition: ({SETTLED}) and ({LIVE})',
        ),
    )
    result = DSLScanner(config).scan_data(_data(problem))
    assert result.completion_state == "incomplete"
    assert result.rule_matches == ["Settled"]
    assert set(result.evaluation_errors) == {"Live", "Mixed"}


@pytest.mark.parametrize("problem", ["list-subclass", "entry-subclass", "entry-key", "unread-key"])
def test_malformed_clock_objects_fail_only_the_flow(problem):
    class BadList(list):
        def __iter__(self):
            raise AssertionError("timeline list hook ran")

    class BadDict(dict):
        def get(self, *args):
            raise AssertionError("timeline entry hook ran")

    class BadKey:
        def __hash__(self):
            return hash("rogue" if problem == "unread-key" else "input")

        def __eq__(self, other):
            raise AssertionError("timeline key equality hook ran")

    config = QuarryConfig(
        config_version=2,
        inputs={**INPUTS, "audit.records": {"facts": {"kind": {"type": "string"}}}},
        safety_limits=SafetyLimits(rule_timeout=None),
        quarry_rule_texts=(
            f'rule "Live" condition: {LIVE}',
            'rule "Constant" condition: true',
        ),
    )
    data = _data("valid")
    if problem == "list-subclass":
        data["timeline"] = BadList(data["timeline"])
    elif problem == "entry-subclass":
        data["timeline"][0] = BadDict(data["timeline"][0])
    elif problem == "entry-key":
        data["timeline"][0] = {BadKey(): 0}
    else:
        data["audit"] = {BadKey(): []}
        data["timeline"].insert(1, {"input": "audit.rogue", "id": "r"})

    result = DSLScanner(config).scan_data(data)
    assert result.rule_matches == ["Constant"]
    assert set(result.evaluation_errors) == {"Live"}
