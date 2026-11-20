# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Prepared calls carry exactly the child artifacts the executable automata can read."""

import pytest

from quarry.config import QuarryConfig
from quarry.policy_serialization import _artifact_from_wire, _artifact_to_wire
from tests.unit.test_nested_rule_runtime_contracts import _policy


@pytest.mark.parametrize(
    "expression,matched",
    [
        ("(2 of ($a) in system_prompt) and leaf(messages)", False),
        ("leaf(messages) and (2 of ($a) in system_prompt)", False),
        ("not (2 of ($a) in system_prompt) or leaf(messages)", True),
        ("leaf(messages) or not (2 of ($a) in system_prompt)", True),
    ],
)
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
def test_lowering_eliminates_unread_children_before_serialization(
    expression, matched, restored, explain
):
    source = (
        'rule "Leaf" condition: user_said("needle")\nrule "Caller" imports: leaf="Leaf" strings: $a="a" condition: '
        + expression
    )
    prepared = _policy(source, QuarryConfig(config_version=2), restored)
    artifact = prepared._scanner.compiled_rules["Caller"]
    assert not artifact.split_checks
    assert _artifact_to_wire(_artifact_from_wire(_artifact_to_wire(artifact))) == _artifact_to_wire(
        artifact
    )
    result = prepared.scan(
        {"messages": [{"role": "system", "content": "a"}, {"role": "user", "content": "needle"}]},
        explain=explain,
    )
    assert result.is_complete and result.rule_matches == (["Caller"] if matched else [])


@pytest.mark.parametrize(
    "expression",
    [
        "leaf(messages)",
        "any m in messages where leaf(m)",
        'any t in tools where t.name=="lookup" and leaf(messages)',
    ],
)
def test_live_root_event_and_tool_body_checks_survive_reload(expression):
    source = (
        'rule "Leaf" condition: user_said("needle")\nrule "Caller" imports: leaf="Leaf" condition: '
        + expression
    )
    prepared = _policy(source, QuarryConfig(config_version=2), True)
    assert prepared._scanner.compiled_rules["Caller"].split_checks
    result = prepared.scan(
        {
            "messages": [{"role": "user", "content": "needle"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }
    )
    assert result.is_complete and result.rule_matches == ["Caller"]
