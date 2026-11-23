# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Temporal lowering retains executable tool plans and reloads its own policies."""

import json

import pytest

from quarry import compile_policy, load_compiled_policy


def source(*, capture=False, quantifier="any", bound=False):
    body = 'count_tool_calls("x") != 0'
    if bound:
        body = 'v2.name == "x" and ' + body
    condition = (
        '(any v0 in get_tool_responses("x") where '
        '(v0.content contains "a" until '
        f"(any o1 in assistant_messages where ({quantifier} v2 in tools where {body}))))"
    )
    if capture:
        condition = "(any k in assistant_messages where $word in k.content as v) and " + condition
    return 'rule "Roundtrip"\nstrings:\n $word = "a" fullword\ncondition:\n ' + condition


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("quantifier", ["any", "all"])
@pytest.mark.parametrize("bound", [False, True])
def test_temporal_tool_quantifiers_roundtrip_with_original_verdicts(capture, quantifier, bound):
    policy = compile_policy(source(capture=capture, quantifier=quantifier, bound=bound))
    saved = policy.to_bytes()
    artifact = json.loads(saved)["rule"]["artifact"]
    live = set(artifact["dfa"]["atoms"])
    for _, child in artifact["closed_conditions"]:
        live.update(child["atoms"])
    assert {row[0] for row in artifact["tool_quantifiers"]} <= live
    restored = load_compiled_policy(saved)
    assert restored.to_bytes() == saved
    tools = [{"type": "function", "function": {"name": "x", "parameters": {"type": "object"}}}]
    linked = [
        {
            "role": "assistant",
            "content": "a",
            "tool_calls": [
                {
                    "id": "synthetic-call",
                    "type": "function",
                    "function": {"name": "x", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "synthetic-call", "content": "a"},
        {"role": "assistant", "content": "a"},
    ]
    for conversation in [
        {"messages": [{"role": "user", "content": "a"}], "tools": []},
        {"messages": [{"role": "user", "content": "a"}], "tools": tools},
        {"messages": linked, "tools": tools},
        {"messages": [*linked[:1], {**linked[1], "content": "b"}, *linked[2:]], "tools": tools},
    ]:
        for explain in [False, True]:
            expected, actual = (
                policy.scan(conversation, explain=explain),
                restored.scan(conversation, explain=explain),
            )
            assert actual.is_complete and expected.is_complete
            assert actual.rule_matches == expected.rule_matches
            assert actual.evaluation_errors == expected.evaluation_errors


def test_loader_still_rejects_an_unread_tool_plan_in_supplied_policy():
    policy = compile_policy('rule "Roundtrip" condition: any t in tools where t.name == "x"')
    document = json.loads(policy.to_bytes())
    document["rule"]["artifact"]["tool_quantifiers"].append(
        ["synthetic-unread", "any", {"kind": "true"}, []]
    )
    with pytest.raises(ValueError, match="tool quantifier ids must be DFA atoms"):
        load_compiled_policy(json.dumps(document).encode())
