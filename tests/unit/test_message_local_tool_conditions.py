# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Synthetic local-call collections keep their owning message and original call links."""

import pytest

from quarry import compile_policy, load_compiled_policy

DATA = {
    "messages": [
        {"role": "user", "content": "request"},
        {
            "role": "assistant",
            "content": "plan",
            "tool_calls": [
                {
                    "id": "call-a",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"limit":2}'},
                },
                {
                    "id": "call-b",
                    "type": "function",
                    "function": {"name": "other", "arguments": '{"limit":3}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call-a", "content": "found"},
        {"role": "tool", "tool_call_id": "call-b", "content": "ordinary"},
    ]
}


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize(
    "condition,matched",
    [
        ("any m in messages where len(m.tool_calls) == 2", True),
        (
            'any m in messages where count([c for c in m.tool_calls if c.name == "lookup"]) == 1',
            True,
        ),
        ("any m in messages where any c in m.tool_calls where c.arguments.limit == 2", True),
        ("assistant_messages[-1].tool_calls[0].arguments.limit == 2", True),
        (
            'any m in messages where any c in m.tool_calls[:1] where c.response.result contains "found"',
            True,
        ),
        (
            'any m in messages where any c in m.tool_calls where c.name == "lookup" and m.content contains "plan"',
            True,
        ),
        ("any m in user_messages where count([c for c in m.tool_calls if true]) == 0", True),
        ("any m in user_messages where len(m.tool_calls) == 0", False),
    ],
)
def test_message_local_call_collection_conditions(condition, matched, restored):
    policy = compile_policy('rule "Read" condition: ' + condition)
    if restored:
        raw = policy.to_bytes()
        policy = load_compiled_policy(raw)
        assert policy.to_bytes() == raw
    result = policy.scan(DATA, explain=True)
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == (["Read"] if matched else [])
