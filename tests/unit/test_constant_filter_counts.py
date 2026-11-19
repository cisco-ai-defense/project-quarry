# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Constant Boolean filters count only members of their selected event collection."""

import asyncio
from operator import eq, ge, gt, le, lt, ne

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.types.exceptions import RuleValidationError

PAYLOAD = {
    "messages": [
        {"role": "user", "content": "one"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "first",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "user", "content": "two"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "second",
                    "type": "function",
                    "function": {"name": "control", "arguments": "{}"},
                }
            ],
        },
    ]
}


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("function", ["count", "len"])
@pytest.mark.parametrize(
    "collection,size",
    [
        ("user_messages", 2),
        ("assistant_messages", 2),
        ("messages", 4),
        ("user_messages[1:]", 1),
        ("user_messages[:0]", 0),
        ("tool_calls", 2),
        ('get_tool_calls("lookup")', 1),
    ],
)
@pytest.mark.parametrize(
    "predicate,keep",
    [
        ("true", True),
        ("false", False),
        ("not false", True),
        ("false or (true and not false)", True),
    ],
)
def test_constant_filters_keep_the_exact_selected_members(
    prepared, function, collection, size, predicate, keep
):
    expected = size if keep else 0
    source = f"{function}([item for item in {collection} if {predicate}])"
    policy = compile_policy(f'rule "R" condition: {source} == {expected}')
    artifact = policy.to_bytes()
    if prepared:
        policy = load_compiled_policy(artifact)
        assert policy.to_bytes() == artifact
    for explain in [False, True]:
        for asynchronous in [False, True]:
            result = (
                asyncio.run(policy.scan_async(PAYLOAD, explain=explain))
                if asynchronous
                else policy.scan(PAYLOAD, explain=explain)
            )
            assert result.is_complete
            assert result.rule_matches == ["R"]


@pytest.mark.parametrize("function", ["count", "len"])
@pytest.mark.parametrize("predicate,size", [("true", 2), ("false", 0)])
@pytest.mark.parametrize(
    "operator,compare", [("==", eq), ("!=", ne), (">", gt), (">=", ge), ("<", lt), ("<=", le)]
)
@pytest.mark.parametrize("flipped", [False, True])
def test_constant_filter_comparisons_keep_both_operand_orders(
    function, predicate, size, operator, compare, flipped
):
    value = f"{function}([item for item in user_messages if {predicate}])"
    condition = f"2 {operator} {value}" if flipped else f"{value} {operator} 2"
    expected = compare(2, size) if flipped else compare(size, 2)
    policy = load_compiled_policy(compile_policy('rule "R" condition: ' + condition).to_bytes())
    result = policy.scan(PAYLOAD, explain=True)
    assert result.is_complete
    assert result.rule_matches == (["R"] if expected else [])


@pytest.mark.parametrize("function", ["count", "len"])
@pytest.mark.parametrize(
    "collection,predicate,bound,error",
    [
        ("user_messages", 'called_tool("lookup")', "2", "only reads the element itself"),
        (
            "user_messages",
            'item.content contains "one" or called_tool("lookup")',
            "2",
            "only reads the element itself",
        ),
        ("user_messages", 'item.content contains "one"', "2.5", "not a meaningful bound"),
        ("user_messages", "true", "-1", "bound must be >= 0"),
        ("tools", "true", "1", "static surface 'tools' is not supported"),
    ],
)
def test_constant_admission_keeps_other_filter_restrictions(
    function, collection, predicate, bound, error
):
    condition = f"{function}([item for item in {collection} if {predicate}]) >= {bound}"
    with pytest.raises(RuleValidationError, match=error):
        compile_policy('rule "R" condition: ' + condition)


@pytest.mark.parametrize("function", ["count", "len"])
@pytest.mark.parametrize(
    "predicate", ['item.content contains "one"', 'not (item.content contains "one")']
)
def test_existing_element_predicates_keep_their_counts(function, predicate):
    policy = compile_policy(
        f'rule "R" condition: {function}([item for item in user_messages if {predicate}]) == 1'
    )
    result = policy.scan(PAYLOAD)
    assert result.is_complete and result.rule_matches == ["R"]
