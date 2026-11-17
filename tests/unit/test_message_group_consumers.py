# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Span and called-rule consumers explicitly combine selected map entries."""

import pytest

from quarry.config import QuarryConfig
from tests.unit.test_nested_rule_runtime_contracts import _policy


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("consumer", ["span", "called-rule", "extract", "inside"])
@pytest.mark.parametrize(
    "selection,message_count,span_count",
    [
        ("[messages for item in messages]", 4, 2),
        ("[messages for item in messages][1]", 2, 1),
        ("[messages for item in messages][:1]", 2, 1),
    ],
)
def test_message_consumers_combine_only_selected_group_entries(
    restored, consumer, selection, message_count, span_count
):
    if consumer == "span":
        source = (
            'rule "Read" strings: $first="first" condition: '
            f"len($first.spans({selection})) == {span_count}"
        )
    elif consumer == "called-rule":
        source = (
            f'rule "Leaf" condition: len(messages) == {message_count} '
            'and messages[0].content == "first" and messages[-1].content == "second"\n'
            f'rule "Read" imports: check="Leaf" condition: check({selection})'
        )
    else:
        body = (
            f'extract({selection}, s) == "first"'
            if consumer == "extract"
            else f"s inside {selection}"
        )
        source = (
            'rule "Read" strings: $first="first" condition: '
            f"any s in $first.spans({selection}) where {body}"
        )
    result = _policy(source, QuarryConfig(config_version=2), restored).scan(
        {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
            ]
        },
        explain=True,
    )
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]
