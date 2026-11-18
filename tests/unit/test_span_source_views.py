# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Span-source searches retain selected occurrences and their owning message view."""

import json
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from quarry import DSLScanner, compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.engine.atoms import ModelExpression as E
from quarry.integrations.model_expression import _BoundMessage, _span_source_texts
from quarry.types.messages import Messages

STRATEGIES = ["direct", "compiled", "reloaded"]
EXPLAINS = [False, True, "support"]
SCALARS = [
    'messages.transform("strip")[0].content',
    'messages.transform("strip")[-1].content',
    'messages.transform("strip").user_messages[0].content',
    '(messages + messages.transform("strip"))[1].content',
    '(messages.transform("strip") + messages)[0].content',
]


def scan(condition, strings, data, strategy, explain):
    cfg = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "rule_timeout": None,
            "transforms": {"strip": {"operations": [{"op": "strip"}]}},
        },
        Path("/tmp"),
        require_rules=False,
    )
    source = 'rule "Read"\nstrings:\n' + strings + "\ncondition: " + condition
    if strategy == "direct":
        return DSLScanner(replace(cfg, quarry_rule_texts=(source,))).scan_data(
            data, enable_explain=explain
        )
    policy = compile_policy(source, runtime_config=cfg)
    if strategy == "compiled":
        return policy.scan(data, explain=explain)
    wire = policy.to_bytes()
    assert json.loads(wire)["rule"]["artifact"]["compiler_version"] == "compiler-30"

    def forbidden(*args, **kwargs):
        raise AssertionError("unchanged compiler21 artifact entered parser/compiler")

    with ExitStack() as guards:
        for target in [
            "quarry.dsl.parser.DSLParser.parse",
            "quarry.scanner.compile_rule",
            "quarry.engine.compiler.compile_rule",
            "quarry.policy.compile_policy",
        ]:
            guards.enter_context(patch(target, forbidden))
        loaded = load_compiled_policy(wire, runtime_config=cfg)
        assert loaded.to_bytes() == wire
        result = loaded.scan(data, explain=explain)
        assert loaded.to_bytes() == wire
        return result


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("explain", EXPLAINS)
@pytest.mark.parametrize(
    ("selection", "cut_count"),
    [
        ('messages + messages.transform("strip")', 1),
        ('messages.transform("strip") + messages', 1),
        ('messages.transform("strip") + messages + messages.transform("strip")', 2),
        (
            '(messages + messages.transform("strip"))[1:] + (messages + messages.transform("strip"))[:1]',
            1,
        ),
    ],
)
def test_each_transformed_occurrence_has_its_own_search_text(
    selection, cut_count, strategy, explain
):
    result = scan(
        f"len($raw.spans({selection})) == 1 and len($cut.spans({selection})) == {cut_count}",
        '$raw="^ prefix $" regex\n$cut="^prefix$" regex',
        {"messages": [{"role": "user", "content": " prefix "}]},
        strategy,
        explain,
    )
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("explain", EXPLAINS)
@pytest.mark.parametrize("text", SCALARS)
def test_indexed_transformed_content_retains_its_owner(text, strategy, explain):
    result = scan(
        f"len($p.spans({text})) == 1",
        '$p="^prefix$" regex',
        {"messages": [{"role": "user", "content": " prefix "}]},
        strategy,
        explain,
    )
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("explain", EXPLAINS)
@pytest.mark.parametrize("text", [SCALARS[0], SCALARS[1], SCALARS[3]])
def test_transformed_content_windows_use_the_same_owned_source(text, strategy, explain):
    result = scan(
        f'{text}.extract($p.spans({text}), left=1, right=1) contains "needle"',
        '$p="needle"',
        {"messages": [{"role": "user", "content": "  prefix needle suffix  "}]},
        strategy,
        explain,
    )
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("explain", EXPLAINS)
@pytest.mark.parametrize(
    "text", ["messages[0].content", "user_messages[0].content", "messages[-1].content"]
)
def test_canonical_content_sources_keep_their_existing_verdict(text, strategy, explain):
    result = scan(
        f"len($p.spans({text})) == 1",
        '$p="needle"',
        {"messages": [{"role": "user", "content": "needle"}]},
        strategy,
        explain,
    )
    assert result.is_complete and result.rule_matches == ["Read"]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("explain", EXPLAINS)
@pytest.mark.parametrize("arguments", ["not-json", '{"n": NaN}'])
def test_content_only_span_sources_do_not_decode_tool_arguments(arguments, strategy, explain):
    data = {
        "messages": [
            {
                "role": "assistant",
                "content": "needle",
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": arguments},
                    }
                ],
            }
        ]
    }
    result = scan("len($p.spans(messages[0].content)) == 1", '$p="needle"', data, strategy, explain)
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("explain", EXPLAINS)
@pytest.mark.parametrize("transformed", [False, True])
def test_tool_result_content_preserves_its_selected_owner(transformed, strategy, explain):
    text = (
        'messages.transform("strip").tool_responses[0].result'
        if transformed
        else "tool_responses[0].result"
    )
    data = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c",
                "content": " prefix " if transformed else "prefix",
            },
        ]
    }
    result = scan(f"len($p.spans({text})) == 1", '$p="^prefix$" regex', data, strategy, explain)
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign_index", [0, 1])
async def test_owned_content_refuses_a_foreign_selection_with_matching_indices(foreign_index):
    canonical = Messages.from_openai([{"role": "user", "content": "same"}] * 2)
    foreign = Messages.from_openai([{"role": "user", "content": "same"}] * 2)
    context = SimpleNamespace(conversation=SimpleNamespace(messages=canonical))
    expression = E("span_text", "content", (E("binding", "selected"),))
    with pytest.raises(ValueError):
        await _span_source_texts(
            expression,
            context,
            None,
            {
                "selected": _BoundMessage(
                    foreign[foreign_index], foreign[foreign_index : foreign_index + 1]
                )
            },
            None,
            None,
        )
