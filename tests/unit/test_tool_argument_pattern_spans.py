# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tool-argument pattern spans retain their actual source and text coordinates."""

import json
from dataclasses import replace

import pytest

from quarry import DSLScanner, compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.safety.limits import SafetyLimits
from quarry.types.messages import Messages
from quarry.types.results import Span

RAW = json.dumps({"q": "é 😀 needle", "nested": {"text": "second needle"}, "number": 7})
DATA = {
    "messages": [
        {"role": "user", "content": "needle outside arguments"},
        {
            "role": "assistant",
            "content": "needle in assistant text",
            "tool_calls": [
                {
                    "id": "first",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": RAW},
                },
                {
                    "id": "second",
                    "type": "function",
                    "function": {"name": "other", "arguments": '{"q":"unrelated"}'},
                },
            ],
        },
    ]
}


def _scan(condition, *, restored, explain):
    source = 'rule "Read"\nstrings:\n    $p = "needle"\ncondition: ' + condition
    config = QuarryConfig(config_version=2, safety_limits=SafetyLimits(rule_timeout=None))
    if restored:
        original = compile_policy(source, runtime_config=config)
        wire = original.to_bytes()
        policy = load_compiled_policy(wire, runtime_config=config)
        assert policy.to_bytes() == wire
        return policy.scan(DATA, explain=explain)
    return DSLScanner(replace(config, quarry_rule_texts=(source,))).scan_data(
        DATA, enable_explain=explain
    )


@pytest.mark.parametrize(
    "receiver",
    [
        "assistant_messages[-1].tool_calls",
        "assistant_messages[-1].tool_calls[0]",
    ],
)
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True])
def test_call_and_call_list_pattern_spans_read_only_decoded_argument_texts(
    receiver, restored, explain
):
    result = _scan(
        f"len($p.spans({receiver})) == 2 and "
        f'all s in $p.spans({receiver}) where messages.extract(s) == "needle"',
        restored=restored,
        explain=explain,
    )
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True])
def test_raw_argument_pattern_spans_extract_from_original_json(restored, explain):
    result = _scan(
        "any c in tool_calls where "
        "(len($p.spans(c.raw_arguments)) == 2 and "
        'all s in $p.spans(c.raw_arguments) where messages.extract(s) == "needle")',
        restored=restored,
        explain=explain,
    )
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]


def _raw_span(messages, start, end, **kwargs):
    return Span(1, start, end, source_field="raw_arguments", **kwargs)._bind_to_source(
        messages.tool_calls[0]
    )


@pytest.mark.parametrize("context", [0, 3, 10000])
def test_raw_argument_extraction_preserves_original_offsets_and_context(context):
    messages = Messages.from_openai(DATA["messages"])
    start = RAW.index("needle")
    span = _raw_span(messages, start, start + 6)
    assert (
        messages.extract(span, context, context)
        == RAW[max(0, start - context) : start + 6 + context]
    )


def test_raw_argument_extraction_rejects_foreign_source_and_invalid_ranges():
    messages = Messages.from_openai(DATA["messages"])
    other = Messages.from_openai(DATA["messages"])
    with pytest.raises(ValueError, match="not present"):
        other.extract(_raw_span(messages, 0, 1))
    with pytest.raises(ValueError, match="outside"):
        messages.extract(_raw_span(messages, 0, len(RAW) + 1))
    with pytest.raises(ValueError, match="outside"):
        messages.extract(_raw_span(messages, 0, 1, end_source_field="arguments.q"))


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True])
def test_bound_tool_call_pattern_spans_keep_the_call_source(restored, explain):
    result = _scan(
        'any c in get_tool_calls("lookup") where '
        "(len($p.spans(c)) == 2 and "
        'all s in $p.spans(c) where messages.extract(s) == "needle")',
        restored=restored,
        explain=explain,
    )
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]


@pytest.mark.parametrize(
    "calls",
    [[], [{"id": "empty", "type": "function", "function": {"name": "empty", "arguments": "{}"}}]],
)
@pytest.mark.parametrize("restored", [False, True])
def test_empty_call_sources_do_not_fall_back_to_assistant_content(calls, restored):
    source = 'rule "Empty"\nstrings:\n    $p = "needle"\ncondition: len($p.spans(assistant_messages[-1].tool_calls)) == 0'
    config = QuarryConfig(config_version=2)
    policy = compile_policy(source, runtime_config=config)
    if restored:
        policy = load_compiled_policy(policy.to_bytes(), runtime_config=config)
    result = policy.scan(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "needle",
                    **({"tool_calls": calls} if calls else {}),
                }
            ]
        }
    )
    assert result.is_complete
    assert result.rule_matches == ["Empty"]


@pytest.mark.asyncio
async def test_call_pattern_spans_reject_calls_from_another_conversation():
    from types import SimpleNamespace

    from quarry.engine.atoms import ModelExpression
    from quarry.engine.matchers import PURE_REGEX
    from quarry.integrations.model_expression import _span_source_texts

    messages = Messages.from_openai(DATA["messages"])
    other = Messages.from_openai(DATA["messages"])
    context = SimpleNamespace(conversation=SimpleNamespace(messages=messages))
    expression = ModelExpression("call_texts", None, (ModelExpression("binding", "calls"),))
    for foreign in (
        other.tool_calls[0],
        other.tool_calls,
        [messages.tool_calls[0], other.tool_calls[0]],
    ):
        with pytest.raises(ValueError, match="scanned input"):
            await _span_source_texts(
                expression, context, None, {"calls": foreign}, None, PURE_REGEX
            )


@pytest.mark.parametrize("restored", [False, True])
def test_raw_and_decoded_argument_spans_compare_in_original_json(restored):
    source = r"""rule "Read"
strings:
    $tail = "needle"
    $encoded = ".ud83d.ude00 needle" regex
    $decoded = "😀 needle"
    $other = "other"
condition:
    (any s in $tail.spans(assistant_messages[0].tool_calls[0].arguments.q) where
        s inside $encoded.spans(assistant_messages[0].tool_calls[0].raw_arguments) and
        messages.extract(s, 2, 0) == "😀 needle") and
    (any s in $encoded.spans(assistant_messages[0].tool_calls[0].raw_arguments) where
        s inside $decoded.spans(assistant_messages[0].tool_calls[0].arguments.q)) and
    (all s in $other.spans(assistant_messages[0].tool_calls[0].raw_arguments) where
        not s inside $decoded.spans(assistant_messages[0].tool_calls[0].arguments.q)) and
    (all s in $other.spans(assistant_messages[0].tool_calls[0].raw_arguments) where
        s inside assistant_messages[0].tool_calls[0])
"""
    config = QuarryConfig(config_version=2, safety_limits=SafetyLimits(rule_timeout=None))
    policy = compile_policy(source, runtime_config=config)
    if restored:
        policy = load_compiled_policy(policy.to_bytes(), runtime_config=config)
    data = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "one",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": r'{"q":"other","z":"other","q":"\u00e9 \ud83d\ude00 needle"}',
                        },
                    },
                ],
            },
        ]
    }
    result = policy.scan(data)
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize(
    "arguments, count",
    [
        ('{"q":"needle"}', 1),
        ('{"q":"old","z":"needle","q":"needle"}', 2),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_outermost_spans_share_raw_argument_coordinates(restored, arguments, count, stream):
    raw = "$raw.spans(assistant_messages[0].tool_calls[0].raw_arguments)"
    decoded = "$decoded.spans(assistant_messages[0].tool_calls[0])"
    outermost = f"outermost(union({raw}, {decoded}))"
    condition = (
        f'all s in {outermost} where messages.extract(s) == "needle"'
        if stream
        else f"len({outermost}) == {count}"
    )
    source = f'rule "Read" strings: $raw="needle" $decoded="eed" condition: {condition}'
    config = QuarryConfig(config_version=2, safety_limits=SafetyLimits(rule_timeout=None))
    policy = compile_policy(source, runtime_config=config)
    if restored:
        policy = load_compiled_policy(policy.to_bytes(), runtime_config=config)
    data = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "one",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": arguments},
                    }
                ],
            }
        ]
    }

    result = policy.scan(data)

    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]
