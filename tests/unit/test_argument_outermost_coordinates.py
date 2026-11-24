# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Outermost selection agrees with containment across raw and decoded arguments."""

import copy
from dataclasses import replace

import pytest

from quarry import DSLScanner, compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.integrations.model_expression import _outermost_spans
from quarry.safety.limits import SafetyLimits
from quarry.types.messages import Messages, ToolCall
from quarry.types.results import InputText, ResultItem, Score, Span


def _call(raw, name="first"):
    return {"id": name, "type": "function", "function": {"name": "lookup", "arguments": raw}}


def _case(name):
    raw = '{"q":"needle"}'
    outer, inner = '"needle"', '"eed"'
    first = "assistant_messages[0].tool_calls[0]"
    raw_receiver, decoded_receiver = first + ".raw_arguments", first + ".arguments.q"
    expected, fields = ["needle"], ["raw_arguments"]
    second = None
    if name == "decoded-outer":
        outer, inner = inner, outer
        fields = ["arguments.q"]
    elif name == "equal-range":
        inner = outer
        fields = None
    elif name == "escaped-outer":
        raw = r'{"q":"\ud83d\ude00 needle"}'
        outer = '".ud83d.ude00 needle" regex'
        expected = [r"\ud83d\ude00 needle"]
    elif name == "different-calls":
        second = _call(raw, "second")
        decoded_receiver = "assistant_messages[0].tool_calls[1].arguments.q"
        expected, fields = ["needle", "eed"], ["raw_arguments", "arguments.q"]
    elif name == "different-fields":
        raw = '{"q":"needle","r":"feed"}'
        decoded_receiver = first + ".arguments.r"
        expected, fields = ["needle", "eed"], ["raw_arguments", "arguments.r"]
    elif name == "duplicate-key-order":
        raw = '{"q":"old","r":"needle","q":"needle"}'
        expected, fields = ["needle", "needle"], ["raw_arguments", "raw_arguments"]
    else:
        assert name == "raw-outer"
    data = {"messages": [{"role": "assistant", "content": "", "tool_calls": [_call(raw)]}]}
    if second is not None:
        data["messages"][0]["tool_calls"].append(second)
    return raw, outer, inner, raw_receiver, decoded_receiver, data, expected, fields


@pytest.mark.parametrize("strategy", ["scanner", "compiled", "restored"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "name",
    [
        "raw-outer",
        "decoded-outer",
        "equal-range",
        "escaped-outer",
        "different-calls",
        "different-fields",
        "duplicate-key-order",
    ],
)
def test_outermost_uses_original_argument_coordinates(name, reverse, strategy):
    raw, outer, inner, raw_receiver, decoded_receiver, data, expected, fields = _case(name)
    sources = [f"$outer.spans({raw_receiver})", f"$inner.spans({decoded_receiver})"]
    if reverse:
        sources.reverse()
    selected = "outermost(union(" + ", ".join(sources) + "))"
    text = (
        'rule "Read"\nstrings:\n    $outer = '
        + outer
        + "\n    $inner = "
        + inner
        + f"\ncondition: len({selected}) == {len(expected)}\ncollect: [s for s in {selected} if true]"
    )
    config = QuarryConfig(config_version=2, safety_limits=SafetyLimits(rule_timeout=None))
    original = copy.deepcopy(data)
    if strategy == "scanner":
        result = DSLScanner(replace(config, quarry_rule_texts=(text,))).scan_data(data)
    else:
        policy = compile_policy(text, runtime_config=config)
        if strategy == "restored":
            wire = policy.to_bytes()
            policy = load_compiled_policy(wire, runtime_config=config)
            assert policy.to_bytes() == wire
        result = policy.scan(data)
    assert data == original
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]
    spans = [item.value for item in result.outputs["Read"].items]
    observed = []
    for span in spans:
        assert isinstance(span._source, ToolCall)
        body = span._source.argument_text(span.field)
        assert isinstance(body, str)
        observed.append(body[span.start : span.end])
    assert observed == expected
    if fields is not None:
        assert [span.field for span in spans] == fields
    if name == "duplicate-key-order":
        assert [span.start for span in spans] == [raw.index("needle"), raw.rindex("needle")]


@pytest.mark.parametrize("reverse", [False, True])
def test_equal_original_ranges_keep_the_highest_score(reverse):
    call = Messages.from_openai(
        [{"role": "assistant", "content": "", "tool_calls": [_call('{"q":"needle"}')]}]
    ).tool_calls[0]
    raw = Span(0, 6, 12, score=Score(0.2), source_field="raw_arguments")._bind_to_source(call)
    decoded = Span(0, 0, 6, score=Score(0.8), source_field="arguments.q")._bind_to_source(call)
    items = [ResultItem("raw", raw), ResultItem("decoded", decoded)]
    if reverse:
        items.reverse()
    kept = _outermost_spans(tuple(items))
    assert len(kept) == 1 and kept[0].value is decoded


@pytest.mark.parametrize("whole_call", [False, True])
def test_outermost_keeps_different_explicit_input_owners(whole_call):
    calls = [
        Messages.from_openai(
            [{"role": "assistant", "content": "", "tool_calls": [_call('{"q":"needle"}')]}]
        ).tool_calls[0]
        for _ in range(2)
    ]
    source = InputText("synthetic", "same", ("arguments.q",))
    first = Span(
        0, 0, 6, source=source, source_field="arguments.q", _whole_call=whole_call
    )._bind_to_source(calls[0])
    second = Span(0, 0, 6, source=source, source_field="arguments.q")._bind_to_source(calls[1])
    kept = _outermost_spans((ResultItem("first", first), ResultItem("second", second)))
    assert {item.item_id for item in kept} == {"first", "second"}
