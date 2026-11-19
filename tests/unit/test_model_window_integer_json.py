# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Native integer JSON rendering stays bounded, cooperative and byte compatible."""

import asyncio
import json
import sys

import pytest

from quarry.integrations import model_window_json
from quarry.integrations.model_window_input import json_window_input
from quarry.types.messages import Messages


class HostInteger(int):
    def __int__(self):
        raise AssertionError("integer conversion must not call host overrides")

    def __str__(self):
        raise AssertionError("integer spelling must not call host overrides")

    def __repr__(self):
        raise AssertionError("integer spelling must not call host overrides")


def tools_with(value):
    return [
        {
            "type": "function",
            "function": {
                "name": "synthetic",
                "parameters": {"type": "object", "properties": {"value": {"default": value}}},
            },
        }
    ]


def messages():
    return Messages.from_openai([{"role": "user", "content": "prefix needle"}])


@pytest.mark.asyncio
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("host_integer", [False, True])
async def test_oversized_integer_is_rejected_without_whole_decimal_conversion(
    monkeypatch, negative, host_integer
):
    number = (-(10**2000) if negative else 10**2000) + 17
    number = HostInteger(number) if host_integer else number
    whole_conversions = []
    original = model_window_json.json.dumps

    def observe(value, *args, **kwargs):
        if value is number:
            whole_conversions.append(True)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(model_window_json.json, "dumps", observe)
    monkeypatch.setattr(model_window_json, "MAX_WINDOW_RENDER_CHARACTERS", 256)

    with pytest.raises(ValueError, match="render limit"):
        await json_window_input(messages(), tools_with(number))

    assert whole_conversions == []


@pytest.mark.asyncio
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("host_integer", [False, True])
@pytest.mark.parametrize("turns", [1, 2, 3])
async def test_cancellation_at_large_integer_field_prevents_decimal_output(
    monkeypatch, negative, host_integer, turns
):
    number = -(10**2000) if negative else 10**2000
    number = HostInteger(number) if host_integer else number
    requested = False
    emitted = []
    original = model_window_json._Writer.emit

    def cancel_after(task, remaining):
        if remaining:
            asyncio.get_running_loop().call_soon(cancel_after, task, remaining - 1)
        else:
            task.cancel()

    def observe(writer, text, mapping=None):
        nonlocal requested
        if text == ":" and writer.texts[-2:] == ["default", '"']:
            requested = True
            task = asyncio.current_task()
            assert task is not None
            asyncio.get_running_loop().call_soon(cancel_after, task, turns - 1)
        elif requested and len(text) > 512 and text.lstrip("-").isdecimal():
            emitted.append(len(text))
        return original(writer, text, mapping)

    monkeypatch.setattr(model_window_json._Writer, "emit", observe)
    task = asyncio.create_task(json_window_input(messages(), tools_with(number)))

    with pytest.raises(asyncio.CancelledError):
        await task

    assert requested
    assert emitted == []


@pytest.mark.asyncio
@pytest.mark.parametrize("digits", [511, 512, 513, 1024, 2001])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("host_integer", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
async def test_integer_context_matches_json_and_preserves_input_origins(
    digits, negative, host_integer, sparse
):
    number = 10 ** (digits - 1) + 17 if sparse else 10**digits - 1
    number = -number if negative else number
    number = HostInteger(number) if host_integer else number
    tools = tools_with(number)
    selected = messages()

    rendered = await json_window_input(selected, tools)

    assert rendered.context == json.dumps(tools, ensure_ascii=True, separators=(",", ":"))
    assert rendered.source.text == json.dumps(
        selected.to_openai_messages(), ensure_ascii=False, separators=(",", ":")
    )
    start = rendered.source.text.index("needle")
    span = rendered.source.span(start, start + 6)
    assert (span.message_index, span.start, span.end, span.precision) == (0, 7, 13, "exact")
    assert selected.extract(span) == "needle"
    low, high = rendered.last_turn
    assert json.loads(rendered.source.text[low:high]) == selected[0].to_openai_dict()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, True, False, None, 1.25])
async def test_other_native_json_scalars_retain_their_wire_spelling(value):
    tools = tools_with(value)

    rendered = await json_window_input(messages(), tools)

    assert rendered.context == json.dumps(tools, ensure_ascii=True, separators=(",", ":"))


@pytest.mark.asyncio
@pytest.mark.parametrize("digits", [639, 640, 641])
@pytest.mark.parametrize("negative", [False, True])
async def test_active_python_integer_digit_limit_is_preserved(digits, negative):
    number = 10 ** (digits - 1)
    number = -number if negative else number
    tools = tools_with(number)
    previous = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(640)
        if digits > 640:
            with pytest.raises(ValueError) as standard:
                json.dumps(tools)
            with pytest.raises(ValueError) as rendered:
                await json_window_input(messages(), tools)
            assert str(rendered.value) == str(standard.value)
        else:
            rendered = await json_window_input(messages(), tools)
            assert rendered.context == json.dumps(tools, separators=(",", ":"))
    finally:
        sys.set_int_max_str_digits(previous)


@pytest.mark.asyncio
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("offset", [-1, 0, 1])
async def test_integer_context_retains_the_exact_combined_character_limit(
    monkeypatch, negative, offset
):
    number = -(10**2000) if negative else 10**2000
    selected, tools = messages(), tools_with(number)
    source = json.dumps(selected.to_openai_messages(), ensure_ascii=False, separators=(",", ":"))
    context = json.dumps(tools, separators=(",", ":"))
    monkeypatch.setattr(
        model_window_json, "MAX_WINDOW_RENDER_CHARACTERS", len(source) + len(context) + offset
    )

    if offset < 0:
        with pytest.raises(ValueError, match="character limit"):
            await json_window_input(selected, tools)
    else:
        result = await json_window_input(selected, tools)
        assert result.context == context


@pytest.mark.asyncio
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("host_integer", [False, True])
async def test_disabled_python_digit_limit_keeps_large_integers_exact(
    monkeypatch, negative, host_integer
):
    number = -(10**5000) if negative else 10**5000
    number = HostInteger(number) if host_integer else number
    tools = tools_with(number)
    previous = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(0)
        expected = json.dumps(tools, separators=(",", ":"))
        original = model_window_json.json.dumps

        def no_whole_integer(value, *args, **kwargs):
            if value is number:
                raise AssertionError("large native integers must yield during conversion")
            return original(value, *args, **kwargs)

        monkeypatch.setattr(model_window_json.json, "dumps", no_whole_integer)
        result = await json_window_input(messages(), tools)
        assert result.context == expected
    finally:
        sys.set_int_max_str_digits(previous)
