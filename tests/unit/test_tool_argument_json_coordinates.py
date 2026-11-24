# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Decoded argument containment refers to the exact original JSON characters."""

import json
from dataclasses import replace

import pytest

from quarry.engine.matchers import value_texts
from quarry.integrations.model_expression import _cover_index, _is_covered
from quarry.safety.timer import QuarryTimeoutError
from quarry.types.json_source import JSONSource
from quarry.types.messages import Messages
from quarry.types.results import InputText, Span


@pytest.mark.parametrize(
    ("encoded", "decoded"),
    [
        ("é", "é"),
        (r"\u00e9", "é"),
        ("😀", "😀"),
        (r"\ud83d\ude00", "😀"),
        (r"\uD83D\uDE00", "😀"),
        (r"\\", "\\"),
        (r"\"", '"'),
        (r"\/", "/"),
        (r"\b", "\b"),
        (r"\f", "\f"),
        (r"\n", "\n"),
        (r"\r", "\r"),
        (r"\t", "\t"),
        (r"\ud800", "\ud800"),
    ],
)
@pytest.mark.parametrize("selection", ["middle", "prefix", "suffix", "empty"])
def test_decoded_substring_bounds_select_exact_original_characters(encoded, decoded, selection):
    prefix = '{"q":"'
    raw = prefix + "a" + encoded + 'z"}'
    text = json.loads(raw)["q"]
    assert text == "a" + decoded + "z"
    start, end, expected = {
        "middle": (1, 2, encoded),
        "prefix": (0, 2, "a" + encoded),
        "suffix": (1, 3, encoded + "z"),
        "empty": (2, 2, ""),
    }[selection]
    source = JSONSource(raw)
    # Repeat to cover both a fresh location and the retained escape index.
    for _ in range(2):
        low, high = source.bounds(("q",), text, start, end)
        assert raw[low:high] == expected
        assert low == len(prefix) + (0 if start == 0 else 1 if start == 1 else 1 + len(encoded))
        assert high == low + len(expected)


@pytest.mark.parametrize(
    ("raw", "path", "text", "expected"),
    [
        ('{"q":"old","q":"new"}', ("q",), "new", "new"),
        ('{"q":{"a.b":[null,{"c\\\\d":"needle"}]}}', ("q", "a.b", "1", "c\\d"), "needle", "needle"),
        ('{"q":1e2}', ("q",), "100.0", "1e2"),
        ('{"q":-0.0}', ("q",), "-0.0", "0"),
        ('{"q":true}', ("q",), "true", "r"),
        ('{"q":12345}', ("q",), "12345", "2"),
    ],
)
def test_scalar_paths_and_original_numeric_spelling(raw, path, text, expected):
    # Changed numeric spellings conservatively depend on the complete raw token.
    start, end = (0, len(text)) if text in {"new", "needle"} else (1, 2)
    low, high = JSONSource(raw).bounds(path, text, start, end)
    assert raw[low:high] == expected
    if text == "new":
        assert low == raw.rindex('"new"') + 1


def _call(raw):
    return Messages.from_openai(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "synthetic",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": raw},
                    }
                ],
            }
        ]
    ).tool_calls[0]


def test_multiple_values_cover_original_order_after_duplicate_keys():
    raw = '{"first":"old","last":"tail","first":"replacement"}'
    call = _call(raw)
    low, high = call._argument_bounds("arguments.first", 0, "arguments.last", 4)
    assert raw[low:high] == 'tail","first":"replacement'
    assert (low, high) == (raw.index("tail"), raw.index("replacement") + len("replacement"))


def test_escaped_field_paths_keep_distinct_json_locations():
    raw = r'{"a.b":{"c\\d":"first"},"a":{"b":"second"}}'
    call = _call(raw)
    fields = dict(value_texts(call.arguments, "arguments"))
    assert len(fields) == 2
    for field, text in fields.items():
        low, high = call._argument_bounds(field, 0, field, len(text))
        assert raw[low:high] == text


def test_decoded_source_cache_is_lazy_reused_and_discarded():
    call = _call(r'{"q":"a\u00e9z"}')
    assert call._argument_source is None
    first = call._argument_bounds("arguments.q", 1, "arguments.q", 2)
    retained = call._argument_source
    assert retained is not None
    assert call._argument_bounds("arguments.q", 1, "arguments.q", 2) == first
    assert call._argument_source is retained
    call._discard_decoded_arguments()
    assert call._argument_source is None and call._arguments_cache is None
    assert call._argument_bounds("arguments.q", 1, "arguments.q", 2) == first
    assert call._argument_source is not retained


@pytest.mark.parametrize("kind", ["decoded", "whole-call"])
def test_containment_preserves_explicit_input_owner_identity(kind):
    first, second = _call('{"q":"needle"}'), _call('{"q":"needle"}')
    source = InputText("synthetic", "same-name", ("arguments.q",))
    outer = Span(
        0, 0, 6, source=source, source_field="arguments.q", _whole_call=kind == "whole-call"
    )._bind_to_source(first)
    inner = Span(0, 0, 6, source=source, source_field="arguments.q")._bind_to_source(first)
    assert _is_covered(_cover_index([outer]), inner)
    assert not _is_covered(_cover_index([outer]), inner._bind_to_source(second))
    assert not _is_covered(
        _cover_index([outer]),
        replace(inner, source=InputText("other", "same-name", ("arguments.q",))),
    )


def test_json_coordinate_index_observes_active_deadline(monkeypatch):
    def expired():
        raise QuarryTimeoutError("synthetic coordinate deadline")

    monkeypatch.setattr("quarry.types.json_source.raise_if_evaluation_expired", expired)
    with pytest.raises(QuarryTimeoutError, match="synthetic coordinate deadline"):
        JSONSource('{"q":"needle"}')


def test_lazy_escape_index_observes_active_deadline(monkeypatch):
    source = JSONSource(r'{"q":"a\u00e9z"}')

    def expired():
        raise QuarryTimeoutError("synthetic coordinate deadline")

    monkeypatch.setattr("quarry.types.json_source.raise_if_evaluation_expired", expired)
    with pytest.raises(QuarryTimeoutError, match="synthetic coordinate deadline"):
        source.bounds(("q",), "aéz", 1, 2)
