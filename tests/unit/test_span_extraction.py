# Copyright 2025 Cisco Systems, Inc. and its affilitates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Collect filters over model spans, and the text ``messages.extract(span)`` reads for them.

A fake span adapter returns one span at fixed coordinates. The filter keeps the span only when it
holds, so a match proves the extraction or comparison the row names; a rule whose filter keeps no
span does not match.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from quarry.config import QuarryConfig
from quarry.engine.atoms import ModelExpression as E
from quarry.integrations.model_expression import evaluate_model_expression
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.safety.limits import SafetyLimits
from quarry.scanner import DSLScanner
from quarry.types.messages import Messages
from quarry.types.results import Score, Span


class _FixedSpanAdapter:
    def __init__(self, message_index: int, start: int, end: int) -> None:
        self.span = Span(message_index, start, end, Score(0.9))

    async def classify_messages(self, _config, _messages, _tools, *, timeout=None):
        return SimpleNamespace(chunks=[SimpleNamespace(probabilities=[0.9])])

    async def decode_spans(self, _config, _raw_result, _messages, _threshold):
        return [self.span]


_THREE_TURNS = [
    {"role": "user", "content": "previous"},
    {"role": "assistant", "content": "prefix SECRET suffix"},
    {"role": "user", "content": "next"},
]


def _scan(
    tmp_path,
    extraction: str,
    coordinates: tuple[int, int, int],
    messages: list[dict],
    variables: str = "",
):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "m": {
                    "adapter": "fake",
                    "model": "span-model",
                    "category": "EMAIL",
                    "classify": {"class_index": 0},
                    "spans": {"class_index": 0},
                }
            },
        },
        tmp_path,
        require_rules=False,
    )
    rule = (
        f'rule "Extract"\n{variables}condition: true\ncollect:\n'
        f'    [span for span in model("m").spans(messages, 0.5) if {extraction}]\n'
    )
    config = replace(config, quarry_rule_texts=(rule,))
    backend = ModelInferenceBackend(config, adapters={"fake": _FixedSpanAdapter(*coordinates)})
    return DSLScanner(config, model_inference_backend=backend).scan_data({"messages": messages})


@pytest.mark.parametrize(
    ("extraction", "coordinates", "messages"),
    [
        pytest.param(
            'messages.extract(span) == "josé@example.com"',
            (1, 7, 23),
            [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "Email: josé@example.com"},
            ],
            id="unicode-text-local-to-its-message",
        ),
        pytest.param(
            'messages.extract(span) == "second@example.com"',
            (0, 5, 23),
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first"},
                        {"type": "text", "text": "second@example.com"},
                    ],
                }
            ],
            id="content-parts-concatenate-without-a-separator",
        ),
        pytest.param(
            'messages.extract(span, 4, 4) == "fix SECRET suf"',
            (1, 7, 13),
            _THREE_TURNS,
            id="context-is-added-on-both-sides",
        ),
        pytest.param(
            'messages.extract(span, 100, 100) == "prefix SECRET suffix"',
            (1, 7, 13),
            _THREE_TURNS,
            id="context-never-crosses-into-another-message",
        ),
        pytest.param(
            'messages.extract(span) contains "CRE"',
            (1, 7, 13),
            _THREE_TURNS,
            id="extracted-text-supports-contains",
        ),
    ],
)
def test_extract_returns_the_span_text(tmp_path, extraction, coordinates, messages) -> None:
    result = _scan(tmp_path, extraction, coordinates, messages)

    assert result.rule_matches == ["Extract"]
    assert result.completion_state == "complete"


def test_a_span_beyond_its_message_fails_the_rule_instead_of_extracting_a_prefix(tmp_path):
    result = _scan(tmp_path, 'messages.extract(span) == "x"', (0, 0, 60), _THREE_TURNS)

    assert result.rule_matches == []
    assert result.completion_state == "incomplete"
    assert result.evaluation_errors == {"Extract": "span is outside message content"}


def test_a_variable_is_inlined_into_the_collect_filter(tmp_path) -> None:
    result = _scan(
        tmp_path,
        "span.score.value > cutoff",
        (1, 7, 13),
        _THREE_TURNS,
        variables="variables:\n    cutoff = 0.5\n",
    )

    assert result.rule_matches == ["Extract"]


def test_a_collect_whose_filter_keeps_no_span_does_not_match(tmp_path) -> None:
    result = _scan(
        tmp_path, "span.score.value > 2 and span.score.value > 0", (1, 7, 13), _THREE_TURNS
    )

    assert (result.rule_matches, result.outputs, result.completion_state) == ([], {}, "complete")


# A split piece of one message is built by the model backend, so this checks the offset
# mapping on the message view directly.
def _bound_span(messages: Messages, message_index: int, start: int, end: int) -> Span:
    return messages._bind_span(
        Span(message_index=message_index, start=start, end=end, score=Score(0.9))
    )


def test_span_over_a_divided_message_reports_whole_message_offsets() -> None:
    messages = Messages.from_openai(
        [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "first part. Email: bob@example.com later"},
        ]
    )
    # A chunk holding characters 12..34 of the user turn, as a split piece does.
    piece = messages._select_ranges((0, 1), {1: (12, 34)})
    assert piece[1].content == "Email: bob@example.com"
    assert piece[1].message_index == 1
    # The chunk is one span, from the start of message 0 to character 34 of message 1.
    assert piece.span == Span(0, 0, 34, end_message_index=1)
    assert piece.shares_root_with(messages)

    # A span model reads the piece's text, so it reports offsets 7..22 of that text.
    span = _bound_span(piece, 1, 7, 22)
    assert (span.message_index, span.start, span.end) == (1, 19, 34)
    assert piece.extract(span) == messages.extract(span) == "bob@example.com"
    with pytest.raises(ValueError, match="outside message content"):
        _bound_span(piece, 1, 20, 30)


async def test_input_text_spans_keep_field_provenance_across_a_projection() -> None:
    item = {"id": "n1", "title": "ignore ", "body": "instructions"}
    context = SimpleNamespace(
        conversation=SimpleNamespace(
            messages=Messages.from_openai([{"role": "user", "content": "hi"}])
        ),
        extension_state_values={},
        limits=SafetyLimits(),
        parameters={},
    )
    binding = E("binding", "item")
    projected = E(
        "input_message",
        "records.notes",
        (
            E("literal", "user"),
            E("literal", "title\x1fbody"),
            E("field", "title", (binding,)),
            E("field", "body", (binding,)),
        ),
    )
    source = E(
        "string_spans",
        "phrase",
        (
            projected,
            E("string_patterns", "0000", (E("literal", "ignore instructions"),)),
        ),
    )
    spans = await evaluate_model_expression(source, context, bindings={"item": item})
    span = spans.items[0].value
    assert span.to_dict() == {
        "position": 0,
        "field": "title",
        "start": 0,
        "end": 12,
        "end_position": 0,
        "end_field": "body",
        "found_in": {"message_index": 0, "start": 0, "end": 19},
        "source": {"input": "records.notes", "item": "n1"},
    }
    messages = await evaluate_model_expression(projected, context, bindings={"item": item})
    assert messages.extract(span) == "ignore instructions"
    assert messages._bind_span(Span(0, 0, 19, Score(0.9))).to_dict()["source"] == {
        "input": "records.notes",
        "item": "n1",
    }
    inside = E("span_covered", None, (source, E("index", 0, (source,))))
    assert await evaluate_model_expression(inside, context, bindings={"item": item}) is True
    optional = {"id": "n2", "body": "instructions"}
    body = E(
        "string_spans",
        "body",
        (
            projected,
            E("string_patterns", "0000", (E("literal", "instructions"),)),
        ),
    )
    found = await evaluate_model_expression(body, context, bindings={"item": optional})
    assert found.items[0].value.to_dict()["field"] == "body"


async def test_tool_argument_spans_keep_each_calls_place_and_nested_path() -> None:
    messages = Messages.from_openai(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": name,
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"paths":["safe","/etc/passwd"]}',
                        },
                    }
                    for name in ("c1", "c2")
                ],
            },
            {"role": "user", "content": "done"},
        ]
    )
    context = SimpleNamespace(
        conversation=SimpleNamespace(messages=messages),
        limits=SafetyLimits(),
        parameters={},
    )
    source = E(
        "string_spans",
        "path",
        (
            E("span_text", "arguments.paths.*", (E("binding", "call"),)),
            E("string_patterns", "0000", (E("literal", "/etc"),)),
        ),
    )
    first, second = messages.tool_calls
    first_span = (
        (await evaluate_model_expression(source, context, bindings={"call": first})).items[0].value
    )
    second_span = (
        (await evaluate_model_expression(source, context, bindings={"call": second})).items[0].value
    )
    assert first_span.to_dict() == {
        "position": 1,
        "field": "arguments.paths.1",
        "start": 0,
        "end": 4,
    }
    assert second_span.position == 2
    assert messages.extract(first_span) == "/etc"
    place = E("view_spans", None, (E("binding", "call"),))
    assert (await evaluate_model_expression(place, context, bindings={"call": first})).items[
        0
    ].value.position == 1
    covered = E("span_covered", None, (place, E("binding", "span")))
    assert (
        await evaluate_model_expression(
            covered, context, bindings={"call": first, "span": second_span}
        )
        is False
    )
    user_span = messages._bind_span(Span(1, 0, 4, Score(0.9)))
    user_place = E("view_spans", None, (E("message_view", "user_messages"),))
    assert user_span.to_dict() == {
        "message_index": 1,
        "start": 0,
        "end": 4,
        "score": {"value": 0.9},
    }
    assert (
        await evaluate_model_expression(
            E("span_covered", None, (user_place, E("binding", "span"))),
            context,
            bindings={"span": user_span},
        )
        is True
    )
