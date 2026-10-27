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
from quarry.integrations.model_inference import ModelInferenceBackend
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
