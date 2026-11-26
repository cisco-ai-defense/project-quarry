# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Configured local content transforms and source-character provenance."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import fields, replace

import orjson
import pytest

import quarry.integrations.content_transforms as transform_runtime
import quarry.types.messages as messages_module
from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.dsl.parser import DSLParser
from quarry.engine.atoms import ModelExpression, validate_transform_expression_graph
from quarry.engine.compiler import compile_rule
from quarry.integrations.content_transforms import apply_content_transform
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.types.dsl import FunctionCall
from quarry.types.dsl.render import render_condition
from quarry.types.exceptions import (
    ConfigurationError,
    InvalidInputError,
)
from quarry.types.messages import AssistantMessage, Messages, ToolMessage
from quarry.types.results import Score, Span
from quarry.types.transforms import ContentTransform, TransformOperation
from tests.unit.model_fakes import ModelAdapter, model_entry, v2_scanner


def _transform(*operations: TransformOperation) -> ContentTransform:
    return ContentTransform(operations)


def _definitions(*operations: dict) -> dict:
    return {"clean": {"operations": list(operations)}}


@pytest.mark.asyncio
async def test_chained_operations_keep_exact_and_replacement_provenance() -> None:
    source = Messages.from_openai([{"role": "user", "content": "  Straße TOKEN  "}])
    normalized = await apply_content_transform(
        source,
        _transform(
            TransformOperation("strip"),
            TransformOperation("casefold"),
            TransformOperation("replace", "token", "item"),
        ),
    )

    assert normalized.to_openai_messages() == [{"role": "user", "content": "strasse item"}]
    assert [item.to_dict() for item in normalized.source_ranges(0, 4, 6)] == [
        {"message_index": 0, "start": 6, "end": 7}
    ]
    assert [item.to_dict() for item in normalized.source_ranges(0, 8, 12)] == [
        {"message_index": 0, "start": 9, "end": 14}
    ]
    assert source.to_openai_messages() == [{"role": "user", "content": "  Straße TOKEN  "}]


@pytest.mark.asyncio
async def test_multimodal_text_parts_transform_without_changing_media() -> None:
    wire = {
        "role": "user",
        "content": [
            {"type": "text", "text": " A "},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            {"type": "text", "text": "Straße"},
        ],
    }
    source = Messages.from_openai([wire])
    transformed = await apply_content_transform(
        source,
        _transform(TransformOperation("strip"), TransformOperation("casefold")),
    )

    assert transformed.to_openai_messages() == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "text", "text": "strasse"},
            ],
        }
    ]
    assert [item.to_dict() for item in transformed.source_ranges(0, 0, 1)] == [
        {"message_index": 0, "part_index": 0, "start": 1, "end": 2}
    ]
    assert [item.to_dict() for item in transformed.source_ranges(0, 5, 7)] == [
        {"message_index": 0, "part_index": 2, "start": 4, "end": 5}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("part_count", [250, 4_000])
async def test_multipart_provenance_partition_has_exact_linear_charged_visits(
    part_count,
) -> None:
    source = Messages.from_openai(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "é"} for _ in range(part_count)],
            }
        ]
    )
    metrics = transform_runtime._TransformMetrics()

    transformed = await apply_content_transform(
        source,
        _transform(TransformOperation("strip")),
        _metrics=metrics,
    )

    assert transformed[0].content == "é" * part_count
    assert metrics.part_visits == part_count
    assert metrics.range_visits == part_count
    assert metrics.segment_visits == 2 * part_count
    assert metrics.character_visits == part_count
    assert metrics.output_characters == part_count
    assert metrics.work_units == 6 * part_count


@pytest.mark.asyncio
async def test_multipart_partition_preserves_empty_shared_and_multimodal_boundaries() -> None:
    source = Messages.from_openai(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "É"},
                    {"type": "text", "text": ""},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    {"type": "text", "text": "ß"},
                    {"type": "text", "text": ""},
                ],
            }
        ]
    )
    metrics = transform_runtime._TransformMetrics()

    transformed = await apply_content_transform(
        source,
        _transform(TransformOperation("casefold")),
        _metrics=metrics,
    )

    assert transformed[0].content == "éss"
    assert metrics.part_visits == 5
    assert metrics.range_visits == 4
    assert [item.to_dict() for item in transformed.source_ranges(0, 0, 1)] == [
        {"message_index": 0, "part_index": 0, "start": 0, "end": 1}
    ]
    assert [item.to_dict() for item in transformed.source_ranges(0, 1, 3)] == [
        {"message_index": 0, "part_index": 3, "start": 0, "end": 1}
    ]


@pytest.mark.asyncio
async def test_multipart_partition_work_limit_fails_at_a_deterministic_visit(
    monkeypatch,
) -> None:
    monkeypatch.setattr(transform_runtime, "MAX_TRANSFORM_WORK", 5)
    source = Messages.from_openai(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": ""} for _ in range(4_000)],
            }
        ]
    )
    metrics = transform_runtime._TransformMetrics()

    with pytest.raises(ValueError, match="5 character-work units"):
        await apply_content_transform(
            source,
            _transform(TransformOperation("strip")),
            _metrics=metrics,
        )

    assert (metrics.part_visits, metrics.range_visits, metrics.work_units) == (3, 3, 6)


@pytest.mark.asyncio
async def test_multipart_partition_checks_deadline_and_cancellation_at_bounded_work(
    monkeypatch,
) -> None:
    source = Messages.from_openai(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": ""} for _ in range(4_000)],
            }
        ]
    )
    checks = 0

    def expire_during_partition() -> None:
        nonlocal checks
        checks += 1
        if checks == 4:
            raise RuntimeError("partition deadline reached")

    metrics = transform_runtime._TransformMetrics()
    monkeypatch.setattr(
        transform_runtime,
        "raise_if_evaluation_expired",
        expire_during_partition,
    )
    with pytest.raises(RuntimeError, match="partition deadline reached"):
        await apply_content_transform(
            source,
            _transform(TransformOperation("strip")),
            _metrics=metrics,
        )
    assert (metrics.part_visits, metrics.range_visits, metrics.work_units) == (384, 384, 768)

    monkeypatch.setattr(transform_runtime, "raise_if_evaluation_expired", lambda: None)
    real_sleep = transform_runtime.asyncio.sleep
    yields = 0

    async def counted_sleep(delay: float) -> None:
        nonlocal yields
        yields += 1
        await real_sleep(delay)

    monkeypatch.setattr(transform_runtime.asyncio, "sleep", counted_sleep)
    metrics = transform_runtime._TransformMetrics()
    await apply_content_transform(
        source,
        _transform(TransformOperation("strip")),
        _metrics=metrics,
    )

    assert metrics.checkpoints == 32
    assert yields >= metrics.checkpoints // 16


@pytest.mark.asyncio
async def test_insertions_fail_closed_while_deletions_keep_remaining_positions() -> None:
    source = Messages.from_openai([{"role": "user", "content": "axb"}])
    transformed = await apply_content_transform(
        source,
        _transform(
            TransformOperation("prefix", "<"),
            TransformOperation("replace", "x", ""),
            TransformOperation("suffix", ">"),
        ),
    )

    assert transformed[0].content == "<ab>"
    with pytest.raises(ValueError, match="inserted text"):
        transformed.source_ranges(0, 0, 1)
    assert [item.to_dict() for item in transformed.source_ranges(0, 1, 3)] == [
        {"message_index": 0, "start": 0, "end": 1},
        {"message_index": 0, "start": 2, "end": 3},
    ]
    with pytest.raises(ValueError, match="outside message content"):
        transformed.source_ranges(0, 1, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operations", "expected", "mapped_start"),
    [
        pytest.param(
            (
                TransformOperation("prefix", "X"),
                TransformOperation("replace", "Xa", "Q"),
            ),
            "Qbc",
            1,
            id="prefix-and-source",
        ),
        pytest.param(
            (
                TransformOperation("suffix", "X"),
                TransformOperation("replace", "cX", "Q"),
            ),
            "abQ",
            0,
            id="source-and-suffix",
        ),
        pytest.param(
            (
                TransformOperation("prefix", "X"),
                TransformOperation("replace", "Xa", "Q"),
                TransformOperation("replace", "Qb", "R"),
            ),
            "Rc",
            1,
            id="chained-mixed-replacement",
        ),
    ],
)
async def test_replacement_crossing_inserted_text_never_invents_source_coordinates(
    operations, expected, mapped_start
) -> None:
    source = Messages.from_openai([{"role": "user", "content": "abc"}])
    transformed = await apply_content_transform(source, _transform(*operations))

    assert transformed[0].content == expected
    replacement = expected.index("Q") if "Q" in expected else expected.index("R")
    with pytest.raises(ValueError, match="inserted text"):
        transformed.source_ranges(0, replacement, replacement + 1)
    boundary_start = max(0, replacement - 1)
    boundary_end = min(len(expected), replacement + 2)
    with pytest.raises(ValueError, match="inserted text"):
        transformed.source_ranges(0, boundary_start, boundary_end)
    mapped = transformed.source_ranges(0, mapped_start, mapped_start + 1)
    assert len(mapped) == 1


@pytest.mark.asyncio
async def test_bound_transformed_span_round_trips_and_reports_part_offsets() -> None:
    source = Messages.from_openai(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before "},
                    {"type": "text", "text": "SECRET after"},
                ],
            }
        ]
    )
    transformed = await apply_content_transform(
        source,
        _transform(TransformOperation("replace", "SECRET", "item")),
    )
    span = transformed._bind_span(Span(0, 7, 11, Score(0.9)))

    assert span.to_dict() == {
        "message_index": 0,
        "start": 7,
        "end": 13,
        "score": {"value": 0.9},
        "source_ranges": [{"message_index": 0, "part_index": 1, "start": 0, "end": 6}],
    }
    assert transformed.extract(span) == "item"
    assert source.extract(span) == "SECRET"


@pytest.mark.asyncio
async def test_transform_builds_a_view_local_tool_graph_and_rejects_duplicates() -> None:
    source = Messages.from_openai(
        [
            {
                "role": "assistant",
                "content": "CALL",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "RESULT"},
        ]
    )
    transformed = await apply_content_transform(source, _transform(TransformOperation("casefold")))
    assistant, response = transformed

    assert isinstance(assistant, AssistantMessage)
    assert isinstance(response, ToolMessage)
    assert assistant.message_index == 0
    assert assistant.tool_calls[0] is not source[0].tool_calls[0]
    assert response is not source[1]
    assert assistant.tool_calls[0].response is response
    assert response.call is assistant.tool_calls[0]
    assert source[0].tool_calls[0].response is source[1]
    assert source[1].call is source[0].tool_calls[0]
    transformed.assert_self_contained()
    with pytest.raises(ValueError, match="duplicate tool-call"):
        Messages.combine((transformed, transformed))


@pytest.mark.asyncio
async def test_call_and_response_only_transforms_never_link_to_original_counterparts() -> None:
    source = Messages.from_openai(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": " CALL "},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ],
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": " RESULT "},
        ]
    )
    transform = _transform(TransformOperation("strip"), TransformOperation("casefold"))

    calls = await apply_content_transform(source.assistant_messages, transform)
    call_message = calls[0]
    assert isinstance(call_message, AssistantMessage)
    call = call_message.tool_calls[0]
    assert call_message.content == "call"
    assert call.response is not source[1]
    assert call.response.content == " RESULT "
    assert call.response.call is call

    responses = await apply_content_transform(source.tool_responses, transform)
    response = responses[0]
    assert isinstance(response, ToolMessage)
    assert response.content == "result"
    assert response.call is not source[0].tool_calls[0]
    assert response.call.response is response

    assert source[0].content == " CALL "
    assert source[1].content == " RESULT "
    assert source[0].tool_calls[0].response is source[1]


@pytest.mark.asyncio
async def test_transformed_tool_graph_survives_filters_and_composition_without_aliasing() -> None:
    source = Messages.from_openai(
        [
            {"role": "user", "content": " PROMPT "},
            {
                "role": "assistant",
                "content": " CALL ",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": " RESULT "},
        ]
    )
    transformed = await apply_content_transform(
        source,
        _transform(TransformOperation("strip"), TransformOperation("casefold")),
    )
    composed = Messages.combine((transformed.assistant_messages, transformed.tool_responses))
    assistant, response = composed

    assert isinstance(assistant, AssistantMessage)
    assert isinstance(response, ToolMessage)
    assert assistant.tool_calls[0].response is response
    assert response.call is assistant.tool_calls[0]
    assert (assistant.content, response.content) == ("call", "result")
    assert assistant is not source[1]
    assert response is not source[2]
    composed.assert_self_contained()


@pytest.mark.asyncio
async def test_independently_transformed_tool_views_relink_only_visible_variants() -> None:
    source = Messages.from_openai(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": " CALL "},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ],
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": " RESULT "},
        ]
    )
    call_view = await apply_content_transform(
        source.assistant_messages,
        _transform(TransformOperation("strip")),
    )
    response_view = await apply_content_transform(
        source.tool_responses,
        _transform(TransformOperation("casefold")),
    )

    combined = Messages.combine((call_view.assistant_messages, response_view.tool_responses))
    assistant, response = combined
    assert isinstance(assistant, AssistantMessage)
    assert isinstance(response, ToolMessage)
    assert (assistant.content, response.content) == ("CALL", " result ")
    assert assistant.tool_calls[0].response is response
    assert response.call is assistant.tool_calls[0]
    assert assistant.tool_calls[0] is not call_view[0].tool_calls[0]
    assert response is not response_view[0]
    combined.assert_self_contained()

    calls_only = Messages.combine((call_view.assistant_messages,))
    assert isinstance(calls_only[0], AssistantMessage)
    assert calls_only[0].tool_calls[0].response is None
    responses_only = Messages.combine((response_view.tool_responses,))
    assert isinstance(responses_only[0], ToolMessage)
    with pytest.raises(InvalidInputError, match="no exact call link"):
        _ = responses_only[0].call
    with pytest.raises(ValueError, match="before its originating"):
        responses_only.assert_self_contained()

    assert source[0].tool_calls[0].response is source[1]
    assert source[1].call is source[0].tool_calls[0]


@pytest.mark.asyncio
async def test_combining_independent_equal_tool_transforms_rejects_repeated_units() -> None:
    source = Messages.from_openai(
        [
            {
                "role": "assistant",
                "content": " A ",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": " B "},
        ]
    )
    transform = _transform(TransformOperation("strip"))
    first = await apply_content_transform(source, transform)
    second = await apply_content_transform(source, transform)

    combined = Messages.combine((first.assistant_messages, second.tool_responses))
    assert isinstance(combined[0], AssistantMessage)
    assert isinstance(combined[1], ToolMessage)
    assert combined[0].tool_calls[0].response is combined[1]
    assert combined[1].call is combined[0].tool_calls[0]
    combined.assert_self_contained()

    with pytest.raises(ValueError, match="duplicate tool-call"):
        Messages.combine((first.assistant_messages, second.assistant_messages))
    with pytest.raises(ValueError, match="different content variants"):
        different = await apply_content_transform(
            source.assistant_messages,
            _transform(TransformOperation("casefold")),
        )
        Messages.combine((first.assistant_messages, different.assistant_messages))


@pytest.mark.asyncio
async def test_transformed_constructed_text_composes_without_source_attribution() -> None:
    source = Messages.from_openai(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "source"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    },
                ],
            }
        ]
    )
    constructed = Messages().add_message("assistant", "  MADE  ")
    transformed = await apply_content_transform(
        constructed,
        _transform(TransformOperation("strip"), TransformOperation("casefold")),
    )

    combined = Messages.combine((source, transformed))

    assert combined.to_openai_messages() == [
        source.to_openai_messages()[0],
        {"role": "assistant", "content": "made"},
    ]
    assert combined[0].content_parts[1].type == "image_url"
    with pytest.raises(ValueError, match="inserted text"):
        combined.source_ranges(1, 0, 4)
    assert source.to_openai_messages()[0]["content"][0]["text"] == "source"


@pytest.mark.asyncio
async def test_constructed_mixed_range_reports_mapped_and_unmapped_portions() -> None:
    source = Messages.from_openai([{"role": "user", "content": "source"}])
    copied = Messages()._add_synthetic_message(
        "assistant",
        source[0].content,
        identity=(19, 0),
        provenance=source[0]._content_provenance(),
    )
    transformed = await apply_content_transform(
        copied,
        _transform(TransformOperation("prefix", ">")),
    )

    span = transformed._bind_span(Span(0, 0, 7, Score(0.9)))

    assert span.to_dict() == {
        "message_index": 0,
        "start": 0,
        "end": 7,
        "score": {"value": 0.9},
        "source_ranges": [{"message_index": 0, "start": 0, "end": 6}],
        "source_unmapped": True,
    }
    with pytest.raises(ValueError, match="inserted text"):
        transformed.source_ranges(0, 0, 7)
    assert [item.to_dict() for item in transformed.source_ranges(0, 1, 7)] == [
        {"message_index": 0, "start": 0, "end": 6}
    ]
    assert transformed.extract(span) == ">source"


def _many_tool_call_messages(count: int) -> Messages:
    return Messages.from_openai(
        [
            {
                "role": "assistant",
                "content": " CALL ",
                "tool_calls": [
                    {
                        "id": f"call-{index}",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                    for index in range(count)
                ],
            }
        ]
    )


def _charged_combine_work(metrics) -> int:
    return sum(
        getattr(metrics, field.name)
        for field in fields(metrics)
        if field.name not in {"checkpoints", "work_units"}
    )


@pytest.mark.asyncio
async def test_duplicate_tool_variant_projection_is_identity_memoized_and_fully_metered() -> None:
    source = _many_tool_call_messages(1_000)
    transform = _transform(TransformOperation("strip"))
    first = await apply_content_transform(source.assistant_messages, transform)
    second = await apply_content_transform(source.assistant_messages, transform)
    observed = {}

    for repetitions in (2, 256):
        metrics = messages_module._CombineMetrics()
        parts = tuple((first, second)[index % 2] for index in range(repetitions))
        with pytest.raises(ValueError, match="duplicate tool-call ID"):
            Messages.combine(parts, _metrics=metrics)
        assert metrics.selection_visits == repetitions
        assert metrics.work_units == _charged_combine_work(metrics)
        observed[repetitions] = metrics

    pair = observed[2]
    alternating = observed[256]
    assert pair.variant_projection_visits > 2_000
    assert pair.variant_field_visits > 2_000
    assert pair.variant_character_visits > 2_000
    assert alternating.variant_projection_visits == pair.variant_projection_visits
    assert alternating.variant_field_visits == pair.variant_field_visits
    assert alternating.variant_character_visits == pair.variant_character_visits
    assert alternating.variant_cache_hits == pair.variant_cache_hits + 254
    assert alternating.work_units == pair.work_units + 508
    assert alternating.work_units < messages_module._MAX_COMBINED_WORK
    assert first[0].tool_calls[0].response is None
    assert second[0].tool_calls[0].response is None
    assert source[0].tool_calls[0].id == "call-0"


@pytest.mark.asyncio
async def test_duplicate_variant_projection_obeys_work_deadline_and_cancellation(
    monkeypatch,
) -> None:
    source = _many_tool_call_messages(1_000)
    transform = _transform(TransformOperation("strip"))
    first = await apply_content_transform(source.assistant_messages, transform)
    second = await apply_content_transform(source.assistant_messages, transform)

    monkeypatch.setattr(messages_module, "_MAX_COMBINED_WORK", 10_000)
    bounded = messages_module._CombineMetrics()
    with pytest.raises(ValueError, match="exceeds 10000 relinking work units"):
        Messages.combine((first, second), _metrics=bounded)
    assert bounded.work_units == 10_001
    assert bounded.work_units == _charged_combine_work(bounded)

    monkeypatch.setattr(messages_module, "_MAX_COMBINED_WORK", 1_000_000)
    deadline = messages_module._CombineMetrics()

    def expire_projection() -> None:
        if deadline.variant_projection_visits >= 512:
            raise RuntimeError("variant deadline reached")

    monkeypatch.setattr(messages_module, "_raise_if_combine_expired", expire_projection)
    with pytest.raises(RuntimeError, match="variant deadline reached"):
        Messages.combine((first, second), _metrics=deadline)
    assert 512 <= deadline.variant_projection_visits <= 768

    cancellation = messages_module._CombineMetrics()
    monkeypatch.setattr(messages_module, "_raise_if_combine_expired", lambda: None)

    async def cancel_projection() -> None:
        if cancellation.variant_projection_visits >= 512:
            raise asyncio.CancelledError

    monkeypatch.setattr(messages_module, "_yield_combine_cancellation", cancel_projection)
    with pytest.raises(asyncio.CancelledError):
        await Messages._combine_async((first, second), _metrics=cancellation)
    assert 512 <= cancellation.variant_projection_visits <= 4_608
    assert source[0].tool_calls[0].id == "call-0"


@pytest.mark.asyncio
async def test_duplicate_variant_projection_rejects_incompatible_alternation_and_excess_parts(
    monkeypatch,
) -> None:
    source = _many_tool_call_messages(1_000)
    first = await apply_content_transform(
        source.assistant_messages,
        _transform(TransformOperation("strip")),
    )
    different = await apply_content_transform(
        source.assistant_messages,
        _transform(TransformOperation("casefold")),
    )
    incompatible = messages_module._CombineMetrics()
    with pytest.raises(ValueError, match="different content variants"):
        Messages.combine(
            tuple((first, different)[index % 2] for index in range(256)),
            _metrics=incompatible,
        )
    assert incompatible.selection_visits == 2
    assert incompatible.work_units == _charged_combine_work(incompatible)

    multipart = Messages.from_openai(
        [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": " A "} for _ in range(1_000)],
            }
        ]
    )
    normalized = _transform(TransformOperation("strip"))
    left = await apply_content_transform(multipart, normalized)
    right = await apply_content_transform(multipart, normalized)
    monkeypatch.setattr(messages_module, "_MAX_COMBINED_WORK", 5_000)
    excessive = messages_module._CombineMetrics()
    with pytest.raises(ValueError, match="exceeds 5000 relinking work units"):
        Messages.combine((left, right), _metrics=excessive)
    assert excessive.work_units == 5_001
    assert excessive.work_units == _charged_combine_work(excessive)


@pytest.mark.asyncio
@pytest.mark.parametrize("call_count", [1_000, 16_000])
async def test_tool_relink_index_and_lookups_have_exact_linear_work(call_count) -> None:
    source = _many_tool_call_messages(call_count)
    transformed = await apply_content_transform(
        source.assistant_messages,
        _transform(TransformOperation("strip")),
    )
    metrics = messages_module._CombineMetrics()

    combined = Messages.combine((transformed,), _metrics=metrics)

    assert len(combined[0].tool_calls) == call_count
    assert metrics.selection_visits == 1
    assert metrics.canonical_message_visits == 1
    assert metrics.canonical_call_visits == call_count
    assert metrics.canonical_link_checks == call_count
    assert metrics.selected_message_visits == 1
    assert metrics.selected_call_visits == call_count
    assert metrics.canonical_lookups == call_count
    assert metrics.provenance_visits == 2
    assert metrics.segment_visits == 1
    assert metrics.call_clone_visits == call_count
    assert metrics.assembly_visits == call_count
    assert metrics.message_clone_visits == 1
    assert metrics.work_units == 6 * call_count + 7
    assert metrics.checkpoints == (metrics.work_units + 255) // 256
    assert combined[0].tool_calls[0] is not transformed[0].tool_calls[0]
    assert source[0].tool_calls[0].id == "call-0"


@pytest.mark.asyncio
async def test_tool_relink_deadline_interrupts_index_and_clone_work(
    monkeypatch,
) -> None:
    source = _many_tool_call_messages(16_000)
    indexing = messages_module._CombineMetrics()
    original_expiry_check = messages_module._raise_if_combine_expired

    def expire_index() -> None:
        if indexing.canonical_call_visits >= 512:
            raise RuntimeError("index deadline reached")

    monkeypatch.setattr(messages_module, "_raise_if_combine_expired", expire_index)
    with pytest.raises(RuntimeError, match="index deadline reached"):
        Messages.combine((source.assistant_messages,), _metrics=indexing)
    assert 512 <= indexing.canonical_call_visits <= 768
    monkeypatch.setattr(messages_module, "_raise_if_combine_expired", original_expiry_check)

    transformed = await apply_content_transform(
        source.assistant_messages,
        _transform(TransformOperation("strip")),
    )
    relinking = messages_module._CombineMetrics()

    def expire_relink() -> None:
        if relinking.call_clone_visits >= 512:
            raise RuntimeError("relink deadline reached")

    monkeypatch.setattr(messages_module, "_raise_if_combine_expired", expire_relink)
    with pytest.raises(RuntimeError, match="relink deadline reached"):
        Messages.combine((transformed,), _metrics=relinking)
    assert 512 <= relinking.call_clone_visits <= 768
    assert source[0].tool_calls[0].id == "call-0"


@pytest.mark.asyncio
async def test_async_tool_relink_is_cancellable_during_clone_work(
    monkeypatch,
) -> None:
    source = _many_tool_call_messages(16_000)
    transformed = await apply_content_transform(
        source.assistant_messages,
        _transform(TransformOperation("strip")),
    )
    metrics = messages_module._CombineMetrics()

    async def cancel_relink() -> None:
        if metrics.call_clone_visits >= 512:
            raise asyncio.CancelledError

    monkeypatch.setattr(messages_module, "_yield_combine_cancellation", cancel_relink)
    with pytest.raises(asyncio.CancelledError):
        await Messages._combine_async((transformed,), _metrics=metrics)
    assert 512 <= metrics.call_clone_visits <= 4_608
    assert source[0].tool_calls[0].id == "call-0"


@pytest.mark.asyncio
async def test_tool_relink_preserves_input_faults_and_rejects_incompatible_variants(
    monkeypatch,
) -> None:
    raw = _many_tool_call_messages(2).to_openai_messages()
    raw[0]["tool_calls"][1]["id"] = "call-0"
    duplicate = Messages.from_openai(raw)
    combined = Messages.combine((duplicate.assistant_messages,))
    assert len(combined.tool_calls) == 2
    for call in combined.tool_calls:
        with pytest.raises(InvalidInputError):
            call.guard_read("id")

    source = _many_tool_call_messages(2)
    transformed = await apply_content_transform(
        source.assistant_messages,
        _transform(TransformOperation("strip")),
    )
    malformed = transformed[0].tool_calls[0]
    object.__setattr__(malformed, "id", "")
    with pytest.raises(ValueError, match="does not match its source tool identity"):
        Messages.combine((transformed,))
    assert source[0].tool_calls[0].id == "call-0"

    transformed = await apply_content_transform(
        source.assistant_messages,
        _transform(TransformOperation("strip")),
    )
    incompatible = transformed[0].tool_calls[0]
    object.__setattr__(
        incompatible,
        "function",
        incompatible.function.model_copy(update={"name": "other"}),
    )
    with pytest.raises(ValueError, match="does not match its source tool identity"):
        Messages.combine((transformed,))
    assert source[0].tool_calls[0].name == "lookup"

    transformed = await apply_content_transform(
        source.assistant_messages,
        _transform(TransformOperation("strip")),
    )
    object.__setattr__(
        transformed[0],
        "tool_calls",
        tuple(reversed(transformed[0].tool_calls)),
    )
    with pytest.raises(ValueError, match="does not match its source tool identity"):
        Messages.combine((transformed,))
    assert [call.id for call in source[0].tool_calls] == ["call-0", "call-1"]

    monkeypatch.setattr(messages_module, "_MAX_COMBINED_TOOL_IDENTITIES", 1)
    with pytest.raises(ValueError, match="exceeds 1 tool identities"):
        Messages.combine((source.assistant_messages,))

    monkeypatch.setattr(messages_module, "_MAX_COMBINED_TOOL_IDENTITIES", 65_536)
    monkeypatch.setattr(messages_module, "_MAX_COMBINED_WORK", 5)
    metrics = messages_module._CombineMetrics()
    with pytest.raises(ValueError, match="exceeds 5 relinking work units"):
        Messages.combine((source.assistant_messages,), _metrics=metrics)
    assert metrics.work_units == 6
    assert [call.id for call in source[0].tool_calls] == ["call-0", "call-1"]


@pytest.mark.asyncio
async def test_transformed_views_keep_filters_slices_and_composition() -> None:
    source = Messages.from_openai(
        [
            {"role": "system", "content": " POLICY "},
            {"role": "user", "content": " FIRST "},
            {"role": "assistant", "content": " REPLY "},
            {"role": "user", "content": " LAST "},
        ]
    )
    transformed = await apply_content_transform(
        source, _transform(TransformOperation("strip"), TransformOperation("casefold"))
    )
    selected = Messages.combine((transformed.user_messages[-1:], transformed[:1]))

    assert [(item.role, item.content, item.message_index) for item in selected] == [
        ("user", "last", 3),
        ("system", "policy", 0),
    ]
    assert selected.shares_root_with(source)


@pytest.mark.asyncio
async def test_transformed_text_ranges_keep_source_offsets_for_chunk_selectors() -> None:
    source = Messages.from_openai([{"role": "user", "content": " ABC "}])
    transformed = await apply_content_transform(
        source, _transform(TransformOperation("strip"), TransformOperation("casefold"))
    )
    chunk = transformed._select_ranges((0,), {0: (1, 3)})

    assert chunk[0].content == "bc"
    assert chunk.span.to_dict() == {"message_index": 0, "start": 2, "end": 4}
    assert [item.to_dict() for item in chunk.source_ranges(0, 0, 2)] == [
        {"message_index": 0, "start": 2, "end": 4}
    ]


@pytest.mark.asyncio
async def test_equal_transform_results_have_stable_distinct_cache_identity() -> None:
    source = Messages.from_openai([{"role": "user", "content": " Mixed "}])
    transform = _transform(TransformOperation("strip"), TransformOperation("casefold"))

    first = await apply_content_transform(source, transform)
    second = await apply_content_transform(source, transform)

    assert first.digest == second.digest
    assert first.digest != source.digest


def test_config_parses_a_detached_strict_transform_registry(tmp_path) -> None:
    document = {
        "config_version": 2,
        "transforms": _definitions(
            {"op": "strip"},
            {"op": "replace", "old": "x", "new": ""},
            {"op": "prefix", "text": "["},
            {"op": "suffix", "text": "]"},
            {"op": "casefold"},
        ),
    }
    config = QuarryConfig.from_mapping(document, tmp_path, require_rules=False)
    document["transforms"]["clean"]["operations"][0]["op"] = "prefix"

    assert config.transforms["clean"].operations[0] == TransformOperation("strip")
    with pytest.raises(TypeError):
        config.transforms["other"] = config.transforms["clean"]  # ty: ignore[index]


@pytest.mark.parametrize(
    ("transforms", "message"),
    [
        pytest.param([], "must be a mapping", id="registry-list"),
        pytest.param({"bad name": {"operations": [{"op": "strip"}]}}, "Invalid transform name"),
        pytest.param({"x": []}, "must be a mapping", id="definition-list"),
        pytest.param({"x": {"operations": []}}, "non-empty list", id="empty"),
        pytest.param(
            {"x": {"operations": [{"op": "custom", "value": object()}]}},
            "op must be",
            id="custom-value",
        ),
        pytest.param(
            {"x": {"operations": [{"op": "transform", "name": "x"}]}},
            "op must be",
            id="reference-cycle-is-not-a-supported-value",
        ),
        pytest.param(
            {"x": {"operations": [{"op": "replace", "old": "", "new": "x"}]}},
            "non-empty literal match",
            id="empty-replace-source",
        ),
        pytest.param(
            {"x": {"operations": [{"op": "strip", "text": "x"}]}},
            "unknown",
            id="operation-type-confusion",
        ),
    ],
)
def test_config_rejects_malformed_or_open_ended_transform_values(
    tmp_path, transforms, message
) -> None:
    with pytest.raises((ConfigurationError, ValueError), match=message):
        QuarryConfig.from_mapping(
            {"config_version": 2, "transforms": transforms},
            tmp_path,
            require_rules=False,
        )


def test_transform_contract_rejects_custom_value_subclasses(tmp_path) -> None:
    class CustomOperation(TransformOperation):
        pass

    class CustomTransform(ContentTransform):
        pass

    with pytest.raises(TypeError, match="must be TransformOperation values"):
        ContentTransform((CustomOperation("strip"),))

    custom = CustomTransform((TransformOperation("strip"),))
    with pytest.raises(ConfigurationError, match="must be a ContentTransform immutable value"):
        QuarryConfig(config_version=2, transforms={"custom": custom})

    rule = DSLParser().parse(
        'rule "R"\ncondition: model("judge").classify(messages.transform("custom")) > 0.5\n'
    )
    result = compile_rule(rule, transforms={"custom": custom})
    assert not result.ok
    assert "must use the immutable transform contract" in result.diagnostics.errors[0].message


def test_transform_registry_and_operation_counts_are_bounded(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="at most 64"):
        QuarryConfig.from_mapping(
            {
                "config_version": 2,
                "transforms": {
                    f"transform_{index}": {"operations": [{"op": "strip"}]} for index in range(65)
                },
            },
            tmp_path,
            require_rules=False,
        )
    with pytest.raises((ConfigurationError, ValueError), match="at most 16"):
        QuarryConfig.from_mapping(
            {
                "config_version": 2,
                "transforms": {"many": {"operations": [{"op": "strip"} for _ in range(17)]}},
            },
            tmp_path,
            require_rules=False,
        )


def _maximal_replace_plan(quoted_characters: int) -> tuple[TransformOperation, ...]:
    remaining = quoted_characters
    operations = []
    for _ in range(16):
        literals = []
        for _ in range(2):
            escaped = min(remaining, 4_096)
            remaining -= escaped
            literals.append('"' * escaped + "a" * (4_096 - escaped))
        operations.append(TransformOperation("replace", literals[0], literals[1]))
    assert remaining == 0
    return tuple(operations)


def test_canonical_plan_byte_bound_is_exact_and_every_accepted_plan_round_trips(
    tmp_path,
) -> None:
    accepted = ContentTransform(_maximal_replace_plan(8_367))
    encoded = accepted.canonical_json()

    assert len(encoded.encode("utf-8")) == 140_000
    assert ContentTransform.from_canonical_json(encoded) == accepted
    config = QuarryConfig(
        config_version=2,
        transforms={"largest": accepted},
    )
    assert config.transforms["largest"] == accepted

    with pytest.raises(ValueError, match="140000 canonical JSON bytes"):
        ContentTransform(_maximal_replace_plan(8_368))
    with pytest.raises(ConfigurationError, match="140000 canonical JSON bytes"):
        QuarryConfig.from_mapping(
            {
                "config_version": 2,
                "transforms": {
                    "too_large": {
                        "operations": [
                            {"op": op.kind, "old": op.value, "new": op.replacement}
                            for op in _maximal_replace_plan(8_368)
                        ]
                    }
                },
            },
            tmp_path,
            require_rules=False,
        )


@pytest.mark.parametrize("character", ['"', "\\", "\x00", "é", "😀"])
def test_canonical_plan_bound_counts_escaping_and_utf8_bytes(character) -> None:
    operations = tuple(
        TransformOperation("replace", character * 4_096, character * 4_096) for _ in range(16)
    )

    with pytest.raises(ValueError, match="canonical JSON bytes"):
        ContentTransform(operations)


@pytest.mark.parametrize(
    "value",
    [
        "not-json",
        "{}",
        '[{"op":"custom"}]',
        '[{"op":"replace","old":"","new":"x"}]',
        " " * 140_001,
    ],
)
def test_compiled_transform_identity_rejects_malformed_values(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        ContentTransform.from_canonical_json(value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("attribute", "limit", "transform", "content", "message"),
    [
        pytest.param(
            "MAX_TRANSFORM_INPUT_CHARACTERS",
            2,
            _transform(TransformOperation("strip")),
            "abc",
            "input exceeds",
            id="input",
        ),
        pytest.param(
            "MAX_TRANSFORM_WORK",
            1,
            _transform(TransformOperation("casefold")),
            "abc",
            "character-work",
            id="work",
        ),
        pytest.param(
            "MAX_TRANSFORM_MAPPING_SEGMENTS",
            1,
            _transform(TransformOperation("prefix", "<"), TransformOperation("suffix", ">")),
            "x",
            "mapping segments",
            id="mapping",
        ),
        pytest.param(
            "MAX_TRANSFORM_OUTPUT_CHARACTERS",
            2,
            _transform(TransformOperation("prefix", "abc")),
            "x",
            "output exceeds",
            id="output",
        ),
    ],
)
async def test_runtime_transform_limits_fail_before_results_escape(
    monkeypatch, attribute, limit, transform, content, message
) -> None:
    monkeypatch.setattr(transform_runtime, attribute, limit)
    source = Messages.from_openai([{"role": "user", "content": content}])

    with pytest.raises(ValueError, match=message):
        await apply_content_transform(source, transform)


@pytest.mark.asyncio
async def test_expansion_ratio_and_deadline_checks_are_enforced(monkeypatch) -> None:
    source = Messages.from_openai([{"role": "user", "content": "x"}])
    monkeypatch.setattr(transform_runtime, "MAX_TRANSFORM_EXPANSION_RATIO", 1)
    monkeypatch.setattr(transform_runtime, "MAX_TRANSFORM_INSERTED_CHARACTERS", 0)
    with pytest.raises(ValueError, match="expansion bound"):
        await apply_content_transform(source, _transform(TransformOperation("prefix", "y")))

    def expired() -> None:
        raise RuntimeError("deadline reached")

    monkeypatch.setattr(transform_runtime, "raise_if_evaluation_expired", expired)
    with pytest.raises(RuntimeError, match="deadline reached"):
        await apply_content_transform(source, _transform(TransformOperation("casefold")))


@pytest.mark.asyncio
async def test_long_operation_checks_deadline_inside_its_character_loop(monkeypatch) -> None:
    checks = 0

    def expires_during_loop() -> None:
        nonlocal checks
        checks += 1
        if checks == 4:
            raise RuntimeError("loop deadline reached")

    monkeypatch.setattr(transform_runtime, "raise_if_evaluation_expired", expires_during_loop)
    source = Messages.from_openai([{"role": "user", "content": "A" * 10_000}])

    with pytest.raises(RuntimeError, match="loop deadline reached"):
        await apply_content_transform(source, _transform(TransformOperation("casefold")))
    assert checks == 4


@pytest.mark.asyncio
async def test_max_segment_replace_and_casefold_have_linear_deterministic_visits() -> None:
    length = 8_192
    source = Messages.from_openai([{"role": "user", "content": "a" * length}])
    metrics = transform_runtime._TransformMetrics()
    transformed = await apply_content_transform(
        source,
        _transform(
            TransformOperation("replace", "a", "A"),
            TransformOperation("casefold"),
        ),
        _metrics=metrics,
    )

    assert transformed[0].content == "a" * length
    assert metrics.character_visits == 2 * length
    assert metrics.output_characters == 2 * length
    assert metrics.segment_visits == 2 * length + 1
    assert metrics.source_visits == 0
    assert (metrics.part_visits, metrics.range_visits) == (1, 1)
    assert metrics.matches == length
    assert (
        metrics.checkpoints
        <= (
            metrics.character_visits
            + metrics.output_characters
            + metrics.segment_visits
            + metrics.source_visits
            + metrics.part_visits
            + metrics.range_visits
            + metrics.matches
        )
        // 256
        + 2
    )


@pytest.mark.asyncio
async def test_max_unicode_expansion_visits_each_character_and_segment_once() -> None:
    length = 4_096
    source = Messages.from_openai([{"role": "user", "content": "ß" * length}])
    metrics = transform_runtime._TransformMetrics()
    transformed = await apply_content_transform(
        source,
        _transform(TransformOperation("casefold")),
        _metrics=metrics,
    )

    assert transformed[0].content == "ss" * length
    assert metrics.character_visits == length
    assert metrics.output_characters == 2 * length
    assert metrics.segment_visits == 2
    assert metrics.source_visits == 0
    assert (metrics.part_visits, metrics.range_visits) == (1, 1)
    assert metrics.matches == 0


@pytest.mark.asyncio
async def test_chained_whole_mapping_reuses_sources_with_linear_mapping_work() -> None:
    length = 4_096
    source = Messages.from_openai([{"role": "user", "content": "a" * length}])
    metrics = transform_runtime._TransformMetrics()
    transformed = await apply_content_transform(
        source,
        _transform(
            TransformOperation("replace", "a", "A"),
            TransformOperation("replace", "A" * length, "b" * length),
            TransformOperation("replace", "b", "c"),
        ),
        _metrics=metrics,
    )

    assert transformed[0].content == "c" * length
    assert metrics.character_visits == 3 * length
    assert metrics.output_characters == 3 * length
    assert metrics.segment_visits == 3 * length + 1
    assert metrics.source_visits == length
    assert (metrics.part_visits, metrics.range_visits) == (1, 1)
    assert metrics.matches == 2 * length + 1


def test_compiler_requires_a_literal_configured_messages_transform() -> None:
    config = {"clean": _transform(TransformOperation("strip"))}

    def errors(condition: str) -> str:
        rule = DSLParser().parse(f'rule "R"\ncondition: {condition}\n')
        result = compile_rule(rule, transforms=config)
        return "; ".join(error.message for error in result.diagnostics.errors)

    assert "not configured" in errors(
        'model("judge").classify(messages.transform("missing")) > 0.5'
    )
    assert "receiver must be Messages" in errors(
        'model("judge").classify(tools.transform("clean")) > 0.5'
    )
    assert "literal" in errors('model("judge").classify(messages.transform(1)) > 0.5')


def test_transform_nesting_is_bounded_at_compile_time() -> None:
    receiver = "messages" + '.transform("clean")' * 17
    rule = DSLParser().parse(f'rule "R"\ncondition: model("judge").classify({receiver}) > 0.5\n')
    result = compile_rule(
        rule,
        transforms={"clean": _transform(TransformOperation("strip"))},
    )

    assert not result.ok
    assert "exceeds 16" in result.diagnostics.errors[0].message


@pytest.mark.parametrize(
    "part",
    [
        pytest.param('messages.transform("clean")', id="siblings"),
        pytest.param(
            'messages.transform("clean") if true else messages',
            id="conditional-branches",
        ),
        pytest.param(
            '[m for m in messages.transform("clean") if m.role == "user"]',
            id="filter-composition",
        ),
    ],
)
def test_aggregate_transform_limit_covers_the_final_expression_graph(part) -> None:
    config = {"clean": _transform(TransformOperation("strip"))}

    accepted = " + ".join([f"({part})"] * 16)
    accepted_rule = DSLParser().parse(
        f'rule "R"\ncondition: model("judge").classify({accepted}) > 0.5\n'
    )
    assert compile_rule(accepted_rule, transforms=config).ok

    rejected = " + ".join([f"({part})"] * 17)
    rejected_rule = DSLParser().parse(
        f'rule "R"\ncondition: model("judge").classify({rejected}) > 0.5\n'
    )
    result = compile_rule(rejected_rule, transforms=config)
    assert not result.ok
    assert "exceeds 16 content transform applications" in result.diagnostics.errors[0].message


def test_reused_transform_aliases_are_counted_per_finalized_occurrence() -> None:
    parts = " + ".join(["cleaned"] * 17)
    rule = DSLParser().parse(
        'rule "R"\nvariables:\n'
        '    cleaned = messages.transform("clean")\n'
        f'condition: model("judge").classify({parts}) > 0.5\n'
    )
    result = compile_rule(
        rule,
        transforms={"clean": _transform(TransformOperation("strip"))},
    )

    assert not result.ok
    assert "exceeds 16 content transform applications" in result.diagnostics.errors[0].message


def test_shared_transform_dag_nodes_count_by_occurrence() -> None:
    transform = _transform(TransformOperation("strip"))
    shared = ModelExpression(
        "transform",
        transform.canonical_json(),
        (ModelExpression("messages"),),
    )
    accepted = ModelExpression("concat", None, (shared,) * 16)
    validate_transform_expression_graph(
        accepted,
        maximum=16,
        validate_plan=ContentTransform.from_canonical_json,
    )

    rejected = ModelExpression("concat", None, (shared,) * 17)
    with pytest.raises(ValueError, match="exceeds 16 content transform applications"):
        validate_transform_expression_graph(
            rejected,
            maximum=16,
            validate_plan=ContentTransform.from_canonical_json,
        )


def test_collect_transform_limit_is_aggregate_across_every_finalized_root() -> None:
    config = {"clean": _transform(TransformOperation("strip"))}

    def chain(count: int) -> str:
        return "messages" + '.transform("clean")' * count

    def rule(model_count: int) -> str:
        return (
            'rule "R"\ncondition: true\ncollect:\n'
            f'    [span for span in model("finder").spans({chain(6)}, 0.5) '
            f"if ({chain(3)} == {chain(3)}) and "
            f'model("judge").classify({chain(model_count)}) > 0.5]\n'
        )

    accepted = compile_rule(DSLParser().parse(rule(4)), transforms=config)
    assert accepted.ok
    assert accepted.artifact.collect_plan is not None

    rejected = compile_rule(DSLParser().parse(rule(5)), transforms=config)
    assert not rejected.ok
    assert "exceeds 16 content transform applications" in rejected.diagnostics.errors[0].message

    plan = config["clean"].canonical_json()
    shared = ModelExpression("transform", plan, (ModelExpression("messages"),))
    validate_transform_expression_graph(
        ModelExpression("concat", None, (shared,) * 8),
        maximum=16,
        validate_plan=ContentTransform.from_canonical_json,
        additional_roots=(ModelExpression("concat", None, (shared,) * 8),),
    )
    with pytest.raises(ValueError, match="exceeds 16 content transform applications"):
        validate_transform_expression_graph(
            ModelExpression("concat", None, (shared,) * 8),
            maximum=16,
            validate_plan=ContentTransform.from_canonical_json,
            additional_roots=(
                ModelExpression("concat", None, (shared,) * 8),
                shared,
            ),
        )


def test_authored_transform_rendering_round_trips_without_internal_helper_names() -> None:
    parsed = DSLParser().parse(
        'rule "R"\ncondition: model("judge").classify(messages.transform("clean")) > 0.5\n'
    )
    rendered = render_condition(parsed.condition)
    reparsed = DSLParser().parse(f'rule "R"\ncondition: {rendered}\n')

    assert 'messages.transform("clean")' in rendered
    assert "_transform" not in rendered
    assert render_condition(reparsed.condition) == rendered
    result = compile_rule(
        parsed,
        transforms={"clean": _transform(TransformOperation("strip"))},
    )
    assert result.ok
    assert all("_transform" not in node.label for node in result.artifact.decision_tree.nodes)
    assert any(
        'messages.transform("clean")' in node.label for node in result.artifact.decision_tree.nodes
    )


@pytest.mark.parametrize(
    ("expression", "rendered_call"),
    [
        pytest.param("messages.transform()", "messages.transform()", id="zero"),
        pytest.param(
            'messages.transform("clean", "extra")',
            'messages.transform("clean", "extra")',
            id="multiple",
        ),
        pytest.param(
            'messages.transform().transform("clean", "extra")',
            'messages.transform().transform("clean", "extra")',
            id="nested-malformed",
        ),
    ],
)
def test_invalid_transform_arities_render_only_authored_method_syntax(
    expression, rendered_call
) -> None:
    parsed = DSLParser().parse(
        f'rule "R"\ncondition: model("judge").classify({expression}) > 0.5\n'
    )
    rendered = render_condition(parsed.condition)
    result = compile_rule(
        parsed,
        transforms={"clean": _transform(TransformOperation("strip"))},
    )

    assert rendered_call in rendered
    assert "_transform" not in rendered
    assert not result.ok
    errors = "; ".join(error.message for error in result.diagnostics.errors)
    assert "messages.transform() takes one literal name" in errors
    assert "_transform" not in errors


def test_zero_argument_internal_transform_renderer_never_leaks_helper_name() -> None:
    rendered = render_condition(FunctionCall("_transform", []))

    assert rendered == "messages.transform()"
    assert "_transform" not in rendered


@pytest.mark.parametrize(
    "expression",
    ["messages.transform()", 'messages.transform("clean", "extra")'],
)
def test_invalid_transform_arity_at_span_boundary_keeps_bounded_public_error(
    expression,
) -> None:
    parsed = DSLParser().parse(
        'rule "R"\ncondition: true\ncollect:\n'
        f'    [span for span in model("finder").spans({expression}, 0.5) if true]\n'
    )
    result = compile_rule(
        parsed,
        transforms={"clean": _transform(TransformOperation("strip"))},
    )

    assert not result.ok
    errors = "; ".join(error.message for error in result.diagnostics.errors)
    assert "messages.transform() takes one literal name" in errors
    assert "_transform" not in errors


def test_model_input_transform_chains_after_composition_and_before_filtering(tmp_path) -> None:
    adapter = ModelAdapter()
    scanner = v2_scanner(
        tmp_path,
        {
            "R": 'model("judge").score('
            '[item for item in (messages[:2] + messages[-1:]).transform("clean") '
            'if item.role == "user"], "policy") > 0.5'
        },
        {"judge": model_entry(score=True)},
        adapter,
        transforms=_definitions({"op": "strip"}, {"op": "casefold"}),
    )

    result = scanner.scan_data(
        {
            "messages": [
                {"role": "system", "content": " POLICY "},
                {"role": "user", "content": " HELLO "},
                {"role": "assistant", "content": " REPLY "},
                {"role": "user", "content": " WORLD "},
            ]
        },
        enable_explain=True,
    )

    assert (result.rule_matches, result.evaluation_errors) == (["R"], None)
    assert adapter.score_messages == [
        [{"role": "user", "content": "hello"}, {"role": "user", "content": "world"}]
    ]
    assert result.detailed_matches[0].model_results[0].input_span == {
        "message_index": 1,
        "start": 1,
        "end_message_index": 3,
        "end": 6,
    }


def test_transform_in_an_unselected_conditional_branch_is_lazy(tmp_path) -> None:
    adapter = ModelAdapter()
    scanner = v2_scanner(
        tmp_path,
        {"R": 'model("judge").classify(messages if true else messages.transform("clean")) > 0.5'},
        {"judge": model_entry(classify=0)},
        adapter,
        transforms=_definitions({"op": "prefix", "text": "not selected: "}),
    )

    result = scanner.scan_data({"messages": [{"role": "user", "content": "plain"}]})

    assert (result.rule_matches, result.evaluation_errors) == (["R"], None)
    assert adapter.classification_messages == [[{"role": "user", "content": "plain"}]]


def test_transformed_span_input_maps_exact_ranges_and_survives_prepared_round_trip(
    tmp_path,
) -> None:
    class SpanAdapter(ModelAdapter):
        async def decode_spans(self, config, raw_result, messages, threshold):
            self.decoded_spans.append((config, raw_result, messages, threshold))
            content = messages[0].content
            start = content.index("alice@example.com")
            return [Span(0, start, start + len("alice@example.com"), Score(0.9))]

    adapter = SpanAdapter()
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "finder": model_entry(category="ENTITY", classify=0, spans=0),
            },
            "transforms": _definitions({"op": "strip"}, {"op": "casefold"}),
        },
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(config, adapters={"fake": adapter})
    rule = (
        'rule "R"\ncondition: true\ncollect:\n'
        '    [span for span in model("finder").spans('
        'messages.transform("clean"), 0.5) if true]\n'
    )
    policy = compile_policy(
        rule,
        runtime_config=config,
        model_inference_backend=backend,
    )
    loaded_runtime = replace(config, transforms={})
    loaded = load_compiled_policy(
        policy.to_bytes(),
        runtime_config=loaded_runtime,
        model_inference_backend=ModelInferenceBackend(
            loaded_runtime,
            adapters={"fake": adapter},
        ),
    )

    result = loaded.scan({"messages": [{"role": "user", "content": "  ALICE@EXAMPLE.COM  "}]})

    assert (result.rule_matches, result.evaluation_errors) == (["R"], None)
    assert adapter.classification_messages[-1] == [{"role": "user", "content": "alice@example.com"}]
    span = result.outputs["R"].items[0].value
    assert span.to_dict() == {
        "message_index": 0,
        "start": 2,
        "end": 19,
        "score": {"value": 0.9},
        "source_ranges": [{"message_index": 0, "start": 2, "end": 19}],
    }
    document = orjson.loads(policy.to_bytes())
    collect_source = document["rule"]["artifact"]["collect_plan"]["source"]
    assert _first_transform_wire(collect_source) is not None


def test_transformed_span_input_fails_closed_for_unmapped_replacement(tmp_path) -> None:
    class InsertedSpanAdapter(ModelAdapter):
        async def decode_spans(self, _config, _raw_result, messages, _threshold):
            assert messages[0].content == "item"
            return [Span(0, 0, 4, Score(0.9))]

    adapter = InsertedSpanAdapter()
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "finder": model_entry(category="ENTITY", classify=0, spans=0),
            },
            "transforms": _definitions(
                {"op": "prefix", "text": "X"},
                {"op": "replace", "old": "XALICE", "new": "item"},
            ),
        },
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(config, adapters={"fake": adapter})
    policy = compile_policy(
        'rule "R"\ncondition: true\ncollect:\n'
        '    [span for span in model("finder").spans('
        'messages.transform("clean"), 0.5) if true]\n',
        runtime_config=config,
        model_inference_backend=backend,
    )

    result = policy.scan({"messages": [{"role": "user", "content": "ALICE"}]})

    assert result.rule_matches == []
    assert result.completion_state == "incomplete"
    assert "inserted text with no source position" in result.evaluation_errors["R"]


def test_prepared_policy_embeds_transform_identity_and_loads_without_registry(tmp_path) -> None:
    adapter = ModelAdapter()
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"judge": model_entry(classify=0)},
            "transforms": _definitions({"op": "replace", "old": "UP", "new": "down"}),
        },
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(config, adapters={"fake": adapter})
    policy = compile_policy(
        'rule "R"\ncondition: model("judge").classify(messages.transform("clean")) > 0.5\n',
        runtime_config=config,
        model_inference_backend=backend,
    )
    encoded = policy.to_bytes()
    runtime = replace(config, transforms={})
    loaded = load_compiled_policy(
        encoded,
        runtime_config=runtime,
        model_inference_backend=ModelInferenceBackend(runtime, adapters={"fake": adapter}),
    )

    result = loaded.scan({"messages": [{"role": "user", "content": "go UP"}]})

    assert (result.rule_matches, result.evaluation_errors) == (["R"], None)
    assert adapter.classification_messages[-1] == [{"role": "user", "content": "go down"}]
    assert loaded.policy_digest == policy.policy_digest
    document = orjson.loads(encoded)
    labels = [node[1] for node in document["rule"]["artifact"]["decision_tree"]["nodes"]]
    assert any('messages.transform("clean")' in label for label in labels)
    assert all("_transform" not in label for label in labels)


def _first_transform_wire(value):
    if isinstance(value, dict):
        if value.get("kind") == "transform":
            return value
        for child in value.values():
            found = _first_transform_wire(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _first_transform_wire(child)
            if found is not None:
                return found
    return None


def test_prepared_policy_load_rejects_aggregate_transform_limit_bypass(tmp_path) -> None:
    adapter = ModelAdapter()
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"judge": model_entry(classify=0)},
            "transforms": _definitions({"op": "strip"}),
        },
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(config, adapters={"fake": adapter})
    policy = compile_policy(
        'rule "R"\ncondition: model("judge").classify(messages.transform("clean")) > 0.5\n',
        runtime_config=config,
        model_inference_backend=backend,
    )
    document = orjson.loads(policy.to_bytes())
    target = _first_transform_wire(document["rule"]["artifact"])
    assert target is not None
    value = target["value"]
    nested = copy.deepcopy(target)
    for _ in range(16):
        nested = {"kind": "transform", "value": value, "children": [nested]}
    target.clear()
    target.update(nested)

    with pytest.raises(ValueError, match="exceeds 16 content transform applications"):
        load_compiled_policy(
            orjson.dumps(document),
            runtime_config=config,
            model_inference_backend=backend,
        )


def test_collect_transform_limit_round_trips_and_rejects_cross_root_wire_bypass(
    tmp_path,
) -> None:
    adapter = ModelAdapter()
    definition = _definitions({"op": "strip"})
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "finder": model_entry(category="ENTITY", classify=0, spans=0),
                "judge": model_entry(classify=0),
            },
            "transforms": definition,
        },
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(config, adapters={"fake": adapter})

    def chain(count: int) -> str:
        return "messages" + '.transform("clean")' * count

    rule = (
        'rule "R"\ncondition: true\ncollect:\n'
        f'    [span for span in model("finder").spans({chain(6)}, 0.5) '
        f"if ({chain(3)} == {chain(3)}) and "
        f'model("judge").classify({chain(4)}) > 0.5]\n'
    )
    policy = compile_policy(
        rule,
        runtime_config=config,
        model_inference_backend=backend,
    )
    encoded = policy.to_bytes()
    load_compiled_policy(
        encoded,
        runtime_config=config,
        model_inference_backend=backend,
    )

    document = orjson.loads(encoded)
    collect = document["rule"]["artifact"]["collect_plan"]
    target = _first_transform_wire(collect["source"])
    assert target is not None
    target["children"][0] = {
        "kind": "transform",
        "value": target["value"],
        "children": [target["children"][0]],
    }

    with pytest.raises(ValueError, match="exceeds 16 content transform applications"):
        load_compiled_policy(
            orjson.dumps(document),
            runtime_config=config,
            model_inference_backend=backend,
        )


def test_untransformed_model_span_keeps_existing_public_wire_shape() -> None:
    source = Messages.from_openai([{"role": "user", "content": "sample"}])
    bound = source._bind_span(Span(0, 1, 4, Score(0.9)))
    assert bound.to_dict() == {"message_index": 0, "start": 1, "end": 4, "score": {"value": 0.9}}
    assert source.extract(bound) == "amp"
