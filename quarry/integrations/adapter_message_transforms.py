# Copyright 2026 Cisco Systems, Inc. and its affiliates
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

"""Apply registered part transforms while preserving message graphs and source context."""

from __future__ import annotations

import asyncio
from bisect import bisect_right
from collections.abc import Callable, Mapping
from itertools import islice
from typing import TYPE_CHECKING

from quarry.engine.pacing import Pacer
from quarry.integrations.adapter_transforms import (
    keep_transform,
    read_transform,
    transform_error,
    transform_unit_snapshot,
)
from quarry.integrations.content_transforms import (
    _append_segment,
    _check_intermediate,
    _MappedPiece,
    _OperationMeter,
    _partition_text_units,
    _replacement_parts,
    _replacement_sources,
    _shift_segments,
    _TransformMetrics,
)
from quarry.integrations.shared_transforms import TRANSFORM_BATCH_SIZE, transform_input_size
from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.media import require_readable_media
from quarry.types.messages import Messages, UnreadableMessage, _project_json, _synthetic_key
from quarry.types.transforms import (
    MAX_TRANSFORM_INPUT_CHARACTERS,
    MAX_TRANSFORM_MAPPING_SEGMENTS,
    MAX_TRANSFORM_OUTPUT_CHARACTERS,
    ContentSourcePart,
    ProvenanceSegment,
    slice_provenance,
)

if TYPE_CHECKING:
    from quarry.integrations.model_inference import ModelInferenceBackend
    from quarry.types.adapter_transforms import TransformResult


async def _compose_result(
    result: TransformResult,
    source: tuple[ProvenanceSegment, ...],
    context: tuple[ContentSourcePart, ...],
    meter: _OperationMeter,
) -> tuple[ProvenanceSegment, ...]:
    """Compose bounded, possibly reordered adapter runs through the input's existing map."""

    ends = tuple(segment.end for segment in source)
    mapped: list[ProvenanceSegment] = []
    for run in result.runs:
        meter.record(ranges=1)
        if run.source_start is None:
            _append_segment(mapped, ProvenanceSegment(run.start, run.end, (), "whole", context))
        else:
            if run.source_end is None:
                raise ValueError("a mapped transform run requires its source end")
            # Adapter runs may overlap or reorder input. Locate their first segment with a
            # binary search, then meter every visited mapping and source reference.
            index = bisect_right(ends, run.source_start)
            selected = []
            while index < len(source) and source[index].start < run.source_end:
                segment = source[index]
                selected.append(segment)
                meter.record(segments=1, sources=len(segment.sources) + len(segment.parts))
                if meter._pending >= 256:
                    await meter.checkpoint()
                index += 1
            pieces = slice_provenance(tuple(selected), run.source_start, run.source_end)
            if run.kind == "linear":
                for piece in _shift_segments(pieces, run.start):
                    _append_segment(mapped, piece)
            else:
                ranges = tuple(
                    _MappedPiece(piece.start, piece.end, piece.sources, piece.kind, piece.parts)
                    for piece in pieces
                )
                _append_segment(
                    mapped,
                    ProvenanceSegment(
                        run.start,
                        run.end,
                        _replacement_sources(ranges, meter),
                        "whole",
                        _replacement_parts(ranges, meter),
                    ),
                )
        _check_intermediate(len(result.text), len(mapped))
        if meter._pending >= 256:
            await meter.checkpoint()
    await meter.checkpoint(finish=True)
    return tuple(mapped)


def _part_batches(parts, size):
    iterator = iter(parts)
    while batch := tuple(islice(iterator, size)):
        yield batch


def _selected_batches(selected):
    batch = []
    size = 0
    for item in selected:
        count = transform_input_size(item[1])
        if batch and size + count > MAX_TRANSFORM_INPUT_CHARACTERS:
            yield batch
            batch, size = [], 0
        batch.append(item)
        size += count
    if batch:
        yield batch


async def apply_adapter_transform(
    messages: Messages,
    backend: ModelInferenceBackend,
    name: str,
    *,
    _metrics: _TransformMetrics | None = None,
    _on_part: Callable[[], None] | None = None,
    _message_value: bool = False,
) -> Messages:
    """Replace selected parts with validated text, retaining roles, order and tool links."""

    if not isinstance(messages, Messages):
        raise TypeError("adapter transform receiver must be Messages")
    selected = frozenset(backend.transform_parts(name))
    if not selected:
        await require_readable_media(messages, rewritable_only=True)
        return await backend.transform_messages(name, messages)
    # Unreadable holders stay opaque in the result. A later field read decides
    # whether they matter; a transform must not read them merely to reach another item.
    readable = messages._view_positions(
        position
        for position, message in enumerate(messages)
        if not isinstance(message, UnreadableMessage)
    )
    await require_readable_media(readable, selected)
    meter = _OperationMeter(_metrics if _metrics is not None else _TransformMetrics())
    unique = {id(message): message for message in readable}
    input_characters = sum(len(message.content) for message in unique.values())
    if input_characters > MAX_TRANSFORM_INPUT_CHARACTERS:
        raise transform_error(
            name, "invalid_input", "selected message text exceeds its character limit"
        )
    output_characters = mapping_segments = 0
    overrides = {}
    batch_transform = getattr(backend, "transform_units", None)
    cache_identity = getattr(backend, "transform_cache_identity", None)
    batch_enabled = (
        callable(batch_transform) and callable(cache_identity) and cache_identity(name) is not None
    )
    pacer = Pacer()
    for identity, message in unique.items():
        raise_if_evaluation_expired()
        if pacer.due():
            await asyncio.sleep(0)
        units = await _partition_text_units(message, message._content_provenance(), meter)
        text_units = {part_index: (value, mapping) for part_index, value, mapping in units}
        raw = message.source_content
        if isinstance(raw, str):
            wire_parts = ((None, {"type": "text", "text": raw}),)
        elif isinstance(raw, tuple):
            wire_parts = enumerate(raw)
        else:
            wire_parts = ()
        replacements = {}
        mapped = []
        offset = 0
        for batch in _part_batches(wire_parts, TRANSFORM_BATCH_SIZE if batch_enabled else 1):
            ready = {}
            if batch_enabled:
                selected_units = []
                for index, part in batch:
                    if isinstance(part, Mapping) and part.get("type") in selected:
                        try:
                            snapshot = transform_unit_snapshot(
                                _project_json(part), _check_unicode=False
                            )
                        except (TypeError, ValueError) as exc:
                            raise transform_error(name, "invalid_input", str(exc)) from exc
                        selected_units.append((index, snapshot))
                        if pacer.due():
                            await asyncio.sleep(0)
                for selected_batch in _selected_batches(selected_units):
                    assert callable(batch_transform)
                    results = await batch_transform(
                        name, [unit for _, unit in selected_batch], _message_value=_message_value
                    )
                    if type(results) is not list or len(results) != len(selected_batch):
                        raise transform_error(
                            name, "invalid_output", "invalid transform batch result"
                        )
                    for (index, unit), result in zip(selected_batch, results, strict=True):
                        ready[index] = read_transform(keep_transform(result, unit, name), unit)
                    if (
                        sum(len(result.text) for result in ready.values())
                        > MAX_TRANSFORM_OUTPUT_CHARACTERS
                        or sum(len(result.runs) for result in ready.values())
                        > MAX_TRANSFORM_MAPPING_SEGMENTS
                    ):
                        raise transform_error(
                            name, "invalid_output", "transform batch exceeds its output limit"
                        )
            for part_index, part in batch:
                meter.record(parts=1)
                if meter._pending >= 256:
                    await meter.checkpoint()
                if not isinstance(part, Mapping):
                    continue
                part_type = part.get("type")
                source_text, provenance = text_units.get(part_index, ("", ()))
                if part_type in selected:
                    unit = _project_json(part)
                    transform_unit = (
                        getattr(backend, "transform_message_unit", backend.transform_unit)
                        if _message_value
                        else backend.transform_unit
                    )
                    if batch_enabled:
                        result = ready[part_index]
                    else:
                        # Preserve the same output contract for injected single-unit backends.
                        result = await transform_unit(name, unit)
                        result = read_transform(keep_transform(result, unit, name), unit)
                    if _on_part is not None:
                        _on_part()
                    context = (
                        (
                            ContentSourcePart(
                                message.message_index,
                                None if message._rewrite_content_changed else part_index,
                            ),
                        )
                        if _synthetic_key(message) is None
                        else ()
                    )
                    transformed_map = await _compose_result(result, provenance, context, meter)
                    transformed_text = result.text
                    if (
                        part_type != "text"
                        or transformed_text != source_text
                        or transformed_map != provenance
                    ):
                        replacements[part_index] = transformed_text
                else:
                    transformed_text, transformed_map = source_text, provenance
                for segment in _shift_segments(transformed_map, offset):
                    _append_segment(mapped, segment)
                offset += len(transformed_text)
                meter.record(output=len(transformed_text))
                _check_intermediate(offset, len(mapped))
        output_characters += offset
        mapping_segments += len(mapped)
        _check_intermediate(output_characters, mapping_segments)
        if replacements:
            overrides[identity] = message._with_transformed_content(
                replacements, tuple(mapped), replace_media=True
            )
    await meter.checkpoint(finish=True)
    return (
        messages
        if not overrides
        else await messages._with_position_overrides_async(
            {
                position: overrides[id(message)]
                for position, message in enumerate(messages)
                if id(message) in overrides
            }
        )
    )
