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

"""Apply reviewed local text operations while retaining character provenance."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache

from quarry.engine.pacing import Pacer
from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.messages import Messages, MessageType, UnreadableMessage
from quarry.types.transforms import (
    MAX_TRANSFORM_EXPANSION_RATIO,
    MAX_TRANSFORM_INPUT_CHARACTERS,
    MAX_TRANSFORM_INSERTED_CHARACTERS,
    MAX_TRANSFORM_MAPPING_SEGMENTS,
    MAX_TRANSFORM_OUTPUT_CHARACTERS,
    MAX_TRANSFORM_WORK,
    ContentSourcePart,
    ContentSourceRange,
    ContentTransform,
    MappingKind,
    ProvenanceSegment,
    TransformOperation,
)

_LOOP_CHECK_INTERVAL = 256
_CANCELLATION_CHECK_INTERVAL = 16


@dataclass(slots=True)
class _TransformMetrics:
    """Deterministic operation counters used by adversarial linear-work regressions."""

    character_visits: int = 0
    output_characters: int = 0
    segment_visits: int = 0
    source_visits: int = 0
    part_visits: int = 0
    range_visits: int = 0
    matches: int = 0
    checkpoints: int = 0
    work_units: int = 0


@dataclass(slots=True)
class _Budget:
    """Aggregate work and output bounds for one transform evaluation."""

    input_characters: int
    work: int = 0
    output_characters: int = 0
    segments: int = 0

    def charge_work(self, amount: int) -> None:
        """Charge bounded operation work without allowing an attacker-sized integer."""

        self.work = min(MAX_TRANSFORM_WORK + 1, self.work + amount)
        if self.work > MAX_TRANSFORM_WORK:
            raise ValueError(f"content transform exceeds {MAX_TRANSFORM_WORK} character-work units")

    def keep(self, output_characters: int, segments: tuple[ProvenanceSegment, ...]) -> None:
        self.output_characters += output_characters
        self.segments += len(segments)
        if self.output_characters > MAX_TRANSFORM_OUTPUT_CHARACTERS:
            raise ValueError(
                f"content transform output exceeds {MAX_TRANSFORM_OUTPUT_CHARACTERS} characters"
            )
        allowed = (
            self.input_characters * MAX_TRANSFORM_EXPANSION_RATIO
            + MAX_TRANSFORM_INSERTED_CHARACTERS
        )
        if self.output_characters > allowed:
            raise ValueError(
                "content transform output exceeds its expansion bound of "
                f"{MAX_TRANSFORM_EXPANSION_RATIO}x input plus "
                f"{MAX_TRANSFORM_INSERTED_CHARACTERS} inserted characters"
            )
        if self.segments > MAX_TRANSFORM_MAPPING_SEGMENTS:
            raise ValueError(
                "content transform provenance exceeds "
                f"{MAX_TRANSFORM_MAPPING_SEGMENTS} mapping segments"
            )


class _OperationMeter:
    """Count one operation's real visits and cooperatively check long loops."""

    __slots__ = ("_limit", "_metrics", "_pacer", "_pending", "work")

    def __init__(self, metrics: _TransformMetrics, *, limit: int | None = None) -> None:
        self._metrics = metrics
        self._pacer = Pacer()
        self._pending = 0
        self.work = 0
        self._limit = MAX_TRANSFORM_WORK if limit is None else limit

    def record(
        self,
        *,
        characters: int = 0,
        output: int = 0,
        segments: int = 0,
        sources: int = 0,
        parts: int = 0,
        ranges: int = 0,
        matches: int = 0,
    ) -> bool:
        amount = characters + output + segments + sources + parts + ranges + matches
        self._metrics.character_visits += characters
        self._metrics.output_characters += output
        self._metrics.segment_visits += segments
        self._metrics.source_visits += sources
        self._metrics.part_visits += parts
        self._metrics.range_visits += ranges
        self._metrics.matches += matches
        self._metrics.work_units = min(MAX_TRANSFORM_WORK + 1, self._metrics.work_units + amount)
        self.work = min(self._limit + 1, self.work + amount)
        self._pending += amount
        if self.work > self._limit:
            raise ValueError(f"content transform exceeds {MAX_TRANSFORM_WORK} character-work units")
        return self._pending >= _LOOP_CHECK_INTERVAL

    async def checkpoint(self, *, finish: bool = False) -> None:
        checks = self._pending // _LOOP_CHECK_INTERVAL
        if finish and self._pending % _LOOP_CHECK_INTERVAL:
            checks += 1
        if not checks:
            return
        self._pending %= _LOOP_CHECK_INTERVAL
        for _ in range(checks):
            self._metrics.checkpoints += 1
            raise_if_evaluation_expired()
            if self._metrics.checkpoints % _CANCELLATION_CHECK_INTERVAL == 0 or self._pacer.due():
                await asyncio.sleep(0)


def _slice_source(source: ContentSourceRange, offset: int, length: int) -> ContentSourceRange:
    return ContentSourceRange(
        message_index=source.message_index,
        part_index=source.part_index,
        start=source.start + offset,
        end=source.start + offset + length,
        message_start=source.message_start + offset,
        message_end=source.message_start + offset + length,
    )


def _slice_segments(
    segments: tuple[ProvenanceSegment, ...], start: int, end: int
) -> tuple[ProvenanceSegment, ...]:
    """Return the provenance of one output slice, rebased to zero."""

    sliced: list[ProvenanceSegment] = []
    for segment in segments:
        low, high = max(start, segment.start), min(end, segment.end)
        if low >= high:
            continue
        sources = segment.sources
        kind = segment.kind
        if (
            kind == "linear"
            and len(sources) == 1
            and segment.end - segment.start == sources[0].end - sources[0].start
        ):
            sources = (_slice_source(sources[0], low - segment.start, high - low),)
        sliced.append(ProvenanceSegment(low - start, high - start, sources, kind, segment.parts))
    return _merge_segments(sliced)


def _shift_segments(
    segments: Iterable[ProvenanceSegment], offset: int
) -> tuple[ProvenanceSegment, ...]:
    return tuple(
        ProvenanceSegment(
            segment.start + offset,
            segment.end + offset,
            segment.sources,
            segment.kind,
            segment.parts,
        )
        for segment in segments
    )


def _merge_segments(segments: Iterable[ProvenanceSegment]) -> tuple[ProvenanceSegment, ...]:
    """Coalesce adjacent mappings without losing source precision."""

    merged: list[ProvenanceSegment] = []
    for segment in segments:
        _append_segment(merged, segment)
    return tuple(merged)


def _append_segment(merged: list[ProvenanceSegment], segment: ProvenanceSegment) -> None:
    """Append one segment, merging the exact adjacent case in constant time."""

    last = merged[-1] if merged else None
    if (
        last is None
        or last.end != segment.start
        or last.kind != segment.kind
        or last.parts != segment.parts
    ):
        merged.append(segment)
        return
    if last.kind == "whole" and last.sources == segment.sources:
        merged[-1] = ProvenanceSegment(last.start, segment.end, last.sources, "whole", last.parts)
        return
    if (
        last.kind == "linear"
        and len(last.sources) == len(segment.sources) == 1
        and last.sources[0].message_index == segment.sources[0].message_index
        and last.sources[0].part_index == segment.sources[0].part_index
        and last.sources[0].end == segment.sources[0].start
        and last.sources[0].message_end == segment.sources[0].message_start
    ):
        source = ContentSourceRange(
            last.sources[0].message_index,
            last.sources[0].part_index,
            last.sources[0].start,
            segment.sources[0].end,
            last.sources[0].message_start,
            segment.sources[0].message_end,
        )
        merged[-1] = ProvenanceSegment(last.start, segment.end, (source,), parts=last.parts)
        return
    merged.append(segment)


def _check_intermediate(output_characters: int, segments: int) -> None:
    if output_characters > MAX_TRANSFORM_OUTPUT_CHARACTERS:
        raise ValueError(
            f"content transform output exceeds {MAX_TRANSFORM_OUTPUT_CHARACTERS} characters"
        )
    if segments > MAX_TRANSFORM_MAPPING_SEGMENTS:
        raise ValueError(
            "content transform provenance exceeds "
            f"{MAX_TRANSFORM_MAPPING_SEGMENTS} mapping segments"
        )


@dataclass(frozen=True, slots=True)
class _MappedPiece:
    """One monotonically consumed input interval and its exact current provenance."""

    start: int
    end: int
    sources: tuple[ContentSourceRange, ...]
    kind: MappingKind
    parts: tuple[ContentSourcePart, ...] = ()


class _MappingCursor:
    """Consume ordered provenance once, retaining a segment across adjacent requests."""

    __slots__ = ("_index", "_meter", "_position", "_segments")

    def __init__(
        self,
        segments: tuple[ProvenanceSegment, ...],
        meter: _OperationMeter,
    ) -> None:
        self._segments = segments
        self._meter = meter
        self._index = 0
        self._position = 0

    async def take(self, end: int) -> tuple[_MappedPiece, ...]:
        if end < self._position:
            raise ValueError("content transform provenance cursor cannot move backwards")
        pieces: list[_MappedPiece] = []
        while self._position < end:
            while (
                self._index < len(self._segments)
                and self._segments[self._index].end <= self._position
            ):
                self._index += 1
            segment = self._segments[self._index] if self._index < len(self._segments) else None
            if segment is None or segment.start > self._position:
                high = min(end, segment.start if segment is not None else end)
                self._meter.record(segments=1)
                if self._meter._pending >= _LOOP_CHECK_INTERVAL:
                    await self._meter.checkpoint()
                pieces.append(_MappedPiece(self._position, high, (), "whole"))
                self._position = high
                continue
            high = min(end, segment.end)
            sources = segment.sources
            if (
                segment.kind == "linear"
                and len(sources) == 1
                and segment.end - segment.start == sources[0].end - sources[0].start
            ):
                sources = (
                    _slice_source(
                        sources[0],
                        self._position - segment.start,
                        high - self._position,
                    ),
                )
            self._meter.record(segments=1)
            if self._meter._pending >= _LOOP_CHECK_INTERVAL:
                await self._meter.checkpoint()
            pieces.append(_MappedPiece(self._position, high, sources, segment.kind, segment.parts))
            self._position = high
        return tuple(pieces)


def _append_source_range(ranges: list[ContentSourceRange], source: ContentSourceRange) -> None:
    last = ranges[-1] if ranges else None
    if (
        last is not None
        and last.message_index == source.message_index
        and last.part_index == source.part_index
        and last.end == source.start
        and last.message_end == source.message_start
    ):
        ranges[-1] = ContentSourceRange(
            last.message_index,
            last.part_index,
            last.start,
            source.end,
            last.message_start,
            source.message_end,
        )
    else:
        ranges.append(source)


def _replacement_sources(
    pieces: tuple[_MappedPiece, ...],
    meter: _OperationMeter,
) -> tuple[ContentSourceRange, ...]:
    """Map a replacement only when every matched character is source-backed."""

    if not pieces or any(not piece.sources for piece in pieces):
        return ()
    if len(pieces) == 1:
        return pieces[0].sources
    ranges: list[ContentSourceRange] = []
    seen: set[ContentSourceRange] = set()
    visits = 0
    for piece in pieces:
        for source in piece.sources:
            meter.record(sources=1)
            visits += 1
            if visits % _LOOP_CHECK_INTERVAL == 0:
                raise_if_evaluation_expired()
            if source in seen:
                continue
            seen.add(source)
            _append_source_range(ranges, source)
            seen.add(ranges[-1])
    return tuple(ranges)


def _replacement_parts(
    pieces: tuple[_MappedPiece, ...], meter: _OperationMeter
) -> tuple[ContentSourcePart, ...]:
    if not pieces or any(not (piece.sources or piece.parts) for piece in pieces):
        return ()
    parts: dict[ContentSourcePart, None] = {}
    contextual = any(piece.parts for piece in pieces)
    for piece in pieces:
        for part in piece.parts:
            parts[part] = None
            meter.record(sources=1)
        if contextual:
            for source in piece.sources:
                parts[ContentSourcePart(source.message_index, source.part_index)] = None
                meter.record(sources=1)
        if len(parts) > MAX_TRANSFORM_MAPPING_SEGMENTS:
            raise ValueError("content transform has too many source parts")
        if meter._pending >= _LOOP_CHECK_INTERVAL:
            raise_if_evaluation_expired()
    return tuple(parts)


def _append_kept_pieces(
    mapped: list[ProvenanceSegment],
    pieces: tuple[_MappedPiece, ...],
    output_at: int,
) -> int:
    for index, piece in enumerate(pieces, start=1):
        length = piece.end - piece.start
        _append_segment(
            mapped,
            ProvenanceSegment(
                output_at, output_at + length, piece.sources, piece.kind, piece.parts
            ),
        )
        output_at += length
        if index % _LOOP_CHECK_INTERVAL == 0:
            raise_if_evaluation_expired()
    return output_at


async def _strip(
    text: str,
    segments: tuple[ProvenanceSegment, ...],
    meter: _OperationMeter,
) -> tuple[str, tuple[ProvenanceSegment, ...]]:
    start = len(text) - len(text.lstrip())
    end = len(text.rstrip())
    if end < start:
        end = start
    result = text[start:end]
    mapped = _slice_segments(segments, start, end)
    meter.record(characters=len(text), output=len(result), segments=len(segments))
    await meter.checkpoint(finish=True)
    return result, mapped


async def _casefold(
    text: str,
    segments: tuple[ProvenanceSegment, ...],
    meter: _OperationMeter,
) -> tuple[str, tuple[ProvenanceSegment, ...]]:
    pieces: list[str] = []
    mapped: list[ProvenanceSegment] = []
    output = 0
    cursor = _MappingCursor(segments, meter)
    for piece in await cursor.take(len(text)):
        exact_linear = (
            piece.kind == "linear"
            and len(piece.sources) == 1
            and piece.end - piece.start == piece.sources[0].end - piece.sources[0].start
        )
        for index in range(piece.start, piece.end):
            folded = text[index].casefold()
            meter.record(characters=1, output=len(folded))
            if folded:
                sources = (
                    (_slice_source(piece.sources[0], index - piece.start, 1),)
                    if exact_linear
                    else piece.sources
                )
                kind = "linear" if exact_linear and len(folded) == 1 else "whole"
                _append_segment(
                    mapped,
                    ProvenanceSegment(output, output + len(folded), sources, kind, piece.parts),
                )
                pieces.append(folded)
                output += len(folded)
                _check_intermediate(output, len(mapped))
            if meter._pending >= _LOOP_CHECK_INTERVAL:
                await meter.checkpoint()
    await meter.checkpoint(finish=True)
    return "".join(pieces), tuple(mapped)


async def _replace(
    text: str,
    segments: tuple[ProvenanceSegment, ...],
    old: str,
    new: str,
    meter: _OperationMeter,
) -> tuple[str, tuple[ProvenanceSegment, ...]]:
    pieces: list[str] = []
    mapped: list[ProvenanceSegment] = []
    cursor = _MappingCursor(segments, meter)
    source_at = output_at = 0
    while True:
        found = text.find(old, source_at)
        if found < 0:
            break
        if found > source_at:
            kept = text[source_at:found]
            pieces.append(kept)
            kept_pieces = await cursor.take(found)
            output_at = _append_kept_pieces(mapped, kept_pieces, output_at)
            meter.record(characters=len(kept), output=len(kept))
        matched_pieces = await cursor.take(found + len(old))
        meter.record(characters=len(old), output=len(new), matches=1)
        if new:
            pieces.append(new)
            _append_segment(
                mapped,
                ProvenanceSegment(
                    output_at,
                    output_at + len(new),
                    _replacement_sources(matched_pieces, meter),
                    "whole",
                    _replacement_parts(matched_pieces, meter),
                ),
            )
            output_at += len(new)
        _check_intermediate(output_at, len(mapped))
        source_at = found + len(old)
        if meter._pending >= _LOOP_CHECK_INTERVAL:
            await meter.checkpoint()
    if source_at == 0:
        meter.record(characters=len(text), output=len(text))
        await meter.checkpoint(finish=True)
        return text, segments
    if source_at < len(text):
        kept = text[source_at:]
        pieces.append(kept)
        kept_pieces = await cursor.take(len(text))
        output_at = _append_kept_pieces(mapped, kept_pieces, output_at)
        meter.record(characters=len(kept), output=len(kept))
        _check_intermediate(output_at, len(mapped))
    await meter.checkpoint(finish=True)
    return "".join(pieces), tuple(mapped)


async def _insert(
    text: str,
    segments: tuple[ProvenanceSegment, ...],
    value: str,
    meter: _OperationMeter,
    *,
    prefix: bool,
) -> tuple[str, tuple[ProvenanceSegment, ...]]:
    if not value:
        meter.record(characters=len(text), output=len(text), segments=len(segments))
        await meter.checkpoint(finish=True)
        return text, segments
    _check_intermediate(len(text) + len(value), len(segments) + 1)
    meter.record(
        characters=len(text),
        output=len(text) + len(value),
        segments=len(segments) + 1,
    )
    await meter.checkpoint(finish=True)
    inserted = ProvenanceSegment(0, len(value), (), "whole")
    if prefix:
        return value + text, (inserted, *_shift_segments(segments, len(value)))
    inserted = ProvenanceSegment(len(text), len(text) + len(value), (), "whole")
    return text + value, (*segments, inserted)


async def _apply_operation(
    text: str,
    segments: tuple[ProvenanceSegment, ...],
    operation: TransformOperation,
    meter: _OperationMeter,
) -> tuple[str, tuple[ProvenanceSegment, ...]]:
    if operation.kind == "strip":
        return await _strip(text, segments, meter)
    if operation.kind == "casefold":
        return await _casefold(text, segments, meter)
    if operation.kind == "replace":
        return await _replace(text, segments, operation.value, operation.replacement, meter)
    return await _insert(
        text,
        segments,
        operation.value,
        meter,
        prefix=operation.kind == "prefix",
    )


async def _partition_text_units(
    message: MessageType,
    provenance: tuple[ProvenanceSegment, ...],
    meter: _OperationMeter,
) -> tuple[tuple[int | None, str, tuple[ProvenanceSegment, ...]], ...]:
    """Partition flattened provenance across content parts in one monotonic pass."""

    source = message.source_content
    cursor = _MappingCursor(provenance, meter)
    units: list[tuple[int | None, str, tuple[ProvenanceSegment, ...]]] = []

    async def append_unit(part_index: int | None, text: str, start: int, end: int) -> None:
        meter.record(ranges=1)
        if meter._pending >= _LOOP_CHECK_INTERVAL:
            await meter.checkpoint()
        pieces = await cursor.take(end)
        segments = _merge_segments(
            ProvenanceSegment(
                piece.start - start,
                piece.end - start,
                piece.sources,
                piece.kind,
                piece.parts,
            )
            for piece in pieces
        )
        units.append((part_index, text, segments))

    if isinstance(source, str):
        meter.record(parts=1)
        if meter._pending >= _LOOP_CHECK_INTERVAL:
            await meter.checkpoint()
        await append_unit(None, source, 0, len(source))
        await meter.checkpoint(finish=True)
        return tuple(units)
    if not isinstance(source, tuple):
        return ()
    offset = 0
    for part_index, part in enumerate(source):
        meter.record(parts=1)
        if meter._pending >= _LOOP_CHECK_INTERVAL:
            await meter.checkpoint()
        if isinstance(part, Mapping) and part.get("type") == "text":
            text = part.get("text", "")
            if not isinstance(text, str):
                raise TypeError("text content parts must contain strings")
            end = offset + len(text)
            await append_unit(part_index, text, offset, end)
            offset += len(text)
    await meter.checkpoint(finish=True)
    return tuple(units)


async def apply_content_transform(
    messages: Messages,
    transform: ContentTransform,
    *,
    _metrics: _TransformMetrics | None = None,
) -> Messages:
    """Apply one immutable transform to selected message text parts."""

    if not isinstance(messages, Messages):
        raise TypeError("content transform receiver must be Messages")
    if not isinstance(transform, ContentTransform):
        raise TypeError("content transform definition must be ContentTransform")
    unique = {
        id(message): message for message in messages if not isinstance(message, UnreadableMessage)
    }
    input_characters = sum(len(message.content) for message in unique.values())
    if input_characters > MAX_TRANSFORM_INPUT_CHARACTERS:
        raise ValueError(
            f"content transform input exceeds {MAX_TRANSFORM_INPUT_CHARACTERS} characters"
        )
    budget = _Budget(input_characters)
    metrics = _metrics if _metrics is not None else _TransformMetrics()
    overrides: dict[int, MessageType] = {}
    pacer = Pacer()
    for identity, message in unique.items():
        if pacer.due():
            await asyncio.sleep(0)
        raise_if_evaluation_expired()
        provenance = message._content_provenance()
        _check_intermediate(len(message.content), len(provenance))
        setup_meter = _OperationMeter(
            metrics,
            limit=max(0, MAX_TRANSFORM_WORK - budget.work),
        )
        text_units = await _partition_text_units(message, provenance, setup_meter)
        budget.charge_work(setup_meter.work)
        replacements: dict[int | None, str] = {}
        mapped: list[ProvenanceSegment] = []
        output_offset = 0
        changed = False
        for part_index, source_text, segments in text_units:
            text = source_text
            for operation in transform.operations:
                if pacer.due():
                    await asyncio.sleep(0)
                raise_if_evaluation_expired()
                meter = _OperationMeter(
                    metrics,
                    limit=max(0, MAX_TRANSFORM_WORK - budget.work),
                )
                text, segments = await _apply_operation(text, segments, operation, meter)
                _check_intermediate(len(text), len(segments))
                budget.charge_work(meter.work)
            replacements[part_index] = text
            mapped.extend(_shift_segments(segments, output_offset))
            output_offset += len(text)
            changed = changed or text != source_text
        final_segments = _merge_segments(mapped)
        budget.keep(output_offset, final_segments)
        if changed or final_segments != provenance:
            overrides[identity] = message._with_transformed_content(replacements, final_segments)
    return (
        messages
        if not overrides
        else messages._with_position_overrides(
            {
                position: overrides[id(message)]
                for position, message in enumerate(messages)
                if id(message) in overrides
            }
        )
    )


@lru_cache(maxsize=256)
def compiled_content_transform(value: str) -> ContentTransform:
    """Decode one compiler-owned transform identity once."""

    return ContentTransform.from_canonical_json(value)
