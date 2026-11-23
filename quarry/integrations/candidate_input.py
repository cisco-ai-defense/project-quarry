# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Separate detached candidate input from reader-owned native source bindings.

Candidate providers receive text parts, never the source objects held here. Every
part has one linear native origin, so a valid subrange cannot acquire authority
over a separator, another message, another content part or an enclosing rewrite.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from quarry.engine.local_work import measured_local_work
from quarry.integrations.message_readability import require_model_messages
from quarry.integrations.model_window_input import message_window_origin
from quarry.integrations.request_limits import raise_if_owned_evaluation_expired
from quarry.types.candidates import (
    CandidateBatchLimits,
    CandidateInput,
    CandidateLimitError,
    CandidatePart,
    CandidateRange,
    CandidateReplyError,
)
from quarry.types.messages import Message, Messages, ToolCall
from quarry.types.origins import LINEAR, Derived, Place, SourceText
from quarry.types.results import Score, Span
from quarry.types.transforms import ContentSourceRange


class _BindingBudget:
    def __init__(self, maximum: int):
        self.maximum = maximum
        self.work = 0

    def charge(self, amount=1):
        before = self.work
        self.work += amount
        if self.work > self.maximum:
            raise CandidateLimitError("candidate source binding work limit exceeded")
        if before // 256 != self.work // 256:
            raise_if_owned_evaluation_expired()

    async def checkpoint(self):
        await asyncio.sleep(0)
        raise_if_owned_evaluation_expired()


@dataclass(frozen=True, slots=True, repr=False)
class _BoundPart:
    source: SourceText
    original: ContentSourceRange | None

    def span(self, start: int, end: int) -> Span:
        result = self.source.span(start, end)
        if self.original is not None:
            original = self.original
            result = replace(
                result,
                source_ranges=(
                    ContentSourceRange(
                        original.message_index,
                        original.part_index,
                        original.start + start,
                        original.start + end,
                        original.message_start + start,
                        original.message_start + end,
                    ),
                ),
            )
            assert type(self.source.origin) is Place
            result = result._bind_to_message(self.source.origin.owner)
        return result


@dataclass(slots=True)
class CandidateBindings:
    """A reader's source owners; close drops them independently of detached input."""

    input: CandidateInput = field(repr=False)
    _parts: tuple[_BoundPart, ...] = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def source_texts(self) -> tuple[SourceText, ...]:
        if self._closed:
            raise CandidateReplyError("candidate source bindings are closed")
        return tuple(part.source for part in self._parts)

    def bind(self, value: CandidateRange) -> Span:
        if self._closed:
            raise CandidateReplyError("candidate source bindings are closed")
        if type(value) is not CandidateRange:
            raise CandidateReplyError("invalid native candidate range")
        value = CandidateRange(value.part, value.start, value.end, value.score)
        if value.part >= len(self._parts):
            raise CandidateReplyError("candidate range references a foreign source part")
        part = self._parts[value.part]
        if value.end > len(part.source.text):
            raise CandidateReplyError("candidate range crosses its native source part")
        return replace(part.span(value.start, value.end), score=Score(value.score))

    def close(self) -> None:
        self._closed = True
        self._parts = ()


@measured_local_work("prepare")
def _native_size(message, budget, remaining_parts, remaining_characters):
    """Bound raw parts before accessing the flattening Message.content property."""
    source = message.source_content
    if source is None:
        return 0, 1
    if type(source) is str:
        return len(source), 1
    if type(source) is not tuple:
        raise CandidateReplyError("candidate input requires native text content")
    if len(source) > remaining_parts:
        raise CandidateLimitError("candidate input exceeds its source-part limit")
    characters = 0
    for part in source:
        budget.charge()
        if not isinstance(part, Mapping) or part.get("type") != "text":
            raise CandidateReplyError("candidate input requires text with exact source positions")
        text = part.get("text", "")
        if type(text) is not str:
            raise CandidateReplyError("candidate input requires native text content")
        characters += len(text)
        if characters > remaining_characters:
            raise CandidateLimitError("candidate input exceeds its character limit")
    return characters, max(1, len(source))


@measured_local_work("iterate", iterator=True)
def _exact_places(origin, start, end, budget, depth=0):
    """Split at every origin run; never merge their enclosing source intervals."""
    budget.charge()
    if depth > 100:
        raise CandidateLimitError("candidate source graph depth limit exceeded")
    if type(origin) is Place:
        yield origin, origin.start + start, origin.start + end
        return
    if type(origin) is not Derived:
        raise CandidateReplyError("candidate input has no native source origin")
    cursor = start
    for fields in origin._touched(start, end):
        budget.charge()
        low, high = max(start, fields[0]), min(end, fields[1])
        if low != cursor or fields[5] != LINEAR or fields[1] - fields[0] != fields[4] - fields[3]:
            raise CandidateReplyError("candidate input requires exact linear source positions")
        source_start = fields[3] + low - fields[0]
        yield from _exact_places(
            origin.sources[fields[2]], source_start, source_start + high - low, budget, depth + 1
        )
        cursor = high
    if cursor != end:
        raise CandidateReplyError("candidate input contains text without source positions")


@measured_local_work("iterate", iterator=True)
def _source_parts(place, start, end, budget):
    """Split flat native message coordinates back into original text parts."""
    if not isinstance(place.owner, Message) or place.field:
        yield start, end, None
        return
    cursor = start
    for segment in place.owner._provenance_steps():
        budget.charge()
        if segment is None or segment.end <= start:
            continue
        if segment.start >= end:
            break
        low, high = max(start, segment.start), min(end, segment.end)
        if low != cursor or segment.kind != "linear" or segment.parts or len(segment.sources) != 1:
            raise CandidateReplyError("candidate source is unmapped or spans multiple parts")
        source = segment.sources[0]
        if (
            source.message_index != place.owner.message_index
            or segment.end - segment.start != source.end - source.start
            or segment.end - segment.start != source.message_end - source.message_start
        ):
            raise CandidateReplyError("candidate source has no exact native character mapping")
        offset = low - segment.start
        yield (
            low,
            high,
            ContentSourceRange(
                source.message_index,
                source.part_index,
                source.start + offset,
                source.start + offset + high - low,
                source.message_start + offset,
                source.message_start + offset + high - low,
            ),
        )
        cursor = high
    if cursor != end:
        raise CandidateReplyError("candidate source does not cover its native text")


@measured_local_work("prepare")
def _detached_input(parts):
    return CandidateInput(tuple(parts))


async def bind_candidate_input(
    messages: Messages, limits: CandidateBatchLimits
) -> CandidateBindings:
    """Prove each whole part's geometry before any provider invocation or filtering."""
    raise_if_owned_evaluation_expired()
    if type(messages) is not Messages or type(limits) is not CandidateBatchLimits:
        raise CandidateReplyError("candidate input requires native messages and limits")
    limits = replace(limits)
    budget = _BindingBudget(limits.max_work)
    characters = raw_parts = 0
    for message in messages:
        budget.charge()
        size, count = _native_size(
            message, budget, limits.max_parts - raw_parts, limits.max_input_characters - characters
        )
        characters += size
        raw_parts += count
        for call in getattr(message, "tool_calls", ()):
            budget.charge()
            characters += len(call.raw_arguments)
            raw_parts += 1
        if characters > limits.max_input_characters or raw_parts > limits.max_parts:
            raise CandidateLimitError("candidate input exceeds its source-part or character limits")
    budget.charge((characters + 255) // 256)
    await require_model_messages(messages)
    parts, bindings = [], []

    def append(text, place, low, high, original, role, selected):
        budget.charge()
        if len(parts) >= limits.max_parts:
            raise CandidateLimitError("candidate input exceeds its source-part limit")
        path = (
            "message",
            selected,
            "position",
            place.position,
            "field",
            place.field,
            "part",
            "text" if original is None or original.part_index is None else original.part_index,
            "start",
            low,
            "end",
            high,
        )
        if place.source is not None:
            path += (
                "input",
                place.source.input,
                "item",
                "absent" if place.source.item is None else "present",
                place.source.item or "",
            )
        part = CandidatePart(role, path, text)
        budget.charge(len(part.path))
        for item in part.path:
            budget.charge((len(item) + 255) // 256 if type(item) is str else 1)
        parts.append(part)
        bindings.append(_BoundPart(SourceText(text, replace(place, start=low), selected), original))

    for selected, message in enumerate(messages):
        text = message.content
        origin = await message_window_origin(messages, selected, budget)
        if not text:
            # Empty messages still shape request identity; they admit no nonempty range.
            ordinal = messages._ordinals[selected]
            place = Place(messages._root._positions[ordinal], "", 0, message)
            append("", place, 0, 0, None, message.role, selected)
        offset = 0
        for place, start, end in _exact_places(origin, 0, len(text), budget):
            for low, high, original in _source_parts(place, start, end, budget):
                length = high - low
                if length:
                    append(
                        text[offset : offset + length],
                        place,
                        low,
                        high,
                        original,
                        message.role,
                        selected,
                    )
                offset += length
        if offset != len(text):
            raise CandidateReplyError("candidate input has inconsistent native source geometry")
        ordinal = messages._ordinals[selected]
        for number, call in enumerate(getattr(message, "tool_calls", ())):
            if not isinstance(call, ToolCall):
                raise CandidateReplyError("candidate input requires a readable native tool call")
            text = call.raw_arguments
            place = Place(messages._root._positions[ordinal] + number + 1, "raw_arguments", 0, call)
            append(text, place, 0, len(text), None, "tool_arguments", selected)
        await budget.checkpoint()
    raise_if_owned_evaluation_expired()
    return CandidateBindings(_detached_input(parts), tuple(bindings))
