# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Prepared original-content ranges for repeated window span decoding."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.messages import Message
from quarry.types.origins import Derived, Place, _mapped
from quarry.types.transforms import ProvenanceSegment, source_mapping_for, source_part_mapping_for

MAX_WINDOW_RANGE_WORK = 4_000_000


class WindowRangeBudget:
    def __init__(self):
        self.work = 0
        self.until_check = 256

    def charge(self, amount=1):
        self.work += amount
        self.until_check -= amount
        if self.work > MAX_WINDOW_RANGE_WORK:
            raise ValueError("model window source mapping exceeds its work limit")
        if self.until_check <= 0:
            raise_if_evaluation_expired()
            self.until_check = 256

    async def checkpoint(self):
        await asyncio.sleep(0)
        raise_if_evaluation_expired()


@dataclass(frozen=True, slots=True)
class WindowRangeIndex:
    text_length: int
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    segments: tuple[ProvenanceSegment, ...]
    budget: WindowRangeBudget

    @classmethod
    async def prepare(cls, message: Message, budget: WindowRangeBudget):
        raise_if_evaluation_expired()
        text_length = len(message.content)
        starts, ends, segments = [], [], []
        previous = 0
        for ordinal, segment in enumerate(message._provenance_steps()):
            budget.charge()
            if segment is not None:
                budget.charge(len(segment.sources) + len(segment.parts))
                if segment.start < previous or segment.end > text_length:
                    raise ValueError("model window source map is not ordered")
                starts.append(segment.start)
                ends.append(segment.end)
                segments.append(segment)
                previous = segment.end
            if ordinal % 256 == 0:
                await budget.checkpoint()
        return cls(text_length, tuple(starts), tuple(ends), tuple(segments), budget)

    def _bound(self, values, value, right=False):
        low, high = 0, len(values)
        while low < high:
            self.budget.charge()
            middle = (low + high) // 2
            if values[middle] < value or right and values[middle] == value:
                low = middle + 1
            else:
                high = middle
        return low

    def mapping(self, start, end):
        if not 0 <= start < end <= self.text_length:
            raise ValueError("model window source range is outside its message")
        self.budget.charge()
        first = self._bound(self.ends, start, right=True)
        beyond = self._bound(self.starts, end)
        selected = self.segments[first:beyond]
        for segment in selected:
            self.budget.charge(1 + len(segment.sources) + len(segment.parts))
        ranges, unmapped = source_mapping_for(selected, start, end)
        parts, located = source_part_mapping_for(selected, start, end)
        return ranges, unmapped, parts, located


def origin_intervals(origin, start, end, budget, depth=0):
    """Visit touched runs separately so dropped source text never becomes claimed coverage."""
    budget.charge()
    if depth > 100:
        raise ValueError("model window origin exceeds its depth limit")
    if type(origin) is Place:
        yield origin, origin.start + start, origin.start + end
    elif type(origin) is Derived:
        for fields in origin._touched(start, end):
            budget.charge()
            low, _ = _mapped(fields, max(start, fields[0]), end=False)
            high, _ = _mapped(fields, min(end, fields[1]), end=True)
            if low < high:
                yield from origin_intervals(origin.sources[fields[2]], low, high, budget, depth + 1)
    else:
        raise TypeError("model window origin must contain native source places")
