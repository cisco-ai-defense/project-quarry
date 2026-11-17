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

"""Scan-local indexed text ranges with bounded provenance work."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Coroutine, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.exceptions import EvaluationError
from quarry.types.messages import MessageType
from quarry.types.transforms import ProvenanceSegment, slice_provenance

T = TypeVar("T")


async def gather_split_tasks(awaitables: Iterable[Coroutine[Any, Any, T]]) -> list[T]:
    """Join every owned child before returning or propagating the first failure.

    Shield the group so cancellation reaches each child once. Later cancellation requests
    must not interrupt a child's async cleanup or let its caller release live value owners.
    """

    tasks: list[asyncio.Task[T]] = []
    pending: list[asyncio.Task[T]] = []
    group = None
    coroutine = task = None
    try:
        for coroutine in awaitables:
            try:
                tasks.append(asyncio.create_task(coroutine))
            except BaseException:
                coroutine.close()
                raise
        group = asyncio.gather(*tasks)
        return await asyncio.shield(group)
    finally:
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        while any(not task.done() for task in pending):
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait(pending)
        for task in tasks:
            if not task.cancelled():
                task.exception()
        if group is not None and group.done() and not group.cancelled():
            group.exception()
        # A propagated exception retains this frame. Drop the tasks/futures whose outcomes
        # may contain that same exception, along with their copied request contexts.
        del tasks, pending, group, coroutine, task, awaitables


MAX_SPLIT_RANGE_WORK = 4_000_000
_CHECK_INTERVAL = 256


@dataclass(slots=True)
class RangeMetrics:
    source_parts: int = 0
    projection_steps: int = 0
    prepared_segments: int = 0
    prepared_references: int = 0
    queries: int = 0
    lookup_steps: int = 0
    selected_segments: int = 0
    selected_references: int = 0
    output_characters: int = 0
    checkpoints: int = 0
    work_units: int = 0


class RangeBudget:
    def __init__(self, metrics: RangeMetrics | None = None):
        self.metrics = RangeMetrics() if metrics is None else metrics
        self._pending = 0
        self._until_check = _CHECK_INTERVAL

    def add(self, field: str, amount: int = 1) -> None:
        charged = min(amount, max(0, MAX_SPLIT_RANGE_WORK + 1 - self.metrics.work_units))
        setattr(self.metrics, field, getattr(self.metrics, field) + charged)
        self.metrics.work_units += charged
        self._pending += charged
        self._until_check -= charged
        if self.metrics.work_units > MAX_SPLIT_RANGE_WORK:
            raise EvaluationError(
                "Split text mapping exceeds its work limit", {"code": "split_work_limit"}
            )
        if self._until_check <= 0:
            raise_if_evaluation_expired()
            self._until_check = _CHECK_INTERVAL

    async def checkpoint(self, *, finish: bool = False) -> None:
        if self._pending >= _CHECK_INTERVAL or finish:
            raise_if_evaluation_expired()
            self._pending = 0
            self.metrics.checkpoints += 1
            await asyncio.sleep(0)
            raise_if_evaluation_expired()


@dataclass(frozen=True, slots=True)
class MessageRangeIndex:
    message: MessageType
    text: str
    segments: tuple[ProvenanceSegment, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    budget: RangeBudget

    @classmethod
    async def prepare(cls, message: MessageType, budget: RangeBudget) -> MessageRangeIndex:
        raise_if_evaluation_expired()
        source = message.source_content
        if isinstance(source, tuple):
            if not source:
                raise ValueError("only a message with text-only content can be divided")
            for part in source:
                budget.add("source_parts")
                if not isinstance(part, Mapping) or part.get("type") != "text":
                    raise ValueError("only a message with text-only content can be divided")
                await budget.checkpoint()
        elif not isinstance(source, str):
            raise ValueError("only a message with text-only content can be divided")
        text = message.content
        segments = []
        starts = []
        ends = []
        previous_end = 0
        for segment in message._provenance_steps():
            budget.add("projection_steps")
            if segment is not None:
                budget.add("prepared_segments")
                budget.add("prepared_references", len(segment.sources) + len(segment.parts))
                if segment.start < previous_end or segment.end > len(text):
                    raise ValueError("split text provenance is not an ordered message map")
                segments.append(segment)
                starts.append(segment.start)
                ends.append(segment.end)
                previous_end = segment.end
            await budget.checkpoint()
        await budget.checkpoint(finish=True)
        return cls(message, text, tuple(segments), tuple(starts), tuple(ends), budget)

    def _bound(self, positions: tuple[int, ...], value: int, *, right: bool) -> int:
        low, high = 0, len(positions)
        while low < high:
            self.budget.add("lookup_steps")
            middle = (low + high) // 2
            if positions[middle] < value or right and positions[middle] == value:
                low = middle + 1
            else:
                high = middle
        return low

    def segment(self, start: int, end: int) -> MessageType:
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(self.text):
            raise ValueError("message segment is outside the message content")
        self.budget.add("queries")
        self.budget.add("output_characters", end - start)
        first = self._bound(self.ends, start, right=True)
        beyond = self._bound(self.starts, end, right=False)
        selected = []
        for index in range(first, beyond):
            segment = self.segments[index]
            self.budget.add("selected_segments")
            self.budget.add("selected_references", len(segment.sources) + len(segment.parts))
            selected.append(segment)
        mapping = slice_provenance(tuple(selected), start, end)
        return self.message._copy_text_range(start, end, self.text, mapping)
