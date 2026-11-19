# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Exact, bounded interval bookkeeping for validated logical input measurements.

Input identities are opaque digests supplied by the reader. Character positions and canonical
text-token positions are independent coordinate systems. A token provider must establish its
source mapping before supplying intervals here; unavailable or failed measurements belong in
the reader's status record. This layer retains only identities, integer intervals and totals.
"""

from __future__ import annotations

import os
import threading
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Literal

ReadUnit = Literal["characters", "text_tokens"]
_MAX_OFFSET = (1 << 63) - 1
_MAX_BUCKETS = 65_536
_MAX_INTERVALS = 262_144
_MAX_WORK = 10_000_000
_INVALID_RANGE = "invalid logical input coverage interval"


class CoverageLimitError(ValueError):
    """Optional accounting exhausted its independent storage or traversal budget."""


class CoverageOwnerError(RuntimeError):
    """A process cannot resume inherited accounting locks or state."""


@dataclass(frozen=True, slots=True)
class ReadInterval:
    """One half-open range in a stable input's character or canonical token coordinates.

    The reader's identity must include the input variant and measurement basis. Equal
    offsets from different inputs, transformations or tokenization bases cannot share it.
    Empty ranges contribute no covered units and consume no retained interval storage.
    """

    source: bytes = field(repr=False)
    unit: ReadUnit
    start: int
    end: int

    def __post_init__(self) -> None:
        _validate_interval(self)


def _validate_interval(value: object) -> None:
    if type(value) is not ReadInterval:
        raise ValueError(_INVALID_RANGE)
    try:
        source, unit, start, end = value.source, value.unit, value.start, value.end
    except AttributeError:
        raise ValueError(_INVALID_RANGE) from None
    if (
        type(source) is not bytes
        or len(source) != 32
        or type(unit) is not str
        or unit not in {"characters", "text_tokens"}
        or type(start) is not int
        or type(end) is not int
        or not 0 <= start <= end <= _MAX_OFFSET
    ):
        raise ValueError(_INVALID_RANGE)


@dataclass(frozen=True, slots=True)
class CoverageTotals:
    """Totals over supplied validated ranges, with no retained input identifiers."""

    characters: int
    text_tokens: int
    buckets: int
    intervals: int
    work: int


@dataclass(frozen=True, slots=True)
class CoverageDelta:
    """Newly covered units committed by one atomic observation batch."""

    characters: int
    text_tokens: int


class ReadCoverage:
    """Union validated intervals atomically within independent accounting limits.

    Each (input identity, unit) bucket contains sorted, disjoint, nonadjacent intervals.
    Duplicate and overlapping observations count their union. Rejected batches leave all
    coverage unchanged; inspection work still counts against the lifetime budget. Work
    charges conservative bounds for references inspected or copied, including failed batches.
    The containing reader decides how to report incomplete or unavailable measurements.
    """

    def __init__(
        self, *, max_buckets: int = 4096, max_intervals: int = 16_384, max_work: int = 100_000
    ) -> None:
        for value, ceiling in [
            (max_buckets, _MAX_BUCKETS),
            (max_intervals, _MAX_INTERVALS),
            (max_work, _MAX_WORK),
        ]:
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError("invalid logical input coverage limit")
        self.max_buckets = max_buckets
        self.max_intervals = max_intervals
        self.max_work = max_work
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._ranges: dict[tuple[bytes, ReadUnit], list[tuple[int, int]]] = {}
        self._units: dict[ReadUnit, int] = {"characters": 0, "text_tokens": 0}
        self._interval_count = 0
        self._work = 0

    def _require_owner(self) -> None:
        # Check before taking a lock that may have been held by a vanished parent thread.
        if self._pid != os.getpid():
            raise CoverageOwnerError("logical input coverage belongs to another process")

    def _charge(self, cost: int) -> None:
        if cost > self.max_work - self._work:
            self._work = self.max_work
            raise CoverageLimitError("logical input coverage work limit exceeded")
        self._work += cost

    def add(self, intervals: tuple[ReadInterval, ...] | list[ReadInterval]) -> CoverageDelta:
        """Validate and commit a bounded batch, or retain none of its new coverage."""

        self._require_owner()
        if type(intervals) not in (tuple, list):
            raise ValueError("logical input coverage must be an exact tuple or list")
        if len(intervals) > self.max_intervals:
            raise CoverageLimitError("logical input coverage batch limit exceeded")
        with self._lock:
            self._charge(len(intervals))
            # Revalidate even a frozen value: a host may bypass dataclass construction.
            for interval in intervals:
                _validate_interval(interval)
            changed: dict[tuple[bytes, ReadUnit], list[tuple[int, int]]] = {}
            totals = dict(self._units)
            count = self._interval_count
            new_buckets = 0
            for interval in intervals:
                if interval.start == interval.end:
                    continue
                key = (interval.source, interval.unit)
                if key not in changed:
                    if key not in self._ranges:
                        new_buckets += 1
                        if len(self._ranges) + new_buckets > self.max_buckets:
                            raise CoverageLimitError("logical input coverage key limit exceeded")
                    previous = self._ranges.get(key, ())
                    self._charge(len(previous))
                    changed[key] = list(previous)
                bucket = changed[key]
                self._charge(4 * len(bucket) + 8)
                start, end = interval.start, interval.end
                left = bisect_left(bucket, (start, -1))
                if left and bucket[left - 1][1] >= start:
                    left -= 1
                right = left
                removed = 0
                while right < len(bucket) and bucket[right][0] <= end:
                    low, high = bucket[right]
                    start, end = min(start, low), max(end, high)
                    removed += high - low
                    right += 1
                count += 1 - (right - left)
                if count > self.max_intervals:
                    raise CoverageLimitError("logical input coverage interval limit exceeded")
                totals[interval.unit] += end - start - removed
                bucket[left:right] = [(start, end)]
            delta = CoverageDelta(
                totals["characters"] - self._units["characters"],
                totals["text_tokens"] - self._units["text_tokens"],
            )
            self._ranges.update(changed)
            self._units = totals
            self._interval_count = count
            return delta

    def snapshot(self) -> CoverageTotals:
        """Return immutable scalar totals; reader status establishes their completeness."""

        self._require_owner()
        with self._lock:
            return CoverageTotals(
                self._units["characters"],
                self._units["text_tokens"],
                len(self._ranges),
                self._interval_count,
                self._work,
            )

    def clear(self) -> None:
        """Release owned metadata and begin a fresh accounting lifetime."""

        self._require_owner()
        with self._lock:
            self._ranges.clear()
            self._units = {"characters": 0, "text_tokens": 0}
            self._interval_count = self._work = 0
