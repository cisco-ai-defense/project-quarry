# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Bounded native range replies for an explicit external-span adapter contract.

These native values describe reply geometry and exact request identity. They do not
authorize a policy result. Adapter integration preserves incomplete-result failure
semantics, validates outside cache/observation locks, and binds each reader's native
source provenance independently.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field, replace
from hashlib import sha256
from math import isfinite

_MAX_CHARACTERS = 4_000_000
_MAX_RANGES = 4096


class ExternalSpanReplyError(ValueError):
    """An external reply does not describe a valid, bound native input observation."""


class ExternalSpanLimitError(ExternalSpanReplyError):
    """External reply inspection exceeded the reader's explicit resource budget."""


def external_span_input_digest(text: str) -> str:
    """Identify the exact immutable request, including unpaired Unicode surrogates."""
    if type(text) is not str or len(text) > _MAX_CHARACTERS:
        raise ExternalSpanLimitError("invalid external span request text")
    return sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _interval(start, end, *, empty):
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start <= end <= _MAX_CHARACTERS
        or not empty
        and start == end
    ):
        raise ExternalSpanReplyError("invalid external span character interval")


@dataclass(frozen=True, slots=True)
class ExternalReadRange:
    """An actually inspected half-open interval in the requested text's characters."""

    start: int
    end: int

    def __post_init__(self):
        _interval(self.start, self.end, empty=True)


@dataclass(frozen=True, slots=True)
class ExternalSpanRange:
    """A detection inside inspected input, without text, transport objects or labels."""

    start: int
    end: int
    score: float

    def __post_init__(self):
        _interval(self.start, self.end, empty=False)
        if (
            type(self.score) not in (int, float)
            or not 0 <= self.score <= 1
            or not isfinite(self.score)
        ):
            raise ExternalSpanReplyError("invalid external span score")
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True, slots=True)
class ExternalSpanLimits:
    """Reader limits are independent from a host's own native reply construction."""

    max_input_characters: int = 1_000_000
    max_spans: int = 4096
    max_coverage_ranges: int = 4096
    max_work: int = 262_144

    def __post_init__(self):
        for value, low, high in (
            (self.max_input_characters, 0, _MAX_CHARACTERS),
            (self.max_spans, 0, _MAX_RANGES),
            (self.max_coverage_ranges, 0, _MAX_RANGES),
            (self.max_work, 1, 4_000_000),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ExternalSpanLimitError("invalid external span limit")


def _native_sequence(values, maximum):
    if type(values) not in (tuple, list):
        raise ExternalSpanReplyError("external span ranges require a native tuple or list")
    count = len(values)
    if count > maximum:
        raise ExternalSpanLimitError("external span range count limit exceeded")
    # Native bounded slicing does not walk later additions or invoke a host override.
    snapshot = values[:count]
    if len(snapshot) != count:
        raise ExternalSpanReplyError("external span ranges changed during inspection")
    return snapshot


def _union(ranges):
    merged = []
    for start, end in sorted((r.start, r.end) for r in ranges if r.start != r.end):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


@dataclass(frozen=True, slots=True)
class ExternalSpanReply:
    """A detached native reply whose declared coverage is explicit, including emptiness.

    Complete replies cover the whole declared input. Partial replies can retain valid
    detections within their known coverage, but this type grants no verdict authority.
    A reader must bind and clone the reply with ``validate_external_span_reply``.
    """

    input_characters: int
    spans: tuple[ExternalSpanRange, ...]
    coverage: tuple[ExternalReadRange, ...]
    complete: bool
    input_digest: str | None = None
    _coverage_union: tuple[tuple[int, int], ...] = field(init=False, repr=False)

    def __post_init__(self):
        if (
            type(self.input_characters) is not int
            or not 0 <= self.input_characters <= _MAX_CHARACTERS
        ):
            raise ExternalSpanReplyError("invalid external span input length")
        if type(self.complete) is not bool:
            raise ExternalSpanReplyError("external span completeness must be a boolean")
        if self.input_digest is not None and (
            type(self.input_digest) is not str
            or len(self.input_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.input_digest)
        ):
            raise ExternalSpanReplyError("invalid external span request digest")
        spans, coverage = [], []
        for value in _native_sequence(self.spans, _MAX_RANGES):
            if type(value) is not ExternalSpanRange:
                raise ExternalSpanReplyError("invalid external detection range")
            try:
                copied = ExternalSpanRange(value.start, value.end, value.score)
            except AttributeError:
                raise ExternalSpanReplyError("invalid external detection range") from None
            if copied.end > self.input_characters:
                raise ExternalSpanReplyError("external detection exceeds the requested input")
            spans.append(copied)
        for value in _native_sequence(self.coverage, _MAX_RANGES):
            if type(value) is not ExternalReadRange:
                raise ExternalSpanReplyError("invalid external read range")
            try:
                copied = ExternalReadRange(value.start, value.end)
            except AttributeError:
                raise ExternalSpanReplyError("invalid external read range") from None
            if copied.end > self.input_characters:
                raise ExternalSpanReplyError("external coverage exceeds the requested input")
            coverage.append(copied)
        merged = _union(coverage)
        fully_covered = not self.input_characters or merged == ((0, self.input_characters),)
        if self.complete and not fully_covered:
            raise ExternalSpanReplyError("complete external reply omits input coverage")
        for span in spans:
            index = bisect_right(merged, (span.start, _MAX_CHARACTERS)) - 1
            if index < 0 or merged[index][1] < span.end:
                raise ExternalSpanReplyError("external detection lies outside inspected input")
        object.__setattr__(self, "spans", tuple(spans))
        object.__setattr__(self, "coverage", tuple(coverage))
        object.__setattr__(self, "_coverage_union", merged)

    @property
    def covered_characters(self) -> int:
        """The exact interval union, independent from detection count or overlaps."""
        return sum(end - start for start, end in self._coverage_union)

    @classmethod
    def for_input(cls, text, spans, coverage, *, complete):
        """Construct a reply bound to the exact text the host actually inspected.

        The digest detects stale replies for a different, equally sized request. It
        cannot prove that a provider's declared inspection actually took place.
        """
        digest = external_span_input_digest(text)
        return cls(len(text), spans, coverage, complete, digest)


def validate_external_span_reply(
    reply: object, input_characters: int, limits: ExternalSpanLimits | None = None
) -> ExternalSpanReply:
    """Bind geometry to actual input length and defensively copy before admission.

    Incompleteness stays explicit even when coverage happens to span the whole input.
    Budget checks precede copying and sorting, including replies mutated after creation.
    """
    if limits is not None and type(limits) is not ExternalSpanLimits:
        raise ExternalSpanLimitError("invalid external span limits")
    try:
        selected = ExternalSpanLimits() if limits is None else replace(limits)
    except AttributeError:
        raise ExternalSpanLimitError("invalid external span limits") from None
    if (
        type(input_characters) is not int
        or not 0 <= input_characters <= selected.max_input_characters
    ):
        raise ExternalSpanLimitError("external span input character limit exceeded")
    if type(reply) is not ExternalSpanReply:
        raise ExternalSpanReplyError("external adapter must return ExternalSpanReply")
    try:
        declared = reply.input_characters
        span_values, coverage_values, complete = reply.spans, reply.coverage, reply.complete
        digest = reply.input_digest
    except AttributeError:
        raise ExternalSpanReplyError("invalid external span reply fields") from None
    if type(declared) is not int or declared != input_characters:
        raise ExternalSpanReplyError("external reply does not match the requested input")
    if type(span_values) not in (tuple, list) or type(coverage_values) not in (tuple, list):
        raise ExternalSpanReplyError("external span ranges require a native tuple or list")
    spans, coverage = len(span_values), len(coverage_values)
    if spans > selected.max_spans or coverage > selected.max_coverage_ranges:
        raise ExternalSpanLimitError("external span range count limit exceeded")
    width = max(1, coverage.bit_length())
    # Conservative bound for copying, scalar checks, coverage sorting/merging and lookups.
    if 1 + 8 * (spans + coverage) + 2 * coverage * width + spans * width > selected.max_work:
        raise ExternalSpanLimitError("external span validation work limit exceeded")
    span_snapshot, coverage_snapshot = span_values[:spans], coverage_values[:coverage]
    if len(span_snapshot) != spans or len(coverage_snapshot) != coverage:
        raise ExternalSpanReplyError("external span ranges changed during inspection")
    return ExternalSpanReply(input_characters, span_snapshot, coverage_snapshot, complete, digest)
