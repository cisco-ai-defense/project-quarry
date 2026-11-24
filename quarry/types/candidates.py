# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Detached, bounded part geometry for explicitly registered candidate batches.

These values establish request identity and reply geometry, not policy authority.
Readers must validate the complete group and bind ranges to their own native parts
before filtering, checking candidates, accepting a response or returning a verdict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256

from quarry.types.external_spans import (
    ExternalReadRange,
    ExternalSpanRange,
    ExternalSpanReplyError,
)

MAX_CANDIDATE_PARTS = 4096
MAX_CANDIDATE_CHANNELS = 16
MAX_CANDIDATE_CHARACTERS = 4_000_000
MAX_CANDIDATE_RANGES = 4096


class CandidateReplyError(ValueError):
    """A candidate reply does not describe its exact requested parts and channels."""


class CandidateLimitError(CandidateReplyError):
    """Candidate input, reply inspection or work exceeds an explicit finite bound."""


def _name(value: object) -> str:
    if (
        type(value) is not str
        or not 1 <= len(value) <= 64
        or not value.isascii()
        or not value[0].isalpha()
        or not value.replace("_", "").replace("-", "").replace(".", "").isalnum()
    ):
        raise CandidateReplyError("candidate capability names must be bounded ASCII identifiers")
    return value


def _digest(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise CandidateReplyError("invalid candidate request digest")
    return value


def _sequence(value: object, maximum: int) -> tuple:
    if type(value) is not tuple and type(value) is not list:
        raise CandidateLimitError("candidate sequence requires a native tuple or list")
    if len(value) > maximum:
        raise CandidateLimitError("candidate sequence exceeds its native bounded contract")
    return tuple(value[: len(value)])


@dataclass(frozen=True, slots=True)
class CandidatePart:
    """One text part and its detached source address; text and paths stay out of repr."""

    kind: str
    path: tuple[str | int, ...] = field(repr=False)
    text: str = field(repr=False)

    def __post_init__(self):
        _name(self.kind)
        path = _sequence(self.path, 64)
        if any(
            not (type(item) is str and len(item) <= 256)
            and not (type(item) is int and 0 <= item <= 2**31 - 1)
            for item in path
        ):
            raise CandidateReplyError("invalid candidate source-part address")
        if type(self.text) is not str or len(self.text) > MAX_CANDIDATE_CHARACTERS:
            raise CandidateLimitError("candidate part text exceeds its character bound")
        object.__setattr__(self, "path", path)


def _feed(hasher, value: str) -> None:
    wire = value.encode("utf-8", errors="surrogatepass")
    hasher.update(len(wire).to_bytes(8, "big"))
    hasher.update(wire)


@dataclass(frozen=True, slots=True)
class CandidateInput:
    parts: tuple[CandidatePart, ...] = field(repr=False)
    digest: str = field(init=False)
    characters: int = field(init=False)

    def __post_init__(self):
        parts = _sequence(self.parts, MAX_CANDIDATE_PARTS)
        copied = []
        characters = 0
        hasher = sha256(b"quarry-candidate-parts-v1")
        hasher.update(len(parts).to_bytes(4, "big"))
        for part in parts:
            if type(part) is not CandidatePart:
                raise CandidateReplyError("invalid native candidate input part")
            part = CandidatePart(part.kind, part.path, part.text)
            characters += len(part.text)
            if characters > MAX_CANDIDATE_CHARACTERS:
                raise CandidateLimitError("candidate input exceeds its character bound")
            _feed(hasher, part.kind)
            hasher.update(len(part.path).to_bytes(4, "big"))
            for item in part.path:
                _feed(hasher, "s" + item if type(item) is str else "i" + str(item))
            _feed(hasher, part.text)
            copied.append(part)
        object.__setattr__(self, "parts", tuple(copied))
        object.__setattr__(self, "characters", characters)
        object.__setattr__(self, "digest", hasher.hexdigest())


@dataclass(frozen=True, slots=True)
class CandidateBatchIdentity:
    namespace: str
    version: str
    batch: str

    def __post_init__(self):
        for value in (self.namespace, self.batch):
            _name(value)
        if (
            type(self.version) is not str
            or not 1 <= len(self.version) <= 128
            or not self.version.isascii()
            or any(not 33 <= ord(char) <= 126 for char in self.version)
        ):
            raise CandidateReplyError("candidate versions require bounded visible ASCII text")


@dataclass(frozen=True, slots=True)
class CandidateBatchRequest:
    provider: CandidateBatchIdentity
    input: CandidateInput = field(repr=False)
    channels: tuple[str, ...]
    settings_digest: str
    digest: str = field(init=False)

    def __post_init__(self):
        if (
            type(self.provider) is not CandidateBatchIdentity
            or type(self.input) is not CandidateInput
        ):
            raise CandidateReplyError("invalid native candidate request")
        provider = CandidateBatchIdentity(
            self.provider.namespace, self.provider.version, self.provider.batch
        )
        source = CandidateInput(self.input.parts)
        channels = _sequence(self.channels, MAX_CANDIDATE_CHANNELS)
        if not channels or len(set(map(_name, channels))) != len(channels):
            raise CandidateReplyError("candidate channels must be nonempty and distinct")
        channels = tuple(sorted(channels))
        _digest(self.settings_digest)
        hasher = sha256(b"quarry-candidate-batch-v1")
        for value in (
            provider.namespace,
            provider.version,
            provider.batch,
            source.digest,
            self.settings_digest,
            *channels,
        ):
            _feed(hasher, value)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "input", source)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "digest", hasher.hexdigest())


def _part_number(value: object) -> None:
    if type(value) is not int or not 0 <= value < MAX_CANDIDATE_PARTS:
        raise CandidateReplyError("invalid candidate source-part number")


@dataclass(frozen=True, slots=True)
class CandidateRange:
    part: int
    start: int
    end: int
    score: float

    def __post_init__(self):
        _part_number(self.part)
        try:
            checked = ExternalSpanRange(self.start, self.end, self.score)
        except ExternalSpanReplyError:
            raise CandidateReplyError("invalid candidate range geometry or score") from None
        object.__setattr__(self, "score", checked.score)


@dataclass(frozen=True, slots=True)
class CandidateCoverage:
    part: int
    start: int
    end: int

    def __post_init__(self):
        _part_number(self.part)
        try:
            ExternalReadRange(self.start, self.end)
        except ExternalSpanReplyError:
            raise CandidateReplyError("invalid candidate coverage geometry") from None


@dataclass(frozen=True, slots=True)
class CandidateChannelReply:
    channel: str
    ranges: tuple[CandidateRange, ...]
    coverage: tuple[CandidateCoverage, ...]
    complete: bool

    def __post_init__(self):
        _name(self.channel)
        if type(self.complete) is not bool:
            raise CandidateReplyError("candidate completeness must be a boolean")
        spans, coverage = [], []
        for value in _sequence(self.ranges, MAX_CANDIDATE_RANGES):
            if type(value) is not CandidateRange:
                raise CandidateReplyError("invalid native candidate range")
            spans.append(CandidateRange(value.part, value.start, value.end, value.score))
        for value in _sequence(self.coverage, MAX_CANDIDATE_RANGES):
            if type(value) is not CandidateCoverage:
                raise CandidateReplyError("invalid native candidate coverage")
            coverage.append(CandidateCoverage(value.part, value.start, value.end))
        object.__setattr__(self, "ranges", tuple(spans))
        object.__setattr__(self, "coverage", tuple(coverage))


@dataclass(frozen=True, slots=True)
class CandidateBatchReply:
    request_digest: str
    channels: tuple[CandidateChannelReply, ...]

    def __post_init__(self):
        _digest(self.request_digest)
        copied = []
        for value in _sequence(self.channels, MAX_CANDIDATE_CHANNELS):
            if type(value) is not CandidateChannelReply:
                raise CandidateReplyError("invalid native candidate channel reply")
            copied.append(
                CandidateChannelReply(value.channel, value.ranges, value.coverage, value.complete)
            )
        if not copied or len({value.channel for value in copied}) != len(copied):
            raise CandidateReplyError("candidate reply channels must be nonempty and distinct")
        object.__setattr__(self, "channels", tuple(sorted(copied, key=lambda value: value.channel)))


@dataclass(frozen=True, slots=True)
class CandidateBatchLimits:
    max_input_characters: int = 1_000_000
    max_parts: int = MAX_CANDIDATE_PARTS
    max_ranges_per_channel: int = MAX_CANDIDATE_RANGES
    max_coverage_per_channel: int = MAX_CANDIDATE_RANGES
    max_work: int = 262_144

    def __post_init__(self):
        for value, low, high in (
            (self.max_input_characters, 0, MAX_CANDIDATE_CHARACTERS),
            (self.max_parts, 0, MAX_CANDIDATE_PARTS),
            (self.max_ranges_per_channel, 0, MAX_CANDIDATE_RANGES),
            (self.max_coverage_per_channel, 0, MAX_CANDIDATE_RANGES),
            (self.max_work, 1, 4_000_000),
        ):
            if type(value) is not int or not low <= value <= high:
                raise CandidateLimitError("invalid candidate reader limit")


def validate_candidate_batch_reply(
    raw: CandidateBatchReply,
    request: CandidateBatchRequest,
    limits: CandidateBatchLimits,
    *,
    check_deadline: Callable[[], None],
) -> CandidateBatchReply:
    """Validate every channel before any caller threshold/check or cache admission.

    Partial replies remain partial. This function supplies no authority to a
    policy reader and retains no source binding, host callback or request object.
    """

    check_deadline()
    if type(raw) is not CandidateBatchReply or type(request) is not CandidateBatchRequest:
        raise CandidateReplyError("invalid native candidate batch")
    if type(limits) is not CandidateBatchLimits:
        raise CandidateLimitError("invalid native candidate reader limits")
    limits = CandidateBatchLimits(
        limits.max_input_characters,
        limits.max_parts,
        limits.max_ranges_per_channel,
        limits.max_coverage_per_channel,
        limits.max_work,
    )
    _digest(raw.request_digest)
    _digest(request.digest)
    remaining = limits.max_work

    def spend(units):
        nonlocal remaining
        if units > remaining:
            raise CandidateLimitError("candidate inspection work limit exceeded")
        before, remaining = remaining, remaining - units
        if before // 256 != remaining // 256:
            check_deadline()

    # Check native lengths and charge inspection before cloning or hashing text.
    # A low reader allowance must refuse work before a host-controlled payload
    # can make identity reconstruction consume its larger constructor allowance.
    if type(request.input) is not CandidateInput:
        raise CandidateReplyError("invalid native candidate input")
    parts = _sequence(request.input.parts, limits.max_parts)
    requested_channels = _sequence(request.channels, MAX_CANDIDATE_CHANNELS)
    spend(len(parts) + len(requested_channels) + 1)
    characters = 0
    for part in parts:
        if type(part) is not CandidatePart or type(part.text) is not str:
            raise CandidateReplyError("invalid native candidate input part")
        characters += len(part.text)
        if characters > limits.max_input_characters:
            raise CandidateLimitError("candidate input exceeds its reader limits")
        spend((len(part.text) + 255) // 256)
        path = _sequence(part.path, 64)
        spend(len(path))
        for item in path:
            if type(item) is str and len(item) <= 256:
                spend((len(item) + 255) // 256)
            elif type(item) is int and 0 <= item <= 2**31 - 1:
                spend(1)
            else:
                raise CandidateReplyError("invalid candidate source-part address")
    check_deadline()
    checked_request = CandidateBatchRequest(
        request.provider, request.input, requested_channels, request.settings_digest
    )
    if checked_request.digest != request.digest or raw.request_digest != request.digest:
        raise CandidateReplyError("candidate reply belongs to a different request")
    source = checked_request.input
    check_deadline()
    channels = _sequence(raw.channels, MAX_CANDIDATE_CHANNELS)
    if len(channels) != len(checked_request.channels):
        raise CandidateReplyError("candidate reply does not cover exactly the requested channels")
    copied = []
    names = set()
    for value in channels:
        spend(1)
        if type(value) is not CandidateChannelReply:
            raise CandidateReplyError("invalid native candidate channel")
        spans = _sequence(value.ranges, limits.max_ranges_per_channel)
        coverage = _sequence(value.coverage, limits.max_coverage_per_channel)
        spend(len(spans) + len(coverage) * (len(coverage).bit_length() + 1))
        channel = CandidateChannelReply(value.channel, spans, coverage, value.complete)
        if channel.channel not in checked_request.channels or channel.channel in names:
            raise CandidateReplyError("candidate reply contains an unexpected or repeated channel")
        names.add(channel.channel)
        by_part: dict[int, list[tuple[int, int]]] = {}
        for item in channel.coverage:
            if item.part >= len(source.parts) or item.end > len(source.parts[item.part].text):
                raise CandidateReplyError("candidate coverage lies outside its requested part")
            if item.start != item.end:
                by_part.setdefault(item.part, []).append((item.start, item.end))
        unions = {}
        for part, intervals in by_part.items():
            union = []
            for start, end in sorted(intervals):
                if union and start <= union[-1][1]:
                    union[-1] = (union[-1][0], max(end, union[-1][1]))
                else:
                    union.append((start, end))
            unions[part] = tuple(union)
        for item in channel.ranges:
            if item.part >= len(source.parts) or item.end > len(source.parts[item.part].text):
                raise CandidateReplyError("candidate range lies outside its requested part")
            intervals = unions.get(item.part, ())
            spend(len(intervals) + 1)
            if not any(start <= item.start and item.end <= end for start, end in intervals):
                raise CandidateReplyError("candidate range lies outside declared coverage")
        if channel.complete:
            spend(len(source.parts))
            for part, item in enumerate(source.parts):
                if item.text and unions.get(part) != ((0, len(item.text)),):
                    raise CandidateReplyError("complete candidate channel has incomplete coverage")
        copied.append(channel)
    check_deadline()
    return CandidateBatchReply(checked_request.digest, tuple(copied))
