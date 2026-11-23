# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Caller-authored RE2 candidates and named, synchronous bounded checks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from quarry.engine.pattern_checks import check_names, run_pattern_check
from quarry.safety.local_match_work import LocalMatchWorkError
from quarry.safety.regex_backend import RE2_REGEX
from quarry.safety.timer import QuarryTimeoutError
from quarry.types.candidates import (
    CandidateBatchReply,
    CandidateChannelReply,
    CandidateCoverage,
    CandidateLimitError,
    CandidateRange,
    CandidateReplyError,
    _name,
    validate_candidate_batch_reply,
)


@dataclass(frozen=True, slots=True)
class CandidateChannelChecks:
    channel: str
    checks: tuple[str, ...]

    def __post_init__(self):
        _name(self.channel)
        object.__setattr__(self, "checks", check_names(self.checks))


@dataclass(frozen=True, slots=True)
class LocalCandidatePattern:
    """One caller-authored RE2 expression, with no automatic catalog or checks."""

    channel: str
    pattern: str = field(repr=False)
    checks: tuple[str, ...] = ()
    score: float = 1.0
    flags: int = 0

    def __post_init__(self):
        _name(self.channel)
        if type(self.pattern) is not str or not 1 <= len(self.pattern) <= 4096:
            raise CandidateLimitError("candidate patterns require 1..4096 characters")
        if type(self.flags) is not int or self.flags & ~(re.IGNORECASE | re.DOTALL):
            raise CandidateReplyError(
                "candidate patterns support only RE2 ignore-case and dot-all flags"
            )
        checked = CandidateRange(0, 0, 1, self.score)
        object.__setattr__(self, "score", checked.score)
        object.__setattr__(self, "checks", check_names(self.checks))


def filter_candidate_reply(request, reply, assignments, checks, job):
    """All ranges have been counted and validated before the first check runs."""
    job.checkpoint(sum(len(channel.ranges) for channel in reply.channels))
    channels = []
    for channel in reply.channels:
        ranges = []
        plans = assignments.get(channel.channel)
        for ordinal, value in enumerate(channel.ranges):
            job.checkpoint()
            names = plans[ordinal] if plans is not None else ()
            accepted = True
            for name in names:
                job.checkpoint(1 + (value.end - value.start + 255) // 256)
                accepted = run_pattern_check(
                    checks[name],
                    name,
                    request.input.parts[value.part].text[value.start : value.end],
                    control_errors=(QuarryTimeoutError, LocalMatchWorkError),
                )
                job.checkpoint()
                if not accepted:
                    break
            if accepted:
                ranges.append(value)
        channels.append(
            CandidateChannelReply(
                channel.channel, tuple(ranges), channel.coverage, channel.complete
            )
        )
    return CandidateBatchReply(reply.request_digest, tuple(channels))


def generate_local_candidates(request, patterns, checks, limits, work_limits, job):
    empty = CandidateBatchReply(
        request.digest,
        tuple(CandidateChannelReply(channel, (), (), False) for channel in request.channels),
    )
    validate_candidate_batch_reply(empty, request, limits, check_deadline=job.checkpoint)
    ranges = {channel: [] for channel in request.channels}
    assignments = {channel: [] for channel in request.channels}
    maximum = min(limits.max_ranges_per_channel, work_limits.max_candidates_per_channel)
    for pattern in patterns:
        for part, source in enumerate(request.input.parts):
            job.checkpoint(1 + (len(source.text) + 255) // 256)
            # RE2 alone supplies production enumeration; no stdlib regex fallback.
            for low, high in RE2_REGEX.iter_match_spans(
                pattern.pattern, source.text, pattern.flags
            ):
                job.checkpoint()
                if len(ranges[pattern.channel]) >= maximum:
                    raise CandidateLimitError("local candidate count limit exceeded before checks")
                if low == high:
                    raise CandidateReplyError(
                        "local candidate patterns must produce nonempty ranges"
                    )
                ranges[pattern.channel].append(CandidateRange(part, low, high, pattern.score))
                assignments[pattern.channel].append(pattern.checks)
    coverage = tuple(
        CandidateCoverage(i, 0, len(p.text)) for i, p in enumerate(request.input.parts)
    )
    reply = CandidateBatchReply(
        request.digest,
        tuple(
            CandidateChannelReply(channel, tuple(ranges[channel]), coverage, True)
            for channel in request.channels
        ),
    )
    reply = validate_candidate_batch_reply(reply, request, limits, check_deadline=job.checkpoint)
    return filter_candidate_reply(request, reply, assignments, checks, job)
