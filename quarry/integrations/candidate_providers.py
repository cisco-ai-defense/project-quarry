# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Explicit host candidate capabilities; no discovery or built-in rule catalog."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from quarry.cache.keys import input_key
from quarry.engine.pattern_checks import copy_pattern_checks
from quarry.integrations.candidate_checks import (
    CandidateChannelChecks,
    LocalCandidatePattern,
    filter_candidate_reply,
    generate_local_candidates,
)
from quarry.integrations.candidate_worker import CandidateWorkerOwner, CandidateWorkLimits
from quarry.types.candidates import (
    MAX_CANDIDATE_CHANNELS,
    CandidateBatchIdentity,
    CandidateBatchRequest,
    CandidateLimitError,
    CandidateReplyError,
    _digest,
    _name,
    _sequence,
)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class HostCandidateProvider:
    """A caller-registered batch with a fixed identity, channel set and settings.

    The callback receives only the detached request and the remaining timeout.
    Changing its implementation requires a new version/settings digest. Register
    the same provider instance for aliases that may share one complete reply.
    """

    identity: CandidateBatchIdentity
    channels: tuple[str, ...]
    settings_digest: str
    callback: Callable = field(repr=False, compare=False)
    channel_checks: tuple[CandidateChannelChecks, ...] = ()
    checks: Mapping[str, Callable] | None = field(default=None, repr=False, compare=False)
    work: CandidateWorkLimits = field(default_factory=CandidateWorkLimits)
    _worker: CandidateWorkerOwner = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if type(self.identity) is not CandidateBatchIdentity:
            raise CandidateReplyError("candidate provider requires an explicit native identity")
        identity = CandidateBatchIdentity(
            self.identity.namespace, self.identity.version, self.identity.batch
        )
        channels = tuple(sorted(map(_name, _sequence(self.channels, MAX_CANDIDATE_CHANNELS))))
        if not channels or len(set(channels)) != len(channels):
            raise CandidateReplyError("candidate provider channels must be nonempty and distinct")
        _digest(self.settings_digest)
        if not callable(self.callback):
            raise CandidateReplyError("candidate provider requires a caller-registered callback")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "channels", channels)
        checks = copy_pattern_checks(self.checks)
        plans = []
        for plan in _sequence(self.channel_checks, MAX_CANDIDATE_CHANNELS):
            if type(plan) is not CandidateChannelChecks:
                raise CandidateReplyError("candidate channel checks require a native declaration")
            plan = CandidateChannelChecks(plan.channel, plan.checks)
            if plan.channel not in channels or any(name not in checks for name in plan.checks):
                raise CandidateReplyError("candidate channel check is unregistered")
            plans.append(plan)
        if len({p.channel for p in plans}) != len(plans):
            raise CandidateReplyError("candidate channel check declarations must be distinct")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "channel_checks", tuple(sorted(plans, key=lambda p: p.channel)))
        worker = CandidateWorkerOwner(self.work)
        object.__setattr__(self, "work", worker.limits)
        object.__setattr__(self, "_worker", worker)
        # The effective settings bind every declared check and execution limit.
        # Callback implementation changes additionally require the promised revision.
        object.__setattr__(
            self,
            "settings_digest",
            input_key(
                "host-candidate-definition-v1",
                identity,
                channels,
                self.settings_digest,
                self.channel_checks,
                tuple(sorted(checks)),
                worker.limits,
            ).hex(),
        )

    async def candidate_batch(self, _config, request: CandidateBatchRequest, *, timeout=None):
        if (
            type(request) is not CandidateBatchRequest
            or request.provider != self.identity
            or request.channels != self.channels
            or request.settings_digest != self.settings_digest
        ):
            raise CandidateReplyError("candidate request does not match its registered capability")
        reply = self.callback(request, timeout=timeout)
        return await reply if inspect.isawaitable(reply) else reply

    async def filter_reply(self, request, reply, *, timeout=None):
        if any(
            len(channel.ranges) > self.work.max_candidates_per_channel for channel in reply.channels
        ):
            raise CandidateLimitError("host candidate count limit exceeded before checks")
        if not any(plan.checks for plan in self.channel_checks):
            return reply
        plans = {plan.channel: plan.checks for plan in self.channel_checks}
        assignments = {
            channel.channel: (plans[channel.channel],) * len(channel.ranges)
            for channel in reply.channels
            if channel.channel in plans
        }
        return await self._worker.run(
            lambda job: filter_candidate_reply(request, reply, assignments, self.checks, job),
            timeout=timeout,
        )

    def close(self):
        self._worker.close()


@dataclass(frozen=True, slots=True, weakref_slot=True)
class LocalCandidateProvider:
    """Caller-authored patterns and checks, executed under finite native ownership."""

    identity: CandidateBatchIdentity
    channels: tuple[str, ...]
    patterns: tuple[LocalCandidatePattern, ...] = field(repr=False)
    checks: Mapping[str, Callable] | None = field(default=None, repr=False, compare=False)
    work: CandidateWorkLimits = field(default_factory=CandidateWorkLimits)
    settings_digest: str = field(init=False)
    _worker: CandidateWorkerOwner = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if type(self.identity) is not CandidateBatchIdentity:
            raise CandidateReplyError("local candidates require an explicit native identity")
        identity = CandidateBatchIdentity(
            self.identity.namespace, self.identity.version, self.identity.batch
        )
        channels = tuple(sorted(map(_name, _sequence(self.channels, MAX_CANDIDATE_CHANNELS))))
        if not channels or len(set(channels)) != len(channels):
            raise CandidateReplyError("local candidate channels must be nonempty and distinct")
        checks = copy_pattern_checks(self.checks)
        patterns = []
        for pattern in _sequence(self.patterns, 64):
            if type(pattern) is not LocalCandidatePattern:
                raise CandidateReplyError("local candidates require native pattern declarations")
            pattern = LocalCandidatePattern(
                pattern.channel, pattern.pattern, pattern.checks, pattern.score, pattern.flags
            )
            if pattern.channel not in channels or any(
                name not in checks for name in pattern.checks
            ):
                raise CandidateReplyError(
                    "local candidate pattern channel or check is unregistered"
                )
            patterns.append(pattern)
        if not patterns:
            raise CandidateReplyError("local candidates require at least one explicit pattern")
        worker = CandidateWorkerOwner(self.work)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "patterns", tuple(patterns))
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "work", worker.limits)
        object.__setattr__(self, "_worker", worker)
        object.__setattr__(
            self,
            "settings_digest",
            input_key(
                "local-candidate-definition-v1",
                identity,
                channels,
                self.patterns,
                tuple(sorted(checks)),
                worker.limits,
            ).hex(),
        )

    async def candidate_batch(self, config, request, *, timeout=None):
        if (
            type(request) is not CandidateBatchRequest
            or request.provider != self.identity
            or request.channels != self.channels
            or request.settings_digest != self.settings_digest
        ):
            raise CandidateReplyError(
                "local candidate request does not match its registered capability"
            )
        return await self._worker.run(
            lambda job: generate_local_candidates(
                request, self.patterns, self.checks, config.spans.limits, self.work, job
            ),
            timeout=timeout,
        )

    def close(self):
        self._worker.close()


def require_candidate_provider(model, adapters):
    """Validate registration and saved identity without invoking host properties."""
    from quarry.config import CandidateSpanConfig
    from quarry.types.exceptions import ConfigurationError

    config = model.spans
    provider = adapters.get(model.adapter)
    if (
        type(provider) not in (HostCandidateProvider, LocalCandidateProvider)
        or type(config) is not CandidateSpanConfig
    ):
        raise ConfigurationError(
            "candidate spans require an explicitly registered native capability"
        )
    if (
        provider.identity != CandidateBatchIdentity(model.model, model.revision, config.batch)
        or config.channel not in provider.channels
        or config.settings_digest != provider.settings_digest
    ):
        raise ConfigurationError(
            "candidate model does not match its registered capability identity"
        )
    return provider


def registered_local_candidate_aliases(config, backend=None, adapters=None):
    """Attest native registrations without invoking host properties or callbacks."""
    from types import MethodType

    from quarry.config import CandidateSpanConfig
    from quarry.integrations.model_inference import ModelInferenceBackend

    if backend is None:
        adapters, models = adapters or {}, config.models
    elif type(backend) is ModelInferenceBackend:
        operation = backend.model_span_messages
        if (
            type(operation) is not MethodType
            or operation.__func__ is not ModelInferenceBackend.model_span_messages
        ):
            return frozenset()
        adapters, models = backend.adapters, backend.models
    else:
        return frozenset()
    aliases = []
    for alias, model in models.items():
        if (
            type(model.spans) is CandidateSpanConfig
            and type(adapters.get(model.adapter)) is LocalCandidateProvider
        ):
            require_candidate_provider(model, adapters)
            aliases.append(alias)
    return frozenset(aliases)
