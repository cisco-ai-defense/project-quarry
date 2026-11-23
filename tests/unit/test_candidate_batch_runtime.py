# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Real policy reads admit complete candidate groups and bind each reader locally."""

import asyncio
from dataclasses import replace

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.integrations.candidate_providers import HostCandidateProvider, LocalCandidateProvider
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.safety.limits import SafetyLimits
from quarry.types.candidates import (
    CandidateBatchIdentity,
    CandidateBatchReply,
    CandidateBatchRequest,
    CandidateChannelReply,
    CandidateCoverage,
    CandidateInput,
    CandidatePart,
    CandidateRange,
)
from quarry.types.exceptions import ConfigurationError
from tests.unit.model_fakes import NoWaitLimiter

DATA = {"messages": [{"role": "user", "content": "abcdef"}]}
SOURCE = (
    'rule "Read" condition: any s in model("left").spans(messages, 0.5) where true '
    'collect: [s for s in model("right").spans(messages, 0.5) if true]'
)


def entry(channel, **extra):
    return {
        "adapter": "registered",
        "model": "synthetic-candidates",
        "revision": "1",
        "category": "SYNTHETIC_" + channel.upper(),
        "spans": {
            "backend": "candidates",
            "batch": "heads",
            "channel": channel,
            "settings_digest": "0" * 64,
        },
        **extra,
    }


def response(request, *, invalid=False, partial=False, excessive=False):
    coverage = tuple(
        CandidateCoverage(i, 0, len(p.text)) for i, p in enumerate(request.input.parts)
    )
    return CandidateBatchReply(
        request.digest,
        tuple(
            CandidateChannelReply(
                channel,
                (
                    CandidateRange(
                        0,
                        0 if channel == "left" else 2,
                        100 if invalid and channel == "right" else 3,
                        0.1 if invalid else 0.9,
                    ),
                )
                * (2 if excessive else 1),
                coverage,
                not (partial and channel == "right"),
            )
            for channel in request.channels
        ),
    )


def policy(tmp_path, callback, *, source=SOURCE, restored=False, entries=None, provider=None):
    provider = provider or HostCandidateProvider(
        CandidateBatchIdentity("synthetic-candidates", "1", "heads"),
        ("left", "right"),
        "0" * 64,
        callback,
    )
    entries = entries or {channel: entry(channel) for channel in ("left", "right")}
    if type(provider) in (HostCandidateProvider, LocalCandidateProvider):
        for model in entries.values():
            model["spans"]["settings_digest"] = provider.settings_digest
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": entries,
        },
        tmp_path,
        require_rules=False,
    )
    config = replace(config, safety_limits=SafetyLimits(rule_timeout=None))
    backend = ModelInferenceBackend(
        config, adapters={"registered": provider}, rate_limiter=NoWaitLimiter()
    )
    compiled = compile_policy(source, runtime_config=config, model_inference_backend=backend)
    if restored:
        wire = compiled.to_bytes()
        compiled = load_compiled_policy(
            wire, runtime_config=config, model_inference_backend=backend
        )
        assert compiled.to_bytes() == wire
    return compiled


@pytest.mark.parametrize("restored", [False, True])
def test_aliases_share_one_complete_batch_and_keep_distinct_native_outputs(tmp_path, restored):
    requests = []

    async def callback(request, *, timeout):
        requests.append(request)
        return response(request)

    compiled = policy(tmp_path, callback, restored=restored)
    result = compiled.scan(DATA)
    assert result.is_complete, result.evaluation_errors
    assert len(requests) == 1
    assert requests[0].channels == ("left", "right")
    assert result.rule_matches == ["Read"]
    collection = result.outputs["Read"]
    assert collection.category == "SYNTHETIC_RIGHT"
    span = collection.items[0].value
    assert (span.message_index, span.start, span.end, span.precision) == (0, 2, 3, "exact")
    assert result.model_usage["calls"] == 1
    assert result.model_usage["records"][0]["operation"] == "spans"


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize(
    "condition",
    [
        'any s in model("left").spans(messages, 0.5) where true',
        'not any s in model("left").spans(messages, 0.5) where true',
    ],
)
@pytest.mark.parametrize(
    "partial,category", [(True, "model_partial_read"), (False, "model_invalid_output")]
)
def test_unselected_channel_blocks_whole_group_before_threshold_or_negation(
    tmp_path, restored, condition, partial, category
):
    calls = []

    def callback(request, *, timeout):
        calls.append(request.digest)
        return response(request, invalid=not partial, partial=partial)

    compiled = policy(
        tmp_path, callback, source='rule "Read" condition: ' + condition, restored=restored
    )
    for _ in range(2):
        result = compiled.scan(DATA)
        assert not result.is_complete and not result.rule_matches and not result.outputs
        assert result.evaluation_error_categories["Read"] == category
    assert len(calls) == 2, (
        "invalid or partial groups must not become reusable authoritative replies"
    )


def test_candidate_count_limit_precedes_low_score_filter(tmp_path):
    entries = {"left": entry("left")}
    entries["left"]["spans"]["limits"] = {"max_ranges_per_channel": 1}
    compiled = policy(
        tmp_path,
        lambda request, **kwargs: response(request, excessive=True),
        entries=entries,
        source='rule "Read" condition: any s in model("left").spans(messages, 1.0) where true',
    )
    result = compiled.scan(DATA)
    assert (
        not result.is_complete and result.evaluation_error_categories["Read"] == "evaluation_limit"
    )


@pytest.mark.parametrize("restored", [False, True])
def test_unselected_candidate_branch_does_not_dispatch(tmp_path, restored):
    def callback(*args, **kwargs):
        raise AssertionError("unselected candidates must not dispatch")

    compiled = policy(
        tmp_path,
        callback,
        restored=restored,
        source='rule "Read" condition: true or any s in model("left").spans(messages, 0.5) where true',
    )
    result = compiled.scan(DATA)
    assert result.is_complete and result.rule_matches == ["Read"]
    assert result.model_usage["calls"] == 0


@pytest.mark.parametrize("field,value", [("model", "different"), ("revision", "2")])
def test_registration_identity_is_checked_at_policy_binding(tmp_path, field, value):
    entries = {"left": entry("left", **{field: value})}
    with pytest.raises(ConfigurationError, match="capability identity"):
        policy(
            tmp_path,
            lambda *args, **kwargs: None,
            entries=entries,
            source='rule "Read" condition: any s in model("left").spans(messages, 0.5) where true',
        )


def test_generic_adapter_cannot_claim_native_candidate_capability(tmp_path):
    class Impostor:
        async def candidate_batch(self, *args, **kwargs):
            raise AssertionError("generic adapter must never be called")

    with pytest.raises(ConfigurationError, match="explicitly registered"):
        policy(tmp_path, None, provider=Impostor())


def test_host_cannot_rewrite_the_expected_request_through_its_snapshot(tmp_path):
    def callback(request, *, timeout):
        foreign = CandidateBatchRequest(
            request.provider,
            CandidateInput(
                tuple(CandidatePart(p.kind, p.path, "UVWXYZ") for p in request.input.parts)
            ),
            request.channels,
            request.settings_digest,
        )
        object.__setattr__(request, "input", foreign.input)
        object.__setattr__(request, "digest", foreign.digest)
        return response(request)

    result = policy(tmp_path, callback).scan(DATA)
    assert not result.is_complete and not result.outputs
    assert result.evaluation_error_categories["Read"] == "model_invalid_output"


def test_aliases_with_different_admission_limits_do_not_share_acceptance(tmp_path):
    requests = []

    def callback(request, *, timeout):
        requests.append(request.digest)
        return response(request)

    entries = {channel: entry(channel) for channel in ("left", "right")}
    entries["right"]["spans"]["limits"] = {"max_work": 5000}
    result = policy(tmp_path, callback, entries=entries).scan(DATA)
    assert result.is_complete, result.evaluation_errors
    assert len(requests) == 2 and result.model_usage["calls"] == 2


@pytest.mark.asyncio
async def test_pending_shared_batch_survives_one_cancelled_reader(tmp_path, monkeypatch):
    from quarry.cache.async_cache import AsyncFunctionCache
    from quarry.integrations import model_inference

    cache = AsyncFunctionCache.expiring(max_entries=100, ttl_seconds=60)
    monkeypatch.setattr(model_inference, "result_cache", cache)
    started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def callback(request, *, timeout):
        calls.append(request.digest)
        started.set()
        try:
            await release.wait()
            return response(request)
        finally:
            finished.set()

    compiled = policy(tmp_path, callback)
    first = asyncio.create_task(compiled.scan_async(DATA))
    second = None
    try:
        await asyncio.wait_for(started.wait(), 3)
        second = asyncio.create_task(compiled.scan_async(DATA))

        async def joined():
            while not any(getattr(value, "waiters", 0) >= 2 for value in cache.store.values()):
                await asyncio.sleep(0)

        await asyncio.wait_for(joined(), 3)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not finished.is_set()
        release.set()
        result = await asyncio.wait_for(second, 3)
        assert result.is_complete and len(calls) == 1
    finally:
        release.set()
        await asyncio.gather(
            first, *([second] if second is not None else []), return_exceptions=True
        )


@pytest.mark.asyncio
async def test_last_candidate_reader_cancellation_awaits_host_cleanup(tmp_path):
    started, finished = asyncio.Event(), asyncio.Event()

    async def callback(request, *, timeout):
        started.set()
        try:
            await asyncio.Future()
        finally:
            await asyncio.sleep(0)
            finished.set()

    compiled = policy(tmp_path, callback)
    task = asyncio.create_task(compiled.scan_async(DATA))
    try:
        await asyncio.wait_for(started.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
