# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Real candidate groups close together without retaining callbacks or native input."""

import asyncio
import gc
import os
import threading
import weakref
from types import SimpleNamespace

import pytest

from quarry.cache import keys
from quarry.cache.async_cache import AsyncFunctionCache, retain_scan, scan_scope
from quarry.config import QuarryConfig
from quarry.engine.local_work import local_work_scope
from quarry.integrations import model_inference
from quarry.integrations.candidate_checks import LocalCandidatePattern
from quarry.integrations.candidate_providers import HostCandidateProvider, LocalCandidateProvider
from quarry.integrations.candidate_worker import CandidateWorkLimits
from quarry.integrations.model_usage import model_call_budget
from quarry.integrations.request_limits import is_owned_request_limit
from quarry.safety import local_match_work, timer
from quarry.types.candidates import (
    CandidateBatchIdentity,
    CandidateBatchReply,
    CandidateChannelReply,
)
from quarry.types.exceptions import (
    CandidateInvalidOutputError,
    CandidatePartialReadError,
    CandidateProviderError,
)
from quarry.types.messages import Messages
from tests.unit.model_fakes import NoWaitLimiter
from tests.unit.test_candidate_batch_runtime import DATA, entry, policy, response

IDENTITY = CandidateBatchIdentity("synthetic-candidates", "1", "heads")


class HostCallback:
    def __init__(self):
        self.calls = 0

    def __call__(self, request, *, timeout):
        self.calls += 1
        return response(request)


class LocalCheck:
    def __init__(self):
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        return True


def provider_for(kind):
    if kind == "host":
        callback = HostCallback()
        provider = HostCandidateProvider(IDENTITY, ("left", "right"), "0" * 64, callback)
    else:
        callback = LocalCheck()
        provider = local_provider(callback)
    return provider, callback


def local_provider(check):
    return LocalCandidateProvider(
        IDENTITY,
        ("left", "right"),
        (
            LocalCandidatePattern("left", "abc", ("synthetic",)),
            LocalCandidatePattern("right", "c"),
        ),
        checks={"synthetic": check},
        work=CandidateWorkLimits(timeout_seconds=10),
    )


def backend(tmp_path, provider, *, retention="process"):
    entries = {channel: entry(channel) for channel in ("left", "right")}
    for model in entries.values():
        model["spans"]["settings_digest"] = provider.settings_digest
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": entries, "model_cache": {"retention": retention}},
        tmp_path,
        require_rules=False,
    )
    return model_inference.ModelInferenceBackend(
        config,
        adapters={"registered": provider},
        rate_limiter=NoWaitLimiter(),
        span_models_to_load=("left", "right"),
    )


async def drain():
    for _ in range(4):
        await asyncio.sleep(0)
    gc.collect()


async def joined(cache):
    async def wait():
        while not any(getattr(value, "waiters", 0) >= 2 for value in cache.store.values()):
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["host", "local"])
@pytest.mark.parametrize("retention", ["process", "scan"])
async def test_all_aliases_share_owned_group_until_last_deferred_lease(
    tmp_path, monkeypatch, kind, retention
):
    cache = AsyncFunctionCache.expiring(
        max_entries=100, ttl_seconds=60, max_bytes=1 if retention == "process" else 1_000_000
    )
    monkeypatch.setattr(model_inference, "result_cache", cache)
    provider, callback = provider_for(kind)
    service = backend(tmp_path, provider, retention=retention)
    provider_ref, callback_ref = weakref.ref(provider), weakref.ref(callback)
    provider_identity = id(provider)
    try:
        messages = Messages.from_openai(DATA["messages"])
        with scan_scope(cache_policy=service.config.model_cache) as scope:
            first_lease, last_lease = retain_scan(), retain_scan()
            left = await service.model_span_messages("left", messages, 0.5)
            right = await service.model_span_messages("right", messages, 0.5)
            assert callback.calls == 1 and len(scope._replies) == 1
        assert len(scope._replies) == 1
        first_lease.close()
        assert len(scope._replies) == 1
        with scan_scope(scope):
            later = await service.model_span_messages("right", messages, 0.5)
        assert callback.calls == 1 and len(scope._replies) == 1
        last_lease.close()
        assert not scope._replies and not scope._revisions and not scope._release_callbacks
        assert not cache.store
        assert (
            left.items[0].value.start == 0
            and right.items[0].value.start == later.items[0].value.start == 2
        )
        assert provider_identity not in keys._kept
    finally:
        provider.close()
    del provider, callback, service
    await drain()
    assert provider_ref() is None and callback_ref() is None
    assert provider_identity not in keys._identities


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["host", "local"])
async def test_detached_process_cache_reply_does_not_retain_registration(
    tmp_path, monkeypatch, kind
):
    cache = AsyncFunctionCache.expiring(max_entries=100, ttl_seconds=60)
    monkeypatch.setattr(model_inference, "result_cache", cache)
    provider, callback = provider_for(kind)
    service = backend(tmp_path, provider)
    provider_ref, callback_ref = weakref.ref(provider), weakref.ref(callback)
    with scan_scope() as scope:
        result = await service.model_span_messages(
            "left", Messages.from_openai(DATA["messages"]), 0.5
        )
    assert len(result.items) == 1 and not scope._replies and len(cache.store) == 1
    provider.close()
    del provider, callback, service
    await drain()
    assert provider_ref() is None and callback_ref() is None
    assert len(cache.store) == 1


@pytest.mark.asyncio
async def test_partial_failure_closes_owned_entries_without_retaining_callback(
    tmp_path, monkeypatch
):
    cache = AsyncFunctionCache.expiring(max_entries=100, ttl_seconds=60)
    monkeypatch.setattr(model_inference, "result_cache", cache)

    class Partial:
        def __call__(self, request, *, timeout):
            return CandidateBatchReply(
                request.digest,
                tuple(
                    CandidateChannelReply(channel, (), (), False) for channel in request.channels
                ),
            )

    callback = Partial()
    callback_ref = weakref.ref(callback)
    provider = HostCandidateProvider(IDENTITY, ("left", "right"), "0" * 64, callback)
    service = backend(tmp_path, provider)
    with scan_scope() as scope:
        for alias in ("left", "right"):
            with pytest.raises(CandidatePartialReadError):
                await service.model_span_messages(
                    alias, Messages.from_openai(DATA["messages"]), 0.5
                )
        assert not scope._replies and not cache.store
    provider.close()
    del provider, callback, service
    await drain()
    assert callback_ref() is None and not scope._replies and not cache.store


@pytest.mark.asyncio
async def test_real_closed_starter_cpu_owner_allows_one_viable_alias_retry(tmp_path, monkeypatch):
    cache = AsyncFunctionCache.expiring(max_entries=100, ttl_seconds=60)
    monkeypatch.setattr(model_inference, "result_cache", cache)
    started, release = threading.Event(), threading.Event()
    calls = []

    def check(text):
        calls.append(text)
        if len(calls) == 1:
            started.set()
            assert release.wait(10)
        return True

    provider = local_provider(check)
    service = backend(tmp_path, provider)
    messages = Messages.from_openai(DATA["messages"])
    owners = [
        local_match_work.LocalMatchWorkOwner(
            local_match_work.LocalMatchWorkGroup(name, (name,), 1_000_000_000, "enforce")
        )
        for name in ("starter", "viable")
    ]
    tasks = []
    try:
        with scan_scope(), model_call_budget(1) as budget:
            with local_work_scope(owners[0]):
                tasks.append(
                    asyncio.create_task(service.model_span_messages("left", messages, 0.5))
                )
            assert await asyncio.to_thread(started.wait, 5)
            with local_work_scope(owners[1]):
                tasks.append(
                    asyncio.create_task(service.model_span_messages("right", messages, 0.5))
                )
            await joined(cache)
            owners[0].close()
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
            assert type(results[0]) is local_match_work.LocalMatchWorkError
            assert is_owned_request_limit(results[0])
            assert results[1].items[0].value.start == 2 and len(calls) == 2
            assert budget.calls_by_alias == {}
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        for owner in owners:
            owner.close()
        provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["scan", "rule"])
async def test_real_native_deadline_preserves_viable_alias_and_physical_work_count(
    tmp_path, monkeypatch, kind
):
    cache = AsyncFunctionCache.expiring(max_entries=100, ttl_seconds=60)
    monkeypatch.setattr(model_inference, "result_cache", cache)
    now = [0.0]
    monkeypatch.setattr(timer, "time", SimpleNamespace(monotonic=lambda: now[0]))
    started, release = threading.Event(), threading.Event()
    calls = []

    def check(text):
        calls.append(text)
        if len(calls) == 1:
            started.set()
            assert release.wait(10)
        return True

    provider = local_provider(check)
    service = backend(tmp_path, provider)
    messages = Messages.from_openai(DATA["messages"])
    tasks = []
    try:
        with scan_scope(), model_call_budget(1) as budget:
            with (
                timer.scan_deadline(10 if kind == "scan" else 100),
                timer.evaluation_deadline(10, "synthetic starter"),
            ):
                tasks.append(
                    asyncio.create_task(service.model_span_messages("left", messages, 0.5))
                )
            assert await asyncio.to_thread(started.wait, 5)
            with timer.scan_deadline(100), timer.evaluation_deadline(100, "synthetic viable"):
                tasks.append(
                    asyncio.create_task(service.model_span_messages("right", messages, 0.5))
                )
            await joined(cache)
            now[0] = 11
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
            assert type(results[0]) is timer.QuarryTimeoutError
            assert results[1].items[0].value.start == 2
            assert len(calls) == (2 if kind == "scan" else 1)
            assert budget.calls_by_alias == {}
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["deadline", "local_work"])
async def test_host_control_exception_without_native_denial_never_retries_group(
    tmp_path, monkeypatch, kind
):
    cache = AsyncFunctionCache.expiring(max_entries=100, ttl_seconds=60)
    monkeypatch.setattr(model_inference, "result_cache", cache)
    started, release = threading.Event(), threading.Event()
    calls = []

    def check(text):
        calls.append(text)
        started.set()
        assert release.wait(10)
        if kind == "deadline":
            raise timer.QuarryTimeoutError("synthetic host control failure")
        raise local_match_work.LocalMatchWorkError("synthetic host control failure")

    provider = local_provider(check)
    service = backend(tmp_path, provider)
    messages = Messages.from_openai(DATA["messages"])
    tasks = []
    try:
        with scan_scope():
            tasks.append(asyncio.create_task(service.model_span_messages("left", messages, 0.5)))
            assert await asyncio.to_thread(started.wait, 5)
            tasks.append(asyncio.create_task(service.model_span_messages("right", messages, 0.5)))
            await joined(cache)
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
            expected = (
                timer.QuarryTimeoutError
                if kind == "deadline"
                else local_match_work.LocalMatchWorkError
            )
            assert all(
                type(result) is expected and not is_owned_request_limit(result)
                for result in results
            )
            assert len(calls) == 1
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        provider.close()


@pytest.mark.asyncio
async def test_cancelled_group_is_replaced_on_next_readers_actual_host_loop(tmp_path):
    started, finished = asyncio.Event(), asyncio.Event()
    loops = []

    async def callback(request, *, timeout):
        loops.append(asyncio.get_running_loop())
        if len(loops) == 1:
            started.set()
            try:
                await asyncio.Future()
            finally:
                finished.set()
        return response(request)

    compiled = policy(tmp_path, callback)
    task = asyncio.create_task(compiled.scan_async(DATA))
    try:
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set() and loops[0] is asyncio.get_running_loop()
        result = await asyncio.wait_for(
            asyncio.to_thread(lambda: asyncio.run(compiled.scan_async(DATA))), 5
        )
        assert result.is_complete and len(loops) == 2 and loops[1] is not loops[0]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_stale_worker_process_guard_refuses_before_inherited_mutex(tmp_path):
    provider, _ = provider_for("local")
    owner = provider._worker
    original_pid = owner._pid
    owner._pid = os.getpid() + 1
    owner._lock.acquire()
    task = asyncio.create_task(asyncio.to_thread(lambda: asyncio.run(owner.run(lambda job: True))))
    try:
        with pytest.raises(CandidateProviderError):
            await asyncio.wait_for(asyncio.shield(task), 5)
    finally:
        owner._lock.release()
        owner._pid = original_pid
        await asyncio.gather(task, return_exceptions=True)
        provider.close()


@pytest.mark.asyncio
async def test_last_alias_cancellation_closes_whole_pending_group(tmp_path, monkeypatch):
    cache = AsyncFunctionCache.expiring(max_entries=100, ttl_seconds=60)
    monkeypatch.setattr(model_inference, "result_cache", cache)
    started, finished = asyncio.Event(), asyncio.Event()
    calls = []

    async def callback(request, *, timeout):
        calls.append(True)
        started.set()
        try:
            await asyncio.Future()
        finally:
            await asyncio.sleep(0)
            finished.set()

    provider = HostCandidateProvider(IDENTITY, ("left", "right"), "0" * 64, callback)
    service = backend(tmp_path, provider, retention="scan")
    messages = Messages.from_openai(DATA["messages"])
    tasks = []
    try:
        with scan_scope(cache_policy=service.config.model_cache) as scope:
            tasks.append(asyncio.create_task(service.model_span_messages("left", messages, 0.5)))
            await asyncio.wait_for(started.wait(), 5)
            tasks.append(asyncio.create_task(service.model_span_messages("right", messages, 0.5)))
            await joined(cache)
            tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
            assert not finished.is_set() and len(cache.store) == 1
            tasks[1].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[1]
            assert finished.is_set() and not cache.store and not scope._replies
        assert not scope._release_callbacks and len(calls) == 1
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        provider.close()


@pytest.mark.asyncio
async def test_rejected_foreign_packet_is_not_inspected_or_retained(tmp_path):
    references = []

    class Packet:
        @property
        def request_digest(self):
            pytest.fail("foreign host properties must not be inspected")

    def callback(request, *, timeout):
        packet = Packet()
        references.append(weakref.ref(packet))
        return packet

    provider = HostCandidateProvider(IDENTITY, ("left", "right"), "0" * 64, callback)
    service = backend(tmp_path, provider)
    try:
        with pytest.raises(CandidateInvalidOutputError):
            await service.model_span_messages("left", Messages.from_openai(DATA["messages"]), 0.5)
        await drain()
        assert len(references) == 1 and references[0]() is None
    finally:
        provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["caller", "configured"])
async def test_native_request_timeout_retries_only_when_starter_owns_the_shorter_cap(
    tmp_path, monkeypatch, boundary
):
    from quarry.integrations import candidate_worker
    from quarry.types.candidates import CandidateLimitError
    from quarry.types.exceptions import LLMError

    cache = AsyncFunctionCache.expiring(max_entries=100, ttl_seconds=60)
    monkeypatch.setattr(model_inference, "result_cache", cache)
    now = [0.0]
    monkeypatch.setattr(candidate_worker, "monotonic", lambda: now[0])
    started, release = threading.Event(), threading.Event()
    calls = []

    def check(text):
        calls.append(text)
        if len(calls) == 1:
            started.set()
            assert release.wait(10)
        return True

    provider = local_provider(check)  # A real, finite registered ten-second work cap.
    service = backend(tmp_path, provider)
    messages = Messages.from_openai(DATA["messages"])
    tasks = []
    try:
        with scan_scope(), model_call_budget(1) as budget:
            tasks.append(
                asyncio.create_task(
                    service.model_span_messages(
                        "left", messages, 0.5, timeout=1 if boundary == "caller" else 10
                    )
                )
            )
            assert await asyncio.to_thread(started.wait, 5)
            tasks.append(
                asyncio.create_task(
                    service.model_span_messages(
                        "right", messages, 0.5, timeout=5 if boundary == "caller" else 20
                    )
                )
            )
            await joined(cache)
            now[0] = 2 if boundary == "caller" else 11
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
            if boundary == "caller":
                assert isinstance(results[0], LLMError) and results[0].details["code"] == "timeout"
                assert is_owned_request_limit(results[0])
                assert results[1].items[0].value.start == 2 and len(calls) == 2
            else:
                assert all(
                    type(result) is CandidateLimitError and not is_owned_request_limit(result)
                    for result in results
                )
                assert len(calls) == 1
            assert budget.calls_by_alias == {}
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        provider.close()
