# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Owned retention ends with active scan work, independently of process reuse."""

import asyncio
import gc
import weakref
from dataclasses import FrozenInstanceError, dataclass

import pytest

from quarry.cache.async_cache import AsyncFunctionCache, SlidingTTLCache, retain_scan, scan_scope
from quarry.config import ModelCacheConfig, QuarryConfig
from quarry.types.exceptions import ConfigurationError


@dataclass
class Reply:
    content: bytes


async def drain():
    for _ in range(4):
        await asyncio.sleep(0)


def bounded(**overrides):
    return AsyncFunctionCache.expiring(
        **{"max_entries": 8, "ttl_seconds": 60, "max_bytes": 32_000, **overrides}
    )


@pytest.mark.parametrize("retention", ["process", "scan"])
@pytest.mark.parametrize("categories", [[], ["token_counts"], ["model_responses", "probes"]])
def test_closed_immutable_retention_configuration(tmp_path, retention, categories):
    categories = list(categories)
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "model_cache": {
                "retention": retention,
                "reusable_categories": categories,
            },
        },
        tmp_path,
        require_rules=False,
    ).model_cache
    expected = tuple(categories)
    categories.append("local_transforms")
    assert config.reusable_categories == expected
    for category in ["model_responses", "token_counts", "local_transforms", "probes"]:
        assert config.retention_for(category) == ("process" if category in expected else retention)
    with pytest.raises(FrozenInstanceError):
        config.retention = "scan"


@pytest.mark.parametrize("value", [None, False, 1, "", "forever"])
def test_invalid_retention_is_a_configuration_error(value):
    with pytest.raises(ConfigurationError, match="retention"):
        ModelCacheConfig(retention=value)


@pytest.mark.parametrize(
    "value",
    [None, "token_counts", {}, ["unknown"], [1], ["token_counts", "token_counts"], ["probes"] * 5],
)
def test_invalid_reusable_categories_are_rejected(value):
    with pytest.raises(ConfigurationError, match="reusable_categories"):
        ModelCacheConfig(reusable_categories=value)


@pytest.mark.asyncio
async def test_scan_results_drop_without_gc_even_if_scope_token_is_kept():
    cache = bounded()
    references = []
    calls = 0

    async def provider():
        nonlocal calls
        calls += 1
        reply = Reply(b"synthetic")
        references.append(weakref.ref(reply))
        return reply

    enabled = gc.isenabled()
    gc.disable()
    try:
        with scan_scope() as token:
            first = await cache.call(b"k", provider, retention="scan")
            assert await cache.call(b"k", provider, retention="scan") is first
            del first
            assert len(cache.store) == 1
        await drain()
        assert calls == 1 and len(cache.store) == 0
        assert references[0]() is None
        assert token is not None
        with scan_scope():
            await cache.call(b"k", provider, retention="scan")
        await drain()
        assert calls == 2 and all(reference() is None for reference in references)
    finally:
        if enabled:
            gc.enable()


@pytest.mark.asyncio
async def test_default_process_cache_reuses_completed_results_across_scans():
    cache = bounded()
    calls = []

    async def provider():
        calls.append(1)
        return "synthetic"

    for _ in range(2):
        with scan_scope():
            assert await cache.call(b"k", provider) == "synthetic"
    assert calls == [1] and list(cache.store.values()) == ["synthetic"]


@pytest.mark.asyncio
async def test_calls_without_a_scan_do_not_retain_scan_results():
    cache = bounded()
    calls = 0

    async def provider():
        nonlocal calls
        calls += 1
        return Reply(b"synthetic")

    await cache.call(b"k", provider, retention="scan")
    await cache.call(b"k", provider, retention="scan")
    assert calls == 2 and not cache.store


@pytest.mark.asyncio
async def test_process_and_scan_modes_partition_values_and_running_work():
    cache = bounded()
    calls = []

    async def provider(value):
        calls.append(value)
        await asyncio.sleep(0)
        return value

    with scan_scope():
        assert await cache.call(b"k", provider, "process") == "process"
        assert await cache.call(b"k", provider, "scan", retention="scan") == "scan"
        assert await cache.call(b"k", provider, "unused") == "process"
        assert await cache.call(b"k", provider, "unused", retention="scan") == "scan"
    assert calls == ["process", "scan"]
    assert list(cache.store.values()) == ["process"]
    with scan_scope():
        assert await cache.call(b"k", provider, "new", retention="scan") == "new"
    assert calls == ["process", "scan", "new"]


@pytest.mark.asyncio
async def test_overlapping_scans_each_own_a_shared_result_until_the_last_finishes():
    cache = bounded()
    ready = asyncio.Event()
    borrowed = asyncio.Event()
    leave_first = asyncio.Event()
    leave_second = asyncio.Event()
    calls = []
    references = []

    async def provider():
        calls.append(1)
        value = Reply(b"shared")
        references.append(weakref.ref(value))
        return value

    async def first():
        with scan_scope():
            await cache.call(b"k", provider, retention="scan")
            ready.set()
            await leave_first.wait()

    async def second():
        await ready.wait()
        with scan_scope():
            await cache.call(b"k", provider, retention="scan")
            borrowed.set()
            await leave_second.wait()

    tasks = [asyncio.create_task(first()), asyncio.create_task(second())]
    try:
        await borrowed.wait()
        leave_first.set()
        await tasks[0]
        assert len(cache.store) == 1 and references[0]() is not None
        leave_second.set()
        await tasks[1]
        await drain()
        assert not cache.store and calls == [1] and references[0]() is None
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_deferred_lease_preserves_then_releases_retention():
    cache = bounded()

    async def provider():
        return Reply(b"deferred")

    with scan_scope() as token:
        lease = retain_scan()
        result = await cache.call(b"k", provider, retention="scan")
        reference = weakref.ref(result)
        del result
    assert len(cache.store) == 1
    with scan_scope(token):
        assert await cache.call(b"k", provider, retention="scan") is reference()
    lease.close()
    lease.close()
    await drain()
    assert not cache.store and reference() is None


@pytest.mark.asyncio
async def test_completion_after_owner_closes_returns_but_does_not_retain():
    cache = bounded()
    started = asyncio.Event()
    finish = asyncio.Event()

    async def provider():
        started.set()
        await finish.wait()
        return Reply(b"late")

    async def child():
        result = await cache.call(b"k", provider, retention="scan")
        return weakref.ref(result)

    with scan_scope():
        task = asyncio.create_task(child())
        await started.wait()
    finish.set()
    reference = await task
    await drain()
    assert not cache.store and reference() is None


@pytest.mark.asyncio
async def test_waiter_retries_failed_preparation_through_its_own_request():
    class RequestFailure(Exception):
        pass

    cache = bounded()
    started = asyncio.Event()
    fail = asyncio.Event()
    calls = []

    async def rejected():
        calls.append("first request")
        started.set()
        await fail.wait()
        raise RequestFailure("synthetic owner failure")

    async def fresh():
        calls.append("waiting request")
        return "accepted"

    async def owner():
        with scan_scope():
            return await cache.call(b"k", rejected, retention="scan", retry_on=lambda e: True)

    first = asyncio.create_task(owner())
    await started.wait()
    try:
        with scan_scope():
            second = asyncio.create_task(
                cache.call(
                    b"k",
                    fresh,
                    retention="scan",
                    retry_on=lambda e: isinstance(e, RequestFailure),
                )
            )
            await drain()
            fail.set()
            with pytest.raises(RequestFailure):
                await first
            assert await second == "accepted"
            assert len(cache.store) == 1
        assert calls == ["first request", "waiting request"] and not cache.store
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["evict", "expire", "clear", "reconfigure", "raw-clear"])
async def test_retained_metadata_follows_store_lifecycle(action):
    now = [0.0]
    store = SlidingTTLCache(2, 10, lambda: now[0], max_bytes=32_000)
    cache = AsyncFunctionCache(store=store)

    async def provider(value):
        return Reply(value)

    with scan_scope() as token:
        for index in range(30):
            await cache.call(str(index).encode(), provider, b"data", retention="scan")
            if action == "expire":
                now[0] += 11
            elif action == "clear":
                cache.clear()
            elif action == "reconfigure":
                cache.configure(max_entries=2, ttl_seconds=10 + index % 2, max_bytes=32_000)
            elif action == "raw-clear":
                with cache.lock:
                    cache.store.clear()
            assert len(cache._scan_entries) <= 2
            assert sum(len(entries) for entries in cache._scan_owners.values()) <= 2
            assert len(token._release_callbacks) == 1
    assert not cache.store and not cache._scan_entries and not cache._scan_owners


@pytest.mark.asyncio
async def test_accepted_response_stays_stable_after_oversized_admission_refusal():
    cache = bounded(max_bytes=128)
    calls = []

    async def provider():
        calls.append(1)
        return Reply(b"x" * 4096)

    with scan_scope():
        first = cache.response_reader(b"k")
        try:
            result = await first.call(provider, retention="scan")
            reference = weakref.ref(result)
            first.accept()
        finally:
            first.close()
        del result
        assert not cache.store
        second = cache.response_reader(b"k")
        try:
            assert await second.call(provider, retention="scan") is reference()
            second.accept()
        finally:
            second.close()
        assert calls == [1]
    await drain()
    assert reference() is None
