# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Accepted raw replies belong to scan work independently of process-cache admission."""

import asyncio
import gc
import weakref
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from quarry.cache.async_cache import (
    AsyncFunctionCache,
    SlidingTTLCache,
    retain_scan,
    scan_scope,
)
from quarry.config import QuarryConfig
from quarry.integrations import model_inference as inference
from quarry.integrations.model_usage import model_call_budget
from quarry.safety.limits import SafetyLimits
from quarry.scanner import DSLScanner
from quarry.types.exceptions import LLMError
from quarry.types.messages import Messages


@pytest.fixture
def no_collection():
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


async def read(cache, key, request, validate=lambda value: value):
    reader = cache.response_reader(key)
    try:
        value = await reader.call(request)
        try:
            result = validate(value)
            if asyncio.iscoroutine(result):
                result = await result
        except Exception:
            reader.reject()
            raise
        else:
            reader.accept()
            return result
    finally:
        reader.close()


@dataclass
class Reply:
    value: int


async def drain():
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_first_accepted_reply_survives_process_admission_refusal_and_scope_boundaries():
    cache = AsyncFunctionCache.expiring(max_entries=1, ttl_seconds=60, max_bytes=1)
    calls = []

    async def request():
        calls.append(1)
        return len(calls)

    with scan_scope():
        assert await read(cache, b"k", request) == 1
        assert cache.store._byte_total == 0 and not cache.store
        assert await read(cache, b"k", request) == 1
    with scan_scope():
        assert await read(cache, b"k", request) == 2
    assert await read(cache, b"k", request) == 3
    assert await read(cache, b"k", request) == 4
    assert calls == [1, 1, 1, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["evict", "clear", "configure"])
async def test_process_table_changes_do_not_change_an_accepted_scan_reply(action):
    cache = AsyncFunctionCache.expiring(max_entries=1, ttl_seconds=60)
    calls = []

    async def request():
        calls.append(1)
        return len(calls)

    with scan_scope():
        assert await read(cache, b"k", request) == 1
        if action == "evict":
            await cache.call(b"other", request)
        elif action == "clear":
            cache.clear()
        else:
            cache.configure(max_entries=2, ttl_seconds=10, max_bytes=1)
        assert b"k" not in cache.store
        assert await read(cache, b"k", request) == 1
    assert len(calls) == (2 if action == "evict" else 1)


@pytest.mark.asyncio
async def test_scan_hits_do_not_refresh_the_process_ttl_and_real_expiry_is_exercised():
    now = [0.0]
    cache = AsyncFunctionCache(store=SlidingTTLCache(2, 1, timer=lambda: now[0]))
    calls = []

    async def request():
        calls.append(1)
        return len(calls)

    with scan_scope():
        assert await read(cache, b"k", request) == 1
        now[0] = 0.75
        assert await read(cache, b"k", request) == 1
        now[0] = 1.1
        with pytest.raises(KeyError):
            cache.store[b"k"]
        assert await read(cache, b"k", request) == 1
    assert await read(cache, b"k", request) == 2


@pytest.mark.asyncio
async def test_fixed_zero_lifetime_reply_actually_expires_without_a_clock_mock_or_sleep():
    class ImmediatelyExpires(Reply):
        cache_lifetime = 0

    cache = AsyncFunctionCache.expiring(max_entries=2, ttl_seconds=60)
    calls = []

    async def request():
        calls.append(1)
        return ImmediatelyExpires(len(calls))

    with scan_scope():
        assert await read(cache, b"k", request, lambda value: value.value) == 1
        with pytest.raises(KeyError):
            cache.store[b"k"]
        assert await read(cache, b"k", request, lambda value: value.value) == 1
    assert await read(cache, b"k", request, lambda value: value.value) == 2


@pytest.mark.asyncio
async def test_each_reader_validates_independently_and_rejection_invalidates_accepted_reply():
    cache = AsyncFunctionCache()
    calls, reads = [], []

    async def request():
        calls.append(1)
        return {"left": len(calls), "right": 10 * len(calls)}

    def left(value):
        reads.append("left")
        return value["left"]

    def right(value):
        reads.append("right")
        return value["right"]

    with scan_scope():
        assert await read(cache, b"k", request, left) == 1
        assert await read(cache, b"k", request, right) == 10
        with pytest.raises(KeyError):
            await read(cache, b"k", request, lambda value: value["missing"])
        assert not cache.store
        assert await read(cache, b"k", request, left) == 2
    assert reads == ["left", "right", "left"] and len(calls) == 2


@pytest.mark.asyncio
async def test_late_rejection_does_not_remove_a_newer_process_reply():
    cache = AsyncFunctionCache()
    validating, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def request():
        calls.append(1)
        return len(calls)

    async def reject(_value):
        validating.set()
        await finish.wait()
        raise ValueError("rejected old reply")

    with scan_scope():
        old = asyncio.create_task(read(cache, b"k", request, reject))
        await validating.wait()
        cache.clear()
        with scan_scope():
            assert await read(cache, b"k", request) == 2
        finish.set()
        with pytest.raises(ValueError, match="rejected old reply"):
            await old
    assert cache.store[b"k"] == 2
    assert await read(cache, b"k", request) == 2 and len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("another_accepts", [False, True])
async def test_cancelled_validation_drops_only_unaccepted_provisional_replies(another_accepts):
    cache = AsyncFunctionCache.expiring(max_entries=1, ttl_seconds=60, max_bytes=1)
    validating = asyncio.Event()
    calls = []

    async def request():
        calls.append(1)
        return len(calls)

    async def blocked(value):
        validating.set()
        await asyncio.Event().wait()
        return value

    with scan_scope():
        task = asyncio.create_task(read(cache, b"k", request, blocked))
        await validating.wait()
        if another_accepts:
            assert await read(cache, b"k", request) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await read(cache, b"k", request) == (1 if another_accepts else 2)
    assert len(calls) == (1 if another_accepts else 2)


@pytest.mark.asyncio
async def test_a_rejected_reply_cannot_be_repinned_by_an_older_successful_reader():
    cache = AsyncFunctionCache.expiring(max_entries=1, ttl_seconds=60, max_bytes=1)
    validating, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def request():
        calls.append(1)
        return len(calls)

    async def slow(value):
        validating.set()
        await finish.wait()
        return value

    def reject(_value):
        raise ValueError("bad reader")

    with scan_scope():
        first = asyncio.create_task(read(cache, b"k", request, slow))
        await validating.wait()
        with pytest.raises(ValueError, match="bad reader"):
            await read(cache, b"k", request, reject)
        assert await read(cache, b"k", request) == 2
        finish.set()
        assert await first == 1
        assert await read(cache, b"k", request) == 2
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["complete", "failure"])
async def test_closed_scope_releases_replies_without_gc_even_if_its_token_is_kept(outcome):
    cache = AsyncFunctionCache.expiring(max_entries=1, ttl_seconds=60, max_bytes=1)
    references = []

    async def request():
        value = Reply(1)
        references.append(weakref.ref(value))
        return value

    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        with suppress(ValueError), scan_scope() as token:
            assert await read(cache, b"k", request, lambda value: value.value) == 1
            assert references[0]() is not None
            if outcome == "failure":
                raise ValueError("scan failed")
        await drain()
        assert token is not None and references[0]() is None
    finally:
        if enabled:
            gc.enable()


@pytest.mark.asyncio
async def test_deferred_lease_keeps_replies_until_closed_and_is_idempotent(no_collection):
    cache = AsyncFunctionCache.expiring(max_entries=1, ttl_seconds=60, max_bytes=1)
    calls, references = [], []

    async def request():
        calls.append(1)
        value = Reply(len(calls))
        references.append(weakref.ref(value))
        return value

    with scan_scope() as token:
        lease = retain_scan()
        assert await read(cache, b"k", request, lambda value: value.value) == 1
    await drain()
    assert references[0]() is not None
    with scan_scope(token):
        assert await read(cache, b"k", request, lambda value: value.value) == 1
    lease.close()
    lease.close()
    await drain()
    assert references[0]() is None and calls == [1]


@pytest.mark.asyncio
async def test_new_scope_generation_does_not_accept_late_results_from_an_ended_scope():
    cache = AsyncFunctionCache.expiring(max_entries=1, ttl_seconds=60, max_bytes=1)
    started, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def request():
        calls.append(1)
        if len(calls) == 1:
            started.set()
            await finish.wait()
        return len(calls)

    with scan_scope() as token:
        old = asyncio.create_task(read(cache, b"k", request))
        await started.wait()
    with scan_scope(token):
        finish.set()
        assert await old == 1
        assert await read(cache, b"k", request) == 2
        assert await read(cache, b"k", request) == 2
    assert len(calls) == 2


def configured(monkeypatch, values):
    class Adapter:
        calls = 0

        async def score(self, config, messages, *, timeout=None):
            self.calls += 1
            return values(self.calls)

    adapter = Adapter()
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "j": {
                    "adapter": "owned",
                    "model": "owned",
                    "score": {
                        "message_builder": "constitution",
                        "boundary_seed": 7,
                        "tokens": {"negative": "0", "positive": "1"},
                    },
                }
            },
        },
        Path("."),
        require_rules=False,
    )
    backend = inference.ModelInferenceBackend(config, adapters={"owned": adapter})
    cache = AsyncFunctionCache.expiring(max_entries=1, ttl_seconds=60, max_bytes=1)
    monkeypatch.setattr(inference, "result_cache", cache)

    async def call(validate=None):
        return await backend._call(
            "j",
            "owned",
            adapter,
            "score",
            config.models["j"],
            lambda: ([],),
            key=("owned",),
            timeout=None,
            read=validate,
        )

    return config, backend, adapter, cache, call


@pytest.mark.asyncio
async def test_backend_keeps_raw_reply_but_still_runs_each_reader_and_detaches_raw_results(
    monkeypatch,
):
    _, _, adapter, cache, call = configured(monkeypatch, lambda n: {"value": n, "items": [n]})
    seen = []

    def validate(value):
        seen.append(value["value"])
        return value["value"]

    with scan_scope():
        assert await call(validate) == 1
        value = await call()
        value["items"].append(99)
        cache.clear()
        assert await call() == {"value": 1, "items": [1]}
        assert await call(validate) == 1
    assert seen == [1, 1] and adapter.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["abandon", "finish", "failure", "cancel"])
async def test_real_scanner_deferred_owner_releases_accepted_reply(
    monkeypatch, action, no_collection
):
    class WeakFloat(float):
        pass

    references = []

    def value(_n):
        response = WeakFloat(0.8)
        references.append(weakref.ref(response))
        return response

    config, backend, adapter, _, _ = configured(monkeypatch, value)
    config = replace(
        config,
        safety_limits=SafetyLimits(rule_timeout=None),
        quarry_rule_texts=('rule "R" condition: model("j").score(messages,"criterion") > 0.5',),
    )
    scanner = DSLScanner(config, model_inference_backend=backend)
    deferred = []
    result = await scanner._scan_data_with_parameters_async(
        {"messages": [{"role": "user", "content": "owned"}]},
        {},
        enable_explain=True,
        rules_subset={"R"},
        decision_paths=True,
        deferred_evidence=deferred,
    )
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(deferred) == 1 and references[0]() is not None
    owner = deferred.pop()
    if action == "abandon":
        owner.close()
    elif action == "finish":
        result = await owner(result)
    else:

        async def failed(_result):
            if action == "cancel":
                raise asyncio.CancelledError
            raise ValueError("evidence failed")

        owner._finish = failed
        with pytest.raises(asyncio.CancelledError if action == "cancel" else ValueError):
            await owner(result)
    del result
    await drain()
    assert references[0]() is None and adapter.calls == 1


@pytest.mark.asyncio
async def test_cache_namespaces_and_none_replies_are_kept_separately():
    left = AsyncFunctionCache.expiring(max_entries=1, ttl_seconds=60, max_bytes=1)
    right = AsyncFunctionCache.expiring(max_entries=1, ttl_seconds=60, max_bytes=1)
    calls = []

    async def empty():
        calls.append("empty")
        return None

    async def value():
        calls.append("right")
        return "right"

    with scan_scope():
        assert await read(left, b"k", empty) is None
        assert await read(right, b"k", value) == "right"
        assert await read(left, b"k", empty) is None
        assert await read(right, b"k", value) == "right"
    assert calls == ["empty", "right"]


@pytest.mark.asyncio
async def test_accepted_replies_can_be_read_on_another_thread_and_event_loop():
    cache = AsyncFunctionCache.expiring(max_entries=1, ttl_seconds=60, max_bytes=1)
    calls = []

    async def request():
        calls.append(1)
        return len(calls)

    async def resumed(token):
        with scan_scope(token):
            return await read(cache, b"k", request)

    with scan_scope() as token:
        assert await read(cache, b"k", request) == 1
        assert await asyncio.to_thread(lambda: asyncio.run(resumed(token))) == 1
    assert calls == [1]


@pytest.mark.asyncio
async def test_cancelled_scan_releases_other_replies_while_a_shared_provider_call_still_runs(
    no_collection,
):
    cache = AsyncFunctionCache.expiring(max_entries=1, ttl_seconds=60, max_bytes=1)
    started, finish = asyncio.Event(), asyncio.Event()
    references, tokens, calls = [], [], []

    async def accepted():
        reply = Reply(1)
        references.append(weakref.ref(reply))
        return reply

    async def shared():
        calls.append(1)
        started.set()
        await finish.wait()
        return 2

    async def first():
        with scan_scope() as token:
            tokens.append(token)
            assert await read(cache, b"accepted", accepted, lambda value: value.value) == 1
            return await read(cache, b"pending", shared)

    async def second():
        with scan_scope():
            return await read(cache, b"pending", shared)

    first_task = asyncio.create_task(first())
    await started.wait()
    second_task = asyncio.create_task(second())

    async def both_waiting():
        while cache.store[b"pending"].waiters != 2:
            await asyncio.sleep(0)

    await asyncio.wait_for(both_waiting(), 3)
    first_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_task
    del first_task
    await drain()
    assert tokens and references[0]() is None and not second_task.done()
    finish.set()
    assert await second_task == 2 and calls == [1]


@pytest.mark.asyncio
async def test_backend_reader_validation_and_physical_budget_accounting_remain_independent(
    monkeypatch,
):
    _, _, adapter, cache, call = configured(monkeypatch, lambda n: {"value": n})
    seen = []

    def valid(value):
        seen.append(value["value"])
        return value["value"]

    with scan_scope(), model_call_budget(2) as budget:
        assert await call(valid) == 1
        assert await call(valid) == 1
        assert budget.calls_by_alias == {"j": 1}
        with pytest.raises(KeyError):
            await call(lambda value: value["missing"])
        assert await call(valid) == 2
        assert budget.calls_by_alias == {"j": 2}
    assert seen == [1, 1, 2] and adapter.calls == 2
    assert cache.store._byte_total == 0


@pytest.mark.asyncio
async def test_resuming_a_foreign_process_token_does_not_take_its_inherited_lock():
    cache = AsyncFunctionCache.expiring(max_entries=1, ttl_seconds=60, max_bytes=1)
    calls = []

    async def request():
        calls.append(1)
        return len(calls)

    with scan_scope() as parent:
        assert await read(cache, b"k", request) == 1
        pid = parent.pid
        parent._lock.acquire()
        parent.pid = -1
        try:
            with scan_scope(parent) as child:
                assert child is not parent
                assert await read(cache, b"k", request) == 2
        finally:
            parent.pid = pid
            parent._lock.release()
        assert await read(cache, b"k", request) == 1
    assert calls == [1, 1]


@pytest.mark.asyncio
async def test_reply_destruction_can_reenter_the_closed_scan_without_its_lock(no_collection):
    cache = AsyncFunctionCache.expiring(max_entries=1, ttl_seconds=60, max_bytes=1)
    closed = []
    with scan_scope() as token:

        class Reentrant:
            def __del__(self):
                closed.append(token.generation())

        async def request():
            return Reentrant()

        assert await read(cache, b"k", request, lambda _value: True)
        await drain()
    await drain()
    assert closed == [None]


@pytest.mark.asyncio
async def test_late_rejection_does_not_refresh_a_replacement_reply_ttl():
    now = [0.0]
    cache = AsyncFunctionCache(store=SlidingTTLCache(2, 1, timer=lambda: now[0]))
    validating, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def request():
        calls.append(1)
        return len(calls)

    async def reject(_value):
        validating.set()
        await finish.wait()
        raise ValueError("old reply rejected")

    with scan_scope():
        old = asyncio.create_task(read(cache, b"k", request, reject))
        await validating.wait()
        cache.clear()
        now[0] = 0.25
        with scan_scope():
            assert await read(cache, b"k", request) == 2
        now[0] = 0.75
        finish.set()
        with pytest.raises(ValueError, match="old reply rejected"):
            await old
        now[0] = 1.26
        with pytest.raises(KeyError):
            cache.store[b"k"]
    assert calls == [1, 1]


@pytest.mark.asyncio
async def test_classifier_aliases_read_their_own_heads_from_one_accepted_physical_reply(
    monkeypatch,
):
    from types import SimpleNamespace

    class Adapter:
        calls = 0

        async def classify_messages(self, config, messages, tools, *, timeout=None):
            self.calls += 1
            return SimpleNamespace(
                chunks=[
                    SimpleNamespace(
                        probabilities=[self.calls / 10, 1 - self.calls / 10],
                    )
                ]
            )

    adapter = Adapter()
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                name: {
                    "adapter": "owned",
                    "model": "owned",
                    "classify": {"class_index": index},
                }
                for name, index in [("left", 0), ("right", 1), ("bad", 3)]
            },
        },
        Path("."),
        require_rules=False,
    )
    backend = inference.ModelInferenceBackend(config, adapters={"owned": adapter})
    cache = AsyncFunctionCache.expiring(max_entries=1, ttl_seconds=60, max_bytes=1)
    monkeypatch.setattr(inference, "result_cache", cache)
    messages = Messages.from_openai([{"role": "user", "content": "owned"}])
    with scan_scope(), model_call_budget(2) as budget:
        assert await backend.model_classify_messages("left", messages) == 0.1
        assert await backend.model_classify_messages("right", messages) == 0.9
        assert adapter.calls == 1 and not cache.store
        with pytest.raises(LLMError, match="outside"):
            await backend.model_classify_messages("bad", messages)
        assert await backend.model_classify_messages("left", messages) == 0.2
        assert await backend.model_classify_messages("right", messages) == 0.8
        assert budget.calls_by_alias == {"left": 2}
    assert adapter.calls == 2
