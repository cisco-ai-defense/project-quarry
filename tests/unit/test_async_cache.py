# Copyright 2025 Cisco Systems, Inc. and its affilitates
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

"""The result cache keeps results, and shares each call still running among its callers."""

import asyncio
import contextvars
import functools
import threading
import time

import pytest
from cachetools import TTLCache

from quarry.cache.async_cache import AsyncFunctionCache, SlidingTTLCache, _Pending, scan_scope
from quarry.cache.keys import object_identity
from quarry.safety.timer import QuarryTimeoutError, evaluation_deadline


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_in_flight_call_per_key() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def slow_provider(value: str) -> str:
        calls.append(value)
        if len(calls) == 3:
            started.set()
        await release.wait()
        return value

    cache = AsyncFunctionCache()
    keys = ["shared", "second", "third", "shared"]
    tasks = [asyncio.create_task(cache.call(key.encode(), slow_provider, key)) for key in keys]
    await started.wait()
    release.set()

    assert await asyncio.gather(*tasks) == keys
    assert sorted(calls) == ["second", "shared", "third"]


def test_scans_on_other_threads_share_a_running_call_and_retry_one_whose_loop_ended() -> None:
    """Each synchronous scan runs its own event loop; a scan on another thread waits for the
    call one started, and a caller, scanning or not, makes its own if the starter's loop ends
    before the call does."""
    cache = AsyncFunctionCache()
    release = threading.Event()
    calls = []

    async def provider(name: str) -> str:
        calls.append(name)
        while not release.is_set():
            await asyncio.sleep(0.01)
        return f"answer for {name}"

    def waiting_on(key: bytes, waiters: int) -> None:
        """Wait until ``waiters`` callers wait for the call running under ``key``."""
        deadline = time.monotonic() + 5
        while getattr(cache.store.get(key), "waiters", 0) != waiters:
            assert time.monotonic() < deadline, f"{waiters} callers never waited on {key!r}"
            time.sleep(0.01)

    async def scanned(key: bytes, name: str) -> str:
        with scan_scope():
            return await cache.call(key, provider, name)

    def scan(key: bytes, name: str, results: list) -> threading.Thread:
        thread = threading.Thread(target=lambda: results.append(asyncio.run(scanned(key, name))))
        thread.start()
        return thread

    shared: list[str] = []
    threads = [scan(b"shared", "first", shared)]
    waiting_on(b"shared", 1)
    threads.append(scan(b"shared", "second", shared))
    waiting_on(b"shared", 2)
    release.set()
    for thread in threads:
        thread.join()
    assert (shared, calls) == (["answer for first"] * 2, ["first"])

    release.clear()
    calls.clear()

    returned = threading.Event()

    async def return_early() -> None:
        # A host calling outside any scan returns without its call; asyncio.run then cancels
        # the call it started.
        asyncio.get_running_loop().create_task(cache.call(b"left", provider, "left"))
        while not returned.is_set():
            await asyncio.sleep(0.01)

    left = threading.Thread(target=lambda: asyncio.run(return_early()))
    left.start()
    waiting_on(b"left", 1)
    waiting: list[str] = []
    waiter = threading.Thread(
        target=lambda: waiting.append(asyncio.run(cache.call(b"left", provider, "waiter")))
    )
    waiter.start()
    waiting_on(b"left", 2)
    returned.set()
    left.join()
    release.set()
    waiter.join()
    assert (waiting, calls) == (["answer for waiter"], ["left", "waiter"])


@pytest.mark.asyncio
async def test_a_call_that_cancels_itself_fails_its_own_scan_once() -> None:
    """A call that cancels itself, as an adapter's own timeout can, is not made again by its
    own scan, which would only cancel it again."""
    calls = []

    async def provider() -> str:
        calls.append(1)
        raise asyncio.CancelledError

    cache = AsyncFunctionCache()
    with scan_scope(), pytest.raises(Exception, match="cancelled"):
        await cache.call(b"k", provider)
    assert calls == [1]


def test_the_store_keeps_running_calls_and_drops_results_by_age_and_bound() -> None:
    """With one time-to-live, the least recently used result is the next to expire, and a
    running call is neither expired nor dropped."""
    now = [0.0]
    store = SlidingTTLCache(maxsize=2, ttl=10, timer=lambda: now[0])
    running = [_Pending(key, None, None, None) for key in ("p", "q")]  # type: ignore[arg-type]
    for pending in running:
        store[pending.key] = pending
    store["a"], store["b"] = 1, 2
    store["c"] = 3  # past the bound: "a", the least recently used result, goes
    assert sorted(store) == ["b", "c", "p", "q"]

    now[0] = 20  # past every result's time-to-live
    with pytest.raises(KeyError):
        store["b"]
    store["d"] = 4  # a write drops the expired "c" too
    assert sorted(store) == ["d", "p", "q"]
    assert (store["p"], store["q"]) == tuple(running)

    del store["p"]
    assert sorted(store) == ["d", "q"]
    store.clear()
    assert len(store) == 0


@pytest.mark.asyncio
async def test_a_long_lived_cache_expires_results_after_their_last_hit_and_bounds_them() -> None:
    now = [0.0]
    calls = []

    async def provider(value: str) -> str:
        calls.append(value)
        return value

    cache = AsyncFunctionCache(store=SlidingTTLCache(maxsize=2, ttl=10, timer=lambda: now[0]))
    await cache.call(b"a", provider, "a")
    now[0] = 8
    await cache.call(b"a", provider, "a")  # a hit renews the result
    now[0] = 16
    await cache.call(b"a", provider, "a")
    assert calls == ["a"]
    now[0] = 27
    await cache.call(b"a", provider, "a")  # 11 seconds after its last hit
    assert calls == ["a", "a"]
    await cache.call(b"b", provider, "b")
    await cache.call(b"c", provider, "c")  # past two results, "a" is the oldest hit
    await cache.call(b"a", provider, "a")
    assert calls == ["a", "a", "b", "c", "a"]

    # A call still running neither expires nor is dropped, however long it takes and however
    # many results are written meanwhile.
    release = asyncio.Event()

    async def slow(value: str) -> str:
        calls.append(value)
        await release.wait()
        return value

    first = asyncio.create_task(cache.call(b"slow", slow, "slow"))
    await asyncio.sleep(0)
    for value in ("x", "y", "z"):
        await cache.call(value.encode(), provider, value)
    now[0] = 100
    second = asyncio.create_task(cache.call(b"slow", slow, "slow"))
    await asyncio.sleep(0)
    release.set()
    assert (await first, await second, calls.count("slow")) == ("slow", "slow", 1)


@pytest.mark.asyncio
async def test_aliases_of_one_scan_share_a_request_backed_call_other_scans_do_not_join() -> None:
    """A request-backed call is shared by every alias of its scan, and not joined by
    another scan, whose speed it is not."""
    release = asyncio.Event()
    calls = []

    async def provider(name: str) -> str:
        calls.append(name)
        await release.wait()
        return "answer"

    cache = AsyncFunctionCache()
    with scan_scope():
        scan = contextvars.copy_context()
    with scan_scope():
        other = contextvars.copy_context()
    tasks = [
        asyncio.create_task(
            cache.call(b"k", provider, name, budget=budget, alone=True), context=ctx
        )
        for name, budget, ctx in (
            ("a", "alias-a", scan),
            ("b", "alias-b", scan),
            ("c", "alias-a", other),
        )
    ]
    await asyncio.sleep(0)
    release.set()
    assert await asyncio.gather(*tasks) == ["answer"] * 3
    assert calls == ["a", "c"]


@pytest.mark.asyncio
async def test_a_caller_retries_once_when_a_call_another_caller_started_fails() -> None:
    """A shared call fails under the deadline and budget of the request that started it,
    so a caller that joined it retries once with a call of its own; the starter does not."""
    release = asyncio.Event()
    attempts = []

    async def provider(name: str) -> str:
        attempts.append(name)
        await release.wait()
        if len(attempts) == 1:
            raise RuntimeError("the starter's budget is spent")
        return "scored"

    cache = AsyncFunctionCache(store=TTLCache(maxsize=10, ttl=60))
    with scan_scope():
        first = contextvars.copy_context()
    with scan_scope():
        second = contextvars.copy_context()

    def budget(error: BaseException) -> bool:
        return "budget" in str(error)

    starter = asyncio.create_task(
        cache.call(b"k", provider, "starter", retry_on=budget), context=first
    )
    await asyncio.sleep(0)
    joiner = asyncio.create_task(
        cache.call(b"k", provider, "joiner", retry_on=budget), context=second
    )
    same_scan = asyncio.create_task(
        cache.call(b"k", provider, "same", retry_on=budget), context=first
    )
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(RuntimeError, match="budget is spent"):
        await starter
    # A caller of the same scan shares its budget and deadline, so it does not retry.
    with pytest.raises(RuntimeError, match="budget is spent"):
        await same_scan
    assert await joiner == "scored"
    assert attempts == ["starter", "joiner"]


@pytest.mark.asyncio
async def test_a_caller_stops_waiting_at_its_own_deadline_and_keeps_others_call() -> None:
    """A caller whose deadline ends gives up on time, without asking again; the call it
    joined goes on for its starter, and is not dropped by an unreadable result's eviction."""
    release = asyncio.Event()
    calls = []

    async def provider() -> str:
        calls.append(1)
        await release.wait()
        return "scored"

    cache = AsyncFunctionCache()
    with scan_scope():
        starter = asyncio.create_task(cache.call(b"k", provider, retry_on=lambda _: True))
    await asyncio.sleep(0)
    with pytest.raises(QuarryTimeoutError), scan_scope(), evaluation_deadline(0.1):
        await asyncio.wait_for(cache.call(b"k", provider, retry_on=lambda _: True), timeout=10)
    # Forgetting a key keeps the call running under it.
    cache.forget(b"k")
    joiner = asyncio.create_task(cache.call(b"k", provider))
    await asyncio.sleep(0)
    release.set()
    assert (await starter, await joiner, calls) == ("scored", "scored", [1])


def test_an_object_is_named_by_itself() -> None:
    """An instance or a partial, whose configuration its class does not show, is named by
    itself, so two configurations never share a key."""

    class Prober:
        def __init__(self, threshold: float) -> None:
            self.threshold = threshold

    strict, loose = Prober(0.9), Prober(0.1)
    first, second = functools.partial(max, 1), functools.partial(max, 2)

    assert object_identity(strict) == object_identity(strict) != object_identity(loose)
    assert object_identity(first) != object_identity(second)
    assert object_identity(None) == object_identity(None)


@pytest.mark.asyncio
async def test_a_rule_retries_a_call_another_rules_deadline_ended_in_one_scan() -> None:
    """Rules of one scan share its alias budget but each has its own deadline, so a call
    the starter's deadline ended is made again for a rule that still has time."""
    release = asyncio.Event()
    attempts = []

    async def provider(name: str) -> str:
        attempts.append(name)
        await release.wait()
        if len(attempts) == 1:
            raise TimeoutError("the starter's deadline passed")
        return "scored"

    cache = AsyncFunctionCache()
    timed_out = lambda error: isinstance(error, TimeoutError)  # noqa: E731
    with scan_scope():
        with evaluation_deadline(30):
            short = contextvars.copy_context()
        with evaluation_deadline(60):
            long = contextvars.copy_context()
    starter = asyncio.create_task(
        cache.call(b"k", provider, "short", budget="judge", retry_on=timed_out), context=short
    )
    await asyncio.sleep(0)
    joiner = asyncio.create_task(
        cache.call(b"k", provider, "long", budget="judge", retry_on=timed_out), context=long
    )
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(TimeoutError):
        await starter
    assert (await joiner, attempts) == ("scored", ["short", "long"])


@pytest.mark.asyncio
async def test_a_failure_not_tied_to_the_starter_reaches_every_waiter() -> None:
    """A shared call that fails for every caller alike is not sent again by each scan."""
    release = asyncio.Event()
    attempts = []

    async def provider(name: str) -> str:
        attempts.append(name)
        await release.wait()
        raise RuntimeError("the provider rejected the request")

    cache = AsyncFunctionCache()
    with scan_scope():
        first = contextvars.copy_context()
    with scan_scope():
        second = contextvars.copy_context()
    budget = lambda error: "budget" in str(error)  # noqa: E731
    starter = asyncio.create_task(cache.call(b"k", provider, "a", retry_on=budget), context=first)
    await asyncio.sleep(0)
    joiner = asyncio.create_task(cache.call(b"k", provider, "b", retry_on=budget), context=second)
    await asyncio.sleep(0)
    release.set()

    for task in (starter, joiner):
        with pytest.raises(RuntimeError, match="rejected"):
            await task
    assert attempts == ["a"]
