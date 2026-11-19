# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Scan retention survives independent loops and closes on actual ownership exits."""

import asyncio
import concurrent.futures
import os
import subprocess
import sys
import threading
import weakref

import pytest

from quarry.cache.async_cache import AsyncFunctionCache, scan_scope


def cache():
    return AsyncFunctionCache.expiring(max_entries=8, ttl_seconds=60, max_bytes=32_000)


async def drain():
    for _ in range(4):
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["clear", "reconfigure"])
async def test_running_scan_work_survives_table_mutation_and_releases(mutation):
    results = cache()
    started, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def provider():
        calls.append(1)
        started.set()
        await finish.wait()
        return "synthetic"

    with scan_scope():
        first = asyncio.create_task(results.call(b"k", provider, retention="scan"))
        await started.wait()
        if mutation == "clear":
            results.clear()
        else:
            results.configure(max_entries=2, ttl_seconds=30, max_bytes=16_000)
        second = asyncio.create_task(results.call(b"k", provider, retention="scan"))
        await drain()
        finish.set()
        assert await asyncio.gather(first, second) == ["synthetic", "synthetic"]
        assert len(results.store) == 1
    assert calls == [1] and not results.store and not results._scan_owners


@pytest.mark.asyncio
async def test_clearing_an_old_store_cannot_detach_new_scan_result_ownership():
    results = cache()
    calls = []

    async def provider():
        calls.append(1)
        return len(calls)

    with scan_scope():
        assert await results.call(b"k", provider, retention="scan") == 1
        old_store = results.store
        results.configure(max_entries=2, ttl_seconds=30, max_bytes=16_000)
        assert await results.call(b"k", provider, retention="scan") == 2
        with results.lock:
            old_store.clear()
    assert calls == [1, 1] and not results.store and not results._scan_owners


@pytest.mark.asyncio
@pytest.mark.parametrize("waiters", [1, 4])
async def test_last_waiter_cancellation_stops_running_work_and_removes_ownership(waiters):
    results = cache()
    started, stopped = asyncio.Event(), asyncio.Event()

    async def provider():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    with scan_scope():
        tasks = [
            asyncio.create_task(results.call(b"k", provider, retention="scan"))
            for _ in range(waiters)
        ]
        await started.wait()
        for task in tasks:
            task.cancel()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(outcome, asyncio.CancelledError) for outcome in outcomes)
        assert stopped.is_set()
        await drain()
        assert not results.store and not results._scan_owners and not results._scan_entries
    assert not results.store


@pytest.mark.asyncio
async def test_one_cancelled_scan_does_not_cancel_another_scans_shared_work():
    results = cache()
    started, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def provider():
        calls.append(1)
        started.set()
        await finish.wait()
        return "synthetic"

    async def scan():
        with scan_scope():
            return await results.call(b"k", provider, retention="scan")

    first = asyncio.create_task(scan())
    await started.wait()
    second = asyncio.create_task(scan())
    await drain()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    finish.set()
    assert await second == "synthetic"
    await drain()
    assert calls == [1] and not results.store and not results._scan_owners


@pytest.mark.asyncio
async def test_reusing_a_closed_token_does_not_join_its_previous_alone_generation():
    results = cache()
    started, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def old_request():
        calls.append("old")
        started.set()
        await finish.wait()
        return "old"

    async def new_request():
        calls.append("new")
        return "new"

    with scan_scope() as token:
        old = asyncio.create_task(results.call(b"k", old_request, retention="scan", alone=True))
        await started.wait()
    with scan_scope(token):
        assert await results.call(b"k", new_request, retention="scan", alone=True) == "new"
        finish.set()
        assert await old == "old"
        assert await results.call(b"k", new_request, retention="scan") == "new"
    assert calls == ["old", "new"] and not results.store


@pytest.mark.asyncio
async def test_last_scan_result_destructor_can_reenter_cache_without_deadlock():
    results = cache()
    destroyed = []

    class Reply:
        def __del__(self):
            with results.lock:
                destroyed.append(len(results.store))

    async def provider():
        return Reply()

    with scan_scope():
        value = await results.call(b"k", provider, retention="scan")
        reference = weakref.ref(value)
        del value
        await drain()
    assert reference() is None and destroyed == [0]


def test_two_thread_event_loops_share_running_work_and_last_scan_owns_the_result():
    results = cache()
    started, joined, finish = threading.Event(), threading.Event(), threading.Event()
    leave_first, leave_second, borrowed = threading.Event(), threading.Event(), threading.Event()
    calls = []

    async def wait(event):
        assert await asyncio.to_thread(event.wait, 5), "synthetic coordination timed out"

    async def provider():
        calls.append(1)
        started.set()
        await wait(finish)
        return "synthetic"

    async def scan(first):
        if not first:
            await wait(started)
        with scan_scope():
            task = asyncio.create_task(results.call(b"k", provider, retention="scan"))
            if not first:
                # Wait until the call coroutine has actually joined the pending future.
                await drain()
                joined.set()
            assert await task == "synthetic"
            if not first:
                borrowed.set()
            await wait(leave_first if first else leave_second)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(asyncio.run, scan(True))
        second = pool.submit(asyncio.run, scan(False))
        try:
            assert joined.wait(5)
            finish.set()
            assert borrowed.wait(5)
            leave_first.set()
            first.result(timeout=5)
            with results.lock:
                assert len(results.store) == 1
            leave_second.set()
            second.result(timeout=5)
        finally:
            finish.set()
            leave_first.set()
            leave_second.set()
    assert calls == [1] and not results.store and not results._scan_owners


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires process fork")
def test_fork_resets_inherited_locks_replies_and_running_work_in_child():
    script = r"""
import asyncio, json, os, signal
from quarry.cache.async_cache import AsyncFunctionCache, current_scan, scan_scope

results = AsyncFunctionCache.expiring(max_entries=7, ttl_seconds=19, max_bytes=12345)
async def ready(): return "parent"
asyncio.run(results.call(b"process", ready))
loop = asyncio.new_event_loop()
async def running(): await asyncio.Event().wait()
with scan_scope() as token:
    asyncio.run(results.call(b"scan", ready, retention="scan"))
    task = loop.create_task(results.call(b"pending", running, retention="scan"))
    loop.run_until_complete(asyncio.sleep(0))
    results.lock.acquire()
    token._lock.acquire()
    pid = os.fork()
    if pid == 0:
        signal.alarm(8)
        try:
            assert current_scan() is None
            assert not results.store and not results._scan_owners and not results._scan_entries
            assert (results.store.maxsize, results.store.ttl, results.store.max_bytes) == (7, 19, 12345)
            assert token.generation() is None
            async def child(): return "child"
            async def check():
                with scan_scope(token):
                    assert await results.call(b"scan", child, retention="scan") == "child"
                assert not results.store
                assert await results.call(b"process", child) == "child"
            asyncio.run(check())
            os._exit(0)
        except BaseException:
            import traceback
            traceback.print_exc()
            os._exit(1)
    token._lock.release()
    results.lock.release()
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert asyncio.run(results.call(b"process", ready)) == "parent"
    assert token.generation() is not None
    task.cancel()
    loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
loop.run_until_complete(asyncio.sleep(0))
loop.close()
assert list(results.store.values()) == ["parent"]
print(json.dumps({"child": "fresh", "parent": "preserved"}))
"""
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stderr
    assert '"parent": "preserved"' in run.stdout


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires process fork")
@pytest.mark.parametrize("accepted_only", [False, True])
def test_fork_reply_destruction_can_reenter_another_cache_and_scan(accepted_only):
    script = r"""
import os, signal
# Register first so a deadlock in Quarry's later child callback is bounded too.
os.register_at_fork(after_in_child=lambda: signal.alarm(5))
import asyncio, json, sys
from quarry.cache.async_cache import AsyncFunctionCache, scan_scope
parent_pid = os.getpid()
accepted_only = sys.argv[1] == "accepted"
first = AsyncFunctionCache.expiring(max_entries=2, ttl_seconds=60, max_bytes=1 if accepted_only else 32000)
second = AsyncFunctionCache.expiring(max_entries=2, ttl_seconds=60)
destroyed = []
with scan_scope() as token:
    class Raw:
        def __del__(self):
            if os.getpid() != parent_pid:
                with second.lock:
                    destroyed.append(token.generation())
    async def provider(): return Raw()
    async def prime():
        if not accepted_only:
            await first.call(b"raw", provider)
        else:
            reader = first.response_reader(b"raw")
            try:
                await reader.call(provider, retention="scan")
                reader.accept()
            finally:
                reader.close()
    asyncio.run(prime())
    first.lock.acquire()
    second.lock.acquire()
    token._lock.acquire()
    pid = os.fork()
    if pid == 0:
        try:
            assert not first.store and not second.store
            assert destroyed == [None], destroyed
            os._exit(0)
        except BaseException:
            import traceback
            traceback.print_exc()
            os._exit(1)
    token._lock.release()
    second.lock.release()
    first.lock.release()
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0, status
    assert len(first.store) == (0 if accepted_only else 1)
print(json.dumps({"child": "destructor completed", "parent": "unchanged"}))
"""
    run = subprocess.run(
        [sys.executable, "-c", script, "accepted" if accepted_only else "generic"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert run.returncode == 0, run.stderr
    assert '"child": "destructor completed"' in run.stdout
