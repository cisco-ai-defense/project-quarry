# Copyright 2026 Cisco Systems, Inc. and its affiliates
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

"""Async cache for expensive function calls."""

import asyncio
import concurrent.futures
import inspect
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from quarry.cache.constants import RESULT_CACHE_MAX_ENTRIES, RESULT_CACHE_TTL_SECONDS
from quarry.safety.timer import (
    QuarryTimeoutError,
    current_evaluation_deadline,
    raise_if_evaluation_expired,
    remaining_evaluation_time,
    without_rule_deadline,
)

_MISSING = object()

# The scan a call runs for, so a caller can tell another scan's failed call from its own.
_scan = ContextVar[object]("quarry_scan", default=None)


@contextmanager
def scan_scope(scan: object = None) -> Generator[object, None, None]:
    """Run one scan, or resume ``scan`` (as its deferred evidence does): the calls it starts
    are its own."""

    scan = object() if scan is None else scan
    token = _scan.set(scan)
    try:
        yield scan
    finally:
        _scan.reset(token)


def current_scan() -> object:
    """Return the running scan, for resuming it with ``scan_scope``."""

    return _scan.get()


class SlidingTTLCache(MutableMapping):
    """Results that expire ``ttl`` seconds after their last use; past ``maxsize`` results the
    least recently used is dropped.

    Every read and write sets an entry's expiry again, ``ttl`` from now, and makes it the most
    recent, so with one time-to-live the least recently used result is also the next to
    expire: one ordered dict keeps both orders, and each operation is O(1). A result whose type
    sets ``cache_lifetime`` instead expires that many seconds (at most ``ttl``) after it is
    stored, however often it is read; until it is read again or reaches the front, an expired
    one can wait behind newer results. A call still running is neither expired nor dropped;
    its time-to-live starts with its result. It is not thread-safe: every caller holds the
    cache's lock.
    """

    def __init__(self, maxsize: int, ttl: float, timer: Callable[[], float] = time.monotonic):
        self.maxsize = maxsize
        self.ttl = ttl
        self.timer = timer
        # Each key's value and when it expires, least recently used first.
        self._entries: OrderedDict[Any, tuple[Any, float]] = OrderedDict()
        # How many entries are running calls, which ``maxsize`` does not count.
        self._running = 0

    def _expiry(self, value: Any, now: float) -> float:
        if isinstance(value, _Pending):
            return math.inf
        return now + min(getattr(type(value), "cache_lifetime", self.ttl), self.ttl)

    def __getitem__(self, key: Any) -> Any:
        value, expires = self._entries[key]
        now = self.timer()
        if now >= expires:
            del self._entries[key]
            raise KeyError(key)
        if not hasattr(type(value), "cache_lifetime"):
            expires = self._expiry(value, now)
        self._entries[key] = (value, expires)
        self._entries.move_to_end(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        now = self.timer()
        previous = self._entries.get(key)
        if previous is not None and isinstance(previous[0], _Pending):
            self._running -= 1
        if isinstance(value, _Pending):
            self._running += 1
        self._entries[key] = (value, self._expiry(value, now))
        self._entries.move_to_end(key)
        self._trim(now)

    def _trim(self, now: float) -> None:
        """Drop expired results, and the least recently used past ``maxsize``, oldest first;
        a running call is moved past them instead."""

        entries = self._entries
        skipped = 0
        while skipped < len(entries):
            key, (value, expires) = next(iter(entries.items()))
            if isinstance(value, _Pending):
                entries.move_to_end(key)
                skipped += 1
            elif now >= expires or len(entries) - self._running > self.maxsize:
                del entries[key]
            else:
                return

    def __delitem__(self, key: Any) -> None:
        value, _ = self._entries.pop(key)
        if isinstance(value, _Pending):
            self._running -= 1

    def __iter__(self) -> Iterator[Any]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._running = 0


@dataclass
class _Pending:
    """A call still running, kept under its key until its result replaces it. Its future is
    thread-safe, so a caller on any thread or event loop waits on it."""

    key: Any
    task: asyncio.Task[Any]
    future: concurrent.futures.Future
    # The scan and budget the call runs for; other callers retry a failed call under their
    # own deadline and budget.
    owner: object
    waiters: int = 0


class _Abandoned(Exception):
    """A shared call was cancelled before it finished: its event loop ended, as
    ``asyncio.run`` does when the scan that started it returns, or the call cancelled
    itself."""


class AsyncFunctionCache:
    """Results of expensive calls by key, with each in-flight call shared by its callers.

    The process keeps one, ``result_cache``, for every result Quarry computes from its inputs:
    model responses and regex results. Its store is a ``SlidingTTLCache``, whose results
    expire a time-to-live after their last use. Each scan runs in ``scan_scope()``, which
    names the scan its calls run for.

    Results, and calls still running, serve every thread and event loop that runs scans, as
    the synchronous APIs do with ``asyncio.run``.
    """

    def __init__(self, *, store: MutableMapping[bytes, Any] | None = None):
        """
        Initialize the cache.

        Args:
            store: Where results are kept; None keeps them in a dict for the cache's life
        """
        self._store: MutableMapping[bytes, Any] = store if store is not None else {}
        # cachetools stores are not thread-safe, and a store may serve several threads.
        self._lock = threading.Lock()

    @classmethod
    def expiring(cls, *, max_entries: int, ttl_seconds: float) -> "AsyncFunctionCache":
        """Return a cache that keeps at most ``max_entries`` results, each until
        ``ttl_seconds`` after its last use."""

        store: SlidingTTLCache = SlidingTTLCache(maxsize=max_entries, ttl=ttl_seconds)
        return cls(store=store)

    @property
    def store(self) -> MutableMapping[bytes, Any]:
        """The results, by the digest of their inputs."""

        return self._store

    @property
    def lock(self) -> threading.Lock:
        """The lock every reader and writer of ``store`` holds."""

        return self._lock

    def configure(self, *, max_entries: int, ttl_seconds: float) -> None:
        """Bound the cache; new bounds drop what it holds. A long-lived host calls it at
        startup."""

        with self._lock:
            store = self._store
            if isinstance(store, SlidingTTLCache) and (store.maxsize, store.ttl) == (
                max_entries,
                ttl_seconds,
            ):
                return
            self._store = SlidingTTLCache(maxsize=max_entries, ttl=ttl_seconds)

    def clear(self) -> None:
        """Drop every result."""

        with self._lock:
            self._store.clear()

    async def call(
        self,
        key: bytes,
        func: Callable,
        *args,
        budget: object = None,
        alone: bool = False,
        retry_on: Callable[[BaseException], bool] | None = None,
        **kwargs,
    ) -> Any:
        """Return ``func(*args, **kwargs)``, kept under ``key``, the digest of every
        input of the result (see ``quarry.cache.keys.input_key``).

        Concurrent callers of one key, on any thread or event loop, share one call: the first
        keeps a ``_Pending`` under the key, and its result replaces it. The call is a task the
        cache starts on the first caller's loop. It does not inherit a caller's rule deadline,
        but it retains the inspection deadline, so a caller that stops waiting cannot cancel
        work another still needs; the last one to stop cancels it. It runs under the deadline and
        model-call budget of the caller that started it, named by the scan (see
        ``scan_scope``), the caller's deadline and ``budget`` (a model alias, whose calls are
        limited apart). A shared call another scan, rule deadline or budget started that fails
        with an error ``retry_on`` ties
        to the starter's limits, such as its deadline, or whose loop ended first, is retried
        until a call this caller started fails; any other failure reaches every waiter. An
        ``alone`` call is not joined by other scans while it runs, since its speed is its own
        scan's; its result is shared like any other.
        """

        # A call outside any scan, as a host calling the backend directly makes, is its own
        # owner, so it retries what another caller's limits ended. Each rule of a scan has its
        # own deadline, so a call another rule's deadline ended is retried under this one's.
        scan = _scan.get()
        owner = (object() if scan is None else scan, budget, current_evaluation_deadline())
        # A request-backed call is shared only within its scan, by every alias that sends it.
        pending_key: Any = (owner[0], key) if alone else key
        while True:
            with self._lock:
                value = self._store.get(key, _MISSING)
                if value is not _MISSING and not isinstance(value, _Pending):
                    return value
                pending = self._store.get(pending_key)
                if pending is None:
                    pending = self._start(key, pending_key, owner, func, args, kwargs)
                pending.waiters += 1
            try:
                return await self._wait(pending)
            except Exception:
                # Only the shared call's own failure is retried; a wait this caller's
                # deadline ends is not.
                error = pending.future.exception() if pending.future.done() else None
                # Another owner's cancelled call is made again; this owner's own is not, as
                # it would cancel again.
                if (
                    error is None
                    or pending.owner == owner
                    or not isinstance(error, _Abandoned)
                    and (retry_on is None or not retry_on(error))
                ):
                    raise
                # A caller whose own deadline has passed does not start the call again.
                raise_if_evaluation_expired()
            finally:
                await self._leave(pending)

    def forget(self, key: bytes) -> None:
        """Drop the result kept under ``key``, so the next caller calls again; a call
        running under it is kept."""

        with self._lock:
            if not isinstance(self._store.get(key), _Pending):
                self._store.pop(key, None)

    def _start(
        self,
        key: bytes,
        pending_key: Any,
        owner: object,
        func: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> _Pending:
        """Start the call for ``key`` on the running loop and keep it under ``pending_key``;
        the caller holds the lock."""

        with without_rule_deadline():
            task = asyncio.get_running_loop().create_task(call_async_or_sync(func, *args, **kwargs))
        # asyncio keeps only a weak reference to a task; the pending entry and its waiters
        # hold this one.
        pending = _Pending(pending_key, task, concurrent.futures.Future(), owner)
        # Running, the future cannot be cancelled by a caller that stops waiting.
        pending.future.set_running_or_notify_cancel()
        self._store[pending_key] = pending

        def finish(task: asyncio.Task[Any]) -> None:
            failure = (
                _Abandoned("the shared call was cancelled before it finished")
                if task.cancelled()
                else task.exception()
            )
            with self._lock:
                if self._store.get(pending_key) is pending:
                    del self._store[pending_key]
                if failure is None:
                    self._store[key] = task.result()
            if failure is None:
                pending.future.set_result(task.result())
            else:
                pending.future.set_exception(failure)

        task.add_done_callback(finish)
        return pending

    async def _leave(self, pending: _Pending) -> None:
        """Release one waiter; the last one drops the call and cancels it on its loop."""

        task = pending.task
        with self._lock:
            pending.waiters -= 1
            if pending.waiters > 0 or task.done():
                return
            if self._store.get(pending.key) is pending:
                del self._store[pending.key]
        loop = task.get_loop()
        if loop is asyncio.get_running_loop():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        elif not loop.is_closed():
            loop.call_soon_threadsafe(task.cancel)

    @staticmethod
    async def _wait(pending: _Pending) -> Any:
        """Wait for a shared call no longer than the caller's own deadline allows."""

        deadline = current_evaluation_deadline()
        remaining = remaining_evaluation_time()
        result = asyncio.wrap_future(pending.future)
        if remaining is None:
            return await result
        timeout = asyncio.timeout(remaining)
        try:
            async with timeout:
                return await result
        except TimeoutError as exc:
            if timeout.expired() and deadline is not None:
                raise QuarryTimeoutError(deadline.error_message) from exc
            raise


async def call_async_or_sync(func: Callable, *args, **kwargs) -> Any:
    """Call function whether it's sync or async."""
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return func(*args, **kwargs)


result_cache = AsyncFunctionCache.expiring(
    max_entries=RESULT_CACHE_MAX_ENTRIES, ttl_seconds=RESULT_CACHE_TTL_SECONDS
)
