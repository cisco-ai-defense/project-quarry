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

"""Async rate limiter for LLM integrations."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from time import monotonic

from quarry.safety.timer import QuarryTimeoutError, current_evaluation_deadline
from quarry.types.rate_limit import RateLimitConfig


class _BucketLimiter:
    """Combine throughput and concurrency limits for one model API."""

    def __init__(self, config: RateLimitConfig):
        self.requests_per_interval = (
            config.requests_per_interval
            if config.requests_per_interval and config.requests_per_interval > 0
            else None
        )
        self.interval_seconds = (
            config.interval_seconds
            if config.interval_seconds and config.interval_seconds > 0
            else 0.0
        )
        self.max_concurrent = (
            config.max_concurrent if config.max_concurrent and config.max_concurrent > 0 else None
        )

        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()
        self._active = 0
        self._waiters: set[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = set()

    async def acquire(self) -> float:
        """Acquire allowance for a call and return total time spent waiting."""

        started = monotonic()
        loop = asyncio.get_running_loop()
        wake = asyncio.Event()
        with self._lock:
            self._waiters.add((loop, wake))
        try:
            while True:
                with self._lock:
                    wake.clear()
                    now = monotonic()
                    deadline = current_evaluation_deadline()
                    deadline_remaining = None
                    if deadline is not None:
                        deadline_remaining = deadline.expires_at - now
                        if deadline_remaining <= 0:
                            raise QuarryTimeoutError(deadline.error_message)

                    self._prune_expired(now)
                    concurrency_open = (
                        self.max_concurrent is None or self._active < self.max_concurrent
                    )
                    rate_open = (
                        self.requests_per_interval is None
                        or len(self._timestamps) < self.requests_per_interval
                    )
                    if concurrency_open and rate_open:
                        self._active += 1
                        if self.requests_per_interval is not None:
                            self._timestamps.append(now)
                        return now - started

                    wait_for = None
                    if not rate_open:
                        wait_for = max(
                            0.0,
                            self.interval_seconds - (now - self._timestamps[0]),
                        )
                        # A rate window that opens after the deadline cannot yield a usable
                        # ticket, so fail without extending the operation.
                        if (
                            deadline is not None
                            and deadline_remaining is not None
                            and wait_for >= deadline_remaining
                        ):
                            raise QuarryTimeoutError(deadline.error_message)

                    deadline_error = None
                    # A concurrency slot has no predictable release time. Bound that wait by
                    # the active deadline and preserve its safe, caller-owned error.
                    if wait_for is None and deadline is not None and deadline_remaining is not None:
                        wait_for = deadline_remaining
                        deadline_error = deadline.error_message

                if wait_for is None:
                    await wake.wait()
                else:
                    try:
                        await asyncio.wait_for(wake.wait(), timeout=wait_for)
                    except TimeoutError:
                        if deadline_error is not None:
                            raise QuarryTimeoutError(deadline_error) from None
        finally:
            with self._lock:
                self._waiters.discard((loop, wake))

    def release(self) -> None:
        """Release any concurrency slot held."""

        with self._lock:
            if self._active <= 0:
                raise RuntimeError("rate limiter release without a matching acquire")
            self._active -= 1
            waiters = tuple(self._waiters)
        for loop, wake in waiters:
            try:
                loop.call_soon_threadsafe(wake.set)
            except RuntimeError:
                # A cancelled scan may close its loop before a different loop releases.
                continue

    def _prune_expired(self, now: float) -> None:
        """Drop timestamps outside the rolling window."""

        if not self._timestamps:
            return

        cutoff = now - self.interval_seconds
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()


class LLMRateLimiter:
    """Shared rate limiter with one independent bucket per upstream model."""

    def __init__(self, limits: Mapping[str, RateLimitConfig]):
        self._limits_config = limits
        self._limiters: dict[str, _BucketLimiter] = {}
        self._lock = threading.Lock()

    @asynccontextmanager
    async def limit(self, model: str) -> AsyncGenerator[float, None]:
        """Enforce the limits configured for an upstream model API."""

        limiter = self._get_or_create_limiter(model)
        wait_time = await limiter.acquire()

        try:
            yield wait_time
        finally:
            limiter.release()

    def _get_or_create_limiter(self, model: str) -> _BucketLimiter:
        """Return the model bucket, applying provider and default fallbacks."""

        with self._lock:
            if model not in self._limiters:
                config = self._limits_config.get(model)
                if config is None and "/" in model:
                    config = self._limits_config.get(model.split("/", 1)[0].lower())
                if config is None:
                    config = self._limits_config.get("default", RateLimitConfig())
                self._limiters[model] = _BucketLimiter(config)

            return self._limiters[model]
