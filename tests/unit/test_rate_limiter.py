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

"""Deadline and cancellation safety for LLM limiter permits."""

import asyncio

import pytest

from quarry.integrations.rate_limiter import _BucketLimiter
from quarry.safety.timer import QuarryTimeoutError, evaluation_deadline, scan_deadline
from quarry.types.rate_limit import RateLimitConfig


@pytest.mark.asyncio
async def test_throughput_wait_past_the_evaluation_deadline_fails_without_a_ticket() -> None:
    limiter = _BucketLimiter(RateLimitConfig(requests_per_interval=1, interval_seconds=10))
    await limiter.acquire()
    limiter.release()

    with (
        pytest.raises(QuarryTimeoutError, match="model operation deadline"),
        evaluation_deadline(0.1, "model operation deadline"),
    ):
        await asyncio.wait_for(limiter.acquire(), timeout=0.5)

    limiter.requests_per_interval = None
    await asyncio.wait_for(limiter.acquire(), timeout=0.1)
    limiter.release()


@pytest.mark.asyncio
async def test_concurrency_wait_stops_at_the_evaluation_deadline() -> None:
    limiter = _BucketLimiter(RateLimitConfig(max_concurrent=1))
    await limiter.acquire()

    with (
        pytest.raises(QuarryTimeoutError, match="model operation deadline"),
        evaluation_deadline(0.01, "model operation deadline"),
    ):
        await asyncio.wait_for(limiter.acquire(), timeout=0.5)

    limiter.release()
    await asyncio.wait_for(limiter.acquire(), timeout=0.1)
    limiter.release()


@pytest.mark.asyncio
async def test_already_expired_deadline_takes_no_open_ticket() -> None:
    limiter = _BucketLimiter(RateLimitConfig(requests_per_interval=1, max_concurrent=1))
    with (
        scan_deadline(0, "already expired"),
        pytest.raises(QuarryTimeoutError, match="already expired"),
    ):
        await limiter.acquire()
    assert limiter._active == 0
    assert list(limiter._timestamps) == []
    assert not limiter._waiters
    await asyncio.wait_for(limiter.acquire(), timeout=0.1)
    limiter.release()


@pytest.mark.asyncio
async def test_concurrency_slot_released_before_deadline_remains_usable() -> None:
    limiter = _BucketLimiter(RateLimitConfig(max_concurrent=1))
    await limiter.acquire()

    async def waiting():
        with evaluation_deadline(1, "waiting deadline"):
            return await limiter.acquire()

    task = asyncio.create_task(waiting())
    await asyncio.sleep(0)
    assert limiter._waiters
    assert not task.done()
    limiter.release()
    assert await asyncio.wait_for(task, timeout=0.5) >= 0
    assert limiter._active == 1
    assert not limiter._waiters
    limiter.release()


def test_cancellation_during_wait_releases_shared_bucket_across_event_loops() -> None:
    limiter = _BucketLimiter(
        RateLimitConfig(
            requests_per_interval=1,
            interval_seconds=10,
            max_concurrent=1,
        )
    )

    async def cancel_waiter() -> None:
        await limiter.acquire()
        waiting = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        limiter.requests_per_interval = None
        limiter.release()

    async def reuse_from_a_fresh_loop() -> None:
        await asyncio.wait_for(limiter.acquire(), timeout=0.1)
        limiter.release()

    asyncio.run(cancel_waiter())
    asyncio.run(reuse_from_a_fresh_loop())
