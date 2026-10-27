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

"""Cancellation safety for LLM concurrency permits."""

import asyncio

import pytest

from quarry.integrations.rate_limiter import _BucketLimiter
from quarry.types.rate_limit import RateLimitConfig


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
