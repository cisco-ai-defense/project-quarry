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

"""The process cache keeps results and shares in-flight calls by key."""

import asyncio

import pytest

from quarry.cache.async_cache import AsyncFunctionCache


@pytest.mark.asyncio
async def test_evicting_idle_locks_keeps_cached_results() -> None:
    calls = []

    async def provider(value: int) -> int:
        calls.append(value)
        return value * 2

    cache = AsyncFunctionCache()

    results = [await cache.call(f"key-{i}".encode(), provider, i) for i in range(1, 5)]
    again = await cache.call(b"key-1", provider, 1)

    assert (results, again, calls) == ([2, 4, 6, 8], 2, [1, 2, 3, 4])


@pytest.mark.asyncio
async def test_a_contended_cache_still_shares_one_in_flight_call_per_key() -> None:
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
