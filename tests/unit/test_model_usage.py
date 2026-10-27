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

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from quarry.integrations import model_usage
from quarry.integrations.model_usage import ModelUsage, model_call_budget
from quarry.types.exceptions import LLMError


@pytest.mark.asyncio
async def test_model_usage_retries_with_one_limiter_ticket_per_attempt(monkeypatch) -> None:
    tickets: list[str] = []
    limiter_active = False
    timeouts: list[float] = []

    @asynccontextmanager
    async def limit(bucket: str):
        nonlocal limiter_active
        tickets.append(bucket)
        limiter_active = True
        try:
            yield 0.0
        finally:
            limiter_active = False

    async def no_delay(_delay: float) -> None:
        assert limiter_active is False

    async def operation(timeout: float | None) -> str:
        assert timeout is not None
        assert limiter_active is True
        timeouts.append(timeout)
        if len(timeouts) == 1:
            raise LLMError("try again", {"code": "rate_limit"})
        return "ok"

    monkeypatch.setattr(model_usage, "_sleep", no_delay)
    usage = ModelUsage(SimpleNamespace(limit=limit))

    assert await usage.run("provider/model", operation, timeout=1.0) == "ok"
    assert tickets == ["provider/model", "provider/model"]
    assert 0 < timeouts[1] < timeouts[0] <= 1.0

    async def unavailable(_timeout: float | None) -> str:
        raise LLMError("try again", {"code": "connection"})

    with pytest.raises(LLMError) as exc_info:
        await usage.run("provider/model", unavailable, timeout=0.01)
    assert exc_info.value.details["code"] == "timeout"


@pytest.mark.asyncio
async def test_model_call_cost_limit_counts_attempts_per_alias(monkeypatch) -> None:
    attempts = 0

    async def no_delay(_delay: float) -> None:
        return None

    async def retry_once(_timeout: float | None) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise LLMError("try again", {"code": "connection"})
        return "ok"

    async def succeed(_timeout: float | None) -> str:
        return "ok"

    monkeypatch.setattr(model_usage, "_sleep", no_delay)
    usage = ModelUsage()

    with model_call_budget(2):
        assert (
            await usage.run("provider/model", retry_once, budget_alias="safety", timeout=1.0)
            == "ok"
        )
        with pytest.raises(LLMError) as exc_info:
            await usage.run("provider/model", succeed, budget_alias="safety")
        assert exc_info.value.details == {
            "code": "model_call_limit",
            "model": "safety",
            "max_model_calls": 2,
        }
        assert await usage.run("provider/model", succeed, budget_alias="other") == "ok"

    with model_call_budget(2):
        assert await usage.run("provider/model", succeed, budget_alias="safety") == "ok"
