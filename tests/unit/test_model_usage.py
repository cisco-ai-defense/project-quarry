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

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import Barrier
from types import SimpleNamespace

import pytest

from quarry.integrations import model_usage
from quarry.integrations.model_usage import ModelUsage, model_call_budget
from quarry.telemetry import (
    ModelUsageLedger,
    elapsed_seconds,
    merge_model_usage,
    public_model_operation,
)
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
async def test_model_call_cost_limit_counts_attempts_per_alias_with_telemetry(monkeypatch) -> None:
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
    telemetry = ModelUsageLedger()

    with model_call_budget(2, telemetry):
        assert (
            await usage.run(
                "provider/model",
                retry_once,
                budget_alias="safety",
                operation_name="score",
                timeout=1.0,
            )
            == "ok"
        )
        with pytest.raises(LLMError) as exc_info:
            await usage.run(
                "provider/model",
                succeed,
                budget_alias="safety",
                operation_name="score",
            )
        assert exc_info.value.details == {
            "code": "model_call_limit",
            "model": "safety",
            "max_model_calls": 2,
        }
        assert (
            await usage.run(
                "provider/model",
                succeed,
                budget_alias="other",
                operation_name="complete",
            )
            == "ok"
        )

    assert telemetry.snapshot() == {
        "calls": 3,
        "records": [
            {"model": "other", "operation": "complete", "calls": 1},
            {"model": "safety", "operation": "score", "calls": 2},
        ],
        "truncated": False,
    }

    with model_call_budget(2):
        assert await usage.run("provider/model", succeed, budget_alias="safety") == "ok"


@pytest.mark.asyncio
async def test_unlimited_concurrent_calls_are_accounted_by_public_alias_and_operation() -> None:
    usage = ModelUsage()
    telemetry = ModelUsageLedger()

    async def succeed(_timeout: float | None) -> str:
        return "ok"

    with model_call_budget(None, telemetry):
        await asyncio.gather(
            usage.run("private-a", succeed, budget_alias="alpha", operation_name="score"),
            usage.run("private-b", succeed, budget_alias="beta", operation_name="classify"),
        )

    assert telemetry.snapshot() == {
        "calls": 2,
        "records": [
            {"model": "alpha", "operation": "score", "calls": 1},
            {"model": "beta", "operation": "classify", "calls": 1},
        ],
        "truncated": False,
    }


@pytest.mark.asyncio
async def test_concurrent_asyncio_tasks_keep_usage_ledgers_isolated() -> None:
    entered = 0
    both_entered = asyncio.Event()

    async def worker(alias: str) -> dict:
        ledger = ModelUsageLedger()

        async def overlap(_timeout: float | None) -> str:
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await both_entered.wait()
            return "ok"

        with model_call_budget(None, ledger):
            await ModelUsage().run(
                f"private-{alias}",
                overlap,
                budget_alias=alias,
                operation_name="score",
            )
        return ledger.snapshot()

    first, second = await asyncio.gather(worker("alpha"), worker("beta"))

    assert first["records"] == [{"model": "alpha", "operation": "score", "calls": 1}]
    assert second["records"] == [{"model": "beta", "operation": "score", "calls": 1}]


def test_concurrent_threads_keep_usage_ledgers_isolated() -> None:
    overlap = Barrier(2)

    def worker(alias: str) -> dict:
        ledger = ModelUsageLedger()

        async def attempt(_timeout: float | None) -> str:
            overlap.wait(timeout=5)
            return "ok"

        async def run() -> None:
            with model_call_budget(None, ledger):
                await ModelUsage().run(
                    f"private-{alias}",
                    attempt,
                    budget_alias=alias,
                    operation_name="classify",
                )

        asyncio.run(run())
        return ledger.snapshot()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(worker, "alpha")
        second = executor.submit(worker, "beta")
        reports = [first.result(timeout=10), second.result(timeout=10)]

    assert [report["calls"] for report in reports] == [1, 1]
    assert [[record["model"] for record in report["records"]] for report in reports] == [
        ["alpha"],
        ["beta"],
    ]


@pytest.mark.asyncio
async def test_failed_physical_attempt_is_counted_without_changing_the_error() -> None:
    usage = ModelUsage()
    telemetry = ModelUsageLedger()

    async def fail(_timeout: float | None) -> str:
        raise LLMError("synthetic failure", {"code": "invalid_output"})

    with model_call_budget(None, telemetry), pytest.raises(LLMError, match="synthetic failure"):
        await usage.run(
            "private-name",
            fail,
            budget_alias="public-alias",
            operation_name="score",
        )

    assert telemetry.snapshot() == {
        "calls": 1,
        "records": [{"model": "public-alias", "operation": "score", "calls": 1}],
        "truncated": False,
    }


@pytest.mark.asyncio
async def test_usage_counts_started_timeouts_and_cancellation_but_not_limiter_waits() -> None:
    provider_started = asyncio.Event()
    usage = ModelUsage()
    telemetry = ModelUsageLedger()

    async def pending(_timeout: float | None) -> str:
        provider_started.set()
        await asyncio.Future()
        return "unreachable"

    with model_call_budget(None, telemetry), pytest.raises(LLMError) as exc_info:
        await usage.run(
            "private-name",
            pending,
            budget_alias="public-alias",
            operation_name="score",
            timeout=0.001,
        )
    assert exc_info.value.details["code"] == "timeout"

    provider_started.clear()
    with model_call_budget(None, telemetry):
        task = asyncio.create_task(
            usage.run(
                "private-name",
                pending,
                budget_alias="public-alias",
                operation_name="score",
            )
        )
        await provider_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    limiter_entered = asyncio.Event()

    @asynccontextmanager
    async def blocked_limit(_bucket: str):
        limiter_entered.set()
        await asyncio.Future()
        yield 0.0

    waiting = ModelUsage(SimpleNamespace(limit=blocked_limit))
    with model_call_budget(None, telemetry):
        task = asyncio.create_task(
            waiting.run(
                "private-name",
                pending,
                budget_alias="waiting-alias",
                operation_name="score",
            )
        )
        await limiter_entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with model_call_budget(None, telemetry), pytest.raises(LLMError):
        await usage._attempt(
            "private-name",
            pending,
            deadline=0.0,
            budget_alias="expired-alias",
            operation_name="score",
        )

    assert telemetry.snapshot() == {
        "calls": 2,
        "records": [{"model": "public-alias", "operation": "score", "calls": 2}],
        "truncated": False,
    }


@pytest.mark.parametrize(
    ("adapter_method", "operation"),
    [
        ("score", "score"),
        ("classify", "classify"),
        ("classify_messages", "classify"),
        ("complete", "complete"),
        ("complete_message", "complete"),
        ("tokenize", "input_tokens"),
    ],
)
def test_adapter_methods_map_to_generic_operations(adapter_method, operation) -> None:
    assert public_model_operation(adapter_method) == operation


def test_usage_and_duration_bounds_saturate_without_negative_values(monkeypatch) -> None:
    from quarry import telemetry

    monkeypatch.setattr(telemetry, "MAX_MODEL_USAGE_CALLS", 2)
    monkeypatch.setattr(telemetry, "MAX_MODEL_USAGE_ALIASES", 1)
    ledger = ModelUsageLedger()

    ledger.record("alpha", "score", 2)
    ledger.record("beta", "complete")

    assert ledger.snapshot() == {
        "calls": 2,
        "records": [{"model": "alpha", "operation": "score", "calls": 2}],
        "truncated": True,
    }
    assert elapsed_seconds(2.0, 1.0) == 0.0
    assert elapsed_seconds(0.0, float("inf")) == telemetry.MAX_SCAN_SECONDS

    monkeypatch.setattr(telemetry, "MAX_MODEL_USAGE_CALLS", 10)
    monkeypatch.setattr(telemetry, "MAX_MODEL_USAGE_OPERATIONS_PER_ALIAS", 1)
    bounded = ModelUsageLedger()
    bounded.record("alpha", "score")
    bounded.record("alpha", "complete")
    bounded.record("beta", "score")

    assert bounded.snapshot() == {
        "calls": 3,
        "records": [{"model": "alpha", "operation": "score", "calls": 1}],
        "truncated": True,
    }

    invalid = ModelUsageLedger()
    invalid.record("alpha", "provider_payload")  # ty: ignore[invalid-argument-type]
    assert invalid.snapshot() == {"calls": 1, "records": [], "truncated": True}

    assert merge_model_usage(
        [
            {
                "calls": 2,
                "records": [{"model": "alpha", "operation": "score", "calls": 1}],
                "truncated": False,
            }
        ]
    ) == {
        "calls": 2,
        "records": [{"model": "alpha", "operation": "score", "calls": 1}],
        "truncated": True,
    }


def test_parallel_records_and_snapshots_remain_consistent() -> None:
    ledger = ModelUsageLedger()

    def record(alias):
        for _ in range(200):
            ledger.record(alias, "score")
            snapshot = ledger.snapshot()
            assert snapshot["calls"] == sum(row["calls"] for row in snapshot["records"])
            assert not snapshot["truncated"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(record, [f"model-{index}" for index in range(8)]))
    assert ledger.snapshot()["calls"] == 1600


@pytest.mark.asyncio
async def test_payload_preparation_failure_records_no_adapter_attempt(tmp_path) -> None:
    from quarry.config import QuarryConfig
    from quarry.integrations.model_inference import ModelInferenceBackend

    class Adapter:
        calls = 0

        async def score(self, *_args, **_kwargs):
            self.calls += 1
            return 0.8

    adapter = Adapter()
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"judge": {"adapter": "synthetic", "model": "synthetic"}}},
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(config, adapters={"synthetic": adapter})
    ledger = ModelUsageLedger()

    def invalid_payload():
        raise ValueError("synthetic payload preparation failure")

    with model_call_budget(None, ledger), pytest.raises(ValueError, match="payload preparation"):
        await backend._call(
            "judge",
            "synthetic",
            adapter,
            "score",
            config.models["judge"],
            invalid_payload,
            key=("payload-preparation-regression",),
            timeout=None,
        )
    assert adapter.calls == 0
    assert ledger.snapshot() == {"calls": 0, "records": [], "truncated": False}


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
