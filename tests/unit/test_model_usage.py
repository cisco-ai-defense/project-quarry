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


@pytest.mark.asyncio
async def test_retry_timeout_names_the_alias_and_last_completed_failure(monkeypatch) -> None:
    times = iter((0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 1.0))
    monkeypatch.setattr(model_usage, "monotonic", lambda: next(times))

    async def no_delay(_delay: float) -> None:
        return None

    failures = iter(("rate_limit", "connection"))
    attempts = 0

    async def fail(_timeout: float | None) -> str:
        nonlocal attempts
        attempts += 1
        raise LLMError("provider-controlled detail", {"code": next(failures)})

    monkeypatch.setattr(model_usage, "_sleep", no_delay)

    with pytest.raises(LLMError) as exc_info:
        await ModelUsage().run(
            "provider/physical-model",
            fail,
            budget_alias="guard",
            timeout=1.0,
        )

    assert attempts == 2
    assert exc_info.value.details == {
        "code": "timeout",
        "model": "guard",
        "last_failure": "connection",
    }
    assert str(exc_info.value) == (
        "Inference model 'guard' ran out of time after 2 failed attempts; "
        "last recorded failure: connection error"
    )
    assert "provider-controlled detail" not in str(exc_info.value)
    assert "physical-model" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_timeout_waiting_for_a_retry_slot_keeps_the_completed_attempt(
    monkeypatch,
) -> None:
    now = 0.0
    tickets = 0
    attempts = 0

    @asynccontextmanager
    async def limit(_bucket: str):
        nonlocal now, tickets
        tickets += 1
        if tickets == 2:
            now = 2.0
        yield 0.0

    async def no_delay(_delay: float) -> None:
        return None

    async def fail(_timeout: float | None) -> str:
        nonlocal attempts
        attempts += 1
        raise LLMError("provider-controlled detail", {"code": "connection"})

    monkeypatch.setattr(model_usage, "monotonic", lambda: now)
    monkeypatch.setattr(model_usage, "_sleep", no_delay)
    usage = ModelUsage(SimpleNamespace(limit=limit))

    with model_call_budget(2) as budget, pytest.raises(LLMError) as exc_info:
        await usage.run(
            "provider/physical-model",
            fail,
            budget_alias="guard",
            timeout=1.0,
        )

    assert tickets == 2
    assert attempts == 1
    assert budget is not None
    assert budget.calls_by_alias == {"guard": 1}
    assert exc_info.value.details == {
        "code": "timeout",
        "model": "guard",
        "last_failure": "connection",
    }
    assert str(exc_info.value) == (
        "Inference model 'guard' ran out of time after 1 failed attempt; "
        "last recorded failure: connection error"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("deadline_owner", ["model", "rule", "unlimited"])
@pytest.mark.parametrize("preparation_seconds", [0.25, 2.0])
@pytest.mark.parametrize("threaded", [False, True])
async def test_payload_preparation_preserves_remaining_deadline_before_adapter_start(
    tmp_path, monkeypatch, deadline_owner, preparation_seconds, threaded
):
    import asyncio
    from contextlib import asynccontextmanager, nullcontext
    from threading import get_ident
    from types import SimpleNamespace

    from quarry.async_runtime import run_engine
    from quarry.config import QuarryConfig
    from quarry.integrations import model_usage
    from quarry.integrations.model_inference import ModelInferenceBackend
    from quarry.integrations.model_usage import model_call_budget
    from quarry.safety import timer
    from quarry.safety.timer import QuarryTimeoutError, evaluation_deadline
    from quarry.telemetry import ModelUsageLedger
    from quarry.types.exceptions import LLMError

    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(model_usage, "monotonic", lambda: clock.now)
    monkeypatch.setattr(timer, "time", SimpleNamespace(monotonic=lambda: clock.now))
    host = get_ident(), asyncio.get_running_loop()
    observed = []
    adapter_timeouts = []

    def record(kind):
        observed.append((kind, get_ident(), asyncio.get_running_loop()))

    @asynccontextmanager
    async def slot(_model):
        record("limiter")
        yield

    class Adapter:
        async def score(self, _config, *, timeout=None):
            record("adapter")
            adapter_timeouts.append(timeout)
            return 0.8

    adapter = Adapter()
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"reader": {"adapter": "synthetic", "model": "synthetic"}}},
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(
        config, adapters={"synthetic": adapter}, rate_limiter=SimpleNamespace(limit=slot)
    )

    def payload():
        record("payload")
        clock.now += preparation_seconds
        return ()

    def request():
        return backend._call(
            "reader",
            "synthetic",
            adapter,
            "score",
            config.models["reader"],
            payload,
            key=("preparation-budget-regression", deadline_owner, preparation_seconds, threaded),
            timeout=1.0 if deadline_owner == "model" else None,
        )

    async def invoke():
        return await run_engine(request) if threaded else await request()

    deadline = (
        evaluation_deadline(1.0, "synthetic payload deadline")
        if deadline_owner == "rule"
        else nullcontext()
    )
    ledger = ModelUsageLedger()
    expired = deadline_owner != "unlimited" and preparation_seconds > 1.0
    with model_call_budget(1, ledger) as budget:
        if expired:
            error_type = QuarryTimeoutError if deadline_owner == "rule" else LLMError
            with pytest.raises(error_type) as error, deadline:
                await invoke()
            if deadline_owner == "model":
                assert error.value.details == {"code": "timeout", "model": "reader"}
            else:
                assert str(error.value) == "synthetic payload deadline"
            assert adapter_timeouts == []
            assert budget.calls_by_alias == {}
            assert ledger.snapshot() == {"calls": 0, "records": [], "truncated": False}
            assert [kind for kind, *_ in observed] == ["limiter", "payload"]
        else:
            with deadline:
                assert await invoke() == 0.8
            if deadline_owner == "unlimited":
                assert adapter_timeouts == [None]
            else:
                assert adapter_timeouts == pytest.approx([1.0 - preparation_seconds])
            assert budget.calls_by_alias == {"reader": 1}
            assert ledger.snapshot() == {
                "calls": 1,
                "records": [{"model": "reader", "operation": "score", "calls": 1}],
                "truncated": False,
            }
            assert [kind for kind, *_ in observed] == ["limiter", "payload", "adapter"]
    assert all((thread, loop) == host for _, thread, loop in observed)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_failure", ["builtin", "structured"])
@pytest.mark.parametrize("threaded", [False, True])
async def test_expired_requesting_rule_keeps_timeout_ownership_for_shared_work(
    tmp_path, monkeypatch, provider_failure, threaded
):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from quarry.async_runtime import run_engine
    from quarry.config import QuarryConfig
    from quarry.integrations import model_usage
    from quarry.integrations.model_inference import ModelInferenceBackend
    from quarry.integrations.model_usage import model_call_budget
    from quarry.safety import timer
    from quarry.safety.timer import QuarryTimeoutError, evaluation_deadline
    from quarry.telemetry import ModelUsageLedger
    from quarry.types.exceptions import LLMError

    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(model_usage, "monotonic", lambda: clock.now)
    monkeypatch.setattr(timer, "time", SimpleNamespace(monotonic=lambda: clock.now))
    calls = []

    class Adapter:
        async def score(self, _config, *, timeout=None):
            calls.append(timeout)
            clock.now += 2.0
            if provider_failure == "builtin":
                raise TimeoutError("synthetic provider timeout")
            raise LLMError("synthetic provider timeout", {"code": "timeout"})

    @asynccontextmanager
    async def slot(_model):
        yield

    adapter = Adapter()
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"reader": {"adapter": "synthetic", "model": "synthetic"}}},
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(
        config, adapters={"synthetic": adapter}, rate_limiter=SimpleNamespace(limit=slot)
    )

    def request():
        return backend._call(
            "reader",
            "synthetic",
            adapter,
            "score",
            config.models["reader"],
            lambda: (),
            key=("shared-request-timeout-owner", provider_failure, threaded),
            timeout=None,
        )

    ledger = ModelUsageLedger()
    with model_call_budget(1, ledger) as budget:
        with (
            pytest.raises(QuarryTimeoutError, match="requesting rule deadline"),
            evaluation_deadline(1.0, "requesting rule deadline"),
        ):
            if threaded:
                await run_engine(request)
            else:
                await request()
        assert calls == pytest.approx([1.0])
        assert budget.calls_by_alias == {"reader": 1}
        assert ledger.snapshot() == {
            "calls": 1,
            "records": [{"model": "reader", "operation": "score", "calls": 1}],
            "truncated": False,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("completion_stage", ["adapter", "limiter_cleanup"])
@pytest.mark.parametrize("deadline_owner", ["model", "rule", "unlimited"])
@pytest.mark.parametrize("operation_seconds", [0.25, 2.0])
@pytest.mark.parametrize("threaded", [False, True])
async def test_successful_adapter_and_limiter_cleanup_keep_logical_deadline(
    tmp_path, monkeypatch, completion_stage, deadline_owner, operation_seconds, threaded
):
    from contextlib import asynccontextmanager, nullcontext
    from types import SimpleNamespace

    from quarry.async_runtime import run_engine
    from quarry.cache.async_cache import result_cache
    from quarry.config import QuarryConfig
    from quarry.integrations import model_usage
    from quarry.integrations.model_inference import ModelInferenceBackend
    from quarry.integrations.model_usage import model_call_budget
    from quarry.safety import timer
    from quarry.safety.timer import QuarryTimeoutError, evaluation_deadline
    from quarry.telemetry import ModelUsageLedger
    from quarry.types.exceptions import LLMError
    from quarry.types.messages import Messages
    from tests.unit.model_fakes import model_entry, passthrough_builder

    result_cache.clear()
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(model_usage, "monotonic", lambda: clock.now)
    monkeypatch.setattr(timer, "time", SimpleNamespace(monotonic=lambda: clock.now))
    timeouts = []

    @asynccontextmanager
    async def limit(_model):
        yield
        if completion_stage == "limiter_cleanup":
            clock.now += operation_seconds

    class Adapter:
        async def score(self, _model, _messages, *, timeout=None):
            timeouts.append(timeout)
            if completion_stage == "adapter":
                clock.now += operation_seconds
            return 0.8

    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"reader": model_entry(score=True)}},
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(
        config,
        adapters={"fake": Adapter()},
        message_builders={"test": passthrough_builder},
        rate_limiter=SimpleNamespace(limit=limit),
    )
    messages = Messages.from_openai([{"role": "user", "content": "synthetic"}])

    def request():
        return backend.model_score_messages(
            "reader",
            "synthetic policy",
            messages,
            timeout=1.0 if deadline_owner == "model" else None,
        )

    async def invoke():
        return await run_engine(request) if threaded else await request()

    scope = (
        evaluation_deadline(1.0, "synthetic public rule deadline")
        if deadline_owner == "rule"
        else nullcontext()
    )
    expired = deadline_owner != "unlimited" and operation_seconds > 1.0
    ledger = ModelUsageLedger()
    with model_call_budget(1, ledger) as budget:
        if expired:
            error_type = LLMError if deadline_owner == "model" else QuarryTimeoutError
            with pytest.raises(error_type) as caught, scope:
                await invoke()
            if deadline_owner == "model":
                assert caught.value.details == {"code": "timeout", "model": "reader"}
            else:
                assert str(caught.value) == "synthetic public rule deadline"
        else:
            with scope:
                assert await invoke() == 0.8
        assert budget.calls_by_alias == {"reader": 1}
        assert timeouts == ([None] if deadline_owner == "unlimited" else pytest.approx([1.0]))
        assert ledger.snapshot() == {
            "calls": 1,
            "records": [{"model": "reader", "operation": "score", "calls": 1}],
            "truncated": False,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("threaded", [False, True])
async def test_expired_failed_payload_keeps_original_error_and_no_attempt(
    tmp_path, monkeypatch, threaded
):
    from types import SimpleNamespace

    from quarry.async_runtime import run_engine
    from quarry.config import QuarryConfig
    from quarry.integrations import model_usage
    from quarry.integrations.model_inference import ModelInferenceBackend
    from quarry.integrations.model_usage import model_call_budget
    from quarry.telemetry import ModelUsageLedger

    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(model_usage, "monotonic", lambda: clock.now)
    error = ValueError("synthetic failed payload after budget")
    adapter_calls = []

    class Adapter:
        async def score(self, _config, *, timeout=None):
            adapter_calls.append(timeout)
            return 0.8

    adapter = Adapter()
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"reader": {"adapter": "synthetic", "model": "synthetic"}}},
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(config, adapters={"synthetic": adapter})

    def payload():
        clock.now += 2.0
        raise error

    def request():
        return backend._call(
            "reader",
            "synthetic",
            adapter,
            "score",
            config.models["reader"],
            payload,
            key=("expired-failed-payload", threaded),
            timeout=1.0,
        )

    ledger = ModelUsageLedger()
    with model_call_budget(1, ledger) as budget, pytest.raises(ValueError) as caught:
        if threaded:
            await run_engine(request)
        else:
            await request()
    assert caught.value is error
    assert budget.calls_by_alias == {} and adapter_calls == []
    assert ledger.snapshot() == {"calls": 0, "records": [], "truncated": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("completion_stage", ["adapter", "limiter_cleanup"])
@pytest.mark.parametrize("threaded", [False, True])
async def test_late_success_is_not_cached_and_next_funded_score_can_complete(
    tmp_path, monkeypatch, completion_stage, threaded
):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from quarry.async_runtime import run_engine
    from quarry.cache.async_cache import result_cache
    from quarry.config import QuarryConfig
    from quarry.integrations import model_usage
    from quarry.integrations.model_inference import ModelInferenceBackend
    from quarry.integrations.model_usage import model_call_budget
    from quarry.telemetry import ModelUsageLedger
    from quarry.types.exceptions import LLMError
    from quarry.types.messages import Messages
    from tests.unit.model_fakes import model_entry, passthrough_builder

    result_cache.clear()
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(model_usage, "monotonic", lambda: clock.now)
    timeouts = []

    @asynccontextmanager
    async def limit(_model):
        yield
        if completion_stage == "limiter_cleanup" and len(timeouts) == 1:
            clock.now += 2.0

    class Adapter:
        async def score(self, _model, _messages, *, timeout=None):
            timeouts.append(timeout)
            if len(timeouts) == 1:
                if completion_stage == "adapter":
                    clock.now += 2.0
                return 0.8
            return 0.6

    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"reader": model_entry(score=True)}},
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(
        config,
        adapters={"fake": Adapter()},
        message_builders={"test": passthrough_builder},
        rate_limiter=SimpleNamespace(limit=limit),
    )
    messages = Messages.from_openai([{"role": "user", "content": "synthetic"}])

    def request():
        return backend.model_score_messages("reader", "synthetic policy", messages, timeout=1.0)

    async def invoke():
        return await run_engine(request) if threaded else await request()

    ledger = ModelUsageLedger()
    with model_call_budget(2, ledger) as budget:
        with pytest.raises(LLMError) as caught:
            await invoke()
        assert caught.value.details == {"code": "timeout", "model": "reader"}
        assert await invoke() == 0.6
        assert await invoke() == 0.6
    assert timeouts == pytest.approx([1.0, 1.0])
    assert budget.calls_by_alias == {"reader": 2}
    assert ledger.snapshot() == {
        "calls": 2,
        "records": [{"model": "reader", "operation": "score", "calls": 2}],
        "truncated": False,
    }
