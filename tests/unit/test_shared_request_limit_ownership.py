# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""A host's failure does not authorize spending another shared request."""

import asyncio
import contextvars
from types import SimpleNamespace

import pytest

from quarry.cache.async_cache import AsyncFunctionCache, scan_scope
from quarry.config import LocalTransformConfig, QuarryConfig
from quarry.integrations import model_inference as inference
from quarry.integrations.model_usage import ModelCallBudget, ModelUsage
from quarry.integrations.request_limits import is_owned_request_limit
from quarry.safety.timer import QuarryTimeoutError
from quarry.types.adapter_transforms import TransformCacheIdentity
from quarry.types.exceptions import LLMError
from quarry.types.messages import Messages


@pytest.fixture
def cache(monkeypatch):
    cache = AsyncFunctionCache.expiring(max_entries=100, ttl_seconds=60)
    monkeypatch.setattr(inference, "result_cache", cache)
    return cache


def host_failure(kind):
    return {
        "timeout": TimeoutError("synthetic host failure"),
        "evaluation": QuarryTimeoutError("synthetic host failure"),
        "timeout-code": LLMError("synthetic host failure", {"code": "timeout"}),
        "call-limit-code": LLMError("synthetic host failure", {"code": "model_call_limit"}),
    }[kind]


async def wait_for_join(cache):
    for _ in range(100):
        if any(getattr(value, "waiters", 0) >= 2 for value in cache.store.values()):
            return
        await asyncio.sleep(0)
    raise AssertionError("both callers must join the same physical request")


@pytest.mark.parametrize("kind", ["timeout", "evaluation", "timeout-code", "call-limit-code"])
@pytest.mark.parametrize("family", ["model", "local"])
@pytest.mark.parametrize("different_alias", [False, True])
@pytest.mark.asyncio
async def test_host_failure_is_shared_without_another_dispatch(
    cache, tmp_path, kind, family, different_alias
):
    started = asyncio.Event()
    release = asyncio.Event()
    failure = host_failure(kind)
    calls = []

    class Adapter:
        local = True
        transform_parts = ("text",)
        transform_cache = TransformCacheIdentity("synthetic.timeout", "1")

        async def operation(self):
            calls.append(None)
            started.set()
            await release.wait()
            if len(calls) == 1:
                raise failure

        async def classify_messages(self, config, messages, tools, *, timeout=None):
            await self.operation()
            return SimpleNamespace(chunks=[SimpleNamespace(probabilities=[0.8, 0.2])])

        async def transform(self, config, part, *, timeout=None):
            await self.operation()
            return "SYNTHETIC"

    if family == "model":
        entry = {"adapter": "synthetic", "model": "synthetic-model", "classify": {"class_index": 0}}
        config = QuarryConfig.from_mapping(
            {"config_version": 2, "models": {"left": entry, "right": entry}},
            tmp_path,
            require_rules=False,
        )
    else:
        config = QuarryConfig(
            config_version=2,
            transforms={name: LocalTransformConfig("synthetic", {}) for name in ["left", "right"]},
        )
    backend = inference.ModelInferenceBackend(config, adapters={"synthetic": Adapter()})
    messages = Messages.from_openai([{"role": "user", "content": "synthetic"}])

    async def call(alias):
        if family == "model":
            return await backend.model_classify_messages(alias, messages)
        return await backend.transform_unit(alias, {"type": "text", "text": "synthetic"})

    first = asyncio.create_task(call("left"))
    await asyncio.wait_for(started.wait(), 3)
    second = asyncio.create_task(call("right" if different_alias else "left"))
    await wait_for_join(cache)
    release.set()
    outcomes = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 3)
    assert len(calls) == 1
    assert all(isinstance(outcome, BaseException) for outcome in outcomes)
    assert not is_owned_request_limit(failure)


@pytest.mark.parametrize("kind", ["timeout", "evaluation", "timeout-code", "call-limit-code"])
def test_diagnostic_labels_and_counterfeit_attributes_do_not_establish_ownership(kind):
    failure = host_failure(kind)
    failure.__dict__["_quarry_owned_request_limit"] = True
    assert not is_owned_request_limit(failure)
    assert not inference._starters_limit(failure)


def test_actual_alias_call_limit_keeps_ownership_through_safe_attribution():
    budget = ModelCallBudget(1)
    budget.consume("physical")
    with pytest.raises(LLMError) as error:
        budget.check("physical")
    assert is_owned_request_limit(error.value)
    attributed = inference._attributed_failure(
        error.value.message, error.value, "authored", private_model="physical"
    )
    assert is_owned_request_limit(attributed)
    assert attributed.details == {
        "code": "model_call_limit",
        "model": "authored",
        "max_model_calls": 1,
    }
    assert "physical" not in attributed.message
    assert "_quarry_owned_request_limit" not in attributed.details


@pytest.mark.asyncio
async def test_actual_native_sender_expiry_allows_one_viable_waiter_retry(cache):
    release = asyncio.Event()
    calls = []

    async def request(name):
        calls.append(name)
        await release.wait()
        if len(calls) == 1:
            return await ModelUsage().run(
                "synthetic", lambda timeout: asyncio.Future(), timeout=0.05
            )
        return 0.8

    with scan_scope():
        first_scope = contextvars.copy_context()
    with scan_scope():
        second_scope = contextvars.copy_context()
    first = asyncio.create_task(
        cache.call(b"owned", request, "starter", retry_on=inference._starters_limit),
        context=first_scope,
    )
    second = asyncio.create_task(
        cache.call(b"owned", request, "joiner", retry_on=inference._starters_limit),
        context=second_scope,
    )
    await wait_for_join(cache)
    release.set()
    outcomes = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 3)
    assert isinstance(outcomes[0], LLMError)
    assert outcomes[0].details["code"] == "timeout"
    assert is_owned_request_limit(outcomes[0])
    assert outcomes[1] == 0.8
    assert calls == ["starter", "joiner"]
