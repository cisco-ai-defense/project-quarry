# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Public rule attribution with synthetic in-process adapters only."""

import asyncio
from contextlib import nullcontext
from contextvars import copy_context
from dataclasses import replace

import pytest

import quarry
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.safety.limits import SafetyLimits


def _policy(tmp_path, name, adapter):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "judge": {
                    "adapter": "owned",
                    "model": "synthetic",
                    "score": {"message_builder": "owned", "boundary_seed": 7},
                }
            },
        },
        tmp_path,
        require_rules=False,
    )
    config = replace(config, safety_limits=SafetyLimits(rule_timeout=None))
    backend = ModelInferenceBackend(
        config,
        adapters={"owned": adapter},
        message_builders={"owned": lambda _c, messages, _t, _b, _s: messages},
    )
    return quarry.compile_policy(
        f'rule "{name}" condition: model("judge").score(messages, "synthetic") > 0.5',
        runtime_config=config,
        model_inference_backend=backend,
    )


def _input(text="synthetic"):
    return {"messages": [{"role": "user", "content": text}]}


def test_no_rule_is_reported_outside_evaluation():
    assert quarry.current_rule_name() is None


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("accounting", [False, True])
@pytest.mark.parametrize("failing", [False, True])
async def test_callbacks_have_live_ownership_without_retaining_it(
    tmp_path, sync, accounting, failing
):
    names = []
    contexts = []

    class Adapter:
        async def score(self, _config, _messages, *, timeout=None):
            names.append(quarry.current_rule_name())
            contexts.append(copy_context())
            names.append(await asyncio.to_thread(contexts[-1].run, quarry.current_rule_name))

            async def child():
                return quarry.current_rule_name()

            names.append(await asyncio.create_task(child()))
            if failing:
                raise RuntimeError("synthetic provider failure")
            return 0.8

    policy = _policy(tmp_path, "Selected", Adapter())
    with quarry.model_read_usage() if accounting else nullcontext():
        if sync:
            result = await asyncio.to_thread(policy.scan, _input())
        else:
            result = await policy.scan_async(_input())
        assert result.is_complete is not failing
        assert result.rule_matches == ([] if failing else ["Selected"])
        assert names == ["Selected", "Selected", "Selected"]
        assert quarry.current_rule_name() is None
        assert contexts[0].run(quarry.current_rule_name) is None


async def test_nested_public_scans_restore_the_callers_owner(tmp_path):
    names = []

    class Inner:
        async def score(self, _config, _messages, *, timeout=None):
            names.append(quarry.current_rule_name())
            return 0.8

    inner = _policy(tmp_path, "Inner", Inner())

    class Outer:
        async def score(self, _config, _messages, *, timeout=None):
            names.append(quarry.current_rule_name())
            result = await inner.scan_async(_input("inside"))
            assert result.is_complete and result.rule_matches == ["Inner"]
            names.append(quarry.current_rule_name())
            return 0.8

    result = await _policy(tmp_path, "Outer", Outer()).scan_async(_input("outside"))
    assert result.is_complete and result.rule_matches == ["Outer"]
    assert names == ["Outer", "Inner", "Outer"]
    assert quarry.current_rule_name() is None


async def test_parallel_scans_do_not_exchange_callback_owners(tmp_path):
    names = []
    entered = asyncio.Event()

    class Adapter:
        async def score(self, _config, _messages, *, timeout=None):
            name = quarry.current_rule_name()
            names.append(name)
            if len(names) == 2:
                entered.set()
            await entered.wait()
            assert quarry.current_rule_name() == name
            return 0.8

    first = _policy(tmp_path, "First", Adapter())
    second = _policy(tmp_path, "Second", Adapter())
    results = await asyncio.wait_for(
        asyncio.gather(first.scan_async(_input("one")), second.scan_async(_input("two"))), 10
    )
    assert [r.rule_matches for r in results] == [["First"], ["Second"]]
    assert all(r.is_complete for r in results)
    assert sorted(names) == ["First", "Second"]
    assert quarry.current_rule_name() is None


async def test_cancelled_callbacks_keep_ownership_until_their_cleanup_ends(tmp_path):
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    contexts = []
    names = []

    class Adapter:
        async def score(self, _config, _messages, *, timeout=None):
            contexts.append(copy_context())
            names.append(quarry.current_rule_name())
            entered.set()
            try:
                await asyncio.Future()
            finally:
                names.append(quarry.current_rule_name())
                cleaned.set()

    policy = _policy(tmp_path, "Cancelled", Adapter())
    scan = asyncio.create_task(policy.scan_async(_input()))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        scan.cancel()
        with pytest.raises(asyncio.CancelledError):
            await scan
        assert cleaned.is_set()
        assert names == ["Cancelled", "Cancelled"]
        assert contexts[0].run(quarry.current_rule_name) is None
        assert quarry.current_rule_name() is None
    finally:
        if not scan.done():
            scan.cancel()
        await asyncio.gather(scan, return_exceptions=True)
