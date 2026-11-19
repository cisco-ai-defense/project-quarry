# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Real batch owners close on every exit, including work abandoned between phases."""

import asyncio
import gc
import weakref
from dataclasses import replace
from pathlib import Path

import pytest

from quarry import compile_policy
from quarry.cache.async_cache import AsyncFunctionCache
from quarry.config import QuarryConfig
from quarry.integrations import model_inference as inference
from quarry.safety.limits import SafetyLimits
from quarry.safety.timer import QuarryTimeoutError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    [
        "complete",
        "verdict_failure",
        "verdict_cancel",
        "deadline",
        "raising_deadline",
        "evidence_failure",
        "evidence_cancel",
    ],
)
async def test_policy_closes_real_response_owners_on_every_batch_exit(monkeypatch, action):
    class Raw(float):
        pass

    references, owners = [], []

    class Adapter:
        calls = 0

        async def score(self, config, messages, *, timeout=None):
            self.calls += 1
            value = Raw(0.8)
            references.append(weakref.ref(value))
            return value

    adapter = Adapter()
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "model_cache": {"max_bytes": 1},
            "models": {
                "j": {
                    "adapter": "owned",
                    "model": "owned",
                    "score": {
                        "message_builder": "constitution",
                        "boundary_seed": 7,
                        "tokens": {"negative": "0", "positive": "1"},
                    },
                }
            },
        },
        Path("."),
        require_rules=False,
    )
    config = replace(
        config,
        safety_limits=SafetyLimits(rule_timeout=None, fail_on_timeout=action == "raising_deadline"),
    )
    backend = inference.ModelInferenceBackend(config, adapters={"owned": adapter})
    policy = compile_policy(
        'rule "R" condition: model("j").score(messages,"criterion") > .5',
        runtime_config=config,
        model_inference_backend=backend,
    )
    monkeypatch.setattr(
        inference,
        "result_cache",
        AsyncFunctionCache.expiring(max_entries=2, ttl_seconds=60, max_bytes=1),
    )
    original = policy._scanner._scan_data_with_parameters_async
    started, stopped = asyncio.Event(), asyncio.Event()

    async def blocked(_result):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def failed(_result):
        await started.wait()
        raise ValueError("evidence failed")

    async def scan(*args, **kwargs):
        first = len(kwargs["deferred_evidence"])
        result = await original(*args, **kwargs)
        added = kwargs["deferred_evidence"][first:]
        assert len(added) == 1
        owners.extend(added)
        assert all(ref() is not None for ref in references)
        if action in {"deadline", "raising_deadline"}:
            # Exercise a real elapsed shared deadline after an owner was produced.
            await asyncio.sleep(1.05)
        if len(owners) == 2:
            if action == "verdict_failure":
                raise ValueError("second verdict failed")
            if action == "verdict_cancel":
                raise asyncio.CancelledError
        if action == "evidence_failure":
            added[0]._finish = failed if len(owners) == 1 else blocked
        elif action == "evidence_cancel":
            added[0]._finish = blocked
        return result

    monkeypatch.setattr(policy._scanner, "_scan_data_with_parameters_async", scan)
    conversations = [{"messages": [{"role": "user", "content": str(i)}]} for i in range(2)]
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        operation = policy.scan_many_async(
            conversations,
            explain=True,
            decision_paths=True,
            deadline_seconds=1.0 if "deadline" in action else None,
        )
        if action == "evidence_cancel":
            task = asyncio.create_task(operation)
            await asyncio.wait_for(started.wait(), 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            del task
            assert stopped.is_set()
        elif action in {"verdict_failure", "evidence_failure"}:
            with pytest.raises(ValueError, match="failed"):
                await operation
            if action == "evidence_failure":
                assert stopped.is_set()
        elif action == "verdict_cancel":
            with pytest.raises(asyncio.CancelledError):
                await operation
        elif action == "raising_deadline":
            with pytest.raises(QuarryTimeoutError):
                await operation
        else:
            results = await operation
            assert len(results) == 2
            if action == "complete":
                assert all(r.is_complete and r.rule_matches == ["R"] for r in results)
            else:
                assert all(
                    r.completion_state == "incomplete" and r.timed_out_rules == ["R"]
                    for r in results
                )
            del results
        del operation
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert owners and all(owner._finish is None and owner._close is None for owner in owners)
        assert references and all(ref() is None for ref in references)
        assert adapter.calls == (1 if "deadline" in action else 2)
    finally:
        if enabled:
            gc.enable()
