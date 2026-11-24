# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Completed native scans must not wait for cyclic GC to release their rule owners."""

import asyncio
import gc
import weakref

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.types.context import ScanContext
from quarry.types.exceptions import LLMError


@pytest.fixture
def tracked_contexts_without_gc(monkeypatch):
    references = []
    original = ScanContext.copy

    def copy(self):
        result = original(self)
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(ScanContext, "copy", copy)
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield references
    finally:
        if enabled:
            gc.enable()
        gc.collect()


def tool(name, description):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("explain", [False, True, "support"])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("quantifier", ["any", "all"])
@pytest.mark.parametrize("matches", [False, True])
async def test_tool_quantifier_contexts_release_with_results_alive(
    tracked_contexts_without_gc, explain, prepared, quantifier, matches
):
    source = (
        f'rule "R" condition: {quantifier} t in tools where '
        't.name == "read" and t.description contains "synthetic"'
    )
    policy = compile_policy(source)
    if prepared:
        policy = load_compiled_policy(policy.to_bytes())
    results = []
    for index in range(4):
        results.append(
            await policy.scan_async(
                {
                    "messages": [{"role": "user", "content": f"safe {index}"}],
                    "tools": [tool("read", "synthetic" if matches else "unrelated")],
                },
                explain=explain,
            )
        )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert all(r.is_complete and r.rule_matches == (["R"] if matches else []) for r in results)
    assert tracked_contexts_without_gc
    assert all(reference() is None for reference in tracked_contexts_without_gc)


@pytest.mark.asyncio
@pytest.mark.parametrize("explain", [False, True, "support"])
@pytest.mark.parametrize("prepared", [False, True])
async def test_failed_tool_callback_releases_context_without_discarding_diagnostics(
    tracked_contexts_without_gc, explain, prepared
):
    calls = []

    def fail(value):
        calls.append(value)
        raise ValueError("synthetic private callback detail")

    source = (
        'rule "R" strings: $p = "synthetic" check("fail") condition: '
        'any t in tools where t.name == "read" and t.description ~ $p'
    )
    policy = compile_policy(source, pattern_checks={"fail": fail})
    if prepared:
        policy = load_compiled_policy(policy.to_bytes(), pattern_checks={"fail": fail})
    result = await policy.scan_async(
        {"messages": [{"role": "user", "content": "safe"}], "tools": [tool("read", "synthetic")]},
        explain=explain,
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not result.is_complete and result.rule_matches == []
    assert result.evaluation_errors and "R" in result.evaluation_errors
    assert "synthetic private callback detail" not in str(result.evaluation_errors)
    assert calls == ["synthetic"]
    assert tracked_contexts_without_gc
    assert all(reference() is None for reference in tracked_contexts_without_gc)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["complete", "failure", "cancel", "deadline"])
@pytest.mark.parametrize("overlap", [0, 2])
async def test_recursive_split_releases_native_sources_after_owned_tasks_drain(
    tracked_contexts_without_gc, tmp_path, monkeypatch, action, overlap
):
    from quarry.cache.async_cache import AsyncFunctionCache
    from quarry.config import QuarryConfig
    from quarry.integrations import model_inference as inference
    from quarry.types.messages import Messages

    started = asyncio.Event()
    stopped = asyncio.Event()
    text = "alpha beta gamma delta epsilon zeta eta theta"
    calls = []

    class Counter:
        async def tokenize(self, config, messages, tools, *, timeout=None):
            content = "".join(message["content"] for message in messages)
            calls.append(content)
            if len(content) < len(text):
                if action == "failure":
                    raise ValueError("synthetic tokenizer failure")
                if action in {"cancel", "deadline"}:
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        stopped.set()
            return {"count": len(content)}

    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "counter": {
                    "adapter": "counter",
                    "model": "synthetic",
                    "tokens": {"adapter": "counter", "extraction": {"count": "count"}},
                }
            },
        },
        tmp_path,
        require_rules=False,
    )
    monkeypatch.setattr(
        inference, "result_cache", AsyncFunctionCache.expiring(max_entries=100, ttl_seconds=60)
    )
    backend = inference.ModelInferenceBackend(config, adapters={"counter": Counter()})
    references = []

    async def operation():
        source = Messages.from_openai([{"role": "user", "content": text}])
        references.append(weakref.ref(source[0]))
        return await backend._split_messages("counter", source, 10, overlap, None, timeout=None)

    if action == "complete":
        chunks = await operation()
        assert len(chunks) > 1
        assert all(len(chunk[0].content) <= 10 for chunk in chunks)
        if not overlap:
            assert "".join(chunk[0].content for chunk in chunks) == text
        assert any(reference() is not None for reference in references)
        del chunks
    elif action == "failure":
        with pytest.raises(LLMError, match="Inference model .counter. failed"):
            await operation()
    else:
        task = asyncio.create_task(operation())
        await asyncio.wait_for(started.wait(), 3)
        if action == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.01):
                    await task
        assert stopped.is_set()
        del task
    for _ in range(4):
        await asyncio.sleep(0)
    assert len(calls) > 1
    assert references and all(reference() is None for reference in references)


@pytest.mark.parametrize("kind", ["slice", "role", "combine", "append"])
def test_live_native_views_keep_their_source_then_release_without_collection(
    tracked_contexts_without_gc, kind
):
    from quarry.types.messages import Messages

    source = Messages.from_openai(
        [{"role": "user", "content": "first"}, {"role": "assistant", "content": "second"}]
    )
    reference = weakref.ref(source[0])
    assert source._root is source and source._source_root is source
    if kind == "slice":
        view = source[:1]
    elif kind == "role":
        view = source.user_messages
    elif kind == "combine":
        view = Messages.combine((source[:1], source[1:]))
    else:
        view = source.add_message("user", "third")
    wire = view.to_openai_messages()
    del source
    assert reference() is not None
    assert view.to_openai_messages() == wire
    with pytest.raises(AttributeError, match="immutable"):
        view._root = None
    del view
    assert reference() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("retention", ["process", "scan"])
@pytest.mark.parametrize("waiters", [1, 3])
async def test_shared_failures_release_request_owners_and_allow_fresh_retry(
    tracked_contexts_without_gc, retention, waiters
):
    from quarry.cache.async_cache import AsyncFunctionCache, scan_scope

    class Owner:
        pass

    cache = AsyncFunctionCache.expiring(max_entries=4, ttl_seconds=60)
    references = []
    calls = []

    async def run_round():
        owner = Owner()
        references.append(weakref.ref(owner))
        gate = asyncio.Event()

        async def fail(value):
            calls.append(value is owner)
            await gate.wait()
            raise ValueError("synthetic shared failure")

        tasks = [
            asyncio.create_task(cache.call(b"same", fail, owner, retention=retention))
            for _ in range(waiters)
        ]
        await asyncio.sleep(0)
        gate.set()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(
            type(error) is ValueError and error.args == ("synthetic shared failure",)
            for error in outcomes
        )
        assert all(error is outcomes[0] for error in outcomes)

    with scan_scope():
        for _ in range(2):
            await run_round()
            for _ in range(4):
                await asyncio.sleep(0)
            assert references and all(reference() is None for reference in references)
    assert calls == [True, True]
    assert not cache.store
