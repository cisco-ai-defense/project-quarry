# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Typed scans plan only compatible selected inputs; head replies keep cache bounds."""

import asyncio

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.cache.async_cache import AsyncFunctionCache, SlidingTTLCache, result_cache
from quarry.integrations import model_inference
from quarry.types.exceptions import ConfigurationError
from tests.unit.model_fakes import model_entry
from tests.unit.test_model_token_head_contracts import HeadAdapter, LegacyAdapter, backend

DATA = {
    "messages": [
        {"role": "user", "content": "synthetic first"},
        {"role": "assistant", "content": "synthetic second"},
    ]
}


def runtime(tmp_path, adapter):
    return backend(
        tmp_path,
        {
            "classes": model_entry(classify=0),
            "segments": model_entry(spans=0, category="SYNTHETIC"),
        },
        adapter,
    )


def policy_for(runtime, mode, restored=False):
    selected = "messages[:1]" if mode == "different" else "messages"
    source = f'rule "Read" condition: model("classes").classify({selected}) > 0.5'
    if mode != "class-only":
        selected = "messages[-1:]" if mode == "different" else "messages"
        source += f'\ncollect: [s for s in model("segments").spans({selected}, 0.5) if true]'
    policy = compile_policy(source, runtime_config=runtime.config, model_inference_backend=runtime)
    if restored:
        raw = policy.to_bytes()
        policy = load_compiled_policy(
            raw, runtime_config=runtime.config, model_inference_backend=runtime
        )
        assert policy.to_bytes() == raw
    return policy


@pytest.mark.asyncio
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("mode", ["class-only", "shared", "different"])
async def test_compiled_inputs_plan_only_the_heads_the_selected_policy_can_read(
    tmp_path, mode, restored
):
    adapter = HeadAdapter()
    service = runtime(tmp_path, adapter)
    policy = policy_for(service, mode, restored)
    result_cache.clear()
    result = await policy.scan_async(DATA)
    assert result.is_complete and result.rule_matches == ["Read"], result.evaluation_errors
    expected = [False] if mode == "class-only" else [True] if mode == "shared" else [False, True]
    assert sorted(adapter.head_requests) == expected
    assert adapter.classifications == len(expected)
    assert adapter.head_decodes == ([] if mode == "class-only" else [True])
    if mode == "different":
        by_text = {
            payload[0]["content"]: head
            for payload, head in zip(
                adapter.classification_messages, adapter.head_requests, strict=True
            )
        }
        assert by_text == {"synthetic first": False, "synthetic second": True}


@pytest.mark.asyncio
async def test_a_headless_cached_reply_cannot_satisfy_a_later_span_plan(tmp_path):
    adapter = HeadAdapter()
    service = runtime(tmp_path, adapter)
    plain = policy_for(service, "class-only")
    spans = policy_for(service, "shared")
    result_cache.clear()
    first = await plain.scan_async(DATA)
    second = await spans.scan_async(DATA)
    assert first.is_complete and second.is_complete
    assert adapter.head_requests == [False, True]
    assert adapter.head_decodes == [True]
    assert adapter.classifications == 2


@pytest.mark.asyncio
async def test_concurrent_selected_plans_do_not_leak_between_scans(tmp_path, monkeypatch):
    original = model_inference.require_model_messages

    async def cooperate(messages):
        await original(messages)
        await asyncio.sleep(0)

    monkeypatch.setattr(model_inference, "require_model_messages", cooperate)
    adapter = HeadAdapter()
    service = runtime(tmp_path, adapter)
    plain = policy_for(service, "class-only")
    spans = policy_for(service, "shared")
    left = {"messages": [{"role": "user", "content": "synthetic left"}]}
    right = {"messages": [{"role": "user", "content": "synthetic right"}]}
    result_cache.clear()
    results = await asyncio.gather(plain.scan_async(left), spans.scan_async(right))
    assert all(r.is_complete and r.rule_matches == ["Read"] for r in results)
    assert {
        payload[0]["content"]: head
        for payload, head in zip(
            adapter.classification_messages, adapter.head_requests, strict=True
        )
    } == {"synthetic left": False, "synthetic right": True}
    assert adapter.classifications == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("head", [False, True])
async def test_only_head_entries_have_a_five_minute_fixed_expiry(tmp_path, monkeypatch, head):
    now = [0.0]
    cache = AsyncFunctionCache(store=SlidingTTLCache(maxsize=16, ttl=1000, timer=lambda: now[0]))
    monkeypatch.setattr(model_inference, "result_cache", cache)
    adapter = HeadAdapter() if head else LegacyAdapter()
    service = runtime(tmp_path, adapter)
    from quarry.types.messages import Messages

    messages = Messages.from_openai(DATA["messages"])
    for instant in [0, 250, 299, 300]:
        now[0] = instant
        assert await service.model_classify_messages("classes", messages) == 0.8
    assert adapter.classifications == (2 if head else 1)


@pytest.mark.asyncio
async def test_known_head_payload_bytes_remain_subject_to_the_cache_bound(tmp_path, monkeypatch):
    class LargeHead(HeadAdapter):
        async def classify_messages(self, *args, **kwargs):
            raw = await super().classify_messages(*args, **kwargs)
            raw.token_head = bytearray(20_000)
            return raw

    cache = AsyncFunctionCache(store=SlidingTTLCache(maxsize=16, ttl=1000, max_bytes=4096))
    monkeypatch.setattr(model_inference, "result_cache", cache)
    adapter = LargeHead()
    service = runtime(tmp_path, adapter)
    from quarry.types.messages import Messages

    messages = Messages.from_openai(DATA["messages"])
    for _ in range(2):
        assert await service.model_classify_messages("classes", messages) == 0.8
    assert adapter.classifications == 2
    assert len(cache.store) == 0


@pytest.mark.asyncio
async def test_an_optin_with_an_incompatible_decoder_fails_before_dispatch(tmp_path):
    class InvalidHead(HeadAdapter):
        async def decode_spans(self, config, raw, messages, threshold):
            return []

    adapter = InvalidHead()
    service = runtime(tmp_path, adapter)
    from quarry.types.messages import Messages

    with pytest.raises(ConfigurationError, match="token_head"):
        await service.model_classify_messages("classes", Messages.from_openai(DATA["messages"]))
    assert adapter.classifications == 0


@pytest.mark.asyncio
async def test_a_different_loaded_revision_does_not_request_an_unusable_head(tmp_path):
    models = {
        "classes": {**model_entry(classify=0), "revision": "r1"},
        "segments": {**model_entry(spans=0, category="SYNTHETIC"), "revision": "r2"},
    }
    adapter = HeadAdapter()
    service = backend(tmp_path, models, adapter)
    from quarry.types.messages import Messages

    messages = Messages.from_openai(DATA["messages"])
    result_cache.clear()
    assert await service.model_classify_messages("classes", messages) == 0.8
    result = await service.model_span_messages("segments", messages, 0.5)
    assert len(result.items) == 1
    assert adapter.head_requests == [False, True]
    assert adapter.head_decodes == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("frame", ["called", "composed"])
@pytest.mark.parametrize("shared", [False, True])
async def test_nested_message_frames_do_not_borrow_an_unrelated_root_head(
    tmp_path, restored, frame, shared
):
    from tests.unit.test_nested_rule_runtime_contracts import _policy

    adapter = HeadAdapter()
    service = runtime(tmp_path, adapter)
    selected = "messages" if shared else "messages[-1:]"
    if frame == "composed":
        selected = f"({selected} + messages[:0])"
    source = (
        'rule "Leaf" condition: model("classes").classify(messages) > 0.5\n'
        f'rule "Outer" imports: leaf = "Leaf" condition: leaf({selected})'
    )
    source += '\ncollect: [s for s in model("segments").spans(messages, 0.5) if true]'
    policy = _policy(source, service.config, restored, service)
    result_cache.clear()
    result = await policy.scan_async(DATA)
    assert result.is_complete and result.rule_matches == ["Outer"], result.evaluation_errors
    assert sorted(adapter.head_requests) == ([True] if shared else [False, True])
    assert adapter.head_decodes == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("offered", [False, True])
async def test_native_classification_with_offered_tools_cannot_supply_a_span_head(
    tmp_path, offered
):
    from quarry.types.messages import Messages, Tool

    adapter = HeadAdapter()
    service = runtime(tmp_path, adapter)
    messages = Messages.from_openai(DATA["messages"])
    tools = [Tool.model_validate({"type": "function", "function": {"name": "synthetic"}})]
    result_cache.clear()
    assert (
        await service.model_classify_messages("classes", messages, tools if offered else []) == 0.8
    )
    result = await service.model_span_messages("segments", messages, 0.5)
    assert len(result.items) == 1
    assert adapter.head_requests == ([False, True] if offered else [True])
    assert adapter.head_decodes == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("restored", [False, True])
async def test_called_rule_collect_is_not_a_selected_span_reader(tmp_path, restored):
    from tests.unit.test_nested_rule_runtime_contracts import _policy

    adapter = HeadAdapter()
    service = runtime(tmp_path, adapter)
    source = (
        'rule "Leaf" condition: true\n'
        'collect: [s for s in model("segments").spans(messages, 0.5) if true]\n'
        'rule "Outer" imports: leaf = "Leaf" condition: '
        'model("classes").classify(messages) > 0.5 and leaf(messages)'
    )
    policy = _policy(source, service.config, restored, service)
    result_cache.clear()
    result = await policy.scan_async(DATA)
    assert result.is_complete and result.rule_matches == ["Outer"], result.evaluation_errors
    assert adapter.head_requests == [False]
    assert adapter.head_decodes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("restored", [False, True])
async def test_computed_item_classification_does_not_borrow_the_whole_input_head(
    tmp_path, restored
):
    adapter = HeadAdapter()
    service = runtime(tmp_path, adapter)
    source = (
        'rule "Read" condition: any m in (messages + messages[:0]) '
        'where model("classes").classify(m) > 0.5\n'
        'collect: [s for s in model("segments").spans(messages, 0.5) if true]'
    )
    policy = compile_policy(source, runtime_config=service.config, model_inference_backend=service)
    if restored:
        raw = policy.to_bytes()
        policy = load_compiled_policy(
            raw, runtime_config=service.config, model_inference_backend=service
        )
        assert policy.to_bytes() == raw
    result_cache.clear()
    result = await policy.scan_async(DATA)
    assert result.is_complete and result.rule_matches == ["Read"], result.evaluation_errors
    assert sorted(adapter.head_requests) == [False, True]
    assert adapter.head_decodes == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["resolve", "discard", "cancel"])
async def test_deferred_evidence_resumes_the_selected_plan_in_a_new_task(
    tmp_path, monkeypatch, finish
):
    from dataclasses import replace

    import quarry.scanner as scanner_module
    from quarry.integrations.model_head_plan import head_plan_active, head_requests

    adapter = HeadAdapter()
    service = runtime(tmp_path, adapter)
    policy = policy_for(service, "shared")
    original = scanner_module.evaluate_async
    entered = asyncio.Event()
    observed = []

    async def evaluate(*args, **kwargs):
        result = await original(*args, **kwargs)
        if kwargs.get("defer_minimization"):
            refine = result.refine

            async def observe(time_left):
                observed.append(head_requests(service))
                entered.set()
                if finish == "cancel":
                    await asyncio.Event().wait()
                return await refine(time_left) if refine is not None else result

            return replace(result, refine=observe)
        return result

    monkeypatch.setattr(scanner_module, "evaluate_async", evaluate)
    result_cache.clear()
    callbacks = []
    result = await policy._scanner._scan_data_with_parameters_async(
        DATA, {}, enable_explain=True, deferred_evidence=callbacks
    )
    assert result.is_complete and result.rule_matches == ["Read"]
    assert len(callbacks) == 1 and not head_plan_active()
    if finish == "discard":
        callbacks[0].close()
        assert observed == []
    else:
        task = asyncio.create_task(callbacks[0](result))
        if finish == "cancel":
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            finished = await task
            assert finished.rule_matches == ["Read"]
        assert observed and all(plan for plan in observed)
    assert not head_plan_active() and head_requests(service) is None
    assert adapter.head_requests == [True]
