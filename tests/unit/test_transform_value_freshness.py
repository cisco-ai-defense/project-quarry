# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Stateful adapters expose stale scans and accidental per-part value deduplication."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

import quarry.cache.message_values as memo_module
from quarry import load_compiled_policy
from quarry.cache.message_values import MessageValueMemo, _retained_bytes
from quarry.engine.atoms import ModelExpression
from quarry.integrations.model_expression import (
    ModelExpressionTrace,
    _message_value_key,
    evaluate_model_expression,
)
from quarry.safety.limits import SafetyLimits
from quarry.safety.timer import QuarryTimeoutError, scan_deadline
from quarry.types.context import ScanContext
from quarry.types.messages import Message, Messages
from tests.unit.test_adapter_transform_dsl import Adapter, compile_condition, configured


class CounterAdapter(Adapter):
    transform_parts = frozenset({"text"})

    async def transform(self, config, part, *, timeout=None):
        self.parts.append(part)
        ordinal = len(self.parts)
        await asyncio.sleep(0)
        return f"{part['text']}:{ordinal}"


class WholeCounterAdapter(CounterAdapter):
    transform_parts = ()

    async def transform_messages(self, config, messages):
        self.parts.append(messages)
        return [
            {**message, "role": "assistant" if len(self.parts) == 1 else "user"}
            for message in messages
        ]


def policy_for(tmp_path, condition, *, prepared=False, whole=False, definitions=None):
    provider = WholeCounterAdapter() if whole else CounterAdapter()
    config, backend, _ = configured(tmp_path, adapter=provider, definitions=definitions)
    policy = compile_condition(config, backend, condition)
    if prepared:
        policy = load_compiled_policy(
            policy.to_bytes(), runtime_config=config, model_inference_backend=backend
        )
    return policy, provider


PAYLOAD = {"messages": [{"role": "user", "content": "same"}]}
FIRST = 'messages.transform("convert")[0].content contains ":1"'


def first_read(whole):
    return 'messages.transform("convert")[0].role == "assistant"' if whole else FIRST


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("whole", [False, True])
def test_new_sync_scan_recomputes_a_local_message_value(tmp_path, prepared, whole):
    policy, provider = policy_for(tmp_path, first_read(whole), prepared=prepared, whole=whole)
    results = [policy.scan(PAYLOAD) for _ in range(3)]
    assert all(result.is_complete for result in results)
    assert [result.rule_matches for result in results] == [["R"], [], []]
    assert len(provider.parts) == 3


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("whole", [False, True])
@pytest.mark.asyncio
async def test_new_async_scan_recomputes_on_the_same_loop(tmp_path, prepared, whole):
    policy, provider = policy_for(tmp_path, first_read(whole), prepared=prepared, whole=whole)
    results = [await policy.scan_async(PAYLOAD) for _ in range(3)]
    assert all(result.is_complete for result in results)
    assert [result.rule_matches for result in results] == [["R"], [], []]
    assert len(provider.parts) == 3


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("whole", [False, True])
def test_repeated_expression_is_one_stable_value_within_a_rule(tmp_path, prepared, whole):
    policy, provider = policy_for(
        tmp_path,
        first_read(whole)
        + (
            ' and messages.transform("convert")[0].role == "user"'
            if whole
            else ' and messages.transform("convert")[0].content contains ":2"'
        ),
        prepared=prepared,
        whole=whole,
    )
    result = policy.scan(PAYLOAD)
    assert result.is_complete and result.rule_matches == []
    assert len(provider.parts) == 1


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("parts_in_one_message", [False, True])
def test_identical_parts_in_distinct_locations_are_processed_separately(
    tmp_path, prepared, parts_in_one_message
):
    policy, provider = policy_for(
        tmp_path, 'messages.transform("convert")[-1].content contains ":2"', prepared=prepared
    )
    messages = (
        [{"role": "user", "content": [{"type": "text", "text": "same"}] * 2}]
        if parts_in_one_message
        else PAYLOAD["messages"] * 2
    )
    result = policy.scan({"messages": messages})
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(provider.parts) == 2


@pytest.mark.parametrize("prepared", [False, True])
def test_aliases_do_not_share_a_local_message_value(tmp_path, prepared):
    policy, provider = policy_for(
        tmp_path,
        'messages.transform("one")[0].content contains ":1" '
        'and messages.transform("two")[0].content contains ":2"',
        prepared=prepared,
        definitions={"one": {"adapter": "fake"}, "two": {"adapter": "fake"}},
    )
    result = policy.scan(PAYLOAD)
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(provider.parts) == 2


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize(
    "left,right,matched,parts",
    [
        ("messages", "messages[:]", True, 2),
        ("messages[:1]", "messages[:1]", False, 1),
        ("messages", 'messages.transform("convert")', True, 2),
    ],
)
def test_source_expression_identity_and_nested_transforms(
    tmp_path, prepared, left, right, matched, parts
):
    policy, provider = policy_for(
        tmp_path,
        f'{left}.transform("convert")[0].content contains ":1" and '
        f'{right}.transform("convert")[0].content contains ":2"',
        prepared=prepared,
    )
    result = policy.scan(PAYLOAD)
    assert result.is_complete and result.rule_matches == (["R"] if matched else [])
    assert len(provider.parts) == parts


@pytest.mark.asyncio
async def test_concurrent_scans_have_separate_local_values(tmp_path):
    policy, provider = policy_for(tmp_path, FIRST)
    results = await asyncio.gather(*(policy.scan_async(PAYLOAD) for _ in range(6)))
    assert all(result.is_complete for result in results)
    assert sum(result.rule_matches == ["R"] for result in results) == 1
    assert len(provider.parts) == 6


@pytest.mark.asyncio
async def test_concurrent_reads_in_a_rule_share_one_computation():
    memo = MessageValueMemo()
    value = Messages.from_openai(PAYLOAD["messages"])
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return value

    tasks = [asyncio.create_task(memo.evaluate(("one",), compute)) for _ in range(5)]
    await started.wait()
    release.set()
    assert all(result is value for result in await asyncio.gather(*tasks))
    assert calls == 1


@pytest.mark.asyncio
async def test_failed_or_cancelled_computation_does_not_become_a_value():
    memo = MessageValueMemo()
    value = Messages.from_openai(PAYLOAD["messages"])
    started = asyncio.Event()
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("synthetic failure")
        if calls == 2:
            started.set()
            await asyncio.Event().wait()
        return value

    with pytest.raises(ValueError, match="synthetic failure"):
        await memo.evaluate(("one",), compute)
    task = asyncio.create_task(memo.evaluate(("one",), compute))
    await started.wait()
    waiter = asyncio.create_task(memo.evaluate(("one",), compute))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await waiter is value
    assert await memo.evaluate(("one",), compute) is value
    assert calls == 3


@pytest.mark.asyncio
async def test_expired_deadline_prevents_a_memoized_read():
    memo = MessageValueMemo()
    value = Messages.from_openai(PAYLOAD["messages"])

    async def compute():
        return value

    assert await memo.evaluate(("one",), compute) is value
    with scan_deadline(0.001):
        await asyncio.sleep(0.01)
        with pytest.raises(QuarryTimeoutError):
            await memo.evaluate(("one",), compute)


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", ["entries", "bytes"])
async def test_exhaustion_faults_without_evicting_a_stable_value(monkeypatch, bound):
    memo = MessageValueMemo()
    value = Messages.from_openai(PAYLOAD["messages"])

    async def compute():
        return value

    assert await memo.evaluate(("first",), compute) is value
    if bound == "entries":
        monkeypatch.setattr(memo_module, "MAX_MESSAGE_VALUES", 1)
    else:
        monkeypatch.setattr(memo_module, "RESULT_CACHE_MAX_BYTES", memo._bytes)
    with pytest.raises(ValueError, match="per-rule .* limit"):
        await memo.evaluate(("second",), compute)
    assert await memo.evaluate(("first",), compute) is value


@pytest.mark.asyncio
async def test_sizing_counts_frozen_text_and_skips_wire_and_field_hooks(monkeypatch):
    text = "\U0001f680" * 1000
    value = Messages.from_openai(
        [
            {"role": "user", "content": "opaque"},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]
    )

    # Admission still rejects unknown roles in this public branch. Exercise the sizing
    # boundary with an opaque host-created object without changing conversation admission.
    opaque = Message.model_construct(source_content="opaque", role="alien")
    value = (opaque, value)

    def forbidden(*args, **kwargs):
        raise AssertionError("sizing must not read guarded fields or project JSON")

    monkeypatch.setattr(Message, "to_openai_dict", forbidden)
    monkeypatch.setattr(Message, "content", property(forbidden))
    assert await _retained_bytes(value, 1_000_000) >= sys.getsizeof(text)
    assert await _retained_bytes(value, 100) == 101


@pytest.mark.asyncio
async def test_sizing_work_is_bounded_and_does_not_invoke_host_sizing(monkeypatch):
    class Host:
        def __sizeof__(self):
            raise AssertionError("host sizing hook must not run")

    assert await _retained_bytes(Host(), 1_000_000) > 0
    monkeypatch.setattr(memo_module, "MAX_MESSAGE_VALUE_REFERENCES", 3)
    assert await _retained_bytes([1, 2, 3, 4], 1_000_000) == 1_000_001


def test_context_copy_isolates_rule_values():
    context = ScanContext(PAYLOAD["messages"])
    assert context.copy().message_value_memo is not context.message_value_memo


@pytest.mark.asyncio
async def test_message_part_dispatch_preserves_injected_host_hook_and_restores_cache(tmp_path):
    provider = CounterAdapter()
    _, backend, _ = configured(tmp_path, adapter=provider)
    original = backend.transform_unit
    calls = []

    async def injected(name, unit):
        calls.append(unit)
        return await original(name, unit)

    backend.transform_unit = injected
    unit = {"type": "text", "text": "same"}
    assert (await backend.transform_message_unit("convert", unit)).text == "same:1"
    assert (await backend.transform_message_unit("convert", unit)).text == "same:2"
    assert (await backend.transform_unit("convert", unit)).text == "same:3"
    assert (await backend.transform_unit("convert", unit)).text == "same:3"
    assert len(calls) == 4 and len(provider.parts) == 3


def test_value_key_tracks_free_bindings_and_ignores_internal_and_unused_bindings():
    first, second = object(), object()
    root = ModelExpression("local_transform", "synthetic", (ModelExpression("messages"),))
    assert _message_value_key(root, {"unused": first}, None) == _message_value_key(root, {}, None)
    bound = ModelExpression("local_transform", "synthetic", (ModelExpression("binding", "item"),))
    assert _message_value_key(bound, {"item": first}, None) != _message_value_key(
        bound, {"item": second}, None
    )
    selected = ModelExpression(
        "message_select", "item", (ModelExpression("messages"), ModelExpression("binding", "item"))
    )
    assert _message_value_key(selected, {}, SimpleNamespace(source=first)) == _message_value_key(
        selected, {}, SimpleNamespace(source=second)
    )


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize(
    "value,parts",
    [
        ('[item.transform("convert") for item in messages]', 2),
        ('[message("user", item.content).transform("convert") for item in messages]', 2),
        ('[item for item in messages if item.role == "user"].transform("convert")', 2),
        ('[messages.transform("convert") for item in messages]', 2),
        ('[message("user", "same").transform("convert") for item in messages]', 1),
    ],
)
def test_repeated_map_is_one_value_and_hoists_item_independent_transforms(
    tmp_path, prepared, value, parts
):
    provider = CounterAdapter()
    config, backend, _ = configured(tmp_path, adapter=provider)
    config = replace(config, safety_limits=SafetyLimits(max_model_calls=1, rule_timeout=None))
    policy = compile_condition(
        config,
        backend,
        f'model("judge").classify({value}) > 0.5 and model("judge").classify({value}) < 0.95',
    )
    if prepared:
        policy = load_compiled_policy(
            policy.to_bytes(), runtime_config=config, model_inference_backend=backend
        )
    result = policy.scan({"messages": PAYLOAD["messages"] * 2}, explain=True)
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(provider.parts) == parts and len(provider.inputs) == 1


@pytest.mark.parametrize("prepared", [False, True])
def test_different_map_sources_keep_their_own_values(tmp_path, prepared):
    provider = CounterAdapter()
    config, backend, _ = configured(tmp_path, adapter=provider)
    policy = compile_condition(
        config,
        backend,
        'model("judge").classify([item.transform("convert") for item in messages]) > 0.5 '
        'and model("judge").classify([item.transform("convert") for item in messages[:]]) < 0.95',
    )
    if prepared:
        policy = load_compiled_policy(
            policy.to_bytes(), runtime_config=config, model_inference_backend=backend
        )
    result = policy.scan(PAYLOAD)
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(provider.parts) == len(provider.inputs) == 2


def test_item_values_are_released_without_losing_the_complete_map(tmp_path, monkeypatch):
    monkeypatch.setattr(memo_module, "MAX_MESSAGE_VALUES", 3)
    provider = CounterAdapter()
    config, backend, _ = configured(tmp_path, adapter=provider)
    value = '[item.transform("convert") for item in messages]'
    policy = compile_condition(
        config,
        backend,
        f'model("judge").classify({value}) > 0.5 and model("judge").classify({value}) < 0.95',
    )
    result = policy.scan({"messages": PAYLOAD["messages"] * 10})
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(provider.parts) == 10 and len(provider.inputs) == 1


@pytest.mark.asyncio
async def test_memoized_map_replays_only_its_actual_parameter_reads():
    context = ScanContext(PAYLOAD["messages"], parameters={"label": "private-text"})
    expression = ModelExpression(
        "message_map",
        "item",
        (
            ModelExpression("messages"),
            ModelExpression("literal", True),
            ModelExpression(
                "message",
                None,
                (ModelExpression("literal", "user"), ModelExpression("parameter", "label")),
            ),
        ),
    )
    # First computation happens without explain; a later trace must still report the read.
    original = await evaluate_model_expression(expression, context)
    trace = ModelExpressionTrace()
    result = await evaluate_model_expression(expression, context, trace=trace)
    assert result is original and trace.parameters == [("label", "«redacted»")]
    assert "private-text" not in repr(trace)


@pytest.mark.asyncio
async def test_sizing_unknown_host_objects_never_reads_class_or_size_hooks():
    class Host:
        @property
        def __class__(self):
            raise AssertionError("sizing must not ask host objects for their class")

        def __sizeof__(self):
            raise AssertionError("sizing must not invoke host size hooks")

    assert await _retained_bytes(Host(), 1_000_000) > 0
