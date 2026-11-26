# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Closed immediate filter operands are reused within their owning frame."""

import asyncio
import weakref
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest

from quarry import compile_policy
from quarry.config import QuarryConfig
from quarry.integrations import model_expression
from quarry.safety.limits import SafetyLimits


def conversation(size, *, alternating=False):
    return {
        "messages": [
            {"role": "user", "content": "synthetic"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"c{index}",
                        "type": "function",
                        "function": {
                            "name": "other" if alternating and index % 2 else "lookup",
                            "arguments": "{}",
                        },
                    }
                    for index in range(size)
                ],
            },
        ]
    }


def policy(source):
    return compile_policy(
        source,
        runtime_config=QuarryConfig(
            config_version=2, safety_limits=SafetyLimits(rule_timeout=None)
        ),
    )


def observe_count_operands(monkeypatch):
    reads = []
    original = model_expression._evaluate_local_calls

    async def observed(expression, context, event, environment, trace, backend):
        value = await original(expression, context, event, environment, trace, backend)
        if expression.kind == "call_count" and expression.children[0].kind != "call_filter":
            reads.append((id(context), value))
        return value

    monkeypatch.setattr(model_expression, "_evaluate_local_calls", observed)
    return reads


@pytest.mark.parametrize("size", [1, 8, 32, 128])
@pytest.mark.parametrize("explain", [False, True, "support"])
def test_closed_comparison_does_not_repeat_its_count_operand(monkeypatch, size, explain):
    subject = policy(
        'rule "Read" condition: count([c for c in assistant_messages[-1].tool_calls '
        f'if len(assistant_messages[-1].tool_calls) == {size} and c.name == "lookup"]) == {size}'
    )
    reads = observe_count_operands(monkeypatch)
    result = subject.scan(conversation(size), explain=explain)
    assert result.is_complete and result.rule_matches == ["Read"]
    assert reads and all(value == size for _, value in reads)
    assert set(Counter(frame for frame, _ in reads).values()) == {1}


def test_separate_scans_keep_their_own_closed_operand_values(monkeypatch):
    subject = policy(
        'rule "Read" condition: count([c for c in assistant_messages[-1].tool_calls '
        'if len(assistant_messages[-1].tool_calls) == 8 and c.name == "lookup"]) == 8'
    )
    reads = observe_count_operands(monkeypatch)
    first = subject.scan(conversation(8))
    second = subject.scan(conversation(3))
    assert first.is_complete and first.rule_matches == ["Read"]
    assert second.is_complete and second.rule_matches == []
    assert Counter(value for _, value in reads) == {8: 1, 3: 1}


@pytest.mark.parametrize("expected", [7, 8])
def test_parameter_dependent_comparisons_still_read_each_item(monkeypatch, expected):
    subject = policy(
        'rule "Read" parameters: expected: number\n'
        "condition: count([c for c in assistant_messages[-1].tool_calls "
        "if len(assistant_messages[-1].tool_calls) == params.expected "
        'and c.name == "lookup"]) == 8'
    )
    reads = []
    original = model_expression._evaluate_children

    async def observed(expression, context, event, bindings, trace, backend):
        values = await original(expression, context, event, bindings, trace, backend)
        if expression.kind == "compare" and any(c.kind == "parameter" for c in expression.children):
            reads.append(values)
        return values

    monkeypatch.setattr(model_expression, "_evaluate_children", observed)
    result = subject.scan(conversation(8), parameters={"expected": expected})
    assert result.is_complete and result.rule_matches == (["Read"] if expected == 8 else [])
    assert reads == [(8, expected)] * 8


def test_item_dependent_name_reads_are_not_reused_across_calls(monkeypatch):
    subject = policy(
        'rule "Read" condition: count([c for c in assistant_messages[-1].tool_calls '
        'if c.name == "lookup"]) == 4'
    )
    names = []
    original = model_expression.field_access

    def observed(value, name):
        result = original(value, name)
        if name == "name":
            names.append(result)
        return result

    monkeypatch.setattr(model_expression, "field_access", observed)
    result = subject.scan(conversation(8, alternating=True))
    assert result.is_complete and result.rule_matches == ["Read"]
    assert names == ["lookup", "other"] * 4


def test_unreached_closed_operands_do_no_work(monkeypatch):
    subject = policy(
        'rule "Read" condition: count([c for c in assistant_messages[-1].tool_calls '
        'if c.name == "absent" and len(assistant_messages[-1].tool_calls) == 8]) == 0'
    )
    reads = observe_count_operands(monkeypatch)
    result = subject.scan(conversation(8))
    assert result.is_complete and result.rule_matches == ["Read"]
    assert reads == []


@pytest.mark.parametrize("inner_name", ["c", "d"])
def test_closed_nested_filters_keep_each_inner_item_and_lexical_binder(monkeypatch, inner_name):
    subject = policy(
        'rule "Read" condition: count([c for c in assistant_messages[-1].tool_calls '
        f"if len([{inner_name} for {inner_name} in assistant_messages[-1].tool_calls "
        f'if {inner_name}.name == "lookup"]) == 4 and c.name == "lookup"]) == 4'
    )
    names = []
    original = model_expression.field_access

    def observed(value, name):
        result = original(value, name)
        if name == "name":
            names.append(result)
        return result

    monkeypatch.setattr(model_expression, "field_access", observed)
    result = subject.scan(conversation(8, alternating=True), explain="support")
    assert result.is_complete and result.rule_matches == ["Read"]
    # The closed inner count is computed once. Both the inner and the outer
    # predicate still visit all eight distinct names, including a shadowed binder.
    assert names == ["lookup", "other"] * 8


def test_single_item_filter_does_not_admit_optional_reuse_metadata(monkeypatch):
    subject = policy(
        'rule "Read" condition: count([c for c in assistant_messages[-1].tool_calls '
        'if len(assistant_messages[-1].tool_calls) == 1 and c.name == "lookup"]) == 1'
    )
    classified = []
    original = model_expression._filter_immediate_plan

    async def observed(expression):
        classified.append(expression)
        return await original(expression)

    monkeypatch.setattr(model_expression, "_filter_immediate_plan", observed)
    result = subject.scan(conversation(1))
    assert result.is_complete and result.rule_matches == ["Read"]
    assert classified == []


@pytest.mark.asyncio
async def test_cancelled_scan_releases_already_completed_immediate_values(monkeypatch):
    from quarry.cache.immediate_values import ImmediateValueMemo

    subject = policy(
        'rule "Read" condition: count([c for c in assistant_messages[-1].tool_calls '
        'if len(assistant_messages[-1].tool_calls) == 8 and c.name == "lookup"]) == 8'
    )
    started = asyncio.Event()
    memos = []
    original = ImmediateValueMemo.keep

    async def observed(self, *args, **kwargs):
        await original(self, *args, **kwargs)
        memos.append(self)
        assert self.values and self.budget.bytes > 0
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(ImmediateValueMemo, "keep", observed)
    task = asyncio.create_task(subject.scan_async(conversation(8)))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert memos and all(
        memo.closed
        and not memo.values
        and not memo.plans
        and memo.budget.bytes == 0
        and memo.budget.entries == 0
        for memo in memos
    )


def test_concurrent_scans_keep_values_local_and_release_frame_owners(monkeypatch):
    from quarry.cache.immediate_values import ImmediateValueMemo

    subject = policy(
        'rule "Read" condition: count([c for c in assistant_messages[-1].tool_calls '
        'if len(assistant_messages[-1].tool_calls) == 8 and c.name == "lookup"]) == 8'
    )
    references = []
    original = ImmediateValueMemo.__init__

    def observe(self, *args, **kwargs):
        original(self, *args, **kwargs)
        references.append(weakref.ref(self))

    monkeypatch.setattr(ImmediateValueMemo, "__init__", observe)

    def scan(size):
        result = subject.scan(conversation(size), explain="support")
        return result.is_complete, result.rule_matches

    sizes = [8, 3, 8, 5] * 4
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(scan, sizes))
    assert results == [(True, ["Read"] if size == 8 else []) for size in sizes]
    assert references and all(reference() is None for reference in references)


def test_model_readers_are_called_for_every_required_filter_item(monkeypatch, tmp_path):
    from quarry.cache.async_cache import result_cache
    from quarry.integrations.model_inference import ModelInferenceBackend
    from tests.unit.model_fakes import ModelAdapter, NoWaitLimiter, model_entry, passthrough_builder

    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"reader": model_entry(score=True)}},
        tmp_path,
        require_rules=False,
    )
    adapter = ModelAdapter()
    backend = ModelInferenceBackend(
        config,
        adapters={"fake": adapter},
        message_builders={"test": passthrough_builder},
        rate_limiter=NoWaitLimiter(),
    )
    calls = []
    original = backend.model_score_messages

    async def observed(*args, **kwargs):
        calls.append(True)
        return await original(*args, **kwargs)

    monkeypatch.setattr(backend, "model_score_messages", observed)
    subject = compile_policy(
        'rule "Read" condition: count([c for c in assistant_messages[-1].tool_calls '
        'if model("reader").score(message("user", "fixed"), "synthetic") > 0.5]) == 8',
        runtime_config=config,
        model_inference_backend=backend,
    )
    result_cache.clear()
    result = subject.scan(conversation(8))
    assert result.is_complete and result.rule_matches == ["Read"]
    assert len(calls) == 8 and adapter.score_calls == 1
