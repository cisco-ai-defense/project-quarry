# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Completed scalar reuse has bounded storage, weak owners and honest evidence."""

import asyncio
import gc
import weakref

import pytest

import quarry.safety.timer as timer
from quarry.cache.immediate_values import ImmediateValueMemo
from quarry.cache.message_values import MAX_MESSAGE_VALUES, MessageValueBudget
from quarry.engine.atoms import ModelExpression as Expr
from quarry.engine.matchers import MISSING, RegexBackend
from quarry.integrations import model_expression
from quarry.integrations.read_usage import current_read_usage, model_read_usage, read_usage_scope
from quarry.safety.limits import SafetyLimits
from quarry.types.context import ScanContext
from tests.unit.test_message_value_read_usage import measured_read


def context():
    return ScanContext(
        [{"role": "user", "content": "synthetic"}], limits=SafetyLimits(rule_timeout=None)
    )


def addition():
    return Expr("+", children=(Expr("literal", 2), Expr("literal", 3)))


@pytest.mark.asyncio
@pytest.mark.parametrize("comparison", [False, True])
async def test_each_cached_expression_replays_its_own_complete_evidence(comparison):
    ctx = context()
    expression = (
        Expr("compare", ">", (addition(), Expr("literal", 3))) if comparison else addition()
    )
    trace = model_expression.ModelExpressionTrace(expression_value=99, operator="==", threshold=99)
    expected = (5, ">", 3) if comparison else (None, None, None)
    for stale in [99, 88]:
        trace.expression_value, trace.operator, trace.threshold = stale, "==", stale
        value = await model_expression._evaluate_filter(expression, ctx, trace=trace)
        assert value == (True if comparison else 5)
        assert (trace.expression_value, trace.operator, trace.threshold) == expected
    ctx.close_message_values()
    assert ctx._immediate_values.budget.bytes == ctx._immediate_values.budget.entries == 0


@pytest.mark.asyncio
async def test_regex_backend_identity_and_failed_work_do_not_become_shared_answers():
    class Backend(RegexBackend):
        def __init__(self, reject=False, fail=False):
            self.reject, self.fail, self.calls = reject, fail, 0

        def search_match(self, *args, **kwargs):
            self.calls += 1
            if self.fail:
                self.fail = False
                raise ValueError("synthetic regex failure")
            return None if self.reject else super().search_match(*args, **kwargs)

    ctx = context()
    expression = Expr("string_compare", "~", (Expr("literal", "abc"), Expr("literal", "abc")))
    first, second = Backend(), Backend(reject=True)
    for backend, expected in [(first, True), (second, False), (first, True), (second, False)]:
        assert (
            await model_expression._evaluate_filter(expression, ctx, regex_backend=backend)
            is expected
        )
    assert (first.calls, second.calls) == (1, 1)
    failing = Backend(fail=True)
    with pytest.raises(ValueError, match="synthetic regex failure"):
        await model_expression._evaluate_filter(expression, ctx, regex_backend=failing)
    assert await model_expression._evaluate_filter(expression, ctx, regex_backend=failing) is True
    assert failing.calls == 2
    ctx.close_message_values()


@pytest.mark.asyncio
async def test_missing_is_a_completed_value_and_does_not_become_an_uncomputed_entry(monkeypatch):
    ctx = context()
    expression = Expr("index", 99, (Expr("messages"),))
    calls = []
    original = model_expression._evaluate_uncached_model_expression

    async def observed(node, *args, **kwargs):
        if node is expression:
            calls.append(True)
        return await original(node, *args, **kwargs)

    monkeypatch.setattr(model_expression, "_evaluate_uncached_model_expression", observed)
    for _ in range(2):
        assert await model_expression._evaluate_filter(expression, ctx) is MISSING
    assert len(calls) == 1
    ctx.close_message_values()


@pytest.mark.asyncio
async def test_reused_measurement_is_attributed_to_the_current_logical_reader():
    class Measured(RegexBackend):
        calls = 0

        def search_match(self, *args, **kwargs):
            self.calls += 1
            measured_read(current_read_usage())
            return super().search_match(*args, **kwargs)

    ctx = context()
    expression = Expr("string_compare", "~", (Expr("literal", "abc"), Expr("literal", "abc")))
    backend = Measured()
    with model_read_usage() as session:
        for name in ["first", "second"]:
            with read_usage_scope(name):
                assert await model_expression._evaluate_filter(
                    expression, ctx, regex_backend=backend
                )
        rows = session.snapshot()["rows"]
        reference = weakref.ref(session)
    assert backend.calls == 1
    assert [row["rule"] for row in rows] == ["first", "second"]
    assert all(row["coverage"]["characters"]["value"] == 12 for row in rows)
    del session
    gc.collect()
    assert reference() is None
    ctx.close_message_values()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_stopped_classification_cannot_install_a_late_plan(cancel):
    memo = ImmediateValueMemo()
    expression = addition()
    entered, release = asyncio.Event(), asyncio.Event()

    async def classify(_):
        entered.set()
        await release.wait()
        return True

    task = asyncio.create_task(memo.eligible(expression, classify))
    await entered.wait()
    if cancel:
        task.cancel()
    else:
        memo.close()
    release.set()
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await task
    memo.close()
    assert not memo.plans and not memo.values
    assert memo.budget.bytes == memo.budget.entries == 0


@pytest.mark.asyncio
async def test_release_during_value_admission_does_not_leak_a_late_lease(monkeypatch):
    memo = ImmediateValueMemo()
    expression, backend = addition(), RegexBackend()
    entered, release = asyncio.Event(), asyncio.Event()
    original = memo.budget.retain

    async def blocked(value):
        entered.set()
        await release.wait()
        return await original(value)

    monkeypatch.setattr(memo.budget, "retain", blocked)
    task = asyncio.create_task(memo.keep(expression, backend, (5, (None, None, None), ()), None))
    await entered.wait()
    memo.close()
    release.set()
    with pytest.raises(RuntimeError, match="released"):
        await task
    assert not memo.values and memo.budget.bytes == memo.budget.entries == 0


@pytest.mark.asyncio
async def test_sibling_frames_share_an_allowance_without_holding_their_expression_owners():
    budget = MessageValueBudget(5000)
    first, second = ImmediateValueMemo(budget), ImmediateValueMemo(budget)
    backend = RegexBackend()
    nodes = [Expr("literal", index) for index in range(64)]
    for index, node in enumerate(nodes):
        memo = first if index % 2 else second
        await memo.keep(node, backend, (index, (None, None, None), ()), None)
        assert budget.bytes <= 5000 and budget.entries <= MAX_MESSAGE_VALUES
    assert first.values and second.values
    refs = [weakref.ref(node) for node in nodes]
    del node, nodes
    gc.collect()
    assert all(reference() is None for reference in refs)
    first.close()
    assert budget.bytes > 0
    second.close()
    assert budget.bytes == budget.entries == 0


@pytest.mark.asyncio
async def test_storage_exhaustion_keeps_the_uncached_result_and_existing_message_budget():
    ctx = context()
    original_budget = ctx.message_value_memo.budget
    ctx._immediate_budget = MessageValueBudget(1)
    for _ in range(2):
        assert await model_expression._evaluate_filter(addition(), ctx) == 5
    assert ctx._immediate_budget.bytes == ctx._immediate_budget.entries == 0
    assert ctx.message_value_memo.budget is original_budget
    assert original_budget.bytes == original_budget.entries == 0
    ctx.close_message_values()


@pytest.mark.asyncio
async def test_entry_allowance_is_aggregate_across_sibling_frames():
    budget = MessageValueBudget(1 << 20)
    first, second = ImmediateValueMemo(budget), ImmediateValueMemo(budget)
    backend = RegexBackend()
    nodes = [Expr("literal", index) for index in range(MAX_MESSAGE_VALUES + 32)]
    for index, node in enumerate(nodes):
        memo = first if index % 2 else second
        await memo.keep(node, backend, (index, (None, None, None), ()), None)
        assert budget.entries <= MAX_MESSAGE_VALUES and budget.bytes <= 1 << 20
    assert len(first.values) + len(second.values) == MAX_MESSAGE_VALUES
    first.close()
    second.close()
    assert budget.bytes == budget.entries == 0


@pytest.mark.asyncio
async def test_reuse_without_prior_measurement_marks_unknown_instead_of_measured_zero():
    ctx = context()
    expression = addition()
    assert await model_expression._evaluate_filter(expression, ctx) == 5
    with model_read_usage() as session:
        with read_usage_scope("later"):
            assert await model_expression._evaluate_filter(expression, ctx) == 5
        assert not session.snapshot()["observations_complete"]
    ctx.close_message_values()


@pytest.mark.asyncio
async def test_selected_frames_keep_their_own_inputs_and_release_shared_storage(monkeypatch):
    root = ScanContext(
        [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}],
        limits=SafetyLimits(rule_timeout=None),
    )
    first = root.for_messages(root.conversation.messages[:1])
    second = root.for_messages(root.conversation.messages[1:])
    other_rule = root.copy()
    expression = Expr(
        "string_compare",
        "~",
        (
            Expr("field", "content", (Expr("index", 0, (Expr("messages"),)),)),
            Expr("literal", "one"),
        ),
    )
    reads = []
    original = model_expression.field_access

    def observed(value, name):
        result = original(value, name)
        if name == "content":
            reads.append(result)
        return result

    monkeypatch.setattr(model_expression, "field_access", observed)
    for ctx, expected in [(root, True), (first, True), (second, False), (other_rule, True)]:
        for _ in range(2):
            assert await model_expression._evaluate_filter(expression, ctx) is expected
    assert reads == ["one", "one", "two", "one"]
    shared = root._immediate_budget
    assert first._immediate_budget is second._immediate_budget is shared
    assert other_rule._immediate_budget is not shared
    first.close_message_values()
    assert shared.bytes > 0
    second.close_message_values()
    root.close_message_values()
    other_rule.close_message_values()
    assert shared.bytes == shared.entries == 0
    assert other_rule._immediate_budget.bytes == other_rule._immediate_budget.entries == 0


@pytest.mark.asyncio
async def test_an_expired_frame_cannot_return_an_already_cached_value(monkeypatch):
    ctx = context()
    expression = addition()
    assert await model_expression._evaluate_filter(expression, ctx) == 5
    monkeypatch.setattr(
        timer,
        "current_evaluation_deadline",
        lambda: timer.EvaluationDeadline(0, "synthetic expiry"),
    )
    with pytest.raises(timer.QuarryTimeoutError, match="synthetic expiry"):
        await model_expression._evaluate_filter(expression, ctx)
    ctx.close_message_values()
    assert ctx._immediate_budget.bytes == ctx._immediate_budget.entries == 0


@pytest.mark.asyncio
async def test_expiry_during_storage_admission_cannot_publish_a_value(monkeypatch):
    memo = ImmediateValueMemo()
    expression, backend = addition(), RegexBackend()
    entered, release = asyncio.Event(), asyncio.Event()
    original = memo.budget.retain

    async def blocked(value):
        lease = await original(value)
        entered.set()
        await release.wait()
        return lease

    monkeypatch.setattr(memo.budget, "retain", blocked)
    task = asyncio.create_task(memo.keep(expression, backend, (5, (None, None, None), ()), None))
    await asyncio.wait_for(entered.wait(), 2)
    assert memo.budget.bytes > 0
    monkeypatch.setattr(
        timer,
        "current_evaluation_deadline",
        lambda: timer.EvaluationDeadline(0, "synthetic expiry"),
    )
    release.set()
    with pytest.raises(timer.QuarryTimeoutError, match="synthetic expiry"):
        await task
    assert not memo.values and memo.budget.bytes == memo.budget.entries == 0
    memo.close()
