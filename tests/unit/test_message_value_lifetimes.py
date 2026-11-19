# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Live message values share rule storage without sharing frame cache identities."""

import asyncio
from dataclasses import replace

import pytest

import quarry.cache.message_values as value_module
import quarry.scanner as scanner_module
import quarry.types.context as context_module
from quarry.cache.message_values import MessageValueMemo
from quarry.engine.events import build_events
from quarry.types.context import ScanContext
from quarry.types.messages import Messages
from tests.unit.test_computed_message_conditions import PAYLOAD, FoldingAdapter, prepare
from tests.unit.test_transform_value_freshness import policy_for


def messages(text):
    return Messages.from_openai([{"role": "user", "content": text}])


async def retain(memo, key, value, **kwargs):
    async def compute():
        return value

    return await memo.evaluate((key,), compute, **kwargs)


def observe_budgets(monkeypatch):
    from quarry.cache.message_values import MessageValueBudget

    budgets = []
    original = MessageValueBudget.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        budgets.append(self)

    monkeypatch.setattr(MessageValueBudget, "__init__", initialize)
    return budgets


def assert_released(budgets):
    assert budgets and all(
        budget.bytes == 0 and budget.entries == 0 and budget._objects == {} for budget in budgets
    )


def test_child_frames_share_only_storage_accounting():
    root = ScanContext(PAYLOAD["messages"])
    child = root.for_messages(root.conversation.messages[:1])
    sibling = root.for_messages(root.conversation.messages[1:])
    other_rule = root.copy()
    assert child.message_value_memo is not root.message_value_memo
    assert sibling.message_value_memo is not child.message_value_memo
    assert (
        child.message_value_memo.budget
        is sibling.message_value_memo.budget
        is root.message_value_memo.budget
    )
    assert other_rule.message_value_memo.budget is not root.message_value_memo.budget


@pytest.mark.asyncio
async def test_shared_graph_storage_stays_charged_until_its_last_owner():
    from quarry.cache.message_values import MessageValueBudget

    budget = MessageValueBudget()
    value = messages("retained" * 100)
    first = await budget.retain(value)
    charged = budget.bytes
    second = await budget.retain(value)
    assert charged > 1000 and budget.bytes == charged
    first.close()
    first.close()
    assert budget.bytes == charged
    second.close()
    assert budget.bytes == 0 and not budget._objects


@pytest.mark.asyncio
async def test_child_cache_keys_remain_local_and_reuse_original_objects():
    from quarry.cache.message_values import MessageValueBudget

    budget = MessageValueBudget()
    parent, child = MessageValueMemo(budget), MessageValueMemo(budget)
    one, two = messages("one"), messages("two")
    assert await retain(parent, "same-key", one) is one
    assert await retain(child, "same-key", two) is two
    assert await retain(parent, "same-key", two) is one
    child.close()
    assert budget.bytes > 0 and await retain(parent, "same-key", two) is one
    parent.close()
    assert budget.bytes == 0


@pytest.mark.asyncio
async def test_entry_limit_is_shared_by_child_frames(monkeypatch):
    from quarry.cache.message_values import MessageValueBudget

    monkeypatch.setattr(value_module, "MAX_MESSAGE_VALUES", 2)
    budget = MessageValueBudget()
    parent, child = MessageValueMemo(budget), MessageValueMemo(budget)
    value = messages("shared storage")
    await retain(parent, "parent", value)
    await retain(child, "child", value)
    assert budget.entries == 2
    with pytest.raises(ValueError, match="per-rule entry limit"):
        await retain(child, "third", value)
    child.close()
    assert budget.entries == 1
    await retain(parent, "replacement", value)
    parent.close()
    assert budget.entries == 0 and budget.bytes == 0


@pytest.mark.asyncio
async def test_aggregate_limit_refuses_a_child_without_evicting_parent():
    from quarry.cache.message_values import MessageValueBudget

    budget = MessageValueBudget()
    parent, child = MessageValueMemo(budget), MessageValueMemo(budget)
    one, two = messages("parent-" * 1000), messages("child-" * 1000)
    await retain(parent, "parent", one)
    parent_size = budget.bytes
    await retain(child, "child", two)
    combined = budget.bytes
    child.close()
    assert budget.bytes == parent_size
    budget.limit = (parent_size + combined) // 2
    with pytest.raises(ValueError, match="per-rule byte limit"):
        await retain(child, "child", two)
    assert budget.bytes == parent_size and await retain(parent, "parent", two) is one
    child.close()
    parent.close()
    assert budget.bytes == 0


@pytest.mark.asyncio
async def test_concurrent_graph_installation_rechecks_the_shared_limit(monkeypatch):
    from quarry.cache.message_values import MessageValueBudget, MessageValueLease

    budget = MessageValueBudget()
    one, two = messages("first-" * 1000), messages("second-" * 1000)
    first = await budget.retain(one)
    first_size = budget.bytes
    second = await budget.retain(two)
    combined = budget.bytes
    first.close()
    second.close()
    budget.limit = (first_size + combined) // 2
    original = value_module._retained_graph
    arrived, release = asyncio.Event(), asyncio.Event()
    walks = 0

    async def synchronized(value, limit):
        nonlocal walks
        graph = await original(value, limit)
        walks += 1
        if walks == 2:
            arrived.set()
        await release.wait()
        return graph

    monkeypatch.setattr(value_module, "_retained_graph", synchronized)
    tasks = [asyncio.create_task(budget.retain(value)) for value in [one, two]]
    await asyncio.wait_for(arrived.wait(), 2)
    release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert sum(isinstance(result, MessageValueLease) for result in results) == 1
    assert sum(isinstance(result, ValueError) for result in results) == 1
    assert budget.bytes <= budget.limit
    for result in results:
        if isinstance(result, MessageValueLease):
            result.close()
    assert budget.bytes == 0


@pytest.mark.asyncio
async def test_cancelled_measurement_never_installs_a_partial_graph(monkeypatch):
    from quarry.cache.message_values import MessageValueBudget

    budget = MessageValueBudget()
    started = asyncio.Event()
    original = value_module._retained_graph

    async def waiting(value, limit):
        graph = await original(value, limit)
        started.set()
        await asyncio.Event().wait()
        return graph

    monkeypatch.setattr(value_module, "_retained_graph", waiting)
    task = asyncio.create_task(budget.retain(messages("cancelled")))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert budget.bytes == 0 and not budget._objects


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_in_progress_map_parts_are_owned_until_completion(outcome):
    memo = MessageValueMemo()
    part = messages("held part" * 1000)
    started, release = asyncio.Event(), asyncio.Event()

    async def compute():
        await memo.retain_part(part)
        started.set()
        await release.wait()
        if outcome == "failure":
            raise ValueError("synthetic failure")
        return part

    task = asyncio.create_task(memo.evaluate(("whole-map",), compute))
    await asyncio.wait_for(started.wait(), 2)
    assert memo.budget.bytes > 8000
    if outcome == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        release.set()
        if outcome == "failure":
            with pytest.raises(ValueError, match="synthetic failure"):
                await task
        else:
            assert await task is part and memo.budget.bytes > 8000
    memo.close()
    assert memo.budget.bytes == 0 and not memo.budget._objects


@pytest.mark.asyncio
async def test_temporary_source_remains_charged_after_its_cache_scope_is_released(tmp_path):
    from quarry.integrations.model_expression import scoped_model_expression

    policy, _ = prepare(
        tmp_path,
        'all outer in messages where any inner in message("user", outer.content).transform("convert") where inner.content contains "block"',
        False,
    )
    context = ScanContext(PAYLOAD["messages"], resource_manager=policy._scanner.resource_manager)
    (plan,) = policy._scanner.compiled_rules["R"].computed_conditions.values()
    async with scoped_model_expression(plan.source, context, event=build_events(context)[0]) as (
        selected,
        _,
    ):
        assert not context.message_value_memo._entries
        budget = context.message_value_memo.budget
        assert budget.bytes > 0 and selected[0].content == "block"
        child = context.for_messages(selected)
        budget.limit = budget.bytes
        with pytest.raises(ValueError, match="per-rule byte limit"):
            await retain(child.message_value_memo, "new child", messages("new child"))
        child.message_value_memo.close()
        assert budget.bytes > 0
    assert context.message_value_memo.budget.bytes == 0


@pytest.mark.asyncio
async def test_evidence_ownership_does_not_consume_the_decision_budget():
    from quarry.cache.message_values import MessageValueBudget
    from quarry.engine.witness import DeferredValueEvidence

    decisions, evidence = MessageValueBudget(), MessageValueBudget()
    lease = await decisions.retain(messages("evidence" * 1000))
    snapshot = lease.copy_to(evidence)
    lease.close()
    assert decisions.bytes == 0 and evidence.bytes > 8000
    holder = DeferredValueEvidence(lambda: {"ready": True}, release=snapshot.close)
    assert holder.resolve() == {"ready": True}
    assert evidence.bytes == 0 and holder.factory is None and holder.release is None


@pytest.mark.asyncio
@pytest.mark.parametrize("resolve", [False, True])
async def test_unused_or_failed_evidence_releases_its_runtime_snapshot(resolve):
    from quarry.cache.message_values import MessageValueBudget
    from quarry.engine.witness import DeferredValueEvidence

    budget = MessageValueBudget()
    lease = await budget.retain(messages("private synthetic input"))

    def broken():
        raise ValueError("synthetic failure")

    holder = DeferredValueEvidence(broken, release=lease.close)
    if resolve:
        assert holder.resolve() == {"unavailable": True}
    holder.discard()
    holder.discard()
    assert budget.bytes == 0 and holder.factory is None


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("explain", [False, True])
def test_finished_scan_releases_child_and_root_values(tmp_path, monkeypatch, prepared, explain):
    budgets = observe_budgets(monkeypatch)
    policy, adapter = prepare(
        tmp_path,
        'any m in messages.transform("convert") where any n in messages.transform("convert") where n.content contains "block"',
        prepared,
    )
    result = policy.scan(PAYLOAD, explain=explain)
    assert result.is_complete and result.rule_matches == ["R"]
    # The equal inner source is rebased to the already computed outer value.
    assert len(adapter.parts) == 2
    assert_released(budgets)


@pytest.mark.parametrize("prepared", [False, True])
def test_exhausted_explanation_storage_preserves_the_verdict(tmp_path, monkeypatch, prepared):
    monkeypatch.setattr(context_module, "MAX_VALUE_EVIDENCE_BYTES", 0)
    budgets = observe_budgets(monkeypatch)
    policy, adapter = prepare(
        tmp_path,
        'any m in messages.transform("convert") where m.content contains "block"',
        prepared,
    )
    plain = policy.scan(PAYLOAD)
    explained = policy.scan(PAYLOAD, explain=True)
    assert plain.is_complete and explained.is_complete
    assert plain.rule_matches == explained.rule_matches == ["R"]
    assert len(adapter.parts) == 4
    assert explained.explanations[0].omitted_value_conditions > 0
    assert_released(budgets)


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("explain", [False, True])
def test_real_child_evaluation_enforces_the_aggregate_live_graph_limit(
    tmp_path, monkeypatch, prepared, explain
):
    from quarry.cache.message_values import MessageValueBudget

    budgets = observe_budgets(monkeypatch)
    observations = []
    original = MessageValueBudget._retain_graph

    def observed(self, graph):
        result = original(self, graph)
        if self.limit is None:
            observations.append((self.bytes, sum(size for _, size in graph.values())))
        return result

    monkeypatch.setattr(MessageValueBudget, "_retain_graph", observed)
    policy, adapter = policy_for(
        tmp_path,
        'any m in messages.transform("first") where any n in m.transform("second") where n.content contains "block"',
        prepared=prepared,
        definitions={"first": {"adapter": "fake"}, "second": {"adapter": "fake"}},
    )
    result = policy.scan(PAYLOAD, explain=explain)
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(adapter.parts) == 3
    peak = max(total for total, _ in observations)
    largest_graph = max(size for _, size in observations)
    assert peak > largest_graph
    assert_released(budgets)
    # Every single retained graph fits; only the simultaneous parent/child ownership
    # crosses this reduced cap. No production-scale allocation is needed.
    monkeypatch.setattr(value_module, "RESULT_CACHE_MAX_BYTES", (peak + largest_graph) // 2)
    adapter.parts.clear()
    refused = policy.scan(PAYLOAD, explain=explain)
    assert not refused.is_complete and not refused.rule_matches
    assert "per-rule byte limit" in refused.evaluation_errors["R"]
    assert_released(budgets)


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_mode", ["success", "cancel"])
async def test_deferred_evidence_owns_values_until_it_finishes(tmp_path, monkeypatch, finish_mode):
    budgets = observe_budgets(monkeypatch)
    policy, _ = prepare(
        tmp_path, 'any m in messages.transform("convert") where m.content contains "block"', False
    )
    original = scanner_module.evaluate_async
    started = asyncio.Event()

    async def evaluate(*args, **kwargs):
        result = await original(*args, **kwargs)
        if finish_mode == "cancel" and kwargs.get("defer_minimization"):

            async def waiting(_time_left):
                started.set()
                await asyncio.Event().wait()

            return replace(result, refine=waiting)
        return result

    monkeypatch.setattr(scanner_module, "evaluate_async", evaluate)
    callbacks = []
    result = await policy._scanner._scan_data_with_parameters_async(
        PAYLOAD, {}, enable_explain=True, deferred_evidence=callbacks
    )
    assert result.is_complete and len(callbacks) == 1
    assert any(budget.bytes > 0 for budget in budgets)
    task = asyncio.create_task(callbacks[0](result))
    if finish_mode == "cancel":
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        finished = await task
        assert finished.rule_matches == ["R"]
    assert_released(budgets)


@pytest.mark.asyncio
async def test_cancelled_scan_closes_all_rule_value_ownership(tmp_path, monkeypatch):
    budgets = observe_budgets(monkeypatch)
    started = asyncio.Event()

    class WaitingAdapter(FoldingAdapter):
        async def transform(self, config, part, *, timeout=None):
            if self.parts:
                started.set()
                await asyncio.Event().wait()
            self.parts.append(part)
            return part["text"].casefold()

    policy, _ = prepare(
        tmp_path,
        'any m in messages.transform("convert") where m.content contains "block"',
        False,
        adapter=WaitingAdapter(),
    )
    task = asyncio.create_task(policy.scan_async(PAYLOAD))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert_released(budgets)
