# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Deferred child explanations own and bound their complete runtime snapshots."""

import asyncio
from dataclasses import replace

import pytest

import quarry.cache.message_values as value_module
import quarry.scanner as scanner_module
import quarry.types.context as context_module
from quarry.engine.contracts import ScanRecord
from quarry.engine.events import ConversationEvent, StaticEnvironment, build_evaluation_input
from quarry.types.context import ScanContext
from quarry.types.messages import Message, Messages
from tests.unit.test_computed_message_conditions import PAYLOAD, prepare
from tests.unit.test_message_value_lifetimes import assert_released, observe_budgets


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["resolve", "discard", "cancel"])
async def test_deferred_snapshot_owns_record_and_events_until_its_owner_finishes(
    tmp_path, monkeypatch, finish
):
    budgets = observe_budgets(monkeypatch)
    policy, _ = prepare(
        tmp_path,
        'any m in messages.transform("convert") where m.content contains "block"',
        False,
    )
    entered = asyncio.Event()
    original = scanner_module.evaluate_async

    async def evaluate(*args, **kwargs):
        result = await original(*args, **kwargs)
        if finish == "cancel" and kwargs.get("defer_minimization"):

            async def wait(_time_left):
                entered.set()
                await asyncio.Event().wait()

            return replace(result, refine=wait)
        return result

    monkeypatch.setattr(scanner_module, "evaluate_async", evaluate)
    callbacks = []
    result = await policy._scanner._scan_data_with_parameters_async(
        PAYLOAD, {}, enable_explain=True, deferred_evidence=callbacks
    )
    assert result.is_complete and len(callbacks) == 1
    assert result.explanations is None
    retained = [
        node.value
        for budget in budgets
        if budget.limit == context_module.MAX_VALUE_EVIDENCE_BYTES
        for node in budget._objects.values()
    ]
    assert any(type(value) is ScanRecord for value in retained)
    assert any(type(value) is ConversationEvent for value in retained)
    if finish == "discard":
        callbacks[0].close()
    else:
        task = asyncio.create_task(callbacks[0](result))
        if finish == "cancel":
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert (await task).rule_matches == ["R"]
    assert_released(budgets)


@pytest.mark.asyncio
@pytest.mark.parametrize("prepared", [False, True])
async def test_explanation_metadata_can_exhaust_storage_when_the_source_alone_fits(
    tmp_path, monkeypatch, prepared
):
    from quarry.cache.message_values import MessageValueBudget, _retained_bytes

    graphs = []
    original = MessageValueBudget._retain_graph

    def observed(self, graph):
        lease = original(self, graph)
        if self.limit == context_module.MAX_VALUE_EVIDENCE_BYTES:
            graphs.append((graph, self.bytes))
        return lease

    monkeypatch.setattr(MessageValueBudget, "_retain_graph", observed)
    budgets = observe_budgets(monkeypatch)
    policy, _ = prepare(
        tmp_path,
        'any m in messages.transform("convert") where m.content contains "block"',
        prepared,
    )
    callbacks = []
    result = await policy._scanner._scan_data_with_parameters_async(
        PAYLOAD, {}, enable_explain=True, deferred_evidence=callbacks
    )
    assert result.is_complete and graphs
    graph, charged = graphs[-1]
    sources = [value for value, _ in graph.values() if type(value) is Messages]
    source_bytes = max([await _retained_bytes(value, 10**9) for value in sources])
    assert charged > source_bytes
    await callbacks[0](result)
    assert_released(budgets)
    monkeypatch.setattr(context_module, "MAX_VALUE_EVIDENCE_BYTES", (source_bytes + charged) // 2)
    capped = await policy.scan_async(PAYLOAD, explain=True)
    assert capped.is_complete and capped.rule_matches == ["R"]
    assert capped.explanations[0].omitted_value_conditions > 0
    assert_released(budgets)


@pytest.mark.parametrize("prepared", [False, True])
def test_snapshot_work_exhaustion_omits_details_and_preserves_the_verdict(
    tmp_path, monkeypatch, prepared
):
    monkeypatch.setattr(value_module, "MAX_VALUE_EVIDENCE_REFERENCES", 1)
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


@pytest.mark.parametrize("hostile_metaclass", [False, True])
def test_snapshot_storage_walk_never_runs_field_or_host_hooks(monkeypatch, hostile_metaclass):
    from quarry.cache.message_values import MessageValueBudget

    context = ScanContext([{"role": "user", "content": "block"}])
    evaluation = build_evaluation_input(context)
    record = ScanRecord()

    class Meta(type):
        def __hash__(cls):
            raise AssertionError("snapshot must not hash a host type")

        def __eq__(cls, other):
            raise AssertionError("snapshot must not compare host types")

    class Host(metaclass=Meta if hostile_metaclass else type):
        def __getattribute__(self, name):
            raise AssertionError("snapshot must not read host attributes")

        def __sizeof__(self):
            raise AssertionError("snapshot must not invoke host sizing")

    def forbidden(*args, **kwargs):
        raise AssertionError("snapshot must not semantically read message text")

    original = StaticEnvironment.__getattribute__

    def guard(self, name):
        if name == "system_prompt":
            return forbidden()
        return original(self, name)

    monkeypatch.setattr(StaticEnvironment, "__getattribute__", guard)
    monkeypatch.setattr(Message, "content", property(forbidden))
    budget = MessageValueBudget(10**6)
    lease = budget.try_retain_snapshot((context.conversation.messages, evaluation, record, Host()))
    assert lease is not None and budget.bytes > 0
    lease.close()
    assert budget.bytes == 0 and not budget._objects


def test_snapshot_work_refusal_does_not_evict_an_existing_owner():
    from quarry.cache.message_values import MessageValueBudget

    budget = MessageValueBudget(10**6)
    first = budget.try_retain_snapshot(("retained",))
    assert first is not None
    charged = budget.bytes
    assert budget.try_retain_snapshot([None] * 100_000) is None
    assert budget.bytes == charged
    first.close()
    assert budget.bytes == 0 and not budget._objects
