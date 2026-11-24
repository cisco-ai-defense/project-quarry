# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Actual selected external reads keep incomplete input non-authoritative."""

import asyncio
import gc
import threading
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.integrations import model_usage
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.safety import timer
from quarry.safety.limits import SafetyLimits
from quarry.scanner import DSLScanner
from quarry.types.exceptions import ConfigurationError
from quarry.types.external_spans import ExternalReadRange, ExternalSpanRange, ExternalSpanReply
from tests.unit.model_fakes import NoWaitLimiter

DATA = {"messages": [{"role": "user", "content": "abcdef"}]}
COLLECT = (
    'rule "Read" condition: true collect: [s for s in model("reader").spans(messages, 0.5) if true]'
)


class ExternalOnly:
    supports_token_head = True  # An unrelated capability must never be requested.

    def __init__(self, reply=None):
        self.calls = 0
        self.reply = reply
        self.inputs = []

    async def external_spans(self, config, text, *, timeout=None):
        self.calls += 1
        self.inputs.append(text)
        if self.reply is not None:
            return self.reply(text)
        return ExternalSpanReply.for_input(
            text,
            (ExternalSpanRange(1, 3, 0.9),),
            (ExternalReadRange(0, len(text)),),
            complete=True,
        )


def policy(
    tmp_path,
    adapter=None,
    source=COLLECT,
    *,
    restored=False,
    spans=None,
    classify=None,
    limits=None,
    **extra,
):
    entry = {
        "adapter": "external",
        "model": "synthetic-v1",
        "category": "SYNTHETIC",
        "spans": {"backend": "external", **(spans or {})},
    }
    if classify is not None:
        entry["classify"] = classify
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"reader": entry}, **extra},
        tmp_path,
        require_rules=False,
    )
    config = replace(config, safety_limits=limits or SafetyLimits(rule_timeout=None))
    adapter = adapter or ExternalOnly()
    backend = ModelInferenceBackend(
        config, adapters={"external": adapter}, rate_limiter=NoWaitLimiter()
    )
    compiled = compile_policy(source, runtime_config=config, model_inference_backend=backend)
    if restored:
        raw = compiled.to_bytes()
        compiled = load_compiled_policy(raw, runtime_config=config, model_inference_backend=backend)
        assert compiled.to_bytes() == raw
    return compiled, adapter


@pytest.mark.parametrize("restored", [False, True])
def test_external_only_adapter_binds_native_spans_without_classifier_heads(tmp_path, restored):
    compiled, adapter = policy(tmp_path, restored=restored)
    result = compiled.scan(DATA, logical_reads=True, window_observations=True)
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]
    output = result.outputs["Read"]
    assert len(output.items) == 1
    found = output.items[0].value
    assert (found.message_index, found.start, found.end, found.score.value) == (0, 1, 3, 0.9)
    assert adapter.inputs == ["abcdef"]
    assert result.model_usage["records"] == [{"model": "reader", "operation": "spans", "calls": 1}]
    assert result.window_observations["rows"] == []


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("detections", [False, True])
@pytest.mark.parametrize("restored", [False, True])
def test_incomplete_reply_never_supplies_positive_or_negative_authority(
    tmp_path, complete, detections, restored
):
    def reply(text):
        return ExternalSpanReply.for_input(
            text,
            (ExternalSpanRange(0, 1, 0.9),) if detections else (),
            (ExternalReadRange(0, len(text) if complete else 1),),
            complete=complete,
        )

    compiled, adapter = policy(tmp_path, ExternalOnly(reply), restored=restored)
    result = compiled.scan(DATA, logical_reads=True)
    assert result.is_complete is complete
    if complete:
        if detections:
            assert len(result.outputs["Read"].items) == 1
        else:
            assert result.outputs == {}
    else:
        assert result.evaluation_error_categories["Read"] == "model_partial_read"
        assert not result.outputs
        row = next(r for r in result.logical_reads["rows"] if r["model"] == "reader")
        coverage = row["coverage"]["characters"]
        assert coverage["value"] is None and coverage["known_units"] == 1
        assert coverage["measurement_states"] == {"partial": 1}
    assert adapter.calls == 1


@pytest.mark.parametrize(
    "condition",
    [
        'all s in model("reader").spans(messages, 0.5) where true',
        'not any s in model("reader").spans(messages, 0.5) where true',
        'any s in model("reader").spans(messages, 0.5) where true',
    ],
)
def test_selected_partial_read_stays_unknown_under_negation_and_quantifiers(tmp_path, condition):
    adapter = ExternalOnly(lambda text: ExternalSpanReply.for_input(text, (), (), complete=False))
    compiled, _ = policy(tmp_path, adapter, source='rule "Read" condition: ' + condition)
    result = compiled.scan(DATA)
    assert not result.is_complete and not result.rule_matches
    assert result.evaluation_error_categories["Read"] == "model_partial_read"


def test_unselected_branch_does_not_dispatch_external_work(tmp_path):
    compiled, adapter = policy(
        tmp_path,
        source='rule "Read" condition: true or any s in model("reader").spans(messages, 0.5) where true',
    )
    result = compiled.scan(DATA)
    assert result.is_complete and result.rule_matches == ["Read"]
    assert adapter.calls == 0


@pytest.mark.parametrize(
    "reply",
    [
        lambda text: {"complete": True},
        lambda text: ExternalSpanReply(len(text), (), (ExternalReadRange(0, len(text)),), True),
        lambda text: ExternalSpanReply.for_input(
            "z" * len(text), (), (ExternalReadRange(0, len(text)),), complete=True
        ),
    ],
)
def test_invalid_missing_or_stale_request_binding_is_typed_and_not_retained(tmp_path, reply):
    compiled, adapter = policy(tmp_path, ExternalOnly(reply))
    for _ in range(2):
        result = compiled.scan(DATA)
        assert not result.is_complete and not result.outputs
        assert result.evaluation_error_categories["Read"] == "model_invalid_output"
    assert adapter.calls == 2


def test_partial_replies_are_not_reused_across_scans(tmp_path):
    compiled, adapter = policy(
        tmp_path,
        ExternalOnly(lambda text: ExternalSpanReply.for_input(text, (), (), complete=False)),
    )
    for _ in range(2):
        result = compiled.scan(DATA)
        assert result.evaluation_error_categories["Read"] == "model_partial_read"
    assert adapter.calls == 2


def test_complete_reuse_rebinds_to_each_actual_input_owner(tmp_path):
    compiled, adapter = policy(tmp_path)
    first = compiled.scan(DATA)
    second = compiled.scan({"messages": [{"role": "user", "content": "abcdef"}]})
    assert first.is_complete and second.is_complete
    assert adapter.calls == 1
    assert first.outputs["Read"].items[0].value == second.outputs["Read"].items[0].value
    assert (
        first.outputs["Read"].items[0].value._found_in[0]
        is not second.outputs["Read"].items[0].value._found_in[0]
    )


@pytest.mark.parametrize("ranges,known", [([], 0), ([(0, 1), (4, 6)], 3), ([(0, 2), (1, 3)], 3)])
def test_partial_coverage_preserves_gaps_and_empty_unknowns(tmp_path, ranges, known):
    adapter = ExternalOnly(
        lambda text: ExternalSpanReply.for_input(
            text,
            (),
            tuple(ExternalReadRange(*r) for r in ranges),
            complete=False,
        )
    )
    compiled, _ = policy(tmp_path, adapter)
    result = compiled.scan(DATA, logical_reads=True)
    assert result.evaluation_error_categories["Read"] == "model_partial_read"
    row = result.logical_reads["rows"][0]
    coverage = row["coverage"]["characters"]
    assert coverage["value"] is None and coverage["known_units"] == known
    assert coverage["measurement_states"] == {"partial": 1}


def test_unused_classifier_configuration_does_not_require_a_classifier_callback(tmp_path):
    compiled, adapter = policy(tmp_path, classify={"class_index": 0})
    assert compiled.scan(DATA).is_complete
    assert adapter.calls == 1


@pytest.mark.parametrize("restored", [False, True])
def test_selected_message_provenance_is_rebound_to_original_index(tmp_path, restored):
    compiled, adapter = policy(
        tmp_path,
        source=COLLECT.replace("spans(messages,", "spans(messages[1:],"),
        restored=restored,
    )
    data = {"messages": [{"role": "user", "content": "ignored"}, *DATA["messages"]]}
    result = compiled.scan(data, logical_reads=True)
    assert result.is_complete, result.evaluation_errors
    found = result.outputs["Read"].items[0].value
    assert (found.message_index, found.start, found.end) == (1, 1, 3)
    assert adapter.inputs == ["abcdef"]
    assert result.logical_reads["rows"][0]["coverage"]["characters"]["value"] == 6


@pytest.mark.parametrize("complete", [False, True])
async def test_shared_pending_reply_has_separate_logical_readers(tmp_path, monkeypatch, complete):
    started, gate = asyncio.Event(), asyncio.Event()
    entered = threading.Event()

    class Waiting(ExternalOnly):
        async def external_spans(self, config, text, *, timeout=None):
            self.calls += 1
            started.set()
            await gate.wait()
            return ExternalSpanReply.for_input(
                text, (), (ExternalReadRange(0, len(text)),), complete=complete
            )

    source = COLLECT.replace('"Read"', '"A"') + "\n" + COLLECT.replace('"Read"', '"B"')
    compiled, adapter = policy(tmp_path, Waiting())
    backend = compiled._scanner.resource_manager.model_inference_backend
    scanner = DSLScanner(
        replace(compiled._scanner.config, quarry_rule_texts=(source,)),
        model_inference_backend=backend,
    )
    call = backend._call
    reads = 0

    async def counted(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 2:
            entered.set()
        return await call(*args, **kwargs)

    monkeypatch.setattr(backend, "_call", counted)
    task = asyncio.create_task(scanner.scan_data_async(DATA, logical_reads=True))
    joined = False
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        joined = await asyncio.to_thread(entered.wait, 2)
    finally:
        gate.set()
        result = await asyncio.wait_for(task, timeout=2)
    assert joined
    assert result.is_complete is complete
    assert reads == 2 and adapter.calls == result.model_usage["calls"] == 1
    assert {row["rule"] for row in result.logical_reads["rows"]} == {"A", "B"}
    if not complete:
        assert result.evaluation_error_categories == {
            "A": "model_partial_read",
            "B": "model_partial_read",
        }


def test_input_character_bound_fails_before_any_host_callback(tmp_path):
    compiled, adapter = policy(tmp_path, spans={"limits": {"max_input_characters": 5}})
    result = compiled.scan(DATA)
    assert not result.is_complete
    assert result.evaluation_error_categories["Read"] == "evaluation_limit"
    assert adapter.calls == 0


def test_provider_failure_has_separate_bounded_category(tmp_path):
    class Broken(ExternalOnly):
        async def external_spans(self, config, text, *, timeout=None):
            raise RuntimeError("synthetic host detail must stay private")

    compiled, _ = policy(tmp_path, Broken())
    result = compiled.scan(DATA)
    assert result.evaluation_error_categories["Read"] == "model_provider"
    assert "synthetic host detail" not in str(result.evaluation_errors)


async def test_rejected_host_objects_are_not_retained_or_inspected(tmp_path):
    references = []

    class HostPacket:
        @property
        def complete(self):
            pytest.fail("a foreign host object's properties must not be inspected")

    def reply(text):
        packet = HostPacket()
        references.append(weakref.ref(packet))
        return packet

    compiled, _ = policy(tmp_path, ExternalOnly(reply))
    result = await compiled.scan_async(DATA)
    assert result.evaluation_error_categories["Read"] == "model_invalid_output"
    for _ in range(3):
        await asyncio.sleep(0)
        gc.collect()
    assert len(references) == 1 and references[0]() is None


def test_classification_capability_is_still_required_when_actually_selected(tmp_path):
    with pytest.raises(ConfigurationError):
        policy(
            tmp_path,
            source='rule "Read" condition: model("reader").classify(messages) > 0.5',
            classify={"class_index": 0},
        )


async def test_last_reader_cancellation_awaits_owned_host_cleanup(tmp_path):
    started, released = asyncio.Event(), asyncio.Event()

    class Waiting(ExternalOnly):
        async def external_spans(self, config, text, *, timeout=None):
            self.calls += 1
            started.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                released.set()

    compiled, _ = policy(tmp_path, Waiting())
    task = asyncio.create_task(compiled.scan_async(DATA))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released.is_set()


async def test_one_cancelled_waiter_preserves_the_other_readers_host_work(tmp_path, monkeypatch):
    started, gate, released = asyncio.Event(), asyncio.Event(), asyncio.Event()
    joined = threading.Event()

    class Waiting(ExternalOnly):
        async def external_spans(self, config, text, *, timeout=None):
            self.calls += 1
            started.set()
            try:
                await gate.wait()
                return ExternalSpanReply.for_input(
                    text, (), (ExternalReadRange(0, len(text)),), complete=True
                )
            finally:
                released.set()

    compiled, adapter = policy(tmp_path, Waiting())
    backend = compiled._scanner.resource_manager.model_inference_backend
    call = backend._call
    readers = 0

    async def counted(*args, **kwargs):
        nonlocal readers
        readers += 1
        if readers == 2:
            joined.set()
        return await call(*args, **kwargs)

    monkeypatch.setattr(backend, "_call", counted)
    first = asyncio.create_task(compiled.scan_async(DATA))
    second = None
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        second = asyncio.create_task(compiled.scan_async(DATA))
        assert await asyncio.to_thread(joined.wait, 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not released.is_set()
        gate.set()
        result = await asyncio.wait_for(second, timeout=2)
        assert result.is_complete and adapter.calls == 1
    finally:
        gate.set()
        await asyncio.gather(
            first, *([second] if second is not None else []), return_exceptions=True
        )


async def test_after_last_cancel_the_next_reader_uses_its_own_host_loop(tmp_path):
    started, released = asyncio.Event(), asyncio.Event()
    loops = []

    class WaitingOnce(ExternalOnly):
        async def external_spans(self, config, text, *, timeout=None):
            self.calls += 1
            loops.append(asyncio.get_running_loop())
            if self.calls == 1:
                started.set()
                try:
                    await asyncio.Future()
                finally:
                    released.set()
            return ExternalSpanReply.for_input(
                text, (), (ExternalReadRange(0, len(text)),), complete=True
            )

    compiled, adapter = policy(tmp_path, WaitingOnce())
    first = asyncio.create_task(compiled.scan_async(DATA))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
    finally:
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    assert released.is_set()
    result = await asyncio.to_thread(lambda: asyncio.run(compiled.scan_async(DATA)))
    assert result.is_complete and adapter.calls == 2
    assert len(loops) == 2 and loops[0] is not loops[1]


async def test_reply_at_the_inspection_deadline_never_returns_an_authoritative_result(
    tmp_path, monkeypatch
):
    now = timer.time.monotonic()
    deadlines = []

    def clock():
        return now

    monkeypatch.setattr(timer, "time", SimpleNamespace(monotonic=clock))
    monkeypatch.setattr(model_usage, "monotonic", clock)

    class Expiring(ExternalOnly):
        async def external_spans(self, config, text, *, timeout=None):
            nonlocal now
            reply = await super().external_spans(config, text, timeout=timeout)
            now = deadlines[0].expires_at
            return reply

    compiled, adapter = policy(tmp_path, Expiring())
    backend = compiled._scanner.resource_manager.model_inference_backend
    call = backend._call

    async def capture(*args, **kwargs):
        deadlines.append(timer.current_evaluation_deadline())
        return await call(*args, **kwargs)

    monkeypatch.setattr(backend, "_call", capture)
    result = await asyncio.wait_for(compiled.scan_async(DATA, deadline_seconds=10), timeout=2)
    assert not result.is_complete and not result.outputs
    assert result.timed_out_rules == ["Read"] and adapter.calls == 1


def test_external_dispatch_retains_the_actual_alias_call_budget(tmp_path):
    source = COLLECT.replace(
        'model("reader").spans(messages, 0.5)',
        'union(model("reader").spans(messages[:1], 0.5), model("reader").spans(messages[1:], 0.5))',
    )
    compiled, adapter = policy(
        tmp_path, source=source, limits=SafetyLimits(rule_timeout=None, max_model_calls=1)
    )
    result = compiled.scan({"messages": [*DATA["messages"], {"role": "user", "content": "ghijkl"}]})
    assert not result.is_complete and not result.outputs
    assert adapter.calls == result.model_usage["calls"] == 1
    assert "model-call limit" in result.evaluation_errors["Read"]
    assert result.evaluation_error_categories["Read"] == "model"
