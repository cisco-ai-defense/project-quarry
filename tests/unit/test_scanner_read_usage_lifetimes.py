# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Actual scanner ownership, scopes and logical producers with synthetic adapters."""

import asyncio
from dataclasses import replace

import pytest

import quarry.scanner as scanner_module
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.integrations.read_usage import (
    ReadUsageLimits,
    current_read_usage,
    model_read_usage,
)
from tests.unit.model_fakes import ModelAdapter, model_entry, passthrough_builder, v2_scanner
from tests.unit.test_nested_rule_runtime_contracts import _policy

DATA = {"messages": [{"role": "user", "content": "synthetic input"}]}
CHECK = 'model("judge").score(messages, "synthetic policy") > 0.5'


@pytest.fixture
def logical_producer():
    # The production evaluator/backend now emits observations. Preserve these
    # lifetime cases while ensuring no synthetic producer masks missing coverage.
    from quarry.integrations.read_operations import current_model_read

    assert current_model_read() is None


def policy(tmp_path, source, *, prepared=False):
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"judge": model_entry(score=True)}},
        tmp_path,
        require_rules=False,
    )
    adapter = ModelAdapter()
    backend = ModelInferenceBackend(
        config, adapters={"fake": adapter}, message_builders={"test": passthrough_builder}
    )
    return _policy(source, config, prepared, backend), adapter


async def test_concurrent_top_level_rules_keep_requesting_names_with_one_physical_reply(
    tmp_path, logical_producer
):
    adapter = ModelAdapter()
    scanner = v2_scanner(
        tmp_path, {"First": CHECK, "Second": CHECK}, {"judge": model_entry(score=True)}, adapter
    )
    with model_read_usage() as session:
        result = await scanner.scan_data_async(DATA)
    assert result.is_complete and sorted(result.rule_matches) == ["First", "Second"]
    assert adapter.score_calls == 1
    rows = session.snapshot()["rows"]
    assert [
        (row["scope"], row["rule"], row["phase"], row["model"], row["observations"]) for row in rows
    ] == [(1, "First", "decision", "judge", 1), (1, "Second", "decision", "judge", 1)]


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("kind", ["direct", "called", "computed"])
async def test_nested_models_are_attributed_to_the_requesting_entry_rule(
    tmp_path, logical_producer, prepared, kind
):
    if kind == "called":
        source = f'rule "Leaf" condition: {CHECK}\nrule "Outer" imports: leaf = "Leaf" condition: leaf(messages)'
    elif kind == "computed":
        source = (
            f'rule "Leaf" condition: {CHECK}\n'
            'rule "Outer" imports: leaf = "Leaf" '
            "condition: any m in (messages + messages) where leaf(m)"
        )
    else:
        source = f'rule "Outer" condition: {CHECK}'
    compiled, adapter = policy(tmp_path, source, prepared=prepared)
    if kind == "computed":
        assert compiled._scanner.compiled_rules["Outer"].computed_conditions
    with model_read_usage() as session:
        result = await compiled.scan_async(DATA)
    assert result.is_complete and result.rule_matches == ["Outer"]
    rows = session.snapshot()["rows"]
    assert rows and {row["rule"] for row in rows} == {"Outer"}
    assert {row["scope"] for row in rows} == {1}
    assert {row["model"] for row in rows} == {"judge"}
    assert adapter.score_calls == 1


@pytest.mark.parametrize("limit", [None, "owners", "scopes"])
async def test_repeated_batch_inputs_stay_distinct_without_changing_provider_work(
    tmp_path, logical_producer, limit
):
    compiled, adapter = policy(tmp_path, f'rule "Outer" condition: {CHECK}')
    limits = ReadUsageLimits(**({f"max_{limit}": 1} if limit else {}))
    with model_read_usage(limits) as session:
        results = await compiled.scan_many_async([DATA, DATA, DATA])
        assert session._owners == 1
    assert all(result.is_complete and result.rule_matches == ["Outer"] for result in results)
    assert adapter.score_calls == 1
    report = session.snapshot()
    expected = {1} if limit == "scopes" else {1, 2, 3}
    assert {row["scope"] for row in report["rows"]} == expected
    assert all(row["rule"] == "Outer" and row["observations"] == 1 for row in report["rows"])
    assert report["observations_complete"] == (limit is None)
    assert session._closed and not session._events and not session._bindings


@pytest.mark.parametrize("finish_mode", ["success", "discard", "error", "cancel"])
async def test_deferred_work_resumes_original_metadata_and_releases_on_every_exit(
    tmp_path, monkeypatch, logical_producer, finish_mode
):
    compiled, adapter = policy(tmp_path, f'rule "Outer" condition: {CHECK}')
    original = scanner_module.evaluate_async
    entered = asyncio.Event()

    async def evaluate(*args, **kwargs):
        result = await original(*args, **kwargs)
        if not kwargs.get("defer_minimization"):
            return result
        previous = result.refine

        async def refine(time_left):
            active = current_read_usage()
            assert active is not None
            marker = active.begin("synthetic-evidence", "score")
            assert marker is not None
            entered.set()
            try:
                if finish_mode == "cancel":
                    await asyncio.Event().wait()
                if finish_mode == "error":
                    raise RuntimeError("synthetic evidence error")
                return result if previous is None else await previous(time_left)
            except BaseException as error:
                marker.finish(
                    "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
                )
                raise
            finally:
                marker.finish("succeeded")

        return replace(result, refine=refine)

    monkeypatch.setattr(scanner_module, "evaluate_async", evaluate)
    callbacks = []
    with model_read_usage() as session:
        result = await compiled._scanner._scan_data_with_parameters_async(
            DATA, {}, enable_explain=True, deferred_evidence=callbacks
        )
        assert result.is_complete and result.explanations is None
    assert not session._closed and session._owners == 1 and len(callbacks) == 1
    try:
        with model_read_usage() as unrelated:
            if finish_mode == "discard":
                callbacks[0].close()
            else:
                task = asyncio.create_task(callbacks[0](result))
                try:
                    if finish_mode == "cancel":
                        await asyncio.wait_for(entered.wait(), 2)
                        task.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await task
                    else:
                        finished = await task
                        assert finished.is_complete and finished.rule_matches == ["Outer"]
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            assert current_read_usage() is unrelated
        assert unrelated.snapshot()["rows"] == []
        assert adapter.score_calls == 1 and result.rule_matches == ["Outer"]
        assert session._closed and session._owners == 0
        rows = session.snapshot()["rows"]
        assert {row["scope"] for row in rows} == {1}
        assert {row["rule"] for row in rows} == {"Outer"}
        markers = [row for row in rows if row["model"] == "synthetic-evidence"]
        if finish_mode == "discard":
            assert markers == []
        else:
            assert [(row["phase"], row["outcome"]) for row in markers] == [
                (
                    "evidence",
                    {"success": "succeeded", "error": "failed", "cancel": "cancelled"}[finish_mode],
                )
            ]
    finally:
        for callback in callbacks:
            callback.close()


async def test_deferred_work_created_with_accounting_off_does_not_adopt_later_request(
    tmp_path, monkeypatch
):
    compiled, adapter = policy(tmp_path, f'rule "Outer" condition: {CHECK}')
    original = scanner_module.evaluate_async
    resumed = []

    async def evaluate(*args, **kwargs):
        result = await original(*args, **kwargs)
        if not kwargs.get("defer_minimization"):
            return result

        async def refine(_time_left):
            resumed.append(current_read_usage())
            return result

        return replace(result, refine=refine)

    monkeypatch.setattr(scanner_module, "evaluate_async", evaluate)
    callbacks = []
    result = await compiled._scanner._scan_data_with_parameters_async(
        DATA, {}, enable_explain=True, deferred_evidence=callbacks
    )
    try:
        with model_read_usage() as session:
            assert (await callbacks[0](result)).rule_matches == ["Outer"]
        assert resumed == [None] and session.snapshot()["rows"] == []
        assert adapter.score_calls == 1
    finally:
        for callback in callbacks:
            callback.close()


async def test_failed_deferred_handoff_releases_its_metadata_owner(tmp_path, logical_producer):
    class Reject:
        def append(self, _owner):
            raise RuntimeError("synthetic deferred handoff failure")

    compiled, adapter = policy(tmp_path, f'rule "Outer" condition: {CHECK}')
    with model_read_usage() as session:
        with pytest.raises(RuntimeError, match="synthetic deferred handoff failure"):
            await compiled._scanner._scan_data_with_parameters_async(
                DATA, {}, enable_explain=True, deferred_evidence=Reject()
            )
        assert session._owners == 1
    assert session._closed and adapter.score_calls == 1


async def test_cancelled_batch_finisher_stops_siblings_before_releasing_metadata(
    tmp_path, monkeypatch, logical_producer
):
    compiled, adapter = policy(tmp_path, f'rule "Outer" condition: {CHECK}')
    original = scanner_module.evaluate_async
    first_started = asyncio.Event()
    running = set()
    evaluated = 0

    async def evaluate(*args, **kwargs):
        nonlocal evaluated
        result = await original(*args, **kwargs)
        if not kwargs.get("defer_minimization"):
            return result
        index = evaluated
        evaluated += 1

        async def refine(_time_left):
            running.add(index)
            try:
                if index == 0:
                    first_started.set()
                    await asyncio.Event().wait()
                await first_started.wait()
                raise asyncio.CancelledError
            finally:
                running.remove(index)

        return replace(result, refine=refine)

    monkeypatch.setattr(scanner_module, "evaluate_async", evaluate)
    with model_read_usage() as session:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(compiled.scan_many_async([DATA, DATA, DATA], explain=True), 3)
        assert not running and session._owners == 1
        assert evaluated == 3 and adapter.score_calls == 1
    assert session._closed and not session._events
