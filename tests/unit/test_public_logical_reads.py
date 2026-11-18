# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Opt-in public snapshots preserve native verdicts, payloads and request ownership."""

import asyncio
import gc
import weakref
from dataclasses import replace

import pytest

import quarry.policy as policy_module
import quarry.scanner as scanner_module
from quarry import ReadUsageLimits, ReadUsageSession, TextTokenBasis, model_read_usage
from quarry.integrations import read_results
from quarry.integrations.read_usage import current_read_usage, read_usage_scope
from quarry.types.rules import ScanResult
from tests.unit.model_fakes import ModelAdapter, model_entry, v2_scanner
from tests.unit.test_read_usage import finish, measure
from tests.unit.test_scanner_read_usage_lifetimes import CHECK, DATA, policy


def summary(result):
    return result.logical_reads["aggregates"]["totals"]


def characters(result):
    return summary(result)["coverage"]["characters"]["value"]


def scanner(tmp_path):
    adapter = ModelAdapter()
    return v2_scanner(
        tmp_path, {"Outer": CHECK}, {"judge": model_entry(score=True)}, adapter
    ), adapter


async def test_default_scanner_and_policy_do_not_create_a_collection(tmp_path, monkeypatch):
    direct, adapter = scanner(tmp_path)
    compiled, second = policy(tmp_path, f'rule "Outer" condition: {CHECK}')

    def unexpected(*args, **kwargs):
        pytest.fail("default call created logical accounting")

    monkeypatch.setattr(ReadUsageSession, "__init__", unexpected)
    for result in [await direct.scan_data_async(DATA), await compiled.scan_async(DATA)]:
        assert result.is_complete and result.rule_matches == ["Outer"]
        assert result.logical_reads is None
    assert adapter.score_calls == second.score_calls == 1


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("batch", [False, True])
async def test_policy_reports_unique_conversation_scopes_after_native_cleanup(
    tmp_path, prepared, batch
):
    compiled, adapter = policy(tmp_path, f'rule "Outer" condition: {CHECK}', prepared=prepared)
    if batch:
        results = await compiled.scan_many_async([DATA, DATA, DATA], logical_reads=True)
    else:
        results = [await compiled.scan_async(DATA, logical_reads=True)]
    assert adapter.score_calls == 1
    assert current_read_usage() is None
    assert [result.logical_reads["scope"] for result in results] == list(range(1, len(results) + 1))
    for result in results:
        assert result.is_complete and result.rule_matches == ["Outer"]
        assert result.logical_reads["finalized"]
        assert {row["rule"] for row in result.logical_reads["rows"]} == {"Outer"}
        assert {row["scope"] for row in result.logical_reads["rows"]} == {
            result.logical_reads["scope"]
        }
        assert characters(result) == len("synthetic input") + len("synthetic policy")
        assert summary(result)["observations"] == summary(result)["evaluations"] == 1
        assert summary(result)["coverage"]["text_tokens"]["value"] is None
        assert not result.logical_reads["measurements_complete"]


def test_sync_public_entry_points_forward_the_option(tmp_path):
    direct, adapter = scanner(tmp_path)
    compiled, second = policy(tmp_path, f'rule "Outer" condition: {CHECK}')
    results = [
        direct.scan_data(DATA, logical_reads=True),
        compiled.scan(DATA, logical_reads=True),
        *compiled.scan_many([DATA, DATA], logical_reads=True),
    ]
    assert all(result.logical_reads["finalized"] and result.is_complete for result in results)
    assert adapter.score_calls == second.score_calls == 1


async def test_scanner_borrows_reference_maps_and_excludes_unrelated_reader_metadata(tmp_path):
    direct, adapter = scanner(tmp_path)
    with model_read_usage(token_bases={"judge": TextTokenBasis("synthetic-whole", "1")}) as session:
        reference = weakref.ref(session)
        assert session.provide_text_tokens("judge", "synthetic input", ((0, 15),))
        assert session.provide_text_tokens("judge", "synthetic policy", ((0, 16),))
        with read_usage_scope("unrelated"):
            finish(session, model="unrelated-model", chars=measure())
        result = await direct.scan_data_async(DATA, logical_reads=True)
        assert result.logical_reads["scope"] == 1
        assert not result.logical_reads["finalized"]
        assert result.logical_reads["aggregates"]["complete"]
        assert summary(result)["coverage"]["text_tokens"]["value"] == 2
        assert summary(result)["models"] == ["judge"]
        assert {row["rule"] for row in result.logical_reads["rows"]} == {"Outer"}
    assert session.report()["finalized"]
    assert not result.logical_reads["finalized"]  # Result is a snapshot, not a mutable live view.
    del session
    # Native completed model tasks release their inherited Context on the next loop turn.
    await asyncio.sleep(0)
    gc.collect()
    assert reference() is None
    assert adapter.score_calls == 1 and adapter.tokenizations == 0


@pytest.mark.parametrize("limits", [ReadUsageLimits(max_scopes=1), ReadUsageLimits(max_bindings=1)])
async def test_refused_scope_or_association_is_unknown_without_changing_verdicts(tmp_path, limits):
    compiled, adapter = policy(tmp_path, f'rule "Outer" condition: {CHECK}')
    with model_read_usage(limits):
        results = await compiled.scan_many_async([DATA, DATA, DATA], logical_reads=True)
    assert all(result.is_complete for result in results)
    assert adapter.score_calls == 1
    assert results[0].logical_reads["scope"] == 1
    for result in results[1:]:
        assert result.logical_reads["state"] == "scope_unavailable"
        assert result.logical_reads["rows"] == []
        assert characters(result) is None


@pytest.mark.parametrize("condition", ["true", "false"])
async def test_known_empty_model_coverage_is_zero_without_influencing_matches(tmp_path, condition):
    compiled, adapter = policy(tmp_path, f'rule "Outer" condition: {condition}')
    result = await compiled.scan_async(DATA, logical_reads=True)
    assert result.is_complete
    assert result.rule_matches == (["Outer"] if condition == "true" else [])
    assert result.logical_reads["aggregates"]["complete"]
    assert characters(result) == 0
    assert summary(result)["observations"] == 0
    assert adapter.score_calls == 0


async def test_whole_input_validation_still_fails_closed_with_no_model_requests(tmp_path):
    direct, adapter = scanner(tmp_path)
    result = await direct.scan_data_async({"messages": "invalid"}, logical_reads=True)
    assert result.validation_error and not result.is_complete
    assert result.logical_reads["scope"] == 1 and result.logical_reads["finalized"]
    assert characters(result) == 0 and summary(result)["observations"] == 0
    assert adapter.score_calls == 0


@pytest.mark.parametrize("after_scan", [False, True])
async def test_batch_timeouts_distinguish_a_scanned_conversation_from_unrun_items(
    tmp_path, monkeypatch, after_scan
):
    compiled, adapter = policy(tmp_path, f'rule "Outer" condition: {CHECK}')
    remaining = iter([1.0, -1.0] if after_scan else [-1.0])
    monkeypatch.setattr(policy_module, "remaining_evaluation_time", lambda: next(remaining))
    results = await compiled.scan_many_async([DATA, DATA, DATA], logical_reads=True)
    assert all(result.timed_out_rules == ["Outer"] and not result.is_complete for result in results)
    assert adapter.score_calls == int(after_scan)
    if after_scan:
        assert results[0].logical_reads["scope"] == 1
        assert characters(results[0]) == 31
    for result in results[int(after_scan) :]:
        assert result.logical_reads["state"] == "scope_unavailable"
        assert characters(result) is None
        assert result.logical_reads["rows"] == []


async def test_deferred_replacement_keeps_original_scope_and_includes_evidence_reads(
    tmp_path, monkeypatch
):
    compiled, adapter = policy(tmp_path, f'rule "Outer" condition: {CHECK}')
    original = scanner_module.evaluate_async

    async def evaluate(*args, **kwargs):
        result = await original(*args, **kwargs)
        if not kwargs.get("defer_minimization"):
            return result
        previous = result.refine

        async def refine(time_left):
            active = current_read_usage()
            finish(active, model="synthetic-evidence", chars=measure(end=3))
            return result if previous is None else await previous(time_left)

        return replace(result, refine=refine)

    monkeypatch.setattr(scanner_module, "evaluate_async", evaluate)
    results = await compiled.scan_many_async([DATA, DATA], explain=True, logical_reads=True)
    assert adapter.score_calls == 1
    for index, result in enumerate(results, 1):
        assert result.is_complete and result.logical_reads["scope"] == index
        assert result.logical_reads["finalized"]
        assert {row["scope"] for row in result.logical_reads["rows"]} == {index}
        assert {row["phase"] for row in result.logical_reads["rows"]} == {"decision", "evidence"}
        assert summary(result)["observations"] == 2
        assert characters(result) == 34


async def test_per_result_reports_do_not_share_mutable_basis_labels(tmp_path):
    compiled, _ = policy(tmp_path, f'rule "Outer" condition: {CHECK}')
    with model_read_usage(token_bases={"judge": TextTokenBasis("synthetic-whole", "1")}):
        first, second = await compiled.scan_many_async([DATA, DATA], logical_reads=True)
        first.logical_reads["token_bases"][0]["name"] = "changed"
        first.logical_reads["aggregates"]["totals"]["models"].clear()
        assert second.logical_reads["token_bases"][0]["name"] == "synthetic-whole"
        assert summary(second)["models"] == ["judge"]


@pytest.mark.parametrize("flag", [None, 1, "yes", object()])
async def test_option_rejects_non_boolean_host_values_without_dispatch(tmp_path, flag):
    direct, adapter = scanner(tmp_path)
    compiled, second = policy(tmp_path, f'rule "Outer" condition: {CHECK}')
    for call in [direct.scan_data_async, compiled.scan_async]:
        with pytest.raises(ValueError, match="logical_reads must be a boolean"):
            await call(DATA, logical_reads=flag)
    assert adapter.score_calls == second.score_calls == 0


def test_result_associations_are_weak_bounded_and_guarded_before_inherited_locks(monkeypatch):
    class Poison:
        def __enter__(self):
            pytest.fail("inherited report lock touched")

    with model_read_usage(ReadUsageLimits(max_bindings=1)) as session:
        collection = read_results._ReadReports(session)
        result = ScanResult(None, [], 0.0)
        reference = weakref.ref(result)
        collection.record(result, 1)
        del result
        gc.collect()
        assert reference() is None
        collection.record(ScanResult(None, [], 0.0), 2)
        assert len(collection._scopes) == 1
        collection._lock = Poison()
        collection.note_read_scope(object(), 0)
        monkeypatch.setattr(read_results.os, "getpid", lambda: collection._pid + 1)
        collection.observe_scope(3)
        collection.note_read_scope(session, 0)
        collection.stop()
        collection.record(ScanResult(None, [], 0.0), 3)
        collection.transfer(ScanResult(None, [], 0.0), ScanResult(None, [], 0.0))


async def test_cancellation_closes_the_owned_session_and_all_native_work(tmp_path):
    entered = asyncio.Event()
    active = []

    class Slow(ModelAdapter):
        async def score(self, *args, **kwargs):
            active.append(weakref.ref(current_read_usage()))
            entered.set()
            await asyncio.Event().wait()

    direct = v2_scanner(tmp_path, {"Outer": CHECK}, {"judge": model_entry(score=True)}, Slow())
    task = asyncio.create_task(direct.scan_data_async(DATA, logical_reads=True))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert current_read_usage() is None
    assert all(reference() is None or reference()._closed for reference in active)
