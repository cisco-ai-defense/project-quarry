# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Opt-in observations describe accepted native replies without changing evaluation."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from quarry import (
    WindowObservationLimits,
    current_rule_name,
    observe_model_windows,
    window_observation_report_schema,
)
from quarry.cache.async_cache import result_cache, scan_scope
from quarry.integrations.read_usage import read_usage_scope
from quarry.types.exceptions import MODEL_REFUSED, LLMError
from quarry.types.messages import Messages
from tests.unit.model_fakes import model_entry, v2_scanner
from tests.unit.test_model_token_head_contracts import backend
from tests.unit.test_model_window_evidence import EvidenceAdapter, policy
from tests.unit.test_model_window_read_usage import MappedAdapter

DATA = {
    "messages": [
        {"role": "user", "content": "PRIVATE"},
        {"role": "assistant", "content": "abcdef"},
    ]
}


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["characters", "tokens"])
@pytest.mark.parametrize("restored", [False, True])
async def test_observed_reply_is_detached_and_covers_only_the_selected_input(
    tmp_path, unit, restored
):
    compiled, adapter = policy(
        tmp_path,
        'model("classes").classify(messages) > 0.5',
        unit=unit,
        restored=restored,
        adapter=MappedAdapter(),
    )
    result_cache.clear()
    with observe_model_windows() as observations:
        result = await compiled.scan_async(DATA)
        live = observations.report()
        assert live["finalized"] is False
    report = observations.report()
    Draft202012Validator(window_observation_report_schema()).validate(report)
    assert report["finalized"] is True and report["omitted"] == 0
    assert result.is_complete and result.rule_matches == ["R"], result.evaluation_errors
    assert result.logical_reads is None
    (row,) = report["rows"]
    assert (row["rule"], row["model"], row["phase"], row["operation"], row["outcome"]) == (
        "R",
        "classes",
        "decision",
        "classify",
        "succeeded",
    )
    assert (row["unit"], row["window_count"], row["state"], row["cached"], row["reads"]) == (
        unit,
        3,
        "measured",
        False,
        1,
    )
    assert row["source_characters"] == {"state": "measured", "value": 6}
    assert row["covered_tokens"] == (
        {"state": "measured", "value": 6}
        if unit == "tokens"
        else {"state": "unavailable", "value": None}
    )
    assert row["probabilities"] == [[0.3, 0.2], [0.8, 0.2], [0.8, 0.2]]
    assert row["probability_states"] == ["measured"] * 3
    wire = json.dumps(report)
    assert "PRIVATE" not in wire and "abcdef" not in wire and "token_ids" not in wire
    assert len(adapter.requests) == 1 and len(adapter.encodings) == (unit == "tokens")
    row["probabilities"][0][0] = 999
    row["source_characters"]["value"] = 999
    assert observations.report()["rows"][0]["probabilities"][0][0] == 0.3
    assert observations.report()["rows"][0]["source_characters"]["value"] == 6


@pytest.mark.asyncio
async def test_unknown_character_offsets_are_explicit_without_an_extra_encoding(tmp_path):
    compiled, adapter = policy(tmp_path, 'model("classes").classify(messages) >= 0', unit="tokens")
    result_cache.clear()
    with observe_model_windows() as observations:
        result = await compiled.scan_async(DATA)
    (row,) = observations.report()["rows"]
    assert result.is_complete
    assert row["source_characters"] == {"state": "unavailable", "value": None}
    assert row["covered_tokens"] == {"state": "measured", "value": 6}
    assert len(adapter.requests) == len(adapter.encodings) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["characters", "tokens"])
async def test_empty_input_reports_zero_without_a_provider_call(tmp_path, unit):
    compiled, adapter = policy(tmp_path, 'model("classes").classify(messages) == 0', unit=unit)
    result_cache.clear()
    with observe_model_windows() as observations:
        result = await compiled.scan_async({"messages": [{"role": "user", "content": ""}]})
    (row,) = observations.report()["rows"]
    assert result.is_complete
    assert (row["state"], row["window_count"], row["outcome"], row["cached"]) == (
        "not_sent",
        0,
        "not_sent",
        None,
    )
    assert row["probabilities"] == row["probability_states"] == []
    assert row["source_characters"] == row["covered_tokens"] == {"state": "measured", "value": 0}
    assert adapter.requests == adapter.encodings == []


@pytest.mark.asyncio
async def test_aliases_rules_and_cache_hits_replay_the_accepted_reply(tmp_path):
    windows = {"unit": "characters", "size": 3, "step": 2, "scope": "last_turn"}
    adapter = EvidenceAdapter()
    compiled = v2_scanner(
        tmp_path,
        {
            "First": 'model("classes").classify(messages) > 0.5',
            "Second": 'model("same").classify(messages) > 0.5',
            "Other": 'model("other").classify(messages) < 0.5',
        },
        {
            name: {**model_entry(classify=index), "windows": windows}
            for name, index in (("classes", 0), ("same", 0), ("other", 1))
        },
        adapter,
    )
    result_cache.clear()
    reports = []
    for _ in range(2):
        with observe_model_windows() as observations:
            result = await compiled.scan_data_async(DATA)
        reports.append(observations.report())
        assert result.is_complete and set(result.rule_matches) == {"First", "Second", "Other"}
    assert len(adapter.requests) == 1
    for report in reports:
        assert len(report["rows"]) == 3
        assert {(r["rule"], r["model"]) for r in report["rows"]} == {
            ("First", "classes"),
            ("Second", "same"),
            ("Other", "other"),
        }
        assert all(
            r["state"] == "measured" and r["source_characters"]["value"] == 6
            for r in report["rows"]
        )
        assert all(
            r["probabilities"] == [[0.3, 0.2], [0.8, 0.2], [0.8, 0.2]] for r in report["rows"]
        )
    assert sum(r["cached"] is False for r in reports[0]["rows"]) == 1
    assert all(r["cached"] is True for r in reports[1]["rows"])


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["unused_nan", "unused_object", "unused_empty"])
async def test_invalid_unused_classes_do_not_change_native_selected_class_verdict(tmp_path, fault):
    class Opaque:
        def __float__(self):
            raise AssertionError("observation must not invoke a numeric conversion")

    class Adapter(EvidenceAdapter):
        async def classify_window_batch(self, *args, **kwargs):
            raw = await super().classify_window_batch(*args, **kwargs)
            for chunk in raw.chunks:
                chunk.probabilities[1] = {
                    "unused_nan": float("nan"),
                    "unused_object": Opaque(),
                    "unused_empty": [],
                }[fault]
            return raw

    compiled, adapter = policy(
        tmp_path, 'model("classes").classify(messages) > 0.5', adapter=Adapter()
    )
    result_cache.clear()
    with observe_model_windows() as observations:
        result = await compiled.scan_async(DATA)
    (row,) = observations.report()["rows"]
    assert result.is_complete and result.rule_matches == ["R"]
    assert row["state"] == "partial" and row["probabilities"] == [None] * 3
    assert row["probability_states"] == ["invalid_probability_vector"] * 3
    assert len(adapter.requests) == 1


@pytest.mark.asyncio
async def test_selected_probability_property_is_not_read_an_extra_time(tmp_path):
    accesses = []

    class Chunk:
        @property
        def probabilities(self):
            accesses.append(1)
            return [0.8, 0.2]

    class Adapter(EvidenceAdapter):
        async def classify_window_batch(self, *args, **kwargs):
            raw = await super().classify_window_batch(*args, **kwargs)
            return SimpleNamespace(chunks=[Chunk() for _ in raw.chunks])

    compiled, _ = policy(tmp_path, 'model("classes").classify(messages) > 0.5', adapter=Adapter())
    for enabled in (False, True):
        result_cache.clear()
        accesses.clear()
        if enabled:
            with observe_model_windows() as observations:
                result = await compiled.scan_async(DATA)
            assert observations.report()["rows"][0]["probabilities"] == [[0.8, 0.2]] * 3
        else:
            result = await compiled.scan_async(DATA)
        assert result.is_complete and len(accesses) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limits",
    [
        WindowObservationLimits(max_work=1),
        WindowObservationLimits(max_windows=1),
        WindowObservationLimits(max_probabilities=1),
        WindowObservationLimits(max_label_bytes=1),
    ],
)
async def test_exhausted_optional_limits_leave_the_verdict_and_requests_unchanged(tmp_path, limits):
    compiled, adapter = policy(tmp_path, 'model("classes").classify(messages) > 0.5')
    result_cache.clear()
    with observe_model_windows(limits) as observations:
        result = await compiled.scan_async(DATA)
    report = observations.report()
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(adapter.requests) == 1
    assert all(
        r["probabilities"] is None or all(v is None for v in r["probabilities"])
        for r in report["rows"]
    )
    assert len(json.dumps(report)) < 2000


@pytest.mark.parametrize("field", list(WindowObservationLimits.__dataclass_fields__))
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "1", 10**20])
def test_limits_reject_non_native_positive_bounded_integers(field, value):
    with pytest.raises(ValueError):
        replace(WindowObservationLimits(), **{field: value})


@pytest.mark.parametrize("failure", ["encoding", "provider", "refusal", "shape", "selected"])
async def test_reached_failures_keep_unknowns_and_never_publish_unaccepted_probabilities(
    tmp_path, failure
):
    class Adapter(EvidenceAdapter):
        async def encode_window_input(self, *args, **kwargs):
            if failure == "encoding":
                self.encodings.append(1)
                raise LLMError("synthetic encoding failure", {"code": "invalid_output"})
            return await super().encode_window_input(*args, **kwargs)

        async def classify_window_batch(self, *args, **kwargs):
            raw = await super().classify_window_batch(*args, **kwargs)
            if failure in ("provider", "refusal"):
                raise LLMError(
                    "synthetic failure",
                    {"code": MODEL_REFUSED if failure == "refusal" else "invalid_output"},
                )
            if failure == "shape":
                raw.chunks.pop()
            if failure == "selected":
                raw.chunks[-1].probabilities[0] = float("nan")
            return raw

    compiled, adapter = policy(
        tmp_path,
        'model("classes").classify(messages) > 0.5',
        unit="tokens" if failure == "encoding" else "characters",
        adapter=Adapter(),
    )
    result_cache.clear()
    with observe_model_windows() as observations:
        result = await compiled.scan_async(DATA)
    report = observations.report()
    Draft202012Validator(window_observation_report_schema()).validate(report)
    assert not result.is_complete and result.rule_matches == []
    (row,) = report["rows"]
    # Native logical accounting reserves the distinct refusal outcome for score/complete.
    # A classifier refusal remains its existing failed read; observations do not relabel it.
    assert row["state"] == row["outcome"] == "failed"
    assert row["probabilities"] is None and row["probability_states"] is None
    assert row["window_count"] == (None if failure == "encoding" else 3)
    assert row["source_characters"]["value"] == (None if failure == "encoding" else 6)
    assert len(adapter.requests) == (0 if failure == "encoding" else 1)


@pytest.mark.parametrize("unit", ["characters", "tokens"])
@pytest.mark.parametrize("span_first", [False, True])
async def test_native_class_and_span_readers_observe_the_same_accepted_head_without_extra_calls(
    tmp_path, unit, span_first
):
    windows = {"unit": unit, "size": 3, "step": 2, "scope": "last_turn"}
    adapter = MappedAdapter()
    service = backend(
        tmp_path,
        {
            "classes": {**model_entry(classify=0), "windows": windows},
            "segments": {**model_entry(spans=0, category="SYNTHETIC"), "windows": windows},
        },
        adapter,
    )
    messages = Messages.from_openai(DATA["messages"])
    result_cache.clear()
    with observe_model_windows() as observations, scan_scope(), read_usage_scope("R"):
        if span_first:
            spans = await service.model_span_messages("segments", messages, 0.5)
            score = await service.model_classify_messages("classes", messages)
        else:
            score = await service.model_classify_messages("classes", messages)
            spans = await service.model_span_messages("segments", messages, 0.5)
    report = observations.report()
    Draft202012Validator(window_observation_report_schema()).validate(report)
    assert score == 0.8 and len(spans.items) == 3
    assert len(adapter.requests) == len(adapter.decodes) == 1
    assert len(adapter.encodings) == (unit == "tokens")
    assert {r["model"] for r in report["rows"]} == {"classes", "segments"}
    assert sum(r["cached"] is False for r in report["rows"]) == 1
    assert all(r["probabilities"] == [[0.3, 0.2], [0.8, 0.2], [0.8, 0.2]] for r in report["rows"])
    assert all(r["source_characters"]["value"] == 6 for r in report["rows"])


async def test_span_reader_does_not_inspect_unused_probability_properties(tmp_path):
    class Chunk:
        @property
        def probabilities(self):
            raise AssertionError("span evaluation does not read this head")

    class Adapter(MappedAdapter):
        async def classify_window_batch(self, *args, **kwargs):
            raw = await super().classify_window_batch(*args, **kwargs)
            raw.chunks = [Chunk() for _ in raw.chunks]
            return raw

    adapter = Adapter()
    service = backend(
        tmp_path,
        {
            "segments": {
                **model_entry(spans=0, category="SYNTHETIC"),
                "windows": {"unit": "tokens", "size": 3, "step": 2, "scope": "last_turn"},
            }
        },
        adapter,
    )
    result_cache.clear()
    with observe_model_windows() as observations, scan_scope(), read_usage_scope("R"):
        spans = await service.model_span_messages(
            "segments", Messages.from_openai(DATA["messages"]), 0.5
        )
    (row,) = observations.report()["rows"]
    assert len(spans.items) == 3 and len(adapter.requests) == 1
    assert row["state"] == "partial" and row["probabilities"] == [None] * 3
    assert row["probability_states"] == ["unavailable"] * 3


async def test_cancelled_starter_does_not_claim_live_rule_ownership_for_a_surviving_waiter(
    tmp_path,
):
    entered, release = asyncio.Event(), asyncio.Event()
    names = []

    class Adapter(EvidenceAdapter):
        async def classify_window_batch(self, *args, **kwargs):
            names.append(current_rule_name())
            entered.set()
            await release.wait()
            names.append(current_rule_name())
            return await super().classify_window_batch(*args, **kwargs)

    adapter = Adapter()
    service = backend(
        tmp_path,
        {
            "classes": {
                **model_entry(classify=0),
                "windows": {"unit": "characters", "size": 3, "step": 2, "scope": "last_turn"},
            }
        },
        adapter,
    )
    messages = Messages.from_openai(DATA["messages"])

    async def run(name):
        with scan_scope(), read_usage_scope(name):
            return await service.model_classify_messages("classes", messages)

    result_cache.clear()
    tasks = []
    try:
        with observe_model_windows() as observations:
            tasks.append(asyncio.create_task(run("Starter")))
            await asyncio.wait_for(entered.wait(), 3)
            tasks.append(asyncio.create_task(run("Waiter")))
            async with asyncio.timeout(3):
                while not any(
                    getattr(entry, "waiters", 0) == 2
                    for entry in tuple(result_cache.store.values())
                ):
                    await asyncio.sleep(0)
            tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
            assert not tasks[1].done()
            release.set()
            assert await asyncio.wait_for(tasks[1], 3) == 0.8
        report = observations.report()
        Draft202012Validator(window_observation_report_schema()).validate(report)
        rows = {r["rule"]: r for r in report["rows"]}
        assert rows["Starter"]["state"] == "cancelled"
        assert rows["Starter"]["probabilities"] is None
        assert rows["Waiter"]["state"] == "measured" and rows["Waiter"]["cached"] is True
        assert rows["Waiter"]["probabilities"] == [[0.3, 0.2], [0.8, 0.2], [0.8, 0.2]]
        assert names == ["Starter", None]
        assert len(adapter.requests) == 1 and current_rule_name() is None
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_repeated_same_cut_reads_do_not_multiply_coverage_or_vectors(tmp_path):
    call = 'model("classes").classify(messages)'
    compiled, adapter = policy(tmp_path, f"{call} == {call}")
    result_cache.clear()
    with observe_model_windows() as observations:
        result = await compiled.scan_async(DATA)
    (row,) = observations.report()["rows"]
    assert result.is_complete and result.rule_matches == ["R"]
    assert row["reads"] == 2 and row["window_count"] == 3
    assert row["source_characters"]["value"] == 6
    assert row["probabilities"] == [[0.3, 0.2], [0.8, 0.2], [0.8, 0.2]]
    assert len(adapter.requests) == 1


async def test_report_copy_budget_is_shared_across_aliases_and_does_not_copy_over_limit_states(
    tmp_path,
):
    windows = {"unit": "characters", "size": 3, "step": 2, "scope": "last_turn"}
    adapter = EvidenceAdapter()
    direct = v2_scanner(
        tmp_path,
        {name: f'model("{name}").classify(messages) > 0.5' for name in ("A", "B", "C")},
        {name: {**model_entry(classify=0), "windows": windows} for name in ("A", "B", "C")},
        adapter,
    )
    result_cache.clear()
    with observe_model_windows(WindowObservationLimits(max_probabilities=6)) as observations:
        result = await direct.scan_data_async(DATA)
    report = observations.report()
    assert result.is_complete and set(result.rule_matches) == {"A", "B", "C"}
    assert len(adapter.requests) == 1 and len(report["rows"]) == 3
    assert sum(r["state"] == "measured" for r in report["rows"]) == 1
    for row in report["rows"]:
        if row["state"] == "limit_exceeded":
            assert row["probabilities"] is None and row["probability_states"] is None
    assert (
        sum(len(v) for r in report["rows"] for v in r["probabilities"] or [] if v is not None) == 6
    )


@pytest.mark.parametrize("limit", ["max_events", "max_rows", "max_replies"])
async def test_frame_reply_and_row_caps_do_not_silently_report_complete_empty_reads(
    tmp_path, limit
):
    compiled, adapter = policy(tmp_path, 'model("classes").classify(messages) > 0.5')
    result_cache.clear()
    with observe_model_windows(replace(WindowObservationLimits(), **{limit: 1})) as observations:
        results = await compiled.scan_many_async(
            [DATA, {"messages": [{"role": "user", "content": "ghijkl"}]}]
        )
    report = observations.report()
    Draft202012Validator(window_observation_report_schema()).validate(report)
    assert all(r.is_complete for r in results) and len(adapter.requests) == 2
    assert report["omitted"] > 0
    assert len(report["rows"]) <= (1 if limit != "max_replies" else 2)
    if limit == "max_replies":
        assert sum(r["state"] == "limit_exceeded" for r in report["rows"]) == 1


async def test_many_native_windows_are_not_retained_or_copied_after_optional_work_refusal(tmp_path):
    compiled, adapter = policy(tmp_path, 'model("classes").classify(messages) > 0.5')
    result_cache.clear()
    with observe_model_windows(WindowObservationLimits(max_work=64)) as observations:
        result = await compiled.scan_async({"messages": [{"role": "user", "content": "a" * 2000}]})
    report = observations.report()
    (row,) = report["rows"]
    assert result.is_complete and len(adapter.requests) == 1
    assert row["window_count"] == 1000 and row["state"] == "limit_exceeded"
    assert row["probabilities"] is None and row["probability_states"] is None
    assert len(json.dumps(report)) < 2000 and report["omitted"] > 0


async def test_mutating_host_storage_never_publishes_a_vector_conflicting_with_selected_values(
    tmp_path,
):
    accesses = []

    class Chunk:
        def __init__(self):
            self.probabilities = [0.3, 0.7]

        def __getattribute__(self, name):
            if name == "probabilities":
                accesses.append(1)
                object.__setattr__(self, "probabilities", [0.9, 0.1])
            return object.__getattribute__(self, name)

    class Adapter(EvidenceAdapter):
        async def classify_window_batch(self, *args, **kwargs):
            raw = await super().classify_window_batch(*args, **kwargs)
            raw.chunks = [Chunk() for _ in raw.chunks]
            return raw

    compiled, adapter = policy(
        tmp_path, 'model("classes").classify(messages) > 0.85', adapter=Adapter()
    )
    result_cache.clear()
    with observe_model_windows() as observations:
        result = await compiled.scan_async(DATA)
    (row,) = observations.report()["rows"]
    assert result.is_complete and result.rule_matches == ["R"]
    assert row["state"] == "partial" and row["probabilities"] == [None] * 3
    assert row["probability_states"] == ["conflicting_probability_vector"] * 3
    assert accesses == [1] * 3 and len(adapter.requests) == 1


@pytest.mark.parametrize("unit", ["characters", "tokens"])
@pytest.mark.parametrize("failed_first", [False, True])
@pytest.mark.parametrize("failure", ["failed", "cancelled"])
async def test_same_cut_and_reply_keep_distinct_read_outcomes(
    tmp_path, unit, failed_first, failure
):
    class Adapter(MappedAdapter):
        async def decode_window_spans(self, config, raw, source, windows, threshold, **kwargs):
            if threshold == 0.9:
                if failure == "cancelled":
                    raise asyncio.CancelledError
                raise LLMError("synthetic decoder failure", {"code": "invalid_output"})
            return await super().decode_window_spans(
                config, raw, source, windows, threshold, **kwargs
            )

    adapter = Adapter()
    service = backend(
        tmp_path,
        {
            "segments": {
                **model_entry(spans=0, category="SYNTHETIC"),
                "windows": {"unit": unit, "size": 3, "step": 2, "scope": "last_turn"},
            }
        },
        adapter,
    )
    messages = Messages.from_openai(DATA["messages"])
    result_cache.clear()
    with observe_model_windows() as observations, scan_scope(), read_usage_scope("R"):
        for threshold in (0.9, 0.5) if failed_first else (0.5, 0.9):
            if threshold == 0.9:
                with pytest.raises(asyncio.CancelledError if failure == "cancelled" else LLMError):
                    await service.model_span_messages("segments", messages, threshold)
            else:
                spans = await service.model_span_messages("segments", messages, threshold)
                assert len(spans.items) == 3
    report = observations.report()
    Draft202012Validator(window_observation_report_schema()).validate(report)
    assert report["omitted"] == 0 and len(report["rows"]) == 2
    rows = {row["outcome"]: row for row in report["rows"]}
    assert set(rows) == {"succeeded", failure}
    assert all(row["reads"] == 1 for row in rows.values())
    assert rows["succeeded"]["state"] == "measured"
    assert rows["succeeded"]["probabilities"] is not None
    assert rows[failure]["state"] == failure
    assert rows[failure]["probabilities"] is None
    assert len(adapter.requests) == (2 if failed_first and failure == "failed" else 1)
