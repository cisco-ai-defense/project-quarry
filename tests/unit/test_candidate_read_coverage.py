# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Per-channel coverage remains separate from group authority and physical attempts."""

import gc
import weakref

import pytest

from quarry.integrations import candidate_adapter, read_candidates
from quarry.integrations.candidate_checks import LocalCandidatePattern
from quarry.integrations.candidate_providers import LocalCandidateProvider
from quarry.integrations.read_usage import ReadUsageLimits, model_read_usage
from quarry.types.candidates import (
    CandidateBatchIdentity,
    CandidateBatchReply,
    CandidateChannelReply,
    CandidateCoverage,
)
from quarry.types.messages import Messages
from tests.unit.test_candidate_batch_runtime import DATA, policy, response


def rows(result):
    return {row["model"]: row for row in result.logical_reads["rows"]}


@pytest.mark.parametrize("kind", ["host", "local"])
@pytest.mark.parametrize("restored", [False, True])
def test_each_alias_observes_its_own_complete_input_from_one_group(tmp_path, kind, restored):
    calls = []

    def callback(request, *, timeout):
        calls.append(request.digest)
        return response(request)

    provider = (
        LocalCandidateProvider(
            CandidateBatchIdentity("synthetic-candidates", "1", "heads"),
            ("left", "right"),
            (LocalCandidatePattern("left", "abc"), LocalCandidatePattern("right", "c")),
        )
        if kind == "local"
        else None
    )
    try:
        compiled = policy(tmp_path, callback, provider=provider, restored=restored)
        result = compiled.scan(DATA, logical_reads=True, window_observations=True)
        assert result.is_complete, result.evaluation_errors
        observed = rows(result)
        assert set(observed) == {"left", "right"}
        for row in observed.values():
            assert row["operation"] == "spans" and row["outcome"] == "succeeded"
            assert row["observations"] == 1
            assert row["coverage"]["characters"]["value"] == 6
            assert row["coverage"]["characters"]["measurement_states"] == {"measured": 1}
            assert row["unique_cuts"] == 1 and row["cut_identity_complete"]
        assert result.model_usage["calls"] == int(kind == "host")
        assert len(calls) == int(kind == "host")
        assert result.window_observations["rows"] == []
    finally:
        if provider is not None:
            provider.close()


@pytest.mark.parametrize("alias", ["left", "right"])
@pytest.mark.parametrize("coverage,known", [((), 0), (((0, 1), (4, 6)), 3), (((0, 2), (1, 3)), 3)])
@pytest.mark.parametrize("restored", [False, True])
def test_partial_group_reports_selected_channel_without_authoritative_cut(
    tmp_path, alias, coverage, known, restored
):
    calls = []

    def callback(request, *, timeout):
        calls.append(True)
        return CandidateBatchReply(
            request.digest,
            (
                CandidateChannelReply("left", (), (CandidateCoverage(0, 0, 6),), True),
                CandidateChannelReply(
                    "right",
                    (),
                    tuple(CandidateCoverage(0, low, high) for low, high in coverage),
                    False,
                ),
            ),
        )

    compiled = policy(
        tmp_path,
        callback,
        restored=restored,
        source=f'rule "Read" condition: not any s in model("{alias}").spans(messages, 0.5) where true',
    )
    for _ in range(2):
        result = compiled.scan(DATA, logical_reads=True)
        assert not result.is_complete and not result.rule_matches and not result.outputs
        assert result.evaluation_error_categories["Read"] == "model_partial_read"
        row = rows(result)[alias]
        assert row["outcome"] == "failed"
        measured = row["coverage"]["characters"]
        assert measured["value"] is None and measured["known_units"] == (
            6 if alias == "left" else known
        )
        assert measured["measurement_states"] == {"partial": 1}
        assert not row["cut_identity_complete"]
    assert len(calls) == 2


def test_multipart_measurement_unions_real_ranges_without_joined_text(tmp_path):
    provider = LocalCandidateProvider(
        CandidateBatchIdentity("synthetic-candidates", "1", "heads"),
        ("left", "right"),
        (LocalCandidatePattern("left", "ab"), LocalCandidatePattern("right", "cd")),
    )
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "ab"},
                    {"type": "text", "text": "cd"},
                ],
            }
        ]
    }
    try:
        result = policy(tmp_path, None, provider=provider).scan(data, logical_reads=True)
        assert result.is_complete, result.evaluation_errors
        assert all(row["coverage"]["characters"]["value"] == 4 for row in rows(result).values())
        assert result.outputs["Read"].items[0].value.source_ranges[0].part_index == 1
    finally:
        provider.close()


def test_invalid_group_cannot_claim_known_inspection_coverage(tmp_path):
    result = policy(tmp_path, lambda request, **kwargs: response(request, invalid=True)).scan(
        DATA, logical_reads=True
    )
    assert not result.is_complete
    row = rows(result)["left"]
    assert row["coverage"]["characters"]["value"] is None
    assert row["coverage"]["characters"]["known_units"] == 0
    assert not row["cut_identity_complete"]


def test_disabled_observations_do_not_run_coverage_projection(tmp_path, monkeypatch):
    def no_projection(*args, **kwargs):
        raise AssertionError("disabled optional observation must not inspect coverage")

    monkeypatch.setattr(read_candidates, "collect_candidate_input", no_projection)
    result = policy(tmp_path, lambda request, **kwargs: response(request)).scan(DATA)
    assert result.is_complete and result.rule_matches == ["Read"]


@pytest.mark.asyncio
async def test_telemetry_input_work_limit_preserves_native_result_and_attempt_count(tmp_path):
    calls = []

    def callback(request, *, timeout):
        calls.append(True)
        return response(request)

    compiled = policy(tmp_path, callback)
    service = compiled._scanner.resource_manager.model_inference_backend
    with model_read_usage(ReadUsageLimits(max_input_work=1)) as session:
        result = await service.model_span_messages(
            "left", Messages.from_openai(DATA["messages"]), 0.5
        )
    assert len(result.items) == 1 and len(calls) == 1
    row = session.snapshot()["rows"][0]
    assert row["outcome"] == "succeeded" and row["coverage"]["characters"]["value"] is None
    assert row["coverage"]["characters"]["measurement_states"] == {"limit_exceeded": 1}


def test_empty_complete_channel_is_measured_zero(tmp_path):
    def callback(request, *, timeout):
        return CandidateBatchReply(
            request.digest,
            tuple(
                CandidateChannelReply(
                    channel,
                    (),
                    tuple(
                        CandidateCoverage(i, 0, len(p.text))
                        for i, p in enumerate(request.input.parts)
                    ),
                    True,
                )
                for channel in request.channels
            ),
        )

    compiled = policy(
        tmp_path,
        callback,
        source='rule "Read" condition: any s in model("left").spans(messages, 0.5) where true',
    )
    result = compiled.scan({"messages": [{"role": "user", "content": ""}]}, logical_reads=True)
    assert result.is_complete and not result.rule_matches
    measured = rows(result)["left"]["coverage"]["characters"]
    assert measured["value"] == 0 and measured["measurement_states"] == {"measured": 1}


def test_coverage_report_and_closed_group_do_not_hold_source_owners(tmp_path, monkeypatch):
    owners = []
    original = candidate_adapter.bind_candidate_input

    async def observed(*args, **kwargs):
        binding = await original(*args, **kwargs)
        owners.extend(weakref.ref(source.origin.owner) for source in binding.source_texts())
        return binding

    monkeypatch.setattr(candidate_adapter, "bind_candidate_input", observed)

    def callback(request, *, timeout):
        return CandidateBatchReply(
            request.digest,
            tuple(
                CandidateChannelReply(
                    channel,
                    (),
                    tuple(
                        CandidateCoverage(i, 0, len(p.text))
                        for i, p in enumerate(request.input.parts)
                    ),
                    True,
                )
                for channel in request.channels
            ),
        )

    compiled = policy(
        tmp_path,
        callback,
        source='rule "Read" condition: not any s in model("left").spans(messages, 0.5) where true',
    )
    result = compiled.scan(DATA, logical_reads=True)
    assert result.is_complete and result.rule_matches == ["Read"]
    gc.collect()
    assert owners and all(owner() is None for owner in owners)
    assert rows(result)["left"]["coverage"]["characters"]["value"] == 6
