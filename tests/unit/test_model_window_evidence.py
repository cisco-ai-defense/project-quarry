# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Window evidence owns the actual maximum and contains only bounded coordinates."""

import asyncio
import json
from dataclasses import replace

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.cache.async_cache import result_cache
from tests.unit.model_fakes import model_entry
from tests.unit.test_model_token_head_contracts import backend
from tests.unit.test_model_window_runtime import WindowAdapter


class EvidenceAdapter(WindowAdapter):
    async def classify_window_batch(self, *args, **kwargs):
        raw = await super().classify_window_batch(*args, **kwargs)
        for ordinal, chunk in enumerate(raw.chunks):
            chunk.probabilities[0] = 0.3 if ordinal == 0 else 0.8
        return raw


def policy(
    tmp_path, condition, *, unit="characters", restored=False, scope="last_turn", adapter=None
):
    adapter = adapter or EvidenceAdapter()
    windows = {"unit": unit, "size": 3, "step": 2, "scope": scope}
    service = backend(
        tmp_path, {"classes": {**model_entry(classify=0), "windows": windows}}, adapter
    )
    compiled = compile_policy(
        'rule "R" condition: ' + condition,
        runtime_config=service.config,
        model_inference_backend=service,
    )
    if restored:
        wire = compiled.to_bytes()
        compiled = load_compiled_policy(
            wire, runtime_config=service.config, model_inference_backend=service
        )
        assert compiled.to_bytes() == wire
    return compiled, adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["characters", "tokens"])
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("comparison", ["> 0.5", "< 0.9"])
async def test_deciding_window_uses_the_maximum_for_both_comparison_directions(
    tmp_path, unit, restored, comparison
):
    compiled, adapter = policy(
        tmp_path, 'model("classes").classify(messages) ' + comparison, unit=unit, restored=restored
    )
    result_cache.clear()
    data = {
        "messages": [
            {"role": "user", "content": "PRIVATE"},
            {"role": "assistant", "content": "abcdef"},
        ]
    }
    result = await compiled.scan_async(data, explain=True)
    assert result.is_complete and result.rule_matches == ["R"], result.evaluation_errors
    (evidence,) = result.detailed_matches[0].model_results
    assert evidence.operation == "classify" and evidence.expression_value == 0.8
    assert evidence.score_summary is not None, "window classifiers need deciding-window evidence"
    summary = evidence.score_summary["windows"]
    assert set(summary) == {
        "unit",
        "count",
        "minimum",
        "maximum",
        "deciding",
        "samples",
        "omitted_windows",
    }
    assert (
        summary["count"],
        summary["minimum"],
        summary["maximum"],
        summary["omitted_windows"],
    ) == (3, 0.3, 0.8, 0)
    deciding = summary["deciding"]
    assert (deciding["ordinal"], deciding["value"]) == (1, 0.8)
    assert set(deciding) == {
        "ordinal",
        "value",
        "range",
        "pieces",
        "pieces_truncated",
        "source_geometry",
    }
    if unit == "characters":
        assert deciding["range"] == {"unit": "characters", "intervals": [[10, 13]]}
        assert deciding["pieces"] == [
            {"position": 1, "field": "", "start": 2, "end": 5, "precision": "exact"}
        ]
        assert deciding["source_geometry"] == "mapped"
    else:
        assert deciding["range"] == {"unit": "tokens", "intervals": [[0, 1], [11, 14], [15, 16]]}
        assert deciding["pieces"] == [] and deciding["source_geometry"] == "unavailable"
    wire = json.dumps(evidence.to_dict())
    assert "PRIVATE" not in wire and "abcdef" not in wire and "token_ids" not in wire
    assert len(adapter.requests) == 1


@pytest.mark.asyncio
async def test_arithmetic_keeps_a_window_calls_score_separate_from_the_expression_value(tmp_path):
    compiled, _ = policy(tmp_path, '0.5 * model("classes").classify(messages) > 0.3')
    result_cache.clear()
    result = await compiled.scan_async(
        {"messages": [{"role": "user", "content": "abcdef"}]}, explain=True
    )
    assert result.is_complete and result.rule_matches == ["R"]
    (evidence,) = result.detailed_matches[0].model_results
    assert evidence.operation == "numeric_expression" and evidence.expression_value == 0.4
    assert "windows" not in evidence.score_summary
    (call,) = evidence.score_summary["window_calls"]
    assert (call["call"], call["model"], call["windows"]["maximum"]) == (0, "classes", 0.8)


@pytest.mark.asyncio
@pytest.mark.parametrize("aggregate", ["count", "all"])
async def test_outer_aggregates_do_not_inherit_a_child_window_decision(tmp_path, aggregate):
    from tests.unit.test_message_local_tool_conditions import DATA

    selected = '[c for c in assistant_messages[-1].tool_calls if model("classes").classify(message("user", c.name)) > 0.5]'
    condition = (
        f"count({selected}) == 2" if aggregate == "count" else f"all c in {selected} where true"
    )
    compiled, _ = policy(tmp_path, condition, scope="all")
    result_cache.clear()
    result = await compiled.scan_async(DATA, explain=True)
    assert result.is_complete and result.rule_matches == ["R"], result.evaluation_errors
    (evidence,) = result.detailed_matches[0].model_results
    assert evidence.operation == (
        "numeric_expression" if aggregate == "count" else "boolean_expression"
    )
    assert evidence.expression_value == (2 if aggregate == "count" else True)
    assert evidence.score_summary is evidence.input_span is None


@pytest.mark.asyncio
async def test_two_classifier_operands_keep_separate_call_evidence(tmp_path):
    call = 'model("classes").classify(messages)'
    compiled, adapter = policy(tmp_path, f"{call} == {call}")
    result_cache.clear()
    result = await compiled.scan_async(
        {"messages": [{"role": "user", "content": "abcdef"}]}, explain=True
    )
    assert result.is_complete and result.rule_matches == ["R"]
    (evidence,) = result.detailed_matches[0].model_results
    assert evidence.operation == "numeric_expression"
    assert "windows" not in evidence.score_summary
    assert [item["call"] for item in evidence.score_summary["window_calls"]] == [0, 1]
    assert len(adapter.requests) == 1


@pytest.mark.asyncio
async def test_window_samples_are_bounded_and_always_keep_a_late_deciding_window(tmp_path):
    class Late(EvidenceAdapter):
        async def classify_window_batch(self, *args, **kwargs):
            raw = await super().classify_window_batch(*args, **kwargs)
            for ordinal, chunk in enumerate(raw.chunks):
                chunk.probabilities[0] = 0.9 if ordinal == len(raw.chunks) - 1 else 0.8
            return raw

    compiled, _ = policy(tmp_path, 'model("classes").classify(messages) > 0.5', adapter=Late())
    result_cache.clear()
    result = await compiled.scan_async(
        {"messages": [{"role": "user", "content": "abcdefghijklmnopqrstuv"}]}, explain=True
    )
    (evidence,) = result.detailed_matches[0].model_results
    summary = evidence.score_summary["windows"]
    assert summary["count"] == 11 and len(summary["samples"]) == 8
    assert summary["omitted_windows"] == 3
    assert summary["deciding"]["ordinal"] == 10
    assert summary["deciding"] in summary["samples"]
    assert summary["deciding"]["pieces"][0]["start"] == 20
    assert len(json.dumps(summary)) < 10_000


@pytest.mark.asyncio
async def test_default_scans_do_not_build_window_geometry(tmp_path, monkeypatch):
    from quarry.integrations import model_window_evidence

    def unexpected(*args, **kwargs):
        raise AssertionError("a default scan must not build optional window geometry")

    monkeypatch.setattr(model_window_evidence, "_character_pieces", unexpected)
    compiled, _ = policy(tmp_path, 'model("classes").classify(messages) > 0.5')
    result_cache.clear()
    result = await compiled.scan_async({"messages": [{"role": "user", "content": "abcdef"}]})
    assert result.is_complete and result.rule_matches == ["R"]


@pytest.mark.asyncio
async def test_window_capture_cannot_be_written_by_an_inherited_child_task(tmp_path):
    from quarry.integrations.model_window_evidence import capture_window_result
    from quarry.types.messages import Messages

    service = backend(
        tmp_path,
        {"classes": {**model_entry(classify=0), "windows": {"unit": "characters", "size": 3}}},
        EvidenceAdapter(),
    )
    messages = Messages.from_openai([{"role": "user", "content": "abcdef"}])
    result_cache.clear()
    with capture_window_result() as outer:
        assert (
            await asyncio.create_task(service.model_classify_messages("classes", messages)) == 0.8
        )
        assert outer.value is None
        assert await service.model_classify_messages("classes", messages) == 0.8
        assert outer.value["maximum"] == 0.8


@pytest.mark.asyncio
async def test_empty_window_input_does_not_invent_a_deciding_score(tmp_path):
    compiled, adapter = policy(tmp_path, 'model("classes").classify(messages) < 0.5')
    result_cache.clear()
    result = await compiled.scan_async(
        {"messages": [{"role": "user", "content": ""}]}, explain=True
    )
    assert result.is_complete and result.rule_matches == ["R"]
    (evidence,) = result.detailed_matches[0].model_results
    assert evidence.expression_value == 0.0 and evidence.score_summary is None
    assert adapter.requests == []


@pytest.mark.asyncio
async def test_explicit_encoder_character_offsets_map_token_decisions_to_original_sources(tmp_path):
    class Mapped(EvidenceAdapter):
        async def encode_window_input(self, config, text, context, last_turn, *, timeout=None):
            value = await super().encode_window_input(
                config, text, context, last_turn, timeout=timeout
            )
            return replace(
                value, character_offsets=(None, *((i, i + 1) for i in range(len(text))), None)
            )

    compiled, adapter = policy(
        tmp_path, 'model("classes").classify(messages) > 0.5', unit="tokens", adapter=Mapped()
    )
    result_cache.clear()
    result = await compiled.scan_async(
        {
            "messages": [
                {"role": "user", "content": "PRIVATE"},
                {"role": "assistant", "content": "abcdef"},
            ]
        },
        explain=True,
    )
    assert result.is_complete and result.rule_matches == ["R"], result.evaluation_errors
    (evidence,) = result.detailed_matches[0].model_results
    deciding = evidence.score_summary["windows"]["deciding"]
    assert deciding["source_geometry"] == "mapped"
    assert deciding["pieces"] == [
        {"position": 1, "field": "", "start": 2, "end": 5, "precision": "exact"}
    ]
    assert len(adapter.encodings) == len(adapter.requests) == 1
