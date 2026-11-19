# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Logical reads describe actual window cuts without extra tokenizer or provider calls."""

import json
from dataclasses import replace

import pytest

from quarry import compile_policy
from quarry.cache.async_cache import result_cache
from tests.unit.model_fakes import model_entry
from tests.unit.test_model_token_head_contracts import backend
from tests.unit.test_model_window_evidence import EvidenceAdapter


class MappedAdapter(EvidenceAdapter):
    async def encode_window_input(self, config, text, context, last_turn, *, timeout=None):
        value = await super().encode_window_input(config, text, context, last_turn, timeout=timeout)
        return replace(
            value, character_offsets=(None, *((i, i + 1) for i in range(len(text))), None)
        )


def policy(tmp_path, unit="characters", *, renderer="join", mapped=True):
    adapter = MappedAdapter() if mapped else EvidenceAdapter()
    windows = {"unit": unit, "size": 3, "step": 2, "scope": "last_turn", "renderer": renderer}
    service = backend(
        tmp_path, {"classes": {**model_entry(classify=0), "windows": windows}}, adapter
    )
    compiled = compile_policy(
        'rule "R" condition: model("classes").classify(messages) >= 0',
        runtime_config=service.config,
        model_inference_backend=service,
    )
    return compiled, adapter


def coverage(result):
    return result.logical_reads["aggregates"]["totals"]["coverage"]["characters"]


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["characters", "tokens"])
async def test_last_turn_reports_only_the_consumed_source_cut(tmp_path, unit):
    compiled, adapter = policy(tmp_path, unit)
    result_cache.clear()
    data = {
        "messages": [
            {"role": "user", "content": "PRIVATE"},
            {"role": "assistant", "content": "abcdef"},
        ]
    }
    result = await compiled.scan_async(data, logical_reads=True)
    assert result.is_complete and result.rule_matches == ["R"], result.evaluation_errors
    assert coverage(result)["value"] == 6
    assert len(adapter.requests) == 1 and len(adapter.encodings) == (unit == "tokens")
    assert "PRIVATE" not in json.dumps(result.logical_reads)


@pytest.mark.asyncio
async def test_token_ids_without_character_offsets_report_unavailable_character_coverage(tmp_path):
    compiled, adapter = policy(tmp_path, "tokens", mapped=False)
    result_cache.clear()
    data = {
        "messages": [
            {"role": "user", "content": "PRIVATE"},
            {"role": "assistant", "content": "abcdef"},
        ]
    }
    result = await compiled.scan_async(data, logical_reads=True)
    assert result.is_complete
    assert coverage(result)["value"] is None
    assert coverage(result)["measurement_states"] == {"unavailable": 1}
    assert len(adapter.encodings) == len(adapter.requests) == 1


@pytest.mark.asyncio
async def test_json_logical_reads_use_the_selected_serialized_object(tmp_path):
    compiled, adapter = policy(tmp_path, renderer="json")
    result_cache.clear()
    data = {
        "messages": [
            {"role": "user", "content": "PRIVATE"},
            {"role": "assistant", "content": "abcdef"},
        ]
    }
    result = await compiled.scan_async(data, logical_reads=True)
    expected = len(json.dumps(data["messages"][-1], separators=(",", ":"), ensure_ascii=False))
    assert result.is_complete
    assert coverage(result)["value"] == expected
    assert len(adapter.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["characters", "tokens"])
async def test_empty_last_turn_is_a_not_sent_read_with_zero_coverage(tmp_path, unit):
    compiled, adapter = policy(tmp_path, unit)
    result_cache.clear()
    data = {
        "messages": [{"role": "user", "content": "PRIVATE"}, {"role": "assistant", "content": ""}]
    }
    result = await compiled.scan_async(data, logical_reads=True)
    assert result.is_complete
    assert coverage(result)["value"] == 0
    assert result.logical_reads["aggregates"]["totals"]["outcomes"] == {"not_sent": 1}
    assert adapter.encodings == adapter.requests == []


@pytest.mark.asyncio
async def test_gaps_in_actual_encoder_offsets_are_not_counted_as_read_characters(tmp_path):
    from quarry.integrations.model_windows import TokenEncoding

    class Gapped(EvidenceAdapter):
        async def encode_window_input(self, config, text, context, last_turn, *, timeout=None):
            self.encodings.append((text, context, last_turn))
            return TokenEncoding(
                (111, 97, 98, 99, 100, 222),
                True,
                True,
                (1, 5),
                character_offsets=(None, (0, 1), (1, 2), (6, 7), (7, 8), None),
            )

    adapter = Gapped()
    windows = {"unit": "tokens", "size": 3, "step": 2}
    service = backend(
        tmp_path, {"classes": {**model_entry(classify=0), "windows": windows}}, adapter
    )
    compiled = compile_policy(
        'rule "R" condition: model("classes").classify(messages) >= 0',
        runtime_config=service.config,
        model_inference_backend=service,
    )
    result_cache.clear()
    result = await compiled.scan_async(
        {"messages": [{"role": "user", "content": "abDROPcd"}]}, logical_reads=True, explain=True
    )
    assert result.is_complete and coverage(result)["value"] == 4
    (evidence,) = result.detailed_matches[0].model_results
    first = evidence.score_summary["windows"]["samples"][0]
    assert [(piece["start"], piece["end"]) for piece in first["pieces"]] == [(0, 2), (6, 7)]
    assert len(adapter.encodings) == len(adapter.requests) == 1


@pytest.mark.asyncio
async def test_encoder_offsets_outside_the_full_input_fail_before_classifier_dispatch(tmp_path):
    class Outside(MappedAdapter):
        async def encode_window_input(self, config, text, context, last_turn, *, timeout=None):
            value = await super().encode_window_input(
                config, text, context, last_turn, timeout=timeout
            )
            offsets = list(value.character_offsets)
            offsets[-2] = (len(text) - 1, len(text) + 1)
            return replace(value, character_offsets=offsets)

    adapter = Outside()
    windows = {"unit": "tokens", "size": 3}
    service = backend(
        tmp_path, {"classes": {**model_entry(classify=0), "windows": windows}}, adapter
    )
    compiled = compile_policy(
        'rule "R" condition: model("classes").classify(messages) >= 0',
        runtime_config=service.config,
        model_inference_backend=service,
    )
    result_cache.clear()
    result = await compiled.scan_async(
        {"messages": [{"role": "user", "content": "abcdef"}]}, logical_reads=True
    )
    assert not result.is_complete
    assert len(adapter.encodings) == 1 and adapter.requests == []
