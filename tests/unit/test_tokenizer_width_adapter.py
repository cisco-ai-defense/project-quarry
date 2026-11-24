# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Proved overflow preserves canonical request validation, limits and ownership."""

import asyncio
import threading
from dataclasses import replace

import pytest
from tokenizers import AddedToken, normalizers

from quarry.integrations import local_tokenizer as local
from quarry.integrations.local_tokenizer_artifact import _LoadedTokenizer
from quarry.types.exceptions import (
    LocalTokenizerInvalidInputError,
    LocalTokenizerLimitError,
    LocalTokenizerUnavailableError,
)
from quarry.types.local_tokenizer import TokenFramingMargins
from quarry.types.tokenizer_width import CappedTokenCount
from tests.unit.test_local_tokenizer_artifact import artifact
from tests.unit.test_tokenizer_width_attestation import bpe, claim

TEMPLATE = "{{ messages|tojson }}{{ tools|tojson }}"


def setup(tmp_path, reference=None, **options):
    cfg = artifact(tmp_path, reference or bpe(), options={"template": TEMPLATE, **options})
    return cfg, claim(cfg)


@pytest.mark.parametrize("text", ["a" * 256, "é e\u0301 👩\u200d💻" * 10, " " * 150, ""])
async def test_capped_answers_compare_against_actual_canonical_whole_request(tmp_path, text):
    cfg, requested = setup(tmp_path, framing=TokenFramingMargins(7, 3))
    messages = [{"role": "user", "content": text}, {"role": "assistant", "content": "aaaa"}]
    tools = [{"type": "function", "function": {"name": "synthetic"}}]
    adapter = local.LocalTokenizerAdapter()
    try:
        exact = await adapter.count_request(cfg, messages, tools)
        for cap in [0, 1, exact // 2, exact - 1, exact, exact + 1, (1 << 63) - 1]:
            result = await adapter.count_request_capped(cfg, requested, messages, tools, cap=cap)
            assert type(result) is CappedTokenCount
            assert result.cap == cap and result.lower_bound <= exact
            assert result.exceeds_cap == (exact > cap)
            if result.is_exact:
                assert result.exact_count == result.lower_bound == exact
                assert result.proof_fingerprint is None
            else:
                assert result.exact_count is None and result.lower_bound > cap
                assert len(result.proof_fingerprint) == 64
        assert await adapter.count_request(cfg, messages, tools) == exact
    finally:
        adapter.close()


async def test_proved_overflow_avoids_native_encode_and_attestation_runs_off_loop(
    tmp_path, monkeypatch
):
    cfg, requested = setup(tmp_path)
    caller, observed = threading.get_ident(), []
    original = local.attest_loaded_width

    def attest(*args):
        observed.append(threading.get_ident())
        return original(*args)

    def forbidden(*args):
        raise AssertionError("overflow path encoded the request")

    monkeypatch.setattr(local, "attest_loaded_width", attest)
    monkeypatch.setattr(_LoadedTokenizer, "encode", forbidden)
    adapter = local.LocalTokenizerAdapter()
    try:
        proof = await adapter.attest_width(cfg, requested)
        assert proof.is_compatible(cfg, requested)
        value = await adapter.count_request_capped(
            cfg, requested, [{"role": "user", "content": "a" * 512}], [], cap=10
        )
        assert not value.is_exact and value.lower_bound > 10
        assert observed and all(thread != caller for thread in observed)
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "kind", ["normalization", "strip", "missing", "collision", "bound", "settings", "revision"]
)
async def test_unproven_pipelines_and_claims_use_canonical_exact_count(tmp_path, kind):
    reference = bpe(missing=kind == "missing", collision=kind == "collision")
    if kind == "normalization":
        reference.normalizer = normalizers.Replace("x", "")
    elif kind == "strip":
        reference.add_tokens([AddedToken("X", lstrip=True)])
    if kind == "collision":
        import json

        document = json.loads(reference.to_str())
        document["model"]["vocab"] = reference.get_vocab(with_added_tokens=False)
        document["model"]["merges"] = [["a", "a"]]
        cfg = artifact(tmp_path, document=document, options={"template": TEMPLATE})
        requested = claim(cfg)
    else:
        cfg, requested = setup(tmp_path, reference)
    if kind == "bound":
        requested = replace(requested, max_source_characters_per_token=1)
    elif kind == "settings":
        requested = replace(requested, count_settings_sha256="0" * 64)
    elif kind == "revision":
        requested = replace(requested, tokenizer_revision="other")
    messages = [{"role": "user", "content": "x" * 64 + " " * 64 + "X" + "a" * 128}]
    adapter = local.LocalTokenizerAdapter()
    try:
        exact = await adapter.count_request(cfg, messages, [])
        assert await adapter.attest_width(cfg, requested) is None
        result = await adapter.count_request_capped(cfg, requested, messages, [], cap=0)
        assert result.is_exact and result.exact_count == exact
    finally:
        adapter.close()


async def test_overflow_cannot_mask_a_canonical_output_token_limit(tmp_path):
    cfg, _ = setup(tmp_path)
    cfg = replace(cfg, limits=replace(cfg.limits, max_tokens=1))
    requested = claim(cfg)
    messages = [{"role": "user", "content": "a" * 128}]
    adapter = local.LocalTokenizerAdapter()
    try:
        assert await adapter.attest_width(cfg, requested) is not None
        with pytest.raises(LocalTokenizerLimitError):
            await adapter.count_request(cfg, messages, [])
        with pytest.raises(LocalTokenizerLimitError):
            await adapter.count_request_capped(cfg, requested, messages, [], cap=0)
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "kind", ["media", "opaque", "unicode", "omitted_tools", "input_limit", "artifact_digest"]
)
async def test_full_request_validation_precedes_any_overflow_proof(tmp_path, kind):
    cfg, _ = setup(tmp_path)
    messages, tools = [{"role": "user", "content": "a" * 128}], []
    error = LocalTokenizerInvalidInputError
    if kind == "media":
        messages[0]["content"] = [{"type": "image_url", "image_url": {"url": "synthetic"}}]
    elif kind == "opaque":
        messages[0]["content"] = object()
    elif kind == "unicode":
        messages[0]["content"] = "\ud800"
    elif kind == "omitted_tools":
        cfg = replace(cfg, template="{{ messages|tojson }}")
        tools = [{"type": "function", "function": {"name": "synthetic"}}]
    elif kind == "input_limit":
        cfg = replace(cfg, limits=replace(cfg.limits, max_input_characters=8))
        error = LocalTokenizerLimitError
    elif kind == "artifact_digest":
        cfg = replace(cfg, artifact_sha256="0" * 64)
        error = LocalTokenizerUnavailableError
    requested = claim(cfg)
    adapter = local.LocalTokenizerAdapter()
    try:
        with pytest.raises(error):
            await adapter.count_request_capped(cfg, requested, messages, tools, cap=0)
    finally:
        adapter.close()


@pytest.mark.parametrize("cap", [True, -1, 1.0, float("inf"), float("nan"), 1 << 63, "1"])
async def test_invalid_caps_never_start_a_worker(tmp_path, cap):
    cfg, requested = setup(tmp_path)
    adapter = local.LocalTokenizerAdapter()
    try:
        with pytest.raises(LocalTokenizerInvalidInputError):
            await adapter.count_request_capped(cfg, requested, [], [], cap=cap)
        assert adapter._owner._executor is None
    finally:
        adapter.close()


async def test_attestation_metadata_cache_is_bounded_and_rechecks_evicted_revisions(
    tmp_path, monkeypatch
):
    from quarry.integrations.local_tokenizer_worker import _LocalWorkerOwner

    cfg, _ = setup(tmp_path)
    adapter = local.LocalTokenizerAdapter()
    adapter._owner = _LocalWorkerOwner(workers=1, capacity=2)
    observed = []
    original = local.attest_loaded_width

    def attest(config, requested, loaded, job):
        observed.append(config.tokenizer_revision)
        return original(config, requested, loaded, job)

    monkeypatch.setattr(local, "attest_loaded_width", attest)
    try:
        for revision in ["one", "one", "two", "three", "one"]:
            current = replace(cfg, tokenizer_revision=revision)
            verified = await adapter.attest_width(current, claim(current))
            assert verified.is_compatible(current, claim(current))
        assert observed == ["one", "two", "three", "one"]
    finally:
        adapter.close()


async def test_structured_calls_results_and_every_offered_tool_retain_one_whole_request(tmp_path):
    cfg, requested = setup(tmp_path, framing=TokenFramingMargins(5, 7))
    messages = [
        {"role": "user", "content": "a" * 256},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "synthetic-call",
                    "type": "function",
                    "function": {"name": "synthetic", "arguments": '{"x":1}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "synthetic-call", "content": "aaaa", "name": "synthetic"},
    ]
    tools = [
        {"type": "function", "function": {"name": "synthetic", "parameters": {"type": "object"}}}
    ]
    adapter = local.LocalTokenizerAdapter()
    try:
        exact = await adapter.count_request(cfg, messages, tools)
        value = await adapter.count_request_capped(cfg, requested, messages, tools, cap=1)
        assert not value.is_exact and value.lower_bound <= exact
        reordered = list(reversed(messages))
        assert (
            await adapter.count_request_capped(cfg, requested, reordered, tools, cap=exact + 1)
        ).exact_count == await adapter.count_request(cfg, reordered, tools)
    finally:
        adapter.close()


async def test_cancelled_attestation_holds_native_slot_until_actual_termination(
    tmp_path, monkeypatch
):
    cfg, requested = setup(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = local.attest_loaded_width

    def held(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(local, "attest_loaded_width", held)
    adapter = local.LocalTokenizerAdapter()
    task = asyncio.create_task(adapter.attest_width(cfg, requested))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert adapter.report()["pending"] == adapter.report()["active"] == 1
        adapter.close()
        assert not adapter.report()["complete"]
        with pytest.raises(LocalTokenizerUnavailableError):
            await adapter.attest_width(cfg, requested)
    finally:
        release.set()
        adapter.close()
        async with asyncio.timeout(5):
            while adapter.report()["pending"]:
                await asyncio.sleep(0.001)
    assert adapter.report()["complete"]
