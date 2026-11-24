# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Width opt-in is lazy, identity-safe and validated before cached overflow escapes."""

import json
import sys
from dataclasses import asdict, make_dataclass, replace

import pytest

from quarry.cache.async_cache import scan_scope
from quarry.cache.keys import input_key
from quarry.config import QuarryConfig
from quarry.integrations.local_tokenizer import LocalTokenizerAdapter
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.types.exceptions import (
    ConfigurationError,
    LocalTokenizerInvalidInputError,
    LocalTokenizerInvalidOutputError,
)
from quarry.types.messages import Messages
from quarry.types.tokenizer_width import CappedTokenCount
from tests.unit.model_fakes import ModelAdapter, passthrough_builder
from tests.unit.test_tokenizer_width_adapter import setup
from tests.unit.test_tokenizer_width_attestation import claim


def document(tmp_path):
    local, width = setup(tmp_path)
    raw = asdict(local)
    raw["template_variables"] = dict(local.template_variables)
    return {
        "config_version": 2,
        "models": {
            "counter": {
                "adapter": "local_tokenizer",
                "model": "synthetic",
                "revision": local.model_revision,
                "tokens": {
                    "adapter": "local_tokenizer",
                    "extraction": {"count": "count"},
                    "local": raw,
                    "source_width": asdict(width),
                },
            }
        },
    }


def config(tmp_path, retention="process"):
    raw = document(tmp_path)
    raw["model_cache"] = {"retention": retention}
    return QuarryConfig.from_mapping(raw, tmp_path, require_rules=False)


def test_claim_parse_is_lazy_and_binds_all_semantics_without_artifact_io(tmp_path, monkeypatch):
    raw = document(tmp_path)
    raw["models"]["counter"]["tokens"]["local"]["artifact_path"] = "not-created.json"
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    monkeypatch.setitem(sys.modules, "jinja2", None)
    cfg = QuarryConfig.from_mapping(raw, tmp_path, require_rules=False)
    model = cfg.models["counter"]
    assert model.tokens.source_width == claim(model.tokens.local)
    identity = json.dumps(model.identity_dict())
    assert str(tmp_path) not in identity and model.tokens.local.template not in identity
    raw["models"]["counter"]["tokens"]["source_width"]["max_source_characters_per_token"] = 1
    assert model.tokens.source_width.max_source_characters_per_token == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_source_characters_per_token", True),
        ("max_source_characters_per_token", float("nan")),
        ("max_source_characters_per_token", 0),
        ("max_source_characters_per_token", -1),
        ("method", "trust-me"),
        ("artifact_sha256", "0" * 64),
        ("count_settings_sha256", "0" * 64),
        ("model_revision", "other"),
        ("tokenizer_revision", "other"),
        ("verified", True),
    ],
)
def test_invalid_or_mismatched_opt_in_is_rejected_at_configuration(tmp_path, field, value):
    raw = document(tmp_path)
    raw["models"]["counter"]["tokens"]["source_width"][field] = value
    with pytest.raises((ValueError, ConfigurationError)):
        QuarryConfig.from_mapping(raw, tmp_path, require_rules=False)


@pytest.mark.parametrize("value", [None, False, 1, [], {"max_source_characters_per_token": 4}])
def test_width_requires_a_full_native_mapping(tmp_path, value):
    raw = document(tmp_path)
    raw["models"]["counter"]["tokens"]["source_width"] = value
    with pytest.raises((ValueError, ConfigurationError)):
        QuarryConfig.from_mapping(raw, tmp_path, require_rules=False)


def test_remote_tokenizer_cannot_claim_a_native_width(tmp_path):
    raw = document(tmp_path)
    tokens = raw["models"]["counter"]["tokens"]
    tokens.pop("local")
    tokens["adapter"] = "vllm"
    with pytest.raises((ValueError, ConfigurationError)):
        QuarryConfig.from_mapping(raw, tmp_path, require_rules=False)


@pytest.mark.parametrize("count", [257, 300])
async def test_oversized_wire_lists_keep_canonical_invalid_input_failure(tmp_path, count):
    cfg = config(tmp_path)
    backend = ModelInferenceBackend(cfg)
    wire = [{"role": "user", "content": "aaaa"}] * count
    try:
        with pytest.raises(LocalTokenizerInvalidInputError):
            await backend._count_messages("counter", wire, [], key=(wire, []))
        with pytest.raises(LocalTokenizerInvalidInputError):
            await backend._count_messages_capped("counter", wire, [], cap=0, key=(wire, []))
    finally:
        backend.adapters["local_tokenizer"].close()


def test_absent_width_preserves_previous_local_cache_and_public_model_identity(tmp_path):
    cfg = config(tmp_path)
    model = cfg.models["counter"]
    current = replace(model.tokens, source_width=None)
    old = make_dataclass(
        "InputTokensConfig",
        [
            ("adapter", str),
            ("endpoint", str),
            ("count_field", str),
            ("model", str),
            ("local", object),
        ],
        frozen=True,
    )
    old.__module__, old.__qualname__ = "quarry.config", "InputTokensConfig"
    old_value = old(
        current.adapter, current.endpoint, current.count_field, current.model, current.local
    )
    assert input_key(current) == input_key(old_value)
    identity = replace(model, tokens=current).identity_dict()
    assert "source_width" not in identity["tokens"]
    assert input_key(current) != input_key(model.tokens)
    assert identity != model.identity_dict()


@pytest.mark.parametrize("retention,expected", [("process", 1), ("scan", 2)])
async def test_capped_cache_retains_by_policy_and_partitions_alias_cap_and_revision(
    tmp_path, monkeypatch, retention, expected
):
    cfg = config(tmp_path, retention)
    model = cfg.models["counter"]
    local = replace(model.tokens.local, model_revision="other")
    other = replace(
        model,
        revision="other",
        tokens=replace(model.tokens, local=local, source_width=claim(local)),
    )
    cfg = replace(cfg, models={"counter": model, "alias": model, "revision": other})
    seen = []
    original = LocalTokenizerAdapter.tokenize_capped

    async def observed(self, *args, **kwargs):
        seen.append((kwargs["alias"], kwargs["cap"]))
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(LocalTokenizerAdapter, "tokenize_capped", observed)
    backend = ModelInferenceBackend(cfg)
    wire = [{"role": "user", "content": "a" * 128}]

    async def read(alias="counter", cap=0):
        return await backend._count_messages_capped(alias, wire, [], cap=cap, key=(wire, []))

    try:
        for _ in range(2):
            with scan_scope():
                first = await read()
                assert await read() == first
                assert not first.is_exact
        assert len(seen) == expected
        with scan_scope():
            cap, alias, revision = await read(cap=1), await read("alias"), await read("revision")
        assert len(seen) == expected + 3
        assert (
            len(
                {
                    first.proof_fingerprint,
                    cap.proof_fingerprint,
                    alias.proof_fingerprint,
                    revision.proof_fingerprint,
                }
            )
            == 4
        )
        # Ordinary input_tokens keeps its canonical scalar namespace/count.
        assert (
            await backend.model_input_tokens_messages("counter", Messages.from_openai(wire))
            > first.lower_bound
        )
    finally:
        backend.adapters["local_tokenizer"].close()


@pytest.mark.parametrize("mutation", ["cap", "binding", "type", "forged_bound", "excess_bound"])
async def test_invalid_full_reply_is_rejected_before_retention_and_next_call_retries(
    tmp_path, monkeypatch, mutation
):
    cfg = config(tmp_path)
    backend = ModelInferenceBackend(cfg)
    original = LocalTokenizerAdapter.tokenize_capped
    calls = 0

    async def malformed(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        valid = await original(self, *args, **kwargs)
        if calls > 1:
            return valid
        if mutation == "cap":
            return replace(valid, cap=1)
        if mutation == "binding":
            return replace(valid, proof_fingerprint="0" * 64)
        if mutation == "type":
            return {"cap": 0, "lower_bound": 999}
        if mutation == "excess_bound":
            return replace(valid, lower_bound=1 << 62)
        object.__setattr__(valid, "lower_bound", True)
        return valid

    monkeypatch.setattr(LocalTokenizerAdapter, "tokenize_capped", malformed)
    wire = [{"role": "user", "content": "a" * 128}]
    try:
        with scan_scope():
            with pytest.raises(LocalTokenizerInvalidOutputError):
                await backend._count_messages_capped("counter", wire, [], cap=0, key=(wire, []))
            value = await backend._count_messages_capped("counter", wire, [], cap=0, key=(wire, []))
            assert type(value) is CappedTokenCount and not value.is_exact and calls == 2
            assert (
                await backend._count_messages_capped("counter", wire, [], cap=0, key=(wire, []))
                == value
            )
            assert calls == 2
    finally:
        backend.adapters["local_tokenizer"].close()


@pytest.mark.parametrize("direction", ["start", "end"])
async def test_actual_trim_agrees_with_canonical_selection_and_confirms_final_cut(
    tmp_path, monkeypatch, direction
):
    cfg = config(tmp_path)
    model = cfg.models["counter"]
    canonical = replace(
        cfg, models={"counter": replace(model, tokens=replace(model.tokens, source_width=None))}
    )
    backend, reference = ModelInferenceBackend(cfg), ModelInferenceBackend(canonical)
    exacts, capped = [], []
    old_exact, old_capped = backend._count_messages, backend._count_messages_capped

    async def exact(*args, **kwargs):
        value = await old_exact(*args, **kwargs)
        exacts.append((args[1], value))
        return value

    async def bounded(*args, **kwargs):
        value = await old_capped(*args, **kwargs)
        capped.append(value)
        return value

    monkeypatch.setattr(backend, "_count_messages", exact)
    monkeypatch.setattr(backend, "_count_messages_capped", bounded)
    messages = Messages.from_openai(
        [{"role": "user", "content": text} for text in ["a" * 512, "é" * 30, "a" * 256]]
    )
    try:
        expected = await reference.model_trim_messages(
            "counter", "classify", messages, 60, direction=direction
        )
        actual = await backend.model_trim_messages(
            "counter", "classify", messages, 60, direction=direction
        )
        assert actual.messages.to_openai_messages() == expected.messages.to_openai_messages()
        assert actual.omitted_messages == expected.omitted_messages
        assert actual.omitted_characters == expected.omitted_characters
        assert any(not value.is_exact for value in capped)
        assert exacts and exacts[-1][0] == actual.messages.to_openai_messages()
        assert exacts[-1][1] <= 60
    finally:
        backend.adapters["local_tokenizer"].close()
        reference.adapters["local_tokenizer"].close()


@pytest.mark.parametrize("alias", ["counter", "c" * 257, "counter\n"])
@pytest.mark.parametrize("size", [60, 1 << 63])
async def test_trim_outside_capped_binding_limits_preserves_canonical_selection(
    tmp_path, alias, size
):
    cfg = config(tmp_path)
    model = cfg.models["counter"]
    cfg = replace(cfg, models={alias: model})
    canonical = replace(
        cfg, models={alias: replace(model, tokens=replace(model.tokens, source_width=None))}
    )
    backend, reference = ModelInferenceBackend(cfg), ModelInferenceBackend(canonical)
    messages = Messages.from_openai([{"role": "user", "content": "a" * 512}])
    try:
        expected = await reference.model_trim_messages(alias, "classify", messages, size)
        actual = await backend.model_trim_messages(alias, "classify", messages, size)
        assert actual.messages.to_openai_messages() == expected.messages.to_openai_messages()
        assert actual.omitted_messages == expected.omitted_messages
        assert actual.omitted_characters == expected.omitted_characters
    finally:
        backend.adapters["local_tokenizer"].close()
        reference.adapters["local_tokenizer"].close()


async def test_implicit_custom_score_builder_and_split_arithmetic_keep_canonical_counts(
    tmp_path, monkeypatch
):
    raw = document(tmp_path)
    raw["models"]["counter"].update(
        {
            "adapter": "fake",
            "score": {
                "message_builder": "test",
                "boundary_seed": 7,
                "max_input_tokens": 60,
            },
        }
    )
    cfg = QuarryConfig.from_mapping(raw, tmp_path, require_rules=False)
    backend = ModelInferenceBackend(
        cfg, adapters={"fake": ModelAdapter()}, message_builders={"test": passthrough_builder}
    )

    async def forbidden(*args, **kwargs):
        pytest.fail("custom conversion/split arithmetic used a lower bound")

    monkeypatch.setattr(backend, "_count_messages_capped", forbidden)
    messages = Messages.from_openai([{"role": "user", "content": "a" * 256}])
    try:
        trimmed = await backend.model_trim_messages(
            "counter", "score", messages, None, constitution="synthetic"
        )
        assert await backend.model_input_tokens_messages("counter", trimmed.messages) <= 60
        chunks = await backend.model_split_messages("counter", messages, 60)
        assert chunks and all(
            [await backend.model_input_tokens_messages("counter", chunk) <= 60 for chunk in chunks]
        )
    finally:
        backend.adapters["local_tokenizer"].close()


@pytest.mark.parametrize("entry", ["scanner", "compiled", "saved"])
def test_opt_in_configuration_preserves_exact_public_count_and_saved_policy(tmp_path, entry):
    from quarry import compile_policy, load_compiled_policy
    from quarry.scanner import DSLScanner

    cfg = config(tmp_path)
    rule = 'rule "Count" condition: model("counter").input_tokens(messages) > 0'
    data = {"messages": [{"role": "user", "content": "a" * 128}]}
    cfg = replace(cfg, quarry_rule_texts=(rule,))
    backend = ModelInferenceBackend(cfg)
    try:
        if entry == "scanner":
            result = DSLScanner(cfg, model_inference_backend=backend).scan_data(data)
        else:
            policy = compile_policy(rule, runtime_config=cfg, model_inference_backend=backend)
            if entry == "saved":
                artifact = policy.to_bytes()
                policy = load_compiled_policy(
                    artifact, runtime_config=cfg, model_inference_backend=backend
                )
                assert policy.to_bytes() == artifact
            result = policy.scan(data)
        assert result.is_complete and result.rule_matches == ["Count"]
        assert result.model_usage["calls"] == 0 and result.model_usage["records"] == []
    finally:
        backend.adapters["local_tokenizer"].close()
