# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Actual config, cache, scanner, trim/split, CLI and MCP local-count paths."""

import json
import subprocess
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict, make_dataclass, replace

import pytest

from quarry import compile_policy
from quarry.cache.keys import input_key
from quarry.config import InputTokensConfig, QuarryConfig
from quarry.integrations.local_tokenizer import LocalTokenizerAdapter
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.scanner import DSLScanner
from quarry.types.exceptions import ConfigurationError, LocalTokenizerInvalidOutputError
from quarry.types.messages import Messages
from tests.unit.test_local_tokenizer_artifact import artifact

TEMPLATE = "{% for m in messages %}{{ m.content }} {% endfor %}"
RULE = 'rule "Count" condition: model("counter").input_tokens(messages) == 2'
DATA = {"messages": [{"role": "user", "content": "hello world"}]}


def document(tmp_path, *, local=None):
    cfg = artifact(tmp_path, options={"template": TEMPLATE})
    raw = asdict(cfg)
    raw["template_variables"] = dict(cfg.template_variables)
    if local:
        raw.update(local)
    return {
        "config_version": 2,
        "models": {
            "counter": {
                "adapter": "local_tokenizer",
                "model": "synthetic",
                "revision": cfg.model_revision,
                "tokens": {
                    "adapter": "local_tokenizer",
                    "extraction": {"count": "count"},
                    "local": raw,
                },
            }
        },
    }


def configuration(tmp_path, *, local=None, rules=()):
    doc = document(tmp_path, local=local)
    cfg = QuarryConfig.from_mapping(doc, tmp_path, require_rules=False)
    return replace(cfg, quarry_rule_texts=rules)


def test_config_is_lazy_strict_detached_and_relative_to_the_document(tmp_path, monkeypatch):
    doc = document(tmp_path)
    raw = doc["models"]["counter"]["tokens"]["local"]
    raw["artifact_path"] = "not-created.json"
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    monkeypatch.setitem(sys.modules, "jinja2", None)
    cfg = QuarryConfig.from_mapping(doc, tmp_path, require_rules=False)
    assert cfg.models["counter"].tokens.local.artifact_path == str(tmp_path / "not-created.json")
    raw["limits"]["max_tokens"] = 1
    assert cfg.models["counter"].tokens.local.limits.max_tokens == 262144
    backend = ModelInferenceBackend(cfg)
    assert backend.adapters["local_tokenizer"]._owner._executor is None
    backend.adapters["local_tokenizer"].close()


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "null",
        "bad_limits",
        "bad_variables",
        "bad_flag",
        "remote",
        "model_override",
        "revision",
    ],
)
def test_invalid_native_local_configuration_fails_before_artifact_loading(tmp_path, change):
    doc = document(tmp_path)
    model = doc["models"]["counter"]
    local = model["tokens"]["local"]
    if change == "unknown":
        local["untrusted_callback"] = "not.imported"
    elif change == "null":
        model["tokens"]["local"] = None
    elif change == "bad_limits":
        local["limits"]["max_tokens"] = True
    elif change == "bad_variables":
        local["template_variables"] = {"_private": "x"}
    elif change == "bad_flag":
        local["add_generation_prompt"] = 1
    elif change == "remote":
        model["tokens"]["endpoint"] = "https://example.invalid/tokenize"
    elif change == "model_override":
        model["tokens"]["model"] = "other"
    else:
        model["revision"] = "other"
    with pytest.raises((ValueError, ConfigurationError)):
        QuarryConfig.from_mapping(doc, tmp_path, require_rules=False)


def test_absent_extension_preserves_legacy_config_and_cache_identity():
    old = make_dataclass(
        "InputTokensConfig",
        [("adapter", str), ("endpoint", str), ("count_field", str), ("model", str)],
        frozen=True,
    )
    old.__module__ = "quarry.config"
    old.__qualname__ = "InputTokensConfig"
    current = InputTokensConfig("synthetic", None, "count")
    assert input_key(current) == input_key(old("synthetic", None, "count", None))
    doc = {
        "config_version": 2,
        "models": {
            "old": {
                "model": "synthetic",
                "tokens": {
                    "adapter": "vllm",
                    "endpoint": "/tokenize",
                    "extraction": {"count": "count"},
                },
            }
        },
    }
    cfg = QuarryConfig.from_mapping(doc, __import__("pathlib").Path.cwd(), require_rules=False)
    assert cfg.models["old"].identity_dict()["tokens"] == {
        "adapter": "vllm",
        "endpoint": "/tokenize",
        "count_field": "count",
        "model": None,
    }


def test_local_policy_identity_hides_paths_and_template_plaintext_and_names_every_semantic_change(
    tmp_path,
):
    cfg = configuration(tmp_path)
    model = cfg.models["counter"]
    identity = json.dumps(model.identity_dict())
    assert model.tokens.local.artifact_path not in identity and TEMPLATE not in identity
    original = input_key(model)
    for local in (
        replace(model.tokens.local, tokenizer_revision="different"),
        replace(model.tokens.local, template=TEMPLATE + "different"),
        replace(model.tokens.local, add_generation_prompt=True),
        replace(model.tokens.local, trim_blocks=True),
        replace(model.tokens.local, template_variables=(("bos_token", "x"),)),
        replace(model.tokens.local, framing=replace(model.tokens.local.framing, per_request=1)),
    ):
        changed = replace(model, tokens=replace(model.tokens, local=local))
        assert input_key(changed) != original and changed.identity_dict() != model.identity_dict()
    policy = compile_policy(RULE, runtime_config=cfg)
    assert model.tokens.local.artifact_path.encode() not in policy.to_bytes()


@pytest.mark.parametrize("entry", ["scanner", "compiled"])
def test_actual_scan_counts_locally_without_provider_calls_or_rate_limit_tickets(tmp_path, entry):
    cfg = configuration(tmp_path, rules=(RULE,))

    class NoProviderTickets:
        @asynccontextmanager
        async def limit(self, model):
            pytest.fail("local tokenization requested a provider rate-limit ticket")
            yield

    backend = ModelInferenceBackend(cfg, rate_limiter=NoProviderTickets())
    try:
        policy = (
            DSLScanner(cfg, model_inference_backend=backend)
            if entry == "scanner"
            else compile_policy(RULE, runtime_config=cfg, model_inference_backend=backend)
        )
        result = policy.scan_data(DATA) if entry == "scanner" else policy.scan(DATA)
        assert result.is_complete and result.rule_matches == ["Count"]
        assert result.model_usage["calls"] == 0 and result.model_usage["records"] == []
    finally:
        backend.adapters["local_tokenizer"].close()


async def test_actual_backend_trim_and_split_use_local_whole_request_counts(tmp_path):
    cfg = configuration(tmp_path)
    backend = ModelInferenceBackend(cfg)
    messages = Messages.from_openai(
        [{"role": "user", "content": "hello world hello world hello world"}]
    )
    try:
        assert await backend.model_input_tokens_messages("counter", messages) == 6
        trimmed = await backend.model_trim_messages("counter", "classify", messages, 2)
        assert await backend.model_input_tokens_messages("counter", trimmed.messages) <= 2
        chunks = await backend.model_split_messages("counter", messages, 2)
        assert len(chunks) >= 3
        assert all(
            [await backend.model_input_tokens_messages("counter", chunk) <= 2 for chunk in chunks]
        )
    finally:
        backend.adapters["local_tokenizer"].close()


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("missing", "model_provider"),
        ("limit", "evaluation_limit"),
        ("output", "model_invalid_output"),
    ],
)
def test_native_failures_keep_their_trusted_categories_through_real_scans(
    tmp_path, monkeypatch, failure, expected
):
    cfg = configuration(tmp_path, rules=(RULE,))
    if failure == "missing":
        (tmp_path / "synthetic-tokenizer.json").unlink()
    elif failure == "limit":
        model = cfg.models["counter"]
        local = replace(model.tokens.local, limits=replace(model.tokens.local.limits, max_tokens=1))
        cfg = replace(
            cfg, models={"counter": replace(model, tokens=replace(model.tokens, local=local))}
        )
    else:

        def invalid(*args):
            raise LocalTokenizerInvalidOutputError("Local tokenizer output is invalid")

        monkeypatch.setattr(LocalTokenizerAdapter, "_loaded", invalid)
    result = DSLScanner(cfg).scan_data(DATA)
    assert not result.is_complete and result.rule_matches == []
    assert result.evaluation_error_categories == {"Count": expected}
    assert str(tmp_path) not in json.dumps(result.evaluation_errors)


def test_actual_cli_uses_the_optional_local_adapter(tmp_path):
    doc = document(tmp_path)
    rules, data, path = tmp_path / "count.quarry", tmp_path / "input.json", tmp_path / "config.yaml"
    rules.write_text(RULE)
    data.write_text(json.dumps(DATA))
    doc["rules"] = [str(rules)]
    path.write_text(json.dumps(doc))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "quarry.cli",
            "scan",
            "--config",
            str(path),
            "--target",
            str(data),
            "--output-format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stderr
    (payload,) = json.loads(result.stdout)
    assert payload["complete"] and payload["matches"] == [{"rule": "Count"}]
    assert payload["model_usage"]["calls"] == 0


async def test_real_mcp_transport_preserves_local_count_result(tmp_path, monkeypatch):
    from quarry.plugin import server
    from quarry.plugin.service import QuarryPlugin

    monkeypatch.setattr(server, "_service", QuarryPlugin(runtime_config=configuration(tmp_path)))
    response = await server.mcp.call_tool(
        "quarry_scan", {"rule_text": RULE, "conversation": DATA, "explain": False}
    )
    payload = response.structured_content
    assert json.loads(response.content[0].text) == payload
    assert payload["completion_state"] == "complete"
    assert payload["matches"] == [{"rule_name": "Count", "metadata": {}}]
    assert payload["model_usage"]["calls"] == 0


def test_unused_configuration_creates_no_native_owner(tmp_path, monkeypatch):
    cfg = configuration(tmp_path)
    other = replace(cfg.models["counter"], adapter="synthetic", tokens=None)
    cfg = replace(cfg, models={**cfg.models, "other": other})

    def forbidden(*args):
        pytest.fail("unused local configuration created an owner")

    monkeypatch.setattr(LocalTokenizerAdapter, "__init__", forbidden)
    backend = ModelInferenceBackend(cfg, models_to_load=["other"], adapters={"synthetic": object()})
    assert "local_tokenizer" not in backend.adapters
