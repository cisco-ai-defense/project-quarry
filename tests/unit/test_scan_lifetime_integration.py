# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Retention policy reaches public APIs without changing accepted model semantics."""

import asyncio
import gc
import json
import sys
import weakref
from dataclasses import replace
from unittest.mock import patch

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.cache.async_cache import AsyncFunctionCache, scan_scope
from quarry.config import LocalTransformConfig, ModelCacheConfig, QuarryConfig
from quarry.integrations import model_inference as inference
from quarry.plugin.service import QuarryPlugin
from quarry.safety.limits import SafetyLimits
from quarry.scanner import DSLScanner
from quarry.sdk import QuarryGuard
from quarry.types.adapter_transforms import TransformCacheIdentity
from quarry.types.exceptions import LLMError
from tests.unit.model_fakes import ModelAdapter, model_entry, set_config_environment, v1_scanner

CONVERSATION = {"messages": [{"role": "user", "content": "synthetic"}]}
RULE = (
    'rule "R" condition: model("m").classify(messages) > .5 '
    'and model("m").complete(messages).content == "blocked" '
    'and model("m").input_tokens(messages) > 0'
)
PROFILES = [
    ("process", [], (1, 1, 1)),
    ("scan", [], (2, 2, 2)),
    ("scan", ["token_counts"], (2, 2, 1)),
    ("scan", ["model_responses"], (1, 1, 2)),
]


@pytest.fixture
def no_collection():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


@pytest.fixture
def results(monkeypatch):
    from quarry import scanner

    value = AsyncFunctionCache.expiring(max_entries=100, ttl_seconds=60, max_bytes=1_000_000)
    monkeypatch.setattr(inference, "result_cache", value)
    monkeypatch.setattr(scanner, "result_cache", value)
    return value


def config_for(tmp_path, retention="process", categories=()):
    return QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"m": model_entry(classify=0, tokens=True)},
            "model_cache": {"retention": retention, "reusable_categories": list(categories)},
            "llm_rate_limits": {"default": {"requests_per_interval": None}},
        },
        tmp_path,
        require_rules=False,
    )


def forbidden(*args, **kwargs):
    raise AssertionError("prepared policy unexpectedly parsed or compiled")


@pytest.mark.parametrize("retention,categories,expected", PROFILES)
@pytest.mark.parametrize(
    "entry", ["scanner", "async_scanner", "policy", "prepared", "sdk", "plugin", "cli"]
)
def test_public_entry_points_respect_each_category_lifetime(
    tmp_path, results, monkeypatch, capsys, retention, categories, expected, entry
):
    adapter = ModelAdapter()
    config = replace(
        config_for(tmp_path, retention, categories),
        quarry_rule_texts=(RULE,),
        safety_limits=SafetyLimits(rule_timeout=None),
    )
    backend = inference.ModelInferenceBackend(config, adapters={"fake": adapter})
    if entry in {"scanner", "async_scanner"}:
        scanner = DSLScanner(config, model_inference_backend=backend)
        run = (
            scanner.scan_data
            if entry == "scanner"
            else lambda c: asyncio.run(scanner.scan_data_async(c))
        )
    elif entry in {"sdk", "cli"}:
        rule = tmp_path / "r.rule"
        rule.write_text(RULE)
        path = tmp_path / "config.yaml"
        path.write_text(
            json.dumps(
                {
                    "config_version": 2,
                    "rules": [str(rule)],
                    "models": {"m": model_entry(classify=0, tokens=True)},
                    "model_cache": {"retention": retention, "reusable_categories": categories},
                }
            )
        )
        if entry == "sdk":
            guard = QuarryGuard(path, model_inference_backend=backend)
            run = guard.scan
        else:
            from quarry import cli

            monkeypatch.setattr(inference, "_default_adapters", lambda: {"fake": adapter})
            data = tmp_path / "conversation.json"
            data.write_text(json.dumps(CONVERSATION))

            def run(c):
                monkeypatch.setattr(
                    sys, "argv", ["quarry", "scan", "-c", str(path), "-t", str(data), "-o", "json"]
                )
                assert cli.main() == 1
                return json.loads(capsys.readouterr().out)[0]
    elif entry == "plugin":
        monkeypatch.setattr(inference, "_default_adapters", lambda: {"fake": adapter})
        service = QuarryPlugin(config)

        def run(c):
            return asyncio.run(service.scan(rule_text=RULE, conversation=c))
    else:
        policy = compile_policy(RULE, runtime_config=config, model_inference_backend=backend)
        if entry == "prepared":
            raw = policy.to_bytes()
            with (
                patch("quarry.dsl.parser.DSLParser.parse", forbidden),
                patch("quarry.scanner.compile_rule", forbidden),
                patch("quarry.engine.compiler.compile_rule", forbidden),
            ):
                policy = load_compiled_policy(
                    raw, runtime_config=config, model_inference_backend=backend
                )
                assert policy.to_bytes() == raw
        run = policy.scan
    for _ in range(2):
        result = run(CONVERSATION)
        if entry == "plugin":
            assert result["matched"] and result["completion_state"] == "complete"
        elif entry == "sdk":
            assert [item["rule"] for item in result] == ["R"]
        elif entry == "cli":
            assert result["complete"] and result["matches"] == [{"rule": "R"}]
        else:
            assert result.is_complete and result.rule_matches == ["R"]
    assert (adapter.classifications, adapter.completions, adapter.tokenizations) == expected
    if retention == "scan" and not categories:
        assert not results.store and not results._scan_entries and not results._scan_owners


@pytest.mark.parametrize("backend_mode", ["process", "scan"])
def test_shared_backend_uses_each_scanners_policy_without_reconfiguring_it(
    tmp_path, results, backend_mode
):
    adapter = ModelAdapter()
    config = config_for(tmp_path, backend_mode)
    backend = inference.ModelInferenceBackend(config, adapters={"fake": adapter})
    scanners = {
        mode: DSLScanner(
            replace(
                config, model_cache=ModelCacheConfig(retention=mode), quarry_rule_texts=(RULE,)
            ),
            model_inference_backend=backend,
        )
        for mode in ["process", "scan"]
    }
    for mode in ["process", "scan", "process", "scan"]:
        result = scanners[mode].scan_data(CONVERSATION)
        assert result.is_complete and result.rule_matches == ["R"]
    assert (adapter.classifications, adapter.completions, adapter.tokenizations) == (3, 3, 3)
    assert backend.config.model_cache.retention == backend_mode


@pytest.mark.asyncio
@pytest.mark.parametrize("versioned", [False, True])
@pytest.mark.parametrize(
    "retention,reusable,expected",
    [("process", [], 1), ("scan", [], 2), ("scan", ["local_transforms"], 1)],
)
async def test_local_transforms_follow_scan_policy_for_shared_and_instance_reuse(
    tmp_path, results, versioned, retention, reusable, expected
):
    class Adapter:
        local = True
        transform_parts = ("text",)
        transform_cache = TransformCacheIdentity("synthetic.retention", "1") if versioned else None
        calls = 0

        async def transform(self, config, part, *, timeout=None):
            self.calls += 1
            return part["text"].upper()

    adapter = Adapter()
    # The backend has process defaults; a scanner supplies the different policy.
    config = QuarryConfig(config_version=2, transforms={"clean": LocalTransformConfig("owned")})
    backend = inference.ModelInferenceBackend(config, adapters={"owned": adapter})
    policy = compile_policy(
        'rule "R" condition: messages.transform("clean")[0].content == "SYNTHETIC"',
        runtime_config=replace(
            config, model_cache=ModelCacheConfig(retention=retention, reusable_categories=reusable)
        ),
        model_inference_backend=backend,
    )
    for _ in range(2):
        result = await policy.scan_async(CONVERSATION)
        assert result.is_complete and result.rule_matches == ["R"]
    # Unversioned DSL parts deliberately preserve independent stateful adapter calls.
    assert adapter.calls == (expected if versioned else 2)
    if retention == "scan" and not reusable:
        assert not results.store
    results.clear()
    adapter.calls = 0
    for _ in range(2):
        with scan_scope(
            cache_policy=ModelCacheConfig(retention=retention, reusable_categories=reusable)
        ):
            for _ in range(2):
                value = await backend.transform_unit("clean", {"type": "text", "text": "synthetic"})
                assert value.text == "SYNTHETIC"
    assert adapter.calls == expected


@pytest.mark.parametrize(
    "retention,reusable,expected", [("process", [], 1), ("scan", [], 2), ("scan", ["probes"], 1)]
)
def test_real_prober_work_obeys_category_policy(tmp_path, results, retention, reusable, expected):
    class Topic:
        calls = 0

        def off_topic(self, intent, action):
            assert (intent, action) == ("synthetic", "unrelated")
            self.calls += 1
            return True

    topic = Topic()
    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            allow_stub_probers=True,
            model_cache=ModelCacheConfig(retention=retention, reusable_categories=reusable),
            quarry_rule_texts=(
                'rule "R" condition: any m in assistant_messages where off_topic(m.content)',
            ),
        )
    )
    scanner.resource_manager.topic_prober = topic
    data = {"messages": [*CONVERSATION["messages"], {"role": "assistant", "content": "unrelated"}]}
    for _ in range(2):
        result = scanner.scan_data(data)
        assert result.is_complete and result.rule_matches == ["R"]
    assert topic.calls == expected


@pytest.mark.parametrize("retention,expected", [("process", 1), ("scan", 2)])
def test_version_one_model_requests_use_the_same_lifetime_policy(
    tmp_path, results, monkeypatch, retention, expected
):
    from tests.unit.model_fakes import FakeAdapter

    set_config_environment(monkeypatch)
    adapter = FakeAdapter()
    original = v1_scanner(tmp_path, 'llm_input_tokens("gemma4", messages) > 0', adapter)
    scanner = DSLScanner(
        replace(original.config, model_cache=ModelCacheConfig(retention=retention)),
        model_inference_backend=original.model_inference_backend,
    )
    for _ in range(2):
        result = scanner.scan_data(CONVERSATION)
        assert result.is_complete and result.rule_matches == ["R"]
    assert len(adapter.token_configs) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("exit", ["finish", "close", "failure", "cancel"])
async def test_actual_deferred_work_keeps_scan_table_then_releases_without_collection(
    tmp_path, results, exit, no_collection
):
    class Raw(float):
        pass

    references = []

    class Adapter(ModelAdapter):
        async def score(self, _config, _messages, *, timeout=None):
            result = Raw(0.9)
            references.append(weakref.ref(result))
            return result

    config = replace(
        config_for(tmp_path, "scan"),
        models=QuarryConfig.from_mapping(
            {"config_version": 2, "models": {"m": model_entry(score=True)}},
            tmp_path,
            require_rules=False,
        ).models,
        quarry_rule_texts=('rule "R" condition: model("m").score(messages,"criterion") > .5',),
    )
    from tests.unit.model_fakes import passthrough_builder

    backend = inference.ModelInferenceBackend(
        config, adapters={"fake": Adapter()}, message_builders={"test": passthrough_builder}
    )
    scanner = DSLScanner(config, model_inference_backend=backend)
    deferred = []
    result = await scanner._scan_data_with_parameters_async(
        CONVERSATION,
        {},
        enable_explain=True,
        rules_subset={"R"},
        decision_paths=True,
        deferred_evidence=deferred,
    )
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(deferred) == 1 and len(results.store) == 1 and references[0]() is not None
    owner = deferred.pop()
    if exit == "close":
        owner.close()
    elif exit == "finish":
        result = await owner(result)
    else:

        async def failed(_result):
            raise asyncio.CancelledError if exit == "cancel" else ValueError("synthetic")

        owner._finish = failed
        with pytest.raises(asyncio.CancelledError if exit == "cancel" else ValueError):
            await owner(result)
    for _ in range(4):
        await asyncio.sleep(0)
    assert not results.store and not results._scan_owners and references[0]() is None


@pytest.mark.asyncio
async def test_direct_backend_calls_use_backend_policy_and_scoped_override(tmp_path, results):
    from quarry.types.messages import Messages

    adapter = ModelAdapter()
    backend = inference.ModelInferenceBackend(
        config_for(tmp_path, "scan"), adapters={"fake": adapter}
    )
    messages = Messages.from_openai(CONVERSATION["messages"])
    for _ in range(2):
        assert await backend.model_classify_messages("m", messages) > 0.5
        assert not results.store
    with scan_scope(cache_policy=ModelCacheConfig(retention="process")):
        for _ in range(2):
            assert await backend.model_classify_messages("m", messages) > 0.5
    assert adapter.classifications == 3 and len(results.store) == 1


@pytest.mark.parametrize("retention,categories,expected", PROFILES)
def test_runtime_retention_policy_preserves_compiled_bytes_and_decision_identity(
    tmp_path, results, retention, categories, expected
):
    adapter = ModelAdapter()
    config = config_for(tmp_path)
    backend = inference.ModelInferenceBackend(config, adapters={"fake": adapter})
    policy = compile_policy(RULE, runtime_config=config, model_inference_backend=backend)
    raw = policy.to_bytes()
    selected = replace(
        config, model_cache=ModelCacheConfig(retention=retention, reusable_categories=categories)
    )
    current = compile_policy(RULE, runtime_config=selected, model_inference_backend=backend)
    assert current.to_bytes() == raw
    loaded = load_compiled_policy(raw, runtime_config=selected, model_inference_backend=backend)
    assert loaded.to_bytes() == raw
    assert loaded.decision_tree() == policy.decision_tree()


@pytest.mark.asyncio
@pytest.mark.parametrize("category", [[], ["model_responses"]])
async def test_scan_readers_validate_alias_heads_and_reject_only_the_physical_reply(
    tmp_path, results, category
):
    from types import SimpleNamespace

    from quarry.integrations.model_usage import model_call_budget
    from quarry.types.messages import Messages

    class Adapter:
        calls = 0

        async def classify_messages(self, config, messages, tools, *, timeout=None):
            self.calls += 1
            return SimpleNamespace(
                chunks=[SimpleNamespace(probabilities=[self.calls / 10, 1 - self.calls / 10])]
            )

    adapter = Adapter()
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "model_cache": {"retention": "scan", "reusable_categories": category},
            "models": {
                name: {"adapter": "owned", "model": "owned", "classify": {"class_index": index}}
                for name, index in [("left", 0), ("right", 1), ("bad", 3)]
            },
        },
        tmp_path,
        require_rules=False,
    )
    backend = inference.ModelInferenceBackend(config, adapters={"owned": adapter})
    messages = Messages.from_openai(CONVERSATION["messages"])
    with scan_scope(cache_policy=config.model_cache), model_call_budget(2) as budget:
        assert await backend.model_classify_messages("left", messages) == 0.1
        assert await backend.model_classify_messages("right", messages) == 0.9
        assert adapter.calls == 1
        with pytest.raises(LLMError, match="outside"):
            await backend.model_classify_messages("bad", messages)
        assert await backend.model_classify_messages("left", messages) == 0.2
        assert await backend.model_classify_messages("right", messages) == 0.8
        assert budget.calls_by_alias == {"left": 2}
    assert adapter.calls == 2
    assert len(results.store) == (1 if category else 0)


@pytest.mark.parametrize("retention,expected", [("process", 1), ("scan", 2)])
@pytest.mark.parametrize("malformed", [False, True])
def test_refusal_values_and_invalid_replies_keep_their_error_contract(
    tmp_path, results, retention, expected, malformed
):
    class Adapter(ModelAdapter):
        async def complete_message(self, config, messages, tools, *, timeout=None):
            self.completions += 1
            if malformed:
                return {"role": "assistant", "content": object()}
            raise LLMError(
                "synthetic provider text", {"code": "model_refused", "reason": "provider_stop"}
            )

    adapter = Adapter()
    config = replace(
        config_for(tmp_path, retention),
        quarry_rule_texts=('rule "R" condition: refused(model("m").complete(messages))',),
    )
    backend = inference.ModelInferenceBackend(config, adapters={"fake": adapter})
    scanner = DSLScanner(config, model_inference_backend=backend)
    for _ in range(2):
        result = scanner.scan_data(CONVERSATION)
        if malformed:
            assert not result.is_complete and not result.rule_matches
            assert "synthetic provider text" not in str(result.evaluation_errors)
        else:
            assert result.is_complete and result.rule_matches == ["R"]
    assert adapter.completions == (2 if malformed else expected)
