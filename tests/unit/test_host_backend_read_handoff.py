# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Delegating host backends retain one owned logical model operation."""

import asyncio
import threading
from dataclasses import replace

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.cache.async_cache import result_cache
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.integrations.read_usage import model_read_usage, read_usage_scope
from quarry.safety.limits import SafetyLimits
from quarry.types.adapter_transforms import TransformCacheIdentity
from tests.unit.model_fakes import ModelAdapter, NoWaitLimiter, model_entry, passthrough_builder
from tests.unit.test_adapter_transform_dsl import configured

DATA = {"messages": [{"role": "user", "content": "alice@example.com"}]}
OPERATIONS = [
    ("model_score_messages", 'llm_score("reader", "policy", conversation()) > 0.5'),
    ("model_input_tokens_messages", 'llm_input_tokens("reader", conversation()) > 0'),
    ("model_score_messages", 'model("reader").score(messages, "policy") > 0.5'),
    ("model_classify_messages", 'model("reader").classify(messages) > 0.5'),
    ("model_complete", 'model("reader").complete(messages).content == "safe"'),
    ("model_input_tokens_messages", 'model("reader").input_tokens(messages) > 0'),
    ("model_span_messages", 'any s in model("reader").spans(messages, 0.5) where true'),
    (
        "model_classification",
        'all m in user_messages where model_score("reader", m.content, 0) > 0.5',
    ),
    (
        "model_classification",
        'all m in user_messages where model_scan("reader", m.content, 0.5, 0)',
    ),
]


def policy(tmp_path, method, condition, seen, *, wrapped, restored, delegation="plain"):
    class Delegating(ModelInferenceBackend):
        pass

    async def delegate(self, *args, **kwargs):
        seen.append((threading.get_ident(), asyncio.get_running_loop()))
        native = getattr(super(Delegating, self), method)
        if delegation == "child":
            return await asyncio.create_task(native(*args, **kwargs))
        if delegation == "reader":
            with read_usage_scope("Other"):
                return await native(*args, **kwargs)
        if delegation == "twice":
            await native(*args, **kwargs)
        return await native(*args, **kwargs)

    setattr(Delegating, method, delegate)
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "reader": model_entry(
                    score=True,
                    classify=0,
                    tokens=True,
                    spans=0,
                    category="EMAIL",
                    physical_model="text-model",
                )
            },
        },
        tmp_path,
        require_rules=False,
    )
    config = replace(config, safety_limits=SafetyLimits(rule_timeout=None))
    adapter = ModelAdapter()
    service = (Delegating if wrapped else ModelInferenceBackend)(
        config,
        adapters={"fake": adapter},
        message_builders={"test": passthrough_builder},
        rate_limiter=NoWaitLimiter(),
    )
    result = compile_policy(
        'rule "Read" condition: ' + condition,
        runtime_config=config,
        model_inference_backend=service,
    )
    if restored:
        result = load_compiled_policy(
            result.to_bytes(),
            runtime_config=config,
            model_inference_backend=service,
        )
    return result


@pytest.mark.parametrize(("method", "condition"), OPERATIONS)
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("batch", [False, True])
async def test_delegating_host_backend_adopts_the_original_logical_read(
    tmp_path,
    method,
    condition,
    restored,
    batch,
):
    observed = []
    host = (threading.get_ident(), asyncio.get_running_loop())
    results = []
    for wrapped in [False, True]:
        result_cache.clear()
        scanner = policy(tmp_path, method, condition, observed, wrapped=wrapped, restored=restored)
        result = (
            (await scanner.scan_many_async([DATA], logical_reads=True))[0]
            if batch
            else (await scanner.scan_async(DATA, logical_reads=True))
        )
        assert result.is_complete, result.evaluation_errors
        results.append(result)
    assert observed == [host]
    assert results[1].rule_matches == results[0].rule_matches
    assert results[1].logical_reads == results[0].logical_reads
    assert results[1].logical_reads["aggregates"]["totals"]["observations"] == 1
    assert results[1].model_usage == results[0].model_usage
    assert results[1].model_usage["calls"] == 1
    assert not results[1].model_usage["truncated"]


@pytest.mark.parametrize("delegation", ["child", "reader", "twice"])
async def test_transferred_handoff_stays_single_use_and_owned(tmp_path, delegation):
    result_cache.clear()
    observed = []
    scanner = policy(
        tmp_path,
        "model_classify_messages",
        'model("reader").classify(messages) > 0.5',
        observed,
        wrapped=True,
        restored=False,
        delegation=delegation,
    )
    with model_read_usage() as session:
        result = await scanner.scan_async(DATA, logical_reads=True)
    assert result.is_complete and result.rule_matches == ["Read"]
    assert observed == [(threading.get_ident(), asyncio.get_running_loop())]
    rows = session.snapshot()["rows"]
    assert sum(row["observations"] for row in rows) == 2
    if delegation == "reader":
        assert {row["rule"]: row["observations"] for row in rows} == {
            "Read": 1,
            "Other": 1,
        }


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize(
    ("kind", "method"),
    [
        (kind, method)
        for kind in ["local", "versioned", "model", "model-batch"]
        for method in [
            "transform_unit",
            "transform_message_unit",
            "transform_parts",
            "transform_cache_identity",
        ]
    ]
    + [(kind, "transform_units") for kind in ["versioned", "model-batch"]],
)
async def test_delegating_transform_hooks_keep_host_and_read_provenance(
    tmp_path, restored, kind, method
):
    host = (threading.get_ident(), asyncio.get_running_loop())
    seen = []

    class Batched(ModelInferenceBackend):
        def transform_cache_identity(self, name):
            return TransformCacheIdentity("synthetic.host-handoff.batch", "1")

    base = Batched if kind == "model-batch" else ModelInferenceBackend

    class Delegating(base):
        pass

    async def delegate(self, *args, **kwargs):
        seen.append((threading.get_ident(), asyncio.get_running_loop()))
        return await getattr(super(Delegating, self), method)(*args, **kwargs)

    def sync_delegate(self, *args, **kwargs):
        seen.append((threading.get_ident(), asyncio.get_running_loop()))
        return getattr(super(Delegating, self), method)(*args, **kwargs)

    setattr(
        Delegating,
        method,
        sync_delegate if method in {"transform_parts", "transform_cache_identity"} else delegate,
    )
    results = []
    for wrapped in [False, True]:
        result_cache.clear()
        config, _, adapter = configured(tmp_path, model=kind.startswith("model"))
        config = replace(config, safety_limits=SafetyLimits(rule_timeout=None))
        if kind == "versioned":
            adapter.transform_cache = TransformCacheIdentity("synthetic.host-handoff", "1")
        service = (Delegating if wrapped else base)(
            config, adapters={"fake": adapter}, rate_limiter=NoWaitLimiter()
        )
        scanner = compile_policy(
            'rule "Read" condition: model("judge").classify(messages.transform("convert")) > 0.5',
            runtime_config=config,
            model_inference_backend=service,
        )
        if restored:
            scanner = load_compiled_policy(
                scanner.to_bytes(), runtime_config=config, model_inference_backend=service
            )
        result = await scanner.scan_async(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "same"},
                            {"type": "text", "text": "next"},
                        ],
                    }
                ]
            },
            logical_reads=True,
        )
        assert result.is_complete and result.rule_matches == ["Read"], result.evaluation_errors
        results.append(result)
    assert seen and all(value == host for value in seen)
    assert results[1].logical_reads == results[0].logical_reads
