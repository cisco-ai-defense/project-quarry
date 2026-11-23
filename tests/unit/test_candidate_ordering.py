# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Only attested native local candidates receive the explicit pre-service cost."""

from dataclasses import replace
from types import SimpleNamespace

import orjson
import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.dsl.parser import DSLParser
from quarry.engine.atoms import Cost
from quarry.engine.compiler import compile_rule
from quarry.integrations.candidate_checks import LocalCandidatePattern
from quarry.integrations.candidate_providers import LocalCandidateProvider
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.safety.limits import SafetyLimits
from quarry.safety.local_match_work import LocalMatchWorkConfig, LocalMatchWorkGroup
from quarry.types.candidates import CandidateBatchIdentity
from tests.unit.model_fakes import NoWaitLimiter

DATA = {"messages": [{"role": "user", "content": "abcdef"}]}


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("matches", [False, True])
@pytest.mark.parametrize("alias", ["zeta", "paid-service"])
def test_explicit_local_candidates_precede_written_earlier_service(
    tmp_path, restored, matches, alias
):
    order = []

    def check(text):
        order.append("candidate")
        return True

    provider = LocalCandidateProvider(
        CandidateBatchIdentity("synthetic-candidates", "1", "heads"),
        ("left",),
        (LocalCandidatePattern("left", "abc" if matches else "xyz", ("synthetic",)),),
        checks={"synthetic": check},
    )

    class Remote:
        async def classify_messages(self, config, messages, tools, *, timeout=None):
            order.append("service")
            return SimpleNamespace(chunks=[SimpleNamespace(probabilities=[0.9, 0.1])])

    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                alias: {
                    "adapter": "native",
                    "model": "synthetic-candidates",
                    "revision": "1",
                    "category": "SYNTHETIC",
                    "spans": {
                        "backend": "candidates",
                        "batch": "heads",
                        "channel": "left",
                        "settings_digest": provider.settings_digest,
                    },
                },
                "remote": {
                    "adapter": "remote",
                    "model": "synthetic-service",
                    "classify": {"class_index": 0},
                },
            },
        },
        tmp_path,
        require_rules=False,
    )
    config = replace(config, safety_limits=SafetyLimits(rule_timeout=None))
    backend = ModelInferenceBackend(
        config, adapters={"native": provider, "remote": Remote()}, rate_limiter=NoWaitLimiter()
    )
    source = f'rule "Read" condition: model("remote").classify(messages) > 0.5 and any s in model("{alias}").spans(messages, 0.5) where true'
    try:
        compiled = compile_policy(source, runtime_config=config, model_inference_backend=backend)
        costs = {spec.cost for spec in compiled._scanner.compiled_rules["Read"].atom_specs.values()}
        assert Cost.MODEL in costs and Cost.LLM in costs
        if restored:
            wire = compiled.to_bytes()
            with pytest.raises(ValueError, match="incompatible"):
                load_compiled_policy(wire, runtime_config=config)
            # A remote atom cannot borrow another alias's native ordering proof.
            forged = orjson.loads(wire)
            for spec in forged["rule"]["artifact"]["atom_specs"]:
                if spec["cost"] == Cost.LLM.value:
                    spec["cost"] = Cost.MODEL.value
            with pytest.raises(ValueError, match="incompatible"):
                load_compiled_policy(
                    orjson.dumps(forged), runtime_config=config, model_inference_backend=backend
                )
            compiled = load_compiled_policy(
                wire, runtime_config=config, model_inference_backend=backend
            )
            assert compiled.to_bytes() == wire
        result = compiled.scan(DATA)
        assert result.is_complete, result.evaluation_errors
        assert result.rule_matches == (["Read"] if matches else [])
        assert order == (["candidate", "service"] if matches else [])
        assert result.model_usage["calls"] == int(matches)
    finally:
        provider.close()


def test_unknown_or_unbound_names_keep_default_service_cost_and_compiler_context_is_restored():
    rule = DSLParser().parse(
        'rule "Read" condition: any s in model("local-patterns").spans(messages, 0.5) where true'
    )
    ordinary = compile_rule(rule)
    assert ordinary.ok
    hinted = compile_rule(rule, local_candidate_aliases=frozenset({"local-patterns"}))
    assert hinted.ok
    assert {spec.cost for spec in ordinary.artifact.atom_specs.values()} == {Cost.LLM}
    assert {spec.cost for spec in hinted.artifact.atom_specs.values()} == {Cost.MODEL}
    after = compile_rule(rule)
    assert after.ok and after.artifact.dfa_hash == ordinary.artifact.dfa_hash
    assert after.artifact.atom_specs == ordinary.artifact.atom_specs


def test_nested_service_dependency_keeps_conservative_service_cost():
    rule = DSLParser().parse(
        'rule "Read" condition: any s in model("candidate").spans(messages, 0.5) where model("remote").classify(messages) > 0.5'
    )
    result = compile_rule(rule, local_candidate_aliases=frozenset({"candidate"}))
    assert result.ok
    assert {spec.cost for spec in result.artifact.atom_specs.values()} == {Cost.LLM}


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("matches", [False, True])
@pytest.mark.parametrize("metered", [False, True])
def test_deferred_candidate_collection_gates_dependent_service_and_keeps_work_owner(
    tmp_path, restored, matches, metered
):
    order = []

    def check(text):
        order.append("candidate")
        return True

    provider = LocalCandidateProvider(
        CandidateBatchIdentity("synthetic-candidates", "1", "heads"),
        ("left",),
        (LocalCandidatePattern("left", "abc" if matches else "xyz", ("synthetic",)),),
        checks={"synthetic": check},
    )

    class Remote:
        async def classify_messages(self, config, messages, tools, *, timeout=None):
            order.append("service")
            return SimpleNamespace(chunks=[SimpleNamespace(probabilities=[0.9, 0.1])])

    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "candidate": {
                    "adapter": "native",
                    "model": "synthetic-candidates",
                    "revision": "1",
                    "category": "SYNTHETIC",
                    "spans": {
                        "backend": "candidates",
                        "batch": "heads",
                        "channel": "left",
                        "settings_digest": provider.settings_digest,
                    },
                },
                "remote": {
                    "adapter": "remote",
                    "model": "synthetic-service",
                    "classify": {"class_index": 0},
                },
            },
        },
        tmp_path,
        require_rules=False,
    )
    config = replace(
        config,
        safety_limits=SafetyLimits(rule_timeout=None),
        local_match_work=(
            LocalMatchWorkConfig(
                (LocalMatchWorkGroup("synthetic", ("Read",), 1_000_000_000, "enforce"),)
            )
            if metered
            else None
        ),
    )
    backend = ModelInferenceBackend(
        config, adapters={"native": provider, "remote": Remote()}, rate_limiter=NoWaitLimiter()
    )
    source = 'rule "Read" condition: true collect: [s for s in model("candidate").spans(messages, 0.5) if model("remote").classify(messages) > 0.5]'
    try:
        compiled = compile_policy(source, runtime_config=config, model_inference_backend=backend)
        if restored:
            wire = compiled.to_bytes()
            compiled = load_compiled_policy(
                wire, runtime_config=config, model_inference_backend=backend
            )
        result = compiled.scan(DATA, logical_reads=True)
        assert result.is_complete, result.evaluation_errors
        assert result.rule_matches == (["Read"] if matches else [])
        assert order == (["candidate", "service"] if matches else [])
        assert result.model_usage["calls"] == int(matches)
        if matches:
            span = result.outputs["Read"].items[0].value
            assert (span.start, span.end) == (0, 3)
        if metered:
            assert result.local_match_work["complete"]
            assert result.local_match_work["groups"][0]["known_cpu_ns"] > 0
    finally:
        provider.close()
