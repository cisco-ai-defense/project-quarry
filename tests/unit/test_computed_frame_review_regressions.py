# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Folded frame reload and resource attribution through computed frames."""

import asyncio
import json

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.cache.async_cache import result_cache
from quarry.config import QuarryConfig
from quarry.decision_report import path_report, report_tree, scanner_digests
from quarry.integrations.model_inference import ModelInferenceBackend
from tests.unit.model_fakes import NoWaitLimiter, model_entry, passthrough_builder
from tests.unit.test_adapter_transform_dsl import Adapter
from tests.unit.test_nested_rule_runtime_contracts import _policy

DATA = {"messages": [{"role": "user", "content": "needle"}]}
KEY = bytes(range(32))


class ResourceAdapter(Adapter):
    local = False

    async def score(self, config, messages, *, timeout=None):
        self.inputs.append(messages)
        return 0.9


def runtime(
    tmp_path,
    *,
    model="synthetic-one",
    endpoint="https://a.example/v1",
    unused="unread-one",
    text="synthetic criterion",
):
    reader = model_entry(classify=0, score=True, physical_model=model)
    reader["endpoint"] = endpoint
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"reader": reader, "unused": model_entry(classify=0, physical_model=unused)},
            "transforms": {"rewrite": {"model": "reader"}},
            "texts": {"criterion": text},
        },
        tmp_path,
        require_rules=False,
    )
    adapter = ResourceAdapter()
    backend = ModelInferenceBackend(
        config,
        adapters={"fake": adapter},
        message_builders={"test": passthrough_builder},
        rate_limiter=NoWaitLimiter(),
    )
    return config, backend, adapter


@pytest.mark.parametrize("called", [False, True])
@pytest.mark.parametrize("model", [False, True])
@pytest.mark.parametrize(
    "form,matched", [("or", True), ("and", False), ("not", False), ("reverse", True)]
)
def test_zero_bound_filtered_counts_keep_reload_and_dispatch(
    tmp_path, monkeypatch, called, model, form, matched
):
    config, backend, adapter = runtime(tmp_path)
    body = 'model("reader").classify(m) > 0.5' if model else 'm.content contains "x"'
    frame = f"(any m in (messages + messages) where {body})"
    count = 'count([n for n in messages if n.content contains "y"])'
    condition = {
        "or": f"{frame} or {count} >= 0",
        "and": f"{frame} and {count} < 0",
        "not": f"not ({frame} or {count} >= 0)",
        "reverse": f"{count} >= 0 or {frame}",
    }[form]
    source = 'rule "Read" condition:' + condition
    if called:
        source += '\nrule "Caller" imports: read="Read" condition: read(messages)'
    policy = _policy(source, config, False, backend)
    wire = policy.to_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("reload reached source parsing or compilation")

    monkeypatch.setattr("quarry.dsl.parser.DSLParser.parse", forbidden)
    monkeypatch.setattr("quarry.scanner.compile_rule", forbidden)
    restored = load_compiled_policy(wire, runtime_config=config, model_inference_backend=backend)
    assert restored.to_bytes() == wire
    observations = []
    for current in [policy, restored]:
        current_observations = []
        for data in [DATA, {"messages": [{"role": "user", "content": "different"}]}]:
            result_cache.clear()
            adapter.inputs.clear()
            result = current.scan(data, explain=True)
            assert result.is_complete
            assert result.rule_matches == (["Caller" if called else "Read"] if matched else [])
            current_observations.append((result.rule_matches, list(adapter.inputs)))
            if not model:
                assert adapter.inputs == []
        observations.append(current_observations)
    assert observations[0] == observations[1]
    assert adapter.parts == []


@pytest.mark.parametrize("mutation", ["missing-plan", "missing-spec", "bad-source", "bad-cost"])
def test_computed_plans_still_require_validated_data(tmp_path, mutation):
    config, backend, _ = runtime(tmp_path)
    source = (
        'rule "Read" condition: '
        '(any m in (messages + messages) where model("reader").classify(m)>0.5) '
        'or count([n for n in messages if n.content contains "y"]) >= 0'
    )
    policy = compile_policy(source, runtime_config=config, model_inference_backend=backend)
    wire = json.loads(policy.to_bytes())
    artifact = wire["rule"]["artifact"]
    atom_id, _, _ = artifact["computed_conditions"][0]
    if mutation == "missing-plan":
        artifact["computed_conditions"].clear()
    elif mutation == "missing-spec":
        artifact["atom_specs"] = [s for s in artifact["atom_specs"] if s["atom_id"] != atom_id]
    elif mutation == "bad-source":
        artifact["computed_conditions"][0][1] = {"kind": "literal", "value": 4, "children": []}
    else:
        spec = next(s for s in artifact["atom_specs"] if s["atom_id"] == atom_id)
        spec["cost"] = "pure"
    with pytest.raises(ValueError):
        load_compiled_policy(
            json.dumps(wire).encode(), runtime_config=config, model_inference_backend=backend
        )


CONDITIONS = {
    "child": 'any m in (messages+messages) where model("reader").classify(m)>0.5',
    "source": 'any m in messages.transform("rewrite") where m.content contains "NEEDLE"',
    "called-child": "any m in (messages+messages) where leaf(m)",
    "text-child": "any m in (messages+messages) where "
    'model("reader").score(m,text_ref("criterion"))>0.5',
}


@pytest.mark.parametrize(
    "shape,restored",
    [
        (shape, restored)
        for shape in CONDITIONS
        for restored in [False, True]
        if shape != "text-child" or not restored
    ],
)
@pytest.mark.parametrize("change", ["model", "location", "unread", "text"])
def test_computed_decision_paths_attribute_only_the_resources_they_read(
    tmp_path, restored, shape, change
):
    def observe(**settings):
        result_cache.clear()
        config, backend, adapter = runtime(tmp_path, **settings)
        source = 'rule "Read" condition:' + CONDITIONS[shape]
        if shape == "called-child":
            source = (
                'rule "Leaf" condition:model("reader").classify(messages)>0.5\n'
                'rule "Read" imports: leaf="Leaf" condition:' + CONDITIONS[shape]
            )
            policy = _policy(source, config, restored, backend)
        else:
            policy = compile_policy(
                source, runtime_config=config, model_inference_backend=backend, evidence_key=KEY
            )
            if restored:
                wire = policy.to_bytes()
                policy = load_compiled_policy(
                    wire, runtime_config=config, model_inference_backend=backend
                )
                assert policy.to_bytes() == wire
        result = asyncio.run(policy.scan_many_async([DATA], decision_paths=True))[0]
        assert result.is_complete and result.rule_matches == ["Read"]
        assert adapter.parts or adapter.inputs
        artifact = policy._scanner.compiled_rules["Read"]
        tree = report_tree(artifact, scanner_digests(policy._scanner, KEY))
        return path_report(
            tree,
            result.decision_paths["Read"],
            matched=True,
            conversation=DATA,
            parameter_digest=result.parameter_digest,
            explain=False,
            explore_rate=0.0,
        )

    original = observe()
    settings = {
        "model": {"model": "synthetic-two"},
        "location": {"endpoint": "https://b.example/v1"},
        "unread": {"unused": "unread-two"},
        "text": {"text": "different synthetic criterion"},
    }[change]
    changed = observe(**settings)

    def node_ids(path):
        return [entry["node"] for entry in path["path"]]

    relevant = change in {"model", "location"} or change == "text" and shape == "text-child"
    assert (changed["config_sha256"] != original["config_sha256"]) is relevant
    if relevant and change != "location":
        assert node_ids(changed) != node_ids(original)
        assert changed["tree_sha256"] != original["tree_sha256"]
    else:
        assert node_ids(changed) == node_ids(original)
        assert changed["tree_sha256"] == original["tree_sha256"]


def test_text_resource_serialization_keeps_its_existing_refusal(tmp_path):
    config, backend, _ = runtime(tmp_path)
    policy = compile_policy(
        'rule "Read" condition:' + CONDITIONS["text-child"],
        runtime_config=config,
        model_inference_backend=backend,
        evidence_key=KEY,
    )
    with pytest.raises(ValueError, match="text_ref resources.*cannot serialize"):
        policy.to_bytes()
