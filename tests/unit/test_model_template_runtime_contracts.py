# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Model templates remain bounded and transparent to policy execution."""

from copy import deepcopy

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.types.exceptions import ConfigurationError
from tests.unit.model_fakes import ModelAdapter, model_entry, passthrough_builder


@pytest.mark.parametrize("depth", [32, 33])
def test_default_inheritance_depth_counts_the_final_model(tmp_path, depth):
    templates = {"t00": {"model": "owned/base"}}
    for number in range(1, depth - 1):
        templates[f"t{number:02}"] = {"extends": f"t{number - 1:02}"}
    document = {
        "model_templates": templates,
        "models": {"z_model": {"extends": f"t{depth - 2:02}"}},
    }
    if depth == 33:
        with pytest.raises(ConfigurationError, match="inheritance limit of 32"):
            QuarryConfig.from_mapping(document, tmp_path, require_rules=False)
    else:
        config = QuarryConfig.from_mapping(document, tmp_path, require_rules=False)
        assert config.models["z_model"].model == "owned/base"


@pytest.mark.parametrize("count", [256, 257])
def test_default_definition_limit_includes_unused_templates(tmp_path, count):
    document = {
        "model_templates": {f"t{n:03}": {"model": "owned/base"} for n in range(count - 1)},
        "models": {"runtime": {"model": "owned/model"}},
    }
    if count == 257:
        with pytest.raises(ConfigurationError, match="more than 256"):
            QuarryConfig.from_mapping(document, tmp_path, require_rules=False)
    else:
        config = QuarryConfig.from_mapping(document, tmp_path, require_rules=False)
        assert set(config.models) == {"runtime"}


def test_recursive_model_template_container_fails_before_schema_parsing(tmp_path):
    schema = {"type": "object"}
    schema["properties"] = {"recursive": schema}
    document = {"model_templates": {"base": {"complete": {"response_schema": schema}}}}
    with pytest.raises(ConfigurationError, match="recursive container"):
        QuarryConfig.from_mapping(document, tmp_path, require_rules=False)
    assert schema["properties"]["recursive"] is schema


def test_shared_nonrecursive_definitions_are_valid_and_not_mutated(tmp_path):
    shared = {"model": "owned/model"}
    document = {"model_templates": {"base": shared}, "models": {"first": shared, "second": shared}}
    before = deepcopy(document)
    config = QuarryConfig.from_mapping(document, tmp_path, require_rules=False)
    assert config.models["first"] == config.models["second"]
    assert document == before
    assert document["models"]["first"] is document["model_templates"]["base"]


def test_oversized_expanded_environment_value_is_not_exposed_in_error(tmp_path, monkeypatch):
    marker = "OWNED_PRIVATE_MODEL_SETTING_"
    monkeypatch.setenv("QUARRY_TEMPLATE_PRIVATE_VALUE", marker + "x" * 1_000_000)
    document = {
        "model_templates": {"base": {"model": "${QUARRY_TEMPLATE_PRIVATE_VALUE}"}},
        "models": {"runtime": {"extends": "base"}},
    }
    with pytest.raises(ConfigurationError, match="aggregate limit of 1000000 bytes") as failure:
        QuarryConfig.from_mapping(document, tmp_path, require_rules=False)
    assert marker not in str(failure.value)


@pytest.mark.parametrize("version", [1, 2])
def test_template_and_written_out_models_have_identical_immutable_runtime_values(tmp_path, version):
    base = {
        "adapter": "owned",
        "model": "owned/base",
        "score": {"boundary_seed": 7, "tokens": {"negative": "0", "positive": "1"}},
    }
    inherited = QuarryConfig.from_mapping(
        {
            "config_version": version,
            "model_templates": {"base": base},
            "models": {"judge": {"extends": "base", "model": "owned/child"}},
        },
        tmp_path,
        require_rules=False,
    )
    flat = QuarryConfig.from_mapping(
        {"config_version": version, "models": {"judge": {**base, "model": "owned/child"}}},
        tmp_path,
        require_rules=False,
    )
    assert inherited.models == flat.models
    assert inherited.get_safety_limits() == flat.get_safety_limits()
    with pytest.raises(TypeError):
        inherited.models["new"] = inherited.models["judge"]


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True])
def test_template_policy_uses_the_same_model_identity_and_request_as_flat_config(
    tmp_path, restored, explain
):
    definition = model_entry(physical_model="owned/judge", score=True)
    flat = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"judge": definition}}, tmp_path, require_rules=False
    )
    inherited = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "model_templates": {"base": definition},
            "models": {"judge": {"extends": "base"}},
        },
        tmp_path,
        require_rules=False,
    )
    results = []
    source = 'rule "Read" condition: model("judge").score(messages, "criterion") > 0.5'
    for config in (flat, inherited):
        adapter = ModelAdapter()
        backend = ModelInferenceBackend(
            config, adapters={"fake": adapter}, message_builders={"test": passthrough_builder}
        )
        policy = compile_policy(source, runtime_config=config, model_inference_backend=backend)
        if restored:
            wire = policy.to_bytes()
            policy = load_compiled_policy(
                wire, runtime_config=config, model_inference_backend=backend
            )
            assert policy.to_bytes() == wire
        result = policy.scan(
            {"messages": [{"role": "user", "content": "owned text"}]}, explain=explain
        )
        assert result.is_complete and result.rule_matches == ["Read"]
        assert adapter.score_calls == 1
        results.append((policy.policy_digest, policy._model_hashes, adapter.score_messages))
    assert results[0] == results[1]
