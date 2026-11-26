# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Constant folding must preserve source admission through prepared-policy reload."""

import orjson
import pytest

from quarry import DSLScanner, compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.integrations.resources import ResourcePreparationError
from tests.unit.model_fakes import ModelAdapter, NoWaitLimiter, model_entry, passthrough_builder
from tests.unit.test_nested_rule_runtime_contracts import _policy


def _runtime(tmp_path, *, physical_model="synthetic-judge"):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"judge": model_entry(classify=0, score=True, physical_model=physical_model)},
        },
        tmp_path,
        require_rules=False,
    )
    adapter = ModelAdapter()
    backend = ModelInferenceBackend(
        config,
        adapters={"fake": adapter},
        message_builders={"test": passthrough_builder},
        rate_limiter=NoWaitLimiter(),
    )
    return config, backend, adapter


def _forbidden(*_args, **_kwargs):
    raise AssertionError("prepared-policy loading must not parse or compile source")


@pytest.mark.parametrize("operation", ["classify", "score"])
@pytest.mark.parametrize("gate,matched", [("true or", True), ("false and", False)])
@pytest.mark.parametrize("nested", [False, True])
def test_folded_models_retain_declarations_without_dispatch(
    tmp_path, monkeypatch, operation, gate, matched, nested
):
    config, backend, adapter = _runtime(tmp_path)
    source = f'rule "Read" condition: {gate} model("judge").{operation}(messages) > 0.5'
    if nested:
        source += '\nrule "Caller" imports: read = "Read" condition: read(messages)'
        policy = _policy(source, config, False, backend)
        name = "Caller"
    else:
        policy = compile_policy(source, runtime_config=config, model_inference_backend=backend)
        name = "Read"
    wire = policy.to_bytes()
    document = orjson.loads(wire)
    assert set(document["models"]) == {"judge"}
    declared = document["rule"]["artifact"]["declared_resources"]
    assert declared["llm_models"] == ["judge"]
    assert ["judge", operation] in declared["model_operations"]
    monkeypatch.setattr(DSLScanner, "_parse_rule_text", _forbidden)
    monkeypatch.setattr(DSLScanner, "_compile_candidate_rules", _forbidden)
    restored = load_compiled_policy(wire, runtime_config=config, model_inference_backend=backend)
    assert restored.to_bytes() == wire
    for current in (policy, restored):
        result = current.scan({"messages": [{"role": "user", "content": "synthetic"}]})
        assert result.is_complete, result.evaluation_errors
        assert result.rule_matches == ([name] if matched else [])
    assert adapter.classifications == adapter.score_calls == 0


@pytest.mark.parametrize("gate", ["true or", "false and"])
@pytest.mark.parametrize("changed", [False, True])
def test_folded_model_configuration_is_still_required(tmp_path, gate, changed):
    config, backend, adapter = _runtime(tmp_path)
    source = f'rule "Read" condition: {gate} model("judge").classify(messages) > 0.5'
    wire = compile_policy(source, runtime_config=config, model_inference_backend=backend).to_bytes()
    if changed:
        other_config, other_backend, other_adapter = _runtime(
            tmp_path, physical_model="synthetic-other-judge"
        )
        message = "runtime model configuration differs"
    else:
        other_config, other_backend = QuarryConfig(config_version=2), None
        other_adapter = None
        message = "runtime configuration is missing models: judge"
    with pytest.raises(ValueError, match=message):
        load_compiled_policy(
            wire, runtime_config=other_config, model_inference_backend=other_backend
        )
    assert adapter.classifications == adapter.score_calls == 0
    if other_adapter is not None:
        assert other_adapter.classifications == other_adapter.score_calls == 0


@pytest.mark.parametrize("gate", ["true or", "false and"])
def test_source_still_rejects_missing_folded_model(gate):
    source = f'rule "Read" condition: {gate} model("judge").classify(messages) > 0.5'
    with pytest.raises(ResourcePreparationError, match="judge"):
        compile_policy(source, runtime_config=QuarryConfig(config_version=2))


@pytest.mark.parametrize(
    "declarations,message",
    [
        (None, "declared_resources must be an object"),
        ({"unknown": []}, "unknown resource kind"),
        ({"llm_models": "judge"}, "must be an array"),
        ({"llm_models": [1]}, "must be a string"),
        ({"model_operations": ["judge"]}, "must be an array"),
        ({"model_operations": [["judge"]]}, "requires two values"),
        ({"model_operations": [["judge", "score", "extra"]]}, "requires two values"),
        ({"model_operations": [[0, "classify"]]}, "must be a string"),
        ({"model_operations": [["judge", None]]}, "must be a string"),
        ({"model_operations": [["judge", "unknown"]]}, "operation is unsupported"),
        ({"model_operations": [["judge", "token_windows"]]}, "operation is unsupported"),
        ({"transforms": [[1, "reference"]]}, "must be a Boolean"),
        ({"transforms": [[False, None]]}, "must be a string"),
        ({"transforms": [[False, "invalid"]]}, "invalid adapter transform reference"),
        (
            {"transforms": [[False, "local\x1fjudge\x1f" + "a" * 64]]},
            "local transform reference cannot name a model",
        ),
        ({"functions": [{}]}, "must be a string"),
    ],
)
def test_declared_resource_metadata_has_a_closed_schema(
    tmp_path, monkeypatch, declarations, message
):
    config, backend, adapter = _runtime(tmp_path)
    policy = compile_policy('rule "Read" condition: true', runtime_config=config)
    document = orjson.loads(policy.to_bytes())
    document["rule"]["artifact"]["declared_resources"] = declarations
    monkeypatch.setattr(DSLScanner, "_parse_rule_text", _forbidden)
    monkeypatch.setattr(DSLScanner, "_compile_candidate_rules", _forbidden)
    with pytest.raises(ValueError, match=message):
        load_compiled_policy(
            orjson.dumps(document), runtime_config=config, model_inference_backend=backend
        )
    assert adapter.classifications == adapter.score_calls == 0


def test_removing_folded_model_declarations_cannot_leave_its_fingerprint(tmp_path):
    config, backend, adapter = _runtime(tmp_path)
    policy = compile_policy(
        'rule "Read" condition: true or model("judge").classify(messages) > 0.5',
        runtime_config=config,
        model_inference_backend=backend,
    )
    document = orjson.loads(policy.to_bytes())
    document["rule"]["artifact"]["declared_resources"] = {}
    with pytest.raises(ValueError, match="model fingerprints do not match the compiled artifact"):
        load_compiled_policy(
            orjson.dumps(document), runtime_config=config, model_inference_backend=backend
        )
    assert adapter.classifications == 0
