# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Nested checks keep configured transforms, closed expression guards and cost metadata."""

from copy import deepcopy

import pytest

from quarry.config import QuarryConfig
from quarry.engine import atoms
from quarry.engine.atoms import Cost
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.policy_serialization import _artifact_from_wire, _artifact_to_wire
from quarry.types.exceptions import RuleValidationError
from tests.unit.model_fakes import ModelAdapter, NoWaitLimiter, model_entry, passthrough_builder
from tests.unit.test_nested_rule_runtime_contracts import _policy, _wrappers


def config(tmp_path, models=None):
    return QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "rule_timeout": None,
            "transforms": {"strip": {"operations": [{"op": "strip"}]}},
            "models": models or {},
        },
        tmp_path,
        require_rules=False,
    )


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
def test_called_body_retains_its_configured_transform(tmp_path, depth, restored, explain):
    source, name = _wrappers(
        'rule "Leaf" strings: $p="^needle$" regex condition: len($p.spans(messages.transform("strip"))) == 1',
        depth=depth,
    )
    result = _policy(source, config(tmp_path), restored).scan(
        {"messages": [{"role": "user", "content": " needle "}]}, explain=explain
    )
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == [name]


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
def test_child_frame_memos_do_not_reuse_a_different_selected_conversation(
    tmp_path, restored, explain
):
    source = (
        'rule "Leaf" strings: $p="^needle$" regex condition: len($p.spans(messages.transform("strip"))) == 1\n'
        'rule "Caller" imports: leaf="Leaf" condition: leaf(messages[:1]) and not leaf(messages[1:])'
    )
    result = _policy(source, config(tmp_path), restored).scan(
        {
            "messages": [
                {"role": "user", "content": " needle "},
                {"role": "user", "content": " other "},
            ]
        },
        explain=explain,
    )
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Caller"]


def test_a_local_transform_argument_keeps_pure_cost(tmp_path):
    source = 'rule "Leaf" condition: len(messages)>0\nrule "Caller" imports: leaf="Leaf" condition: leaf(messages.transform("strip"))'
    policy = _policy(source, config(tmp_path), True)
    artifact = policy._scanner.compiled_rules[policy._rule_name]
    assert {artifact.atom_specs[key].cost for key in artifact.split_checks} == {Cost.PURE}


def test_call_arguments_use_existing_constructed_message_graph_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(atoms, "MAX_CONSTRUCTED_MESSAGES", 1)
    source = (
        'rule "Leaf" condition: true\nrule "Caller" imports: leaf="Leaf" condition: '
        'leaf(message("user","one").add_message("assistant","two"))'
    )
    with pytest.raises(RuleValidationError, match="constructed messages"):
        _policy(source, config(tmp_path), False)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("cost", [Cost.PURE, Cost.MODEL])
def test_prepared_called_models_cannot_hide_their_external_cost(tmp_path, depth, cost):
    cfg = config(tmp_path, {"judge": model_entry(score=True)})
    backend = ModelInferenceBackend(
        cfg,
        adapters={"fake": ModelAdapter()},
        message_builders={"test": passthrough_builder},
        rate_limiter=NoWaitLimiter(),
    )
    source, _ = _wrappers(
        'rule "Leaf" condition: model("judge").score(messages,"criterion") > 0.5', depth=depth
    )
    policy = _policy(source, cfg, False, backend)
    raw = _artifact_to_wire(policy._scanner.compiled_rules[policy._rule_name])
    split = raw["split_checks"][0][0]
    tampered = deepcopy(raw)
    next(row for row in tampered["atom_specs"] if row["atom_id"] == split)["cost"] = cost.value
    with pytest.raises(ValueError, match="cost"):
        _artifact_from_wire(tampered)


@pytest.mark.parametrize("field", ["model_expression", "predicate", "detector_arg"])
def test_prepared_called_checks_refuse_invalid_dispatch_shapes(tmp_path, field):
    source = (
        'rule "Leaf" condition: true\nrule "Caller" imports: leaf="Leaf" condition: leaf(messages)'
    )
    policy = _policy(source, config(tmp_path), False)
    raw = _artifact_to_wire(policy._scanner.compiled_rules[policy._rule_name])
    split = raw["split_checks"][0][0]
    next(row for row in raw["atom_specs"] if row["atom_id"] == split)[field] = (
        None if field == "model_expression" else "invalid"
    )
    with pytest.raises(ValueError, match="split|called"):
        _artifact_from_wire(raw)
