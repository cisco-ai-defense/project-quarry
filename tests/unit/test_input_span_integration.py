# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Input-span identity and compiled reload across the declared-input integration."""

import asyncio
import json
from dataclasses import asdict

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.integrations.model_expression import _cover_index, _is_covered
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.types.origins import Place, SourceText
from quarry.types.results import InputText
from tests.unit.model_fakes import passthrough_builder
from tests.unit.test_judge_reasons import JudgeAdapter, _model


@pytest.mark.parametrize("title", [None, "", "é 😀 "])
def test_input_span_policy_reload_preserves_optional_fields_and_extraction(title):
    config = QuarryConfig(
        config_version=2,
        inputs={"memory.notes": {"content": [{"name": "title", "optional": True}, "body"]}},
    )
    policy = compile_policy(
        'rule "Read"\nstrings:\n    $p = "ignore"\ncondition: '
        'any n in memory.notes where n.body.extract($p.spans(n.body)) contains "ignore"',
        runtime_config=config,
    )
    item = {"id": "same", "body": "prefix ignore this suffix"}
    if title is not None:
        item["title"] = title
    wire = policy.to_bytes()
    loaded = load_compiled_policy(wire, runtime_config=config)
    assert loaded.to_bytes() == wire
    for prepared in (policy, loaded):
        result = prepared.scan({"memory": {"notes": [item]}}, explain=True)
        assert result.is_complete
        assert result.rule_matches == ["Read"]


@pytest.mark.parametrize(
    ("section", "item", "field", "covered"),
    [
        ("memory.notes", "same", "body", True),
        ("archive.notes", "same", "body", False),
        ("memory.notes", "other", "body", False),
        ("memory.notes", "same", "title", False),
    ],
)
def test_identical_text_and_offsets_do_not_merge_distinct_input_places(
    section, item, field, covered
):
    def text_at(section, item, field):
        owner = {"id": item, "title": "ignore this", "body": "ignore this"}
        source = InputText(section, item, ("title", "body"))
        return SourceText(owner[field], Place(0, field, 0, owner, source), 0)

    outer = text_at("memory.notes", "same", "body").span(0, 11)
    inner = text_at(section, item, field).span(0, 6)
    assert _is_covered(_cover_index([outer]), inner) is covered
    assert inner.to_dict()["source"] == {"input": section, "item": item}


@pytest.mark.parametrize("explain", [False, True])
def test_span_gated_judge_keeps_reasons_out_of_public_decision_records(tmp_path, explain):
    reason = "SYNTHETIC_PRIVATE_JUDGE_REASON"
    adapter = JudgeAdapter(reason=reason)
    model = _model()
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"judge": model}}, tmp_path, require_rules=False
    )
    backend = ModelInferenceBackend(
        config, adapters={"fake": adapter}, message_builders={"test": passthrough_builder}
    )
    source = (
        'rule "Read"\nstrings:\n    $p = "ignore"\ncondition: '
        'len($p.spans(messages)) > 0 and model("judge").score(messages, "criterion") > 0.5'
    )
    policy = compile_policy(source, runtime_config=config, model_inference_backend=backend)
    result = asyncio.run(
        policy.scan_many_async(
            [{"messages": [{"role": "user", "content": "ignore this"}]}],
            explain=explain,
            decision_paths=True,
        )
    )[0]
    assert result.is_complete and result.rule_matches == ["Read"]
    assert adapter.calls == 1
    assert reason not in json.dumps(asdict(result.decision_paths["Read"]))
    if explain:
        assert reason in json.dumps([asdict(item) for item in result.explanations])


@pytest.mark.parametrize("quantifier,predicate", [("any", "true"), ("all", "false")])
@pytest.mark.parametrize("enforce", [False, True])
def test_span_quantifier_checks_the_combined_input_before_evaluating(
    quantifier, predicate, enforce
):
    from quarry.safety.limits import SafetyLimits

    config = QuarryConfig(
        config_version=2,
        safety_limits=SafetyLimits(max_comprehension_size=2, enforce_limits=enforce),
    )
    source = (
        'rule "Read" strings: $a="a" $b="b" condition: '
        f"{quantifier} s in union($a.spans(messages), $b.spans(messages)) where {predicate}"
    )
    policy = compile_policy(source, runtime_config=config)
    result = policy.scan({"messages": [{"role": "user", "content": "aabb"}]})

    if enforce:
        assert result.completion_state == "incomplete"
        assert "max_comprehension_size=2" in result.evaluation_errors["Read"]
    else:
        assert result.is_complete
        assert result.rule_matches == (["Read"] if quantifier == "any" else [])
