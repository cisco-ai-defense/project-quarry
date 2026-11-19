# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Native local collections preserve paid work, lexical scope and aggregate meaning."""

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.types.exceptions import MODEL_REFUSED, LLMError, RuleValidationError
from tests.unit.model_fakes import ModelAdapter, NoWaitLimiter, model_entry, passthrough_builder
from tests.unit.test_message_local_tool_conditions import DATA

SCORE = 'model("reader").score(message("user", c.name), "synthetic") > 0.5'


def local_policy(tmp_path, condition, *, restored=False, max_calls=10, adapter=None):
    adapter = adapter or ModelAdapter()
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"reader": model_entry(score=True, classify=0)},
            "max_model_calls": max_calls,
        },
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(
        config,
        adapters={"fake": adapter},
        message_builders={"test": passthrough_builder},
        rate_limiter=NoWaitLimiter(),
    )
    policy = compile_policy(
        'rule "Read" condition: ' + condition,
        runtime_config=config,
        model_inference_backend=backend,
    )
    if restored:
        raw = policy.to_bytes()
        policy = load_compiled_policy(raw, runtime_config=config, model_inference_backend=backend)
        assert policy.to_bytes() == raw
    return policy, adapter


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
@pytest.mark.parametrize(
    "body,matched,names",
    [
        (
            f'count([c for c in m.tool_calls if c.name == "lookup" and {SCORE}]) == 1',
            True,
            ["lookup"],
        ),
        (f"len([c for c in m.tool_calls if {SCORE}]) == 2", True, ["lookup", "other"]),
        (f'any c in [c for c in m.tool_calls if {SCORE}] where c.name == "absent"', False, []),
        (
            'any c in [x for x in m.tool_calls if model("reader").score(message("user", x.name), "synthetic") > 0.5] where c.name == "lookup"',
            True,
            ["lookup"],
        ),
        (f"all c in [c for c in m.tool_calls if {SCORE}] where true", True, []),
        (
            f'all c in [c for c in m.tool_calls if {SCORE}] where c.name == "lookup"',
            False,
            ["other"],
        ),
        (
            f'all c in [c for c in m.tool_calls if c.name == "lookup"] where {SCORE}',
            True,
            ["lookup"],
        ),
        (f'any c in m.tool_calls where c.name == "lookup" and {SCORE}', True, ["lookup"]),
    ],
)
def test_local_paid_selection_work_and_boolean_evidence(
    tmp_path, restored, explain, body, matched, names
):
    policy, adapter = local_policy(
        tmp_path,
        "any m in assistant_messages where " + body,
        restored=restored,
        max_calls=max(1, len(names)),
    )
    result = policy.scan(DATA, explain=explain)
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == (["Read"] if matched else [])
    assert adapter.score_messages == [[{"role": "user", "content": name}] for name in names]
    if explain and matched:
        evidence = [e for detail in result.detailed_matches for e in detail.model_results]
        assert evidence
        for item in evidence:
            assert item.expression_value is True
            assert item.operator is item.threshold is item.input_span is None
            assert item.operation != "score"


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [True, "support"])
def test_indexed_count_has_numeric_evidence_without_child_judge_span(tmp_path, restored, explain):
    condition = f'count([c for c in assistant_messages[-1].tool_calls if c.name == "lookup" and {SCORE}]) == 1'
    policy, adapter = local_policy(tmp_path, condition, restored=restored, max_calls=1)
    result = policy.scan(DATA, explain=explain)
    assert result.is_complete and result.rule_matches == ["Read"]
    assert adapter.score_calls == 1
    (evidence,) = result.detailed_matches[0].model_results
    assert evidence.operation == "numeric_expression"
    assert evidence.expression_value == 1 and type(evidence.expression_value) is int
    assert evidence.operator == "=" and evidence.threshold == 1
    assert evidence.input_span is None


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize(
    "body",
    [
        f"all c in tool_calls where {SCORE}",
        f"any m in assistant_messages where all c in m.tool_calls where {SCORE}",
    ],
)
def test_unfiltered_paid_universals_keep_existing_admission(tmp_path, restored, body):
    with pytest.raises(RuleValidationError, match="all comprehension"):
        local_policy(tmp_path, body, restored=restored)


@pytest.mark.parametrize("restored", [False, True])
def test_filter_and_quantifier_binders_do_not_capture_an_enclosing_name(tmp_path, restored):
    condition = (
        "any c in assistant_messages where "
        'any selected in [c for c in c.tool_calls if c.name == "lookup"] '
        'where c.content == "plan" and selected.arguments.limit == 2'
    )
    policy, adapter = local_policy(tmp_path, condition, restored=restored)
    result = policy.scan(DATA, explain=True)
    assert result.is_complete and result.rule_matches == ["Read"]
    assert adapter.score_calls == 0


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [True, "support"])
def test_mixed_score_and_classifier_counts_keep_one_boolean_outcome(tmp_path, restored, explain):
    body = (
        f'count([c for c in m.tool_calls if c.name == "lookup" and {SCORE} '
        'and model("reader").classify(message("user", c.name)) > 0.5]) == 1'
    )
    policy, adapter = local_policy(
        tmp_path, "any m in assistant_messages where " + body, restored=restored, max_calls=2
    )
    result = policy.scan(DATA, explain=explain)
    assert result.is_complete and result.rule_matches == ["Read"], result.evaluation_errors
    assert (
        adapter.score_messages
        == adapter.classification_messages
        == [[{"role": "user", "content": "lookup"}]]
    )
    (evidence,) = result.detailed_matches[0].model_results
    assert evidence.expression_value is True
    assert evidence.operator is evidence.threshold is evidence.input_span is None
    assert evidence._call_count == 2


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [True, "support"])
def test_refusal_fallback_counts_keep_refusals_and_boolean_attribution(tmp_path, restored, explain):
    class Refusing(ModelAdapter):
        async def score(self, config, messages, *, timeout=None):
            self.score_messages.append(messages)
            raise LLMError("Synthetic refusal", {"code": MODEL_REFUSED, "reason": "synthetic"})

    body = (
        'count([c for c in m.tool_calls if c.name == "lookup" and '
        'if_refused(model("reader").score(message("user", c.name), "synthetic"), 1) > 0.5]) == 1'
    )
    policy, adapter = local_policy(
        tmp_path,
        "any m in assistant_messages where " + body,
        restored=restored,
        max_calls=1,
        adapter=Refusing(),
    )
    result = policy.scan(DATA, explain=explain)
    assert result.is_complete and result.rule_matches == ["Read"], result.evaluation_errors
    assert adapter.score_calls == 1
    (evidence,) = result.detailed_matches[0].model_results
    assert evidence.expression_value is True
    assert evidence.operator is evidence.threshold is evidence.input_span is None
    assert evidence.refusals == (("reader", "score", "refusal_field"),)


@pytest.mark.parametrize("restored", [False, True])
def test_local_paid_counts_obey_the_physical_request_allowance(tmp_path, restored):
    policy, adapter = local_policy(
        tmp_path,
        f"any m in assistant_messages where count([c for c in m.tool_calls if {SCORE}]) == 2",
        restored=restored,
        max_calls=1,
    )
    result = policy.scan(DATA, explain=True)
    assert not result.is_complete and result.rule_matches == []
    assert adapter.score_calls == 1
    assert result.failures[0].category == "model"
    assert "max_model_calls=1" in result.evaluation_errors["Read"]
