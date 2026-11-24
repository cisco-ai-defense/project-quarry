# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Enabled collection limits apply before a span quantifier tests its predicate."""

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.integrations import model_expression
from quarry.safety.limits import SafetyLimits


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("enforce", [False, True])
@pytest.mark.parametrize("quantifier", ["any", "all"])
@pytest.mark.parametrize("predicate", [False, True])
@pytest.mark.parametrize(
    ("source", "content", "oversized"),
    [
        pytest.param("$a.spans(messages)", "aaa", True, id="direct-over-limit"),
        pytest.param(
            "[kept for kept in $a.spans(messages) if true]",
            "aaa",
            True,
            id="filtered-over-limit",
        ),
        pytest.param("outermost($a.spans(messages))", "aaa", True, id="outermost-over-limit"),
        pytest.param(
            "union($a.spans(messages), $b.spans(messages))",
            "aabb",
            True,
            id="combined-over-limit",
        ),
        pytest.param("$a.spans(messages)", "aa", False, id="direct-at-limit"),
        pytest.param(
            "union($a.spans(messages), $a.spans(messages))",
            "aa",
            False,
            id="deduplicated-at-limit",
        ),
    ],
)
def test_span_quantifier_input_limits_precede_predicates(
    monkeypatch, restored, enforce, quantifier, predicate, source, content, oversized
):
    config = QuarryConfig(
        config_version=2,
        safety_limits=SafetyLimits(
            rule_timeout=None, max_comprehension_size=2, enforce_limits=enforce
        ),
    )
    policy = compile_policy(
        'rule "Read" strings: $a="a" $b="b" condition: '
        f"{quantifier} checked in {source} where {str(predicate).lower()}",
        runtime_config=config,
    )
    if restored:
        wire = policy.to_bytes()
        policy = load_compiled_policy(wire, runtime_config=config)
        assert policy.to_bytes() == wire

    predicate_calls = 0
    evaluate = model_expression.evaluate_model_expression

    async def record_predicate(expression, *args, **kwargs):
        nonlocal predicate_calls
        if expression.kind == "literal" and "checked" in (kwargs.get("bindings") or {}):
            predicate_calls += 1
        return await evaluate(expression, *args, **kwargs)

    monkeypatch.setattr(model_expression, "evaluate_model_expression", record_predicate)
    result = policy.scan({"messages": [{"role": "user", "content": content}]})

    if enforce and oversized:
        assert result.completion_state == "incomplete"
        assert not result.rule_matches
        assert "max_comprehension_size=2" in result.evaluation_errors["Read"]
        assert predicate_calls == 0
    else:
        assert result.is_complete, result.evaluation_errors
        assert result.rule_matches == (["Read"] if predicate else [])
        expected = 1 if (quantifier == "any") == predicate else (3 if oversized else 2)
        if source.startswith("union(") and oversized and expected > 1:
            expected = 4
        assert predicate_calls == expected
