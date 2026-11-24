# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Span quantifiers stream pattern matches and honor collection and time limits."""

from types import SimpleNamespace

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.engine.atoms import ModelExpression as E
from quarry.integrations import model_expression
from quarry.safety import timer
from quarry.safety.limits import SafetyLimits
from quarry.types.messages import Messages


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize(
    "source",
    [
        "$a.spans(messages)",
        "[s for s in $a.spans(messages) if true]",
        "outermost($a.spans(messages))",
        "union($a.spans(messages), $b.spans(messages))",
    ],
)
@pytest.mark.parametrize(
    ("quantifier", "comparison", "matched", "exhausted"),
    [
        ("any", '== "a"', True, False),
        ("all", '!= "a"', False, False),
        ("all", '== "a"', True, True),
        ("any", '!= "a"', False, True),
    ],
)
def test_quantifiers_stream_past_the_collection_limit(
    monkeypatch, restored, source, quantifier, comparison, matched, exhausted
):
    config = QuarryConfig(
        config_version=2, safety_limits=SafetyLimits(rule_timeout=None, max_comprehension_size=2)
    )
    rule = (
        'rule "Read" strings: $a="a" $b="a" regex condition: '
        f"{quantifier} s in {source} where messages.extract(s) {comparison}"
    )
    policy = compile_policy(rule, runtime_config=config)
    if restored:
        policy = load_compiled_policy(policy.to_bytes(), runtime_config=config)

    read_last_message = False
    match = model_expression.matching_patterns

    def recording_matches(text, *args, **kwargs):
        nonlocal read_last_message
        read_last_message |= text == "a a a a last"
        return match(text, *args, **kwargs)

    monkeypatch.setattr(model_expression, "matching_patterns", recording_matches)
    result = policy.scan(
        {"messages": [{"role": "user", "content": text} for text in ("a a a a", "a a a a last")]}
    )

    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == (["Read"] if matched else [])
    assert read_last_message is exhausted


def test_materializing_the_same_pattern_spans_exceeds_the_collection_limit():
    config = QuarryConfig(
        config_version=2, safety_limits=SafetyLimits(rule_timeout=None, max_comprehension_size=2)
    )
    policy = compile_policy(
        'rule "Read" strings: $a="a" condition: len($a.spans(messages)) == 3',
        runtime_config=config,
    )
    result = policy.scan({"messages": [{"role": "user", "content": "aaa"}]})
    assert result.completion_state == "incomplete"
    assert "max_comprehension_size=2" in result.evaluation_errors["Read"]


@pytest.mark.parametrize("kind", ["span_any", "span_all"])
@pytest.mark.asyncio
async def test_pattern_stream_checks_deadline_after_matching(monkeypatch, kind):
    messages = Messages.from_openai([{"role": "user", "content": "aaaa"}])
    context = SimpleNamespace(conversation=SimpleNamespace(messages=messages))
    matches = E(
        "string_spans", "$a", (E("messages"), E("string_patterns", "0000", (E("literal", "a"),)))
    )
    expression = E(kind, "s", (matches, E("literal", kind == "span_any")))
    now = 0.0

    def expire_during_match(*args, **kwargs):
        nonlocal now
        now = 2.0
        yield (0, 1), (0,)

    monkeypatch.setattr(timer, "time", SimpleNamespace(monotonic=lambda: now))
    monkeypatch.setattr(model_expression, "matching_patterns", expire_during_match)
    with timer.evaluation_deadline(1):
        with pytest.raises(timer.QuarryTimeoutError):
            await model_expression.evaluate_model_expression(expression, context)
        now = 0.0
