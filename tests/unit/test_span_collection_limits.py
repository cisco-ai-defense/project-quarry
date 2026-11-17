# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Span operations respect the configured collection bound without truncating results."""

from dataclasses import replace

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.safety import timer
from quarry.safety.limits import SafetyLimits
from quarry.safety.timer import QuarryTimeoutError, evaluation_deadline
from quarry.types.messages import Messages
from quarry.types.results import CollectionResult, ProducerRef, ResultItem, Score, Span
from tests.unit.model_fakes import ModelAdapter, NoWaitLimiter, model_entry, passthrough_builder


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("enforce", [False, True])
@pytest.mark.parametrize(
    "operation", ["union-distinct", "union-duplicate", "outermost", "quantifier", "filter"]
)
def test_span_operators_apply_limits_before_iterating_or_returning_large_lists(
    tmp_path, restored, enforce, operation
):
    class ThreeSpans(ModelAdapter):
        async def decode_spans(self, config, raw_result, messages, threshold):
            return [Span(0, i, i + 1, Score(0.9)) for i in range(3)]

    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"spans": model_entry(spans=0, category="owned")}},
        tmp_path,
        require_rules=False,
    )
    config = replace(
        config,
        safety_limits=SafetyLimits(
            rule_timeout=None, max_comprehension_size=2, enforce_limits=enforce
        ),
    )
    backend = ModelInferenceBackend(
        config,
        adapters={"fake": ThreeSpans()},
        message_builders={"test": passthrough_builder},
        rate_limiter=NoWaitLimiter(),
    )
    conditions = {
        "union-distinct": "len(union($a.spans(messages), $b.spans(messages))) == 4",
        "union-duplicate": "len(union($a.spans(messages), $a.spans(messages))) == 2",
        "outermost": 'len(outermost(model("spans").spans(messages, 0.5))) == 3',
        "quantifier": 'any s in model("spans").spans(messages, 0.5) where true',
        "filter": 'len([s for s in model("spans").spans(messages, 0.5) if true]) == 3',
    }
    source = 'rule "Read" strings: $a="a" $b="b" condition: ' + conditions[operation]
    policy = compile_policy(source, runtime_config=config, model_inference_backend=backend)
    if restored:
        raw = policy.to_bytes()
        policy = load_compiled_policy(raw, runtime_config=config, model_inference_backend=backend)
        assert policy.to_bytes() == raw
    content = "aabb" if operation.startswith("union") else "abcdef"
    result = policy.scan({"messages": [{"role": "user", "content": content}]})
    if enforce and operation != "union-duplicate":
        assert result.completion_state == "incomplete"
        assert not result.rule_matches
        assert "max_comprehension_size=2" in result.evaluation_errors["Read"]
    else:
        assert result.is_complete, result.evaluation_errors
        assert result.rule_matches == ["Read"]


@pytest.mark.parametrize("operation", ["union", "outermost"])
def test_span_list_synchronous_work_checks_the_active_deadline(monkeypatch, operation):
    from quarry.integrations.model_expression import _outermost_spans, _union_spans

    messages = Messages.from_openai([{"role": "user", "content": "abc"}])
    spans = CollectionResult(
        "span_collection",
        "owned",
        ProducerRef("model", "owned", "synthetic"),
        tuple(
            ResultItem(str(i), messages._bind_span(Span(0, i, i + 1, Score(0.9)))) for i in range(2)
        ),
    )
    with evaluation_deadline(1) as deadline, monkeypatch.context() as clock:
        clock.setattr(timer.time, "monotonic", lambda: deadline.expires_at + 1)
        with pytest.raises(QuarryTimeoutError):
            _union_spans([spans, spans]) if operation == "union" else _outermost_spans(spans.items)
