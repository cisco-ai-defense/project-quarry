# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Provider-input and provenance regressions reported on the transform foundation."""

from dataclasses import replace

import pytest

from quarry import compile_policy
from quarry.config import QuarryConfig
from quarry.integrations.content_transforms import apply_content_transform
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.safety.limits import SafetyLimits
from quarry.types.messages import Messages
from quarry.types.results import Score, Span
from quarry.types.transforms import ContentTransform, TransformOperation
from tests.unit.model_fakes import ModelAdapter, model_entry


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (
            'messages.transform("redact")[0] + message("assistant", "review")',
            [{"role": "user", "content": "redacted"}, {"role": "assistant", "content": "review"}],
        ),
        (
            'messages.transform("redact")[-1] + message("assistant", "review")',
            [{"role": "user", "content": "redacted"}, {"role": "assistant", "content": "review"}],
        ),
        (
            'messages[:1].transform("redact")[0] + message("assistant", "review")',
            [{"role": "user", "content": "redacted"}, {"role": "assistant", "content": "review"}],
        ),
        (
            '(message("user", "constructed") + message("assistant", "prior"))[0] + message("assistant", "review")',
            [
                {"role": "user", "content": "constructed"},
                {"role": "assistant", "content": "review"},
            ],
        ),
        (
            '(message("user", "constructed") + message("assistant", "prior"))[-1] + message("assistant", "review")',
            [{"role": "assistant", "content": "prior"}, {"role": "assistant", "content": "review"}],
        ),
        (
            'message("assistant", "review") + messages.transform("redact")[0]',
            [{"role": "assistant", "content": "review"}, {"role": "user", "content": "redacted"}],
        ),
        (
            'messages.transform("redact")[0] if true else messages[99:]',
            [{"role": "user", "content": "redacted"}],
        ),
        (
            'message("user", messages[0].content) + message("assistant", "review")',
            [{"role": "user", "content": "SECRET"}, {"role": "assistant", "content": "review"}],
        ),
        (
            'message("assistant", "review") + message("user", messages[0].content)',
            [{"role": "assistant", "content": "review"}, {"role": "user", "content": "SECRET"}],
        ),
        (
            'message("user", messages.transform("redact")[0].content) + message("assistant", "review")',
            [{"role": "user", "content": "redacted"}, {"role": "assistant", "content": "review"}],
        ),
    ],
)
def test_selected_variants_and_copied_constructors_reach_provider_unchanged(
    tmp_path, expression, expected
):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"judge": model_entry(classify=0)},
            "transforms": {
                "redact": {"operations": [{"op": "replace", "old": "SECRET", "new": "redacted"}]}
            },
        },
        tmp_path,
        require_rules=False,
    )
    config = replace(config, safety_limits=SafetyLimits(rule_timeout=None))
    adapter = ModelAdapter()
    policy = compile_policy(
        f'rule "R" condition: model("judge").classify({expression}) > 0.5',
        runtime_config=config,
        model_inference_backend=ModelInferenceBackend(config, adapters={"fake": adapter}),
    )
    result = policy.scan({"messages": [{"role": "user", "content": "SECRET"}]})
    assert result.is_complete and result.rule_matches == ["R"]
    assert adapter.classification_messages == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operations",
    [
        (TransformOperation("replace", "ab", ""), TransformOperation("prefix", "ab")),
        (TransformOperation("replace", "ab", ""), TransformOperation("suffix", "ab")),
    ],
)
async def test_identical_output_does_not_restore_deleted_source_provenance(operations):
    source = Messages.from_openai([{"role": "user", "content": "ab"}])
    transformed = await apply_content_transform(source, ContentTransform(operations))
    assert transformed.to_openai_messages() == source.to_openai_messages()
    with pytest.raises(ValueError, match="inserted text"):
        transformed.source_ranges(0, 0, 2)
    with pytest.raises(ValueError, match="inserted text"):
        transformed._bind_span(Span(0, 0, 2, Score(0.9)))
    assert source.source_ranges(0, 0, 2)[0].to_dict() == {"message_index": 0, "start": 0, "end": 2}


@pytest.mark.asyncio
async def test_identical_text_retains_new_whole_match_mapping():
    source = Messages.from_openai([{"role": "user", "content": "ab"}])
    transformed = await apply_content_transform(
        source, ContentTransform((TransformOperation("replace", "ab", "ab"),))
    )
    assert transformed[0].content == "ab"
    assert transformed.source_ranges(0, 0, 1)[0].to_dict() == {
        "message_index": 0,
        "start": 0,
        "end": 2,
    }
    assert source.source_ranges(0, 0, 1)[0].to_dict()["end"] == 1


@pytest.mark.asyncio
async def test_identical_text_preserves_partial_inserted_and_copied_ranges():
    source = Messages.from_openai([{"role": "user", "content": "ab"}])
    transformed = await apply_content_transform(
        source,
        ContentTransform(
            (TransformOperation("replace", "a", ""), TransformOperation("prefix", "a"))
        ),
    )
    assert transformed[0].content == "ab"
    with pytest.raises(ValueError, match="inserted text"):
        transformed.source_ranges(0, 0, 1)
    assert transformed.source_ranges(0, 1, 2)[0].to_dict() == {
        "message_index": 0,
        "start": 1,
        "end": 2,
    }


@pytest.mark.asyncio
async def test_promotion_refuses_a_foreign_bound_message_instead_of_reading_current_turn():
    from types import SimpleNamespace

    from quarry.engine.atoms import ModelExpression
    from quarry.integrations.model_expression import evaluate_model_expression

    current = Messages.from_openai([{"role": "user", "content": "current"}])
    foreign = Messages.from_openai([{"role": "user", "content": "foreign"}])
    expression = ModelExpression("as_messages", None, (ModelExpression("binding", "m"),))
    context = SimpleNamespace(conversation=SimpleNamespace(messages=current))
    with pytest.raises(ValueError, match="owning selection"):
        await evaluate_model_expression(expression, context, bindings={"m": foreign[0]})
