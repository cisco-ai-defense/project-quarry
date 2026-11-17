# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Transforms preserve opaque holders and defer their faults until an actual read."""

from __future__ import annotations

import pytest

from quarry import InvalidInputError, compile_policy, load_compiled_policy
from quarry.engine.matchers import field_access
from quarry.integrations.adapter_message_transforms import apply_adapter_transform
from quarry.integrations.content_transforms import apply_content_transform
from quarry.types.messages import Messages
from quarry.types.transforms import ContentTransform, TransformOperation
from tests.unit.test_adapter_transform_dsl import configured


@pytest.mark.parametrize(
    "role,extra",
    [
        ("alien", {}),
        ("user", {"refusal": "no"}),
        ("assistant", {"tool_call_id": "claimed"}),
        ("tool", {"refusal": "no", "tool_call_id": None}),
    ],
)
@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("adapter", [False, True])
@pytest.mark.asyncio
async def test_transform_keeps_unreadable_holders_opaque_and_transforms_other_items(
    tmp_path, role, extra, first, adapter
):
    opaque = {"role": role, "content": "opaque", **extra}
    good = {"role": "user", "content": "Visible"}
    raw = [opaque, good] if first else [good, opaque]
    source = Messages.from_openai(raw)
    if adapter:
        _, backend, provider = configured(tmp_path)
        changed = await apply_adapter_transform(source, backend, "convert")
        assert provider.parts == [{"type": "text", "text": "Visible"}]
        expected = "VISIBLE"
    else:
        changed = await apply_content_transform(
            source, ContentTransform((TransformOperation("casefold"),))
        )
        expected = "visible"
    good_index, opaque_index = (1, 0) if first else (0, 1)
    assert changed[good_index].content == expected
    assert changed[opaque_index].to_openai_dict() == opaque
    assert changed[opaque_index].message_index == opaque_index
    with pytest.raises(InvalidInputError):
        field_access(changed[opaque_index], "content")
    assert source.to_openai_messages() == raw


@pytest.mark.parametrize("prepared", [False, True])
def test_indexed_dsl_read_does_not_fault_on_a_later_unreadable_holder(tmp_path, prepared):
    config, backend, provider = configured(tmp_path)
    policy = compile_policy(
        'rule "R" condition: messages.transform("convert")[0].content contains "BLOCK"',
        runtime_config=config,
        model_inference_backend=backend,
    )
    if prepared:
        policy = load_compiled_policy(
            policy.to_bytes(), runtime_config=config, model_inference_backend=backend
        )
    result = policy.scan(
        {
            "messages": [
                {"role": "user", "content": "block"},
                {"role": "alien", "content": "opaque"},
            ]
        },
        explain=True,
    )
    assert result.is_complete and result.rule_matches == ["R"]
    assert provider.parts == [{"type": "text", "text": "block"}]


@pytest.mark.asyncio
async def test_opaque_media_is_preserved_without_an_adapter_read(tmp_path):
    opaque = {"role": "alien", "content": [{"type": "image_url", "image_url": {"url": "data:bad"}}]}
    source = Messages.from_openai([opaque, {"role": "user", "content": "visible"}])
    _, backend, provider = configured(tmp_path)
    changed = await apply_adapter_transform(source, backend, "convert")
    assert changed[0].to_openai_dict() == opaque and changed[1].content == "VISIBLE"
    assert provider.parts == [{"type": "text", "text": "visible"}]


@pytest.mark.asyncio
async def test_an_all_opaque_selection_calls_no_adapter(tmp_path):
    source = Messages.from_openai([{"role": "alien", "content": "opaque"}])
    _, backend, provider = configured(tmp_path)
    changed = await apply_adapter_transform(source, backend, "convert")
    assert changed is source and provider.parts == []


@pytest.mark.parametrize("adapter", [False, True])
@pytest.mark.asyncio
async def test_result_text_transform_relinks_an_opaque_call_holder_without_reading_it(
    tmp_path, adapter
):
    raw = [
        {
            "role": "user",
            "content": "opaque",
            "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "content": "Visible", "tool_call_id": "c"},
    ]
    source = Messages.from_openai(raw)
    if adapter:
        _, backend, provider = configured(tmp_path)
        changed = await apply_adapter_transform(source, backend, "convert")
        assert provider.parts == [{"type": "text", "text": "Visible"}]
    else:
        changed = await apply_content_transform(
            source, ContentTransform((TransformOperation("casefold"),))
        )
    assert changed[1].content == ("VISIBLE" if adapter else "visible")
    assert changed[1].call is changed.tool_calls[0]
    assert changed.tool_calls[0]._response is changed[1]
    assert changed[0].to_openai_dict() == raw[0]
    with pytest.raises(InvalidInputError):
        field_access(changed[0], "content")
    with pytest.raises(InvalidInputError):
        field_access(changed[1], "name")
    assert source.to_openai_messages() == raw and source.tool_calls[0]._response is source[1]
