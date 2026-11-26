# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Typed binders share compatible heads while preserving lexical source ownership."""

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.cache.async_cache import result_cache
from tests.unit.model_fakes import model_entry
from tests.unit.test_model_token_head_contracts import HeadAdapter, backend
from tests.unit.test_model_window_runtime import WindowAdapter


def setup(tmp_path, unit):
    adapter = HeadAdapter() if unit is None else WindowAdapter()
    models = {
        "classes": model_entry(classify=0),
        "segments": model_entry(spans=0, category="SYNTHETIC"),
    }
    if unit is not None:
        for model in models.values():
            model["windows"] = {"unit": unit, "size": 100}
    return backend(tmp_path, models, adapter), adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", [None, "characters", "tokens"])
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("mode", ["event", "mapped", "joined", "constructed"])
async def test_bound_class_and_span_inputs_share_the_planned_physical_head(
    tmp_path, unit, restored, mode
):
    service, adapter = setup(tmp_path, unit)
    if mode == "event":
        condition = 'any m in user_messages[:1] where model("classes").classify(message("user",m.content)) > 0.5'
        source = 'message("user",user_messages[0].content)'
    elif mode == "mapped":
        condition = (
            'model("classes").classify([message("user",m.content) for m in user_messages]) > 0.5'
        )
        source = '[message("user",n.content) for n in user_messages]'
    elif mode == "joined":
        condition = 'model("classes").classify(message("user",join([m.name for m in assistant_messages[-1].tool_calls],"\\n"))) > 0.5'
        source = 'message("user",join([n.name for n in assistant_messages[-1].tool_calls],"\\n"))'
    else:
        source = 'message("user",messages[-1].content)'
        condition = f'model("classes").classify({source}) > 0.5'
    compiled = compile_policy(
        'rule "R" condition: '
        + condition
        + '\ncollect: [s for s in model("segments").spans('
        + source
        + ", 0.5) if true]",
        runtime_config=service.config,
        model_inference_backend=service,
    )
    if restored:
        wire = compiled.to_bytes()
        compiled = load_compiled_policy(
            wire, runtime_config=service.config, model_inference_backend=service
        )
        assert compiled.to_bytes() == wire
    data = {
        "messages": [
            {"role": "system", "content": "preamble"},
            {"role": "user", "content": "abcdef"},
            {"role": "user", "content": "ghijkl"},
        ]
    }
    if mode == "joined":
        from tests.unit.test_message_local_tool_conditions import DATA

        data = DATA
    result_cache.clear()
    result = await compiled.scan_async(data)
    assert result.is_complete and result.rule_matches == ["R"], result.evaluation_errors
    flags = adapter.head_requests if unit is None else [entry[-1] for entry in adapter.requests]
    assert flags == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", [None, "characters", "tokens"])
@pytest.mark.parametrize("target,shared", [(-2, True), (-1, False)])
async def test_event_aliases_respect_negative_ordinals_and_distinct_selected_inputs(
    tmp_path, unit, target, shared
):
    service, adapter = setup(tmp_path, unit)
    source = f'rule "R" condition: any m in user_messages[:1] where model("classes").classify(message("user",m.content)) > 0.5\ncollect: [s for s in model("segments").spans(message("user",user_messages[{target}].content),0.5) if true]'
    compiled = compile_policy(
        source, runtime_config=service.config, model_inference_backend=service
    )
    result_cache.clear()
    result = await compiled.scan_async(
        {
            "messages": [
                {"role": "system", "content": "preamble"},
                {"role": "user", "content": "abcdef"},
                {"role": "user", "content": "ghijkl"},
            ]
        }
    )
    assert result.is_complete and result.rule_matches == ["R"], result.evaluation_errors
    flags = adapter.head_requests if unit is None else [entry[-1] for entry in adapter.requests]
    assert flags == ([True] if shared else [False, True])


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["characters", "tokens"])
async def test_authored_window_spans_keep_unmapped_and_constructed_coordinates(tmp_path, unit):
    service, adapter = setup(tmp_path, unit)
    source = 'message("user","authored")'
    compiled = compile_policy(
        f'rule "R" condition: model("classes").classify({source}) > 0.5\ncollect: [s for s in model("segments").spans({source},0.5) if true]',
        runtime_config=service.config,
        model_inference_backend=service,
    )
    result_cache.clear()
    result = await compiled.scan_async(
        {"messages": [{"role": "user", "content": "PRIVATE ROOT"}]}, explain=True
    )
    assert result.is_complete and result.rule_matches == ["R"], result.evaluation_errors
    span = result.outputs["R"].items[0].value
    assert span.source_unmapped and span.source_ranges == ()
    evidence = result.detailed_matches[0].model_results[0].score_summary["windows"]["deciding"]
    if unit == "characters":
        assert evidence["source_geometry"] == "constructed"
        assert evidence["pieces"][0]["constructed"] is True
    assert "PRIVATE ROOT" not in repr(evidence)


@pytest.mark.asyncio
async def test_nested_host_classification_does_not_borrow_the_callers_selected_head(tmp_path):
    from quarry.types.messages import Messages
    from tests.unit.model_fakes import model_entry

    class Nested(HeadAdapter):
        async def classify_messages(
            self, config, messages, tools, *, timeout=None, token_head=False
        ):
            if not getattr(self, "nested", False):
                self.nested = True
                other = Messages.from_openai([{"role": "user", "content": "unrelated"}])
                assert await self.service.model_classify_messages("classes", other) == 0.8
            return await super().classify_messages(
                config, messages, tools, timeout=timeout, token_head=token_head
            )

    adapter = Nested()
    service = backend(
        tmp_path,
        {
            "classes": model_entry(classify=0),
            "segments": model_entry(spans=0, category="SYNTHETIC"),
        },
        adapter,
    )
    adapter.service = service
    compiled = compile_policy(
        'rule "R" condition: model("classes").classify(messages)>0.5\ncollect: [s for s in model("segments").spans(messages,0.5) if true]',
        runtime_config=service.config,
        model_inference_backend=service,
    )
    result_cache.clear()
    result = await compiled.scan_async({"messages": [{"role": "user", "content": "abcdef"}]})
    assert result.is_complete and result.rule_matches == ["R"], result.evaluation_errors
    assert adapter.head_requests == [False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", [None, "characters", "tokens"])
async def test_shadowed_map_binders_match_alpha_renaming_without_capturing_outer_values(
    tmp_path, unit
):
    from tests.unit.test_message_local_tool_conditions import DATA

    service, adapter = setup(tmp_path, unit)
    left = '[message("user",join([m.name for m in m.tool_calls],",")) for m in assistant_messages]'
    right = '[message("user",join([c.name for c in x.tool_calls],",")) for x in assistant_messages]'
    compiled = compile_policy(
        f'rule "R" condition: model("classes").classify({left})>0.5\ncollect: [s for s in model("segments").spans({right},0.5) if true]',
        runtime_config=service.config,
        model_inference_backend=service,
    )
    result_cache.clear()
    result = await compiled.scan_async(DATA)
    assert result.is_complete and result.rule_matches == ["R"], result.evaluation_errors
    flags = adapter.head_requests if unit is None else [entry[-1] for entry in adapter.requests]
    assert flags == [True]


@pytest.mark.asyncio
async def test_media_derived_text_keeps_part_context_without_image_character_coordinates(tmp_path):
    from quarry.integrations.model_window_evidence import capture_window_result
    from quarry.types.messages import Messages
    from quarry.types.transforms import ContentSourcePart, ProvenanceSegment

    service, adapter = setup(tmp_path, "characters")
    original = Messages.from_openai(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
                ],
            }
        ]
    )
    parts = (ContentSourcePart(0, 0),)
    variant = original[0]._with_transformed_content(
        {0: "abcdef"}, (ProvenanceSegment(0, 6, (), "whole", parts),), replace_media=True
    )
    selected = original._with_overrides({0: variant})
    result_cache.clear()
    with capture_window_result() as capture:
        assert await service.model_classify_messages("classes", selected) == 0.8
    spans = await service.model_span_messages("segments", selected, 0.5)
    span = spans.items[0].value
    assert span.source_unmapped and span.source_parts == parts and span.source_ranges == ()
    deciding = capture.value["deciding"]
    assert deciding["source_geometry"] == "context"
    assert deciding["pieces"][0]["source_parts"] == [{"message_index": 0, "part_index": 0}]
    assert "AA==" not in repr(deciding)
    assert len(adapter.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", [None, "characters", "tokens"])
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("function", ["model_score", "model_scan"])
@pytest.mark.parametrize("target,shared", [(0, True), (-2, True), (-1, False)])
async def test_legacy_field_classifiers_share_compatible_typed_span_heads(
    tmp_path, unit, restored, function, target, shared
):
    service, adapter = setup(tmp_path, unit)
    predicate = (
        'model_score("classes",m.content,0)>0.5'
        if function == "model_score"
        else 'model_scan("classes",m.content,0.5,0)'
    )
    compiled = compile_policy(
        f'rule "R" condition: any m in user_messages[:1] where {predicate}\n'
        f'collect: [s for s in model("segments").spans(message("user",user_messages[{target}].content),0.5) if true]',
        runtime_config=service.config,
        model_inference_backend=service,
    )
    if restored:
        wire = compiled.to_bytes()
        compiled = load_compiled_policy(
            wire, runtime_config=service.config, model_inference_backend=service
        )
        assert compiled.to_bytes() == wire
    result_cache.clear()
    result = await compiled.scan_async(
        {
            "messages": [
                {"role": "system", "content": "preamble"},
                {"role": "user", "content": "abcdef"},
                {"role": "user", "content": "ghijkl"},
            ]
        }
    )
    assert result.is_complete and result.rule_matches == ["R"], result.evaluation_errors
    flags = adapter.head_requests if unit is None else [entry[-1] for entry in adapter.requests]
    assert flags == ([True] if shared else [False, True])


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", [None, "characters", "tokens"])
async def test_arbitrary_legacy_readers_run_on_every_read_of_a_shared_reply(tmp_path, unit):
    from quarry.cache.async_cache import scan_scope
    from quarry.types.messages import Messages

    service, adapter = setup(tmp_path, unit)
    calls = []

    def synchronous(raw):
        calls.append(("sync", tuple(raw.chunks[0].probabilities)))
        return len(calls)

    async def asynchronous(raw):
        calls.append(("async", tuple(raw.chunks[0].probabilities)))
        return len(calls)

    result_cache.clear()
    with scan_scope():
        for count, reader in enumerate((synchronous, synchronous, asynchronous), 1):
            assert await service.model_classification("classes", "abcdef", read=reader) == count
        spans = await service.model_span_messages(
            "segments", Messages.from_openai([{"role": "user", "content": "abcdef"}]), 0.5
        )
        assert len(spans.items) == 1
    probabilities = (0.8, 0.4) if unit is None else (0.8, 0.2)
    assert calls == [("sync", probabilities), ("sync", probabilities), ("async", probabilities)]
    flags = adapter.head_requests if unit is None else [entry[-1] for entry in adapter.requests]
    assert flags == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize(
    "unit,stage",
    [
        (None, "tools"),
        ("characters", "tools"),
        ("tokens", "tools"),
        ("tokens", "encode"),
        ("characters", "render"),
        ("tokens", "render"),
    ],
)
async def test_nested_host_hooks_cannot_borrow_the_active_typed_input(
    tmp_path, unit, stage, restored
):
    from quarry.integrations.model_window_input import join_window_input
    from quarry.types.messages import Messages

    class Nested(HeadAdapter if unit is None else WindowAdapter):
        nested = False

        async def visit(self):
            if not self.nested:
                self.nested = True
                other = Messages.from_openai([{"role": "user", "content": "unrelated"}])
                assert await self.service.model_classify_messages("classes", other) == 0.8

        async def classify_messages(self, *args, **kwargs):
            if stage == "tools":
                await self.visit()
            return await super().classify_messages(*args, **kwargs)

        async def classify_window_batch(self, *args, **kwargs):
            if stage == "tools":
                await self.visit()
            return await super().classify_window_batch(*args, **kwargs)

        async def encode_window_input(self, *args, **kwargs):
            if stage == "encode":
                await self.visit()
            return await super().encode_window_input(*args, **kwargs)

        async def render_window_input(self, config, messages, tools):
            await self.visit()
            return await join_window_input(messages, tools)

    adapter = Nested()
    models = {
        "classes": model_entry(classify=0),
        "segments": model_entry(spans=0, category="SYNTHETIC"),
    }
    if unit is not None:
        for model in models.values():
            model["windows"] = {"unit": unit, "size": 100}
            if stage == "render":
                model["windows"]["renderer"] = "host"
    service = backend(tmp_path, models, adapter)
    adapter.service = service
    args = "messages,tools" if stage == "tools" else "messages"
    policy = compile_policy(
        f'rule "R" condition: model("classes").classify({args})>0.5\n'
        'collect: [s for s in model("segments").spans(messages,0.5) if true]',
        runtime_config=service.config,
        model_inference_backend=service,
    )
    if restored:
        wire = policy.to_bytes()
        policy = load_compiled_policy(
            wire, runtime_config=service.config, model_inference_backend=service
        )
        assert policy.to_bytes() == wire
    result_cache.clear()
    result = await policy.scan_async(
        {
            "messages": [{"role": "user", "content": "abcdef"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "synthetic_tool", "parameters": {"type": "object"}},
                }
            ],
        }
    )
    assert result.is_complete and result.rule_matches == ["R"], result.evaluation_errors
    flags = adapter.head_requests if unit is None else [entry[-1] for entry in adapter.requests]
    assert flags == ([False, False, True] if stage == "tools" else [False, True])
