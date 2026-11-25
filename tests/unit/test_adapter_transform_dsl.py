# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Compiled adapter transforms, resource identity and mapped text/media provenance."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import orjson
import pytest

import quarry.types.messages as message_types
from quarry import compile_policy, load_compiled_policy
from quarry.config import LocalTransformConfig, ModelTransformConfig, QuarryConfig
from quarry.dsl.parser import DSLParser
from quarry.engine.atoms import Cost
from quarry.engine.compiler import compile_rule
from quarry.integrations.adapter_message_transforms import apply_adapter_transform
from quarry.integrations.content_transforms import apply_content_transform
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.integrations.resources import ResourcePreparationError
from quarry.types.adapter_transforms import TransformResult, TransformRun
from quarry.types.messages import Messages
from quarry.types.results import Score, Span
from quarry.types.transforms import ContentTransform, TransformOperation
from tests.unit.model_fakes import model_entry


class Adapter:
    local = True
    transform_parts = ("text", "image_url", "input_audio")

    def __init__(self):
        self.parts = []
        self.inputs = []
        self.output = None

    def transform(self, config, part, *, timeout=None):
        self.parts.append(part)
        if self.output is not None:
            return self.output
        return part["text"].upper() if part["type"] == "text" else "LABEL"

    async def classify_messages(self, config, messages, tools, *, timeout=None):
        self.inputs.append(messages)
        return SimpleNamespace(chunks=[SimpleNamespace(probabilities=[0.9])])

    async def decode_spans(self, config, raw, messages, threshold):
        start = messages[0].content.index("LABEL")
        return [Span(0, start, start + 5, Score(0.9))]


def configured(tmp_path, *, model=False, definitions=None, adapter=None):
    adapter = adapter or Adapter()
    adapter.local = not (model or any("model" in item for item in (definitions or {}).values()))
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "extractor": model_entry(physical_model="synthetic-extractor"),
                "judge": model_entry(classify=0),
                "finder": model_entry(category="ENTITY", classify=0, spans=0),
            },
            "transforms": definitions
            or {"convert": {"model": "extractor"} if model else {"adapter": "fake"}},
        },
        tmp_path,
        require_rules=False,
    )
    return config, ModelInferenceBackend(config, adapters={"fake": adapter}), adapter


def compile_condition(config, backend, condition):
    return compile_policy(
        'rule "R" condition: ' + condition, runtime_config=config, model_inference_backend=backend
    )


def wire_nodes(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from wire_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from wire_nodes(child)


@pytest.mark.parametrize("model", [False, True])
@pytest.mark.parametrize("prepared", [False, True])
def test_indexed_transformed_text_executes_with_correct_cost_and_required_models(
    tmp_path, model, prepared
):
    config, backend, adapter = configured(tmp_path, model=model)
    policy = compile_condition(
        config, backend, 'messages.transform("convert")[0].content contains "BLOCK"'
    )
    assert policy._scanner.resource_requirements["llm_models"] == (
        {"extractor"} if model else set()
    )
    specs = policy._scanner.compiled_rules["R"].atom_specs.values()
    assert {spec.cost for spec in specs} == ({Cost.LLM} if model else {Cost.PURE})
    if prepared:
        policy = load_compiled_policy(
            policy.to_bytes(), runtime_config=config, model_inference_backend=backend
        )
    result = policy.scan({"messages": [{"role": "user", "content": "block"}]})
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(adapter.parts) == 1


@pytest.mark.parametrize("model", [False, True])
def test_cheap_false_gate_skips_transform_dispatch(tmp_path, model):
    config, backend, adapter = configured(tmp_path, model=model)
    policy = compile_condition(
        config, backend, 'false and messages.transform("convert")[0].content contains "BLOCK"'
    )
    result = policy.scan({"messages": [{"role": "user", "content": "block"}]})
    assert result.is_complete and result.rule_matches == []
    assert adapter.parts == []


@pytest.mark.parametrize("model", [False, True])
@pytest.mark.parametrize("change", ["missing", "parts", "kind"])
def test_prepared_policy_requires_the_matching_transform_registry(tmp_path, model, change):
    config, backend, adapter = configured(tmp_path, model=model)
    policy = compile_condition(
        config, backend, 'messages.transform("convert")[0].content contains "BLOCK"'
    )
    if change == "missing":
        definitions = {}
    elif change == "kind":
        definitions = {
            "convert": LocalTransformConfig("fake") if model else ModelTransformConfig("extractor")
        }
    else:
        definitions = {"convert": replace(config.transforms["convert"], parts=("text",))}
    changed = replace(config, transforms=definitions)
    changed_backend = ModelInferenceBackend(changed, adapters={"fake": adapter})
    with pytest.raises(ResourcePreparationError, match="configuration differs"):
        load_compiled_policy(
            policy.to_bytes(), runtime_config=changed, model_inference_backend=changed_backend
        )
    assert adapter.parts == []


def test_local_settings_are_bound_but_are_not_serialized(tmp_path):
    config, backend, _ = configured(
        tmp_path,
        definitions={
            "convert": {
                "adapter": "fake",
                "settings": {"marker": "private-setting-text-unique-to-this-test"},
            }
        },
    )
    policy = compile_condition(
        config, backend, 'messages.transform("convert")[0].content contains "BLOCK"'
    )
    payload = policy.to_bytes()
    assert b"private-setting-text-unique-to-this-test" not in payload
    changed = replace(
        config, transforms={"convert": LocalTransformConfig("fake", {"marker": "different"})}
    )
    with pytest.raises(ResourcePreparationError, match="configuration differs"):
        load_compiled_policy(payload, runtime_config=changed, model_inference_backend=backend)


@pytest.mark.parametrize("tamper", ["pure-cost", "local-kind", "invalid-reference"])
def test_prepared_model_transform_cannot_masquerade_as_local_work(tmp_path, tamper):
    config, backend, adapter = configured(tmp_path, model=True)
    policy = compile_condition(
        config, backend, 'messages.transform("convert")[0].content contains "BLOCK"'
    )
    document = orjson.loads(policy.to_bytes())
    if tamper == "pure-cost":
        for atom in document["rule"]["artifact"]["atom_specs"]:
            atom["cost"] = "pure"
    else:
        node = next(item for item in wire_nodes(document) if item.get("kind") == "model_transform")
        if tamper == "local-kind":
            node["kind"] = "local_transform"
        else:
            node["value"] = "x" * 10_000
    with pytest.raises(ValueError):
        load_compiled_policy(
            orjson.dumps(document), runtime_config=config, model_inference_backend=backend
        )
    assert adapter.parts == []


@pytest.mark.parametrize(
    "source",
    [
        'model("judge").classify([message("user", item.content).transform("convert") for item in messages]) > 0.5',
        'model("judge").classify([item for item in messages if message("user", item.content).transform("convert")[0].content contains "x"]) > 0.5',
        'model("judge").classify(messages if messages.transform("convert")[0].content contains "x" else messages) > 0.5',
    ],
)
def test_model_transform_is_rejected_inside_local_values_and_predicates(tmp_path, source):
    config, _, adapter = configured(tmp_path, model=True)
    result = compile_rule(
        DSLParser().parse('rule "R" condition: ' + source), transforms=config.transforms
    )
    assert not result.ok
    assert "cannot call a model" in " ".join(item.message for item in result.diagnostics.errors)
    assert adapter.parts == []


@pytest.mark.asyncio
async def test_exact_and_reordered_adapter_runs_compose_through_existing_transforms(tmp_path):
    config, backend, adapter = configured(tmp_path)
    source = Messages.from_openai([{"role": "user", "content": "  abcd  "}])
    stripped = await apply_content_transform(
        source, ContentTransform((TransformOperation("strip"),))
    )
    adapter.output = TransformResult(
        "cdab",
        (
            TransformRun(0, 2, 2, 4, "linear"),
            TransformRun(2, 4, 0, 2, "linear"),
        ),
    )
    transformed = await apply_adapter_transform(stripped, backend, "convert")
    assert transformed[0].content == "cdab"
    assert [item.to_dict() for item in transformed.source_ranges(0, 0, 4)] == [
        {"message_index": 0, "start": 4, "end": 6},
        {"message_index": 0, "start": 2, "end": 4},
    ]
    assert source[0].content == "  abcd  "
    assert transformed._bind_span(Span(0, 0, 2)).to_dict()["source_ranges"] == [
        {"message_index": 0, "start": 4, "end": 6},
    ]


@pytest.mark.asyncio
async def test_media_replacement_keeps_unselected_parts_and_reports_part_only_origins(tmp_path):
    config, backend, adapter = configured(
        tmp_path, definitions={"convert": {"adapter": "fake", "parts": ["image_url"]}}
    )
    wire = {
        "role": "user",
        "content": [
            {"type": "text", "text": "prefix "},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
        ],
    }
    source = Messages.from_openai([wire])
    transformed = await apply_adapter_transform(source, backend, "convert")
    assert len(adapter.parts) == 1 and adapter.parts[0]["type"] == "image_url"
    assert transformed.to_openai_messages()[0]["content"] == [
        wire["content"][0],
        {"type": "text", "text": "LABEL"},
        wire["content"][2],
    ]
    span = transformed._bind_span(Span(0, 7, 12))
    assert span.to_dict() == {
        "message_index": 0,
        "start": 7,
        "end": 12,
        "source_unmapped": True,
        "source_parts": [{"message_index": 0, "part_index": 1}],
    }
    assert transformed.extract(span) == "LABEL"
    assert transformed.span.to_dict()["end"] == 12
    assert transformed.span.to_dict()["source_parts"] == [{"message_index": 0, "part_index": 1}]
    assert source.to_openai_messages() == [wire]


@pytest.mark.asyncio
async def test_part_origins_survive_local_operations_slicing_and_message_composition(tmp_path):
    config, backend, adapter = configured(tmp_path)
    source = Messages.from_openai(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ],
            }
        ]
    )
    transformed = await apply_adapter_transform(source, backend, "convert")
    lower = await apply_content_transform(
        transformed,
        ContentTransform(
            (
                TransformOperation("casefold"),
                TransformOperation("replace", "label", "marker"),
            )
        ),
    )
    combined = await Messages._combine_async((lower[:1],))
    span = combined._bind_span(Span(0, 1, 4))
    assert span.to_dict()["source_parts"] == [{"message_index": 0, "part_index": 0}]
    assert span.source_ranges == () and span.source_unmapped
    assert combined.extract(span) == "ark"


@pytest.mark.parametrize("model", [False, True])
def test_media_derived_model_span_survives_prepared_round_trip(tmp_path, model):
    config, backend, adapter = configured(tmp_path, model=model)
    source = 'rule "R" condition: true collect: [span for span in model("finder").spans(messages.transform("convert"), 0.5) if true]'
    policy = compile_policy(source, runtime_config=config, model_inference_backend=backend)
    loaded = load_compiled_policy(
        policy.to_bytes(), runtime_config=config, model_inference_backend=backend
    )
    result = loaded.scan(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    ],
                }
            ]
        }
    )
    assert result.is_complete and result.rule_matches == ["R"]
    span = result.outputs["R"].items[0].value
    assert span.to_dict() == {
        "message_index": 0,
        "start": 0,
        "end": 5,
        "score": {"value": 0.9},
        "source_unmapped": True,
        "source_parts": [{"message_index": 0, "part_index": 0}],
    }
    document = orjson.loads(policy.to_bytes())
    document["rule"]["artifact"]["collect_plan"]["has_external_work"] = False
    with pytest.raises(ValueError, match="has_external_work"):
        load_compiled_policy(
            orjson.dumps(document), runtime_config=config, model_inference_backend=backend
        )


@pytest.mark.asyncio
async def test_tool_relinking_preserves_links_and_source_graph(tmp_path):
    config, backend, adapter = configured(tmp_path)
    source = Messages.from_openai(
        [
            {
                "role": "assistant",
                "content": "agent",
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "inspect", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c", "content": "result"},
        ]
    )
    transformed = await apply_adapter_transform(source, backend, "convert")
    call = transformed[0].tool_calls[0]
    assert call.response is transformed[1]
    assert transformed[1].call is call
    assert call.response.content == "RESULT"
    assert source[0].tool_calls[0].response is source[1]
    assert source[1].content == "result"
    transformed.assert_self_contained()


@pytest.mark.asyncio
async def test_async_override_relinking_can_be_cancelled_without_mutating_source(
    tmp_path, monkeypatch
):
    config, backend, adapter = configured(tmp_path)
    calls = [
        {"id": f"c{i}", "type": "function", "function": {"name": "inspect", "arguments": "{}"}}
        for i in range(3_000)
    ]
    source = Messages.from_openai([{"role": "assistant", "content": "agent", "tool_calls": calls}])
    checkpoints = 0

    async def stop():
        nonlocal checkpoints
        checkpoints += 1
        raise asyncio.CancelledError

    monkeypatch.setattr(message_types, "_yield_combine_cancellation", stop)
    with pytest.raises(asyncio.CancelledError):
        await apply_adapter_transform(source, backend, "convert")
    assert checkpoints == 1
    assert source[0].content == "agent"
    assert source[0].tool_calls[0].response is None


def test_mixed_transform_chain_uses_one_aggregate_limit(tmp_path):
    config, _, adapter = configured(
        tmp_path,
        definitions={
            "convert": {"adapter": "fake"},
            "local": {"operations": [{"op": "strip"}]},
        },
    )
    for count, expected in ((16, True), (17, False)):
        expression = "messages" + "".join(
            '.transform("convert")' if i % 2 else '.transform("local")' for i in range(count)
        )
        source = f'rule "R" condition: model("judge").classify({expression}) > 0.5'
        result = compile_rule(DSLParser().parse(source), transforms=config.transforms)
        assert result.ok is expected
    assert adapter.parts == []


@pytest.mark.parametrize("model", [False, True])
@pytest.mark.parametrize("identical", [False, True])
def test_scanner_budget_counts_only_distinct_model_transform_requests(tmp_path, model, identical):
    from quarry.safety.limits import SafetyLimits

    config, backend, adapter = configured(tmp_path, model=model)
    config = replace(config, safety_limits=SafetyLimits(max_model_calls=1, rule_timeout=None))
    policy = compile_condition(
        config, backend, 'messages.transform("convert")[-1].content contains "BLOCK"'
    )
    messages = [
        {"role": "user", "content": "block" if identical else "first"},
        {"role": "user", "content": "block"},
    ]
    result = policy.scan({"messages": messages})
    if model and not identical:
        assert not result.is_complete
        assert "max_model_calls=1" in str(result.evaluation_errors)
    else:
        assert result.is_complete and result.rule_matches == ["R"]
    # Local callbacks run separately for each message, even with identical raw parts.
    # Only the model-backed request cache can deduplicate those parts.
    assert len(adapter.parts) == (1 if model else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "detail": "low"}},
        {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
    ],
)
async def test_shared_media_reply_is_bound_to_each_callers_original_part(tmp_path, part):
    config, backend, adapter = configured(tmp_path)
    source = Messages.from_openai(
        [{"role": "user", "content": [part]}, {"role": "user", "content": [part]}]
    )
    transformed = await apply_adapter_transform(source, backend, "convert")
    assert len(adapter.parts) == 1
    for local in (0, 1):
        span = transformed._bind_span(Span(local, 0, 5))
        assert span.source_ranges == ()
        assert span.to_dict()["source_parts"] == [{"message_index": local, "part_index": 0}]
    selected = transformed[1:]
    assert selected._bind_span(Span(0, 0, 5)).message_index == 1


def test_prepared_map_cannot_hide_a_model_transform_under_a_local_transform_kind(tmp_path):
    from quarry.types.adapter_transforms import adapter_transform_reference

    config, backend, adapter = configured(tmp_path)
    policy = compile_condition(
        config,
        backend,
        'model("judge").classify([item.transform("convert") for item in messages]) > 0.5',
    )
    document = orjson.loads(policy.to_bytes())
    node = next(item for item in wire_nodes(document) if item.get("kind") == "local_transform")
    node["kind"] = "model_transform"
    node["value"] = adapter_transform_reference("convert", ModelTransformConfig("extractor"))
    with pytest.raises(ValueError, match="message map cannot call a model"):
        load_compiled_policy(
            orjson.dumps(document), runtime_config=config, model_inference_backend=backend
        )
    assert adapter.parts == []


@pytest.mark.asyncio
async def test_explicit_unmapped_text_keeps_message_context_without_character_claims(tmp_path):
    config, backend, adapter = configured(tmp_path)
    adapter.output = TransformResult("generated", (TransformRun(0, 9),))
    source = Messages.from_openai([{"role": "user", "content": "original"}])
    transformed = await apply_adapter_transform(source, backend, "convert")
    span = transformed._bind_span(Span(0, 0, 9))
    assert span.source_ranges == () and span.source_unmapped
    assert span.to_dict()["source_parts"] == [{"message_index": 0}]
    with pytest.raises(ValueError, match="no source position"):
        transformed.source_ranges(0, 0, 9)


@pytest.mark.parametrize("selected", [True, False])
def test_model_transform_evidence_names_only_used_aliases_and_contains_no_part_text(
    tmp_path, selected
):
    definitions = {
        "convert": {"model": "extractor", "parts": ["text"] if selected else ["image_url"]}
    }
    config, backend, _ = configured(tmp_path, definitions=definitions)
    policy = compile_condition(config, backend, 'messages.transform("convert")[0].content != ""')
    result = policy.scan(
        {"messages": [{"role": "user", "content": "block sensitive-fragment"}]}, explain=True
    )
    assert result.is_complete
    evidence = result.detailed_matches[0].model_results[0]
    assert evidence.models == (("extractor",) if selected else ())
    assert "sensitive-fragment" not in str(evidence.to_dict()).casefold()


def test_local_adapter_map_executes_and_model_transform_can_feed_a_local_filter(tmp_path):
    for model in (False, True):
        config, backend, adapter = configured(tmp_path, model=model)
        value = (
            '[item for item in messages.transform("convert") if item.role == "user"]'
            if model
            else '[item.transform("convert") for item in messages]'
        )
        policy = compile_condition(config, backend, f'model("judge").classify({value}) > 0.5')
        loaded = load_compiled_policy(
            policy.to_bytes(), runtime_config=config, model_inference_backend=backend
        )
        result = loaded.scan(
            {
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                ]
            }
        )
        assert result.is_complete and result.rule_matches == ["R"]
        assert adapter.inputs[-1] == (
            [{"role": "user", "content": "FIRST"}]
            if model
            else [
                {"role": "user", "content": "FIRST"},
                {"role": "assistant", "content": "SECOND"},
            ]
        )


def test_model_transform_in_unchosen_branch_is_not_dispatched(tmp_path):
    config, backend, adapter = configured(tmp_path, model=True)
    policy = compile_condition(
        config,
        backend,
        'model("judge").classify(messages.transform("convert") if false else messages) > 0.5',
    )
    loaded = load_compiled_policy(
        policy.to_bytes(), runtime_config=config, model_inference_backend=backend
    )
    result = loaded.scan({"messages": [{"role": "user", "content": "unchanged"}]})
    assert result.is_complete and result.rule_matches == ["R"]
    assert adapter.parts == []
    assert adapter.inputs == [[{"role": "user", "content": "unchanged"}]]


@pytest.mark.asyncio
async def test_overlapping_adapter_maps_cannot_exceed_the_shared_work_bound(monkeypatch):
    import quarry.integrations.content_transforms as transforms
    from quarry.integrations.adapter_message_transforms import _compose_result
    from quarry.types.transforms import ContentSourceRange, ProvenanceSegment

    monkeypatch.setattr(transforms, "MAX_TRANSFORM_WORK", 1_000)
    metrics = transforms._TransformMetrics()
    meter = transforms._OperationMeter(metrics)
    mapping = tuple(
        ProvenanceSegment(i, i + 1, (ContentSourceRange(0, None, i, i + 1, i, i + 1),))
        for i in range(32)
    )
    # Every output character asks to visit the entire input map, despite short output.
    result = TransformResult("x" * 32, tuple(TransformRun(i, i + 1, 0, 32) for i in range(32)))
    with pytest.raises(ValueError, match="character-work"):
        await _compose_result(result, mapping, (), meter)
    assert metrics.work_units == 1_001
    assert metrics.source_visits > 0
