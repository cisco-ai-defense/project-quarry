# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Unreadable calls retain wire data, position and links without poisoning other rules."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import ClassVar

import pytest

from quarry import IncompleteScanError, InvalidInputError, compile_policy, load_compiled_policy
from quarry.cli import scan_command
from quarry.config import QuarryConfig
from quarry.engine.events import build_static_environment
from quarry.engine.matchers import field_access
from quarry.integrations.adapter_message_transforms import apply_adapter_transform
from quarry.integrations.content_transforms import apply_content_transform
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.integrations.model_usage import model_call_budget
from quarry.scanner import DSLScanner
from quarry.sdk import QuarryGuard
from quarry.types.context import ScanContext
from quarry.types.messages import Messages, ToolCall, UnreadableToolCall
from quarry.types.transforms import ContentTransform, TransformOperation
from tests.unit.model_fakes import ModelAdapter, model_entry, passthrough_builder
from tests.unit.test_adapter_transform_dsl import configured
from tests.unit.test_message_rewrites import service

FAULTS = [
    pytest.param("other", "lookup", id="unsupported-type"),
    pytest.param("", "lookup", id="empty-type"),
    pytest.param("function", "", id="empty-name"),
]


def call(kind="other", name="lookup", identity="b", arguments="{opaque"):
    return {"id": identity, "type": kind, "function": {"name": name, "arguments": arguments}}


def transcript(kind="other", name="lookup", *, response=True):
    messages = [{"role": "assistant", "content": "held", "tool_calls": [call(kind, name)]}]
    if response:
        messages.append({"role": "tool", "tool_call_id": "b", "content": "reply"})
    messages.append({"role": "user", "content": "visible"})
    return messages


def scanner(**conditions):
    return DSLScanner(
        QuarryConfig(
            quarry_rule_texts=tuple(
                f'rule "{name}" condition: {condition}' for name, condition in conditions.items()
            )
        )
    )


@pytest.mark.parametrize("kind,name", FAULTS)
@pytest.mark.parametrize("explain", [False, True])
def test_unused_values_positions_and_call_sizes_remain_readable(kind, name, explain):
    result = scanner(
        Always="true",
        Text='any m in messages where m.content contains "held"',
        Size="len(tool_calls) == 1",
        Exists="any c in tool_calls where true",
        Reply='any r in tool_responses where r.result contains "reply"',
        Skip='false and (any c in tool_calls where c.name == "lookup")',
    ).scan_data({"messages": transcript(kind, name)}, enable_explain=explain)
    assert result.is_complete and set(result.rule_matches) == {
        "Always",
        "Text",
        "Size",
        "Exists",
        "Reply",
    }
    assert not result.evaluation_errors
    if explain:
        assert len(result.explanations) == 6


@pytest.mark.parametrize("kind,name", FAULTS)
@pytest.mark.parametrize("field", ["id", "type", "name", "raw_arguments", "arguments", "response"])
def test_rule_field_reads_raise_typed_input_faults(kind, name, field):
    messages = Messages.from_openai(transcript(kind, name))
    value = messages.tool_calls[0]
    assert isinstance(value, UnreadableToolCall) and value.message_index == 0
    assert field_access(value, "message_index") == 0
    with pytest.raises(InvalidInputError) as error:
        field_access(value, field)
    assert error.value.details["code"] == "invalid_input" and "messages[0].tool_calls[0]" in str(
        error.value
    )
    assert "{opaque" not in str(error.value)


@pytest.mark.parametrize(
    "condition",
    [
        'any c in tool_calls where c.name == "lookup"',
        'any c in tool_calls where c.id == "b"',
        'any c in tool_calls where c.type == "other"',
        'any c in tool_calls where c.raw_arguments contains "opaque"',
        "any c in tool_calls where c.arguments.x != null",
        "any c in tool_calls where c.response != null",
        'count_tool_calls("lookup") == 1',
        'len(get_tool_calls("lookup")) == 1',
        'len(get_tool_responses("lookup")) == 1',
        'any r in tool_responses where r.name == "lookup"',
    ],
)
def test_only_the_rule_reading_a_fault_is_incomplete(condition):
    result = scanner(
        Text='any m in messages where m.content contains "held"', Read=condition
    ).scan_data({"messages": transcript()})
    assert result.completion_state == "incomplete" and result.rule_matches == ["Text"]
    assert result.validation_error is None and set(result.evaluation_errors) == {"Read"}
    assert [f.to_dict() for f in result.failures] == [
        {"category": "invalid_input", "rule_name": "Read"}
    ]


def test_raw_wire_and_canonical_links_remain_available_to_host_inspection():
    raw = transcript()
    messages = Messages.from_openai(raw)
    bad = messages.tool_calls[0]
    assert bad.id == "b" and bad.type == "other" and bad.function.arguments == "{opaque"
    assert messages[1].call is bad and bad._response is messages[1]
    assert messages.to_openai_messages() == raw
    for read in [
        lambda: bad.name,
        lambda: bad.raw_arguments,
        lambda: bad.arguments,
        lambda: bad.response,
        lambda: bad.texts(),
        lambda: bad.argument_texts("arguments.*"),
        lambda: bad.argument_text("raw_arguments"),
    ]:
        with pytest.raises(InvalidInputError):
            read()
    assert bad.place_texts() == (("raw_arguments", "{opaque"),)
    with pytest.raises(AttributeError):
        bad._fault = "changed"


def test_name_queries_match_positive_and_exact_collection_contracts():
    good = {"role": "assistant", "tool_calls": [call("function", "lookup", "g", "{}")]}
    source = Messages.from_openai([good, *transcript()])
    assert source.called_tool("lookup") is True
    with pytest.raises(InvalidInputError):
        source.called_tool("absent")
    for read in [
        lambda: source.count_tool_calls("lookup"),
        lambda: source.get_tool_calls("lookup"),
        lambda: source.get_tool_responses("lookup"),
        lambda: source[:1].get_tool_responses("lookup"),
    ]:
        with pytest.raises(InvalidInputError):
            read()
    good_view = source[:1]
    assert good_view.called_tool("lookup") and good_view.count_tool_calls("lookup") == 1
    assert len(good_view.get_tool_calls("lookup")) == 1
    assert source[:0].get_tool_calls("lookup") == []


def test_shared_named_counts_are_lazy_and_dont_poison_other_sizes():
    raw = transcript()
    static = build_static_environment(ScanContext(raw))
    assert static.surface_sizes["tool_calls"] == 1
    with pytest.raises(InvalidInputError):
        _ = static.tool_call_name_counts["lookup"]
    with pytest.raises(InvalidInputError):
        _ = static.tool_response_name_counts["lookup"]
    assert static.surface_sizes["messages"] == 3


@pytest.mark.parametrize("prepared", [False, True])
def test_prepared_rule_preserves_read_fault_and_reuses_later_valid_data(prepared):
    policy = compile_policy('rule "Read" condition: any c in tool_calls where c.name == "lookup"')
    if prepared:
        policy = load_compiled_policy(policy.to_bytes())
    bad = policy.scan({"messages": transcript()})
    assert (
        bad.completion_state == "incomplete" and bad.failures[0].category.value == "invalid_input"
    )
    good = policy.scan({"messages": transcript("function", "lookup")})
    assert good.is_complete and good.rule_matches == ["Read"]


def model_backend(tmp_path):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "reader": model_entry(
                    classify=0, score=True, tokens=True, physical_model="text-model"
                )
            },
        },
        tmp_path,
        require_rules=False,
    )
    adapter = ModelAdapter()
    return ModelInferenceBackend(
        config, adapters={"fake": adapter}, message_builders={"test": passthrough_builder}
    ), adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["score", "classify", "complete", "tokens", "split"])
async def test_default_backend_refuses_calls_before_cache_provider_and_budget(
    tmp_path, operation, monkeypatch
):
    backend, adapter = model_backend(tmp_path)
    messages = Messages.from_openai(transcript())

    async def cache(*args, **kwargs):
        pytest.fail("unreadable call reached cache")

    monkeypatch.setattr(backend, "_call", cache)
    with model_call_budget(1) as budget:
        with pytest.raises(InvalidInputError):
            if operation == "score":
                await backend.model_score_messages("reader", "policy", messages)
            elif operation == "classify":
                await backend.model_classify_messages("reader", messages)
            elif operation == "complete":
                await backend.model_complete("reader", messages)
            elif operation == "tokens":
                await backend.model_input_tokens_messages("reader", messages)
            else:
                await backend.model_split_messages("reader", messages, 1000)
        assert budget.calls_by_alias == {}
    assert (
        adapter.classifications
        == adapter.score_calls
        == adapter.completions
        == adapter.tokenizations
        == 0
    )


def test_custom_model_backend_is_guarded_before_dispatch(tmp_path):
    backend, _ = model_backend(tmp_path)

    async def custom(*args, **kwargs):
        pytest.fail("unreadable call reached custom model backend")

    backend.model_classify_messages = custom
    policy = compile_policy(
        'rule "Read" condition: model("reader").classify(messages) > 0.5',
        runtime_config=backend.config,
        model_inference_backend=backend,
    )
    result = policy.scan({"messages": transcript()})
    assert (
        result.completion_state == "incomplete"
        and result.failures[0].category.value == "invalid_input"
    )


@pytest.mark.asyncio
async def test_part_transforms_can_read_text_without_reading_calls(tmp_path):
    _, backend, adapter = configured(tmp_path)
    source = Messages.from_openai(transcript())
    transformed = await apply_adapter_transform(source, backend, "convert")
    assert transformed[0].content == "HELD" and source[0].content == "held"
    assert transformed[1].call is transformed[0].tool_calls[0]
    assert transformed[0].tool_calls[0]._response is transformed[1]
    assert isinstance(transformed[1].call, UnreadableToolCall)
    assert len(adapter.parts) == 3
    lower = await apply_content_transform(
        transformed, ContentTransform((TransformOperation("casefold"),))
    )
    assert lower[0].content == "held" and lower[1].call is lower[0].tool_calls[0]


@pytest.mark.asyncio
async def test_whole_rewrite_skips_bad_call_wire_and_keeps_transformed_links(tmp_path):
    class Rewrite:
        local = True

        def __init__(self):
            self.calls = []

        def transform_messages(self, settings, wire):
            self.calls.append(wire)
            return [dict(m, content=[{"type": "text", "text": m["content"]}]) for m in wire]

    adapter = Rewrite()
    _, backend, _ = service(tmp_path, adapter=adapter)
    source = Messages.from_openai(transcript())
    changed = await backend.transform_messages("rewrite", source)
    assert len(adapter.calls) == 1 and all(m["role"] != "assistant" for m in adapter.calls[0])
    assert (
        changed[0].tool_calls[0]._response is changed[1]
        and changed[1].call is changed[0].tool_calls[0]
    )
    assert changed[1] is not source[1] and source[1].call is source[0].tool_calls[0]


def test_composition_and_selection_preserve_unreadable_call_classes_and_wire():
    source = Messages.from_openai(transcript())
    combined = Messages.combine((source[:1], source[1:]))
    assert combined.to_openai_messages() == source.to_openai_messages()
    assert isinstance(combined.tool_calls[0], UnreadableToolCall)
    assert combined[1].call is combined[0].tool_calls[0]
    assert combined[0].tool_calls[0]._response is combined[1]


def test_name_only_argument_fault_does_not_break_position_evidence():
    result = scanner(Name='called_tool("lookup")').scan_data(
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [call("function", "lookup", arguments='{"x":NaN}')],
                }
            ]
        },
        enable_explain=True,
    )
    assert result.is_complete and result.rule_matches == ["Name"]


@pytest.mark.parametrize("kind", ["function", "other"])
def test_registry_keeps_valid_extensions_and_cannot_legitimize_unreadable_calls(kind):
    class ExtendedCall(ToolCall):
        dsl_fields: ClassVar[frozenset[str]] = frozenset({"command"})

        @property
        def command(self) -> str:
            return "marker"

    messages = Messages.from_openai(
        transcript(kind, "lookup"), tool_call_types={(kind, "lookup"): ExtendedCall}
    )
    value = messages.tool_calls[0]
    if kind == "function":
        assert isinstance(value, ExtendedCall) and field_access(value, "command") == "marker"
    else:
        assert isinstance(value, UnreadableToolCall)
        with pytest.raises(InvalidInputError):
            field_access(value, "command")


def test_field_guard_protocol_ignores_json_keys_and_instance_descriptors():
    class Data:
        value = "held"

        @property
        def guard_read(self):
            pytest.fail("field access evaluated an instance guard descriptor")

    assert field_access({"guard_read": "ignored", "id": "held"}, "id") == "held"
    assert field_access(Data(), "value") == "held"


@pytest.mark.parametrize(
    "raw",
    [
        {"id": "c", "type": "function", "function": {"arguments": "{}"}},
        {"id": "c", "type": "function", "function": {"name": None, "arguments": "{}"}},
        {"id": "c", "type": 4, "function": {"name": "lookup", "arguments": "{}"}},
        {"id": "c", "type": "other"},
        {"id": "c", "type": "function", "function": {"name": "lookup"}},
        {"id": "c", "type": "function", "function": {"name": "lookup", "arguments": {}}},
    ],
)
def test_malformed_call_shapes_remain_structural_whole_input_errors(raw):
    result = scanner(Always="true").scan_data(
        {"messages": [{"role": "assistant", "tool_calls": [raw]}]}
    )
    assert result.completion_state == "invalid" and result.rule_matches == []


def surface_config(tmp_path):
    paths = []
    for name, condition in {
        "Text": 'any m in messages where m.content contains "held"',
        "Read": 'any c in tool_calls where c.id == "b"',
    }.items():
        path = tmp_path / (name + ".rule")
        path.write_text(f'rule "{name}" condition: {condition}')
        paths.append(str(path))
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"rules": paths}))
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_sdk_call_fault_retains_text_matches_and_fails_closed(tmp_path, asynchronous):
    guard = QuarryGuard(surface_config(tmp_path))
    with pytest.raises(IncompleteScanError) as failure:
        if asynchronous:
            await guard.scan_async({"messages": transcript()})
        else:
            await asyncio.to_thread(guard.scan, {"messages": transcript()})
    result = failure.value.scan_result
    assert result.rule_matches == ["Text"] and result.failures[0].category.value == "invalid_input"


def test_cli_call_fault_has_partial_matches_typed_category_and_non_success_exit(tmp_path, capsys):
    config = surface_config(tmp_path)
    target = tmp_path / "input.json"
    target.write_text(json.dumps({"messages": transcript()}))
    status = scan_command(
        SimpleNamespace(
            config=config,
            target=target,
            no_limits=False,
            recursive=False,
            verbose=False,
            batch=False,
            validate_only=False,
            output_format="json",
            explain=False,
            show_timing=False,
            output=None,
        )
    )
    [result] = json.loads(capsys.readouterr().out)
    assert status == 2 and result["completion_state"] == "incomplete"
    assert result["matches"] == [{"rule": "Text"}]
    assert result["failures"] == [{"category": "invalid_input", "rule_name": "Read"}]
