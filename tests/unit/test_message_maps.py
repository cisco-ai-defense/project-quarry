# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Ordered message maps at the provider boundary and prepared-policy boundary."""

import asyncio
from dataclasses import replace

import orjson
import pytest

import quarry.integrations.model_expression as expression_module
from quarry import compile_policy, load_compiled_policy
from quarry.dsl.parser import DSLParser
from quarry.engine.compiler import compile_rule
from quarry.language.example_records import EXAMPLE_RECORDS
from quarry.safety.limits import SafetyLimits
from quarry.types.dsl.render import render_condition
from quarry.types.transforms import ContentTransform, TransformOperation
from tests.unit.model_fakes import ModelAdapter, model_entry, v2_scanner

CONVERSATION = {
    "messages": [
        {"role": "user", "content": "  FIRST  "},
        {"role": "assistant", "content": "  Reply  "},
        {"role": "user", "content": "  LAST  "},
    ]
}


def scanner_for(tmp_path, value, adapter=None):
    adapter = adapter or ModelAdapter()
    scanner = v2_scanner(
        tmp_path,
        {"R": f'model("judge").classify({value}) > 0.5'},
        {"judge": model_entry(classify=0)},
        adapter,
        transforms={"clean": {"operations": [{"op": "strip"}, {"op": "casefold"}]}},
    )
    return scanner, adapter


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "[item for item in messages]",
            [("user", "  FIRST  "), ("assistant", "  Reply  "), ("user", "  LAST  ")],
        ),
        (
            "[[outer for inner in messages[:1]] for outer in messages]",
            [("user", "  FIRST  "), ("assistant", "  Reply  "), ("user", "  LAST  ")],
        ),
        (
            '[item.transform("clean") if item.role == "user" else item for item in messages]',
            [("user", "first"), ("assistant", "  Reply  "), ("user", "last")],
        ),
        (
            '[message("user", item.content) for item in messages if item.role == "user"]',
            [("user", "  FIRST  "), ("user", "  LAST  ")],
        ),
        (
            "[item if true else item for item in "
            '(messages[-1:] + messages[:1]).transform("clean")]',
            [("user", "last"), ("user", "first")],
        ),
        (
            '[item.transform("clean") for item in message("system", "  NOTE  ") + messages]',
            [("system", "note"), ("user", "first"), ("assistant", "reply"), ("user", "last")],
        ),
        (
            "[item if true else item for item in "
            '[item.transform("clean") for item in messages if item.role == "user"]]',
            [("user", "first"), ("user", "last")],
        ),
    ],
)
def test_message_maps_preserve_written_order_and_selected_content(tmp_path, value, expected):
    scanner, adapter = scanner_for(tmp_path, value)

    result = scanner.scan_data(CONVERSATION)

    assert result.is_complete
    assert result.rule_matches == ["R"]
    assert adapter.classification_messages == [
        [{"role": role, "content": content} for role, content in expected]
    ]


def test_indexed_transformed_message_keeps_its_owning_view(tmp_path):
    scanner, adapter = scanner_for(tmp_path, 'messages.transform("clean")[-1]')

    result = scanner.scan_data(CONVERSATION)

    assert result.is_complete
    assert adapter.classification_messages == [[{"role": "user", "content": "last"}]]


def test_map_of_offered_tools_builds_messages_without_sending_other_fields(tmp_path):
    scanner, adapter = scanner_for(
        tmp_path, '[message("user", item.description) for item in tools if item.name == "read"]'
    )
    data = {
        "messages": CONVERSATION["messages"],
        "tools": [
            {"type": "function", "function": {"name": "read", "description": "selected"}},
            {"type": "function", "function": {"name": "write", "description": "excluded"}},
        ],
    }

    result = scanner.scan_data(data)

    assert result.is_complete
    assert adapter.classification_messages == [[{"role": "user", "content": "selected"}]]


def test_message_map_is_not_limited_by_concat_operand_count(tmp_path):
    scanner, adapter = scanner_for(tmp_path, "[item if true else item for item in messages]")
    scanner.limits = SafetyLimits(rule_timeout=None)
    messages = [{"role": "user", "content": f"entry {index}"} for index in range(1_000)]

    result = scanner.scan_data({"messages": messages})

    assert result.is_complete
    assert adapter.classification_messages == [messages]


def test_filtered_items_do_not_evaluate_the_projection(tmp_path, monkeypatch):
    scanner, adapter = scanner_for(
        tmp_path, '[item.transform("clean") for item in messages if false]'
    )

    async def unexpected_transform(*args, **kwargs):
        raise AssertionError("a filtered item must not be transformed")

    monkeypatch.setattr(expression_module, "apply_content_transform", unexpected_transform)
    result = scanner.scan_data(CONVERSATION)

    assert result.is_complete
    assert result.rule_matches == []
    assert adapter.classification_messages == []


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("[item.content for item in messages]", "must produce message values"),
        ("[item for item in 7]", "collection of structured items"),
        (
            '[item.transform("clean") for item in messages if item.content]',
            "filter must be Boolean",
        ),
        ('[item.transform("clean") if true else 1 for item in messages]', "compatible types"),
        ('[message("user", "x") for item in messages if item.message_index == 1]', "provenance"),
        ('[model("judge").complete(messages) for item in messages]', "cannot call a model"),
        (
            '[message("user", "x") for item in messages if model("judge").classify(messages) > 0.5]',
            "cannot call a model",
        ),
    ],
)
def test_impossible_message_maps_fail_during_rule_load(value, message):
    rule = DSLParser().parse(f'rule "R"\ncondition: model("judge").classify({value}) > 0.5\n')

    result = compile_rule(
        rule, transforms={"clean": ContentTransform((TransformOperation("strip"),))}
    )

    assert not result.ok
    assert message in result.diagnostics.errors[0].message


@pytest.mark.parametrize(
    "boundary", ["source", "output", "characters", "source-hard", "output-hard"]
)
def test_message_map_limits_fail_before_provider_dispatch(tmp_path, monkeypatch, boundary):
    value = "[item if true else item for item in messages]"
    if boundary in {"output", "output-hard"}:
        value = "[item + item for item in messages]"
    scanner, adapter = scanner_for(tmp_path, value)
    scanner.limits = SafetyLimits(rule_timeout=None, max_comprehension_size=4)
    if boundary == "source":
        scanner.limits = replace(scanner.limits, max_comprehension_size=2)
    elif boundary == "characters":
        scanner.limits = replace(scanner.limits, enforce_limits=False)
        monkeypatch.setattr(expression_module, "MAX_MESSAGE_MAP_CHARACTERS", 8)
    elif boundary.endswith("-hard"):
        scanner.limits = replace(scanner.limits, enforce_limits=False)
        monkeypatch.setattr(
            expression_module, "MAX_MESSAGE_MAP_ITEMS", 2 if boundary == "source-hard" else 4
        )

    result = scanner.scan_data(CONVERSATION)

    assert not result.is_complete
    assert result.evaluation_errors
    assert adapter.classification_messages == []


def test_prepared_message_map_preserves_transforms_without_a_registry(tmp_path):
    value = '[item.transform("clean") if item.role == "user" else item for item in messages]'
    scanner, adapter = scanner_for(tmp_path, value)
    policy = compile_policy(
        f'rule "R"\ncondition: model("judge").classify({value}) > 0.5\n',
        runtime_config=scanner.config,
        model_inference_backend=scanner.resource_manager.inference_backend,
    )
    loaded = load_compiled_policy(
        policy.to_bytes(),
        runtime_config=replace(scanner.config, transforms={}),
        model_inference_backend=scanner.resource_manager.inference_backend,
    )

    result = loaded.scan(CONVERSATION)

    assert result.is_complete
    assert adapter.classification_messages[0][0]["content"] == "first"


@pytest.mark.parametrize(
    "malformation", ["projection", "filter", "filter-messages", "nested-model"]
)
def test_prepared_message_maps_reject_invalid_bodies(tmp_path, malformation):
    scanner, _ = scanner_for(tmp_path, '[message("user", item.content) for item in messages]')
    policy = compile_policy(
        'rule "R"\ncondition: model("judge").classify('
        '[message("user", item.content) for item in messages]) > 0.5\n',
        runtime_config=scanner.config,
        model_inference_backend=scanner.resource_manager.inference_backend,
    )
    document = orjson.loads(policy.to_bytes())
    pending = [document["rule"]["artifact"]["atom_specs"][0]["model_expression"]]
    while pending:
        mapped = pending.pop()
        if mapped["kind"] == "message_map":
            break
        pending.extend(mapped["children"])
    if malformation == "nested-model":
        mapped["children"][2] = {
            "kind": "model_call",
            "value": "judge\u001fcomplete",
            "children": [
                {
                    "kind": "messages",
                    "value": None,
                    "children": [],
                }
            ],
        }
    elif malformation == "filter-messages":
        mapped["children"][1] = {"kind": "messages", "value": None, "children": []}
    else:
        mapped["children"][1 if malformation == "filter" else 2] = {
            "kind": "literal",
            "value": "wrong type",
            "children": [],
        }

    with pytest.raises(ValueError, match="message map"):
        load_compiled_policy(
            orjson.dumps(document),
            runtime_config=scanner.config,
            model_inference_backend=scanner.resource_manager.inference_backend,
        )


def test_message_map_rendering_round_trips_authored_syntax():
    source = '[item.transform("clean") if item.role == "user" else item for item in messages]'
    parsed = DSLParser().parse(f'rule "R"\ncondition: model("judge").classify({source}) > 0.5\n')

    rendered = render_condition(parsed.condition)

    assert DSLParser().parse(f'rule "R"\ncondition: {rendered}\n').condition == parsed.condition
    assert "_transform" not in rendered


@pytest.mark.parametrize("budget", ["count", "characters"])
def test_constructed_map_output_is_bounded_across_iterations(tmp_path, monkeypatch, budget):
    scanner, adapter = scanner_for(tmp_path, '[message("user", "text") for item in messages]')
    scanner.limits = SafetyLimits(rule_timeout=None, enforce_limits=False)
    monkeypatch.setattr(
        expression_module,
        "MAX_CONSTRUCTED_MESSAGES" if budget == "count" else "MAX_CONSTRUCTED_TEXT_CHARACTERS",
        2 if budget == "count" else 8,
    )

    result = scanner.scan_data(CONVERSATION)

    assert not result.is_complete
    assert "constructed" in result.evaluation_errors["R"]
    assert adapter.classification_messages == []


def test_packaged_message_map_example_with_deterministic_adapter(tmp_path):
    example = next(record for record in EXAMPLE_RECORDS if record.id == "example:message-map")
    scanner, adapter = scanner_for(tmp_path, "messages")
    policy = compile_policy(
        example.rule_text,
        runtime_config=scanner.config,
        model_inference_backend=scanner.resource_manager.inference_backend,
    )

    matching = policy.scan(example.matching)
    non_matching = policy.scan(example.non_matching)

    assert matching.is_complete and matching.rule_matches == ["Mapped model input"]
    assert non_matching.is_complete and non_matching.rule_matches == []
    assert adapter.classification_messages == [[{"role": "user", "content": "selected input"}]]


def test_tool_call_maps_construct_each_selected_call_once(tmp_path):
    scanner, adapter = scanner_for(
        tmp_path, '[message("user", call.name) for call in messages[-1].tool_calls]'
    )
    data = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"c{index}",
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"},
                    }
                    for index, name in enumerate(["lookup", "write"])
                ],
            }
        ]
    }

    result = scanner.scan_data(data)

    assert result.is_complete
    assert adapter.classification_messages == [
        [
            {"role": "user", "content": "lookup"},
            {"role": "user", "content": "write"},
        ]
    ]


def test_mapped_tool_units_keep_selected_links_and_source_offsets(tmp_path, monkeypatch):
    scanner, adapter = scanner_for(tmp_path, '[item.transform("clean") for item in messages]')
    backend = scanner.resource_manager.inference_backend
    original = backend.model_classify_messages
    captured = []

    async def capture(alias, messages, *args, **kwargs):
        captured.append(messages)
        return await original(alias, messages, *args, **kwargs)

    monkeypatch.setattr(backend, "model_classify_messages", capture)
    data = {
        "messages": [
            {"role": "user", "content": "  ASK  "},
            {
                "role": "assistant",
                "content": "  WAIT  ",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "  DONE  "},
        ]
    }

    result = scanner.scan_data(data)

    assert result.is_complete
    assert captured[0][1].tool_calls[0].response is captured[0][2]
    assert captured[0][2].call is captured[0][1].tool_calls[0]
    assert captured[0][2].content == "done"
    assert [item.to_dict() for item in captured[0].source_ranges(2, 0, 4)] == [
        {"message_index": 2, "start": 2, "end": 6}
    ]
    assert data["messages"][2]["content"] == "  DONE  "
    assert len(adapter.classification_messages) == 1


def test_mapped_copies_keep_provenance_for_each_separate_constructor(tmp_path, monkeypatch):
    scanner, _ = scanner_for(tmp_path, '[message("user", item.content) for item in messages]')
    backend = scanner.resource_manager.inference_backend
    original = backend.model_classify_messages
    captured = []

    async def capture(alias, messages, *args, **kwargs):
        captured.append(messages)
        return await original(alias, messages, *args, **kwargs)

    monkeypatch.setattr(backend, "model_classify_messages", capture)

    result = scanner.scan_data(CONVERSATION)

    assert result.is_complete
    for index, message in enumerate(captured[0]):
        ranges = captured[0].source_ranges(message.message_index, 0, len(message.content))
        assert [item.to_dict() for item in ranges] == [
            {"message_index": index, "start": 0, "end": len(message.content)}
        ]


@pytest.mark.asyncio
async def test_cancelled_message_map_dispatches_nothing_and_scanner_recovers(tmp_path, monkeypatch):
    scanner, adapter = scanner_for(tmp_path, '[item.transform("clean") for item in messages]')
    started = asyncio.Event()
    release = asyncio.Event()
    original = expression_module.apply_content_transform

    async def paused_transform(*args, **kwargs):
        started.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(expression_module, "apply_content_transform", paused_transform)
    task = asyncio.create_task(scanner.scan_data_async(CONVERSATION))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert adapter.classification_messages == []
    monkeypatch.setattr(expression_module, "apply_content_transform", original)

    result = await scanner.scan_data_async(CONVERSATION)

    assert result.is_complete
    assert len(adapter.classification_messages) == 1
