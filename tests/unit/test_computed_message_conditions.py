# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Computed message values run complete conditions in an isolated conversation frame."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from operator import eq, ge, gt, le, lt, ne

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.engine.events import build_events
from quarry.policy_serialization import _artifact_from_wire, _artifact_to_wire
from quarry.types.context import ScanContext
from tests.unit.test_adapter_transform_dsl import Adapter, compile_condition, configured


class FoldingAdapter(Adapter):
    transform_parts = frozenset({"text"})

    def transform(self, config, part, *, timeout=None):
        self.parts.append(part)
        return part["text"].casefold()


PAYLOAD = {
    "messages": [
        {"role": "user", "content": "block"},
        {"role": "assistant", "content": "safe"},
    ]
}


def prepare(tmp_path, condition, prepared, *, adapter=None):
    config, backend, adapter = configured(tmp_path, adapter=adapter or FoldingAdapter())
    policy = compile_condition(config, backend, condition)
    if prepared:
        policy = load_compiled_policy(
            policy.to_bytes(), runtime_config=config, model_inference_backend=backend
        )
    return policy, adapter


SOURCES = [
    ("(messages[:1] + messages[1:])", False, False),
    ('(messages[:1] + messages[1:]).transform("convert")', False, False),
    ('(messages[:1].transform("convert") + messages[1:])', False, False),
    ("(messages if true else messages[:0])", False, False),
    ('(messages if true else messages[:0]).transform("convert")', False, False),
    ('[item for item in messages if item.role == "user"]', True, False),
    ('[item for item in messages if item.role == "user"].transform("convert")', True, False),
    ('messages.transform("convert")[:1]', True, False),
    ('(message("user", "block") + messages)', False, False),
    ("(messages[:0] if true else messages)", True, True),
]


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("operation", ["any", "all", "count", "len"])
@pytest.mark.parametrize("source,all_match,empty", SOURCES)
def test_full_conditions_over_message_value_shapes(
    tmp_path, prepared, operation, source, all_match, empty
):
    if operation in {"any", "all"}:
        condition = f'{operation} m in ({source}) where m.content contains "block"'
        expected = all_match if operation == "all" else not empty
    elif operation == "count":
        condition = f'count([m for m in ({source}) if m.content contains "block"]) > 0'
        expected = not empty
    else:
        condition, expected = f"len(({source})) > 0", not empty
    policy, _ = prepare(tmp_path, condition, prepared)
    result = policy.scan(PAYLOAD, explain=True)
    assert result.is_complete
    assert result.rule_matches == (["R"] if expected else [])


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("operation", ["any", "all", "count"])
@pytest.mark.parametrize("called,empty", [(False, False), (True, False), (False, True)])
def test_root_facts_keep_the_parent_conversation_and_short_circuit_contract(
    tmp_path, prepared, operation, called, empty
):
    source = f'messages[:{0 if empty else 1}].transform("convert")'
    body = 'called_tool("lookup") and m.content contains "block"'
    condition = (
        f"count([m for m in {source} if {body}]) > 0"
        if operation == "count"
        else f"{operation} m in {source} where {body}"
    )
    policy, adapter = prepare(tmp_path, condition, prepared)
    payload = {"messages": list(PAYLOAD["messages"])}
    if called:
        payload["messages"].append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "outside",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            }
        )
    result = policy.scan(payload)
    assert result.is_complete
    expected = (empty or called) if operation == "all" else called and not empty
    assert result.rule_matches == (["R"] if expected else [])
    assert len(adapter.parts) == (0 if empty or (not called and operation != "all") else 1)


@pytest.mark.parametrize("prepared", [False, True])
def test_same_value_temporal_body_uses_child_order_and_one_transformation(tmp_path, prepared):
    source = 'messages.transform("convert")'
    condition = (
        f'any m in {source} where (m.content contains "safe") '
        f'after (any n in {source} where n.content contains "block")'
    )
    policy, adapter = prepare(tmp_path, condition, prepared)
    forward = policy.scan(PAYLOAD)
    backward = policy.scan({"messages": list(reversed(PAYLOAD["messages"]))})
    assert forward.is_complete and backward.is_complete
    assert forward.rule_matches == ["R"] and backward.rule_matches == []
    assert len(adapter.parts) == 4


@pytest.mark.parametrize("prepared", [False, True])
def test_folded_computed_branch_preserves_resource_preparation_without_dispatch(tmp_path, prepared):
    policy, adapter = prepare(
        tmp_path,
        'true or (any m in messages.transform("convert") where m.content contains "block")',
        prepared,
    )
    assert policy.scan(PAYLOAD).rule_matches == ["R"]
    assert adapter.parts == []
    assert policy._scanner.resource_requirements["transforms"]


@pytest.mark.parametrize("prepared", [False, True])
def test_child_explain_distinguishes_frame_position_and_original_message(tmp_path, prepared):
    policy, _ = prepare(
        tmp_path,
        'any m in messages[1:].transform("convert") where m.content contains "safe"',
        prepared,
    )
    without = policy.scan(PAYLOAD)
    result = policy.scan(PAYLOAD, explain=True)
    assert result.is_complete and result.rule_matches == without.rule_matches == ["R"]
    (value,) = result.explanations[0].to_dict()["value_conditions"]
    assert (
        result.explanations[0].atoms[0].description == "condition over a computed message selection"
    )
    assert value["minimized"] is False
    assert value["condition"]["witness"][0]["position"] == 0
    assert value["origins"][0]["position"] == 0
    assert value["origins"][0]["message_index"] == 1
    assert value["condition"]["witness"][0]["excerpt"] == "safe"


def test_native_explain_shape_does_not_gain_empty_child_fields():
    result = compile_policy('rule "R" condition: true').scan(PAYLOAD, explain=True)
    assert "value_conditions" not in result.explanations[0].to_dict()


def test_folded_computed_condition_round_trips_without_a_live_dfa_atom():
    source = (
        'rule "R" condition: '
        '(any m in (messages + messages) where m.content contains "x") or '
        'count([n for n in messages if n.content contains "y"]) >= 0'
    )
    policy = compile_policy(source)

    restored = load_compiled_policy(policy.to_bytes())

    assert restored.to_bytes() == policy.to_bytes()


@pytest.mark.asyncio
async def test_decision_paths_never_retain_child_text_or_deferred_payload(tmp_path):
    secret = "private-child-text"
    policy, _ = prepare(
        tmp_path, 'any m in messages.transform("convert") where m.content contains "private"', True
    )
    (result,) = await policy.scan_many_async(
        [{"messages": [{"role": "user", "content": secret}]}], decision_paths=True
    )
    assert result.is_complete and result.rule_matches == ["R"]
    assert secret not in repr(result.decision_paths)
    for record in result.decision_paths.values():
        for check in record.checks:
            assert secret not in json.dumps(asdict(check))
            assert check.evidence.value_details is None


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize(
    "index,word,matched", [(0, "block", True), (-1, "safe", True), (0, "safe", False)]
)
def test_same_value_indexed_reads_are_relative_to_the_child_frame(
    tmp_path, prepared, index, word, matched
):
    source = 'messages.transform("convert")'
    policy, adapter = prepare(
        tmp_path,
        f'any m in {source} where {source}[{index}].content contains "{word}"',
        prepared,
    )
    result = policy.scan(PAYLOAD)
    assert result.is_complete and result.rule_matches == (["R"] if matched else [])
    assert len(adapter.parts) == 2


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize(
    "operator,compare", [("==", eq), ("!=", ne), (">", gt), (">=", ge), ("<", lt), ("<=", le)]
)
@pytest.mark.parametrize("bound", [-1, -0.5, 0.5, 1.5, 2.5])
def test_len_compares_signed_fractional_bounds_without_skipping_value_evaluation(
    tmp_path, prepared, operator, compare, bound
):
    policy, adapter = prepare(
        tmp_path, f'len(messages.transform("convert")) {operator} {bound}', prepared
    )
    result = policy.scan(PAYLOAD)
    assert result.is_complete
    assert result.rule_matches == (["R"] if compare(2, bound) else [])
    assert len(adapter.parts) == 2


@pytest.mark.parametrize(
    "forgery", ["scalar", "binding", "parameter", "child-schema", "child-cost", "missing-plan"]
)
def test_prepared_child_frame_rejects_inconsistent_source_and_contracts(tmp_path, forgery):
    policy, _ = prepare(
        tmp_path, 'any m in messages.transform("convert") where m.content contains "block"', False
    )
    wire = _artifact_to_wire(policy._scanner.compiled_rules["R"])
    plan = wire["computed_conditions"][0]
    if forgery in {"scalar", "binding", "parameter"}:
        plan[1] = {
            "kind": {"scalar": "literal", "binding": "binding", "parameter": "parameter"}[forgery],
            "value": True if forgery == "scalar" else "missing",
            "children": [],
        }
    elif forgery == "child-schema":
        plan[2]["parameter_schema"].append({"name": "missing", "type": "number", "binding": "scan"})
    elif forgery == "child-cost":
        for atom in plan[2]["atom_specs"]:
            atom["cost"] = "llm"
    else:
        wire["computed_conditions"] = []
    with pytest.raises(ValueError, match="computed"):
        _artifact_from_wire(wire)


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize(
    "condition,matched",
    [
        ('any m in messages.transform("convert") where false', False),
        ('all m in messages.transform("convert") where true', True),
        ('len(messages.transform("convert")) > 0', True),
    ],
)
def test_explain_does_not_read_ignored_opaque_content(tmp_path, prepared, condition, matched):
    policy, adapter = prepare(tmp_path, condition, prepared)
    payload = {"messages": [{"role": "alien", "content": "opaque"}]}
    for explain in [False, True]:
        result = policy.scan(payload, explain=explain)
        assert result.is_complete
        assert result.rule_matches == (["R"] if matched else [])
    assert adapter.parts == []


class OrdinalAdapter(FoldingAdapter):
    def transform(self, config, part, *, timeout=None):
        return super().transform(config, part, timeout=timeout) + f":{len(self.parts)}"


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize(
    "form,matched",
    [
        ("repeated-domain", True),
        ("same-element-two-questions", False),
        ("two-comprehensions", False),
    ],
)
def test_bound_source_values_belong_to_each_child_question(tmp_path, prepared, form, matched):
    source = 'message("user", outer.content).transform("convert")'
    first = f'any inner in {source} where inner.content contains "block"'
    second = f'any inner in {source} where inner.content contains ":1"'
    if form == "repeated-domain":
        condition = f"all outer in (messages + messages) where {first}"
    elif form == "same-element-two-questions":
        condition = f"all outer in messages where ({first}) and ({second})"
    else:
        condition = (
            f"(any outer in messages where {first}) and (any outer in messages where {second})"
        )
    policy, adapter = prepare(tmp_path, condition, prepared, adapter=OrdinalAdapter())
    result = policy.scan({"messages": [PAYLOAD["messages"][0]]})
    assert result.is_complete and result.rule_matches == (["R"] if matched else [])
    assert len(adapter.parts) == 2


@pytest.mark.parametrize("prepared", [False, True])
def test_completed_event_sources_do_not_accumulate_in_the_rule_memo(tmp_path, prepared):
    policy, adapter = prepare(
        tmp_path,
        'all outer in messages where any inner in message("user", outer.content).transform("convert") where inner.content contains "block"',
        prepared,
    )
    result = policy.scan({"messages": [PAYLOAD["messages"][0]] * 300})
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(adapter.parts) == 300


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("transform_item,parts", [(False, 2), (True, 4)])
def test_bound_source_keeps_independent_parent_values_stable(
    tmp_path, prepared, transform_item, parts
):
    item = 'message("user", outer.content)' + ('.transform("convert")' if transform_item else "")
    policy, adapter = prepare(
        tmp_path,
        f'all outer in messages where any inner in (messages.transform("convert") + {item}) where inner.content contains "block"',
        prepared,
    )
    result = policy.scan(PAYLOAD)
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(adapter.parts) == parts


@pytest.mark.asyncio
async def test_cancelled_bound_source_releases_its_pending_memo_entry(tmp_path):
    from quarry.integrations.model_expression import evaluate_scoped_model_expression

    started = asyncio.Event()

    class WaitingAdapter(FoldingAdapter):
        async def transform(self, config, part, *, timeout=None):
            started.set()
            await asyncio.Event().wait()

    policy, _ = prepare(
        tmp_path,
        'all outer in messages where any inner in message("user", outer.content).transform("convert") where inner.content contains "block"',
        False,
        adapter=WaitingAdapter(),
    )
    context = ScanContext(PAYLOAD["messages"], resource_manager=policy._scanner.resource_manager)
    (plan,) = policy._scanner.compiled_rules["R"].computed_conditions.values()
    task = asyncio.create_task(
        evaluate_scoped_model_expression(plan.source, context, event=build_events(context)[0])
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    assert len(context.message_value_memo._entries) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert context.message_value_memo._entries == {}
    assert context.message_value_memo._bytes == 0


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("form", ["large-domain", "two-fields", "repeated-domain"])
def test_native_bound_transform_reads_have_separate_question_scopes(tmp_path, prepared, form):
    read = 'm.transform("convert")[0].content'
    if form == "two-fields":
        condition = f'all m in messages where {read} contains ":1" and {read} contains ":2"'
        size, callbacks = 1, 2
    else:
        domain = "(messages + messages)" if form == "repeated-domain" else "messages"
        condition = f'all m in {domain} where {read} contains "block"'
        size = 1 if form == "repeated-domain" else 300
        callbacks = 2 if form == "repeated-domain" else 300
    policy, adapter = prepare(tmp_path, condition, prepared, adapter=OrdinalAdapter())
    result = policy.scan({"messages": [PAYLOAD["messages"][0]] * size})
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(adapter.parts) == callbacks
