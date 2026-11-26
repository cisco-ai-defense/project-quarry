# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Maps retain collection entries until an operation asks for combined messages."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.cache.message_values import _retained_bytes
from quarry.engine.atoms import ModelExpression, validate_model_expression
from quarry.types.exceptions import RuleValidationError
from quarry.types.messages import Messages
from tests.unit.test_computed_message_conditions import PAYLOAD, prepare
from tests.unit.test_message_maps import scanner_for

MAP = '[messages.transform("convert") for item in messages]'


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("function", ["len", "count"])
@pytest.mark.parametrize(
    "value,expected", [(MAP, 2), (f"{MAP}[:1]", 1), (f"{MAP}[:0]", 0), (f"{MAP}[0]", 2)]
)
def test_map_cardinality_counts_entries_before_input_conversion(
    tmp_path, prepared, function, value, expected
):
    policy, adapter = prepare(tmp_path, f"{function}({value}) == {expected}", prepared)
    result = policy.scan(PAYLOAD, explain=True)
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(adapter.parts) == 2


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("operation", ["any", "all", "count"])
def test_quantifiers_consume_the_combined_messages_instead_of_group_objects(
    tmp_path, prepared, operation
):
    condition = f'{operation} m in {MAP} where m.content contains "block"'
    expected = operation != "all"
    if operation == "count":
        condition = f'count([m for m in {MAP} if m.content contains "block"]) == 2'
    policy, adapter = prepare(tmp_path, condition, prepared)
    result = policy.scan(PAYLOAD)
    assert result.is_complete and result.rule_matches == (["R"] if expected else [])
    assert len(adapter.parts) == 2


@pytest.mark.parametrize("prepared", [False, True])
def test_same_value_size_inside_a_quantifier_reads_the_converted_frame(tmp_path, prepared):
    policy, adapter = prepare(tmp_path, f"any m in {MAP} where len({MAP}) == 4", prepared)
    result = policy.scan(PAYLOAD)
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(adapter.parts) == 2


@pytest.mark.parametrize(
    "value,count",
    [
        ("[messages for item in messages]", 4),
        ("[messages for item in messages][0]", 2),
        ("[messages for item in messages][:1]", 2),
    ],
)
def test_model_input_conversion_combines_only_selected_entries(tmp_path, value, count):
    scanner, adapter = scanner_for(tmp_path, value)
    policy = compile_policy(
        f'rule "R" condition: model("judge").classify({value}) > 0.5',
        runtime_config=scanner.config,
        model_inference_backend=scanner.model_inference_backend,
    )
    loaded = load_compiled_policy(
        policy.to_bytes(),
        runtime_config=scanner.config,
        model_inference_backend=scanner.model_inference_backend,
    )
    result = loaded.scan(PAYLOAD)
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(adapter.classification_messages[-1]) == count
    assert adapter.classification_messages[-1] == (PAYLOAD["messages"] * 2)[:count]


@pytest.mark.parametrize("prepared", [False, True])
def test_nested_maps_iterate_collections_before_model_input_conversion(tmp_path, prepared):
    value = '[[messages.transform("convert") for inner in messages] for outer in messages]'
    policy, adapter = prepare(tmp_path, f"len({value}) == 2 and len({value}[0]) == 4", prepared)
    result = policy.scan(PAYLOAD)
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(adapter.parts) == 2


@pytest.mark.parametrize("prepared", [False, True])
def test_indexing_keeps_a_whole_collection_with_canonical_fields_and_provenance(tmp_path, prepared):
    policy, adapter = prepare(
        tmp_path, f'{MAP}[0].user_messages[0].content contains "block"', prepared
    )
    result = policy.scan(PAYLOAD, explain=True)
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(adapter.parts) == 2


def test_a_group_index_is_not_mistyped_as_an_individual_message(tmp_path):
    with pytest.raises(RuleValidationError, match="no field 'content'"):
        prepare(tmp_path, f'{MAP}[0].content contains "block"', False)


def test_group_views_are_immutable_and_retain_the_selected_canonical_values():
    from quarry.types.message_groups import MessageGroups

    source = Messages.from_openai(PAYLOAD["messages"])
    groups = MessageGroups((source, source[:1]), source[:0])
    assert len(groups) == 2 and groups[0] is source
    assert groups[-1][0] is source[0]
    assert groups[:1][0] is source and groups[:0]._empty.shares_root_with(source)
    with pytest.raises(FrozenInstanceError):
        groups._items = ()
    with pytest.raises(TypeError):
        MessageGroups([source], source[:0])


@pytest.mark.parametrize("kind", ["length", "flatten_messages"])
def test_prepared_collection_operations_reject_scalar_sources(kind):
    with pytest.raises(ValueError, match="requires"):
        validate_model_expression(ModelExpression(kind, None, (ModelExpression("literal", True),)))


@pytest.mark.asyncio
async def test_retained_group_sizing_never_serializes_or_reads_message_fields(monkeypatch):
    from quarry.types.message_groups import MessageGroups

    source = Messages.from_openai(
        [{"role": "alien", "content": "opaque"}, {"role": "user", "content": "x" * 1000}]
    )
    groups = MessageGroups((source, source[:1]), source[:0])

    def forbidden(*args, **kwargs):
        raise AssertionError("group sizing must inspect immutable storage only")

    monkeypatch.setattr(Messages, "to_openai_messages", forbidden)
    assert await _retained_bytes(groups, 1_000_000) > 1000
    assert await _retained_bytes(groups, 100) == 101


@pytest.mark.parametrize("function", ["len", "count"])
@pytest.mark.parametrize("form", ["bound", "indexed", "constructed"])
def test_whole_message_measurements_are_not_implicitly_converted_to_collection_sizes(
    tmp_path, function, form
):
    condition = (
        f"any item in messages where {function}(item) == 1"
        if form == "bound"
        else f"{function}(messages[0]) == 1"
        if form == "indexed"
        else f'{function}(message("user", "x")[0]) == 1'
    )
    with pytest.raises(RuleValidationError, match="whole message"):
        prepare(tmp_path, condition, False)
