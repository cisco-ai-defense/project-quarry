# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Actual constructor evaluation keeps source occurrence, binding and frame identity."""

import asyncio
import gc
import weakref
from dataclasses import asdict

import pytest

from quarry.cache.message_values import _retained_bytes
from quarry.engine.atoms import ModelExpression
from quarry.engine.events import build_events
from quarry.integrations.content_transforms import apply_content_transform
from quarry.integrations.model_expression import (
    _READ_CONSTRUCTION_SCOPE,
    evaluate_model_expression,
    scoped_model_expression,
)
from quarry.integrations.read_coverage import ReadCoverage
from quarry.integrations.read_usage import ReadUsageLimits, model_read_usage
from quarry.types.context import ScanContext
from quarry.types.transforms import ContentTransform, TransformOperation
from tests.unit.test_scanner_read_usage_lifetimes import policy


def literal(value):
    return ModelExpression("literal", value)


def turn(content=None):
    return ModelExpression(
        "message", children=(literal("user"), literal("same") if content is None else content)
    )


def interval(session, message):
    value = session.text_measurement(message, message.source_content)
    assert value.state == "measured"
    return value.intervals[0]


def context():
    return ScanContext(
        [{"role": "user", "content": "same"}, {"role": "assistant", "content": "same"}]
    )


async def test_repeated_fresh_constructors_share_one_occurrence_inside_the_same_frame():
    frame, site = context(), turn()
    with model_read_usage() as session:
        one = await evaluate_model_expression(site, frame)
        two = await evaluate_model_expression(site, frame)
        assert one[0] is not two[0]
        assert interval(session, one[0]) == interval(session, two[0])
        coverage = ReadCoverage()
        coverage.add([interval(session, one[0]), interval(session, two[0])])
        assert coverage.snapshot().characters == 4


async def test_concurrent_evaluations_of_the_same_constructor_share_metadata_without_sharing_values():
    frame, site = context(), turn()
    with model_read_usage() as session:
        values = await asyncio.gather(*(evaluate_model_expression(site, frame) for _ in range(20)))
        assert len({id(value[0]) for value in values}) == 20
        assert len({interval(session, value[0]).source for value in values}) == 1


async def test_equal_authored_text_from_other_sites_or_child_frames_stays_distinct():
    frame, site, other = context(), turn(), turn()
    child = frame.for_messages(frame.conversation.messages[:1])
    with model_read_usage() as session:
        first = await evaluate_model_expression(site, frame)
        second = await evaluate_model_expression(other, frame)
        third = await evaluate_model_expression(site, child)
        assert len({interval(session, item[0]).source for item in (first, second, third)}) == 3


async def test_mapped_authored_turns_keep_equal_text_at_different_item_positions_apart():
    frame, site = context(), turn()
    expression = ModelExpression(
        "message_map", "item", (ModelExpression("messages"), literal(True), site)
    )
    with model_read_usage() as session:
        value = await evaluate_model_expression(expression, frame)
        assert [part[0].content for part in value] == ["same", "same"]
        assert interval(session, value[0][0]).source != interval(session, value[1][0]).source


async def test_map_ancestors_distinguish_shared_constructor_nodes_with_the_same_positions():
    frame, site = context(), turn()
    expressions = [
        ModelExpression("message_map", "item", (ModelExpression("messages"), literal(True), site))
        for _ in range(2)
    ]
    with model_read_usage() as session:
        # Bind the actual ancestor scopes explicitly: equal whole-map expressions are
        # structurally memoized by the native evaluator before their bodies are entered.
        outputs = [
            await evaluate_model_expression(
                site, frame, bindings={_READ_CONSTRUCTION_SCOPE: (ancestor, 0)}
            )
            for ancestor in expressions
        ]
        assert interval(session, outputs[0][0]).source != interval(session, outputs[1][0]).source


async def test_temporary_counter_zero_in_separate_native_child_frames_does_not_collide():
    frame = context()
    site = turn(ModelExpression("field", "role", (ModelExpression("binding", "item"),)))
    child = frame.for_messages(frame.conversation.messages[:1])
    with model_read_usage() as session:
        results = []
        for current in (frame, child):
            async with scoped_model_expression(site, current, event=build_events(current)[0]) as (
                value,
                _,
            ):
                results.append(interval(session, value[0]))
        assert results[0].source != results[1].source


async def test_repeated_temporary_evaluations_in_one_frame_have_different_scope_counters():
    frame = context()
    site = turn(ModelExpression("field", "role", (ModelExpression("binding", "item"),)))
    with model_read_usage() as session:
        results = []
        for _ in range(2):
            async with scoped_model_expression(site, frame, event=build_events(frame)[0]) as (
                value,
                _,
            ):
                results.append(interval(session, value[0]))
        assert results[0].source != results[1].source


async def test_constructor_copy_of_selected_content_preserves_original_character_coordinates():
    frame = context()
    raw = frame.conversation.messages
    with model_read_usage() as session:
        # Rebuild the cut while the observer is active so its exact lineage is known.
        cut = raw._select_ranges((0,), {0: (1, 4)})
        source = ModelExpression(
            "field", "content", (ModelExpression("index", 0, (ModelExpression("messages"),)),)
        )
        selected = frame.for_messages(cut)
        copied = await evaluate_model_expression(turn(source), selected)
        assert interval(session, copied[0]) == interval(session, cut[0])
        assert (interval(session, copied[0]).start, interval(session, copied[0]).end) == (1, 4)
        assert interval(session, copied[0]).source == interval(session, raw[0]).source


async def test_transformed_constructed_variants_reuse_their_own_axis_and_slice_offsets():
    frame, site = context(), turn(literal("Straße"))
    plan = ContentTransform((TransformOperation("casefold"),))
    with model_read_usage() as session:
        one = await evaluate_model_expression(site, frame)
        two = await evaluate_model_expression(site, frame)
        left, right = (
            await apply_content_transform(one, plan),
            await apply_content_transform(two, plan),
        )
        assert interval(session, left[0]) == interval(session, right[0])
        first = left._select_ranges((0,), {0: (1, 5)})
        second = right._select_ranges((0,), {0: (3, 7)})
        coverage = ReadCoverage()
        coverage.add([interval(session, first[0]), interval(session, second[0])])
        assert coverage.snapshot().characters == 6


async def test_accounting_does_not_change_constructed_retained_value_bytes():
    frame, site = context(), turn()
    baseline = await evaluate_model_expression(site, frame)
    before = await _retained_bytes(baseline, 10**9)
    with model_read_usage() as session:
        value = await evaluate_model_expression(site, frame)
        interval(session, value[0])
        assert await _retained_bytes(value, 10**9) == before
        assert set(value[0].__pydantic_private__) == set(baseline[0].__pydantic_private__)


async def test_live_accounting_retains_no_constructor_literals_frames_or_finished_messages():
    with model_read_usage() as session:
        frame, site = context(), turn(literal("sensitive synthetic literal"))
        value = await evaluate_model_expression(site, frame)
        references = [
            weakref.ref(owner) for owner in (frame, frame.message_value_memo, site, value[0])
        ]
        interval(session, value[0])
        del frame, site, value
        gc.collect()
        assert all(reference() is None for reference in references)
        assert session._text_inputs.registry.snapshot().owners == 0


async def test_finished_map_scope_does_not_retain_its_ancestor_expression_or_native_memo():
    with model_read_usage() as session:
        frame, site = context(), turn()
        expression = ModelExpression(
            "message_map", "item", (ModelExpression("messages"), literal(True), site)
        )
        value = await evaluate_model_expression(expression, frame)
        references = [weakref.ref(owner) for owner in (expression, site, frame.message_value_memo)]
        interval(session, value[0][0])
        frame.close_message_values()
        del value, frame, site, expression
        gc.collect()
        assert all(reference() is None for reference in references)
        assert session._text_inputs.registry.snapshot().owners == 0


@pytest.mark.parametrize("limit", ["owners", "work"])
async def test_accounting_scope_exhaustion_keeps_native_construction_and_decision_budget(limit):
    frame, site = context(), turn()
    limits = ReadUsageLimits(**{f"max_input_{limit}": 1})
    with model_read_usage(limits) as session:
        value = await evaluate_model_expression(site, frame)
        assert value.to_openai_messages() == [{"role": "user", "content": "same"}]
        assert session.text_measurement(value[0], value[0].source_content).state == "limit_exceeded"


async def test_optional_constructor_metadata_failure_preserves_native_output(monkeypatch):
    frame, site = context(), turn()
    with model_read_usage() as session:

        def fail(*_args):
            raise RuntimeError("sensitive synthetic error")

        monkeypatch.setattr(session._text_inputs.registry, "constructed", fail)
        value = await evaluate_model_expression(site, frame)
        assert value.to_openai_messages() == [{"role": "user", "content": "same"}]
        assert session.text_measurement(value[0], value[0].source_content).state == "failed"
        assert session._text_inputs._failure == "failed"


def test_weak_expression_ownership_does_not_add_serialized_fields_or_change_value_identity():
    site, equal = turn(), turn()
    assert site == equal and hash(site) == hash(equal)
    assert set(asdict(site)) == {"kind", "value", "children"}
    reference = weakref.ref(site)
    del site
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("value", [None, "", "same"])
async def test_omitted_and_empty_constructor_turns_keep_hook_alignment(value):
    frame = context()
    first = turn(literal(value))
    chain = ModelExpression("add_message", children=(first, literal("assistant"), literal("tail")))
    with model_read_usage() as session:
        one = await evaluate_model_expression(chain, frame)
        two = await evaluate_model_expression(chain, frame)
        assert len(one) == (1 if value is None else 2)
        assert one[-1].content == "tail"
        assert [interval(session, item) for item in one] == [
            interval(session, item) for item in two
        ]
        if value == "":
            assert interval(session, one[0]).start == interval(session, one[0]).end == 0


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("shape", ["authored", "copied", "mapped", "computed"])
async def test_real_policy_constructors_preserve_verdict_wire_and_provider_work(
    tmp_path, monkeypatch, prepared, shape
):
    data = {"messages": [{"role": "user", "content": "same"}]}
    if shape == "authored":
        condition = 'model("judge").score(message("user", "same"), "synthetic") > 0.5'
    elif shape == "copied":
        condition = 'model("judge").score(message("user", messages[0].content), "synthetic") > 0.5'
    elif shape == "mapped":
        condition = 'model("judge").score([message("user", m.content) for m in messages], "synthetic") > 0.5'
    else:
        condition = 'any m in (message("user", "same") + messages) where model("judge").score(message("user", m.content), "synthetic") > 0.5'
    compiled, adapter = policy(tmp_path, f'rule "Outer" condition: {condition}', prepared=prepared)
    baseline = await compiled.scan_async(data)
    assert baseline.is_complete and baseline.rule_matches == ["Outer"]
    calls, wires = adapter.score_calls, list(adapter.score_messages)
    recorded = []
    with model_read_usage() as session:
        original = session._text_inputs.constructed

        def observe(frame, site, scope, target, text, source):
            original(frame, site, scope, target, text, source)
            recorded.append(interval(session, target))

        monkeypatch.setattr(session._text_inputs, "constructed", observe)
        result = await compiled.scan_async(data)
    assert result.is_complete and result.rule_matches == baseline.rule_matches
    assert recorded and all(item.end - item.start == 4 for item in recorded)
    # The backend may reuse a physical reply. Accounting never adds provider work.
    assert adapter.score_calls == calls and adapter.score_messages == wires
