# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Semantic non-body input fields, native copy lineage and guarded projections."""

import gc
import weakref

import pytest

from quarry.integrations.content_transforms import apply_content_transform
from quarry.integrations.read_inputs import collect_field
from quarry.integrations.read_operations import model_read, record_model_input
from quarry.integrations.read_usage import ReadUsageLimits, model_read_usage
from quarry.types.messages import Messages, ToolCall
from quarry.types.transforms import ContentTransform, TransformOperation
from tests.unit.test_model_read_operations import backend, rows

BODY, REFUSAL, NAME, ARGUMENTS = "body", "refused", "lookup", '{"q":"same"}'
EXPECTED = sum(map(len, (BODY, REFUSAL, NAME, ARGUMENTS)))


def assistant():
    return Messages.from_openai(
        [
            {
                "role": "assistant",
                "content": BODY,
                "refusal": REFUSAL,
                "tool_calls": [
                    {
                        "id": "synthetic-call",
                        "type": "function",
                        "function": {"name": NAME, "arguments": ARGUMENTS},
                    }
                ],
            }
        ]
    )


def observe(value):
    with model_read("reader", "classify"):
        record_model_input(value)


async def test_real_backend_counts_call_fields_without_decoding_opaque_arguments(
    tmp_path, monkeypatch
):
    service, _ = backend(tmp_path)
    value = assistant()
    call = value[0].tool_calls[0]
    monkeypatch.setattr(ToolCall, "arguments", property(lambda _: pytest.fail("decoded arguments")))
    with model_read_usage() as session:
        await service.model_classify_messages("reader", value)
    measured = rows(session)[0]["coverage"]["characters"]
    assert measured["value"] == EXPECTED and measured["measurement_states"] == {"measured": 1}
    assert call._arguments_cache is None


@pytest.mark.parametrize("copy", ["shallow", "deep", "cut", "transformed", "combined"])
def test_preserving_native_variants_do_not_duplicate_non_body_fields(copy):
    value = assistant()
    with model_read_usage() as session:
        observe(value)
        if copy == "shallow":
            clone = value._with_position_overrides({0: value[0].model_copy()})
        elif copy == "deep":
            clone = value._with_position_overrides({0: value[0].model_copy(deep=True)})
        elif copy == "cut":
            clone = value._select_ranges((0,), {0: (2, 4)})
        elif copy == "transformed":
            # This is the native variant constructor used after a validated transform.
            message = value[0]._with_transformed_content({None: "BODY"}, ())
            clone = value._with_position_overrides({0: message})
        else:
            clone = Messages.combine((value._with_position_overrides({0: value[0].model_copy()}),))
        observe(clone)
    assert rows(session)[0]["coverage"]["characters"]["value"] == EXPECTED + (
        4 if copy == "transformed" else 0
    )


async def test_cut_before_text_transform_preserves_auxiliary_field_axes():
    value = assistant()
    with model_read_usage() as session:
        observe(value)
        suffix = value._select_ranges((0,), {0: (2, 4)})
        changed = await apply_content_transform(
            suffix, ContentTransform((TransformOperation("replace", "dy", "DY"),))
        )
        observe(changed)
    assert rows(session)[0]["coverage"]["characters"]["value"] == EXPECTED + 2


@pytest.mark.parametrize("field", ["raw_arguments", "name", "refusal"])
def test_legacy_field_and_full_model_inputs_share_native_axes(field):
    value = assistant()
    with model_read_usage() as session:
        observe(value)
        owner = value[0] if field == "refusal" else value[0].tool_calls[0]
        text = {"raw_arguments": ARGUMENTS, "name": NAME, "refusal": REFUSAL}[field]
        measurement = collect_field(session, owner, field, text, "tool_calls", 0)
        with model_read("reader", "classify") as read:
            read._add("input", measurement)
    assert rows(session)[0]["coverage"]["characters"]["value"] == EXPECTED


def test_structural_call_links_change_cuts_without_counting_as_text():
    value = assistant()
    with model_read_usage() as session:
        observe(value)
        replacement = value[0].tool_calls[0].model_copy(update={"id": "another-synthetic-call"})
        clone = value._with_position_overrides(
            {0: value[0].model_copy(update={"tool_calls": (replacement,)})}
        )
        observe(clone)
    row = rows(session)[0]
    assert row["unique_cuts"] == 2 and row["coverage"]["characters"]["value"] == EXPECTED


def test_composed_call_response_graph_clones_keep_original_field_coverage():
    wire = assistant()[0].to_openai_dict()
    value = Messages.from_openai(
        [wire, {"role": "tool", "tool_call_id": "synthetic-call", "content": "result"}]
    )
    with model_read_usage() as session:
        observe(value)
        changed = value._with_position_overrides({0: value[0].model_copy()})
        combined = Messages.combine((changed[:1], value[1:]))
        assert combined[0].tool_calls[0] is not value[0].tool_calls[0]
        observe(combined)
    assert rows(session)[0]["coverage"]["characters"]["value"] == EXPECTED + len("result")


def test_equal_call_text_in_distinct_occurrences_stays_distinct():
    one, two = assistant(), assistant()
    with model_read_usage() as session:
        observe(one)
        observe(two)
    assert rows(session)[0]["coverage"]["characters"]["value"] == 2 * EXPECTED


def test_call_and_refusal_metadata_does_not_retain_native_graphs():
    with model_read_usage() as session:
        value = assistant()
        message, call = weakref.ref(value[0]), weakref.ref(value[0].tool_calls[0])
        observe(value)
        del value
        gc.collect()
        assert message() is None and call() is None
    assert rows(session)[0]["coverage"]["characters"]["value"] == EXPECTED


def test_media_payload_variants_affect_cut_identity_without_counting_encoded_data():
    value = Messages.from_openai(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": BODY},
                    {"type": "image_url", "image_url": {"url": "data:synthetic-one"}},
                ],
            }
        ]
    )
    with model_read_usage() as session:
        observe(value)
        wire = value[0].to_openai_dict()
        wire["content"][1]["image_url"]["url"] = "data:synthetic-two"
        observe(value.rewritten([wire]))
    row = rows(session)[0]
    assert row["unique_cuts"] == 2 and row["coverage"]["characters"]["value"] == len(BODY)


def test_non_body_fields_obey_independent_metadata_limits():
    with model_read_usage(ReadUsageLimits(max_input_work=1)) as session:
        observe(assistant())
    row = rows(session)[0]
    assert row["outcome"] == "succeeded"
    assert row["coverage"]["characters"]["value"] is None


@pytest.mark.parametrize("field", ["refusal", "name", "arguments"])
def test_changed_native_field_text_has_its_own_axis_without_duplicating_other_fields(field):
    value = assistant()
    replacement_text = "changed"
    with model_read_usage() as session:
        observe(value)
        if field == "refusal":
            changed = value[0].model_copy(update={"refusal": replacement_text})
        else:
            call = value[0].tool_calls[0]
            function = call.function.model_copy(update={field: replacement_text})
            call = call.model_copy(update={"function": function})
            changed = value[0].model_copy(update={"tool_calls": (call,)})
        observe(value._with_position_overrides({0: changed}))
    assert rows(session)[0]["coverage"]["characters"]["value"] == EXPECTED + len(replacement_text)


def test_a_nonfinal_native_cut_does_not_read_the_dropped_assistant_fields():
    value = assistant()
    with model_read_usage() as session:
        observe(value._select_ranges((0,), {0: (0, 2)}))
    measured = rows(session)[0]["coverage"]["characters"]
    assert measured["value"] == 2 and measured["measurement_states"] == {"measured": 1}


def test_media_position_between_text_ranges_affects_the_complete_input_cut():
    media = {"type": "image_url", "image_url": {"url": "data:synthetic"}}
    value = Messages.from_openai(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "ab"}, media, {"type": "text", "text": "cd"}],
            }
        ]
    )
    with model_read_usage() as session:
        observe(value)
        changed = value.rewritten(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        media,
                        {"type": "text", "text": "bcd"},
                    ],
                }
            ]
        )
        observe(changed)
    row = rows(session)[0]
    assert row["unique_cuts"] == 2 and row["coverage"]["characters"]["value"] == 4
