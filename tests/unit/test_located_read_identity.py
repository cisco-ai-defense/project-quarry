# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Located accounting follows exact runs, never an enclosing evidence span."""

import gc
import weakref
from array import array

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quarry.cache.message_values import _retained_bytes
from quarry.config import QuarryConfig
from quarry.input_registry import InputObject
from quarry.integrations.content_transforms import apply_content_transform
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.integrations.read_coverage import ReadCoverage
from quarry.integrations.read_identity import ReadTextRegistry
from quarry.integrations.read_usage import ReadUsageLimits, model_read_usage
from quarry.types.messages import Messages, ToolCall
from quarry.types.origins import LINEAR, WHOLE, Derived, Place, excerpt, joined
from quarry.types.results import InputText
from quarry.types.transforms import ContentTransform, TransformOperation
from tests.unit.model_fakes import ModelAdapter, model_entry, passthrough_builder
from tests.unit.test_nested_rule_runtime_contracts import _policy


def place(owner, field="body", *, start=0, section="memory.notes", item="same"):
    return Place(0, field, start, owner, InputText(section, item, ("title", "body")))


def project(texts, origin, role="user"):
    return Messages.message_parts(role, texts, origin)


def ranges(session, message):
    result = session.text_measurement(message, message.source_content)
    assert result.state == "measured"
    return result.intervals


def total(*intervals):
    coverage = ReadCoverage()
    coverage.add(intervals)
    return coverage.snapshot().characters


def test_repeated_field_projections_share_one_axis_across_roles_and_part_layouts():
    owner = InputObject({"title": "same", "body": "same"})
    with model_read_usage() as session:
        one = project(["same"], place(owner))
        two = project(["sa", "me"], place(owner), role="assistant")
        other = project(["same"], place(owner, "title"))
        first = ranges(session, one[0])
        assert ranges(session, two[0]) == first
        assert first[0].source != ranges(session, other[0])[0].source
        assert total(*first, *ranges(session, two[0])) == 4


@pytest.mark.parametrize("dimension", ["owner", "section", "item", "field"])
def test_equal_text_and_wire_locations_do_not_merge_different_source_occurrences(dimension):
    owner = InputObject({"title": "same", "body": "same"})
    other = InputObject({"title": "same", "body": "same"}) if dimension == "owner" else owner
    kwargs = (
        {"section": "archive.notes"}
        if dimension == "section"
        else {"item": "other"}
        if dimension == "item"
        else {"field": "title"}
        if dimension == "field"
        else {}
    )
    with model_read_usage() as session:
        first = project(["same"], place(owner))
        second = project(["same"], place(other, **kwargs))
        assert ranges(session, first[0])[0].source != ranges(session, second[0])[0].source


def test_ordered_fields_keep_individual_axes_and_reordering_does_not_double_count():
    owner = InputObject({"title": "ab", "body": "cd"})
    a, b = place(owner, "title"), place(owner)
    with model_read_usage() as session:
        first = project(["ab", "cd"], joined([("ab", a), ("cd", b)]))
        second = project(["cd", "ab"], joined([("cd", b), ("ab", a)]))
        left, right = ranges(session, first[0]), ranges(session, second[0])
        assert left == right[::-1] and len(left) == 2
        assert total(*left, *right) == 4


def disjoint(owner):
    base = place(owner)
    return joined([("ab", excerpt(base, 0, 2)), ("ij", excerpt(base, 8, 10))])


def test_disjoint_runs_stay_disjoint_when_evidence_reports_one_enclosing_span():
    owner = InputObject({"body": "abcdefghij"})
    origin = disjoint(owner)
    assert [(span.start, span.end) for span in origin.places(0, 4)] == [(0, 2), (8, 10)]
    with model_read_usage() as session:
        value = project(["ab", "ij"], origin)
        result = ranges(session, value[0])
        assert [(item.start, item.end) for item in result] == [(0, 2), (8, 10)]
        assert result[0].source == result[1].source and total(*result) == 4


def test_one_source_with_multiple_runs_does_not_inherit_the_evidence_envelope():
    owner = InputObject({"body": "abcdefghij"})
    origin = Derived(
        "abij", (place(owner),), array("q", (0, 2, 0, 0, 2, LINEAR, 2, 4, 0, 8, 10, LINEAR))
    )
    assert [(span.start, span.end) for span in origin.places(0, 4)] == [(0, 10)]
    with model_read_usage() as session:
        value = project(["abij"], origin)
        result = ranges(session, value[0])
        assert [(item.start, item.end) for item in result] == [(0, 2), (8, 10)]
        assert total(*result) == 4


def test_nested_slices_and_preserving_copies_keep_each_original_projected_cut():
    owner = InputObject({"body": "abcdefghij"})
    with model_read_usage() as session:
        value = project(["abij"], disjoint(owner))
        selected = value._select_ranges((0,), {0: (1, 3)})
        copied = selected[0].model_copy()
        result = ranges(session, copied)
        assert [(item.start, item.end) for item in result] == [(1, 2), (8, 9)]
        assert result == ranges(session, selected[0])
        assert total(*result, *ranges(session, value[0])) == 4


async def test_transform_variants_of_equivalent_projection_layouts_share_their_own_axis():
    owner = InputObject({"body": "ABCDEFGHIJ"})
    direct = Derived(
        "ABIJ", (place(owner),), array("q", (0, 2, 0, 0, 2, LINEAR, 2, 4, 0, 8, 10, LINEAR))
    )
    plan = ContentTransform((TransformOperation("casefold"),))
    with model_read_usage() as session:
        alternate = joined(
            [("AB", excerpt(place(owner), 0, 2)), ("IJ", excerpt(place(owner), 8, 10))]
        )
        first, second = project(["ABIJ"], direct), project(["AB", "IJ"], alternate)
        a, b = (
            await apply_content_transform(first, plan),
            await apply_content_transform(second, plan),
        )
        one = ranges(session, a[0])
        assert one == ranges(session, b[0]) and one[0].end == 4
        assert one[0].source != ranges(session, first[0])[0].source


@pytest.mark.parametrize("kind", [LINEAR, WHOLE])
def test_inexact_origin_runs_use_actual_variant_coordinates_instead_of_source_envelopes(kind):
    owner = InputObject({"body": "abcdefghij"})
    origin = Derived("WXYZ", (place(owner),), array("q", (0, 4, 0, 0, 10, kind)))
    with model_read_usage() as session:
        full = project(["WXYZ"], origin)
        cut = project(["YZ"], excerpt(origin, 2, 4))
        result = ranges(session, full[0])
        selected = ranges(session, cut[0])
        assert len(result) == len(selected) == 1
        assert (result[0].start, result[0].end) == (0, 4)
        assert (selected[0].start, selected[0].end) == (2, 4)
        assert result[0].source == selected[0].source and total(*result, *selected) == 4


def test_authored_gaps_count_their_actual_characters_and_preserve_cut_coordinates():
    owner = InputObject({"body": "abcdefghij"})
    origin = Derived(
        "ab::ij", (place(owner),), array("q", (0, 2, 0, 0, 2, LINEAR, 4, 6, 0, 8, 10, LINEAR))
    )
    with model_read_usage() as session:
        first, second = project(["ab::ij"], origin), project(["ab::ij"], origin)
        result = ranges(session, first[0])
        assert len(result) == 3 and total(*result, *ranges(session, second[0])) == 6
        cut = first._select_ranges((0,), {0: (1, 3)})
        assert [(item.start, item.end) for item in ranges(session, cut[0])] == [(1, 2), (2, 3)]


def test_canonical_chat_projection_shares_the_original_message_axis_without_content_cache_reads():
    root = Messages.from_openai(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "ab"},
                    {"type": "image_url", "image_url": {"url": "data:synthetic"}},
                    {"type": "text", "text": "cd"},
                ],
            }
        ]
    )
    owner = root[0]
    before = dict(owner.__pydantic_private__)
    with model_read_usage() as session:
        projected = project(["bc"], Place(0, "", 1, owner))
        full, cut = ranges(session, owner), ranges(session, projected[0])
        assert cut[0].source == full[0].source
        assert (cut[0].start, cut[0].end) == (1, 3)
        assert owner.__pydantic_private__ == before


def test_origin_text_mismatch_fails_metadata_without_rejecting_the_native_projection():
    owner = InputObject({"body": "correct"})
    with model_read_usage() as session:
        value = project(["wrong"], place(owner))
        assert value[0].content == "wrong"
        result = session.text_measurement(value[0], value[0].source_content)
        assert result.state == "failed" and result.intervals == ()


@pytest.mark.parametrize("shape", ["slice", "transform", "authored"])
async def test_known_untracked_native_variants_are_unavailable_instead_of_fresh_exact_inputs(shape):
    root = Messages.from_openai([{"role": "user", "content": "abcdef"}])
    if shape == "slice":
        value = root._select_ranges((0,), {0: (1, 4)})
    elif shape == "transform":
        value = await apply_content_transform(
            root, ContentTransform((TransformOperation("prefix", ">>"),))
        )
    else:
        value = Messages().add_message("user", "authored")
    with model_read_usage() as session:
        result = session.text_measurement(value[0], value[0].source_content)
        assert result.state == "unavailable" and result.intervals == ()
        copied = value[0].model_copy()
        assert session.text_measurement(copied, copied.source_content).state == "unavailable"


def test_projection_reads_only_native_storage_without_input_attribute_hooks(monkeypatch):
    owner = InputObject({"body": "same"})

    def fail(*_args):
        raise AssertionError("host attribute read")

    monkeypatch.setattr(InputObject, "__getattr__", fail)
    with model_read_usage() as session:
        value = project(["same"], place(owner))
        assert total(*ranges(session, value[0])) == 4


async def test_live_projection_metadata_does_not_retain_native_input_owners_or_value_graphs():
    with model_read_usage() as session:
        owner = InputObject({"body": "sensitive synthetic text"})
        value = project(["sensitive synthetic text"], place(owner))
        refs = weakref.ref(owner), weakref.ref(value[0])
        ranges(session, value[0])
        before = await _retained_bytes(value, 10**9)
        ranges(session, value[0])
        assert await _retained_bytes(value, 10**9) == before
        del value, owner
        gc.collect()
        assert all(reference() is None for reference in refs)
        assert session._text_inputs.registry.snapshot().owners == 0


@pytest.mark.parametrize("limit", ["owners", "work", "ranges"])
def test_projection_metadata_limits_never_change_native_payloads(limit):
    owner = InputObject({"body": "abcdefghij"})
    with model_read_usage(ReadUsageLimits(**{f"max_input_{limit}": 1})) as session:
        value = project(["abij"], disjoint(owner))
        assert value.to_openai_messages() == [{"role": "user", "content": "abij"}]
        result = session.text_measurement(value[0], value[0].source_content)
        assert result.state == "limit_exceeded" and not result.intervals


def test_exported_projection_intervals_are_owned_and_cannot_rewrite_internal_ranges():
    owner = InputObject({"body": "abcdefghij"})
    with model_read_usage() as session:
        value = project(["abij"], disjoint(owner))
        exported = ranges(session, value[0])
        object.__setattr__(exported[0], "end", 1000)
        assert [(item.start, item.end) for item in ranges(session, value[0])] == [(0, 2), (8, 10)]


@given(
    st.lists(st.integers(min_value=0, max_value=9), min_size=1, max_size=16),
    st.integers(min_value=0, max_value=16),
    st.integers(min_value=0, max_value=16),
)
def test_arbitrary_ordered_repeated_runs_and_subcuts_match_independent_character_sets(
    positions, a, b
):
    source = "abcdefghij"
    owner = InputObject({"body": source})
    text = "".join(source[index] for index in positions)
    runs = array(
        "q",
        (
            value
            for output, index in enumerate(positions)
            for value in (output, output + 1, 0, index, index + 1, LINEAR)
        ),
    )
    origin = Derived(text, (place(owner),), runs)
    start, end = sorted((min(a, len(text)), min(b, len(text))))
    with model_read_usage() as session:
        value = project([text], origin)
        assert total(*ranges(session, value[0])) == len(set(positions))
        if start == end:
            with pytest.raises(ValueError, match="message segment is outside"):
                value._select_ranges((0,), {0: (start, end)})
            return
        selected = value._select_ranges((0,), {0: (start, end)})
        assert total(*ranges(session, selected[0])) == len(set(positions[start:end]))


@pytest.mark.parametrize(
    ("field", "text"),
    [
        ("raw_arguments", '{"command":"same","items":["x",true,12,1.25],"a.b":"dot"}'),
        ("arguments.command", "same"),
        ("arguments.items.0", "x"),
        ("arguments.items.1", "true"),
        ("arguments.items.2", "12"),
        ("arguments.items.3", "1.25"),
        (r"arguments.a\.b", "dot"),
    ],
)
def test_call_texts_use_existing_native_argument_storage_without_reparsing(
    monkeypatch, field, text
):
    raw = '{"command":"same","items":["x",true,12,1.25],"a.b":"dot"}'
    call = ToolCall.model_validate(
        {"type": "function", "function": {"name": "lookup", "arguments": raw}}
    )
    assert call.arguments["command"] == "same"
    before = dict(call.__pydantic_private__)
    data_before = dict(call.__dict__)

    def fail(*_args):
        raise AssertionError("argument property repeated")

    monkeypatch.setattr(ToolCall, "arguments", property(fail))
    with model_read_usage() as session:
        one = project([text], Place(0, field, 0, call))
        two = project([text], Place(0, field, 0, call))
        result = ranges(session, one[0])
        assert result == ranges(session, two[0]) and total(*result) == len(text)
        assert call.__pydantic_private__ == before
        assert call.__dict__ == data_before


def test_unparsed_call_field_is_unavailable_without_decoding_or_a_fresh_axis():
    call = ToolCall.model_validate(
        {"type": "function", "function": {"name": "lookup", "arguments": '{"command":"same"}'}}
    )
    with model_read_usage() as session:
        value = project(["same"], Place(0, "arguments.command", 0, call))
        assert call._arguments_cache is None
        result = session.text_measurement(value[0], value[0].source_content)
        assert result.state == "unavailable" and result.intervals == ()


def test_unknown_origin_owners_do_not_trigger_host_attribute_hooks():
    class Host:
        def __getattribute__(self, _name):
            raise AssertionError("host access")

    with model_read_usage() as session:
        value = project(["same"], place(Host()))
        result = session.text_measurement(value[0], value[0].source_content)
        assert result.state == "unavailable" and result.intervals == ()


def test_cycle_in_optional_origin_metadata_stops_without_changing_native_text():
    owner = InputObject({"body": "same"})
    cyclic = Derived("same", (place(owner),), array("q", (0, 4, 0, 0, 4, LINEAR)))
    object.__setattr__(cyclic, "sources", (cyclic,))
    target = Messages.from_openai([{"role": "user", "content": "same"}])[0]
    with model_read_usage() as session:
        # The native builder also rejects cyclic origins. Exercise the optional
        # observer directly to verify its independent graph limit and error boundary.
        session._text_inputs.located(target, target.source_content, cyclic)
        assert target.content == "same"
        result = session.text_measurement(target, target.source_content)
        assert result.state == "failed" and result.intervals == ()


def test_empty_native_projection_is_validated_zero_without_retaining_an_owner():
    with model_read_usage() as session:
        owner = InputObject({"body": ""})
        value = project([""], place(owner))
        assert ranges(session, value[0]) == ()
        reference = weakref.ref(owner)
        del owner, value
        gc.collect()
        assert reference() is None


@pytest.mark.parametrize("bad", [0, -1, True, 262145, 1.5])
def test_invalid_projection_range_limits_are_rejected(bad):
    with pytest.raises(ValueError):
        ReadTextRegistry(max_ranges=bad)


@pytest.mark.parametrize("prepared", [False, True])
async def test_actual_declared_input_call_preserves_verdict_provider_work_and_field_axes(
    tmp_path, monkeypatch, prepared
):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "inputs": {"memory.notes": {"content": ["title", "body"]}},
            "models": {"judge": model_entry(score=True)},
        },
        tmp_path,
        require_rules=False,
    )
    adapter = ModelAdapter()
    backend = ModelInferenceBackend(
        config, adapters={"fake": adapter}, message_builders={"test": passthrough_builder}
    )
    source = 'rule "Leaf" condition: model("judge").score(messages, "synthetic") > 0.5\nrule "Outer" imports: leaf = "Leaf" condition: any n in memory.notes where leaf(n)'
    policy = _policy(source, config, prepared, backend)
    data = {"memory": {"notes": [{"id": "n", "title": "same", "body": "same"}]}}
    baseline = await policy.scan_async(data)
    calls, wires = adapter.score_calls, list(adapter.score_messages)
    recorded = []
    with model_read_usage() as session:
        previous = session._text_inputs.located

        def observe(target, content, origin):
            previous(target, content, origin)
            recorded.append(ranges(session, target))

        monkeypatch.setattr(session._text_inputs, "located", observe)
        result = await policy.scan_async(data)
    assert result.is_complete and result.rule_matches == baseline.rule_matches == ["Outer"]
    assert recorded and len(recorded[0]) == 2 and total(*recorded[0]) == 8
    assert recorded[0][0].source != recorded[0][1].source
    assert adapter.score_calls == calls and adapter.score_messages == wires
