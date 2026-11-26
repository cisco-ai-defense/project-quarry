# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Explicit reference token maps, exact source projection and honest failure states."""

import gc
import threading
import weakref
from contextvars import copy_context

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quarry import compile_policy, load_compiled_policy
from quarry.input_registry import InputObject
from quarry.integrations.adapter_message_transforms import apply_adapter_transform
from quarry.integrations.read_operations import model_read, record_model_input
from quarry.integrations.read_tokens import TextTokenBasis
from quarry.integrations.read_usage import (
    ReadUsageLimits,
    capture_read_usage,
    model_read_usage,
    read_usage_scope,
)
from quarry.types.messages import Messages
from tests.unit.test_adapter_transform_dsl import configured
from tests.unit.test_located_read_identity import disjoint, place, project
from tests.unit.test_model_read_operations import backend, messages

BASIS = TextTokenBasis("synthetic-pairs", "1")
PAIRS = ((0, 2), (2, 4), (4, 6))


def measured(session, rule=None):
    rows = session.snapshot()["rows"]
    if rule is not None:
        rows = [row for row in rows if row["rule"] == rule]
    assert len(rows) == 1
    return rows[0]["coverage"]["text_tokens"]


def observe(value):
    with model_read("reader", "classify"):
        record_model_input(value)


async def test_real_backend_uses_explicit_basis_without_an_additional_tokenizer_call(tmp_path):
    service, adapter = backend(tmp_path)
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert session.provide_text_tokens("reader", "abcdef", PAIRS)
        assert session.provide_text_tokens("reader", "policy", ((0, 6),))
        result = await service.model_score_messages("reader", "policy", messages("abcdef"))
    assert result == 0.99 and adapter.score_calls == 1 and adapter.tokenizations == 0
    assert measured(session)["value"] == 4
    assert session.snapshot()["measurements_complete"]
    assert session.snapshot()["token_bases"] == [
        {"model": "reader", "name": "synthetic-pairs", "version": "1"}
    ]


async def test_requested_physical_token_count_is_not_relabelled_as_reference_coverage(tmp_path):
    service, adapter = backend(tmp_path)
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert session.provide_text_tokens("reader", "abcdef", PAIRS)
        result = await service.model_input_tokens_messages("reader", messages("abcdef"))
    assert result == 6 and adapter.tokenizations == 1
    assert measured(session)["value"] == 3


@given(
    st.lists(
        st.tuples(st.integers(0, 7), st.integers(1, 8)).filter(lambda pair: pair[0] < pair[1]),
        min_size=1,
        max_size=15,
    )
)
def test_overlap_projection_matches_an_independent_finite_token_reference(cuts):
    value = messages("abcdefgh")
    offsets = ((0, 2), (2, 4), (4, 6), (6, 8))
    expected = set()
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert session.provide_text_tokens("reader", "abcdefgh", offsets)
        for low, high in cuts:
            observe(value._select_ranges((0,), {0: (low, high)}))
            expected.update(
                index for index, (start, end) in enumerate(offsets) if start < high and low < end
            )
    assert measured(session)["value"] == len(expected)


def test_disjoint_located_fields_use_original_token_positions():
    owner = InputObject({"body": "abcdefghij"})
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert session.provide_text_tokens(
            "reader", "abcdefghij", ((0, 2), (2, 4), (4, 6), (6, 8), (8, 10))
        )
        observe(project(["ab", "ij"], disjoint(owner)))
    assert measured(session)["value"] == 2  # Not the enclosing five-token envelope.


def test_distinct_equal_text_occurrences_keep_distinct_token_axes():
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert session.provide_text_tokens("reader", "abcdef", PAIRS)
        observe(messages("abcdef"))
        observe(messages("abcdef"))
    assert measured(session)["value"] == 6


def test_late_valid_map_refines_every_memoized_reader_without_provider_repetition():
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        with read_usage_scope("first"), capture_read_usage() as capture:
            observe(messages("abcdef"))
        replay = capture.result()
        assert replay is not None
        with read_usage_scope("second"):
            replay.replay()
        assert measured(session, "first")["value"] is None
        assert session.provide_text_tokens("reader", "abcdef", PAIRS)
        assert measured(session, "first")["value"] == 3
        assert measured(session, "second")["value"] == 3
    assert {row["observations"] for row in session.snapshot()["rows"]} == {1}


@pytest.mark.parametrize("state", ["failed", "cancelled", "unavailable"])
def test_failed_auxiliary_registration_is_never_a_validated_zero(state):
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert session.provide_text_tokens("reader", "abcdef", (), state=state)
        observe(messages("abcdef"))
    coverage = measured(session)
    assert coverage["value"] is None and coverage["known_units"] == 0
    assert coverage["measurement_states"] == {state: 1}


@pytest.mark.parametrize(
    "offsets",
    [
        [(0, 6)],
        ((0, 0),),
        ((-1, 1),),
        ((0, 7),),
        ((True, 2),),
        ((0, 3), (2, 2)),
        ((1, 4), (0, 6)),
        ((0, 5), (2, 4)),
        ((0, 1, 2),),
        (object(),),
    ],
)
def test_invalid_offset_maps_fail_closed_without_native_result_changes(offsets):
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert not session.provide_text_tokens("reader", "abcdef", offsets)
        observe(messages("abcdef"))
    assert measured(session)["value"] is None
    assert measured(session)["measurement_states"] == {"failed": 1}


def test_conflicting_map_invalidates_prior_readers_instead_of_reusing_old_counts():
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert session.provide_text_tokens("reader", "abcdef", PAIRS)
        observe(messages("abcdef"))
        assert measured(session)["value"] == 3
        assert not session.provide_text_tokens("reader", "abcdef", ((0, 6),))
        assert measured(session)["value"] is None
        assert measured(session)["measurement_states"] == {"failed": 1}


def test_partial_source_coverage_keeps_a_validated_lower_bound():
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert session.provide_text_tokens("reader", "abcdef", PAIRS)
        value = Messages.from_openai(
            [{"role": "user", "content": "abcdef"}, {"role": "user", "content": "unknown"}]
        )
        observe(value)
    coverage = measured(session)
    assert coverage["value"] is None and coverage["known_units"] == 3
    assert coverage["measurement_states"] == {"partial": 1}


async def test_selected_transform_part_projects_reference_tokens_and_ignores_boundaries(tmp_path):
    _, service, adapter = configured(tmp_path, model=True)
    value = messages("abcdef")
    with model_read_usage(token_bases={"extractor": BASIS}) as session:
        assert session.provide_text_tokens("extractor", "abcdef", PAIRS)
        result = await apply_adapter_transform(
            value._select_ranges((0,), {0: (1, 5)}), service, "convert"
        )
    assert result[0].content == "BCDE" and len(adapter.parts) == 1
    assert measured(session)["value"] == 3


def test_positive_width_reference_tokens_do_not_count_empty_text_reads():
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert session.provide_text_tokens("reader", "", ())
        observe(messages(""))
    assert measured(session)["value"] == 0


def test_overlapping_character_spans_can_represent_multiple_tokens_per_character():
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert session.provide_text_tokens("reader", "éx", ((0, 1), (0, 1), (1, 2)))
        observe(messages("éx"))
    assert measured(session)["value"] == 3


def test_registration_has_an_independent_budget_and_duplicate_maps_do_not_reserve_twice():
    with model_read_usage(
        ReadUsageLimits(max_token_spans=3), token_bases={"reader": BASIS}
    ) as session:
        before = session._text_inputs.registry.snapshot().work
        assert session.provide_text_tokens("reader", "abcdef", PAIRS)
        assert session.provide_text_tokens("reader", "abcdef", PAIRS)
        assert session._text_inputs.registry.snapshot().work == before
        observe(messages("abcdef"))
    assert measured(session)["value"] == 3


def test_token_admission_limit_preserves_already_validated_maps():
    with model_read_usage(
        ReadUsageLimits(max_tokenizations=1), token_bases={"reader": BASIS}
    ) as session:
        assert session.provide_text_tokens("reader", "abcdef", PAIRS)
        assert not session.provide_text_tokens("reader", "another", ((0, 7),))
        observe(messages("abcdef"))
    assert measured(session)["value"] == 3


def test_token_registration_and_projection_retain_no_native_source():
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert session.provide_text_tokens("reader", "abcdef", PAIRS)
        value = messages("abcdef")
        reference = weakref.ref(value[0])
        observe(value)
        del value
        gc.collect()
        assert reference() is None
    assert measured(session)["value"] == 3
    assert session._token_maps._axes == session._token_maps._maps == {}
    assert not session.provide_text_tokens("reader", "abcdef", PAIRS)


def test_declared_basis_is_copied_and_different_aliases_can_have_different_vocabularies():
    first = TextTokenBasis("synthetic-pairs", "1")
    with model_read_usage(
        token_bases={"reader": first, "other": TextTokenBasis("synthetic-whole", "1")}
    ) as session:
        object.__setattr__(first, "name", "changed")
        assert session.provide_text_tokens("reader", "abcdef", PAIRS)
        assert session.provide_text_tokens("other", "abcdef", ((0, 6),))
        value = messages("abcdef")
        observe(value)
        with model_read("other", "classify"):
            record_model_input(value)
    rows = session.snapshot()["rows"]
    assert {row["model"]: row["coverage"]["text_tokens"]["value"] for row in rows} == {
        "other": 1,
        "reader": 3,
    }
    assert {item["name"] for item in session.snapshot()["token_bases"]} == {
        "synthetic-pairs",
        "synthetic-whole",
    }


@pytest.mark.parametrize("limit", ["token_axes", "tokenizations", "token_spans", "token_work"])
def test_independent_token_limits_do_not_erase_validated_character_reads(limit):
    with model_read_usage(
        ReadUsageLimits(**{"max_" + limit: 1}), token_bases={"reader": BASIS}
    ) as session:
        session.provide_text_tokens("reader", "abcdef", PAIRS)
        observe(messages("abcdef"))
    row = session.snapshot()["rows"][0]
    assert row["coverage"]["characters"]["value"] == 6
    if limit != "tokenizations":
        assert row["coverage"]["text_tokens"]["value"] is None


def test_failed_measurement_for_empty_text_does_not_become_validated_zero():
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert session.provide_text_tokens("reader", "", (), state="failed")
        observe(messages(""))
    assert measured(session)["value"] is None
    assert measured(session)["measurement_states"] == {"failed": 1}


def test_aliases_declaring_the_same_basis_share_maps_and_conflicts_invalidate_both():
    with model_read_usage(token_bases={"reader": BASIS, "other": BASIS}) as session:
        assert session.provide_text_tokens("reader", "abcdef", PAIRS)
        value = messages("abcdef")
        observe(value)
        with model_read("other", "classify"):
            record_model_input(value)
        assert {row["coverage"]["text_tokens"]["value"] for row in session.snapshot()["rows"]} == {
            3
        }
        assert not session.provide_text_tokens("other", "abcdef", ((0, 6),))
        assert all(
            row["coverage"]["text_tokens"]["value"] is None for row in session.snapshot()["rows"]
        )


def test_untrusted_offset_objects_are_never_iterated_or_formatted():
    class Hostile:
        def __iter__(self):
            pytest.fail("host iteration")

        def __len__(self):
            pytest.fail("host length")

        def __repr__(self):
            pytest.fail("host formatting")

    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert not session.provide_text_tokens("reader", "abcdef", (Hostile(),))
        observe(messages("abcdef"))
    assert measured(session)["measurement_states"] == {"failed": 1}


@pytest.mark.parametrize("field", ["name", "version"])
@pytest.mark.parametrize("value", ["", None, "bad\nlabel", "\ud800", "x" * 129])
def test_invalid_token_basis_labels_are_rejected_with_a_fixed_error(field, value):
    kwargs = {"name": "synthetic", "version": "1", field: value}
    with pytest.raises(ValueError, match="^invalid logical text-token basis$"):
        TextTokenBasis(**kwargs)


def test_basis_failure_never_executes_inherited_locks_in_a_foreign_process():
    class BadLock:
        def __enter__(self):
            pytest.fail("inherited lock")

    with model_read_usage(token_bases={"reader": BASIS}) as session:
        atlas = session._token_maps
        atlas._pid = -1
        atlas._lock = BadLock()
        assert not atlas.reserve_text(1)
        assert not atlas.provide("reader", b"x" * 32, 1, ((0, 1),))
        assert atlas.project("reader", "measured", (), max_work=10).state == "unavailable"
        atlas.close()
    assert session.snapshot()["finalized"]


@pytest.mark.parametrize("prepared", [False, True])
def test_real_policy_reports_registered_reference_basis_with_unchanged_verdict(tmp_path, prepared):
    service, adapter = backend(tmp_path)
    policy = compile_policy(
        'rule "R" condition: model("reader").input_tokens(messages) > 0',
        runtime_config=service.config,
        model_inference_backend=service,
    )
    if prepared:
        policy = load_compiled_policy(
            policy.to_bytes(), runtime_config=service.config, model_inference_backend=service
        )
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert session.provide_text_tokens("reader", "abcdef", PAIRS)
        result = policy.scan({"messages": [{"role": "user", "content": "abcdef"}]})
    assert result.is_complete and result.rule_matches == ["R"] and adapter.tokenizations == 1
    assert measured(session, "R")["value"] == 3


def test_refusal_and_opaque_argument_fields_use_their_own_reference_tokens():
    from tests.unit.test_model_read_input_fields import ARGUMENTS, BODY, NAME, REFUSAL, assistant

    with model_read_usage(token_bases={"reader": BASIS}) as session:
        for text in (BODY, REFUSAL, NAME, ARGUMENTS):
            assert session.provide_text_tokens("reader", text, ((0, len(text)),))
        observe(assistant())
    assert measured(session)["value"] == 4


def test_empty_window_in_a_nonempty_canonical_axis_has_no_reference_tokens():
    value = messages(
        [
            {"type": "text", "text": "ab"},
            {"type": "text", "text": ""},
            {"type": "text", "text": "cd"},
        ]
    )
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert session.provide_text_tokens("reader", "abcd", ((0, 4),))
        # The selected empty native text part is a real zero-width character read;
        # even a reference token spanning its offset must not be counted.
        native = session._text_inputs
        interval = native.registry.window_ranges(
            value[0], native._parts(value[0].source_content), 2, 2
        )
        from quarry.integrations.read_usage import UnitMeasurement

        ticket = session.begin("reader", "transform", native_tokens=True)
        ticket.finish("succeeded", characters=UnitMeasurement("characters", "measured", interval))
    assert measured(session)["value"] == 0


def test_failed_empty_map_also_remains_failed_for_a_located_empty_text():
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert session.provide_text_tokens("reader", "", (), state="failed")
        owner = InputObject({"body": ""})
        observe(project([""], place(owner)))
    assert measured(session)["value"] is None
    assert measured(session)["measurement_states"] == {"failed": 1}


@pytest.mark.parametrize("action", ["observe", "register"])
def test_busy_token_table_never_blocks_native_weak_cleanup_or_registration(action):
    class Owner:
        pass

    with model_read_usage(token_bases={"reader": BASIS}) as session:
        atlas = session._token_maps
        registry = session._text_inputs.registry
        first, other = Owner(), Owner()
        holder = [first]
        reference = weakref.ref(first)
        registry.observe(first, "first")
        del first
        entered = threading.Event()
        context = copy_context()
        returned = []

        def work():
            with registry._lock:
                entered.set()
                if action == "observe":
                    registry.observe(other, "other")
                else:
                    returned.append(
                        context.run(session.provide_text_tokens, "reader", "abcdef", PAIRS)
                    )

        one = threading.Thread(target=work, daemon=True)
        two = threading.Thread(target=holder.clear, daemon=True)
        atlas._lock.acquire()
        try:
            one.start()
            assert entered.wait(1)
            two.start()
            one.join(1)
            two.join(1)
            assert not one.is_alive() and not two.is_alive()
            assert reference() is None
        finally:
            atlas._lock.release()
            one.join(1)
            if two.ident is not None:
                two.join(1)
        if action == "register":
            assert returned == [False]
        observe(messages("abcdef"))
    assert measured(session)["value"] is None
    assert measured(session)["measurement_states"] == {"unavailable": 1}
