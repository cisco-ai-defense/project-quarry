# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Synthetic logical reads prove attribution without invoking any provider or tokenizer."""

import concurrent.futures
import gc
import json
import weakref
from contextlib import ExitStack

import pytest

from quarry.integrations import read_usage
from quarry.integrations.read_coverage import ReadInterval
from quarry.integrations.read_usage import (
    ReadUsageLimits,
    ReadUsageSession,
    UnitMeasurement,
    capture_read_usage,
    current_read_usage,
    model_read_usage,
    read_usage_scope,
)

A, B, CUT = b"a" * 32, b"b" * 32, b"c" * 32


def measure(unit="characters", start=0, end=10, source=A):
    return UnitMeasurement(unit, "measured", (ReadInterval(source, unit, start, end),))


def finish(session, *, chars=None, tokens=None, outcome="succeeded", model="example", cut=CUT):
    ticket = session.begin(model, "score")
    assert ticket is not None
    assert ticket.finish(outcome, cut=cut, characters=chars, tokens=tokens)
    return ticket


def unit(row, name="characters"):
    return row["coverage"][name]


def test_report_unions_overlapping_repeated_cuts_but_separates_input_and_token_basis():
    with model_read_usage() as session, read_usage_scope("rule"):
        finish(session, chars=measure(end=6), tokens=measure("text_tokens", end=2))
        finish(session, chars=measure(start=4), tokens=measure("text_tokens", start=1, end=4))
        finish(session, chars=measure(start=4), tokens=measure("text_tokens", start=1, end=4))
        finish(
            session, chars=measure(end=3, source=B), tokens=measure("text_tokens", end=2, source=B)
        )
        report = session.snapshot()
    assert report["observations_complete"] is True
    assert report["measurements_complete"] is True
    (row,) = report["rows"]
    assert (row["rule"], row["model"], row["observations"], row["unique_cuts"]) == (
        "rule",
        "example",
        4,
        1,
    )
    assert unit(row)["value"] == 13
    assert unit(row, "text_tokens")["value"] == 6
    assert session.snapshot() == {**report, "finalized": True}


def test_same_memoized_read_is_attributed_once_to_each_requesting_rule_without_recompute():
    calls = 0
    with model_read_usage() as session:
        with read_usage_scope("first"), capture_read_usage() as capture:
            calls += 1
            finish(session, chars=measure(), tokens=measure("text_tokens", end=3))
        replay = capture.result()
        with read_usage_scope("second"):
            replay.replay()
            replay.replay()
        rows = session.snapshot()["rows"]
    assert calls == 1
    assert [row["rule"] for row in rows] == ["first", "second"]
    assert [row["observations"] for row in rows] == [1, 1]
    assert [unit(row)["value"] for row in rows] == [10, 10]


def test_nested_capture_replays_transitive_reads_and_sees_later_measurement_refinement():
    with model_read_usage() as session:
        with read_usage_scope("first"), capture_read_usage() as inner:
            ticket = finish(
                session, chars=measure(), tokens=UnitMeasurement("text_tokens", "pending")
            )
        with read_usage_scope("second"), capture_read_usage() as outer:
            inner.result().replay()
        with read_usage_scope("third"):
            outer.result().replay()
        assert ticket.finish_tokens(measure("text_tokens", end=4))
        assert not ticket.finish_tokens(measure("text_tokens", end=99))
        rows = session.snapshot()["rows"]
    assert [row["rule"] for row in rows] == ["first", "second", "third"]
    assert all(unit(row, "text_tokens")["value"] == 4 for row in rows)


@pytest.mark.parametrize("outcome", ["succeeded", "refused", "failed", "cancelled", "not_sent"])
def test_outcome_never_implies_a_validated_measurement_or_zero(outcome):
    with model_read_usage() as session:
        finish(session, outcome=outcome)
        report = session.snapshot()
    assert report["observations_complete"] is True
    assert report["measurements_complete"] is False
    (row,) = report["rows"]
    assert row["outcome"] == outcome
    for name in ("characters", "text_tokens"):
        assert unit(row, name) == {
            "value": None,
            "known_units": 0,
            "complete": False,
            "measurement_states": {"unavailable": 1},
            "coverage_limited": False,
        }


@pytest.mark.parametrize(
    "state", ["unavailable", "failed", "cancelled", "pending", "abandoned", "limit_exceeded"]
)
def test_auxiliary_measurement_states_preserve_known_character_coverage(state):
    with model_read_usage() as session:
        finish(session, chars=measure(), tokens=UnitMeasurement("text_tokens", state))
        (row,) = session.snapshot()["rows"]
    assert unit(row)["value"] == 10
    assert unit(row, "text_tokens")["value"] is None
    assert unit(row, "text_tokens")["measurement_states"] == {state: 1}


def test_explicit_validated_empty_measurement_is_distinct_from_unavailable():
    with model_read_usage() as session:
        finish(
            session,
            chars=UnitMeasurement("characters", "measured"),
            tokens=UnitMeasurement("text_tokens", "measured"),
        )
        (row,) = session.snapshot()["rows"]
    assert unit(row)["value"] == 0
    assert unit(row, "text_tokens")["value"] == 0


def test_pending_work_at_close_is_abandoned_without_inventing_a_cancellation():
    with model_read_usage() as session:
        ticket = session.begin("example", "complete")
    report = session.snapshot()
    assert report["observations_complete"] is False
    assert report["measurements_complete"] is False
    (row,) = report["rows"]
    assert row["outcome"] == "unknown"
    assert unit(row)["measurement_states"] == {"abandoned": 1}
    assert not ticket.finish("succeeded", characters=measure())
    assert session._events == session._bindings == {}
    report["rows"].clear()
    assert len(session.snapshot()["rows"]) == 1
    session.close()


def test_refinement_cannot_precede_outcome_or_repeat_or_stay_pending():
    with model_read_usage() as session:
        ticket = session.begin("example", "score")
        assert not ticket.finish_tokens(measure("text_tokens"))
        assert ticket.finish("succeeded", tokens=UnitMeasurement("text_tokens", "pending"))
        assert not ticket.finish_tokens(UnitMeasurement("text_tokens", "pending"))
        assert not ticket.finish_tokens(measure("text_tokens"))
        (row,) = session.snapshot()["rows"]
    assert unit(row, "text_tokens")["measurement_states"] == {"failed": 1}


@pytest.mark.parametrize(
    "limits",
    [ReadUsageLimits(max_events=1), ReadUsageLimits(max_bindings=1), ReadUsageLimits(max_rows=1)],
)
def test_omitted_observations_cannot_make_remaining_rows_look_complete(limits):
    with model_read_usage(limits) as session:
        with read_usage_scope("first"):
            finish(session, chars=measure())
        with read_usage_scope("second"):
            ticket = session.begin("example", "score")
            if ticket is not None:
                ticket.finish("succeeded", characters=measure())
        report = session.snapshot()
    assert report["observations_complete"] is False
    assert report["measurements_complete"] is False
    assert report["omitted_observations"] >= 1
    assert all(unit(row)["value"] is None for row in report["rows"])


def test_token_storage_limit_keeps_previously_validated_character_lower_bound():
    with model_read_usage(ReadUsageLimits(max_ranges=1)) as session:
        ticket = finish(session, chars=measure(), tokens=UnitMeasurement("text_tokens", "pending"))
        assert not ticket.finish_tokens(measure("text_tokens", end=3))
        (row,) = session.snapshot()["rows"]
    assert unit(row)["value"] == 10
    assert unit(row, "text_tokens")["measurement_states"] == {"limit_exceeded": 1}
    assert session._range_count == 0


def test_report_work_cap_is_shared_between_rows_and_units():
    with model_read_usage(ReadUsageLimits(max_work=10)) as session:
        with read_usage_scope("first"):
            finish(session, chars=measure())
        with read_usage_scope("second"):
            finish(session, chars=measure())
        report = session.snapshot()
    assert report["measurements_complete"] is False
    assert unit(report["rows"][0])["value"] == 10
    assert unit(report["rows"][0], "text_tokens")["value"] is None
    assert unit(report["rows"][1])["value"] is None
    assert unit(report["rows"][1])["coverage_limited"] is True


def test_malformed_auxiliary_measurement_does_not_discard_valid_character_coverage():
    with model_read_usage() as session:
        ticket = finish(session, chars=measure(), tokens=UnitMeasurement("text_tokens", "pending"))
        assert not ticket.finish_tokens(object.__new__(UnitMeasurement))
        (row,) = session.snapshot()["rows"]
    assert unit(row)["value"] == 10
    assert unit(row, "text_tokens")["value"] is None
    assert unit(row, "text_tokens")["measurement_states"] == {"failed": 1}


def test_ingestion_owns_a_snapshot_of_measurements_ranges_and_limits():
    limits = ReadUsageLimits(max_events=2)
    measurement = measure()
    with model_read_usage(limits) as session:
        finish(session, chars=measurement)
        object.__setattr__(limits, "max_events", 0)
        object.__setattr__(measurement.intervals[0], "end", 1000)
        object.__setattr__(measurement, "state", "failed")
        assert session.limits.max_events == 2
        (row,) = session.snapshot()["rows"]
    assert unit(row)["value"] == 10


def test_bound_is_checked_before_traversing_replayed_measurement_ranges(monkeypatch):
    visits = 0
    original = read_usage._validate_interval

    def visit(value):
        nonlocal visits
        visits += 1
        return original(value)

    with model_read_usage(ReadUsageLimits(max_work=1)) as session:
        pending = UnitMeasurement("text_tokens", "pending")
        measurement = measure()
        monkeypatch.setattr(read_usage, "_validate_interval", visit)
        finish(session, chars=measurement, tokens=pending)
        assert visits == 0
        (row,) = session.snapshot()["rows"]
    assert unit(row)["measurement_states"] == {"limit_exceeded": 1}


def test_capture_depth_and_reference_limits_are_explicitly_incomplete():
    with model_read_usage(ReadUsageLimits(max_capture_depth=2, max_replay_references=1)) as session:
        with ExitStack() as stack:
            first = stack.enter_context(capture_read_usage())
            second = stack.enter_context(capture_read_usage())
            assert stack.enter_context(capture_read_usage()) is None
            finish(session, chars=measure())
        assert first.result() is first.result()
        with read_usage_scope("replayed"):
            second.result().replay()
        assert session._capture_references == 1
        report = session.snapshot()
    assert not report["observations_complete"]
    assert report["omitted_observations"] >= 2


def test_sealed_capture_marks_late_work_incomplete():
    with model_read_usage() as session, capture_read_usage() as capture:
        replay = capture.result()
        finish(session, chars=measure())
        assert not replay._complete
        assert not session.snapshot()["observations_complete"]


def test_cross_request_and_dead_owner_replay_are_incomplete_and_do_not_keep_owner_alive():
    with model_read_usage() as source, capture_read_usage() as capture:
        ticket = finish(source, chars=measure())
    replay = capture.result()
    reference = weakref.ref(source)
    with model_read_usage() as destination:
        replay.replay()
        assert not destination.snapshot()["observations_complete"]
        assert destination.snapshot()["rows"] == []
    del source
    gc.collect()
    assert reference() is None
    assert not ticket.finish("succeeded")
    with model_read_usage() as destination:
        replay.replay()
        assert not destination.snapshot()["observations_complete"]


def test_contexts_restore_after_failure_and_off_path_does_not_allocate_capture():
    assert current_read_usage() is None
    with capture_read_usage() as capture:
        assert capture is None
    with model_read_usage() as outer:
        with pytest.raises(RuntimeError, match="synthetic"), model_read_usage() as inner:
            assert current_read_usage() is inner
            raise RuntimeError("synthetic")
        assert current_read_usage() is outer
    assert current_read_usage() is None


def test_concurrent_finishes_keep_one_consistent_final_measurement():
    session = ReadUsageSession()
    ticket = session.begin("example", "score")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        accepted = list(
            pool.map(lambda _: ticket.finish("succeeded", characters=measure()), range(32))
        )
    assert sum(accepted) == 1
    assert unit(session.snapshot()["rows"][0])["value"] == 10
    session.close()


def test_invalid_native_metadata_is_redacted_and_does_not_commit_false_measurements():
    with model_read_usage() as session:
        ticket = session.begin("example", "score")
        forged = object.__new__(UnitMeasurement)
        assert not ticket.finish("succeeded", characters=forged, cut="sensitive-input")
        (row,) = session.snapshot()["rows"]
    assert row["invalid_measurements"] == 1
    assert unit(row)["value"] is None
    assert unit(row)["measurement_states"] == {"failed": 1}
    assert "sensitive-input" not in json.dumps(session.snapshot())


def test_foreign_process_rejects_before_touching_inherited_locks():
    class Held:
        def __enter__(self):
            raise AssertionError("inherited lock touched")

        def __exit__(self, *args):
            return False

    session = ReadUsageSession()
    ticket = session.begin("example", "score")
    capture = read_usage.ReadCapture(session)
    session._pid = capture._pid = -1
    session._lock = capture._lock = Held()
    assert session.begin("example", "score") is None
    assert not ticket.finish("succeeded", characters=measure())
    assert not ticket.finish_tokens(measure("text_tokens"))
    assert not session._reserve_capture()
    assert session.snapshot()["state"] == "foreign_process"
    assert capture.result() is None
    capture.remember(session, A)
    session.close()


def test_invalid_reader_and_model_labels_do_not_run_host_string_hooks():
    class Label(str):
        def __eq__(self, other):
            raise AssertionError("untrusted equality")

        def encode(self, *args, **kwargs):
            raise AssertionError("untrusted encoding")

    with model_read_usage() as session:
        assert session.begin(Label("example"), "score") is None
        assert session.begin("example", Label("score")) is None
        with read_usage_scope(Label("rule")):
            finish(session, chars=measure())
        assert session.snapshot()["rows"] == []
        assert session.snapshot()["omitted_observations"] == 3


@pytest.mark.parametrize("field", list(ReadUsageLimits.__dataclass_fields__))
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, 1 << 64])
def test_limits_require_bounded_exact_positive_integers(field, value):
    with pytest.raises(ValueError, match="^invalid logical read accounting limit$"):
        ReadUsageLimits(**{field: value})


@pytest.mark.parametrize(
    "unit_name,state,ranges",
    [
        ("characters", "measured", [ReadInterval(A, "characters", 0, 1)]),
        ("characters", "failed", (ReadInterval(A, "characters", 0, 1),)),
        ("text_tokens", "measured", (ReadInterval(A, "characters", 0, 1),)),
        ("characters", "sensitive-input", ()),
    ],
)
def test_measurement_validation_has_fixed_text_free_errors(unit_name, state, ranges):
    with pytest.raises(ValueError, match="^invalid logical input measurement$"):
        UnitMeasurement(unit_name, state, ranges)
