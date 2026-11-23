# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Native CPU ownership primitives; whole scanner integration remains separate."""

import asyncio
import contextvars
import json
import threading
from types import SimpleNamespace

import pytest
import re2

from quarry.safety import local_match_work as work
from quarry.safety import timer


@pytest.mark.parametrize("point", ["before_work", "after_native_return", "authority"])
def test_expired_evaluation_deadline_keeps_priority_over_local_work_cut(monkeypatch, point):
    now = 1024.0
    monkeypatch.setattr(timer, "time", SimpleNamespace(monotonic=lambda: now))
    measured = owner(allowance=1)
    clock(monkeypatch, 0, 10)
    reached = []

    def operation():
        nonlocal now
        reached.append(True)
        now += 10
        return True

    with (
        pytest.raises(timer.QuarryTimeoutError, match="selected rule"),
        timer.evaluation_deadline(10, "selected rule"),
    ):
        if point == "before_work":
            now += 10
            measured.call("match", operation)
        elif point == "authority":
            now += 10
            measured.ensure_authoritative()
        else:
            measured.call("match", operation)
    measured.close()
    report = measured.report()
    assert report["segments"] == (1 if point == "after_native_return" else 0)
    assert bool(reached) is (point == "after_native_return")


def owner(*, allowance=100, mode="enforce"):
    return work.LocalMatchWorkOwner(
        work.LocalMatchWorkGroup("group", ("Selected",), allowance, mode)
    )


def clock(monkeypatch, *ticks):
    values = iter(ticks)
    monkeypatch.setattr(work, "thread_time_ns", lambda: next(values))


@pytest.mark.parametrize(
    "allowance", [True, False, 0, -1, 1.0, float("inf"), float("nan"), "1", 1 << 63]
)
def test_allowance_requires_finite_positive_native_integer(allowance):
    with pytest.raises(ValueError, match="allowance"):
        owner(allowance=allowance)


@pytest.mark.parametrize(
    "rules", [(), "Selected", ("",), ("Selected", "Selected"), (object(),), ("x" * 257,)]
)
def test_rule_selection_is_explicit_bounded_and_unique(rules):
    with pytest.raises(ValueError, match="rule"):
        work.LocalMatchWorkGroup("group", rules, 100)


@pytest.mark.parametrize("label", ["", "x" * 65, True, object()])
def test_group_label_is_bounded_native_text(label):
    with pytest.raises(ValueError, match="label"):
        work.LocalMatchWorkGroup(label, ("Selected",), 100)


@pytest.mark.parametrize("mode", ["", "off", True, object()])
def test_only_explicit_shadow_and_enforce_modes_are_allowed(mode):
    with pytest.raises(ValueError, match="mode"):
        owner(mode=mode)


def test_configuration_is_detached_and_revalidated(monkeypatch):
    names = ["Selected"]
    config = work.LocalMatchWorkGroup("group", names, 10)
    names.clear()
    measured = work.LocalMatchWorkOwner(config)
    object.__setattr__(config, "allowance_cpu_ns", 0)
    with pytest.raises(ValueError, match="allowance"):
        work.LocalMatchWorkOwner(config)
    clock(monkeypatch, 0, 5)
    assert measured.call("match", lambda: True)
    measured.close()
    assert measured.report()["allowance_cpu_ns"] == 10


def test_enforcement_charges_actual_overshoot_before_returning_a_value(monkeypatch):
    measured = owner(allowance=10)
    clock(monkeypatch, 0, 11)
    completed = []
    with pytest.raises(work.LocalMatchWorkError) as error:
        measured.call("match", lambda: completed.append(True) or "value")
    assert error.value.details == {"code": "local_work_limit"}
    assert completed == [True]
    measured.close()
    report = measured.report()
    assert report["known_cpu_ns"] == report["value_cpu_ns"] == 11
    assert report["overshoot_cpu_ns"] == 1 and report["cut_phase"] == "match"
    assert report["exhausted"] and report["complete"]


def test_shadow_keeps_the_outcome_and_reports_exhaustion(monkeypatch):
    measured = owner(allowance=10, mode="shadow")
    clock(monkeypatch, 0, 11, 20, 40)
    assert measured.call("prepare", lambda: "first") == "first"
    assert measured.call("match", lambda: "second") == "second"
    measured.close()
    report = measured.report()
    assert report["known_cpu_ns"] == 31 and report["overshoot_cpu_ns"] == 21
    assert report["segments"] == 2 and report["cut_phase"] == "prepare"


def test_exact_boundary_keeps_completed_work_but_denies_the_next_segment(monkeypatch):
    measured = owner(allowance=10)
    clock(monkeypatch, 0, 10)
    assert measured.call("match", lambda: True)
    unseen = []
    with pytest.raises(work.LocalMatchWorkError):
        measured.call("prepare", lambda: unseen.append(True))
    assert unseen == [] and measured.report()["segments"] == 1
    assert measured.report()["cut_phase"] == "prepare"


@pytest.mark.parametrize(
    "error", [TimeoutError("synthetic"), ValueError("synthetic"), asyncio.CancelledError()]
)
def test_original_failures_keep_precedence_while_work_is_still_charged(monkeypatch, error):
    measured = owner(allowance=10)
    clock(monkeypatch, 0, 20)

    def fail():
        raise error

    with pytest.raises(type(error)) as caught:
        measured.call("match", fail)
    assert caught.value is error
    assert measured.report()["known_cpu_ns"] == 20
    assert measured.report()["active_segments"] == 0


@pytest.mark.parametrize("ticks", [(None, 2), (0, None), (5, 4), (True, 4)])
@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_unmeasurable_work_is_unknown_and_enforcement_is_conservative(monkeypatch, ticks, mode):
    measured = owner(mode=mode)
    clock(monkeypatch, *ticks)
    if mode == "enforce":
        with pytest.raises(work.LocalMatchWorkError):
            measured.call("match", lambda: True)
    else:
        assert measured.call("match", lambda: True)
    measured.close()
    report = measured.report()
    assert report["state"] == "unmetered" and report["unmetered_segments"] == 1
    assert report["known_cpu_ns"] == 0 and report["value_cpu_ns"] is None
    assert not report["complete"]


def test_nested_same_thread_work_is_not_double_charged(monkeypatch):
    measured = owner()
    clock(monkeypatch, 0, 9)
    assert measured.call("prepare", lambda: measured.call("match", lambda: 7)) == 7
    measured.close()
    assert measured.report()["value_cpu_ns"] == 9
    assert measured.report()["segments"] == 1


def test_saturated_counters_remain_bounded_and_do_not_claim_an_exact_total(monkeypatch):
    measured = owner(allowance=work._MAX, mode="shadow")
    clock(monkeypatch, 0, work._MAX + 10)
    assert measured.call("match", lambda: True)
    measured.close()
    report = measured.report()
    assert report["known_cpu_ns"] == work._MAX
    assert report["counter_saturated"] and not report["complete"]
    assert report["value_cpu_ns"] is report["overshoot_cpu_ns"] is None


def test_closing_inside_an_active_segment_rejects_new_nested_work(monkeypatch):
    measured = owner()
    clock(monkeypatch, 0, 9)
    unseen = []

    def outer():
        measured.close()
        return measured.call("match", lambda: unseen.append(True))

    with pytest.raises(work.LocalMatchWorkError):
        measured.call("prepare", outer)
    assert unseen == []
    assert measured.report()["complete"] and measured.report()["value_cpu_ns"] == 9


def test_terminal_iterator_work_cannot_hide_an_overrun(monkeypatch):
    measured = owner(allowance=10)
    clock(monkeypatch, 0, 2, 2, 4, 4, 16)
    values = measured.iterate(iter([1, 2]))
    assert next(values) == 1 and next(values) == 2
    with pytest.raises(work.LocalMatchWorkError):
        next(values)
    assert measured.report()["known_cpu_ns"] == 16
    assert measured.report()["segments"] == 3


def test_actual_native_matching_on_a_copied_worker_context_has_one_owner():
    measured = owner(allowance=work._MAX)
    failures = []

    def native():
        compiled = re2.compile(r"synthetic-[0-9]+")
        for _ in range(1000):
            assert compiled.search("synthetic-12345") is not None

    def parent():
        context = contextvars.copy_context()

        def worker():
            try:
                context.run(measured.call, "match", native)
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(5)
        assert not thread.is_alive()

    measured.call("prepare", parent)
    measured.close()
    report = measured.report()
    assert failures == [] and report["complete"]
    assert report["known_cpu_ns"] > 0 and report["segments"] == 2
    assert "synthetic-12345" not in json.dumps(report)


def test_closing_an_owner_does_not_end_running_worker_work():
    measured = owner(allowance=work._MAX)
    entered, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def native():
        entered.set()
        try:
            assert release.wait(5)
            return True
        finally:
            cleaned.set()

    def worker():
        try:
            measured.call("match", native)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert entered.wait(3)
        measured.close()
        report = measured.report()
        assert not report["complete"] and report["active_segments"] == 1
        assert report["value_cpu_ns"] is None and not cleaned.is_set()
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and cleaned.is_set()
    assert len(errors) == 1 and isinstance(errors[0], work.LocalMatchWorkError)
    assert measured.report()["active_segments"] == 0 and measured.report()["complete"]


def test_closed_owner_rejects_copied_context_work_before_calling_it():
    measured = owner()
    context = contextvars.copy_context()
    measured.close()
    entered = []
    with pytest.raises(work.LocalMatchWorkError):
        context.run(measured.call, "match", lambda: entered.append(True))
    assert entered == []
