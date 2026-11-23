# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Real local primitives and host bridges use explicit, nonoverlapping CPU owners."""

import asyncio
import contextvars

import pytest

from quarry.async_runtime import call_host, run_engine
from quarry.engine import matchers
from quarry.engine.local_work import current_local_work, local_work_scope
from quarry.safety import local_match_work as work
from quarry.safety.regex_backend import RE2_REGEX


def owner(*, mode="enforce", allowance=1_000_000_000):
    return work.LocalMatchWorkOwner(
        work.LocalMatchWorkGroup("group", ("Selected",), allowance, mode)
    )


@pytest.mark.parametrize(
    "operation,expected",
    [
        (lambda: matchers.match_contains("synthetic aba", "aba"), True),
        (lambda: matchers.match_contains_any("synthetic aba", ["none", "aba"]), True),
        (lambda: matchers.match_regex("synthetic aba", "a.a", regex=RE2_REGEX), True),
        (lambda: matchers.find_position("synthetic aba", "aba"), 10),
        (lambda: matchers.count_occurrences("ab ab ab", "ab"), 3),
        (lambda: matchers.extract_match("synthetic aba", "aba"), "aba"),
        (lambda: matchers.match_length("abcd", "==", 4), True),
        (lambda: matchers.field_text({"content": "synthetic"}, "content"), "synthetic"),
    ],
)
def test_real_local_results_match_with_an_explicit_owner(operation, expected):
    baseline = operation()
    measured = owner()
    with local_work_scope(measured):
        actual = operation()
    measured.close()
    assert actual == baseline == expected
    report = measured.report()
    assert report["complete"] and report["known_cpu_ns"] > 0
    assert report["segments"] == 1


def test_disabled_matching_does_not_read_the_cpu_clock(monkeypatch):
    def unexpected():
        pytest.fail("disabled matching read the CPU clock")

    monkeypatch.setattr(work, "thread_time_ns", unexpected)
    with local_work_scope(None):
        assert matchers.match_contains("synthetic aba", "aba")
        assert list(matchers.match_spans("ab ab", "ab")) == [(0, 2), (3, 5)]
    assert current_local_work() is None


def test_preparation_and_native_matching_have_one_owner_without_double_charging(monkeypatch):
    measured = owner()
    ticks = iter([0, 17])
    monkeypatch.setattr(work, "thread_time_ns", lambda: next(ticks))
    with local_work_scope(measured):
        assert matchers.match_contains("synthetic ABA", "aba", matchers.Modifiers(nocase=True))
    measured.close()
    assert measured.report()["value_cpu_ns"] == 17
    assert measured.report()["segments"] == 1


def test_native_result_cannot_escape_an_exhausted_enforcing_owner(monkeypatch):
    measured = owner(allowance=10)
    ticks = iter([0, 11])
    monkeypatch.setattr(work, "thread_time_ns", lambda: next(ticks))
    with local_work_scope(measured), pytest.raises(work.LocalMatchWorkError):
        matchers.match_regex("synthetic aba", "a.a", regex=RE2_REGEX)
    assert current_local_work() is None
    measured.close()
    assert measured.report()["known_overshoot_cpu_ns"] == 1


@pytest.mark.parametrize("regex", [False, True])
def test_lazy_matching_charges_each_actual_step_and_rejects_a_closed_creator(regex):
    measured = owner()
    with local_work_scope(measured):
        values = matchers.match_spans(
            "ab ab ab",
            "ab",
            matchers.Modifiers(regex=regex),
            regex=RE2_REGEX,
        )
    before = measured.report()["segments"]
    assert next(values) == (0, 2)
    assert measured.report()["segments"] > before
    measured.close()
    with pytest.raises(work.LocalMatchWorkError):
        next(values)


def test_scope_restores_an_outer_owner_even_after_an_exception():
    first, second = owner(), owner()
    with local_work_scope(first):
        with pytest.raises(ValueError), local_work_scope(second):
            assert current_local_work() is second
            raise ValueError("synthetic")
        assert current_local_work() is first
        with local_work_scope(None):
            assert current_local_work() is None
        assert current_local_work() is first
    assert current_local_work() is None


async def test_copied_off_loop_context_charges_the_actual_worker():
    measured = owner()
    with local_work_scope(measured):
        assert await asyncio.to_thread(matchers.match_contains, "synthetic aba", "aba")
    measured.close()
    assert measured.report()["complete"] and measured.report()["segments"] == 1
    assert measured.report()["known_cpu_ns"] > 0


@pytest.mark.parametrize("managed_engine", [False, True])
async def test_provider_work_does_not_spend_the_local_matching_allowance(managed_engine):
    measured = owner()

    async def provider():
        assert current_local_work() is None
        assert matchers.match_regex("synthetic aba", "a.a", regex=RE2_REGEX)
        return "synthetic"

    async def engine():
        assert current_local_work() is measured
        assert await call_host(provider) == "synthetic"
        assert current_local_work() is measured
        assert measured.report()["segments"] == 0
        assert matchers.match_contains("synthetic aba", "aba")

    with local_work_scope(measured):
        if managed_engine:
            await run_engine(engine)
        else:
            await engine()
    measured.close()
    assert measured.report()["segments"] == 1
    assert measured.report()["complete"]


def test_a_copied_context_cannot_reopen_closed_matching_work():
    measured = owner()
    with local_work_scope(measured):
        copied = contextvars.copy_context()
    measured.close()
    with pytest.raises(work.LocalMatchWorkError):
        copied.run(matchers.match_contains, "synthetic", "synthetic")
