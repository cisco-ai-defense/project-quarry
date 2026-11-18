# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Logical metadata follows explicit work owners without extending model work or budgets."""

import concurrent.futures
import gc
import os
import signal
import time
import weakref

import pytest

from quarry.integrations import model_usage
from quarry.integrations.model_usage import model_call_budget
from quarry.integrations.read_usage import (
    ReadUsageLease,
    ReadUsageLimits,
    ReadUsageSession,
    UnitMeasurement,
    capture_read_usage,
    current_read_usage,
    model_read_usage,
    new_read_scope,
    read_usage_scope,
    resume_read_usage,
    retain_read_usage,
)
from quarry.safety.timer import current_evaluation_deadline, evaluation_deadline
from tests.unit.test_read_usage import finish, measure, unit


def test_deferred_owner_keeps_metadata_live_until_last_explicit_release():
    with model_read_usage() as session:
        owner = retain_read_usage()
        with read_usage_scope("first"), capture_read_usage() as capture:
            ticket = finish(
                session, chars=measure(), tokens=UnitMeasurement("text_tokens", "pending")
            )
    assert current_read_usage() is None and not session._closed
    assert session._owners == 1
    with resume_read_usage(owner), read_usage_scope("second", phase="evidence"):
        assert current_read_usage() is session
        capture.result().replay()
        assert ticket.finish_tokens(measure("text_tokens", end=3))
    assert current_read_usage() is None
    owner.close()
    owner.close()
    assert session._closed and session._owners == 0
    assert session._events == session._bindings == {}
    rows = session.snapshot()["rows"]
    assert [(row["rule"], row["phase"]) for row in rows] == [
        ("first", "decision"),
        ("second", "evidence"),
    ]
    assert all(unit(row, "text_tokens")["value"] == 3 for row in rows)


def test_validated_live_measurements_do_not_pretend_the_collection_is_finalized():
    with model_read_usage() as session:
        owner = session.retain()
        finish(session, chars=measure(), tokens=measure("text_tokens", end=3))
        report = session.snapshot()
        assert report["observations_complete"] and report["measurements_complete"]
        assert report["finalized"] is False
    assert session.snapshot() == report
    owner.close()
    assert session.snapshot() == {**report, "finalized": True}


def test_request_exception_does_not_close_still_owned_metadata():
    owner = None
    with pytest.raises(RuntimeError, match="synthetic"), model_read_usage() as session:
        owner = session.retain()
        finish(session, chars=measure())
        raise RuntimeError("synthetic")
    try:
        assert not session._closed
        with resume_read_usage(owner):
            assert current_read_usage() is session
    finally:
        owner.close()
    assert session._closed


def test_resume_suppresses_foreign_labels_and_captures_then_restores_caller():
    with model_read_usage() as first:
        owner = first.retain()
    with (
        model_read_usage() as second,
        read_usage_scope("unrelated"),
        capture_read_usage() as capture,
    ):
        with resume_read_usage(owner):
            assert current_read_usage() is first
            finish(first, chars=measure())
        assert current_read_usage() is second
        finish(second, chars=measure())
    owner.close()
    assert first.snapshot()["rows"][0]["rule"] is None
    assert second.snapshot()["rows"][0]["rule"] == "unrelated"
    assert len(capture.result()._identities) == 1


@pytest.mark.parametrize("kind", ["missing", "closed"])
def test_missing_or_closed_owner_never_adopts_an_unrelated_callers_session(kind):
    with model_read_usage() as old:
        owner = old.retain()
    owner.close()
    with model_read_usage() as current, read_usage_scope("unrelated"):
        with resume_read_usage(owner if kind == "closed" else None):
            assert current_read_usage() is None
            assert retain_read_usage() is None
            assert new_read_scope() is None
        assert current_read_usage() is current
    assert current.snapshot()["rows"] == []


def test_nested_requests_start_independent_readers_and_restore_outer_scope():
    with model_read_usage() as outer, read_usage_scope("outer"), capture_read_usage() as capture:
        with model_read_usage() as inner:
            finish(inner, chars=measure())
        finish(outer, chars=measure())
    assert inner.snapshot()["rows"][0]["rule"] is None
    assert outer.snapshot()["rows"][0]["rule"] == "outer"
    assert len(capture.result()._identities) == 1


def test_owner_quota_is_cumulative_and_missing_ownership_never_looks_complete():
    with model_read_usage(ReadUsageLimits(max_owners=2)) as session:
        owner = session.retain()
        assert owner is not None and session.retain() is None
        owner.close()
        assert session.retain() is None
        finish(session, chars=measure())
        assert session._owner_reservations == 2 and session._owners == 1
    report = session.snapshot()
    assert session._closed and report["omitted_observations"] == 2
    assert report["observations_complete"] is False
    assert unit(report["rows"][0])["value"] is None


def test_direct_lease_constructor_also_reserves_and_releases_exactly_one_owner():
    session = ReadUsageSession()
    owner = ReadUsageLease(session)
    assert session._owners == 1
    owner.close()
    owner.close()
    assert session._closed and session._owners == 0
    with pytest.raises(ValueError, match="^invalid logical read accounting owner$"):
        ReadUsageLease("sensitive input")


def test_concurrent_duplicate_closes_finalize_once_after_the_last_owner():
    session = ReadUsageSession()
    owners = [session.retain() for _ in range(24)]
    finish(session, chars=measure())
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda owner: owner.close(), owners * 4))
    assert session._closed and session._owners == 0
    assert unit(session.snapshot()["rows"][0])["value"] == 10
    assert session.retain() is None


def test_concurrent_scope_allocation_is_unique_bounded_and_reports_omissions():
    with model_read_usage(ReadUsageLimits(max_scopes=32)) as session:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            scopes = list(pool.map(lambda _: session.new_scope(), range(64)))
        assert sorted(scope for scope in scopes if scope is not None) == list(range(1, 33))
        assert scopes.count(None) == 32
        assert session._scopes == 32
    assert session.snapshot()["omitted_observations"] == 32


def test_explicit_seal_is_idempotent_even_when_owners_remain():
    with model_read_usage() as session:
        owner = session.retain()
        session.begin("example", "score")
        session.close()
        report = session.snapshot()
        assert current_read_usage() is None
        assert session.retain() is None and session.new_scope() is None
        owner.close()
        session.close()
    assert session.snapshot() == report
    assert unit(report["rows"][0])["measurement_states"] == {"abandoned": 1}


def test_closed_lease_and_replay_handles_do_not_retain_the_session():
    with model_read_usage() as session, capture_read_usage() as capture:
        owner = session.retain()
        reference = weakref.ref(session)
        ticket = finish(session, chars=measure())
    replay = capture.result()
    del session
    gc.collect()
    assert reference() is not None
    owner.close()
    gc.collect()
    assert reference() is None
    assert not ticket.finish("succeeded")
    with resume_read_usage(owner):
        assert current_read_usage() is None
    assert replay._source() is None


def test_resume_preserves_existing_deadline_and_physical_attempt_budget():
    with model_read_usage() as session:
        owner = session.retain()
    with evaluation_deadline(20, "synthetic deadline"), model_call_budget(2) as budget:
        deadline = current_evaluation_deadline()
        with resume_read_usage(owner):
            assert current_evaluation_deadline() is deadline
            assert model_usage._current_model_call_budget.get() is budget
            assert budget.calls_by_alias == {}
    owner.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires process fork")
def test_fork_rejects_parent_owners_before_touching_inherited_held_locks():
    with model_read_usage() as session:
        owner = session.retain()
        session._lock.acquire()
        owner._lock.acquire()
        try:
            child = os.fork()
            if child == 0:
                try:
                    signal.alarm(2)
                    assert current_read_usage() is None
                    assert session.retain() is None and session.new_scope() is None
                    with resume_read_usage(owner):
                        assert current_read_usage() is None
                    owner.close()
                    session.close()
                except BaseException:
                    os._exit(91)
                os._exit(0)
        finally:
            owner._lock.release()
            session._lock.release()
        deadline = time.monotonic() + 4
        while True:
            pid, status = os.waitpid(child, os.WNOHANG)
            if pid:
                assert os.waitstatus_to_exitcode(status) == 0
                break
            if time.monotonic() >= deadline:
                os.kill(child, signal.SIGKILL)
                os.waitpid(child, 0)
                pytest.fail("forked accounting owner touched an inherited lock")
            time.sleep(0.01)
        owner.close()
