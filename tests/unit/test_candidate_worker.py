# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Real worker capacity, copied CPU ownership and native denial provenance."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from quarry.engine.local_work import local_work_scope
from quarry.integrations.candidate_worker import CandidateWorkerOwner, CandidateWorkLimits
from quarry.integrations.request_limits import is_owned_request_limit
from quarry.safety import local_match_work, timer
from quarry.types.candidates import CandidateLimitError
from quarry.types.exceptions import CandidateProviderError


@pytest.mark.parametrize(
    "field,value",
    [
        ("timeout_seconds", True),
        ("timeout_seconds", 0),
        ("timeout_seconds", 31),
        ("timeout_seconds", float("nan")),
        ("timeout_seconds", float("inf")),
        ("max_work", 0),
        ("max_work", True),
        ("max_work", 4_000_001),
        ("max_candidates_per_channel", -1),
        ("workers", 0),
        ("workers", 9),
        ("capacity", 1),
    ],
)
def test_worker_limits_are_native_finite_and_bounded(field, value):
    with pytest.raises(CandidateLimitError):
        CandidateWorkLimits(**{field: value})


async def drained(owner):
    async def wait():
        while not owner.report()["complete"]:
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 3)


@pytest.mark.asyncio
async def test_cancelled_native_work_keeps_slot_until_actual_terminal():
    owner = CandidateWorkerOwner(CandidateWorkLimits(workers=1, capacity=1, timeout_seconds=2))
    started, release = threading.Event(), threading.Event()

    def function(job):
        started.set()
        assert release.wait(3)
        job.checkpoint()
        return 1

    task = asyncio.create_task(owner.run(function))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert owner.report() == {"pending": 1, "active": 1, "closed": False, "complete": False}
        with pytest.raises(CandidateLimitError):
            await owner.run(lambda job: 2)
        release.set()
        await drained(owner)
        assert await owner.run(lambda job: 3) == 3
        await drained(owner)
    finally:
        release.set()
        owner.close()
        await asyncio.gather(task, return_exceptions=True)
        await drained(owner)


@pytest.mark.asyncio
async def test_finite_wait_timeout_does_not_release_running_slot():
    owner = CandidateWorkerOwner(CandidateWorkLimits(workers=1, capacity=1, timeout_seconds=0.02))
    started, release = threading.Event(), threading.Event()

    def function(job):
        started.set()
        assert release.wait(3)
        return True

    try:
        with pytest.raises(CandidateLimitError):
            await owner.run(function)
        assert started.is_set() and owner.report()["pending"] == 1
        release.set()
        await drained(owner)
    finally:
        release.set()
        owner.close()
        await drained(owner)


@pytest.mark.asyncio
async def test_real_work_owner_is_copied_and_charged_on_executing_thread(monkeypatch):
    ticks = iter([0, 10])
    monkeypatch.setattr(local_match_work, "thread_time_ns", lambda: next(ticks))
    work = local_match_work.LocalMatchWorkOwner(
        local_match_work.LocalMatchWorkGroup("synthetic", ("Selected",), 1, "enforce")
    )
    owner = CandidateWorkerOwner(CandidateWorkLimits())
    threads = []
    try:
        with local_work_scope(work), pytest.raises(local_match_work.LocalMatchWorkError) as caught:
            await owner.run(lambda job: threads.append(threading.get_ident()) or True)
        assert is_owned_request_limit(caught.value)
        assert threads and threads[0] != threading.get_ident()
        work.close()
        await drained(owner)
        report = work.report()
        assert report["known_cpu_ns"] == 10 and report["complete"]
        assert report["segments"] == 1 and report["overshoot_cpu_ns"] == 9
    finally:
        work.close()
        owner.close()
        await drained(owner)


@pytest.mark.asyncio
async def test_closed_work_owner_is_a_native_denial_before_callback():
    work = local_match_work.LocalMatchWorkOwner(
        local_match_work.LocalMatchWorkGroup("synthetic", ("Selected",), 1_000_000_000, "enforce")
    )
    work.close()
    owner = CandidateWorkerOwner(CandidateWorkLimits())
    called = []
    try:
        with local_work_scope(work), pytest.raises(local_match_work.LocalMatchWorkError) as caught:
            await owner.run(lambda job: called.append(True))
        assert is_owned_request_limit(caught.value) and called == []
    finally:
        owner.close()
        await drained(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["local_work", "deadline"])
async def test_host_exception_type_does_not_prove_native_exhaustion(kind):
    owner = CandidateWorkerOwner(CandidateWorkLimits())

    def function(job):
        if kind == "local_work":
            raise local_match_work.LocalMatchWorkError("synthetic private callback detail")
        raise timer.QuarryTimeoutError("synthetic private callback detail")

    try:
        with pytest.raises(
            (local_match_work.LocalMatchWorkError, timer.QuarryTimeoutError)
        ) as caught:
            await owner.run(function)
        assert not is_owned_request_limit(caught.value)
        assert "private callback detail" not in str(caught.value)
    finally:
        owner.close()
        await drained(owner)


@pytest.mark.asyncio
async def test_actual_scan_deadline_at_worker_return_has_native_ownership(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(timer, "time", SimpleNamespace(monotonic=lambda: now[0]))
    owner = CandidateWorkerOwner(CandidateWorkLimits())

    def function(job):
        now[0] = 11
        return True

    try:
        with (
            timer.scan_deadline(10, "synthetic scan"),
            pytest.raises(timer.QuarryTimeoutError) as caught,
        ):
            await owner.run(function)
        assert is_owned_request_limit(caught.value)
    finally:
        owner.close()
        await drained(owner)


@pytest.mark.asyncio
async def test_worker_error_future_never_keeps_callback_exception_or_details():
    owner = CandidateWorkerOwner(CandidateWorkLimits())

    def function(job):
        raise RuntimeError("synthetic private callback detail")

    try:
        with pytest.raises(CandidateProviderError) as caught:
            await owner.run(function)
        assert "private callback detail" not in str(caught.value)
        assert caught.value.__cause__ is None and caught.value.__context__ is None
    finally:
        owner.close()
        await drained(owner)


@pytest.mark.asyncio
async def test_failed_native_submission_releases_capacity_without_host_context():
    class FailedExecutor:
        def submit(self, *args):
            raise RuntimeError("synthetic private submission detail")

        def shutdown(self, **kwargs):
            pass

    owner = CandidateWorkerOwner(CandidateWorkLimits())
    owner._executor = FailedExecutor()
    try:
        with pytest.raises(CandidateProviderError) as caught:
            await owner.run(lambda job: True)
        assert owner.report()["pending"] == 0 and owner.report()["complete"]
        assert caught.value.__context__ is None and caught.value.__cause__ is None
        assert "private submission detail" not in str(caught.value)
    finally:
        owner.close()


@pytest.mark.asyncio
async def test_actual_shorter_request_timer_keeps_ownership_and_native_capacity():
    from quarry.types.exceptions import LLMError

    owner = CandidateWorkerOwner(CandidateWorkLimits(workers=1, capacity=1, timeout_seconds=2))
    started, release = threading.Event(), threading.Event()

    def function(job):
        started.set()
        assert release.wait(3)
        return True

    try:
        with pytest.raises(LLMError) as caught:
            await owner.run(function, timeout=0.02)
        assert caught.value.details["code"] == "timeout" and is_owned_request_limit(caught.value)
        assert started.is_set() and owner.report()["pending"] == 1
        release.set()
        await drained(owner)
    finally:
        release.set()
        owner.close()
        await drained(owner)
