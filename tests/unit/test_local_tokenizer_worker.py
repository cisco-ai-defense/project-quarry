# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Actual worker ownership survives cancelled/closed waiters and native failures."""

import asyncio
import gc
import os
import threading
import weakref

import pytest

from quarry.integrations.local_tokenizer_worker import _LocalWorkerOwner
from quarry.types.exceptions import (
    LocalTokenizerInvalidInputError,
    LocalTokenizerInvalidOutputError,
    LocalTokenizerLimitError,
    LocalTokenizerUnavailableError,
)


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.mark.parametrize(
    "workers,capacity", [(True, 2), (0, 2), (9, 9), (2, 1), (1, True), (1, 65)]
)
def test_capacity_is_native_and_finite(workers, capacity):
    with pytest.raises(ValueError):
        _LocalWorkerOwner(workers=workers, capacity=capacity)


async def test_real_work_executes_off_the_caller_loop_and_does_not_hold_owner_mutex():
    owner = _LocalWorkerOwner(workers=1, capacity=1)
    caller = threading.get_ident()

    def operation(job):
        assert threading.get_ident() != caller
        assert owner._lock.acquire(blocking=False)
        owner._lock.release()
        job.checkpoint()
        return 17

    try:
        assert await owner.run(operation, timeout=5) == 17
        await until(lambda: owner.report()["pending"] == 0)
        assert owner.report()["complete"]
    finally:
        owner.close()


async def test_cancelled_waiter_keeps_native_resources_and_capacity_until_actual_end():
    owner = _LocalWorkerOwner(workers=1, capacity=1)
    entered, release, ended = threading.Event(), threading.Event(), threading.Event()
    released = threading.Event()

    class Payload:
        def __del__(self):
            released.set()

    payload = Payload()
    reference = weakref.ref(payload)

    def operation(job, held=payload):
        assert held is not None
        entered.set()
        try:
            assert release.wait(5)
            return True
        finally:
            ended.set()

    task = asyncio.create_task(owner.run(operation, timeout=5))
    try:
        await until(entered.is_set)
        del payload, operation
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert owner.report()["pending"] == owner.report()["active"] == 1
        assert not owner.report()["complete"] and not ended.is_set()
        assert reference() is not None and not released.is_set()
        with pytest.raises(LocalTokenizerLimitError) as busy:
            await owner.run(lambda job: False, timeout=5)
        assert busy.value.details["code"] == "local_tokenizer_busy"
        owner.close()
        assert owner.report()["closed"] and not owner.report()["complete"]
    finally:
        release.set()
        await until(lambda: owner.report()["complete"])
        owner.close()
    gc.collect()
    assert ended.is_set() and released.is_set() and reference() is None


async def test_cancelled_queued_job_never_runs_but_keeps_its_reserved_slot_until_terminal():
    owner = _LocalWorkerOwner(workers=1, capacity=2)
    entered, release = threading.Event(), threading.Event()
    touched = []

    def blocking(job):
        entered.set()
        assert release.wait(5)
        return True

    first = asyncio.create_task(owner.run(blocking, timeout=5))
    try:
        await until(entered.is_set)
        queued = asyncio.create_task(owner.run(lambda job: touched.append(True), timeout=5))
        await until(lambda: owner.report()["pending"] == 2)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert owner.report()["pending"] == 2 and not touched
        with pytest.raises(LocalTokenizerLimitError):
            await owner.run(lambda job: None, timeout=5)
    finally:
        release.set()
        assert await first
        await until(lambda: owner.report()["complete"])
        owner.close()
    assert touched == []


async def test_actual_native_return_after_deadline_is_rejected_before_delivery(monkeypatch):
    import quarry.integrations.local_tokenizer_worker as worker

    now = 1000.0
    monkeypatch.setattr(worker, "monotonic", lambda: now)
    owner = _LocalWorkerOwner(workers=1, capacity=1)

    def native(job):
        nonlocal now
        now += 2
        return 19

    try:
        with pytest.raises(LocalTokenizerLimitError):
            await owner.run(native, timeout=1)
        await until(lambda: owner.report()["complete"])
    finally:
        owner.close()


@pytest.mark.parametrize(
    "kind",
    [
        RuntimeError,
        LocalTokenizerInvalidInputError,
        LocalTokenizerInvalidOutputError,
        LocalTokenizerLimitError,
    ],
)
async def test_native_failure_is_typed_and_does_not_retain_private_frames(kind):
    owner = _LocalWorkerOwner(workers=1, capacity=1)
    refs = []

    class Private:
        pass

    def failing(job):
        source = Private()
        refs.append(weakref.ref(source))
        raise kind("synthetic private data must not escape")

    expected = LocalTokenizerUnavailableError if kind is RuntimeError else kind
    try:
        with pytest.raises(expected) as caught:
            await owner.run(failing, timeout=5)
        assert "private data" not in str(caught.value) + str(caught.value.details)
        assert caught.value.__context__ is None
        await until(lambda: owner.report()["complete"])
        gc.collect()
        assert refs[0]() is None
    finally:
        owner.close()


async def test_closed_owner_rejects_admission_without_executing_code():
    owner = _LocalWorkerOwner(workers=1, capacity=1)
    owner.close()
    with pytest.raises(LocalTokenizerUnavailableError):
        await owner.run(lambda job: pytest.fail("closed owner executed work"), timeout=5)


async def test_inherited_process_guard_does_not_wait_for_parent_mutex(monkeypatch):
    owner = _LocalWorkerOwner(workers=1, capacity=1)
    parent_pid = os.getpid()
    with owner._lock:
        monkeypatch.setattr(os, "getpid", lambda: parent_pid + 1)
        with pytest.raises(LocalTokenizerUnavailableError):
            await owner.run(lambda job: pytest.fail("forked owner executed work"), timeout=5)
        with pytest.raises(LocalTokenizerUnavailableError):
            owner.close()
        assert owner.report()["state"] == "owner_unavailable"
        assert owner.report()["pending"] is None
    monkeypatch.undo()
    owner.close()
