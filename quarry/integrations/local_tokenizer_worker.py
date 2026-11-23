# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Bounded native worker ownership; waiter cancellation does not end native work."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from math import isfinite
from time import monotonic
from typing import Any

from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.exceptions import (
    LocalTokenizerInvalidInputError,
    LocalTokenizerInvalidOutputError,
    LocalTokenizerLimitError,
    LocalTokenizerUnavailableError,
)
from quarry.types.local_tokenizer import LocalTokenizerValidationError


class _Abandoned(Exception):
    pass


class _Expired(Exception):
    pass


def _raise(code: str) -> None:
    raise_if_evaluation_expired()
    kind, message = {
        "input": (
            LocalTokenizerInvalidInputError,
            "Local tokenizer input is unsupported or invalid",
        ),
        "output": (LocalTokenizerInvalidOutputError, "Local tokenizer output is invalid"),
        "limit": (LocalTokenizerLimitError, "Local tokenizer work limit was reached"),
        "busy": (LocalTokenizerLimitError, "Local tokenizer worker capacity was reached"),
    }.get(code, (LocalTokenizerUnavailableError, "Local tokenizer is unavailable"))
    raise kind(message, {"code": "local_tokenizer_" + code})


@dataclass(slots=True)
class _Job:
    deadline: float
    cancelled: threading.Event = field(default_factory=threading.Event)

    def checkpoint(self) -> None:
        if self.cancelled.is_set():
            raise _Abandoned()
        if monotonic() >= self.deadline:
            raise _Expired()


@dataclass(slots=True)
class _Outcome:
    value: Any = None
    error: str | None = None


def _consume(future: asyncio.Future) -> None:
    if not future.cancelled():
        future.exception()


class _LocalWorkerOwner:
    """One adapter's finite capacity. No request/future/result table is retained.

    Accepted jobs, including queued cancelled jobs, retain their slot until the
    concurrent Future actually terminates. The executor never runs host callbacks
    under this owner's mutex. Only the internal adapter uses ``run``.
    """

    def __init__(self, *, workers: int = 2, capacity: int = 8):
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError("local tokenizer worker count must be from1to8")
        if type(capacity) is not int or not workers <= capacity <= 64:
            raise ValueError("local tokenizer capacity must bound its workers and be at most64")
        self._workers = workers
        self._capacity = capacity
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._pending = self._active = 0
        self._closed = False
        self._executor: ThreadPoolExecutor | None = None

    def _process(self) -> None:
        # An inherited mutex may belong to a vanished parent thread.
        if self._pid != os.getpid():
            _raise("unavailable")

    def _execute(self, function: Callable[[_Job], Any], job: _Job) -> _Outcome:
        with self._lock:
            self._active += 1
        try:
            job.checkpoint()
            value = function(job)
            job.checkpoint()
            return _Outcome(value=value)
        except BaseException as error:
            # The concurrent Future never retains the native exception/traceback,
            # which could otherwise retain a template, artifact or request graph.
            kind = type(error)
            if kind in (_Abandoned, _Expired, LocalTokenizerLimitError):
                code = "limit"
            elif kind is LocalTokenizerInvalidInputError:
                code = "input"
            elif kind in (LocalTokenizerInvalidOutputError, LocalTokenizerValidationError):
                code = "output"
            else:
                code = "unavailable"
            return _Outcome(error=code)
        finally:
            with self._lock:
                self._active -= 1
            del function, job

    def _finished(self, _future: Future) -> None:
        with self._lock:
            self._pending -= 1

    def _submit(self, function: Callable[[_Job], Any], job: _Job) -> Future:
        self._process()
        failed = False
        with self._lock:
            if self._closed:
                _raise("unavailable")
            if self._pending >= self._capacity:
                _raise("busy")
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self._workers, thread_name_prefix="quarry-tokenizer"
                )
            executor = self._executor
            self._pending += 1
        try:
            native = executor.submit(self._execute, function, job)
        except Exception:
            failed = True
        if failed:
            with self._lock:
                self._pending -= 1
            _raise("unavailable")
        # add_done_callback can execute inline; never attach it under our mutex.
        native.add_done_callback(self._finished)
        return native

    async def run(self, function: Callable[[_Job], Any], *, timeout: float) -> Any:
        self._process()
        if type(timeout) not in (int, float) or not 0 < timeout <= 30 or not isfinite(timeout):
            _raise("input")
        raise_if_evaluation_expired()
        job = _Job(monotonic() + timeout)
        native = self._submit(function, job)
        observed = asyncio.wrap_future(native)
        try:
            outcome = await asyncio.shield(observed)
        except asyncio.CancelledError:
            job.cancelled.set()
            observed.add_done_callback(_consume)
            raise
        finally:
            del function, native, observed
        if outcome.error is not None:
            _raise(outcome.error)
        raise_if_evaluation_expired()
        if monotonic() >= job.deadline:
            _raise("limit")
        return outcome.value

    def close(self) -> None:
        self._process()
        with self._lock:
            self._closed = True
            executor = self._executor
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=False)

    def report(self) -> dict[str, Any]:
        if self._pid != os.getpid():
            return {
                "state": "owner_unavailable",
                "pending": None,
                "active": None,
                "closed": None,
                "complete": False,
            }
        with self._lock:
            return {
                "state": "collected",
                "pending": self._pending,
                "active": self._active,
                "closed": self._closed,
                "complete": self._pending == 0,
            }
