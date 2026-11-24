# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Finite candidate workers retain capacity until native work actually terminates.

Registered checks are trusted synchronous host functions. Cancellation bounds a
waiter and asks cooperative work to stop; it cannot forcibly terminate arbitrary
Python code. The actual worker retains its slot, context and work owner until it
returns. Completed futures contain only detached values or bounded error codes.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from math import isfinite
from time import monotonic

from quarry.engine.local_work import ensure_local_work_authoritative, measured_local_work
from quarry.integrations.request_limits import (
    owned_request_limit,
    raise_if_owned_evaluation_expired,
)
from quarry.safety.local_match_work import LocalMatchWorkError
from quarry.safety.timer import QuarryTimeoutError
from quarry.types.candidates import CandidateLimitError, CandidateReplyError
from quarry.types.exceptions import CandidateProviderError, LLMError


@dataclass(frozen=True, slots=True)
class CandidateWorkLimits:
    timeout_seconds: float = 1.0
    max_work: int = 262_144
    max_candidates_per_channel: int = 4096
    workers: int = 2
    capacity: int = 8

    def __post_init__(self):
        if (
            type(self.timeout_seconds) not in (int, float)
            or not isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 30
        ):
            raise CandidateLimitError("candidate execution requires a finite timeout in (0, 30]")
        for value, low, high in (
            (self.max_work, 1, 4_000_000),
            (self.max_candidates_per_channel, 0, 4096),
            (self.workers, 1, 8),
            (self.capacity, self.workers, 64),
        ):
            if type(value) is not int or not low <= value <= high:
                raise CandidateLimitError("invalid candidate execution limit")


class _NativeDenial(Exception):
    def __init__(self, code, owned=False):
        self.code, self.owned = code, owned


@dataclass(slots=True)
class CandidateJob:
    """A worker's own cancellation, deadline and finite inspection allowance."""

    deadline: float
    max_work: int
    request_limited: bool = False
    cancelled: threading.Event = field(default_factory=threading.Event)
    work: int = 0

    def checkpoint(self, units=1):
        self.work += units
        if self.cancelled.is_set() or self.work > self.max_work:
            raise _NativeDenial("limit")
        if monotonic() >= self.deadline:
            raise _NativeDenial(
                "request_timeout" if self.request_limited else "limit", owned=self.request_limited
            )
        # Only checks executed here prove that the native caller's ownership ended.
        # Never infer that authority by catching the same type from a host callback.
        try:
            raise_if_owned_evaluation_expired()
        except QuarryTimeoutError:
            raise _NativeDenial("deadline", owned=True) from None
        try:
            ensure_local_work_authoritative()
        except LocalMatchWorkError:
            raise _NativeDenial("local_work", owned=True) from None


@dataclass(slots=True)
class _Outcome:
    value: object = None
    error: str | None = None
    owned: bool = False


def _raise(code, *, owned=False):
    raise_if_owned_evaluation_expired()
    if code == "deadline":
        error = QuarryTimeoutError("Candidate execution deadline exceeded")
    elif code == "local_work":
        error = LocalMatchWorkError("Candidate local work is unavailable or exhausted")
    elif code in {"limit", "busy"}:
        error = CandidateLimitError("Candidate worker allowance or capacity was reached")
    elif code == "request_timeout":
        error = LLMError("Candidate request timed out", {"code": "timeout"})
    elif code == "output":
        error = CandidateReplyError("Candidate execution returned invalid output")
    else:
        error = CandidateProviderError(
            "Candidate check or worker failed", {"code": "provider_error"}
        )
    raise owned_request_limit(error) if owned else error


def _consume(future):
    if not future.cancelled():
        future.exception()


@measured_local_work("match")
def _invoke(function, job):
    return function(job)


def _execute_in_context(function, job):
    job.checkpoint()
    value = _invoke(function, job)
    job.checkpoint()
    return value


class CandidateWorkerOwner:
    """An explicitly registered provider's bounded native capacity, without a job table."""

    def __init__(self, limits: CandidateWorkLimits):
        if type(limits) is not CandidateWorkLimits:
            raise CandidateLimitError("candidate worker requires native finite limits")
        self.limits = CandidateWorkLimits(
            limits.timeout_seconds,
            limits.max_work,
            limits.max_candidates_per_channel,
            limits.workers,
            limits.capacity,
        )
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._executor = None
        self._pending = self._active = 0
        self._closed = False

    def _process(self):
        if os.getpid() != self._pid:
            _raise("unavailable")

    def _execute(self, context, function, job):
        with self._lock:
            self._active += 1

        try:
            return _Outcome(value=context.run(_execute_in_context, function, job))
        except _NativeDenial as error:
            return _Outcome(error=error.code, owned=error.owned)
        except (LocalMatchWorkError, QuarryTimeoutError) as error:
            # A callback can raise either public type. Recheck our own controls
            # outside that callback before claiming the failure belongs to us.
            try:
                context.run(job.checkpoint)
            except _NativeDenial as native:
                return _Outcome(error=native.code, owned=native.owned)
            return _Outcome(
                error="local_work" if type(error) is LocalMatchWorkError else "deadline"
            )
        except CandidateLimitError:
            return _Outcome(error="limit")
        except CandidateReplyError:
            return _Outcome(error="output")
        except BaseException:
            # No callback exception, traceback, text or returned foreign object
            # reaches the concurrent Future or application diagnostics.
            return _Outcome(error="provider")
        finally:
            with self._lock:
                self._active -= 1
            del context, function, job

    def _finished(self, _future: Future):
        with self._lock:
            self._pending -= 1

    async def run(self, function, *, timeout=None):
        self._process()
        raise_if_owned_evaluation_expired()
        duration = self.limits.timeout_seconds
        if timeout is not None:
            if type(timeout) not in (int, float) or not isfinite(timeout) or timeout <= 0:
                raise CandidateLimitError("candidate request timeout must be finite and positive")
            duration = min(duration, timeout)
        request_limited = timeout is not None and timeout < self.limits.timeout_seconds
        job = CandidateJob(monotonic() + duration, self.limits.max_work, request_limited)
        context = contextvars.copy_context()
        with self._lock:
            if self._closed:
                _raise("unavailable")
            if self._pending >= self.limits.capacity:
                _raise("busy")
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.limits.workers, thread_name_prefix="quarry-candidates"
                )
            executor = self._executor
            self._pending += 1
        failed = False
        try:
            native = executor.submit(self._execute, context, function, job)
        except Exception:
            failed = True
        if failed:
            with self._lock:
                self._pending -= 1
            del function, context, executor
            _raise("unavailable")
        native.add_done_callback(self._finished)
        observed = asyncio.wrap_future(native)
        timeout_scope = asyncio.timeout(duration)
        try:
            async with timeout_scope:
                outcome = await asyncio.shield(observed)
        except (TimeoutError, asyncio.CancelledError) as error:
            job.cancelled.set()
            observed.add_done_callback(_consume)
            if isinstance(error, asyncio.CancelledError):
                raise
            if timeout_scope.expired():
                _raise("request_timeout" if request_limited else "limit", owned=request_limited)
            _raise("provider")
        finally:
            del function, context, native, observed
        if outcome.error is not None:
            _raise(outcome.error, owned=outcome.owned)
        raise_if_owned_evaluation_expired()
        return outcome.value

    def close(self):
        self._process()
        with self._lock:
            self._closed = True
            executor = self._executor
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=False)

    def report(self):
        self._process()
        with self._lock:
            return {
                "pending": self._pending,
                "active": self._active,
                "closed": self._closed,
                "complete": self._pending == 0,
            }
