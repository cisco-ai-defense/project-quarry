# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Task-local evaluation deadlines."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from math import isfinite


class QuarryTimeoutError(Exception):
    """Raised when evaluation times out."""


@dataclass(frozen=True)
class EvaluationDeadline:
    """An absolute monotonic deadline owned by the current evaluation task."""

    expires_at: float
    error_message: str


_current_rule_deadline: ContextVar[EvaluationDeadline | None] = ContextVar[
    EvaluationDeadline | None
]("quarry_rule_deadline", default=None)
_current_scan_deadline: ContextVar[EvaluationDeadline | None] = ContextVar[
    EvaluationDeadline | None
]("quarry_scan_deadline", default=None)


def _enabled_timeout(seconds: float | None) -> float | None:
    """Normalize a timeout, preserving the documented non-positive disable value."""

    if seconds is None:
        return None
    timeout = float(seconds)
    if not isfinite(timeout):
        raise ValueError("Timeout must be finite or None")
    return timeout if timeout > 0 else None


def current_evaluation_deadline() -> EvaluationDeadline | None:
    """Return the earlier active rule or scan deadline, if any."""

    rule = _current_rule_deadline.get()
    scan = _current_scan_deadline.get()
    if rule is None:
        return scan
    if scan is None:
        return rule
    return rule if rule.expires_at <= scan.expires_at else scan


def remaining_evaluation_time() -> float | None:
    """Return non-negative seconds remaining in the active evaluation."""

    deadline = current_evaluation_deadline()
    if deadline is None:
        return None
    return max(0.0, deadline.expires_at - time.monotonic())


def raise_if_evaluation_expired() -> None:
    """Raise when the task-local evaluation deadline has expired."""

    deadline = current_evaluation_deadline()
    if deadline is not None and time.monotonic() >= deadline.expires_at:
        raise QuarryTimeoutError(deadline.error_message)


def clamp_to_evaluation_deadline(seconds: float | None) -> float | None:
    """Clamp an operation-local timeout to the active rule deadline."""

    raise_if_evaluation_expired()
    local_timeout = _enabled_timeout(seconds)
    remaining = remaining_evaluation_time()
    if remaining is None:
        return local_timeout
    if remaining <= 0:
        raise_if_evaluation_expired()
    return remaining if local_timeout is None else min(local_timeout, remaining)


@contextmanager
def evaluation_deadline(
    seconds: float | None, error_message: str = "Evaluation timeout"
) -> Iterator[EvaluationDeadline | None]:
    """Install a nested, task-local absolute deadline.

    The earliest nested deadline remains authoritative. This context is
    cooperative for synchronous Python work; evaluator checkpoints and native
    operation timeouts enforce it without process-global signal state.
    """

    timeout = _enabled_timeout(seconds)
    if timeout is None:
        raise_if_evaluation_expired()
        yield current_evaluation_deadline()
        return

    candidate = EvaluationDeadline(time.monotonic() + timeout, error_message)
    current = _current_rule_deadline.get()
    active = candidate if current is None or candidate.expires_at < current.expires_at else current
    token = _current_rule_deadline.set(active)
    try:
        raise_if_evaluation_expired()
        yield active
        raise_if_evaluation_expired()
    finally:
        _current_rule_deadline.reset(token)


@contextmanager
def scan_deadline(
    seconds: float | None, error_message: str = "Scan deadline exceeded"
) -> Iterator[EvaluationDeadline | None]:
    """Install a scan-wide deadline inherited by every rule task."""

    if seconds is None:
        yield _current_scan_deadline.get()
        return
    timeout = float(seconds)
    if not isfinite(timeout):
        raise ValueError("Timeout must be finite or None")
    timeout = max(0.0, timeout)

    candidate = EvaluationDeadline(time.monotonic() + timeout, error_message)
    current = _current_scan_deadline.get()
    active = candidate if current is None or candidate.expires_at < current.expires_at else current
    token = _current_scan_deadline.set(active)
    try:
        yield active
    finally:
        _current_scan_deadline.reset(token)


@contextmanager
def without_rule_deadline() -> Iterator[None]:
    """Clear only the caller's rule deadline while retaining the scan cap."""

    token = _current_rule_deadline.set(None)
    try:
        yield
    finally:
        _current_rule_deadline.reset(token)


@contextmanager
def evidence_deadline(seconds: float) -> Iterator[EvaluationDeadline]:
    """Bound post-verdict evidence without extending the caller's scan deadline."""

    candidate = EvaluationDeadline(time.monotonic() + seconds, "Evidence budget exhausted")
    current = _current_scan_deadline.get()
    deadline = (
        candidate if current is None or candidate.expires_at < current.expires_at else current
    )
    rule = _current_rule_deadline.set(None)
    scan = _current_scan_deadline.set(deadline)
    try:
        yield deadline
    finally:
        _current_scan_deadline.reset(scan)
        _current_rule_deadline.reset(rule)
