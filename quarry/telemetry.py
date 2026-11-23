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

"""Bounded, provider-neutral scan telemetry."""

from __future__ import annotations

import math
from threading import Lock
from time import monotonic
from typing import Any, Final, Literal

MAX_SCAN_SECONDS: Final = 31_536_000.0
MAX_MODEL_USAGE_ALIASES: Final = 128
MAX_MODEL_USAGE_OPERATIONS_PER_ALIAS: Final = 8
MAX_MODEL_USAGE_RECORDS: Final = 512
MAX_MODEL_USAGE_CALLS: Final = 1_000_000
MAX_MODEL_ALIAS_CHARS: Final = 200
MAX_MODEL_OPERATION_CHARS: Final = 64

ModelOperation = Literal["score", "classify", "complete", "input_tokens", "spans"]

_PUBLIC_OPERATIONS: Final[dict[str, ModelOperation]] = {
    "score": "score",
    "classify": "classify",
    "classify_messages": "classify",
    "complete": "complete",
    "complete_message": "complete",
    "tokenize": "input_tokens",
    "encode_window_input": "input_tokens",
    "window_token_counts": "input_tokens",
    "classify_window_batch": "classify",
    "transform": "complete",
    "external_spans": "spans",
    "candidate_batch": "spans",
}


def elapsed_seconds(started: float, ended: float | None = None) -> float:
    """Return a finite nonnegative monotonic duration, saturated at the public limit."""

    elapsed = (monotonic() if ended is None else ended) - started
    if math.isnan(elapsed) or elapsed <= 0.0:
        return 0.0
    if not math.isfinite(elapsed) or elapsed >= MAX_SCAN_SECONDS:
        return MAX_SCAN_SECONDS
    return elapsed


def add_durations(*durations: float) -> float:
    """Add trusted runtime durations without exceeding the public timing contract."""

    total = 0.0
    for duration in durations:
        if not math.isfinite(duration) or duration >= MAX_SCAN_SECONDS:
            return MAX_SCAN_SECONDS
        if duration > 0.0:
            total += duration
            if total >= MAX_SCAN_SECONDS:
                return MAX_SCAN_SECONDS
    return total


def public_model_operation(adapter_method: str) -> ModelOperation:
    """Map an internal adapter method to its provider-neutral public operation."""

    return _PUBLIC_OPERATIONS[adapter_method]


def empty_model_usage() -> dict[str, Any]:
    """Return a new empty public model-usage summary."""

    return {"calls": 0, "records": [], "truncated": False}


class ModelUsageLedger:
    """Collect attempted physical model calls into a bounded deterministic summary."""

    __slots__ = ("_calls", "_records", "_truncated", "_lock")

    def __init__(self) -> None:
        self._lock = Lock()
        self._calls = 0
        self._records: dict[tuple[str, ModelOperation], int] = {}
        self._truncated = False

    def record(self, model: str, operation: ModelOperation, count: int = 1) -> None:
        """Record a physical attempt consistently across concurrent callers."""

        with self._lock:
            self._record(model, operation, count)

    def _record(self, model: str, operation: ModelOperation, count: int = 1) -> None:
        """Record adapter invocations, saturating diagnostic data without affecting a scan."""

        if type(count) is not int or count <= 0:
            self._truncated = True
            return

        remaining = MAX_MODEL_USAGE_CALLS - self._calls
        charged = min(count, max(0, remaining))
        self._calls += charged
        if charged != count:
            self._truncated = True
        if charged == 0:
            return

        if (
            type(model) is not str
            or not model
            or len(model) > MAX_MODEL_ALIAS_CHARS
            or operation not in _PUBLIC_OPERATIONS.values()
        ):
            self._truncated = True
            return

        key = (model, operation)
        if key not in self._records:
            aliases = {alias for alias, _operation in self._records}
            operations = {
                known_operation for alias, known_operation in self._records if alias == model
            }
            if (
                (model not in aliases and len(aliases) >= MAX_MODEL_USAGE_ALIASES)
                or len(operations) >= MAX_MODEL_USAGE_OPERATIONS_PER_ALIAS
                or len(self._records) >= MAX_MODEL_USAGE_RECORDS
            ):
                self._truncated = True
                return

        previous = self._records.get(key, 0)
        next_count = min(MAX_MODEL_USAGE_CALLS, previous + charged)
        self._records[key] = next_count
        if next_count != previous + charged:
            self._truncated = True

    def mark_truncated(self) -> None:
        """Record that some diagnostic detail could not be represented."""

        with self._lock:
            self._truncated = True

    def snapshot(self) -> dict[str, Any]:
        """Detach a summary while concurrent callers may be starting work."""

        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> dict[str, Any]:
        """Return a detached JSON-ready summary sorted by alias and operation."""

        records = [
            {"model": model, "operation": operation, "calls": calls}
            for (model, operation), calls in sorted(self._records.items())
        ]
        return {
            "calls": self._calls,
            "records": records,
            "truncated": self._truncated,
        }


def merge_model_usage(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge trusted scan summaries while preserving all public bounds."""

    merged = ModelUsageLedger()
    inherited_truncation = False
    for report in reports:
        inherited_truncation = inherited_truncation or report.get("truncated") is True
        represented_calls = 0
        for record in report.get("records", ()):
            if type(record) is not dict:
                inherited_truncation = True
                continue
            operation = record.get("operation")
            if operation not in _PUBLIC_OPERATIONS.values():
                inherited_truncation = True
                continue
            count = record.get("calls")
            merged.record(
                record.get("model"),
                operation,
                count,
            )
            if type(count) is int and count > 0:
                represented_calls += count
        reported_calls = report.get("calls")
        if type(reported_calls) is int and reported_calls > represented_calls:
            merged.record("", "score", reported_calls - represented_calls)
        elif reported_calls != represented_calls:
            inherited_truncation = True
    summary = merged.snapshot()
    if inherited_truncation:
        merged.mark_truncated()
        summary = merged.snapshot()
    return summary


def models_called(report: dict[str, Any]) -> list[str]:
    """Return sorted public aliases that have at least one retained usage record."""

    return sorted(
        {
            record["model"]
            for record in report.get("records", ())
            if type(record) is dict and type(record.get("model")) is str
        }
    )
