# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Request-local result/scope association for opt-in logical-read snapshots."""

from __future__ import annotations

import copy
import os
import threading
import weakref
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar

from quarry.integrations.read_reporting import scope_report, scopes_report
from quarry.integrations.read_usage import current_read_usage, model_read_usage, observe_read_scopes
from quarry.types.rules import ScanResult


class _ReadReports:
    def __init__(self, session, *, logical_reads=True, window_observations=False):
        self.session = session
        self.logical_reads = logical_reads
        self.window_observations = window_observations
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._scopes = {}
        self._seen_scopes = set()
        self._scope_unavailable = False
        self._activity_scopes = set()
        self._activity_limited = False
        self._recording = True

    def note_read_scope(self, session, scope):
        if self._pid != os.getpid() or session is not self.session:
            return
        # Session -> report is the only lock edge. This section touches native scalar
        # metadata only; it never calls the session or owns a result/input finalizer.
        with self._lock:
            if not self._recording:
                return
            if (
                scope in self._activity_scopes
                or len(self._activity_scopes) < session.limits.max_bindings
            ):
                self._activity_scopes.add(scope)
            else:
                self._activity_limited = True

    def stop(self):
        if self._pid != os.getpid():
            return
        with self._lock:
            self._recording = False

    def _unassociated(self):
        with self._lock:
            return self._activity_limited or not self._activity_scopes <= self._seen_scopes

    def observe_scope(self, scope):
        if self._pid != os.getpid():
            return
        with self._lock:
            if not self._recording:
                return
            if type(scope) is int and (
                scope in self._seen_scopes
                or len(self._seen_scopes) < self.session.limits.max_scopes
            ):
                self._seen_scopes.add(scope)
            else:
                self._scope_unavailable = True

    def record(self, result, scope):
        if self._pid != os.getpid() or type(result) is not ScanResult:
            return
        with self._lock:
            if not self._recording:
                return
            # No callbacks, result owners or arbitrary host containers are retained.
            if len(self._scopes) < self.session.limits.max_bindings:
                self._scopes[id(result)] = (weakref.ref(result), scope)

    def transfer(self, source, target):
        if self._pid != os.getpid() or type(source) is not ScanResult:
            return
        with self._lock:
            if not self._recording:
                return
            previous = self._scopes.pop(id(source), None)
            if previous is None or previous[0]() is not source or type(target) is not ScanResult:
                return
            self._scopes[id(target)] = (weakref.ref(target), previous[1])

    def attach(self, results):
        """Render once, partition once, then copy only each result's own scalar records."""
        if self.window_observations:
            self._attach_windows(results)
        if not self.logical_reads:
            if self._pid == os.getpid():
                with self._lock:
                    self._scopes.clear()
            return results
        if self._pid != os.getpid() or self._unassociated():
            for result in results:
                result.logical_reads = self.session.scope_report(None)
            return results
        report = self.session.report()
        partitions = {}
        for category in ("rows", "by_reader", "by_model", "by_scope"):
            rows = report[category] if category == "rows" else report["aggregates"][category]
            for row in rows:
                partitions.setdefault(row["scope"], {}).setdefault(category, []).append(row)
        bases = {row["model"]: row for row in report.get("token_bases", ())}
        with self._lock:
            scopes, self._scopes = self._scopes, {}
        for result in results:
            previous = scopes.get(id(result))
            scope = previous[1] if previous is not None and previous[0]() is result else None
            if type(scope) is not int:
                result.logical_reads = self.session.scope_report(None)
                continue
            rows = partitions.get(scope, {})
            selected = {
                **report,
                "rows": rows.get("rows", []),
                "aggregates": {
                    **report["aggregates"],
                    **{key: rows.get(key, []) for key in ("by_reader", "by_model", "by_scope")},
                },
            }
            if "token_bases" in report:
                models = {row["model"] for row in selected["rows"]}
                models.update(row["model"] for row in selected["aggregates"]["by_model"])
                selected["token_bases"] = [
                    bases[model] for model in sorted(models) if model in bases
                ]
            result.logical_reads = copy.deepcopy(scope_report(selected, scope))
        return results

    def attach_combined(self, result):
        if self.window_observations:
            self._attach_windows([result], combined=True)
        if not self.logical_reads:
            if self._pid == os.getpid():
                with self._lock:
                    self._scopes.clear()
            return result
        if self._pid != os.getpid() or self._unassociated():
            result.logical_reads = self.session.scope_report(None)
            return result
        with self._lock:
            selected, self._seen_scopes = self._seen_scopes, set()
            self._scopes.clear()
        if len(selected) == 1 and not self._scope_unavailable:
            result.logical_reads = self.session.scope_report(next(iter(selected)))
        else:
            result.logical_reads = scopes_report(self.session.report(), selected)
        return result

    def _attach_windows(self, results, *, combined=False):
        report = self.session.window_observation_report()
        if self._pid != os.getpid():
            # Do not acquire a mutex inherited from a vanished parent thread.
            for result in results:
                result.window_observations = {
                    **report,
                    "scopes": [],
                    "state": "scope_unavailable",
                    "rows": [],
                    "omitted": max(1, report["omitted"]),
                }
            return
        unavailable = self._unassociated()
        with self._lock:
            scopes = dict(self._scopes)
            selected = set(self._seen_scopes)
            unavailable = unavailable or self._scope_unavailable
        partitions = {}
        for row in report["rows"]:
            partitions.setdefault(row["scope"], []).append(row)
        for result in results:
            previous = scopes.get(id(result))
            scope = previous[1] if previous is not None and previous[0]() is result else None
            valid = not unavailable and (combined or type(scope) is int)
            result.window_observations = {
                **report,
                "scopes": sorted(selected) if combined and valid else [scope] if valid else [],
                "state": "collected" if valid else "scope_unavailable",
                "omitted": report["omitted"] if valid else max(1, report["omitted"]),
                "rows": copy.deepcopy(
                    [row for key in sorted(selected) for row in partitions.get(key, ())]
                    if combined and valid
                    else partitions.get(scope, [])
                    if valid
                    else []
                ),
            }


_REPORTS: ContextVar[_ReadReports | None] = ContextVar("quarry_logical_read_reports", default=None)


def record_result_scope(result: ScanResult, scope: int | None) -> None:
    reports = _REPORTS.get()
    if reports is not None and reports.session is current_read_usage():
        reports.record(result, scope)


def record_scan_scope(scope: int | None) -> None:
    reports = _REPORTS.get()
    if reports is not None and reports.session is current_read_usage():
        reports.observe_scope(scope)


def transfer_result_scope(source: ScanResult, target: ScanResult) -> None:
    reports = _REPORTS.get()
    if reports is not None and reports.session is current_read_usage():
        reports.transfer(source, target)


@contextmanager
def collect_read_reports(*, logical_reads=True, window_observations=False):
    """Borrow an active session, otherwise own one through all native request cleanup."""
    with ExitStack() as stack:
        session = current_read_usage()
        if session is None:
            session = stack.enter_context(model_read_usage())
        if window_observations:
            session.enable_window_observations()
        reports = _ReadReports(
            session, logical_reads=logical_reads, window_observations=window_observations
        )
        token = _REPORTS.set(reports)
        try:
            with observe_read_scopes(reports):
                yield reports
        finally:
            reports.stop()
            _REPORTS.reset(token)
