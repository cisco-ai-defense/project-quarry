# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Scan-owned CPU accounting for explicitly selected local matching groups.

Only synchronous matching segments belong inside ``call``. Thread CPU, rather than
wall time or process CPU, includes native work on the executing worker and excludes
time spent waiting for another thread. A native call cannot be preempted by this
allowance: charge it after return and report the overshoot before returning a value.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from time import thread_time_ns
from typing import Any, TypeVar

from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.exceptions import EvaluationError

_MAX = (1 << 63) - 1
_PHASES = frozenset({"prepare", "match", "iterate"})
_ACTIVE: ContextVar[tuple[object, int] | None] = ContextVar(
    "quarry_local_work_segment", default=None
)
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class LocalMatchWorkGroup:
    """One finite allowance, shared by named rules in one scan only."""

    label: str
    rules: tuple[str, ...]
    allowance_cpu_ns: int
    mode: str = "shadow"

    def __post_init__(self) -> None:
        if type(self.label) is not str or not 1 <= len(self.label) <= 64:
            raise ValueError("local work group label must contain 1 to 64 characters")
        if type(self.rules) not in (tuple, list) or not 1 <= len(self.rules) <= 256:
            raise ValueError("local work group must select 1 to 256 rules")
        if any(type(rule) is not str or not 1 <= len(rule) <= 256 for rule in self.rules):
            raise ValueError("local work rule names must contain 1 to 256 characters")
        if len(set(self.rules)) != len(self.rules):
            raise ValueError("local work group rule names must be unique")
        if type(self.allowance_cpu_ns) is not int or not 1 <= self.allowance_cpu_ns <= _MAX:
            raise ValueError("local work allowance must be a positive signed 64-bit integer")
        if type(self.mode) is not str or self.mode not in {"shadow", "enforce"}:
            raise ValueError("local work mode must be shadow or enforce")
        object.__setattr__(self, "rules", tuple(self.rules))


class LocalMatchWorkError(EvaluationError):
    """Selected work lacks an available, unexhausted accounting owner."""


@dataclass(frozen=True, slots=True)
class LocalMatchWorkConfig:
    """Bounded, nonoverlapping rule groups; absent configuration needs no session."""

    groups: tuple[LocalMatchWorkGroup, ...]

    def __post_init__(self) -> None:
        if type(self.groups) not in (tuple, list) or not 1 <= len(self.groups) <= 64:
            raise ValueError("local work configuration must contain 1 to 64 groups")
        groups = []
        labels: set[str] = set()
        rules: set[str] = set()
        for group in self.groups:
            if type(group) is not LocalMatchWorkGroup:
                raise ValueError("local work groups must be native configuration values")
            copied = LocalMatchWorkGroup(
                group.label, group.rules, group.allowance_cpu_ns, group.mode
            )
            if copied.label in labels:
                raise ValueError("local work group labels must be unique")
            if any(rule in rules for rule in copied.rules):
                raise ValueError("a rule cannot belong to multiple local work groups")
            labels.add(copied.label)
            rules.update(copied.rules)
            if len(rules) > 1024:
                raise ValueError("local work groups must select at most 1024 rules")
            groups.append(copied)
        object.__setattr__(self, "groups", tuple(groups))

    @classmethod
    def from_mapping(cls, raw: object) -> LocalMatchWorkConfig:
        if type(raw) is not dict or len(raw) != 1 or any(type(key) is not str for key in raw):
            raise ValueError("local work configuration requires only a groups field")
        values = raw.get("groups")
        if not isinstance(values, (tuple, list)) or type(values) not in (tuple, list):
            raise ValueError("local work configuration must contain 1 to 64 groups")
        if not 1 <= len(values) <= 64:
            raise ValueError("local work configuration must contain 1 to 64 groups")
        groups = []
        for item in values:
            if type(item) is not dict or not 3 <= len(item) <= 4:
                raise ValueError("invalid local work group fields")
            if any(type(key) is not str for key in item) or not set(item) <= {
                "label",
                "rules",
                "allowance_cpu_ns",
                "mode",
            }:
                raise ValueError("unknown local work group fields")
            if not {"label", "rules", "allowance_cpu_ns"} <= set(item):
                raise ValueError("missing local work group fields")
            groups.append(
                LocalMatchWorkGroup(
                    item["label"],
                    item["rules"],
                    item["allowance_cpu_ns"],
                    item.get("mode", "shadow"),
                )
            )
        return cls(tuple(groups))

    def validate_rules(self, known_rules: tuple[str, ...]) -> None:
        """Reject unresolved selections before any runtime provider is constructed."""
        if type(known_rules) not in (tuple, list) or any(
            type(rule) is not str for rule in known_rules
        ):
            raise ValueError("known rule names must be native strings")
        known = frozenset(known_rules)
        if any(rule not in known for group in self.groups for rule in group.rules):
            raise ValueError("local work group selects an unknown rule")


class LocalMatchWorkSession:
    """Allocate one owner per group for exactly one scan; bind rules explicitly."""

    def __init__(self, config: LocalMatchWorkConfig, *, known_rules: tuple[str, ...]):
        if type(config) is not LocalMatchWorkConfig:
            raise ValueError("local work configuration must be a native value")
        config = LocalMatchWorkConfig(config.groups)
        config.validate_rules(known_rules)
        self._owners = tuple(LocalMatchWorkOwner(group) for group in config.groups)
        self._by_rule = {
            rule: owner
            for group, owner in zip(config.groups, self._owners, strict=True)
            for rule in group.rules
        }
        self._pid = os.getpid()
        self._closed = False
        self._snapshot: dict[str, Any] | None = None

    def owner_for(self, rule: str) -> LocalMatchWorkOwner | None:
        if self._pid != os.getpid() or self._closed:
            raise LocalMatchWorkError(
                "Local matching work owner is unavailable", {"code": "local_work_owner_unavailable"}
            )
        if type(rule) is not str:
            raise ValueError("rule name must be a native string")
        return self._by_rule.get(rule)

    def close(self) -> None:
        if self._pid != os.getpid():
            raise LocalMatchWorkError(
                "Local matching work owner is unavailable", {"code": "local_work_owner_unavailable"}
            )
        self._closed = True
        for owner in self._owners:
            owner.close()
        if self._snapshot is not None:
            self._snapshot.update(self.report())

    def snapshot(self) -> dict[str, Any]:
        """One bounded result view, finalized when deferred value ownership closes."""
        if self._pid != os.getpid():
            return self.report()
        if self._snapshot is None:
            self._snapshot = self.report()
        return self._snapshot

    def report(self) -> dict[str, Any]:
        if self._pid != os.getpid():
            return {
                "version": 1,
                "state": "owner_unavailable",
                "complete": False,
                "scope": "conversation",
                "conversations": 1,
                "groups": [],
            }
        groups = [owner.report() for owner in self._owners]
        return {
            "version": 1,
            "scope": "conversation",
            "conversations": 1,
            "state": "collected",
            "complete": self._closed and all(row["complete"] for row in groups),
            "groups": groups,
        }


def _clock() -> int | None:
    try:
        value = thread_time_ns()
    except Exception:
        return None
    return value if type(value) is int and value >= 0 else None


class LocalMatchWorkOwner:
    """Keep only bounded counters/configuration, never matching inputs or results."""

    def __init__(self, group: LocalMatchWorkGroup):
        if type(group) is not LocalMatchWorkGroup:
            raise ValueError("local work group must be a native configuration value")
        # Revalidate even an instance forged through object.__setattr__.
        self._group = LocalMatchWorkGroup(
            group.label, group.rules, group.allowance_cpu_ns, group.mode
        )
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._used = 0
        self._segments = 0
        self._denied = 0
        self._active = 0
        self._unmetered = 0
        self._saturated = False
        self._closed = False
        self._cut_phase: str | None = None

    def _error(self, code: str) -> LocalMatchWorkError:
        return LocalMatchWorkError(
            "Local matching work is unavailable or exhausted", {"code": code}
        )

    def _check_process(self) -> None:
        # An inherited mutex may be locked by a vanished parent thread.
        if self._pid != os.getpid():
            raise self._error("local_work_owner_unavailable")

    def call(self, phase: str, function: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
        """Measure one synchronous native/worker segment; charge before returning it.

        Nested calls on the same thread count once. A copied context executing on
        another thread gets its own native clock and shares the original owner.
        Original exceptions (including timeout/cancellation) keep their precedence.
        """
        self._check_process()
        raise_if_evaluation_expired()
        if type(phase) is not str or phase not in _PHASES:
            raise ValueError("unknown local work phase")
        identity = (self, threading.get_ident())
        active = _ACTIVE.get()
        if active is not None and active[0] is self and active[1] == identity[1]:
            with self._lock:
                if self._closed:
                    raise self._error("local_work_owner_unavailable")
            return function(*args, **kwargs)
        with self._lock:
            if self._closed:
                raise self._error("local_work_owner_unavailable")
            if self._group.mode == "enforce" and (
                self._used >= self._group.allowance_cpu_ns or self._unmetered
            ):
                self._saturated |= self._denied >= _MAX
                self._denied = min(_MAX, self._denied + 1)
                if self._cut_phase is None:
                    self._cut_phase = phase
                raise self._error("local_work_limit")
            self._active += 1
        token = _ACTIVE.set(identity)
        start = _clock()
        failed = True
        try:
            result = function(*args, **kwargs)
            failed = False
        finally:
            finish = _clock()
            _ACTIVE.reset(token)
            measured = start is not None and finish is not None and finish >= start
            with self._lock:
                self._active -= 1
                self._saturated |= self._segments >= _MAX
                self._segments = min(_MAX, self._segments + 1)
                if measured:
                    total = self._used + (finish - start)
                    self._saturated |= total > _MAX
                    self._used = min(_MAX, total)
                else:
                    self._saturated |= self._unmetered >= _MAX
                    self._unmetered = min(_MAX, self._unmetered + 1)
                exhausted = (
                    self._used > self._group.allowance_cpu_ns
                    or self._unmetered > 0
                    or self._denied > 0
                )
                if exhausted and self._cut_phase is None:
                    self._cut_phase = phase
                closed = self._closed
            if not failed:
                raise_if_evaluation_expired()
            if not failed and (closed or self._group.mode == "enforce" and exhausted):
                raise self._error("local_work_owner_unavailable" if closed else "local_work_limit")
        return result

    def ensure_authoritative(self) -> None:
        """A caught denial cannot turn into a negative result or reusable value."""
        self._check_process()
        raise_if_evaluation_expired()
        with self._lock:
            if self._closed:
                raise self._error("local_work_owner_unavailable")
            if self._group.mode == "enforce" and (
                self._used > self._group.allowance_cpu_ns or self._unmetered or self._denied
            ):
                raise self._error("local_work_limit")

    def iterate(self, iterator: Iterator[_T]) -> Iterator[_T]:
        """Charge each actual iterator step, preserving laziness and owned work."""

        def step():
            try:
                return True, next(iterator)
            except StopIteration:
                return False, None

        while True:
            present, value = self.call("iterate", step)
            if not present:
                return
            yield value

    def close(self) -> None:
        """Reject new work; an active native segment still owns its resources."""
        self._check_process()
        with self._lock:
            self._closed = True

    def report(self) -> dict[str, Any]:
        if self._pid != os.getpid():
            return {"state": "owner_unavailable", "complete": False, "known_cpu_ns": 0}
        with self._lock:
            complete = (
                self._closed and self._active == 0 and not self._unmetered and not self._saturated
            )
            return {
                "label": self._group.label,
                "mode": self._group.mode,
                "allowance_cpu_ns": self._group.allowance_cpu_ns,
                "conversations": 1,
                "exhausted_conversations": int(
                    self._used >= self._group.allowance_cpu_ns or bool(self._unmetered)
                ),
                "state": "unmetered" if self._unmetered else "measured",
                "known_cpu_ns": self._used,
                "value_cpu_ns": self._used if complete else None,
                "known_overshoot_cpu_ns": max(0, self._used - self._group.allowance_cpu_ns),
                "overshoot_cpu_ns": max(0, self._used - self._group.allowance_cpu_ns)
                if complete
                else None,
                "exhausted": self._used >= self._group.allowance_cpu_ns or bool(self._unmetered),
                "segments": self._segments,
                "denied_segments": self._denied,
                "active_segments": self._active,
                "unmetered_segments": self._unmetered,
                "counter_saturated": self._saturated,
                "cut_phase": self._cut_phase,
                "closed": self._closed,
                "complete": bool(complete),
            }
