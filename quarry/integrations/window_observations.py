# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Bounded, detached window observations attached to existing logical-read identities."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from inspect import getattr_static
from math import isfinite
from types import MemberDescriptorType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quarry.integrations.read_usage import ReadUsageLimits, ReadUsageSession


@dataclass(frozen=True, slots=True)
class WindowObservationLimits:
    """Independent bounds for retained replies, logical rows and scalar traversal."""

    max_replies: int = 128
    max_events: int = 1024
    max_rows: int = 1024
    max_windows: int = 4096
    max_probabilities: int = 65_536
    max_label_bytes: int = 65_536
    max_work: int = 262_144

    def __post_init__(self):
        for value, ceiling in (
            (self.max_replies, 4096),
            (self.max_events, 65_536),
            (self.max_rows, 65_536),
            (self.max_windows, 65_536),
            (self.max_probabilities, 1_048_576),
            (self.max_label_bytes, 1_048_576),
            (self.max_work, 10_000_000),
        ):
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError("invalid window observation limit")


def _static_probabilities(chunk):
    """Read ordinary Python storage without invoking an unused host property."""
    probabilities = getattr_static(chunk, "probabilities", None)
    if type(probabilities) is MemberDescriptorType:
        probabilities = probabilities.__get__(chunk, type(chunk))
    return probabilities


def _snapshot_probabilities(probabilities, count):
    """Project a bounded native slice outside locks, including temporary finalizers."""
    # Exact list/tuple slicing cannot invoke a host override and copies at most count
    # references even if another thread grows the original between admission and copy.
    snapshot = probabilities[:count]
    if len(snapshot) != count:
        return None, "conflicting_probability_vector"
    values = []
    for value in snapshot:
        if type(value) not in (float, int) or not 0 <= value <= 1 or not isfinite(value):
            return None, "invalid_probability_vector"
        values.append(float(value))
    return tuple(values), "measured"


class _WindowObservationStore:
    """All methods run under the owning read session's lock; no host callbacks run here."""

    def __init__(self, limits: WindowObservationLimits):
        if type(limits) is not WindowObservationLimits:
            raise ValueError("invalid window observation limits")
        self.limits = replace(limits)
        self.frames: dict[bytes, dict] = {}
        self.replies: dict[bytes, dict] = {}
        self.windows = self.values = self.work = self.label_bytes = self.omitted = 0

    def charge(self, work: int) -> bool:
        if work > self.limits.max_work - self.work:
            self.work = self.limits.max_work
            self.omitted = min(self.omitted + 1, 1 << 63)
            return False
        self.work += work
        return True

    def note_pending(self, identity, unit):
        if identity in self.frames:
            return
        if len(self.frames) >= self.limits.max_events or not self.charge(1):
            self.omitted = min(self.omitted + 1, 1 << 63)
            return
        self.frames[identity] = {
            "unit": unit,
            "window_count": None,
            "source_characters": {"state": "pending", "value": None},
            "covered_tokens": {"state": "pending", "value": None},
            "reply": None,
            "cached": None,
            "provider_model": None,
            "state": "pending",
        }

    def note_input(self, identity, unit, count, characters, tokens):
        if identity not in self.frames:
            self.note_pending(identity, unit)
            if identity not in self.frames:
                return
        if identity in self.frames and self.frames[identity]["window_count"] is not None:
            # A second distinct cut must not overwrite a first observation silently.
            self.frames[identity]["state"] = "unavailable"
            return
        self.frames[identity] = {
            "unit": unit,
            "window_count": count,
            "source_characters": characters,
            "covered_tokens": tokens,
            "reply": None,
            "cached": None,
            "provider_model": None,
            "state": "not_sent" if count == 0 else "pending",
        }

    def begin_reply(self, identity, generation, provider_model, count, cached):
        frame = self.frames.get(identity)
        if frame is None:
            return False
        if frame["state"] == "unavailable" or count != frame["window_count"]:
            frame["state"] = "unavailable"
            return False
        if type(generation) is not bytes or len(generation) != 16 or type(cached) is not bool:
            frame["state"] = "unavailable"
            return False
        if frame["reply"] is not None and frame["reply"] != generation:
            frame["state"] = "unavailable"
            return False
        frame["reply"], frame["cached"] = generation, cached
        if (
            frame["provider_model"] is None
            and type(provider_model) is str
            and len(provider_model) <= 512
        ):
            try:
                size = len(provider_model.encode("utf-8"))
            except UnicodeError:
                size = self.limits.max_label_bytes + 1
            if size <= self.limits.max_label_bytes - self.label_bytes:
                self.label_bytes += size
                frame["provider_model"] = provider_model
        if generation not in self.replies:
            if (
                type(count) is not int
                or count < 0
                or len(self.replies) >= self.limits.max_replies
                or count > self.limits.max_windows - self.windows
                or not self.charge(2 * count + 1)
            ):
                frame["state"] = "limit_exceeded"
                self.omitted = min(self.omitted + 1, 1 << 63)
                return False
            self.windows += count
            self.replies[generation] = {
                "vectors": [None] * count,
                "states": ["unavailable"] * count,
                "values": 0,
                "measured": 0,
            }
        elif len(self.replies[generation]["vectors"]) != count:
            frame["state"] = "unavailable"
            return False
        frame["state"] = "observed"
        return True

    def _drop_vector(self, reply, ordinal, state):
        previous = reply["vectors"][ordinal]
        if previous is not None:
            self.values -= len(previous)
            reply["values"] -= len(previous)
            reply["measured"] -= 1
            reply["vectors"][ordinal] = None
        reply["states"][ordinal] = state

    def prepare_vector(self, generation, ordinal, count, *, selected):
        """Admit bounded projection work using only native scalar metadata."""
        reply = self.replies.get(generation)
        if reply is None or not 0 <= ordinal < len(reply["vectors"]):
            return False
        previous = reply["vectors"][ordinal]
        if (previous is not None and not selected) or reply["states"][
            ordinal
        ] == "conflicting_probability_vector":
            return False
        if not self.charge(1):
            self._drop_vector(reply, ordinal, "limit_exceeded")
            return False
        if count < 0:
            if selected:
                self._drop_vector(reply, ordinal, "unavailable")
            return False
        if count == 0:
            self._drop_vector(reply, ordinal, "invalid_probability_vector")
            return False
        if count > self.limits.max_probabilities - self.values + (
            len(previous) if previous is not None else 0
        ) or not self.charge(3 * count):
            self._drop_vector(reply, ordinal, "limit_exceeded")
            return False
        return True

    def commit_vector(self, generation, ordinal, values, state, *, selected):
        """Commit detached native floats after projection outside the session lock."""
        reply = self.replies.get(generation)
        if reply is None or not 0 <= ordinal < len(reply["vectors"]):
            return
        previous = reply["vectors"][ordinal]
        if (previous is not None and not selected) or reply["states"][
            ordinal
        ] == "conflicting_probability_vector":
            return
        if values is None:
            self._drop_vector(reply, ordinal, state)
            return
        if previous is not None:
            if previous != values:
                self._drop_vector(reply, ordinal, "conflicting_probability_vector")
            return
        count = len(values)
        # Concurrent projections may have committed while this one copied its input.
        if count > self.limits.max_probabilities - self.values:
            self._drop_vector(reply, ordinal, "limit_exceeded")
            return
        self.values += count
        reply["values"] += count
        reply["measured"] += 1
        reply["vectors"][ordinal] = values
        reply["states"][ordinal] = "measured"

    def report(self, bindings, events, *, finalized):
        rows, seen = [], {}
        omitted, values, labels, work = self.omitted, 0, 0, 0
        for reader, identity in bindings:
            work += 1
            if work > self.limits.max_work:
                omitted = min(omitted + 1, 1 << 63)
                break
            frame = self.frames.get(identity)
            event = events.get(identity)
            if frame is None or event is None:
                continue
            key = (
                reader.scope,
                reader.rule,
                reader.phase,
                event.model,
                event.operation,
                event.cut if event.cut is not None else identity,
                frame["reply"],
            )
            if key in seen:
                row = seen[key]
                row["reads"] += 1
                if frame["cached"] is False:
                    row["cached"] = False
                continue
            label_cost = sum(
                len(s.encode("utf-8"))
                for s in (
                    reader.rule,
                    reader.phase,
                    event.model,
                    event.operation,
                    frame["provider_model"],
                    frame["unit"],
                )
                if s is not None
            )
            if (
                len(rows) >= self.limits.max_rows
                or label_cost > self.limits.max_label_bytes - labels
            ):
                omitted = min(omitted + 1, 1 << 63)
                continue
            labels += label_cost
            state, vectors, states = frame["state"], None, None
            reply = self.replies.get(frame["reply"])
            if event.outcome not in ("succeeded", "not_sent"):
                state = event.outcome
            elif reply is not None and state == "observed":
                size = reply["values"]
                work += 2 * len(reply["vectors"]) + size
                if size > self.limits.max_probabilities - values or work > self.limits.max_work:
                    state = "limit_exceeded"
                else:
                    vectors = [None if v is None else list(v) for v in reply["vectors"]]
                    states = list(reply["states"])
                    values += size
                    state = "measured" if reply["measured"] == len(vectors) else "partial"
            elif state == "not_sent":
                vectors, states = [], []
            row = {
                "scope": reader.scope,
                "rule": reader.rule,
                "phase": reader.phase,
                "model": event.model,
                "provider_model": frame["provider_model"],
                "operation": event.operation,
                "outcome": event.outcome,
                "unit": frame["unit"],
                "window_count": frame["window_count"],
                "source_characters": dict(frame["source_characters"]),
                "covered_tokens": dict(frame["covered_tokens"]),
                "cached": frame["cached"],
                "state": state,
                "probabilities": vectors,
                "probability_states": states,
                "reads": 1,
            }
            rows.append(row)
            seen[key] = row
        return {
            "version": 1,
            "enabled": True,
            "finalized": finalized,
            "rows": rows,
            "omitted": omitted,
        }

    def clear(self):
        self.frames.clear()
        self.replies.clear()


class ModelWindowObservations:
    """A bounded read-only view; each report is a detached ordinary JSON-shaped value."""

    def __init__(self, session: ReadUsageSession):
        self._session = session

    def report(self) -> dict:
        return self._session.window_observation_report()


@contextmanager
def observe_model_windows(
    limits: WindowObservationLimits | None = None, *, read_limits: ReadUsageLimits | None = None
) -> Generator[ModelWindowObservations, None, None]:
    """Collect reached window replies without making model or tokenizer requests.

    Await work inside the context. Deferred owned work keeps its existing metadata
    lease; ``finalized`` remains false until the last owner completes.
    """
    from quarry.integrations.read_usage import model_read_usage

    selected = WindowObservationLimits() if limits is None else limits
    with model_read_usage(read_limits, window_limits=selected) as session:
        yield ModelWindowObservations(session)
