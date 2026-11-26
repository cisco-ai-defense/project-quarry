# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Optional logical reader attribution, independent of provider attempt budgets.

Only bounded, validated measurement metadata is retained. Input identities remain opaque;
reports contain rule/model labels, outcomes and counts. Replay is owned by the collecting
session, and missing replay metadata is reported as incomplete rather than measured zero.
"""

from __future__ import annotations

import copy
import os
import secrets
import threading
import weakref
from collections import Counter
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol, cast

from quarry.engine.observations import (
    GroundingReadCapture,
    GroundingReadObserver,
    GroundingReadReplay,
)
from quarry.integrations.read_coverage import (
    CoverageLimitError,
    ReadCoverage,
    ReadInterval,
    ReadUnit,
    _validate_interval,
)
from quarry.integrations.read_identity import (
    ReadIdentityLimitError,
    ReadIdentityOwnerError,
    ReadIdentityUnavailable,
    ReadTextRegistry,
)
from quarry.integrations.read_reporting import build_aggregates, scope_report
from quarry.integrations.read_tokens import TextTokenAtlas, TextTokenBasis
from quarry.types.read_observations import _text_read_observer

MeasurementState = Literal[
    "measured",
    "partial",
    "pending",
    "unavailable",
    "failed",
    "cancelled",
    "abandoned",
    "limit_exceeded",
]
ReadOutcome = Literal["succeeded", "refused", "failed", "cancelled", "not_sent"]
_STATES = frozenset(
    {
        "measured",
        "partial",
        "pending",
        "unavailable",
        "failed",
        "cancelled",
        "abandoned",
        "limit_exceeded",
    }
)
_OUTCOMES = frozenset({"succeeded", "refused", "failed", "cancelled", "not_sent"})
_OPERATIONS = frozenset(
    {"score", "classify", "spans", "complete", "input_tokens", "split", "transform"}
)
_PHASES = frozenset({"decision", "evidence", "measurement"})
_MAX_SINGLE_RANGES = 65_536


def _label(value: object, *, optional: bool = False) -> bool:
    if value is None:
        return optional
    if type(value) is not str or not 1 <= len(value) <= 512:
        return False
    try:
        value.encode("utf-8")
    except UnicodeError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class UnitMeasurement:
    """Validated ranges in one coordinate system, or an explicit unavailable state."""

    unit: ReadUnit
    state: MeasurementState
    intervals: tuple[ReadInterval, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        _validate_measurement(self, self.unit)


def _validate_measurement(value: object, unit: ReadUnit) -> None:
    if type(value) is not UnitMeasurement:
        raise ValueError("invalid logical input measurement")
    try:
        actual_unit, state, intervals = value.unit, value.state, value.intervals
    except AttributeError:
        raise ValueError("invalid logical input measurement") from None
    if (
        type(actual_unit) is not str
        or actual_unit != unit
        or actual_unit not in {"characters", "text_tokens"}
        or type(state) is not str
        or state not in _STATES
        or type(intervals) is not tuple
        or len(intervals) > _MAX_SINGLE_RANGES
        or state not in {"measured", "partial"}
        and intervals
    ):
        raise ValueError("invalid logical input measurement")
    for interval in intervals:
        _validate_interval(interval)
        if interval.unit != unit:
            raise ValueError("invalid logical input measurement")


def _measurement(value: object, unit: ReadUnit) -> tuple[UnitMeasurement, bool]:
    if value is None:
        return UnitMeasurement(unit, "unavailable"), True
    try:
        _validate_measurement(value, unit)
        assert type(value) is UnitMeasurement
        # Own the metadata snapshot. Even frozen host-created dataclasses can be changed
        # with object.__setattr__; later mutation cannot rewrite an accepted measurement.
        return UnitMeasurement(
            unit,
            value.state,
            tuple(
                ReadInterval(item.source, unit, item.start, item.end) for item in value.intervals
            ),
        ), True
    except (ValueError, AttributeError):
        return UnitMeasurement(unit, "failed"), False


@dataclass(frozen=True, slots=True)
class ReadUsageLimits:
    max_events: int = 1024
    max_bindings: int = 4096
    max_rows: int = 256
    max_ranges: int = 16_384
    max_coverage_keys: int = 4096
    max_work: int = 1_000_000
    max_replay_references: int = 16_384
    max_capture_depth: int = 32
    max_captures: int = 4096
    max_owners: int = 4096
    max_scopes: int = 1024
    max_input_owners: int = 8192
    max_input_work: int = 1_000_000
    max_input_ranges: int = 16_384
    max_token_axes: int = 16_384
    max_tokenizations: int = 4096
    max_token_spans: int = 16_384
    max_token_work: int = 1_000_000
    max_aggregate_groups: int = 2048
    max_aggregate_keys: int = 8192
    max_aggregate_ranges: int = 32_768
    max_aggregate_work: int = 2_000_000

    def __post_init__(self) -> None:
        for value, ceiling in [
            (self.max_events, 65_536),
            (self.max_bindings, 262_144),
            (self.max_rows, 4096),
            (self.max_ranges, 262_144),
            (self.max_coverage_keys, 65_536),
            (self.max_work, 10_000_000),
            (self.max_replay_references, 262_144),
            (self.max_capture_depth, 256),
            (self.max_captures, 65_536),
            (self.max_owners, 65_536),
            (self.max_scopes, 65_536),
            (self.max_input_owners, 262_144),
            (self.max_input_work, 10_000_000),
            (self.max_input_ranges, 262_144),
            (self.max_token_axes, 262_144),
            (self.max_tokenizations, 65_536),
            (self.max_token_spans, 262_144),
            (self.max_token_work, 10_000_000),
            (self.max_aggregate_groups, 65_536),
            (self.max_aggregate_keys, 65_536),
            (self.max_aggregate_ranges, 262_144),
            (self.max_aggregate_work, 10_000_000),
        ]:
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError("invalid logical read accounting limit")


@dataclass(frozen=True, slots=True)
class _Reader:
    scope: int
    rule: str | None
    phase: str


@dataclass(frozen=True, slots=True)
class _Read:
    model: str
    operation: str
    native_tokens: bool = False
    outcome: str = "pending"
    cut: bytes | None = field(default=None, repr=False)
    characters: UnitMeasurement = field(
        default_factory=lambda: UnitMeasurement("characters", "pending")
    )
    tokens: UnitMeasurement = field(
        default_factory=lambda: UnitMeasurement("text_tokens", "pending")
    )
    invalid: bool = False

    @property
    def range_count(self) -> int:
        return len(self.characters.intervals) + len(self.tokens.intervals)


@dataclass(slots=True)
class _UsageRow:
    count: int = 0
    cuts: set[bytes] = field(default_factory=set)
    cuts_complete: bool = True
    coverage: ReadCoverage = field(default_factory=ReadCoverage)
    states: dict[ReadUnit, Counter[str]] = field(
        default_factory=lambda: {"characters": Counter(), "text_tokens": Counter()}
    )
    limited: set[ReadUnit] = field(default_factory=set)
    invalid: int = 0


class _NativeTextObserver:
    """Translate already-read native text to weak, independently bounded lineage."""

    def __init__(self, limits: ReadUsageLimits, token_axes: TextTokenAtlas | None = None) -> None:
        self.registry = ReadTextRegistry(
            max_owners=limits.max_input_owners,
            max_work=limits.max_input_work,
            max_ranges=limits.max_input_ranges,
            token_axes=token_axes,
        )
        self._failure: MeasurementState | None = None

    def _failed(self, error: Exception) -> MeasurementState:
        if isinstance(error, ReadIdentityLimitError):
            self._failure = "limit_exceeded"
        elif (
            not isinstance(error, (ReadIdentityUnavailable, ReadIdentityOwnerError))
            and self._failure != "limit_exceeded"
        ):
            self._failure = "failed"
        return self._failure or "unavailable"

    def _parts(self, content: object) -> tuple[str, ...]:
        if type(content) is str:
            return (content,)
        if content is None:
            return ()
        if type(content) is not tuple:
            raise ValueError("invalid native logical input text")
        # Imported only after native construction; no dependency from data types to models.
        from quarry.types.messages import _FrozenDict

        self.registry.inspect(len(content))
        texts = []
        for part in content:
            if type(part) is not _FrozenDict:
                raise ValueError("invalid native logical input text")
            kind = part.get("type")
            if type(kind) is not str:
                raise ValueError("invalid native logical input text")
            if kind == "text":
                text = part.get("text", "")
                if type(text) is not str:
                    raise ValueError("invalid native logical input text")
                texts.append(text)
        return tuple(texts)

    def _ensure(self, owner: object) -> None:
        from quarry.types.messages import (
            AssistantMessage,
            DeveloperMessage,
            Message,
            SystemMessage,
            ToolMessage,
            UnreadableMessage,
            UserMessage,
        )

        if type(owner) is UnreadableMessage:
            self.registry.unavailable(owner)
        elif type(owner) in (
            SystemMessage,
            DeveloperMessage,
            UserMessage,
            AssistantMessage,
            ToolMessage,
        ):
            native = cast(Message, owner)
            if not self.registry.known(owner) and (
                native._segment is not None
                or native._provenance is not None
                or native._rewrite_source is not None
                or native._synthetic_identity is not None
            ):
                self.registry.unavailable(owner)

    def copied(self, source: object, target: object) -> None:
        try:
            self._ensure(source)
            self.registry.copied(source, target)
            self._copied_calls(source, target)
        except Exception as error:
            self._failed(error)

    def _copied_calls(self, source: object, target: object) -> None:
        from quarry.types.messages import AssistantMessage, ToolCall

        if type(source) is not AssistantMessage or type(target) is not AssistantMessage:
            return
        before, after = source.tool_calls, target.tool_calls
        if type(before) is not tuple or type(after) is not tuple or len(before) != len(after):
            raise ReadIdentityUnavailable("logical input call lineage is unavailable")
        self.registry.inspect(len(before))
        for original, clone in zip(before, after, strict=True):
            if original is not clone and type(original) is ToolCall and type(clone) is ToolCall:
                self.registry.copied(original, clone)

    def authored(self, target: object, content: object) -> None:
        try:
            self.registry.observe_ranges(target, self._parts(content))
        except Exception as error:
            self._failed(error)

    def unavailable(self, target: object) -> None:
        try:
            self.registry.unavailable(target)
        except Exception as error:
            self._failed(error)

    def copy_updated(self, source: object, target: object, update: object) -> None:
        try:
            self._ensure(source)
            if type(update) is not dict:
                self.registry.unavailable(target)
                return
            self.registry.inspect(len(update))
            for key in dict.keys(update):
                if type(key) is not str or key == "source_content":
                    self.registry.unavailable(target)
                    return
            self.registry.copied(source, target)
            if "tool_calls" not in update:
                self._copied_calls(source, target)
        except Exception as error:
            self._failed(error)

    def sliced(self, source: object, target: object, start: int, end: int, text: str) -> None:
        try:
            self._ensure(source)
            self.registry.sliced(source, target, start, end, text)
        except Exception as error:
            self._failed(error)

    def derived(self, source: object, target: object, content: object) -> None:
        try:
            self._ensure(source)
            self.registry.derived_parts(source, target, self._parts(content))
        except Exception as error:
            self._failed(error)

    def constructed(
        self,
        frame: object,
        constructor: object,
        scope: tuple,
        target: object,
        text: str,
        source: object | None,
    ) -> None:
        try:
            if source is not None:
                self._ensure(source)
                self.registry.copied(source, target)
            else:
                self.registry.constructed(frame, constructor, scope, target, text)
        except Exception as error:
            self._failed(error)

    def located(self, target: object, content: object, origin: object) -> None:
        try:
            from quarry.integrations.read_origins import ReadOriginProjection

            self.registry.unavailable(target)
            projector = ReadOriginProjection(self.registry, self._parts, self._ensure)
            ranges = projector.ranges(content, origin)
            self.registry.projected(
                target, self._parts(content), ranges, empty_identity=projector.identity(origin)
            )
        except Exception as error:
            self._failed(error)

    def measurement(self, owner: object, content: object) -> UnitMeasurement:
        if self._failure is not None:
            return UnitMeasurement("characters", self._failure)
        try:
            self._ensure(owner)
            intervals = self.registry.observe_ranges(owner, self._parts(content))
        except Exception as error:
            return UnitMeasurement("characters", self._failed(error))
        return UnitMeasurement("characters", "measured", intervals)

    def close(self) -> None:
        self.registry.close()


_DEFAULT_READER = _Reader(0, None, "decision")
_ACTIVE: ContextVar[ReadUsageSession | None] = ContextVar("quarry_read_usage", default=None)
_READER: ContextVar[_Reader | None] = ContextVar(
    "quarry_read_usage_reader", default=_DEFAULT_READER
)
_CAPTURES: ContextVar[tuple[ReadCapture, ...]] = ContextVar("quarry_read_captures", default=())


class _ScopeActivityObserver(Protocol):
    def note_read_scope(self, session: ReadUsageSession, scope: int) -> None: ...


_SCOPE_ACTIVITY: ContextVar[_ScopeActivityObserver | None] = ContextVar(
    "quarry_read_scope_activity", default=None
)


def current_read_usage() -> ReadUsageSession | None:
    session = _ACTIVE.get()
    return (
        session
        if session is not None and session._pid == os.getpid() and not session._closed
        else None
    )


class ReadUsageLease:
    """One explicit metadata owner; stop work using the lease before closing it.

    Obtain leases through ``ReadUsageSession.retain``. This handle retains only the
    collecting session, never a scan, input, provider response or background task.
    """

    def __init__(self, session: ReadUsageSession) -> None:
        if type(session) is not ReadUsageSession:
            raise ValueError("invalid logical read accounting owner")
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._owner: ReadUsageSession | None = session if session._retain_owner() else None

    def _session(self) -> ReadUsageSession | None:
        if self._pid != os.getpid():
            return None
        with self._lock:
            owner = self._owner
        return (
            owner if owner is not None and owner._pid == os.getpid() and not owner._closed else None
        )

    def close(self) -> None:
        if self._pid != os.getpid():
            self._owner = None
            return
        with self._lock:
            owner, self._owner = self._owner, None
        if owner is not None:
            owner._release_owner()


class ReadTicket:
    """A weak producer handle; completed values retain neither this session nor inputs."""

    def __init__(self, session: ReadUsageSession, identity: bytes) -> None:
        self._session = weakref.ref(session)
        self._identity = identity

    def finish(
        self,
        outcome: ReadOutcome,
        *,
        cut: bytes | None = None,
        characters: UnitMeasurement | None = None,
        tokens: UnitMeasurement | None = None,
    ) -> bool:
        session = self._session()
        return session is not None and session._finish(
            self._identity, outcome, cut, characters, tokens
        )

    def finish_tokens(self, measurement: UnitMeasurement) -> bool:
        session = self._session()
        return session is not None and session._finish_tokens(self._identity, measurement)

    def stage_input(self, characters: UnitMeasurement, cut: bytes | None) -> bool:
        """Reserve live input metadata before an operation waits for provider work."""
        session = self._session()
        return session is not None and session._stage_input(self._identity, characters, cut)


class ReadUsageSession:
    """Request-owned metadata and attribution; this class performs no model or token work."""

    def __init__(
        self,
        limits: ReadUsageLimits | None = None,
        *,
        token_bases: dict[str, TextTokenBasis] | None = None,
    ) -> None:
        if limits is not None and type(limits) is not ReadUsageLimits:
            raise ValueError("invalid logical read accounting limits")
        try:
            self.limits = ReadUsageLimits() if limits is None else replace(limits)
        except AttributeError:
            raise ValueError("invalid logical read accounting limits") from None
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._events: dict[bytes, _Read] = {}
        self._bindings: dict[tuple[_Reader, bytes], None] = {}
        self._range_count = 0
        self._validation_work = 0
        self._capture_references = 0
        self._captures = 0
        self._owners = 0
        self._owner_reservations = 0
        self._scopes = 0
        self._omitted = 0
        self._closed = False
        self._final: dict | None = None
        self._final_report: dict | None = None
        self._token_maps = (
            None
            if token_bases is None
            else TextTokenAtlas(
                token_bases,
                max_axes=self.limits.max_token_axes,
                max_maps=self.limits.max_tokenizations,
                max_spans=self.limits.max_token_spans,
                max_work=self.limits.max_token_work,
            )
        )
        self._text_inputs = _NativeTextObserver(self.limits, self._token_maps)

    def provide_text_tokens(
        self,
        model: str,
        text: str,
        offsets: tuple[tuple[int, int], ...],
        *,
        state: str = "measured",
    ) -> bool:
        """Validate an already computed reference token map, with no tokenizer execution.

        Token indexes are the supplied positive-width Unicode character spans in order.
        ``text`` must be the full canonical text for that source axis, not a fragment
        with proportionally scaled counts. Equal declared bases share their maps.
        Register before collection ends; unavailable/failed/cancelled attempts never
        masquerade as an empty successful map. Conflicting maps invalidate all readers.
        Source binding and registration may remain unavailable under resource or lock
        contention; a False return never claims that a map was accepted.
        """
        maps = self._token_maps
        if self._pid != os.getpid() or current_read_usage() is not self or maps is None:
            return False
        if type(model) is not str or type(text) is not str or type(state) is not str:
            maps.fail_model(model)
            return False
        try:
            fingerprint = self._text_inputs.registry.token_fingerprint(text)
            if fingerprint is None:
                return False
            return maps.provide(model, fingerprint, len(text), offsets, state=state)
        except Exception:
            maps.fail_model(model)
            return False

    def text_measurement(self, owner: object, content: object) -> UnitMeasurement:
        """Measure supplied already-readable native content; perform no model/token work."""
        return self._text_inputs.measurement(owner, content)

    def input_scope(self) -> int | None:
        """Return this active reader's input namespace without retaining its context."""
        reader = _READER.get()
        return reader.scope if current_read_usage() is self and reader is not None else None

    def reader_stamp(self) -> tuple[int, str | None, str] | None:
        reader = _READER.get()
        return (
            (reader.scope, reader.rule, reader.phase)
            if current_read_usage() is self and reader is not None
            else None
        )

    def retain(self) -> ReadUsageLease | None:
        """Lease metadata until owned deferred work stops, or mark omitted on exhaustion."""
        lease = ReadUsageLease(self)
        return lease if lease._owner is not None else None

    def _retain_owner(self) -> bool:
        if self._pid != os.getpid():
            return False
        with self._lock:
            if self._closed:
                return False
            if self._owner_reservations >= self.limits.max_owners:
                self._omitted = min(self._omitted + 1, 1 << 63)
                return False
            self._owners += 1
            self._owner_reservations += 1
            return True

    def _release_owner(self) -> None:
        if self._pid != os.getpid():
            return
        with self._lock:
            if self._closed:
                return
            self._owners -= 1
            assert self._owners >= 0
            if not self._owners:
                self._close_locked()

    def new_scope(self) -> int | None:
        """Allocate a request-local conversation identity independently of input content."""
        if self._pid != os.getpid():
            return None
        with self._lock:
            if self._closed:
                return None
            if self._scopes >= self.limits.max_scopes:
                self._omitted = min(self._omitted + 1, 1 << 63)
                return None
            self._scopes += 1
            return self._scopes

    def _missing(self, count: int = 1) -> None:
        if self._pid != os.getpid():
            return
        with self._lock:
            if not self._closed:
                self._omitted = min(self._omitted + count, 1 << 63)

    def _bind(self, identity: bytes) -> bool:
        reader = _READER.get()
        if reader is None:
            self._omitted = min(self._omitted + 1, 1 << 63)
            return False
        key = (reader, identity)
        if key not in self._bindings:
            if len(self._bindings) >= self.limits.max_bindings:
                self._omitted = min(self._omitted + 1, 1 << 63)
                return False
            self._bindings[key] = None
        observer = _SCOPE_ACTIVITY.get()
        if observer is not None:
            # Includes replay of an existing binding: a custom reader still needs a
            # conversation association even when no new event or provider call exists.
            observer.note_read_scope(self, reader.scope)
        return True

    def begin(
        self, model: str, operation: str, *, native_tokens: bool = False
    ) -> ReadTicket | None:
        if self._pid != os.getpid():
            return None
        if not _label(model) or type(operation) is not str or operation not in _OPERATIONS:
            self._missing()
            return None
        with self._lock:
            if self._closed:
                return None
            if len(self._events) >= self.limits.max_events:
                self._omitted = min(self._omitted + 1, 1 << 63)
                return None
            identity = secrets.token_bytes(32)
            self._events[identity] = _Read(model, operation, native_tokens=native_tokens is True)
            self._bind(identity)
        _capture(self, identity)
        return ReadTicket(self, identity)

    def _install(self, identity: bytes, value: _Read) -> bool:
        previous = self._events[identity]
        next_count = self._range_count - previous.range_count
        admitted = True
        units = []
        for measurement in (value.characters, value.tokens):
            if len(measurement.intervals) > self.limits.max_ranges - next_count:
                measurement = UnitMeasurement(measurement.unit, "limit_exceeded")
                admitted = False
            next_count += len(measurement.intervals)
            units.append(measurement)
        value = replace(value, characters=units[0], tokens=units[1])
        self._events[identity] = value
        self._range_count = next_count
        return admitted

    def _validated_measurement(self, value: object, unit: ReadUnit):
        # Check the native container length before inspecting any supplied ranges. Ingestion
        # and report rendering each have a separate request-wide max_work traversal cap.
        if type(value) is UnitMeasurement:
            try:
                intervals = value.intervals
            except AttributeError:
                return UnitMeasurement(unit, "failed"), False
            if type(intervals) is tuple:
                cost = 1 + 4 * len(intervals)
                if cost > self.limits.max_work - self._validation_work:
                    return UnitMeasurement(unit, "limit_exceeded"), True
                self._validation_work += cost
                if len(intervals) > self.limits.max_ranges:
                    return UnitMeasurement(unit, "limit_exceeded"), True
        return _measurement(value, unit)

    def _finish(self, identity, outcome, cut, characters, tokens) -> bool:
        if self._pid != os.getpid():
            return False
        valid_outcome = type(outcome) is str and outcome in _OUTCOMES
        valid_cut = cut is None or type(cut) is bytes and len(cut) == 32
        with self._lock:
            old = self._events.get(identity)
            if self._closed or old is None or old.outcome != "pending":
                return False
            char, char_valid = self._validated_measurement(characters, "characters")
            token, token_valid = self._validated_measurement(tokens, "text_tokens")
            value = replace(
                old,
                outcome=outcome if valid_outcome else "unknown",
                cut=cut if valid_cut else None,
                characters=char,
                tokens=token,
                invalid=not (valid_outcome and valid_cut and char_valid and token_valid),
            )
            return self._install(identity, value) and not value.invalid

    def _stage_input(self, identity, characters, cut) -> bool:
        if self._pid != os.getpid():
            return False
        with self._lock:
            old = self._events.get(identity)
            if self._closed or old is None or old.outcome != "pending":
                return False
            char, valid = self._validated_measurement(characters, "characters")
            valid_cut = cut is None or type(cut) is bytes and len(cut) == 32
            accepted = self._install(
                identity,
                replace(
                    old,
                    characters=char,
                    cut=cut if valid_cut else None,
                    invalid=old.invalid or not valid or not valid_cut,
                ),
            )
            installed = self._events[identity].characters
            return (
                accepted
                and valid
                and valid_cut
                and type(characters) is UnitMeasurement
                and installed.state == characters.state
                and len(installed.intervals) == len(characters.intervals)
            )

    def _finish_tokens(self, identity, measurement) -> bool:
        if self._pid != os.getpid():
            return False
        with self._lock:
            old = self._events.get(identity)
            if (
                self._closed
                or old is None
                or old.outcome == "pending"
                or old.tokens.state != "pending"
            ):
                return False
            token, valid = self._validated_measurement(measurement, "text_tokens")
            if token.state == "pending":
                token, valid = UnitMeasurement("text_tokens", "failed"), False
            return (
                self._install(
                    identity, replace(old, tokens=token, invalid=old.invalid or not valid)
                )
                and valid
            )

    def _replay(self, source: ReadUsageSession, identities: tuple[bytes, ...]) -> None:
        if self._pid != os.getpid():
            return
        # Runtime memo values belong to one request. A foreign or expired owner cannot
        # validate this request's accounting; never silently import its pending state.
        if source is not self:
            self._missing(max(1, len(identities)))
            return
        captured = []
        with self._lock:
            if self._closed:
                return
            for identity in identities:
                if identity not in self._events:
                    self._omitted = min(self._omitted + 1, 1 << 63)
                    continue
                self._bind(identity)
                captured.append(identity)
        for identity in captured:
            _capture(self, identity)

    def _reserve_capture(self) -> bool:
        if self._pid != os.getpid():
            return False
        with self._lock:
            if self._closed or self._capture_references >= self.limits.max_replay_references:
                self._omitted = min(self._omitted + 1, 1 << 63)
                return False
            self._capture_references += 1
            return True

    def _reserve_capture_object(self) -> bool:
        if self._pid != os.getpid():
            return False
        with self._lock:
            if self._closed:
                return False
            if self._captures >= self.limits.max_captures:
                self._omitted = min(self._omitted + 1, 1 << 63)
                return False
            self._captures += 1
            return True

    def snapshot(self) -> dict:
        """Read current counts; only the last owner's release finalizes the collection."""
        if self._pid != os.getpid():
            return {
                "version": 1,
                "finalized": False,
                "observations_complete": False,
                "measurements_complete": False,
                "omitted_observations": 0,
                "rows": [],
                "state": "foreign_process",
            }
        with self._lock:
            if self._closed:
                assert self._final is not None
                return copy.deepcopy(self._final)
            return self._render()

    def report(self) -> dict:
        """Full scalar report with exact unions across operations, readers and models.

        Summaries distinguish reader observations from distinct attributed evaluations.
        Different conversation scopes add separately; incompatible token bases retain
        separate totals and never produce an unqualified numeric token sum.
        """
        if self._pid != os.getpid():
            result = self.snapshot()
            result["aggregates"] = build_aggregates({}, {}, None, self.limits, omitted=1)
            return result
        with self._lock:
            if self._closed:
                assert self._final_report is not None
                return copy.deepcopy(self._final_report)
            return self._render(aggregates=True)

    def scope_report(self, scope: int | None) -> dict:
        """Report one allocated conversation scope; missing scope admission is unknown."""
        if self._pid != os.getpid():
            result = self.report()
            result["scope"] = None
            return result
        if type(scope) is not int or not 0 <= scope <= self._scopes:
            return {
                "version": 1,
                "scope": None,
                "finalized": self._closed,
                "observations_complete": False,
                "measurements_complete": False,
                "omitted_observations": max(1, self._omitted),
                "rows": [],
                "state": "scope_unavailable",
                "aggregates": build_aggregates({}, {}, None, self.limits, omitted=1),
            }
        return scope_report(self.report(), scope)

    def _render(self, *, aggregates=False) -> dict:
        rows: dict[tuple, _UsageRow] = {}
        omitted = self._omitted
        work = 0
        buckets = 0
        interval_count = 0
        token_cache = {}

        def resolve_tokens(identity, event, remaining):
            if identity in token_cache:
                tokens, basis, limited = token_cache[identity]
                return tokens, basis, limited, 0
            tokens, basis, limited, cost = event.tokens, None, False, 0
            if event.native_tokens and self._token_maps is not None:
                basis = self._token_maps.basis_key(event.model)
                projected = self._token_maps.project(
                    event.model,
                    event.characters.state,
                    event.characters.intervals,
                    max_work=remaining,
                )
                tokens = UnitMeasurement("text_tokens", projected.state, projected.intervals)
                limited, cost = projected.limited, projected.work
            elif not event.native_tokens and tokens.state in {"measured", "partial"}:
                basis = ("supplied_intervals", event.model, None)
            token_cache[identity] = (tokens, basis, limited)
            return tokens, basis, limited, cost

        # Admission follows observation order, independent of random opaque event IDs.
        for reader, identity in self._bindings:
            event = self._events[identity]
            key = (
                reader.scope,
                reader.rule,
                reader.phase,
                event.model,
                event.operation,
                event.outcome,
            )
            if key not in rows:
                if len(rows) >= self.limits.max_rows:
                    omitted += 1
                    continue
                rows[key] = _UsageRow()
            row = rows[key]
            row.count += 1
            row.invalid += event.invalid
            if event.cut is None:
                row.cuts_complete = False
            else:
                row.cuts.add(event.cut)
            tokens, _, token_limited, cost = resolve_tokens(
                identity, event, max(0, self.limits.max_work - work)
            )
            work += cost
            if token_limited:
                row.limited.add("text_tokens")
            for measurement in [event.characters, tokens]:
                row.states[measurement.unit][measurement.state] += 1
                if not measurement.intervals:
                    continue
                coverage = row.coverage
                before = coverage.snapshot()
                coverage.max_buckets = before.buckets + self.limits.max_coverage_keys - buckets
                coverage.max_intervals = before.intervals + self.limits.max_ranges - interval_count
                coverage.max_work = before.work + self.limits.max_work - work
                try:
                    # Pass the already bounded native tuple without flattening or walking
                    # it first. The store charges its batch before validation and copying.
                    coverage.add(measurement.intervals)
                except CoverageLimitError:
                    row.limited.add(measurement.unit)
                after = coverage.snapshot()
                work += after.work - before.work
                buckets += after.buckets - before.buckets
                interval_count += after.intervals - before.intervals
        output = []
        measurements_complete = not omitted
        for key, row in sorted(rows.items(), key=lambda x: (x[0][0], x[0][1] or "", *x[0][2:])):
            scope, rule, phase, model, operation, outcome = key
            totals = row.coverage.snapshot()
            coverage = {}
            for unit in ("characters", "text_tokens"):
                states = dict(sorted(row.states[unit].items()))
                complete = unit not in row.limited and not omitted and set(states) == {"measured"}
                measurements_complete = measurements_complete and complete
                known = totals.characters if unit == "characters" else totals.text_tokens
                coverage[unit] = {
                    "value": known if complete else None,
                    "known_units": known,
                    "complete": complete,
                    "measurement_states": states,
                    "coverage_limited": unit in row.limited,
                }
            output.append(
                {
                    "scope": scope,
                    "rule": rule,
                    "phase": phase,
                    "model": model,
                    "operation": operation,
                    "outcome": outcome,
                    "observations": row.count,
                    "unique_cuts": len(row.cuts),
                    "cut_identity_complete": row.cuts_complete,
                    "invalid_measurements": row.invalid,
                    "coverage": coverage,
                }
            )
        result = {
            "version": 1,
            "finalized": False,
            "observations_complete": not omitted
            and all(row["outcome"] not in {"pending", "unknown"} for row in output),
            "measurements_complete": measurements_complete,
            "omitted_observations": omitted,
            "rows": output,
        }
        if self._token_maps is not None:
            result["token_bases"] = self._token_maps.bases()
        if aggregates:
            result["aggregates"] = build_aggregates(
                self._bindings, self._events, resolve_tokens, self.limits, omitted=omitted
            )
        return result

    def close(self) -> None:
        """Explicitly seal metadata; ordinary request exit releases its own lease instead."""
        if self._pid != os.getpid():
            return
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._closed:
            return
        for identity, event in self._events.items():
            self._events[identity] = replace(
                event,
                outcome="unknown" if event.outcome == "pending" else event.outcome,
                characters=UnitMeasurement("characters", "abandoned")
                if event.characters.state == "pending"
                else event.characters,
                tokens=UnitMeasurement("text_tokens", "abandoned")
                if event.tokens.state == "pending"
                else event.tokens,
            )
        self._final_report = self._render(aggregates=True)
        self._final_report["finalized"] = True
        self._final = {
            key: value for key, value in self._final_report.items() if key != "aggregates"
        }
        self._closed = True
        self._events.clear()
        self._bindings.clear()
        self._range_count = self._owners = 0
        self._text_inputs.close()
        if self._token_maps is not None:
            self._token_maps.close()


class ReadReplay:
    def __init__(self, source: ReadUsageSession, identities: tuple[bytes, ...], complete: bool):
        self._source = weakref.ref(source)
        self._identities = identities
        self._complete = complete

    def replay(self) -> None:
        destination = current_read_usage()
        if destination is None:
            return
        source = self._source()
        if not self._complete:
            destination._missing()
        if source is None:
            destination._missing(max(1, len(self._identities)))
        else:
            destination._replay(source, self._identities)


class ReadCapture:
    def __init__(self, session: ReadUsageSession):
        self._source = weakref.ref(session)
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._identities: dict[bytes, None] = {}
        self._complete = True
        self._result: ReadReplay | None = None

    def remember(self, session: ReadUsageSession, identity: bytes) -> None:
        if self._pid != os.getpid() or self._source() is not session:
            return
        with self._lock:
            if self._result is not None:
                self._result._complete = False
                session._missing()
                return
            if identity not in self._identities:
                if session._reserve_capture():
                    self._identities[identity] = None
                else:
                    self._complete = False

    def result(self) -> ReadReplay | None:
        if self._pid != os.getpid():
            return None
        source = self._source()
        with self._lock:
            if source is None:
                return None
            if self._result is None:
                self._result = ReadReplay(source, tuple(self._identities), self._complete)
                self._identities.clear()
            return self._result


def _capture(session: ReadUsageSession, identity: bytes) -> None:
    for capture in _CAPTURES.get():
        capture.remember(session, identity)


def replay_read_usage(replay: GroundingReadReplay | None) -> None:
    """Attribute a runtime memo hit; missing metadata cannot establish a measured zero."""
    session = current_read_usage()
    if session is None:
        return
    if replay is None:
        session._missing()
    else:
        replay.replay()


@contextmanager
def capture_read_usage() -> Generator[ReadCapture | None, None, None]:
    session = current_read_usage()
    if session is None:
        yield None
        return
    if len(_CAPTURES.get()) >= session.limits.max_capture_depth:
        session._missing()
        yield None
        return
    if not session._reserve_capture_object():
        yield None
        return
    capture = ReadCapture(session)
    token = _CAPTURES.set((*_CAPTURES.get(), capture))
    try:
        yield capture
    finally:
        try:
            capture.result()
        finally:
            _CAPTURES.reset(token)


class _LogicalGroundingObserver:
    def capture(self) -> AbstractContextManager[GroundingReadCapture | None]:
        return capture_read_usage()

    def replay(self, reads: GroundingReadReplay | None) -> None:
        replay_read_usage(reads)


_GROUNDING_OBSERVER = _LogicalGroundingObserver()


def grounding_read_observer() -> GroundingReadObserver | None:
    """Return a generic engine hook only while this request collects logical reads."""
    return _GROUNDING_OBSERVER if current_read_usage() is not None else None


def new_read_scope() -> int | None:
    session = current_read_usage()
    return None if session is None else session.new_scope()


@contextmanager
def observe_read_scopes(observer: _ScopeActivityObserver):
    """Bind an owned metadata observer; this internal hook performs no host/model work."""
    token = _SCOPE_ACTIVITY.set(observer)
    try:
        yield
    finally:
        _SCOPE_ACTIVITY.reset(token)


def retain_read_usage() -> ReadUsageLease | None:
    session = current_read_usage()
    return None if session is None else session.retain()


@contextmanager
def resume_read_usage(owner: ReadUsageLease | None) -> Generator[None, None, None]:
    """Resume explicitly owned work, suppressing any unrelated caller's accounting.

    This binds metadata only. It creates no task, performs no model/token work and
    extends no deadline or model-call budget. Missing ownership disables observation.
    """
    session = None if owner is None else owner._session()
    previous = current_read_usage()
    active_token = _ACTIVE.set(session)
    text_token = _text_read_observer.set(None if session is None else session._text_inputs)
    reader_token = captures_token = None
    if session is not previous:
        reader_token = _READER.set(_DEFAULT_READER)
        captures_token = _CAPTURES.set(())
    try:
        yield
    finally:
        _text_read_observer.reset(text_token)
        if captures_token is not None:
            _CAPTURES.reset(captures_token)
        if reader_token is not None:
            _READER.reset(reader_token)
        _ACTIVE.reset(active_token)


@contextmanager
def read_usage_scope(rule: str | None, *, scope: int | None = 0, phase: str = "decision"):
    valid = (
        _label(rule, optional=True)
        and type(scope) is int
        and 0 <= scope < 1 << 31
        and type(phase) is str
        and phase in _PHASES
    )
    token = _READER.set(_Reader(scope, rule, phase) if valid else None)
    try:
        yield
    finally:
        _READER.reset(token)


@contextmanager
def model_read_usage(
    limits: ReadUsageLimits | None = None,
    *,
    token_bases: dict[str, TextTokenBasis] | None = None,
) -> Generator[ReadUsageSession, None, None]:
    session = ReadUsageSession(limits, token_bases=token_bases)
    owner = session.retain()
    assert owner is not None  # A fresh, validated configuration admits its first owner.
    try:
        with resume_read_usage(owner):
            yield session
    finally:
        owner.close()


def _reset_read_usage_after_fork() -> None:
    _ACTIVE.set(None)
    _READER.set(_DEFAULT_READER)
    _CAPTURES.set(())
    _SCOPE_ACTIVITY.set(None)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_read_usage_after_fork)
