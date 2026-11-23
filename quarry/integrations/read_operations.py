# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Owned logical operations and explicit, single-use dispatch-layer adoption."""

from __future__ import annotations

import asyncio
import os
import threading
import weakref
from contextvars import ContextVar

from quarry.integrations.read_inputs import (
    InputMeasurement,
    collect_field,
    collect_messages,
    collect_tools,
    collect_transform_part,
    collect_value,
)
from quarry.integrations.read_usage import (
    ReadTicket,
    ReadUsageLease,
    ReadUsageSession,
    UnitMeasurement,
    current_read_usage,
)
from quarry.types.adapter_transforms import ModelTransformConfig, adapter_transform_reference_parts
from quarry.types.exceptions import MODEL_REFUSED, LLMError


class _Disabled:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


_DISABLED = _Disabled()


def _task():
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


class ModelRead:
    """Keep bounded scalar metadata and one lease, never the operation's input/result."""

    def __init__(
        self,
        session: ReadUsageSession,
        ticket: ReadTicket,
        lease: ReadUsageLease | None,
        model: str,
        operation: str,
        streaming: bool = False,
    ):
        self._pid = os.getpid()
        self._session = weakref.ref(session)
        self._ticket, self._lease = ticket, lease
        self.model, self.operation = model, operation
        self._reader = session.reader_stamp()
        self._inputs: dict[str, InputMeasurement] = {}
        self._not_sent = False
        self._closed = False
        self._limited = False
        self._characters = UnitMeasurement("characters", "unavailable")
        self._cut = None
        self._stream_open = streaming
        self._parts_requested = False
        self._complete_characters = self._characters
        self._complete_cut = None
        self._external_partial = False

    def _slot(self, slot: str) -> bool:
        if type(slot) is str and slot in {"input", "context", "constitution", "tools"}:
            return True
        session = self._session()
        if session is not None:
            session._missing()
        return False

    def messages(self, messages: object, *, slot: str = "input") -> None:
        if self._limited or not self._slot(slot) or slot in self._inputs:
            return
        session = self._session()
        if session is not None and self._pid == os.getpid():
            self._add(slot, collect_messages(session, messages))

    def value(self, kind: str, value: object, *, slot: str) -> None:
        if self._limited or not self._slot(slot) or slot in self._inputs:
            return
        session = self._session()
        if session is not None and self._pid == os.getpid():
            self._add(slot, collect_value(session, kind, value))

    def unavailable(self, *, slot: str) -> None:
        if not self._limited and self._slot(slot) and slot not in self._inputs:
            self._add(slot, InputMeasurement(UnitMeasurement("characters", "unavailable"), None))

    def external_input(self, source, coverage, *, complete):
        if self._limited:
            return
        session = self._session()
        if session is not None and self._pid == os.getpid():
            from quarry.integrations.read_external_spans import collect_external_input

            self._external_partial = not complete
            self._add("input", collect_external_input(session, source, coverage, complete=complete))

    def tools(self, tools: object) -> None:
        if self._limited:
            return
        previous = self._inputs.get("tools")
        if previous is not None and previous.characters.state == "measured":
            return
        session = self._session()
        if session is not None and self._pid == os.getpid():
            self._add("tools", collect_tools(session, tools))

    def window_pending(self, unit):
        session = self._session()
        if session is not None and not self._closed and self._pid == os.getpid():
            try:
                session._window_pending(self._ticket._identity, unit)
            except Exception:
                self._omit_window()

    def window_input(self, source, windows, encoding, context="", *, unit=None) -> None:
        if self._limited:
            return
        session = self._session()
        if session is not None and self._pid == os.getpid():
            from quarry.integrations.read_windows import collect_window_input

            measurement = collect_window_input(session, source, windows, encoding)
            self._add("input", measurement)
            try:
                session._window_input(
                    self._ticket._identity, unit, windows, encoding, measurement.characters
                )
            except Exception:
                self._omit_window()
            if context:
                self.value("window-context", context, slot="context")

    def _omit_window(self):
        session = self._session()
        if session is not None and session._pid == os.getpid():
            with session._lock:
                if not session._closed and session._windows is not None:
                    session._windows.omitted = min(session._windows.omitted + 1, 1 << 63)

    def window_reply(self, generation, provider_model, chunks, cached):
        session = self._session()
        if session is not None and not self._closed and self._pid == os.getpid():
            try:
                session._window_reply(
                    self._ticket._identity, generation, provider_model, chunks, cached
                )
            except Exception:
                self._omit_window()

    def window_vector(self, generation, ordinal, probabilities):
        session = self._session()
        if session is not None and not self._closed and self._pid == os.getpid():
            try:
                session._window_vector(generation, ordinal, probabilities)
            except Exception:
                self._omit_window()

    def _measurement(self, session):
        ordered = sorted(self._inputs.items())
        complete = bool(ordered) and all(item.characters.state == "measured" for _, item in ordered)
        count = sum(len(item.characters.intervals) for _, item in ordered)
        if count > min(session.limits.max_ranges, 65_536):
            return UnitMeasurement("characters", "limit_exceeded"), None
        ranges = tuple(interval for _, item in ordered for interval in item.characters.intervals)
        if complete:
            characters = UnitMeasurement("characters", "measured", ranges)
        elif (
            ranges
            or self._external_partial
            and all(item.characters.state in {"measured", "partial"} for _, item in ordered)
        ):
            characters = UnitMeasurement("characters", "partial", ranges)
        else:
            states = {item.characters.state for _, item in ordered}
            state = next(
                (
                    state
                    for state in ("limit_exceeded", "failed", "cancelled", "unavailable")
                    if state in states
                ),
                "unavailable",
            )
            characters = UnitMeasurement("characters", state)
        cuts = tuple(item.cut for _, item in ordered if item.cut is not None)
        cut = (
            session._text_inputs.registry.origin_identity(None, cuts)
            if complete and len(cuts) == len(ordered)
            else None
        )
        return characters, cut

    def _add(self, slot, measurement):
        session = self._session()
        if session is None:
            return
        previous = self._inputs.get(slot)
        self._inputs[slot] = measurement
        try:
            characters, cut = self._measurement(session)
            if self._stage(characters, cut):
                return
            self._limited = True
            if previous is None:
                self._inputs[slot] = InputMeasurement(
                    UnitMeasurement("characters", "limit_exceeded"), None
                )
            else:
                self._inputs[slot] = InputMeasurement(
                    UnitMeasurement("characters", "partial", previous.characters.intervals), None
                )
            characters, cut = self._measurement(session)
            if not self._stage(characters, cut):
                self._inputs.clear()
                self._characters, self._cut = UnitMeasurement("characters", "limit_exceeded"), None
                self._complete_characters, self._complete_cut = self._characters, None
                self._inputs["input"] = InputMeasurement(self._characters, None)
        except Exception as error:
            self._limited = True
            self._inputs.clear()
            self._characters, self._cut = (
                UnitMeasurement("characters", session._text_inputs._failed(error)),
                None,
            )
            self._inputs["input"] = InputMeasurement(self._characters, None)
            self._complete_characters, self._complete_cut = self._characters, None
            session._missing()

    def _stage(self, characters, cut):
        staged = (
            UnitMeasurement("characters", "partial", characters.intervals)
            if self._stream_open and characters.state == "measured"
            else characters
        )
        staged_cut = None if self._stream_open else cut
        if not self._ticket.stage_input(staged, staged_cut):
            return False
        self._characters, self._cut = staged, staged_cut
        self._complete_characters, self._complete_cut = characters, cut
        return True

    def append_part(self, measurement: InputMeasurement) -> None:
        self._parts_requested = True
        if self._limited:
            return
        session = self._session()
        if session is None:
            return
        previous = self._inputs.get("input")
        if previous is None:
            self._add("input", measurement)
            return
        try:
            before, added = previous.characters, measurement.characters
            if len(before.intervals) + len(added.intervals) > min(
                session.limits.max_ranges, 65_536
            ):
                self._limited = True
                self._add(
                    "input",
                    InputMeasurement(
                        UnitMeasurement("characters", "partial", before.intervals), None
                    ),
                )
                return
            intervals = before.intervals + added.intervals
            complete = before.state == added.state == "measured"
            state = "measured" if complete else "partial" if intervals else "unavailable"
            cut = (
                session._text_inputs.registry.origin_identity(None, (previous.cut, measurement.cut))
                if complete and previous.cut is not None and measurement.cut is not None
                else None
            )
            self._add(
                "input", InputMeasurement(UnitMeasurement("characters", state, intervals), cut)
            )
        except Exception as error:
            self._limited = True
            self._add(
                "input",
                InputMeasurement(
                    UnitMeasurement("characters", session._text_inputs._failed(error)), None
                ),
            )

    def complete_input(self) -> None:
        if not self._stream_open:
            return
        self._stream_open = False
        if not self._parts_requested:
            self._not_sent = True
            session = self._session()
            if session is not None:
                try:
                    cut = session._text_inputs.registry.origin_identity(None, ())
                    self._add(
                        "input", InputMeasurement(UnitMeasurement("characters", "measured"), cut)
                    )
                except Exception as error:
                    self._add(
                        "input",
                        InputMeasurement(
                            UnitMeasurement("characters", session._text_inputs._failed(error)), None
                        ),
                    )
        else:
            # Prepared before provider waits; successful finalization never rehashes raw
            # inputs or consumes identity work exhausted by another concurrent request.
            self._characters, self._cut = self._complete_characters, self._complete_cut

    def not_sent(self) -> None:
        self._not_sent = True

    def _finish(self, error: BaseException | None) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            session = self._session()
            if session is None or self._pid != os.getpid():
                return
            outcome = "not_sent" if self._not_sent else "succeeded"
            if error is not None:
                outcome = (
                    "cancelled" if issubclass(type(error), asyncio.CancelledError) else "failed"
                )
                if self.operation in {"score", "complete"} and issubclass(type(error), LLMError):
                    try:
                        storage = object.__getattribute__(error, "__dict__")
                        details = dict.get(storage, "details")
                        code = dict.get(details, "code") if type(details) is dict else None
                        if type(code) is str and code == MODEL_REFUSED:
                            outcome = "refused"
                    except Exception:
                        pass
            self._ticket.finish(
                outcome,
                cut=self._cut,
                characters=self._characters,
                tokens=UnitMeasurement("text_tokens", "unavailable"),
            )
        except Exception:
            # Optional metadata never replaces a native result or native exception.
            session = self._session()
            if session is not None:
                session._missing()
        finally:
            self._inputs.clear()
            self._characters = UnitMeasurement("characters", "unavailable")
            self._cut = None
            self._complete_characters = UnitMeasurement("characters", "unavailable")
            self._complete_cut = None
            if self._lease is not None:
                self._lease.close()


_CURRENT: ContextVar[ModelRead | None] = ContextVar("quarry_model_read_operation", default=None)


class _Handoff:
    def __init__(self, read: ModelRead):
        self.read = read
        self.pid, self.thread = os.getpid(), threading.get_ident()
        task = _task()
        self.task = None if task is None else weakref.ref(task)
        self.claimed = False

    def claim(self, session, model, operation):
        same_task = _task() is (None if self.task is None else self.task())
        if (
            not self.claimed
            and self.pid == os.getpid()
            and self.thread == threading.get_ident()
            and same_task
            and self.read is _CURRENT.get()
            and not self.read._closed
            and self.read._session() is session
            and self.read._reader == session.reader_stamp()
            and self.read.model == model
            and self.read.operation == operation
        ):
            self.claimed = True
            return self.read
        return None


_PENDING: ContextVar[_Handoff | None] = ContextVar("quarry_model_read_handoff", default=None)


class _FieldInput:
    def __init__(self, session, model, measurement):
        self.session = weakref.ref(session)
        self.model, self.measurement = model, measurement
        self.reader, self.pid = session.reader_stamp(), os.getpid()
        self.thread = threading.get_ident()
        task = _task()
        self.task = None if task is None else weakref.ref(task)

    def take(self, session, model, operation):
        if type(model) is not str or self.model != model or operation != "classify":
            return None
        if (
            self.pid == os.getpid()
            and self.thread == threading.get_ident()
            and _task() is (None if self.task is None else self.task())
            and self.session() is session
            and self.reader == session.reader_stamp()
        ):
            measurement, self.measurement = self.measurement, None
            return measurement
        return None


_FIELD: ContextVar[_FieldInput | None] = ContextVar("quarry_model_field_input", default=None)


class _FieldScope:
    def __init__(self, carrier):
        self.carrier = carrier

    def __enter__(self):
        self.token = _FIELD.set(self.carrier)

    def __exit__(self, *_args):
        self.carrier.measurement = None
        _FIELD.reset(self.token)
        return False


class _HandoffScope:
    def __init__(self, read):
        self.read = read

    def __enter__(self):
        self.token = _PENDING.set(_Handoff(self.read))

    def __exit__(self, *_args):
        _PENDING.reset(self.token)
        return False


class _Operation:
    def __init__(self, session, model, operation, adopt, streaming):
        self.session, self.model, self.operation, self.adopt = session, model, operation, adopt
        self.read = None
        self.owned = False
        self.streaming = streaming

    def __enter__(self):
        pending = _PENDING.get()
        self.pending_token = _PENDING.set(None)
        try:
            self.read = (
                pending.claim(self.session, self.model, self.operation)
                if self.adopt and pending is not None
                else None
            )
            if self.read is None:
                lease = self.session.retain()
                try:
                    ticket = self.session.begin(self.model, self.operation, native_tokens=True)
                    if ticket is not None:
                        self.read = ModelRead(
                            self.session,
                            ticket,
                            lease,
                            self.model,
                            self.operation,
                            self.streaming,
                        )
                        self.owned = True
                finally:
                    if not self.owned and lease is not None:
                        lease.close()
        except Exception:
            self.session._missing()
        self.current_token = _CURRENT.set(self.read)
        try:
            source = _FIELD.get()
            if source is not None:
                measurement = source.take(self.session, self.model, self.operation)
                if (
                    self.read is not None
                    and measurement is not None
                    and "input" not in self.read._inputs
                ):
                    self.read._add("input", measurement)
        except Exception:
            self.session._missing()
        return self.read

    def __exit__(self, _kind, error, _traceback):
        try:
            if self.owned and self.read is not None:
                try:
                    if error is None:
                        self.read.complete_input()
                except Exception:
                    self.session._missing()
                self.read._finish(error)
        finally:
            _CURRENT.reset(self.current_token)
            _PENDING.reset(self.pending_token)
        return False


def model_read(model: str, operation: str, *, adopt: bool = False, streaming: bool = False):
    session = current_read_usage()
    return _DISABLED if session is None else _Operation(session, model, operation, adopt, streaming)


def handoff_model_read():
    read = current_model_read()
    return _DISABLED if read is None or read._closed else _HandoffScope(read)


def transfer_model_read():
    """Consume an owned dispatch handoff and bind its replacement at the destination.

    Copying a Context alone must never grant adoption to another task. The submitting
    task first claims its existing single-use handoff with all normal ownership checks;
    entering this scope then creates a handoff owned by the actual host invocation.
    """
    read = current_model_read()
    pending = _PENDING.get()
    if read is None or pending is None:
        return _DISABLED
    session = current_read_usage()
    claimed = pending.claim(session, read.model, read.operation)
    return _DISABLED if claimed is None else _TransferredRead(claimed)


def current_model_read() -> ModelRead | None:
    read = _CURRENT.get()
    session = current_read_usage()
    return (
        read
        if (
            read is not None
            and not read._closed
            and read._pid == os.getpid()
            and read._session() is session
            and session is not None
            and read._reader == session.reader_stamp()
        )
        else None
    )


_ABSENT = object()


def record_model_input(
    messages: object,
    *,
    constitution: object = _ABSENT,
    tools: object = _ABSENT,
    context: object = None,
) -> None:
    """Called after native guards, before work on a logical operation's input."""
    read = current_model_read()
    if read is None:
        return
    read.messages(messages)
    if constitution is not _ABSENT:
        read.value("constitution", constitution, slot="constitution")
    if tools is not _ABSENT and tools is not None:
        if type(tools) is list or type(tools) is tuple:
            if len(tools):
                read.tools(tools)
        else:
            read.tools(tools)
    if context is not None:
        read.messages(context, slot="context")


def record_model_text(text: object) -> None:
    read = current_model_read()
    if read is not None:
        read.value("input-text", text, slot="input")


def _transform_alias(backend, name):
    session = current_read_usage()
    if session is None:
        return None
    try:
        if type(name) is not str:
            return None
        definition = backend.config.transforms.get(name)
        return definition.model if type(definition) is ModelTransformConfig else None
    except Exception:
        session._missing()
        return None


def transform_model_read(backend, name, *, streaming=False):
    alias = _transform_alias(backend, name)
    return (
        _DISABLED
        if alias is None
        else model_read(alias, "transform", adopt=True, streaming=streaming)
    )


def handoff_transform_read(backend, name):
    read = current_model_read()
    if (
        read is None
        or read.operation != "transform"
        or _transform_alias(backend, name) != read.model
    ):
        return _DISABLED
    return _HandoffScope(read)


def expression_transform_read(reference):
    session = current_read_usage()
    if session is None or reference is None:
        return _DISABLED
    try:
        _, alias = adapter_transform_reference_parts(reference, True)
        return model_read(alias, "transform", streaming=True)
    except Exception:
        session._missing()
        return _DISABLED


class _PartInput:
    def __init__(self, read, signature):
        self.read = weakref.ref(read)
        self.signature = signature
        self.pid, self.thread = os.getpid(), threading.get_ident()
        task = _task()
        self.task = None if task is None else weakref.ref(task)
        self.claimed = False

    def matches(self, read):
        return (
            not self.claimed
            and self.read() is read
            and not read._closed
            and self.pid == os.getpid()
            and self.thread == threading.get_ident()
            and _task() is (None if self.task is None else self.task())
        )

    def claim(self, read, unit):
        if not self.matches(read):
            return False
        self.claimed = True
        signature, self.signature = self.signature, None
        session = read._session()
        if session is not None:
            current = collect_transform_part(session, unit)
            if signature is None or current.cut != signature:
                read.append_part(
                    InputMeasurement(UnitMeasurement("characters", "unavailable"), None)
                )
        return True


_PART: ContextVar[_PartInput | None] = ContextVar("quarry_transform_part_input", default=None)


class _BatchInput:
    def __init__(self, read, signatures):
        self.read = weakref.ref(read)
        self.signatures = signatures
        self.pid, self.thread = os.getpid(), threading.get_ident()

    def matches(self, read):
        return (
            self.read() is read
            and not read._closed
            and self.pid == os.getpid()
            and self.thread == threading.get_ident()
        )

    def take(self, read, index):
        if (
            not self.matches(read)
            or type(index) is not int
            or not 0 <= index < len(self.signatures)
        ):
            return False, None
        signature = self.signatures[index]
        if signature is _CONSUMED_PART:
            return False, None
        self.signatures[index] = _CONSUMED_PART
        return True, signature


_CONSUMED_PART = object()
_BATCH: ContextVar[_BatchInput | None] = ContextVar("quarry_transform_batch_input", default=None)


class _BatchScope:
    def __init__(self, read, signatures):
        self.read = read
        self.carrier = _BatchInput(read, signatures)

    def __enter__(self):
        self.batch_token = _BATCH.set(self.carrier)
        self.handoff_token = _PENDING.set(_Handoff(self.read))

    def __exit__(self, *_args):
        self.carrier.signatures.clear()
        _PENDING.reset(self.handoff_token)
        _BATCH.reset(self.batch_token)
        return False


class _PartScope:
    def __init__(self, read, signature):
        self.read = read
        self.carrier = _PartInput(read, signature)

    def __enter__(self):
        self.part_token = _PART.set(self.carrier)
        self.handoff_token = _PENDING.set(_Handoff(self.read))

    def __exit__(self, *_args):
        self.carrier.claimed = True
        self.carrier.signature = None
        _PENDING.reset(self.handoff_token)
        _PART.reset(self.part_token)
        return False


class _TransferredRead:
    """Move only owned, bounded transform signatures with the approved read handoff."""

    def __init__(self, read: ModelRead):
        self.read = read
        self.part_signature = _ABSENT
        self.batch_signatures = None
        part = _PART.get()
        if part is not None and part.matches(read):
            self.part_signature = part.signature
            part.claimed, part.signature = True, None
        batch = _BATCH.get()
        if self.part_signature is _ABSENT and batch is not None and batch.matches(read):
            self.batch_signatures, batch.signatures = batch.signatures, []

    def __enter__(self):
        read = self.read
        if read is None:
            raise RuntimeError("Read handoff was already released")
        self.handoff = _Handoff(read)
        self.part = (
            None if self.part_signature is _ABSENT else _PartInput(read, self.part_signature)
        )
        self.batch = (
            None if self.batch_signatures is None else _BatchInput(read, self.batch_signatures)
        )
        self.handoff_token = _PENDING.set(self.handoff)
        self.part_token = _PART.set(self.part)
        self.batch_token = _BATCH.set(self.batch)

    def __exit__(self, *_args):
        try:
            self.handoff.claimed = True
            if self.part is not None:
                self.part.claimed, self.part.signature = True, None
            if self.batch_signatures is not None:
                self.batch_signatures.clear()
        finally:
            _BATCH.reset(self.batch_token)
            _PART.reset(self.part_token)
            _PENDING.reset(self.handoff_token)
            del self.batch_token, self.part_token, self.handoff_token
            self.read, self.part_signature, self.batch_signatures = None, _ABSENT, None
            del self.handoff, self.part, self.batch
        return False


def transform_source_positions(backend, name, messages):
    read = current_model_read()
    if (
        read is None
        or read.operation != "transform"
        or _transform_alias(backend, name) != read.model
    ):
        return None
    session = read._session()
    if session is None:
        return None
    try:
        from quarry.types.messages import Messages

        if type(messages) is not Messages or type(messages._root) is not Messages:
            return None
        session._text_inputs.registry.inspect(len(messages) + 1)
        return {
            id(message): ordinal
            for message, ordinal in zip(messages, messages._ordinals, strict=True)
        }
    except Exception:
        session._missing()
        return None


def transform_part_read(
    backend,
    name,
    unit,
    *,
    messages=None,
    message=None,
    index=None,
    positions=None,
    batch_index=None,
):
    read = current_model_read()
    if (
        read is None
        or read.operation != "transform"
        or _transform_alias(backend, name) != read.model
    ):
        return _DISABLED
    pending = _PART.get()
    if pending is not None and pending.matches(read):
        return _HandoffScope(read)
    session = read._session()
    if session is None:
        return _DISABLED
    try:
        batch = _BATCH.get()
        if batch is not None and batch_index is not None:
            claimed, signature = batch.take(read, batch_index)
            if claimed:
                return _PartScope(read, signature)
        origin = None
        if positions is not None and message is not None and messages is not None:
            ordinal = positions.get(id(message))
            if ordinal is not None:
                origin = messages._root._origins[ordinal]
        measured = collect_transform_part(session, unit, message, index, origin)
        signature = collect_transform_part(session, unit).cut
        read.append_part(measured)
        return _PartScope(read, signature)
    except Exception:
        session._missing()
        read.append_part(InputMeasurement(UnitMeasurement("characters", "failed"), None))
        return _PartScope(read, None)


def transform_batch_read(backend, name, units, *, messages, message, positions):
    read = current_model_read()
    if (
        read is None
        or read.operation != "transform"
        or _transform_alias(backend, name) != read.model
    ):
        return _DISABLED
    session = read._session()
    if session is None:
        return _DISABLED
    signatures = []
    try:
        # Native apply_adapter_transform batches contain at most16already-checked parts.
        if type(units) is not list or len(units) > 16:
            raise ValueError("invalid logical transform batch")
        origin = None
        if positions is not None:
            ordinal = positions.get(id(message))
            if ordinal is not None:
                origin = messages._root._origins[ordinal]
        for index, unit in units:
            measured = collect_transform_part(session, unit, message, index, origin)
            signatures.append(collect_transform_part(session, unit).cut)
            read.append_part(measured)
        return _BatchScope(read, signatures)
    except Exception:
        session._missing()
        read.append_part(InputMeasurement(UnitMeasurement("characters", "failed"), None))
        return _BatchScope(read, [None] * min(len(signatures), 16))


def record_transform_unit(unit):
    read = current_model_read()
    if read is None or read.operation != "transform":
        return
    try:
        pending = _PART.get()
        if pending is not None and pending.claim(read, unit):
            return
        session = read._session()
        if session is not None:
            read.append_part(collect_transform_part(session, unit))
    except Exception:
        session = read._session()
        if session is not None:
            session._missing()
        read.append_part(InputMeasurement(UnitMeasurement("characters", "failed"), None))


def model_field_input(model: str, event, field: str, text: object, section: str):
    session = current_read_usage()
    if session is None:
        return _DISABLED
    try:
        measured = collect_field(
            session,
            event.source if event is not None else None,
            field,
            text,
            section,
            event.position if event is not None else 0,
        )
        return _FieldScope(_FieldInput(session, model, measured))
    except Exception:
        session._missing()
        measured = InputMeasurement(UnitMeasurement("characters", "failed"), None)
        return _FieldScope(_FieldInput(session, model, measured))


def _reset_after_fork() -> None:
    _CURRENT.set(None)
    _PENDING.set(None)
    _FIELD.set(None)
    _PART.set(None)
    _BATCH.set(None)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
