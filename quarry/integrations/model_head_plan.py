# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Task-local symbolic input plans for optional classifier token heads."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from quarry.cache.keys import input_key
from quarry.engine.atoms import ModelExpression
from quarry.engine.events import ConversationEvent, build_static_environment


class HeadFrame(NamedTuple):
    messages: bytes
    tools: bytes


ROOT_FRAME = HeadFrame(input_key("head-root-messages"), input_key("head-root-tools"))
_EMPTY_MESSAGES = input_key("head-empty-messages")
_EMPTY_TOOLS = input_key("head-empty-tools")
_BINDERS = frozenset(
    {
        "map",
        "message_map",
        "message_select",
        "split_any",
        "span_any",
        "span_all",
        "span_filter",
        "call_filter",
        "call_any",
        "call_all",
    }
)
_MESSAGE_VIEWS = frozenset(
    {
        "user_messages",
        "assistant_messages",
        "system_messages",
        "developer_messages",
        "tool_responses",
    }
)


def input_signature(expression, frame=ROOT_FRAME, bindings=None):
    """Compose bounded digests without reading messages or expanding frame recipes."""
    memo = {}

    def visit(node, environment):
        identity = id(node), tuple(sorted(environment.items()))
        if identity in memo:
            return memo[identity]
        if node.kind == "messages":
            result = frame.messages
        elif node.kind == "tools":
            result = frame.tools
        elif node.kind == "message_view":
            result = input_key("head-view", node.value, frame.messages)
        elif node.kind in {"tool_calls", "system_prompt"}:
            result = input_key("head-surface", node.kind, frame.messages)
        elif node.kind == "slice" and node.children[1].value in (None, 0):
            stop = node.children[2].value
            result = (
                _EMPTY_MESSAGES
                if stop == 0
                else visit(node.children[0], environment)
                if stop is None
                else input_key(
                    node.kind, node.value, tuple(visit(c, environment) for c in node.children)
                )
            )
        elif node.kind == "concat":
            parts = tuple(visit(c, environment) for c in node.children)
            parts = tuple(p for p in parts if p != _EMPTY_MESSAGES)
            result = (
                _EMPTY_MESSAGES
                if not parts
                else parts[0]
                if len(parts) == 1
                else input_key(node.kind, node.value, parts)
            )
        elif (
            node.kind == "flatten_messages"
            and visit(node.children[0], environment) == _EMPTY_MESSAGES
        ):
            result = _EMPTY_MESSAGES
        elif node.kind == "binding":
            result = environment.get(str(node.value))
            if result is None:
                # Free bindings still require an admitted event or caller scope.
                result = input_key("head-unplanned-binding", id(node), frame)
        elif node.kind in _BINDERS:
            source = visit(node.children[0], environment)
            nested = {**environment, str(node.value): input_key("head-bound-item", source)}
            result = input_key(
                node.kind, None, (source, *(visit(c, nested) for c in node.children[1:]))
            )
        else:
            result = input_key(
                node.kind, node.value, tuple(visit(c, environment) for c in node.children)
            )
        memo[identity] = result
        return result

    return visit(expression, {} if bindings is None else bindings)


def child_frame(expression, frame=ROOT_FRAME):
    return HeadFrame(input_signature(expression, frame), _EMPTY_TOOLS)


class _HeadPlan(NamedTuple):
    owner: int
    requests: frozenset[tuple[bytes, bytes]]


_plan = ContextVar[_HeadPlan | None]("quarry_token_head_plan", default=None)
_input: ContextVar[_HeadInput | None] = ContextVar("quarry_token_head_input", default=None)
_frame = ContextVar("quarry_token_head_frame", default=ROOT_FRAME)
_event: ContextVar[_EventInput | None] = ContextVar("quarry_token_head_event", default=None)


def _task_id():
    try:
        return id(asyncio.current_task())
    except RuntimeError:
        return None


@dataclass
class _EventInput:
    event: ConversationEvent
    context: Any
    sizes: dict[str, int] = field(default_factory=dict)


@dataclass
class _HeadInput:
    candidates: Callable[[], Iterator[bytes]]
    pid: int = field(default_factory=os.getpid)
    thread: int = field(default_factory=threading.get_ident)
    task: int | None = field(default_factory=_task_id)
    claimed: bool = False

    def claim(self):
        if (
            self.claimed
            or self.pid != os.getpid()
            or self.thread != threading.get_ident()
            or self.task != _task_id()
        ):
            return ()
        self.claimed = True
        return self.candidates()


@contextmanager
def resume_head_plan(value):
    token = _plan.set(value)
    input_token = _input.set(None)
    frame_token = _frame.set(ROOT_FRAME)
    event_token = _event.set(None)
    try:
        yield
    finally:
        _event.reset(event_token)
        _frame.reset(frame_token)
        _input.reset(input_token)
        _plan.reset(token)


def head_plan_scope(owner, requests):
    return resume_head_plan(_HeadPlan(id(owner), frozenset(requests)))


def capture_head_plan():
    return _plan.get()


def head_plan_active() -> bool:
    return _plan.get() is not None


def head_requests(owner):
    value = _plan.get()
    return value.requests if value is not None and value.owner == id(owner) else None


def current_head_input():
    value = _input.get()
    return () if value is None else value.claim()


@contextmanager
def head_event(event, context):
    if not head_plan_active() or event is None:
        token = _event.set(None)
    else:
        token = _event.set(_EventInput(event, context))
    try:
        yield
    finally:
        _event.reset(token)


def _surface_signature(surface, frame):
    if surface == "messages":
        return frame.messages
    if surface in _MESSAGE_VIEWS:
        return input_key("head-view", surface, frame.messages)
    if surface in {"tool_calls", "system_prompt"}:
        return input_key("head-surface", surface, frame.messages)
    return input_key("input_state", surface, ())


@contextmanager
def head_input(expression, event_names=()):
    frame, event_input = _frame.get(), _event.get()

    def candidates():
        yield input_signature(expression, frame)
        if not event_names or event_input is None:
            return
        event, context = event_input.event, event_input.context
        for surface, ordinal in event.ordinals.items():
            source = _surface_signature(surface, frame)
            item = input_key("index", ordinal, (source,))
            yield input_signature(expression, frame, dict.fromkeys(event_names, item))
        # Negative ordinals need only lazy surface sizes, never message text or hooks.
        # They are not read when a positive alias already matches the selected plan.
        for surface, ordinal in event.ordinals.items():
            if surface not in event_input.sizes:
                event_input.sizes[surface] = build_static_environment(context).surface_sizes[
                    surface
                ]
            source = _surface_signature(surface, frame)
            item = input_key("index", ordinal - event_input.sizes[surface], (source,))
            yield input_signature(expression, frame, dict.fromkeys(event_names, item))

    token = _input.set(_HeadInput(candidates))
    try:
        yield
    finally:
        _input.reset(token)


@contextmanager
def legacy_head_input(event, context, field):
    """Plan the version-2 single-user-message form of a legacy field classifier."""
    if not head_plan_active():
        yield
        return
    name = "__quarry_legacy_head_event"
    expression = ModelExpression(
        "message",
        children=(
            ModelExpression("literal", "user"),
            ModelExpression("field", field, (ModelExpression("binding", name),)),
        ),
    )
    with head_event(event, context), head_input(expression, (name,)):
        yield


@contextmanager
def head_frame(expression):
    if not head_plan_active():
        yield
        return
    token = _frame.set(child_frame(expression, _frame.get()))
    try:
        yield
    finally:
        _frame.reset(token)


def _reset_after_fork():
    _plan.set(None)
    _input.set(None)
    _frame.set(ROOT_FRAME)
    _event.set(None)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
