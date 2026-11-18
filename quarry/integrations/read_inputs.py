# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Bounded semantic model input collection, without constructing provider payloads."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import cast

from quarry.input_registry import InputObject
from quarry.integrations.read_identity import ReadIdentityLimitError, ReadIdentityUnavailable
from quarry.integrations.read_usage import ReadUsageSession, UnitMeasurement
from quarry.types.messages import (
    AssistantMessage,
    DeveloperMessage,
    Message,
    Messages,
    SystemMessage,
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolFunction,
    ToolMessage,
    UserMessage,
    _FrozenDict,
)
from quarry.types.origins import Place
from quarry.types.results import InputText

_MESSAGES = (SystemMessage, DeveloperMessage, UserMessage, AssistantMessage, ToolMessage)


@dataclass(frozen=True, slots=True)
class InputMeasurement:
    characters: UnitMeasurement
    cut: bytes | None


def collect_field(
    session: ReadUsageSession, owner: object, field: str, text: object, section: str, position: int
) -> InputMeasurement:
    """Name a scanner field already read under the oracle's native semantic guards."""
    native = session._text_inputs
    try:
        from quarry.integrations.read_origins import ReadOriginProjection

        if type(text) is not str or type(field) is not str:
            return InputMeasurement(UnitMeasurement("characters", "unavailable"), None)
        source = None
        if type(owner) is InputObject:
            fields = object.__getattribute__(owner, "_fields")
            item = fields.get("id")
            source = InputText(section, item if type(item) is str else None, (field,))
        elif type(owner) in _MESSAGES:
            if field in {"content", "result"}:
                field = ""
            elif type(owner) is not AssistantMessage or field != "refusal":
                return InputMeasurement(UnitMeasurement("characters", "unavailable"), None)
        origin = Place(position, field, 0, owner, source)
        projector = ReadOriginProjection(native.registry, native._parts, native._ensure)
        intervals = projector.ranges(text, origin)
        if not intervals and not text and session._token_maps is not None:
            scope = session.input_scope()
            if scope is not None:
                intervals = (native.registry.value_interval(scope, "empty-input-text", ""),)
        return InputMeasurement(
            UnitMeasurement("characters", "measured", intervals), projector.identity(origin)
        )
    except Exception as error:
        return InputMeasurement(UnitMeasurement("characters", native._failed(error)), None)


def collect_value(session: ReadUsageSession, kind: str, value: object) -> InputMeasurement:
    """Authored scalar arguments have value identity, independent of source occurrences."""
    native = session._text_inputs
    try:
        scope = session.input_scope()
        if scope is None or type(value) is not str:
            return InputMeasurement(UnitMeasurement("characters", "unavailable"), None)
        interval = native.registry.value_interval(scope, kind, value)
        return InputMeasurement(
            UnitMeasurement("characters", "measured", (interval,)), interval.source
        )
    except Exception as error:
        return InputMeasurement(UnitMeasurement("characters", native._failed(error)), None)


def collect_transform_part(
    session: ReadUsageSession,
    unit: object,
    message: object = None,
    index: int | None = None,
    origin: object = None,
) -> InputMeasurement:
    """Collect exactly one selected native part; binary encodings are not text content."""
    native, registry = session._text_inputs, session._text_inputs.registry
    try:
        scope = session.input_scope()
        if scope is None or type(unit) is not dict:
            raise ReadIdentityUnavailable("logical transform input is unavailable")
        registry.inspect(len(unit) + 1)
        if any(type(key) is not str for key in unit):
            raise ReadIdentityUnavailable("logical transform input is unavailable")
        kind = dict.get(unit, "type")
        if type(kind) is not str or kind not in {"text", "image_url", "input_audio"}:
            raise ReadIdentityUnavailable("logical transform input is unavailable")
        if set(unit) != {"type", "text" if kind == "text" else kind}:
            raise ReadIdentityUnavailable("logical transform input is unavailable")
        keys: list[bytes | int] = [
            registry.value_interval(scope, "transform-part-type", kind).source
        ]
        if message is None:
            if kind == "text":
                return collect_value(session, "input-text", dict.get(unit, "text"))
        elif type(message) not in _MESSAGES:
            raise ReadIdentityUnavailable("logical transform source is unavailable")
        if kind == "text":
            message = cast(Message, message)
            text = dict.get(unit, "text")
            content = message.source_content
            if type(text) is not str:
                raise ReadIdentityUnavailable("logical transform text is unavailable")
            registry.inspect(len(text) + 1)
            if type(content) is str and index is None:
                start, selected = 0, content
            elif type(content) is tuple and type(index) is int and 0 <= index < len(content):
                registry.inspect(len(content))
                start = 0
                for position in range(index):
                    before = content[position]
                    if type(before) is not _FrozenDict:
                        raise ReadIdentityUnavailable("logical transform source is unavailable")
                    if before.get("type") == "text":
                        value = before.get("text", "")
                        if type(value) is not str:
                            raise ReadIdentityUnavailable("logical transform source is unavailable")
                        start += len(value)
                part = content[index]
                if type(part) is not _FrozenDict or part.get("type") != "text":
                    raise ReadIdentityUnavailable("logical transform source is unavailable")
                selected = part.get("text", "")
            else:
                raise ReadIdentityUnavailable("logical transform source is unavailable")
            if type(selected) is not str or selected != text:
                raise ReadIdentityUnavailable("logical transform source text differs")
            if (
                not registry.known(message)
                and origin is not None
                and message._provenance is None
                and message._segment is None
                and not message._rewrite_content_changed
            ):
                native.located(message, content, origin)
            native._ensure(message)
            texts = native._parts(content)
            intervals = registry.window_ranges(message, texts, start, start + len(text))
            if not intervals and not text and session._token_maps is not None:
                intervals = (registry.value_interval(scope, "empty-input-text", ""),)
            local = registry.observe_parts(message, texts)
            keys.extend((local.source, local.start + start, local.start + start + len(text)))
        else:
            payload = dict.get(unit, kind)
            names = ("url", "detail") if kind == "image_url" else ("data", "format")
            if type(payload) is not dict or len(payload) > len(names):
                raise ReadIdentityUnavailable("logical transform media is unavailable")
            if any(type(key) is not str for key in payload) or not set(payload) <= set(names):
                raise ReadIdentityUnavailable("logical transform media is unavailable")
            if message is not None:
                native._ensure(message)
            for name in names:
                text = dict.get(payload, name)
                if text is None and name == "detail":
                    continue
                if type(text) is not str:
                    raise ReadIdentityUnavailable("logical transform media is unavailable")
                interval = (
                    registry.value_interval(scope, "transform-" + kind + "-" + name, text)
                    if message is None
                    else registry.field_interval(message, ("media-" + kind, str(index), name), text)
                )
                keys.append(interval.source)
            intervals = ()
        return InputMeasurement(
            UnitMeasurement("characters", "measured", intervals),
            registry.origin_identity(None, tuple(keys)),
        )
    except Exception as error:
        return InputMeasurement(UnitMeasurement("characters", native._failed(error)), None)


def collect_messages(session: ReadUsageSession, messages: object) -> InputMeasurement:
    """Collect only canonical message text after the operation's native readability guards.

    Roles participate in whole-input identity but are structural, rather than content
    characters. No score builder, boundary, tokenizer or wire serializer runs here.
    """
    native = session._text_inputs
    registry = native.registry
    try:
        scope = session.input_scope()
        if scope is None or type(messages) is not Messages or type(messages._root) is not Messages:
            return InputMeasurement(UnitMeasurement("characters", "unavailable"), None)
        registry.inspect(len(messages) + 1)
        if len(messages) > registry.max_ranges:
            raise ReadIdentityLimitError("logical input collection limit exceeded")
        intervals = []
        keys: list[bytes | int] = [registry.value_interval(scope, "input-shape", "messages").source]
        complete = True
        for position, message in enumerate(messages):
            if type(message) not in _MESSAGES:
                complete = False
                continue
            content = message.source_content
            # A plain located projection constructed before collection can be validated
            # from its retained native origin. Flagged transformed/cut values require
            # their actual lineage; their visible bytes alone are not a fresh source.
            if not registry.known(message):
                ordinal = messages._ordinals[position]
                origin = messages._root._origins[ordinal]
                if (
                    origin is not None
                    and message._provenance is None
                    and message._segment is None
                    and not message._rewrite_content_changed
                ):
                    native.located(message, content, origin)
            measured = native.measurement(message, content)
            if (
                measured.state == "measured"
                and not measured.intervals
                and session._token_maps is not None
                and native._parts(content)
            ):
                # A located empty string still names a text input. Preserve its zero-width
                # reference so a failed supplied token map cannot become measured zero.
                measured = UnitMeasurement(
                    "characters",
                    "measured",
                    (registry.value_interval(scope, "empty-input-text", ""),),
                )
            complete &= measured.state == "measured"
            if len(measured.intervals) > registry.max_ranges - len(intervals):
                raise ReadIdentityLimitError("logical input collection limit exceeded")
            intervals.extend(measured.intervals)
            if measured.state == "measured":
                # The local axis names the exact ordered variant, including an empty
                # authored turn. Original coverage may have several disjoint ranges.
                local = registry.observe_parts(message, native._parts(content))
                role = registry.value_interval(scope, "message-role", message.role)
                keys.extend((role.source, local.source, local.start, local.end))
            extra_ranges, extra_keys = _message_extras(registry, scope, message, content)
            if len(extra_ranges) > registry.max_ranges - len(intervals):
                raise ReadIdentityLimitError("logical input collection limit exceeded")
            intervals.extend(extra_ranges)
            keys.extend(extra_keys)
        cut = registry.origin_identity(None, tuple(keys)) if complete else None
        state = "measured" if complete else "partial" if intervals else "unavailable"
        return InputMeasurement(UnitMeasurement("characters", state, tuple(intervals)), cut)
    except Exception as error:
        return InputMeasurement(UnitMeasurement("characters", native._failed(error)), None)


def _message_extras(registry, scope, message, content):
    """Read guarded native fields, never decoding arguments or projecting a wire graph.

    Refusals, function names and opaque argument strings are text. Roles, call links
    and media payloads affect the cut identity without becoming content characters.
    """
    intervals = []
    keys: list[bytes | int] = [
        registry.value_interval(
            scope,
            "content-shape",
            "parts" if type(content) is tuple else "text" if type(content) is str else "none",
        ).source
    ]

    def field(owner, labels, text, *, counted=True):
        if type(text) is not str:
            raise ReadIdentityUnavailable("logical model input field is unavailable")
        if len(keys) >= registry.max_ranges:
            raise ReadIdentityLimitError("logical input collection limit exceeded")
        interval = registry.field_interval(owner, labels, text)
        if counted:
            intervals.append(interval)
        keys.append(interval.source)

    if type(message) is AssistantMessage:
        refusal, calls = message.refusal, message.tool_calls
        if refusal is not None:
            field(message, (None, None, "refusal"), refusal)
        if type(calls) is not tuple:
            raise ReadIdentityUnavailable("logical model input calls are unavailable")
        registry.inspect(len(calls))
        if len(calls) > registry.max_ranges:
            raise ReadIdentityLimitError("logical input collection limit exceeded")
        for call in calls:
            if type(call) is not ToolCall or type(call.function) is not ToolCallFunction:
                raise ReadIdentityUnavailable("logical model input call is unavailable")
            # These labels match native ToolCall Place coordinates. A legacy field read
            # and a full-message read therefore share their actual input text axis.
            field(call, (None, None, "name"), call.function.name)
            field(call, (None, None, "raw_arguments"), call.function.arguments)
            keys.append(registry.value_interval(scope, "call-id", call.id or "").source)
    elif type(message) is ToolMessage:
        keys.append(
            registry.value_interval(scope, "tool-call-id", message.tool_call_id or "").source
        )
    if type(content) is tuple:
        registry.inspect(len(content))
        for index, part in enumerate(content):
            if type(part) is not _FrozenDict:
                raise ReadIdentityUnavailable("logical model input part is unavailable")
            kind = part.get("type")
            if type(kind) is not str:
                raise ReadIdentityUnavailable("logical model input part is unavailable")
            if len(keys) + 2 > registry.max_ranges:
                raise ReadIdentityLimitError("logical input collection limit exceeded")
            keys.append(registry.value_interval(scope, "content-part-type", kind).source)
            if kind == "text":
                text = part.get("text", "")
                if type(text) is not str:
                    raise ReadIdentityUnavailable("logical model input part is unavailable")
                keys.append(len(text))
                continue
            names = (
                ("url", "detail")
                if kind == "image_url"
                else ("data", "format")
                if kind == "input_audio"
                else ()
            )
            payload = part.get(kind)
            if not names or type(payload) is not _FrozenDict:
                raise ReadIdentityUnavailable("logical model input media is unavailable")
            for name in names:
                text = payload.get(name)
                if text is not None:
                    field(message, ("media-" + kind, str(index), name), text, counted=False)
    return intervals, keys


def collect_tools(session: ReadUsageSession, tools: object) -> InputMeasurement:
    """Collect native offered-tool names/descriptions and semantic JSON schema text.

    Schema keys and scalar values count without JSON syntax/escaping. No defensive
    schema projection, serialization of a whole graph or arbitrary object hook runs.
    """
    native, registry = session._text_inputs, session._text_inputs.registry
    try:
        scope = session.input_scope()
        if scope is None or type(tools) not in (tuple, list):
            return InputMeasurement(UnitMeasurement("characters", "unavailable"), None)
        native_tools = cast(list[Tool] | tuple[Tool, ...], tools)
        registry.inspect(len(native_tools) + 1)
        intervals = []
        cuts: list[bytes | int] = [registry.value_interval(scope, "input-shape", "tools").source]

        def leaf(owner, labels, text):
            if len(intervals) >= registry.max_ranges:
                raise ReadIdentityLimitError("logical input collection limit exceeded")
            interval = registry.field_interval(owner, labels, text)
            intervals.append(interval)
            cuts.append(interval.source)

        for tool in native_tools:
            if (
                type(tool) is not Tool
                or tool.type != "function"
                or type(tool.function) is not ToolFunction
            ):
                raise ReadIdentityUnavailable("logical offered-tool input is unavailable")
            function = tool.function
            leaf(tool, ("tool-name",), function.name)
            leaf(tool, ("tool-description",), function.source_description)
            root = registry.origin_identity(None, ())
            pending = [(function.source_parameters, root, 0)]
            while pending:
                node, path, depth = pending.pop()
                registry.inspect(1)
                if depth > 100 or len(pending) > registry.max_ranges:
                    raise ReadIdentityLimitError("logical input graph limit exceeded")
                if type(node) is _FrozenDict:
                    items = object.__getattribute__(node, "_items")
                    if type(items) is not tuple:
                        raise ReadIdentityUnavailable("logical offered-tool schema is unavailable")
                    registry.inspect(len(items) * 2)
                    if len(items) + len(pending) > registry.max_ranges:
                        raise ReadIdentityLimitError("logical input graph limit exceeded")
                    children = []
                    for key, value in items:
                        if type(key) is not str:
                            raise ReadIdentityUnavailable(
                                "logical offered-tool schema is unavailable"
                            )
                        label = registry.value_interval(scope, "json-field", key).source
                        child_path = registry.origin_identity(None, (path, label))
                        leaf(tool, ("tool-schema-key", child_path.hex()), key)
                        children.append((value, child_path, depth + 1))
                    pending.extend(reversed(children))
                elif type(node) is tuple:
                    registry.inspect(len(node))
                    if len(node) + len(pending) > registry.max_ranges:
                        raise ReadIdentityLimitError("logical input graph limit exceeded")
                    pending.extend(
                        (value, registry.origin_identity(None, (path, index)), depth + 1)
                        for index, value in reversed(tuple(enumerate(node)))
                    )
                else:
                    if type(node) is str:
                        text = node
                    elif node is None:
                        text = "null"
                    elif type(node) is bool:
                        text = "true" if node else "false"
                    elif type(node) is int:
                        registry.inspect(node.bit_length() + 1)
                        text = json.dumps(node)
                    elif type(node) is float and math.isfinite(node):
                        registry.inspect(128)
                        text = json.dumps(node, allow_nan=False)
                    else:
                        raise ReadIdentityUnavailable("logical offered-tool scalar is unavailable")
                    leaf(tool, ("tool-schema-value", path.hex()), text)
        cut = registry.origin_identity(None, tuple(cuts))
        return InputMeasurement(UnitMeasurement("characters", "measured", tuple(intervals)), cut)
    except Exception as error:
        return InputMeasurement(UnitMeasurement("characters", native._failed(error)), None)
