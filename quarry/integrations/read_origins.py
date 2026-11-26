# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Transient validation of native text origins into opaque character coordinates.

Evidence envelopes and interpolated coordinates are not exact read coverage. Exact
runs are followed separately; an inexact transform has its own full visible text axis.
Only the resulting scalar ranges survive this visitor.
"""

from __future__ import annotations

import json
import math
from array import array
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

from quarry.engine.matchers import field_parts
from quarry.input_registry import InputObject
from quarry.integrations.read_coverage import ReadInterval
from quarry.integrations.read_identity import (
    ReadIdentityLimitError,
    ReadIdentityUnavailable,
    ReadTextRegistry,
)
from quarry.types.messages import (
    AssistantMessage,
    DeveloperMessage,
    SystemMessage,
    ToolCall,
    ToolCallFunction,
    ToolMessage,
    UserMessage,
    _FrozenDict,
)
from quarry.types.origins import LINEAR, RUN_FIELDS, WHOLE, Derived, Place
from quarry.types.results import InputText

_MESSAGES = (SystemMessage, DeveloperMessage, UserMessage, AssistantMessage, ToolMessage)


@dataclass(slots=True)
class _Node:
    identity: bytes
    texts: tuple[str, ...]
    length: int
    offset: int = 0
    ranges: tuple[ReadInterval, ...] = ()
    variant: bool = False


class ReadOriginProjection:
    """Visit only validated native storage, with no provider, parsing or host hooks."""

    def __init__(
        self,
        registry: ReadTextRegistry,
        parts: Callable[[object], tuple[str, ...]],
        ensure: Callable[[object], None],
    ) -> None:
        self.registry, self.parts, self.ensure = registry, parts, ensure
        self._nodes: dict[int, _Node] = {}

    def _hash(self, node: _Node, start: int, end: int) -> bytes:
        if not 0 <= start <= end <= node.length:
            raise ValueError("logical input origin range is outside its source")
        return self.registry.fingerprint_window(node.texts, node.offset + start, node.offset + end)

    def _argument(self, owner: ToolCall, name: str) -> str:
        if name in {"raw_arguments", "name", "id"}:
            data = object.__getattribute__(owner, "__dict__")
            if name == "id":
                value = dict.get(data, "id")
                if type(value) is not str:
                    raise ReadIdentityUnavailable("logical input field is unavailable")
                return value
            function = dict.get(object.__getattribute__(owner, "__dict__"), "function")
            if type(function) is not ToolCallFunction:
                raise ReadIdentityUnavailable("logical input field is unavailable")
            value = dict.get(
                object.__getattribute__(function, "__dict__"),
                "arguments" if name == "raw_arguments" else "name",
            )
            if type(value) is not str:
                raise ReadIdentityUnavailable("logical input field is unavailable")
            return value
        if not name.startswith("arguments."):
            raise ReadIdentityUnavailable("logical input field is unavailable")
        # A native argument read has already populated this cache. Do not parse JSON,
        # call the arguments property, or rebuild a defensive full argument graph here.
        private = object.__getattribute__(owner, "__pydantic_private__")
        data = object.__getattribute__(owner, "__dict__")
        current = dict.get(data, "_arguments_cache", dict.get(private, "_arguments_cache"))
        error = dict.get(data, "_arguments_error", dict.get(private, "_arguments_error"))
        if type(current) is not _FrozenDict or error is not None:
            raise ReadIdentityUnavailable("logical input argument text is unavailable")
        for component in field_parts(name)[1:]:
            self.registry.inspect(1)
            if type(current) is _FrozenDict:
                try:
                    current = current[component]
                except KeyError:
                    raise ReadIdentityUnavailable("logical input field is unavailable") from None
            elif type(current) is tuple and component.isascii() and component.isdecimal():
                index = int(component)
                if index >= len(current):
                    raise ReadIdentityUnavailable("logical input field is unavailable")
                current = current[index]
            else:
                raise ReadIdentityUnavailable("logical input field is unavailable")
        if type(current) is str:
            return current
        if type(current) is bool:
            return "true" if current else "false"
        if type(current) is int:
            self.registry.inspect(current.bit_length() + 1)
        elif type(current) is float and math.isfinite(current):
            self.registry.inspect(128)
        else:
            raise ReadIdentityUnavailable("logical input field is unavailable")
        return json.dumps(current, allow_nan=False)

    def _input_field(self, owner: object, name: str) -> str:
        # InputRegistry has already validated and frozen this native storage. Read
        # only its exact immutable map, bypassing attribute/allowlist/host hooks.
        if type(owner) is not InputObject:
            raise ReadIdentityUnavailable("logical input field owner is unavailable")
        data = object.__getattribute__(owner, "_fields")
        if type(data) is not MappingProxyType:
            raise ReadIdentityUnavailable("logical input field storage is unavailable")
        value = data.get(name)
        if type(value) is str:
            return value
        raise ReadIdentityUnavailable("logical input field text is unavailable")

    def _place(self, origin: Place) -> _Node:
        if (
            type(origin.start) is not int
            or origin.start < 0
            or type(origin.field) is not str
            or type(origin.position) is not int
            or origin.position < 0
        ):
            raise ValueError("invalid logical input place")
        self.registry.inspect(len(origin.field) + 1)
        source = origin.source
        if source is not None:
            if (
                type(source) is not InputText
                or type(source.input) is not str
                or type(source.item) not in (str, type(None))
                or type(source.fields) is not tuple
            ):
                raise ValueError("invalid logical input place")
            self.registry.inspect(len(source.fields))
            if any(type(name) is not str for name in source.fields):
                raise ValueError("invalid logical input place")
            self.registry.inspect(sum(map(len, source.fields)))
            if origin.field not in source.fields:
                raise ValueError("invalid logical input place")
        owner = origin.owner
        if source is None and origin.field == "" and type(owner) in _MESSAGES:
            self.ensure(owner)
            texts = self.parts(owner.source_content)
            ranges = self.registry.observe_ranges(owner, texts)
        else:
            if source is None and type(owner) is ToolCall:
                text = self._argument(owner, origin.field)
            elif source is None and type(owner) is AssistantMessage and origin.field == "refusal":
                self.ensure(owner)
                text = owner.refusal
            elif source is not None:
                text = self._input_field(owner, origin.field)
            else:
                raise ReadIdentityUnavailable("logical input origin owner is unavailable")
            if type(text) is not str:
                raise ReadIdentityUnavailable("logical input field text is unavailable")
            labels = (
                None if source is None else source.input,
                None if source is None else source.item,
                origin.field,
            )
            texts, ranges = (text,), (self.registry.field_interval(owner, labels, text),)
        length = sum(map(len, texts))
        if origin.start > length:
            raise ValueError("logical input origin range is outside its source")
        values = (
            origin.start,
            *(value for item in ranges for value in (item.source, item.start, item.end)),
        )
        return _Node(
            self.registry.origin_identity(None, values),
            texts,
            length - origin.start,
            origin.start,
            ranges,
        )

    def _build(self, root: object) -> None:
        pending = [(root, 0, False)]
        visiting: set[int] = set()
        while pending:
            origin, depth, finish = pending.pop()
            self.registry.inspect(1)
            identity = id(origin)
            if identity in self._nodes:
                continue
            if depth > 100 or len(pending) + len(self._nodes) > self.registry.max_ranges:
                raise ReadIdentityLimitError("logical input origin graph limit exceeded")
            if type(origin) is Place:
                self._nodes[identity] = self._place(origin)
                continue
            if type(origin) is not Derived:
                raise ValueError("invalid logical input origin")
            if not finish:
                if identity in visiting:
                    raise ValueError("cyclic logical input origin")
                if (
                    type(origin.text) is not str
                    or type(origin.sources) is not tuple
                    or type(origin.runs) is not array
                    or origin.runs.typecode != "q"
                    or len(origin.runs) % RUN_FIELDS
                ):
                    raise ValueError("invalid logical input origin")
                self.registry.inspect(len(origin.sources) + len(origin.runs))
                if (
                    len(origin.sources) + len(pending) > self.registry.max_ranges
                    or len(origin.runs) // RUN_FIELDS > self.registry.max_ranges
                ):
                    raise ReadIdentityLimitError("logical input origin graph limit exceeded")
                visiting.add(identity)
                pending.append((origin, depth, True))
                pending.extend((child, depth + 1, False) for child in reversed(origin.sources))
                continue
            visiting.remove(identity)
            previous = 0
            variant = False
            node = _Node(b"", (origin.text,), len(origin.text))
            for index in range(0, len(origin.runs), RUN_FIELDS):
                low, high, source, begin, end, kind = origin.runs[index : index + RUN_FIELDS]
                if not (
                    previous <= low < high <= len(origin.text)
                    and 0 <= source < len(origin.sources)
                    and kind in (LINEAR, WHOLE)
                ):
                    raise ValueError("invalid logical input origin run")
                child = self._nodes[id(origin.sources[source])]
                if not 0 <= begin <= end <= child.length:
                    raise ValueError("logical input origin range is outside its source")
                exact = kind == LINEAR and high - low == end - begin
                variant |= not exact
                if exact and self._hash(node, low, high) != self._hash(child, begin, end):
                    raise ValueError("logical input origin text does not match its source")
                previous = high
            values = (
                len(origin.sources),
                *(self._nodes[id(child)].identity for child in origin.sources),
                len(origin.runs),
                *origin.runs,
            )
            node.identity = self.registry.origin_identity(origin.text, values)
            node.variant = variant
            self._nodes[identity] = node

    def ranges(self, content: object, origin: object) -> tuple[ReadInterval, ...]:
        texts = self.parts(content)
        length = sum(map(len, texts))
        self._build(origin)
        root = self._nodes[id(origin)]
        if self.registry.fingerprint_window(texts, 0, length) != self._hash(root, 0, length):
            raise ValueError("logical input origin text does not match its source")
        pending = [(origin, 0, length)]
        result = []
        while pending:
            current, start, end = pending.pop()
            self.registry.inspect(1)
            if type(current) is ReadInterval:
                result.append(current)
                continue
            if start == end:
                continue
            if len(result) + len(pending) > self.registry.max_ranges:
                raise ReadIdentityLimitError("logical input origin range limit exceeded")
            node = self._nodes[id(current)]
            if type(current) is Place:
                self.registry.inspect(len(node.ranges))
                result.extend(
                    self.registry._cut_projection(
                        node.ranges, node.offset + start, node.offset + end
                    )
                )
            elif type(current) is Derived and node.variant:
                result.append(ReadInterval(node.identity, "characters", start, end))
            elif type(current) is Derived:
                tasks = []
                position = start
                for index in range(0, len(current.runs), RUN_FIELDS):
                    low, high, source, begin, _, _ = current.runs[index : index + RUN_FIELDS]
                    low, high = max(start, low), min(end, high)
                    if low >= high:
                        continue
                    if low > position:
                        tasks.append(
                            (
                                ReadInterval(node.identity, "characters", position, low),
                                position,
                                low,
                            )
                        )
                    run_start = current.runs[index]
                    tasks.append(
                        (current.sources[source], begin + low - run_start, begin + high - run_start)
                    )
                    position = high
                if position < end:
                    tasks.append(
                        (ReadInterval(node.identity, "characters", position, end), position, end)
                    )
                # A gap is authored native input text, rather than an inferred original
                # range. Model-builder boundaries are never added to these origins.
                for child, low, high in reversed(tasks):
                    pending.append((child, low, high))
            else:
                raise ValueError("invalid logical input origin")
        return tuple(result)

    def identity(self, origin: object) -> bytes:
        return self._nodes[id(origin)].identity
