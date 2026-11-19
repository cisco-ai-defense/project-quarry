# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Cooperative JSON rendering over immutable native bodies and indexed source runs."""

from __future__ import annotations

import asyncio
import json
import re
from array import array
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from quarry.integrations.model_window_input import (
    MAX_WINDOW_RENDER_CHARACTERS,
    WindowInput,
    message_window_origin,
)
from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.exceptions import LLMError
from quarry.types.messages import (
    AssistantMessage,
    DeveloperMessage,
    OpenAIObject,
    SystemMessage,
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolFunction,
    ToolMessage,
    UserMessage,
    _FrozenDict,
)
from quarry.types.origins import LINEAR, WHOLE, Derived, SourceText

_NATIVE_KINDS = (
    AssistantMessage,
    DeveloperMessage,
    SystemMessage,
    UserMessage,
    ToolMessage,
    ToolCall,
    ToolCallFunction,
    Tool,
    ToolFunction,
)
_NATIVE_TO_DICT = OpenAIObject.to_openai_dict
_ESCAPES = re.compile(r'["\\\x00-\x1f]')
_STRING_STEP = 16_384
_MAX_NODES = 100_000
_MAX_PIECES = 1_000_000


@dataclass(frozen=True, slots=True)
class _Fields:
    items: tuple


def _native_fields(value):
    """Copy only shallow field references; retain frozen multipart/container bodies."""
    kind = type(value)
    if (
        not any(kind is native for native in _NATIVE_KINDS)
        or kind.to_openai_dict is not _NATIVE_TO_DICT
    ):
        return None
    data = BaseModel.__dict__["__dict__"].__get__(value, kind)
    supplied = BaseModel.__dict__["__pydantic_fields_set__"].__get__(value, kind)
    extra = BaseModel.__dict__["__pydantic_extra__"].__get__(value, kind)
    pairs = [
        (field.alias or name, data[name])
        for name, field in kind.model_fields.items()
        if name in supplied and not field.exclude
    ]
    if extra:
        pairs.extend(extra.items())
    return _Fields(tuple(pairs))


class _Writer:
    def __init__(self, *, ascii=False):
        self.ascii = ascii
        self.texts = []
        self.runs = array("q")
        self.length = 0
        self.nodes = 0
        self.pieces = 0

    def emit(self, text, mapping=None):
        if not text:
            return
        end = self.length + len(text)
        self.pieces += 1
        if end > MAX_WINDOW_RENDER_CHARACTERS or self.pieces > _MAX_PIECES:
            raise ValueError("model window JSON exceeds its render limit")
        if mapping is not None:
            source, low, high, kind = mapping
            if high > low:
                previous = self.runs[-6:] if self.runs else None
                if (
                    previous is not None
                    and previous[1] == self.length
                    and previous[2] == source
                    and previous[5] == kind
                    and (
                        kind == WHOLE
                        and (previous[3], previous[4]) == (low, high)
                        or kind == LINEAR
                        and previous[4] == low
                        and previous[1] - previous[0] == previous[4] - previous[3]
                        and len(text) == high - low
                    )
                ):
                    self.runs[-5] = end
                    self.runs[-2] = high
                else:
                    self.runs.extend((self.length, end, source, low, high, kind))
        self.texts.append(text)
        self.length = end

    async def checkpoint(self):
        raise_if_evaluation_expired()
        await asyncio.sleep(0)
        raise_if_evaluation_expired()

    async def opaque_inline_check(self, encoded):
        """Inspect host JSON without re-running its hooks or projecting a second tree."""
        if len(encoded) > MAX_WINDOW_RENDER_CHARACTERS:
            raise ValueError("model window JSON exceeds its render limit")
        decoder = json.JSONDecoder()
        index = previous_check = strings = 0
        while index < len(encoded):
            if encoded[index] != '"':
                index += 1
            else:
                value, index = decoder.raw_decode(encoded, index)
                strings += 1
                if strings > _MAX_NODES:
                    raise ValueError("model window JSON exceeds its graph limit")
                after = index
                while after < len(encoded) and encoded[after].isspace():
                    after += 1
                if value == "url" and after < len(encoded) and encoded[after] == ":":
                    after += 1
                    while after < len(encoded) and encoded[after].isspace():
                        after += 1
                    value, index = decoder.raw_decode(encoded, after)
                    if type(value) is not str or not value.startswith("data:"):
                        raise LLMError(
                            "Inference conversations require inline media",
                            {"code": "invalid_output"},
                        )
            if index - previous_check >= _STRING_STEP:
                await self.checkpoint()
                previous_check = index

    async def string(self, value, mapping, exact=None):
        self.emit('"', mapping)
        for start in range(0, len(value), _STRING_STEP):
            chunk = value[start : start + _STRING_STEP]
            if exact is None or self.ascii:
                self.emit(json.dumps(chunk, ensure_ascii=self.ascii)[1:-1], mapping)
            else:
                index, base = exact
                previous = 0
                for match in _ESCAPES.finditer(chunk):
                    at = match.start()
                    self.emit(
                        chunk[previous:at],
                        (index, base + start + previous, base + start + at, LINEAR),
                    )
                    self.emit(
                        json.dumps(chunk[at], ensure_ascii=False)[1:-1],
                        (index, base + start + at, base + start + at + 1, WHOLE),
                    )
                    previous = at + 1
                self.emit(
                    chunk[previous:],
                    (index, base + start + previous, base + start + len(chunk), LINEAR),
                )
            await self.checkpoint()
        self.emit('"', mapping)

    async def value(self, value, mapping=None, exact_paths=None, *, inline_media=False):
        exact_paths = {} if exact_paths is None else exact_paths
        stack: list[tuple[str, Any, tuple[str | int, ...], int]] = [("value", value, (), 0)]
        while stack:
            tag, value, path, depth = stack.pop()
            self.nodes += 1
            if self.nodes + len(stack) > _MAX_NODES or depth > 100:
                raise ValueError("model window JSON exceeds its graph limit")
            if self.nodes % 256 == 0:
                await self.checkpoint()
            if tag == "emit":
                self.emit(value, mapping)
                continue
            if tag == "key":
                if type(value) is not str:
                    raise ValueError("model window JSON keys must be strings")
                await self.string(value, mapping)
                continue
            kind = type(value)
            if kind is str:
                await self.string(value, mapping, exact_paths.get(path))
                continue
            if value is None or any(kind is native for native in (bool, int, float)):
                self.emit(json.dumps(value, allow_nan=False), mapping)
                continue
            if any(base is OpenAIObject for base in type.__getattribute__(kind, "__mro__")):
                native = _native_fields(value)
                if native is None:
                    # Registered serialization overrides are opaque: preserve their JSON,
                    # while claiming only the enclosing original message coordinates.
                    value = value.to_openai_dict()
                    kind = type(value)
                else:
                    value, kind = native, _Fields
            if kind is _Fields:
                pairs = value.items
            elif kind is _FrozenDict:
                pairs = object.__getattribute__(value, "_items")
            elif kind is dict:
                pairs = tuple(dict.items(value))
            else:
                pairs = None
            if pairs is not None:
                if self.nodes + len(stack) + 4 * len(pairs) > _MAX_NODES:
                    raise ValueError("model window JSON exceeds its graph limit")
                self.emit("{", mapping)
                stack.append(("emit", "}", path, depth))
                for index in range(len(pairs) - 1, -1, -1):
                    key, item = pairs[index]
                    if type(key) is not str:
                        raise ValueError("model window JSON keys must be strings")
                    if (
                        inline_media
                        and key == "url"
                        and (type(item) is not str or not item.startswith("data:"))
                    ):
                        raise LLMError(
                            "Inference conversations require inline media",
                            {"code": "invalid_output"},
                        )
                    if index + 1 < len(pairs):
                        stack.append(("emit", ",", path, depth))
                    stack.extend(
                        (
                            ("value", item, (*path, key), depth + 1),
                            ("emit", ":", path, depth),
                            ("key", key, path, depth),
                        )
                    )
            elif kind is tuple or kind is list:
                if self.nodes + len(stack) + 2 * len(value) > _MAX_NODES:
                    raise ValueError("model window JSON exceeds its graph limit")
                self.emit("[", mapping)
                stack.append(("emit", "]", path, depth))
                for index in range(len(value) - 1, -1, -1):
                    if index + 1 < len(value):
                        stack.append(("emit", ",", path, depth))
                    stack.append(("value", value[index], (*path, index), depth + 1))
            elif any(
                base is native
                for base in type.__getattribute__(kind, "__mro__")
                for native in (str, int, float, list, dict, tuple)
            ):
                encoded = json.dumps(
                    value, ensure_ascii=self.ascii, allow_nan=False, separators=(",", ":")
                )
                if inline_media:
                    await self.opaque_inline_check(encoded)
                self.emit(encoded, mapping)
            else:
                raise TypeError("model window JSON contains a non-JSON value")
        await self.checkpoint()

    def text(self):
        return "".join(self.texts)


async def json_context(tools):
    if not tools:
        return ""
    writer = _Writer(ascii=True)
    await writer.value(tools)
    return writer.text()


async def render_json_input(messages, tools):
    from quarry.integrations.model_window_ranges import WindowRangeBudget

    writer = _Writer()
    budget = WindowRangeBudget()
    sources = []
    last_turn = None
    if messages:
        writer.emit("[")
    for position, message in enumerate(messages):
        if position:
            writer.emit(",")
        sources.append(await message_window_origin(messages, position, budget))
        mapping = (position, 0, len(message.content), WHOLE)
        fields = _native_fields(message)
        exact_paths = {}
        if fields is not None:
            content = message.source_content
            if type(content) is str:
                exact_paths[("content",)] = (position, 0)
            elif type(content) is tuple:
                offset = 0
                for index, part in enumerate(content):
                    if (
                        isinstance(part, (dict, _FrozenDict))
                        and (type(part) is dict or type(part) is _FrozenDict)
                        and part.get("type") == "text"
                    ):
                        text = part.get("text", "")
                        if type(text) is str:
                            exact_paths[("content", index, "text")] = (position, offset)
                        offset += len(str(text))
        start = writer.length
        await writer.value(
            fields if fields is not None else message, mapping, exact_paths, inline_media=True
        )
        last_turn = (start, writer.length)
    if messages:
        writer.emit("]")
    context = await json_context(tools)
    text = writer.text()
    if len(text) + len(context) > MAX_WINDOW_RENDER_CHARACTERS:
        raise ValueError("model window JSON input/context exceeds its character limit")
    origin = Derived(text, tuple(sources), writer.runs)
    return WindowInput(
        SourceText(text, origin, messages._ordinals[0] if messages else 0), context, last_turn
    )
