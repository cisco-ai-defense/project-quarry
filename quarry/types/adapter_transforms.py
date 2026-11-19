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

"""Host adapter configuration and immutable content-transform replies."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

from quarry.types.exceptions import ModelError
from quarry.types.transforms import ContentTransform

TRANSFORM_PART_TYPES = frozenset({"text", "image_url", "input_audio"})
MAX_TRANSFORM_SETTINGS_DEPTH = 16
MAX_TRANSFORM_SETTINGS_VALUES = 10_000
MAX_TRANSFORM_SETTINGS_BYTES = 140_000
_NAME = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")


def _name(value: str, owner: str) -> None:
    if type(value) is not str or _NAME.fullmatch(value) is None:
        raise ValueError(f"{owner} must use 1 to 128 letters, digits, '_', '.' or '-'")


def checked_transform_parts(value: object) -> tuple[str, ...] | None:
    """Detach an explicit, non-empty subset of supported wire part types."""

    if value is None:
        return None
    if type(value) not in {list, tuple}:
        raise ValueError("transform parts must be a non-empty list of supported part types")
    parts = cast(list | tuple, value)
    if not 1 <= len(parts) <= len(TRANSFORM_PART_TYPES):
        raise ValueError("transform parts must be a non-empty list of supported part types")
    if any(type(part) is not str or part not in TRANSFORM_PART_TYPES for part in parts):
        raise ValueError("transform parts support only text, image_url and input_audio")
    if len(set(parts)) != len(parts):
        raise ValueError("transform parts must not repeat a part type")
    return tuple(parts)


def _settings(value: object) -> Mapping[str, object]:
    """Bound and detach JSON settings before recursion, encoding or long-lived storage."""

    if type(value) not in {dict, MappingProxyType}:
        raise TypeError("transform settings must be a JSON object")
    values = characters = 0
    active: set[int] = set()

    def visit(item: object, depth: int) -> object:
        nonlocal values, characters
        values += 1
        if values > MAX_TRANSFORM_SETTINGS_VALUES:
            raise ValueError("transform settings exceed the value limit")
        if depth > MAX_TRANSFORM_SETTINGS_DEPTH:
            raise ValueError("transform settings exceed the nesting limit")
        if type(item) is str:
            characters += len(item)
            if characters > MAX_TRANSFORM_SETTINGS_BYTES:
                raise ValueError("transform settings exceed the byte limit")
            return item
        if item is None or type(item) is bool:
            return item
        if type(item) is int:
            if not -(2**63) <= item < 2**63:
                raise ValueError("transform settings integers must fit signed 64-bit values")
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("transform settings numbers must be finite")
            return item
        if type(item) not in {dict, MappingProxyType, list, tuple}:
            raise TypeError("transform settings must contain only JSON values")
        identity = id(item)
        if identity in active:
            raise ValueError("transform settings must not contain cycles")
        active.add(identity)
        try:
            if type(item) in {dict, MappingProxyType}:
                result = {}
                for key, child in cast(Mapping, item).items():
                    if type(key) is not str:
                        raise TypeError("transform setting names must be strings")
                    visit(key, depth)
                    result[key] = visit(child, depth + 1)
                return result
            return [visit(child, depth + 1) for child in cast(list | tuple, item)]
        finally:
            active.remove(identity)

    plain = visit(value, 0)
    try:
        encoded = json.dumps(plain, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("transform settings must be valid Unicode JSON") from exc
    if len(encoded) > MAX_TRANSFORM_SETTINGS_BYTES:
        raise ValueError("transform settings exceed the byte limit")

    def freeze(item: object) -> object:
        if type(item) is dict:
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if type(item) is list:
            return tuple(freeze(child) for child in item)
        return item

    return cast(Mapping[str, object], freeze(plain))


@dataclass(frozen=True, slots=True)
class LocalTransformConfig:
    """A registered in-process adapter, its JSON settings and optional part selection."""

    adapter: str
    settings: Mapping[str, object] = field(default_factory=dict)
    parts: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _name(self.adapter, "transform adapter")
        object.__setattr__(self, "settings", _settings(self.settings))
        object.__setattr__(self, "parts", checked_transform_parts(self.parts))


@dataclass(frozen=True, slots=True)
class ModelTransformConfig:
    """A configured model alias whose adapter transforms the selected content parts."""

    model: str
    parts: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _name(self.model, "transform model alias")
        object.__setattr__(self, "parts", checked_transform_parts(self.parts))


TransformConfig = ContentTransform | LocalTransformConfig | ModelTransformConfig


def adapter_transform_reference(name: str, definition: object) -> str:
    """Bind a compiler reference to configuration without serializing adapter settings."""

    _name(name, "transform name")
    if type(definition) is LocalTransformConfig:
        alias = ""

        def plain(value):
            if isinstance(value, Mapping):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [plain(item) for item in value]
            return value

        identity = {
            "adapter": definition.adapter,
            "settings": plain(definition.settings),
            "parts": definition.parts,
        }
    elif type(definition) is ModelTransformConfig:
        alias = definition.model
        identity = {"model": alias, "parts": definition.parts}
    else:
        raise TypeError("adapter reference requires local or model transform configuration")
    wire = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return name + "\x1f" + alias + "\x1f" + hashlib.sha256(wire.encode()).hexdigest()


def adapter_transform_reference_parts(value: str, model: bool) -> tuple[str, str]:
    """Strictly validate a bounded reference and return its name and model alias."""

    if type(value) is not str or len(value) > 322 or value.count("\x1f") != 2:
        raise ValueError("invalid adapter transform reference")
    name, alias, digest = value.split("\x1f")
    _name(name, "transform name")
    if model:
        _name(alias, "transform model alias")
    elif alias:
        raise ValueError("local transform reference cannot name a model")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("invalid adapter transform configuration fingerprint")
    return name, alias


@dataclass(frozen=True, slots=True)
class TransformRun:
    """An output interval mapped to one input text interval, or to no character source.

    Linear runs copy equal-sized character ranges. Whole runs associate every output
    character with the complete input interval. Media-derived text has no input offsets.
    Coordinates are validated against the particular input and output before caching.
    """

    start: int
    end: int
    source_start: int | None = None
    source_end: int | None = None
    kind: Literal["linear", "whole"] = "whole"


@dataclass(frozen=True, slots=True)
class TransformResult:
    """Adapter output plus ordered runs that cover every output character exactly once."""

    text: str
    runs: tuple[TransformRun, ...]


class DecodeLimitError(ModelError):
    """A local decoder exhausted its explicit work, depth or output limit."""

    def __init__(self, message: str = "decoder limit reached") -> None:
        super().__init__(message, {"code": "decode_limit"})
