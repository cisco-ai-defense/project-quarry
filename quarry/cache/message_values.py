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

"""Bounded transformed message values belonging to one rule evaluation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from quarry.cache.async_cache import _shallow_bytes
from quarry.cache.constants import RESULT_CACHE_MAX_BYTES
from quarry.engine.pacing import Pacer
from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.messages import Message, Messages, ToolCall, _FrozenDict
from quarry.types.transforms import ContentSourcePart, ContentSourceRange, ProvenanceSegment

MAX_MESSAGE_VALUES = 256
MAX_MESSAGE_VALUE_REFERENCES = 1_000_000


@dataclass(frozen=True, eq=False)
class BindingIdentity:
    """Keep a binding alive only with this rule, comparing it by object identity."""

    value: Any

    def __hash__(self) -> int:
        return id(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BindingIdentity) and self.value is other.value


@dataclass
class _Entry:
    lock: asyncio.Lock
    value: Messages | None = None
    reads: tuple[tuple[str, Any], ...] = ()
    scope: tuple = ()
    size: int = 0


class MessageValueMemo:
    """Keep each computed value stable within a rule without a process-wide result key.

    Values cannot be evicted and recomputed: a local adapter may be stateful. Exhaustion
    instead fails the rule explicitly. Failures and cancelled computations are not values.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple, _Entry] = {}
        self._bytes = 0

    async def evaluate(
        self,
        key: tuple,
        compute: Callable[[], Awaitable[Messages]],
        *,
        reads: list[tuple[str, Any]] | None = None,
        scope: tuple = (),
    ) -> Messages:
        raise_if_evaluation_expired()
        entry = self._entries.get(key)
        if entry is None:
            if len(self._entries) >= MAX_MESSAGE_VALUES:
                raise ValueError("transformed message values exceed their per-rule entry limit")
            entry = self._entries[key] = _Entry(asyncio.Lock(), scope=scope)
        async with entry.lock:
            raise_if_evaluation_expired()
            if entry.value is not None:
                if reads is not None:
                    reads.extend(entry.reads)
                return entry.value
            value = await compute()
            if not isinstance(value, Messages):
                raise TypeError("transformed message value must be Messages")
            # Other values can complete while this cooperative walk yields. Check the
            # current total again after measuring, before installing this value.
            kept_reads = tuple(reads or ())
            size = await _retained_bytes((key, value, kept_reads), RESULT_CACHE_MAX_BYTES)
            if self._bytes + size > RESULT_CACHE_MAX_BYTES:
                raise ValueError("transformed message values exceed their per-rule byte limit")
            self._bytes += size
            entry.size = size
            entry.reads = kept_reads
            entry.value = value
            return value

    def release_scope(self, scope: tuple) -> None:
        """Release item-local values once their complete enclosing map owns the output.

        A whole-map value or a value hoisted outside that item keeps its enclosing scope.
        Running computations are left with their owners rather than invalidated.
        """

        for key, entry in tuple(self._entries.items()):
            if entry.scope[: len(scope)] == scope and not entry.lock.locked():
                self._bytes -= entry.size
                del self._entries[key]


async def _retained_bytes(value: object, limit: int) -> int:
    """Size known immutable message state without projecting wire JSON or reading fields.

    Original graphs shared by multiple values are conservatively charged for each value.
    Pydantic's own storage descriptors bypass custom field guards and properties. Unknown
    host objects count their base size; their methods and serialization hooks never run.
    """

    pending: list[Any] = [value]
    seen: set[int] = set()
    references = 1
    total = 0
    pacer = Pacer()
    while pending:
        raise_if_evaluation_expired()
        if pacer.due():
            await asyncio.sleep(0)
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        total += _shallow_bytes(item)
        if total > limit:
            return limit + 1
        kind = type(item)
        if kind in {dict, MappingProxyType, list, tuple, set, frozenset}:
            edges = len(item) * (2 if kind in {dict, MappingProxyType} else 1)
            if references + edges > MAX_MESSAGE_VALUE_REFERENCES:
                return limit + 1
        if kind is dict:
            children = tuple(dict.keys(item)) + tuple(dict.values(item))
        elif kind is MappingProxyType:
            children = tuple(item.keys()) + tuple(item.values())
        elif kind in {list, tuple, set, frozenset}:
            children = item
        elif kind is _FrozenDict:
            children = (item._items, item._values)
        elif kind is Messages:
            children = tuple(object.__getattribute__(item, name) for name in Messages.__slots__)
        elif any(
            base is Message or base is ToolCall for base in type.__getattribute__(kind, "__mro__")
        ):
            children = tuple(
                BaseModel.__dict__[name].__get__(item, type(item))
                for name in ("__dict__", "__pydantic_extra__", "__pydantic_private__")
            )
        elif kind in {BindingIdentity, ContentSourcePart, ContentSourceRange, ProvenanceSegment}:
            children = tuple(object.__getattribute__(item, field.name) for field in fields(item))
        else:
            continue
        references += len(children)
        if references > MAX_MESSAGE_VALUE_REFERENCES:
            return limit + 1
        pending.extend(children)
    return total
