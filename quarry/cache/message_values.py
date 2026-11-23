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
import os
from collections.abc import Awaitable, Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, fields
from itertools import count
from struct import calcsize
from types import MappingProxyType
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from quarry.cache.async_cache import _SAFE_SHALLOW_TYPES, _shallow_bytes
from quarry.cache.constants import RESULT_CACHE_MAX_BYTES
from quarry.engine.contracts import AtomEvaluation, ExecutionAttempt, ScanRecord, WitnessTrace
from quarry.engine.events import ConversationEvent, EvaluationInput, StaticEnvironment
from quarry.engine.local_work import ensure_local_work_authoritative
from quarry.engine.pacing import Pacer
from quarry.engine.witness import ModelResultEvidence
from quarry.input_registry import InputObject
from quarry.integrations.read_usage import ReadReplay, capture_read_usage, replay_read_usage
from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.message_groups import MessageGroups
from quarry.types.messages import (
    Message,
    Messages,
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolFunction,
    _FrozenDict,
)
from quarry.types.origins import Derived, Place, SourceText
from quarry.types.results import (
    CollectionResult,
    InputText,
    ProducerRef,
    ResultItem,
    Score,
    Span,
)
from quarry.types.transforms import ContentSourcePart, ContentSourceRange, ProvenanceSegment

MAX_MESSAGE_VALUES = 256
MAX_MESSAGE_VALUE_REFERENCES = 1_000_000
# Optional explanation admission is synchronous and small, so it cannot suspend a
# completed child verdict for an unbounded storage walk.
MAX_VALUE_EVIDENCE_REFERENCES = 4096


_Value = TypeVar("_Value", bound=Messages | MessageGroups)
_CALCULATION: ContextVar[list[MessageValueLease] | None] = ContextVar(
    "message_value_calculation", default=None
)


def _reset_calculation_after_fork() -> None:
    _CALCULATION.set(None)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_calculation_after_fork)


@dataclass(slots=True)
class _RetainedObject:
    value: object
    size: int
    owners: int = 1


class MessageValueLease:
    """Keep a known value graph charged until its last runtime owner releases it."""

    def __init__(self, budget: MessageValueBudget, identities: tuple[int, ...], size: int) -> None:
        self._budget = budget
        self._identities = identities
        self.size = size

    def close(self) -> None:
        identities, self._identities = self._identities, ()
        for identity in identities:
            node = self._budget._objects[identity]
            node.owners -= 1
            if not node.owners:
                self._budget.bytes -= node.size
                del self._budget._objects[identity]

    def copy_to(self, budget: MessageValueBudget) -> MessageValueLease:
        """Transfer already measured storage to a separate evidence budget without hooks."""
        if not self._identities:
            raise RuntimeError("a released message value cannot be retained again")
        limit = RESULT_CACHE_MAX_BYTES if budget.limit is None else budget.limit
        if self.size > limit:
            raise ValueError("transformed message values exceed their per-rule byte limit")
        graph = {
            identity: (self._budget._objects[identity].value, self._budget._objects[identity].size)
            for identity in self._identities
        }
        return budget._retain_graph(graph)


class MessageValueBudget:
    """One rule's live object graphs, shared by its separate child-frame caches.

    Shared objects are charged once. The table owns its objects while charged, preventing
    an identity from being recycled between a measurement and the last owner's release.
    """

    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit
        self.bytes = 0
        self.entries = 0
        self._objects: dict[int, _RetainedObject] = {}

    async def retain(self, value: object) -> MessageValueLease:
        limit = RESULT_CACHE_MAX_BYTES if self.limit is None else self.limit
        graph = await _retained_graph(value, limit)
        if graph is None:
            raise ValueError("transformed message values exceed their per-rule byte limit")
        return self._retain_graph(graph)

    def try_retain_snapshot(self, value: object) -> MessageValueLease | None:
        """Own bounded explanation data, or omit it without changing a verdict.

        This inspects only known storage, never message fields or serialization hooks.
        Unlike decision-value admission it does not await or check the verdict deadline.
        Its independent reference cap bounds the synchronous work, including shared edges.
        """
        limit = RESULT_CACHE_MAX_BYTES if self.limit is None else self.limit
        try:
            graph = {}
            for node in _retained_nodes(value, limit, MAX_VALUE_EVIDENCE_REFERENCES):
                if node is not None:
                    item, size = node
                    graph[id(item)] = (item, size)
            return self._retain_graph(graph)
        except (_GraphLimitExceeded, ValueError):
            return None

    def _retain_graph(self, graph: dict[int, tuple[object, int]]) -> MessageValueLease:
        limit = RESULT_CACHE_MAX_BYTES if self.limit is None else self.limit
        added = sum(
            max(0, size - self._objects[identity].size) if identity in self._objects else size
            for identity, (_, size) in graph.items()
        )
        new_objects = sum(identity not in self._objects for identity in graph)
        if (
            self.bytes + added > limit
            or len(self._objects) + new_objects > MAX_MESSAGE_VALUE_REFERENCES
        ):
            raise ValueError("transformed message values exceed their per-rule byte limit")
        # No await occurs between the aggregate check and installation. Concurrent frame
        # walks must recheck against all values that completed before their own walk.
        for identity, (value, size) in graph.items():
            if identity in self._objects:
                node = self._objects[identity]
                node.owners += 1
                node.size = max(node.size, size)
            else:
                self._objects[identity] = _RetainedObject(value, size)
        self.bytes += added
        return MessageValueLease(self, tuple(graph), sum(size for _, size in graph.values()))


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
    value: Messages | MessageGroups | None = None
    reads: tuple[tuple[str, Any], ...] = ()
    scope: tuple = ()
    size: int = 0
    lease: MessageValueLease | None = None
    logical_reads: ReadReplay | None = None


class MessageValueMemo:
    """Keep each computed value stable within a rule without a process-wide result key.

    Values cannot be evicted and recomputed: a local adapter may be stateful. Exhaustion
    instead fails the rule explicitly. Failures and cancelled computations are not values.
    """

    def __init__(self, budget: MessageValueBudget | None = None) -> None:
        self._entries: dict[tuple, _Entry] = {}
        self._bytes = 0
        self._scopes = count()
        self.budget = budget if budget is not None else MessageValueBudget()

    async def retain_part(self, value: Messages) -> None:
        """Keep an in-progress map's selected parts charged until its result is cached."""
        owner = _CALCULATION.get()
        if owner is None:
            raise RuntimeError("a map part needs its enclosing value computation")
        owner.append(await self.budget.retain(value))

    def new_scope(self) -> tuple:
        """Name one temporary source evaluation without retaining its event or result."""
        return ("computed-source", next(self._scopes))

    async def evaluate(
        self,
        key: tuple,
        compute: Callable[[], Awaitable[_Value]],
        *,
        reads: list[tuple[str, Any]] | None = None,
        scope: tuple = (),
    ) -> _Value:
        raise_if_evaluation_expired()
        ensure_local_work_authoritative()
        entry = self._entries.get(key)
        if entry is None:
            if self.budget.entries >= MAX_MESSAGE_VALUES:
                raise ValueError("transformed message values exceed their per-rule entry limit")
            entry = self._entries[key] = _Entry(asyncio.Lock(), scope=scope)
            self.budget.entries += 1
        async with entry.lock:
            raise_if_evaluation_expired()
            ensure_local_work_authoritative()
            if entry.value is not None:
                if reads is not None:
                    reads.extend(entry.reads)
                replay_read_usage(entry.logical_reads)
                return cast(_Value, entry.value)
            retained_parts: list[MessageValueLease] = []
            token = _CALCULATION.set(retained_parts)
            try:
                with capture_read_usage() as logical_reads:
                    value = await compute()
                    ensure_local_work_authoritative()
                    if not isinstance(value, Messages | MessageGroups):
                        raise TypeError("transformed message value must be Messages")
                    kept_reads = tuple(reads or ())
                    lease = await self.budget.retain((key, value, kept_reads))
                    try:
                        ensure_local_work_authoritative()
                    except BaseException:
                        lease.close()
                        raise
                    entry.lease = lease
                    entry.size = entry.lease.size
                    self._bytes += entry.size
                    entry.reads = kept_reads
                    # Accounting has its own quotas and weak request ownership. It is
                    # excluded from the decision value's retained graph and byte budget.
                    entry.logical_reads = None if logical_reads is None else logical_reads.result()
                    entry.value = value
                return value
            finally:
                _CALCULATION.reset(token)
                for lease in retained_parts:
                    lease.close()

    def release_scope(self, scope: tuple) -> None:
        """Release item-local values once their complete enclosing map owns the output.

        A whole-map value or a value hoisted outside that item keeps its enclosing scope.
        Running computations are left with their owners rather than invalidated.
        """

        for key, entry in tuple(self._entries.items()):
            if entry.scope[: len(scope)] == scope and not entry.lock.locked():
                self._bytes -= entry.size
                if entry.lease is not None:
                    entry.lease.close()
                del self._entries[key]
                self.budget.entries -= 1

    def close(self) -> None:
        """Release a finished frame; callers first stop all computations using it."""
        self.release_scope(())


async def _retained_bytes(value: object, limit: int) -> int:
    """Size known immutable message state without projecting wire JSON or reading fields.

    A graph counts each object once; the rule budget also deduplicates across owners.
    Pydantic's own storage descriptors bypass custom field guards and properties. Unknown
    host objects count their base size; their methods and serialization hooks never run.
    """

    graph = await _retained_graph(value, limit)
    return limit + 1 if graph is None else sum(size for _, size in graph.values())


async def _retained_graph(value: object, limit: int) -> dict[int, tuple[object, int]] | None:
    graph: dict[int, tuple[object, int]] = {}
    pacer = Pacer()
    try:
        for node in _retained_nodes(value, limit, MAX_MESSAGE_VALUE_REFERENCES):
            raise_if_evaluation_expired()
            if pacer.due():
                await asyncio.sleep(0)
            if node is not None:
                item, size = node
                graph[id(item)] = (item, size)
    except _GraphLimitExceeded:
        return None
    return graph


class _GraphLimitExceeded(Exception):
    """Known storage is too large or takes too many references to admit."""


def _retained_nodes(
    value: object, limit: int, reference_limit: int
) -> Iterator[tuple[object, int] | None]:
    """Inspect a closed set of storage types without running user-defined hooks."""
    pending: list[Any] = [value]
    seen: set[int] = set()
    references = 1
    total = 0
    while pending:
        item = pending.pop()
        if id(item) in seen:
            # Duplicate edges still consume traversal work. Let the asynchronous
            # caller pace them without counting shared storage a second time.
            yield None
            continue
        seen.add(id(item))
        kind = type(item)
        size = (
            _shallow_bytes(item)
            if any(kind is known for known in _SAFE_SHALLOW_TYPES)
            else object.__sizeof__(item)
        )
        if kind is InputObject:
            # Preserve the historical object header plus one immutable data slot.
            # Weak storage may be an explicit slot or a managed preheader depending
            # on Python; subtracting a pointer from __sizeof__ is not portable.
            # Optional registry owners/ranges have separate cumulative bounds.
            size = object.__basicsize__ + calcsize("P")
        total += size
        if total > limit:
            raise _GraphLimitExceeded
        yield item, size
        if any(kind is known for known in (dict, MappingProxyType, list, tuple, set, frozenset)):
            edges = len(item) * (2 if kind is dict or kind is MappingProxyType else 1)
            if references + edges > reference_limit:
                raise _GraphLimitExceeded
        if kind is dict:
            children = tuple(dict.keys(item)) + tuple(dict.values(item))
        elif kind is MappingProxyType:
            children = tuple(item.keys()) + tuple(item.values())
        elif any(kind is known for known in (list, tuple, set, frozenset)):
            children = item
        elif kind is _FrozenDict:
            children = (item._items, item._values)
        elif kind is InputObject:
            children = (object.__getattribute__(item, "_fields"),)
        elif kind is Messages:
            children = tuple(object.__getattribute__(item, name) for name in Messages.__slots__)
        elif any(
            any(
                base is known for known in (Message, ToolCall, ToolCallFunction, Tool, ToolFunction)
            )
            for base in type.__getattribute__(kind, "__mro__")
        ):
            children = tuple(
                BaseModel.__dict__[name].__get__(item, type(item))
                for name in ("__dict__", "__pydantic_extra__", "__pydantic_private__")
            )
        elif any(
            kind is known
            for known in (
                BindingIdentity,
                Place,
                Derived,
                SourceText,
                MessageGroups,
                ContentSourcePart,
                ContentSourceRange,
                ProvenanceSegment,
                InputText,
                Score,
                Span,
                CollectionResult,
                ProducerRef,
                ResultItem,
                AtomEvaluation,
                ExecutionAttempt,
                WitnessTrace,
                ScanRecord,
                ConversationEvent,
                StaticEnvironment,
                EvaluationInput,
                ModelResultEvidence,
            )
        ):
            children = tuple(object.__getattribute__(item, field.name) for field in fields(kind))
            # Non-slotted dataclasses also own the backing dictionary itself.
            try:
                storage = object.__getattribute__(item, "__dict__")
            except AttributeError:
                pass
            else:
                children = (*children, storage)
        else:
            continue
        references += len(children)
        if references > reference_limit:
            raise _GraphLimitExceeded
        pending.extend(children)
