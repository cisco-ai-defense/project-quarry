# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Scan-owned window renderings, with bounded known-graph storage and source origins."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import weakref
from array import array
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from quarry.cache.constants import RESULT_CACHE_MAX_BYTES
from quarry.cache.message_values import MAX_MESSAGE_VALUES, MessageValueBudget, MessageValueLease
from quarry.integrations.model_window_ranges import WindowRangeBudget
from quarry.integrations.model_windows import CharacterWindow, TokenWindow
from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.messages import Messages, _synthetic_key
from quarry.types.origins import LINEAR, WHOLE, Derived, Place, SourceText, message_origin
from quarry.types.transforms import _native_range_key

MAX_WINDOW_RENDER_CHARACTERS = 4_000_000
_WINDOW_MEMOS: weakref.WeakSet[WindowMemo] = weakref.WeakSet()


class _WindowLease(MessageValueLease):
    """A native window owner cannot release storage from a closed generation."""

    def __init__(self, budget: _WindowBudget, lease: MessageValueLease, generation: object):
        super().__init__(budget, lease._identities, lease.size)
        lease._identities = ()
        self._window_budget = budget
        self._generation = generation

    def close(self):
        budget = self._window_budget
        budget._check_fork()
        removed = []
        with budget._lock:
            identities, self._identities = self._identities, ()
            if self._generation is budget._generation:
                for identity in identities:
                    node = budget._objects[identity]
                    node.owners -= 1
                    if not node.owners:
                        budget.bytes -= node.size
                        removed.append(budget._objects.pop(identity))
        # A native graph may own host finalizers. Drop its last references after unlocking.
        removed.clear()

    def copy_to(self, budget: MessageValueBudget) -> MessageValueLease:
        owner = self._window_budget
        owner._check_fork()
        with owner._lock:
            if self._generation is not owner._generation or not self._identities:
                raise RuntimeError("model window memo has been released")
            graph = {
                identity: (owner._objects[identity].value, owner._objects[identity].size)
                for identity in self._identities
            }
        limit = RESULT_CACHE_MAX_BYTES if budget.limit is None else budget.limit
        if self.size > limit:
            raise ValueError("transformed message values exceed their per-rule byte limit")
        return budget._retain_graph(graph)


class _WindowBudget(MessageValueBudget):
    """Close even admitted leases that have not reached their memo entry yet."""

    def __init__(self, limit):
        super().__init__(limit)
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._generation = object()
        self._closed = False

    def _check_fork(self):
        if self._pid != os.getpid():
            # Another native cache's child cleanup can re-enter before our fork callback.
            # Never acquire the vanished thread's lock or retain its unpublished graph.
            self._lock = threading.RLock()
            self._pid = os.getpid()
            self._closed = True
            self._generation = object()
            removed, self._objects = self._objects, {}
            self.bytes = self.entries = 0
            removed.clear()

    def _retain_graph(self, graph):
        self._check_fork()
        failure = None
        with self._lock:
            if not self._closed:
                generation = self._generation
                try:
                    lease = super()._retain_graph(graph)
                except Exception as error:
                    if not self._closed and generation is self._generation:
                        raise
                    failure = error
                else:
                    if not self._closed and generation is self._generation:
                        return _WindowLease(self, lease, generation)
        # A fork can interrupt this very thread inside the native installation above.
        # Roll back any continued writes after its child callback closed the generation.
        self.close()
        raise RuntimeError("model window memo has been released") from failure

    def close(self):
        self._check_fork()
        with self._lock:
            self._closed = True
            self._generation = object()
            removed, self._objects = self._objects, {}
            self.bytes = self.entries = 0
        removed.clear()


@dataclass(frozen=True, slots=True)
class WindowInput:
    source: SourceText
    context: str = ""
    last_turn: tuple[int, int] | None = None

    def __post_init__(self):
        if (
            type(self.source) is not SourceText
            or type(self.source.text) is not str
            or type(self.context) is not str
            or type(self.source.message_index) is not int
            or self.source.message_index < 0
        ):
            raise ValueError("window input requires SourceText and string context")
        if len(self.source.text) + len(self.context) > MAX_WINDOW_RENDER_CHARACTERS:
            raise ValueError("window input/context exceeds its character limit")
        turn = self.last_turn
        if turn is not None:
            if (
                type(turn) not in (tuple, list)
                or len(turn) != 2
                or any(type(v) is not int for v in turn)
                or not 0 <= turn[0] <= turn[1] <= len(self.source.text)
            ):
                raise ValueError("window last_turn must be an interval in its full input")
            object.__setattr__(self, "last_turn", tuple(turn))


class _WindowEntryLock:
    """Serialize native memo work without binding a waiter to the owner's loop."""

    def __init__(self):
        self._pid = os.getpid()
        self._guard = threading.Lock()
        self._held = False
        self._closed = False
        self._waiters: dict[asyncio.Event, asyncio.AbstractEventLoop] = {}

    def _check_fork(self):
        if self._pid != os.getpid():
            self._guard = threading.Lock()
            self._pid = os.getpid()
            self._held = False
            self._closed = True
            self._waiters = {}

    async def __aenter__(self):
        while True:
            self._check_fork()
            with self._guard:
                if self._closed:
                    raise RuntimeError("model window memo has been released")
                if not self._held:
                    self._held = True
                    return self
                wake = asyncio.Event()
                self._waiters[wake] = asyncio.get_running_loop()
            try:
                await wake.wait()
            finally:
                with self._guard:
                    self._waiters.pop(wake, None)

    async def __aexit__(self, *_error):
        self._check_fork()
        with self._guard:
            self._held = False
            waiters = tuple(self._waiters.items())
        self._wake(waiters)

    @staticmethod
    def _wake(waiters):
        # Never touch another loop's Future or call loop hooks under the native guard.
        for wake, loop in waiters:
            try:
                loop.call_soon_threadsafe(wake.set)
            except RuntimeError:
                if not loop.is_closed():
                    raise

    def close(self):
        self._check_fork()
        with self._guard:
            self._closed = True
            waiters, self._waiters = self._waiters, {}
        self._wake(tuple(waiters.items()))


@dataclass
class _RenderEntry:
    lock: _WindowEntryLock
    value: WindowInput | None = None
    lease: MessageValueLease | None = None


@dataclass
class _PlanEntry:
    lock: _WindowEntryLock
    value: tuple | None = None
    lease: MessageValueLease | None = None


class WindowMemo:
    """Share completed input renders only while the current scan owns their bytes."""

    def __init__(self, limit: int):
        self.budget = _WindowBudget(limit)
        self.entries: dict[bytes, _RenderEntry] = {}
        self.plans: dict[bytes, _PlanEntry] = {}
        self.measurements: dict[bytes, _PlanEntry] = {}
        self.closed = False
        _WINDOW_MEMOS.add(self)

    def _require_open(self):
        if self.closed or self.budget._closed:
            raise RuntimeError("model window memo has been released")

    def _storage_lock(self):
        self.budget._check_fork()
        return self.budget._lock

    def _entry(self, table, key, factory):
        with self._storage_lock():
            self._require_open()
            entry = table.get(key)
            if entry is None:
                if self._entry_count() >= MAX_MESSAGE_VALUES:
                    label = "render" if factory is _RenderEntry else "memo"
                    raise ValueError(f"model windows exceed their {label} entry limit")
                entry = factory(_WindowEntryLock())
                self._require_open()
                table[key] = entry
            return entry

    def _cached(self, entry):
        with self._storage_lock():
            self._require_open()
            return entry.value

    def _publish(self, entry, value, lease):
        with self._storage_lock():
            if not self.closed and not self.budget._closed:
                entry.value, entry.lease = value, lease
                return value
        lease.close()
        raise RuntimeError("model window memo has been released")

    async def render(
        self, key: bytes, compute: Callable[[], Awaitable[WindowInput]]
    ) -> WindowInput:
        raise_if_evaluation_expired()
        entry = self._entry(self.entries, key, _RenderEntry)
        async with entry.lock:
            cached = self._cached(entry)
            if cached is None:
                value = await compute()
                self._require_open()
                lease = await self.budget.retain(
                    (key, value.source, value.context, value.last_turn)
                )
                return self._publish(entry, value, lease)
            return cached

    async def plan(self, key: bytes, compute, encoding=None):
        """Own exact cuts and payloads, including every known native window field."""
        raise_if_evaluation_expired()
        entry = self._entry(self.plans, key, _PlanEntry)
        async with entry.lock:
            cached = self._cached(entry)
            if cached is None:
                value = await compute()
                self._require_open()
                windows, batches = value
                graph = []
                for ordinal, window in enumerate(windows):
                    # These classes use slots. Expose their fields explicitly because the
                    # value budget deliberately does not inspect arbitrary host objects.
                    if type(window) is CharacterWindow:
                        graph.append((window.source, window.start, window.end))
                    elif type(window) is TokenWindow:
                        graph.append((window.token_ids, window.positions))
                    else:
                        raise TypeError("model window plans require native cuts")
                    if ordinal % 256 == 0:
                        await asyncio.sleep(0)
                        raise_if_evaluation_expired()
                if encoding is not None:
                    graph.append(
                        (
                            encoding,
                            encoding.token_ids,
                            encoding.prefix_token,
                            encoding.suffix_token,
                            encoding.last_turn,
                            encoding.context_tokens,
                            encoding.character_offsets,
                        )
                    )
                lease = await self.budget.retain((key, value, windows, batches, tuple(graph)))
                return self._publish(entry, value, lease)
            return cached

    def _entry_count(self):
        return len(self.entries) + len(self.plans) + len(self.measurements)

    async def measured(self, key: bytes, compute):
        """Share admitted primitive score tuples without keeping the opaque host reply."""
        raise_if_evaluation_expired()
        entry = self._entry(self.measurements, key, _PlanEntry)
        async with entry.lock:
            cached = self._cached(entry)
            if cached is None:
                value = await compute()
                self._require_open()
                lease = await self.budget.retain((key, value))
                return self._publish(entry, value, lease)
            return cached

    def close(self, _owner=None):
        with self._storage_lock():
            self.closed = True
            entries, self.entries = self.entries, {}
            plans, self.plans = self.plans, {}
            measurements, self.measurements = self.measurements, {}
        for entry in (*entries.values(), *plans.values(), *measurements.values()):
            entry.lock.close()
            if entry.lease is not None:
                entry.lease.close()
            entry.value = entry.lease = None
        self.budget.close()
        _WINDOW_MEMOS.discard(self)


def _reset_window_memos_after_fork() -> None:
    # Process caches and scan locks reset before this module's child callback runs.
    # An inherited deferred owner cannot release its parent's process-local lease,
    # so drop native window graphs even if a completed task still keeps them alive.
    for memo in tuple(_WINDOW_MEMOS):
        memo.close()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_window_memos_after_fork)


async def message_window_origin(messages: Messages, position: int, budget=None, _depth=0):
    """Use exact declared provenance, or an enclosing original for opaque rewrites."""
    ordinal = messages._ordinals[position]
    message = messages[position]
    original = messages._root._items[ordinal]
    origin = messages._root._origins[ordinal]
    if _depth > 100:
        raise ValueError("model window source graph exceeds its depth limit")
    if budget is None:
        budget = WindowRangeBudget()
    budget.charge()
    if origin is None and (
        _synthetic_key(message) is not None
        or any(segment.parts for segment in (message._provenance or ()))
    ):
        segments = message._content_provenance()
        source_root = messages._root._source_root
        # Authored or media-derived characters keep their own visible axis, with their
        # native unmapped/part metadata attached after decoding instead of invented text
        # coordinates in the scanned conversation.
        if not segments or any(not segment.sources or segment.parts for segment in segments):
            return Place(messages._root._positions[ordinal], "", 0, message)
        if source_root is None or source_root is messages._root:
            raise ValueError("copied model input has no retained original source")
        sources, runs, known = [], array("q"), {}
        for index, segment in enumerate(segments):
            budget.charge(1 + len(segment.sources))
            references = []
            for source in segment.sources:
                if _native_range_key(source) is None or not 0 <= source.message_index < len(
                    source_root._items
                ):
                    raise ValueError("copied model input has invalid source coordinates")
                owner = source_root._items[source.message_index]
                if not 0 <= source.message_start < source.message_end <= len(owner.content):
                    raise ValueError("copied model input is outside its original source")
                if source.message_index not in known:
                    known[source.message_index] = await message_window_origin(
                        source_root, source.message_index, budget, _depth + 1
                    )
                references.append(
                    (known[source.message_index], source.message_start, source.message_end)
                )
            if len(references) == 1:
                mapped, low, high = references[0]
                kind = LINEAR if segment.kind == "linear" else WHOLE
            else:
                # A bounded virtual axis represents every touched original interval;
                # whole mappings preserve disjoint ranges instead of filling their gaps.
                inner = array("q")
                for slot, (_, low, high) in enumerate(references):
                    inner.extend((slot, slot + 1, slot, low, high, WHOLE))
                mapped = Derived(
                    " " * len(references), tuple(item[0] for item in references), inner
                )
                low, high, kind = 0, len(references), WHOLE
            runs.extend((segment.start, segment.end, len(sources), low, high, kind))
            sources.append(mapped)
            if index % 256 == 0:
                await budget.checkpoint()
        return Derived(message.content, tuple(sources), runs)
    if origin is None:
        origin = Place(messages._root._positions[ordinal], "", 0, original)
    if message._rewrite_content_changed:
        runs = array("q")
        if message.content and original.content:
            runs.extend((0, len(message.content), 0, 0, len(original.content), WHOLE))
        return Derived(message.content, (origin,), runs)
    return message_origin(message, origin)


async def join_window_input(messages: Messages, tools: list[dict]) -> WindowInput:
    """Join selected contents while retaining their canonical full source projection."""
    texts, sources, runs = [], [], array("q")
    offset = 0
    content_characters = 0
    last_turn = None
    budget = WindowRangeBudget()
    for position, message in enumerate(messages):
        raise_if_evaluation_expired()
        if position:
            texts.append("\n")
            offset += 1
        text = message.content
        content_characters += len(text)
        if offset + len(text) > MAX_WINDOW_RENDER_CHARACTERS:
            raise ValueError("model window input exceeds its character limit")
        origin = await message_window_origin(messages, position, budget)
        if text:
            runs.extend((offset, offset + len(text), len(sources), 0, len(text), LINEAR))
        sources.append(origin)
        texts.append(text)
        last_turn = (offset, offset + len(text))
        offset += len(text)
        if position % 256 == 0:
            await asyncio.sleep(0)
            raise_if_evaluation_expired()
    text = "".join(texts) if content_characters else ""
    if not content_characters:
        last_turn = (0, 0)
    context = json.dumps(tools, ensure_ascii=True, separators=(",", ":")) if tools else ""
    if len(context) + len(text) > MAX_WINDOW_RENDER_CHARACTERS:
        raise ValueError("model window input/context exceeds its character limit")
    origin = Derived(text, tuple(sources), runs)
    return WindowInput(
        SourceText(text, origin, messages._ordinals[0] if messages else 0), context, last_turn
    )


async def json_window_input(messages: Messages, tools: list[dict]) -> WindowInput:
    from quarry.integrations.model_window_json import render_json_input

    return await render_json_input(messages, tools)
