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

"""Canonical static/event input model.

The engine evaluates a rule against two semantically distinct inputs:

- a **static environment** — zero-width data that is not part of the temporal word:
  tool definitions, the system prompt, rule string sets, and scan limits; and
- an **event sequence** — the ordered timeline of conversation events (messages, tool
  calls, tool responses), each carrying its exact Quarry timeline position, the surfaces
  it belongs to, and a back-reference to the original Quarry object for evidence.

Unlike the pure core (`temporal`, `symbolic_dfa`, …), this module is an **adapter**: it
is allowed to import Quarry *data* types (`Message`/`ToolCall`/`ToolResponse`/`Tool`,
`ScanContext`) because the event list must be a faithful *view* over the
existing ``ScanContext.conversation`` — never a second parser. It must NOT import the
rule AST (`quarry.types.dsl`), the scanner (`quarry.scanner`), or the DSL layer
(`quarry.dsl.*`); the engine stays independent of how rules are parsed. (This boundary is
enforced by ``tests/engine/test_async_runtime.py``.)

Conversation events are built directly from ``ScanContext.timeline``. A scan may also
supply registered extension events and an explicit shared order. Surface membership
and per-surface ordinals are computed by object identity. Inputs that breach a safety
limit are rejected before the engine builds the event word.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Set
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar

from quarry.language.runtime import surface_names
from quarry.types.exceptions import InvalidInputError
from quarry.types.messages import Message, ToolCall, ToolMessage

if TYPE_CHECKING:  # avoid a hard import cycle; only needed for typing
    from quarry.types.context import ScanContext


class _EventSurfaceNames(Set[str]):
    """Backward-compatible surface view that does not read the catalog during import."""

    def _value(self) -> frozenset[str]:
        return surface_names(domain="event")

    def __contains__(self, value: object) -> bool:
        return value in self._value()

    def __iter__(self) -> Iterator[str]:
        return iter(self._value())

    def __len__(self) -> int:
        return len(self._value())


SURFACES = _EventSurfaceNames()


class EventKind(StrEnum):
    """What a timeline event is."""

    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    EXTENSION = "extension"


@dataclass(frozen=True)
class ConversationEvent:
    """One element of the temporal word from a conversation or registered extension.

    Fields:
        position: index in the evaluated timeline.
        kind: message / tool_call / tool_response / extension.
        surface_tags: the surfaces this event belongs to, by identity membership.
        ordinals: per-surface index of this event within each surface it belongs to
            (e.g. ``{"user_messages": 0}`` == ``user_messages[0]``).
        source: the original conversation object or validated extension fact.
        message_index: the source object's ``message_index`` (tool call/response) or the
            timeline position (plain messages), or -1 for extension events.
        source_path: a stable human-readable path to the event, e.g.
            ``"user_messages[0]"`` or ``"tool_calls[2]"`` (first surface + ordinal).
    """

    position: int
    kind: EventKind
    surface_tags: frozenset[str]
    ordinals: dict[str, int]
    source: Any
    message_index: int
    source_path: str

    @property
    def role(self) -> str | None:
        """Message role, if this event is a message; else ``None``."""
        return self.source.role if isinstance(self.source, Message) else None

    @property
    def tool_name(self) -> str | None:
        """Tool name, if this event is a tool call or a tool-result message; else ``None``.

        A tool result is the canonical ``ToolMessage`` (whose ``.name`` is the answered
        function's name), so a ``tool_responses`` event exposes the tool name here too."""
        if isinstance(self.source, (ToolCall, ToolMessage)):
            try:
                return self.source.name
            except InvalidInputError:
                return None
        return None


_Value = TypeVar("_Value")


class _DeferredMapping(Mapping[str, _Value]):
    """Resolve one immutable aggregate only when an atom reads it."""

    def __init__(self, load: Callable[[], Mapping[str, _Value]]):
        self._load = load
        self._resolved: Mapping[str, _Value] | None = None

    def _value(self) -> Mapping[str, _Value]:
        if self._resolved is None:
            self._resolved = self._load()
        return self._resolved

    def __getitem__(self, key: str) -> _Value:
        return self._value()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._value())

    def __len__(self) -> int:
        return len(self._value())


class _SurfaceSizes(Mapping[str, int]):
    """Memoize each surface independently so one faulty surface cannot poison another."""

    def __init__(self, load: Mapping[str, Callable[[], int]]):
        self._load = MappingProxyType(dict(load))
        self._resolved: dict[str, int] = {}

    def __getitem__(self, key: str) -> int:
        if key not in self._resolved:
            self._resolved[key] = self._load[key]()
        return self._resolved[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._load)

    def __len__(self) -> int:
        return len(self._load)


@dataclass(frozen=True)
class StaticEnvironment:
    """Zero-width data available to every event — never part of the temporal word.

    Tool definitions and the system prompt are static: a rule that quantifies over
    ``tools`` or reads ``system_prompt`` asks a question that does not advance the
    timeline. Keeping them here (not as events) is what lets static-tool quantifiers
    stay zero-width in later phases.
    """

    tools: Mapping[str, Any]  # name -> Tool, deferred until read
    system_prompt: str | None
    limits: Any = None  # SafetyLimits handle (opaque here)
    was_truncated: bool = False
    # Per-surface event counts (zero-width aggregate). ``len(messages)`` / ``count(S)``
    # collection-size comparisons are static properties of the whole word — not a temporal
    # count — so they are grounded here once at the boot step (see the compiler's
    # surface-count macro), which scales to any N with a single atom rather than an
    # N-state temporal counter.
    surface_sizes: Mapping[str, int] = field(default_factory=dict)
    # Per-name tool-call counts (``count_tool_calls("x")`` / ``len(get_tool_calls("x"))``):
    # a name-filtered collection-size aggregate, same zero-width treatment.
    tool_call_name_counts: Mapping[str, int] = field(default_factory=dict)
    # Per-name tool-RESPONSE counts (``len(get_tool_responses("x"))``): the response-side
    # twin of ``tool_call_name_counts``. A name-filtered ``get_tool_responses`` count must
    # read responses, not calls — an agent can call a tool without a paired
    # response (or vice-versa), so the two aggregates genuinely differ.
    tool_response_name_counts: Mapping[str, int] = field(default_factory=dict)
    # Diagnostic metadata, never used to answer a rule's size comparison. Explanations
    # can describe skipped tool atoms without semantically reading faulty definitions.
    tool_count: int | None = None
    _system_prompt_reader: Callable[[], str | None] | None = field(
        default=None, repr=False, compare=False
    )

    extension_state: dict[str, Any] = field(default_factory=dict)
    missing_extension_states: frozenset[str] = frozenset()

    def __getattribute__(self, name: str):
        if name == "system_prompt":
            reader = object.__getattribute__(self, "_system_prompt_reader")
            if reader is not None:
                value = reader()
                object.__setattr__(self, "system_prompt", value)
                object.__setattr__(self, "_system_prompt_reader", None)
        return object.__getattribute__(self, name)

    def has_tool(self, name: str) -> bool:
        """Return whether the static environment has a tool with the given name."""
        return name in self.tools

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Return the names of the tools in the static environment."""
        return tuple(self.tools.keys())


@dataclass(frozen=True)
class EvaluationInput:
    """The two inputs to a rule evaluation: static environment + event word."""

    static: StaticEnvironment
    events: tuple[ConversationEvent, ...]

    def on_surface(self, surface: str) -> tuple[ConversationEvent, ...]:
        """Events belonging to ``surface``, in timeline order."""
        return tuple(e for e in self.events if surface in e.surface_tags)

    def in_domain(self, spec) -> tuple[ConversationEvent, ...]:
        """The events an event atom can hold at: those on its surface, inside its index or
        slice window."""
        return tuple(e for e in self.on_surface(spec.selector) if in_window(spec, e, self.static))

    @property
    def length(self) -> int:
        """Return the length of the events."""
        return len(self.events)


def in_window(spec, event: ConversationEvent, static: StaticEnvironment | None) -> bool:
    """Whether ``event`` is inside ``spec``'s index or slice window on its surface; always
    true for an atom without one. A negative index and a slice window are resolved against
    the surface size, giving exact Python indexing and slicing semantics, so
    ``messages[-1]`` is the last message and ``messages[-4:]`` the last four."""
    if not spec.is_slice and spec.ordinal is None:
        return True
    ordinal = event.ordinals.get(spec.selector)
    if ordinal is None:
        return False
    size = static.surface_sizes.get(spec.selector, 0) if static else 0
    if spec.is_slice:
        lo, hi, _ = slice(spec.slice_start, spec.slice_stop).indices(size)
        return lo <= ordinal < hi
    target = spec.ordinal if spec.ordinal >= 0 else size + spec.ordinal
    return ordinal == target


def _build_surface_index(conv) -> dict[str, dict[int, int]]:
    """Per-surface ``{id(obj): ordinal}`` maps, built ONCE per conversation.

    Membership is an O(1) ``id()`` lookup and the whole build is O(N). Keyed by ``id(obj)`` so
    membership is by object identity (not equality) — safe because the maps live only for this
    synchronous build.

    ``tool_responses`` maps the canonical ``conv.tool_responses`` (the ``ToolMessage`` view).
    Those same ``ToolMessage`` objects also live in ``conv.messages``, so a tool result's single
    event is tagged as belonging to BOTH surfaces (with the correct ordinal on each) — one
    canonical event, not a synthesized duplicate."""
    surface_lists = {surface: getattr(conv, surface) for surface in surface_names(domain="event")}
    return {
        surface: {id(item): idx for idx, item in enumerate(seq)}
        for surface, seq in surface_lists.items()
    }


def _surface_membership(
    obj: Any, surface_index: dict[str, dict[int, int]]
) -> tuple[frozenset[str], dict[str, int]]:
    """Surfaces ``obj`` belongs to and its ordinal within each, by identity.

    Membership mirrors exactly what the ``Conversation`` collections contain: a plain
    message may be in ``messages`` and one of ``user_messages``/``assistant_messages``;
    a tool call is in ``tool_calls``; a tool response in ``tool_responses``. Uses the
    prebuilt ``{id(obj): ordinal}`` maps (O(1) per surface) — identical result to the
    prior linear identity scan.
    """
    tags: set[str] = set()
    ordinals: dict[str, int] = {}
    oid = id(obj)
    for surface, members in surface_index.items():
        idx = members.get(oid)
        if idx is not None:
            tags.add(surface)
            ordinals[surface] = idx
    return frozenset(tags), ordinals


def _kind_of(obj: Any) -> EventKind:
    if isinstance(obj, ToolCall):
        return EventKind.TOOL_CALL
    # A tool result is a canonical ``ToolMessage`` (a ``Message``) → MESSAGE kind, tagged onto
    # both the ``messages`` and ``tool_responses`` surfaces. There is no distinct TOOL_RESPONSE
    # timeline object anymore; TOOL_RESPONSE remains a valid surface but not an event kind.
    if isinstance(obj, Message):
        return EventKind.MESSAGE
    raise TypeError(f"unexpected timeline object type: {type(obj).__name__}")


# Preferred surface for the human-readable source path, most specific first.
_PATH_SURFACE_PRIORITY = (
    "user_messages",
    "assistant_messages",
    "tool_calls",
    "tool_responses",
    "messages",
)


def _source_path(surface_tags: frozenset[str], ordinals: dict[str, int]) -> str:
    for surface in _PATH_SURFACE_PRIORITY:
        if surface in surface_tags:
            return f"{surface}[{ordinals[surface]}]"
    return "?"


def build_events(context: ScanContext) -> tuple[ConversationEvent, ...]:
    """Build the event word as a view over ``context.timeline``.

    ``position`` is the true ``ScanContext.timeline`` index of each emitted event.
    Surface tags/ordinals are computed by identity against the ``Conversation``
    collections.

    Every raw timeline entry becomes exactly one event (faithful 1:1 view — including
    the ``tool``-role ``Message`` that mirrors a ``ToolResponse``, so the ``messages``
    surface stays identical to ``Conversation.messages`` by object identity). Flow ``within:N``
    distance correctness for tool turns is handled at the DFA/compile layer, not by
    altering this view (see the compiler's flow lowering).
    """
    conv = context.conversation
    # Build the per-surface identity→ordinal maps ONCE, then O(1) membership per event
    # (was an O(N) linear scan per event → O(N²) build).
    surface_lists = {surface: getattr(conv, surface) for surface in surface_names(domain="event")}
    occurrence_indices: dict[str, dict[int, list[int]]] = {}
    for surface, items in surface_lists.items():
        by_identity: dict[int, list[int]] = {}
        for ordinal, item in enumerate(items):
            by_identity.setdefault(id(item), []).append(ordinal)
        occurrence_indices[surface] = by_identity
    consumed: dict[tuple[str, int], int] = {}
    events: list[ConversationEvent] = []
    for position, obj in context.timeline:
        kind = _kind_of(obj)
        ordinals = {}
        for surface, by_identity in occurrence_indices.items():
            positions = by_identity.get(id(obj), ())
            key = surface, id(obj)
            occurrence = consumed.get(key, 0)
            if occurrence < len(positions):
                ordinals[surface] = positions[occurrence]
                consumed[key] = occurrence + 1
        surface_tags = frozenset(ordinals)
        msg_index = getattr(obj, "message_index", position)
        events.append(
            ConversationEvent(
                position=position,
                kind=kind,
                surface_tags=surface_tags,
                ordinals=ordinals,
                source=obj,
                message_index=msg_index,
                source_path=_source_path(surface_tags, ordinals),
            )
        )
    return tuple(events)


def build_static_environment(context: ScanContext) -> StaticEnvironment:
    """Build the static environment from ``context`` (tool defs, system prompt, limits)."""
    conv = context.conversation
    # Per-surface sizes, taken from the Conversation collections so they match exactly
    # what a rule's ``len(messages)`` / ``count(user_messages)`` sees.
    size_readers: dict[str, Callable[[], int]] = {
        surface: lambda surface=surface: len(getattr(conv, surface))
        for surface in surface_names(domain="event")
    }
    # Deferring each key matters: len(messages) must not read offered tools.
    size_readers.update(
        tools=lambda: len(conv.tools), system_prompt=lambda: len(conv.system_prompt or "")
    )
    size_readers.update(
        {
            surface: lambda items=items: len(items)
            for surface, items in context.extension_event_surfaces.items()
        }
    )
    surface_sizes = _SurfaceSizes(size_readers)
    tool_call_name_counts = _DeferredMapping(conv.messages._name_counts)
    tool_response_name_counts = _DeferredMapping(lambda: conv.messages._name_counts(responses=True))
    return StaticEnvironment(
        tools=_DeferredMapping(conv._tool_index),
        system_prompt=None,
        _system_prompt_reader=lambda: conv.system_prompt,
        limits=context.limits,
        was_truncated=context.was_truncated,
        surface_sizes=surface_sizes,
        tool_call_name_counts=tool_call_name_counts,
        tool_response_name_counts=tool_response_name_counts,
        tool_count=len(conv._tools),
        extension_state=context.extension_state_values,
        missing_extension_states=context.missing_extension_states,
    )


def build_evaluation_input(context: ScanContext) -> EvaluationInput:
    """Assemble the conversation and registered facts into one event word."""
    static = build_static_environment(context)
    conversation_events = build_events(context)
    if not context.extension_event_sequence and context.ordered_timeline is None:
        return EvaluationInput(static=static, events=conversation_events)

    by_source = {id(event.source): event for event in conversation_events}
    extension_surface = {
        id(item): surface
        for surface, items in context.extension_event_surfaces.items()
        for item in items
    }
    word = context.ordered_timeline
    if word is None:
        word = [*context.timeline]
        start = len(word)
        word.extend(
            (start + index, item) for index, item in enumerate(context.extension_event_sequence)
        )
    events: list[ConversationEvent] = []
    extension_ordinals: dict[str, int] = {}
    for position, item in word:
        conversation_event = by_source.get(id(item))
        if conversation_event is not None:
            event = replace(conversation_event, position=position)
        else:
            surface = extension_surface[id(item)]
            ordinal = extension_ordinals.get(surface, 0)
            extension_ordinals[surface] = ordinal + 1
            event = ConversationEvent(
                position=position,
                kind=EventKind.EXTENSION,
                surface_tags=frozenset((surface,)),
                ordinals={surface: ordinal},
                source=item,
                message_index=-1,
                source_path=f"{surface}[{ordinal}]",
            )
        events.append(event)
    return EvaluationInput(static=static, events=tuple(events))
