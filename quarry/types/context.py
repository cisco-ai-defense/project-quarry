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

"""Context for rule evaluation."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import orjson

from quarry.cache.message_values import MessageValueMemo
from quarry.core.conversation import Conversation
from quarry.safety.limits import SafetyLimits
from quarry.types.exceptions import ConversationValidationError
from quarry.types.messages import Message, Messages, ToolCall, ToolCallTypeRegistry

# Offending items named individually inside ONE limit's detail string before it collapses to a
# count. Bounds the error: a 10,000-message conversation that breaches a limit on every message
# must not build a 10,000-clause message. This caps items WITHIN a violation, not the number of
# violations reported — every breached limit is always named.
_MAX_REPORTED_ITEMS = 5
_EXACT_CONTAINER_ERROR = (
    "conversation contains an overridable dict/list subclass; "
    "use exact built-in dict and list containers"
)
_EXACT_KEY_ERROR = "conversation JSON object keys must use exact built-in strings"
_REFERENCE_BYTES = max(1, list.__sizeof__([None]) - list.__sizeof__([]))


def _summarize(details: list[str]) -> str:
    """Join per-item violation details, capping how many are named individually."""

    if len(details) <= _MAX_REPORTED_ITEMS:
        return "; ".join(details)
    shown = "; ".join(details[:_MAX_REPORTED_ITEMS])
    return f"{shown}; and {len(details) - _MAX_REPORTED_ITEMS} more"


def _limit_error(violations: dict[str, str]) -> ConversationValidationError:
    """Build the single error naming every safety limit the input breached.

    Keys are the CONFIG SETTING names (``max_total_size``, ``max_message_size``, ...) rather than
    prose, so the message tells an operator exactly which key to raise."""

    return ConversationValidationError(
        "conversation exceeds configured safety limits: "
        + "; ".join(f"{key}: {detail}" for key, detail in violations.items())
        + ". The conversation is not truncated or partially skipped to fit; raise the limit(s) "
        "above if this input is legitimately large."
    )


def _timeline_items(msg: dict[str, Any]) -> int:
    """Count the timeline events one raw message will materialize.

    One per message, plus one per preserved tool call, including cross-role calls. A tool
    result is one canonical
    ``ToolMessage`` occupying a single position (it belongs to both the ``messages`` and
    ``tool_responses`` surfaces there), so it is not counted twice."""

    return 1 + len(msg.get("tool_calls") or [])


def _exact_dict_children(value: dict) -> Iterator[Any]:
    """Yield an exact dict's keys and values without calling overridable methods."""

    for key, child in dict.items(value):
        if type(key) is not str:
            raise ConversationValidationError(_EXACT_KEY_ERROR)
        yield key
        yield child


def _base_object_size(value: Any) -> int:
    """Measure with built-in descriptors, never an untrusted ``__sizeof__`` override."""

    value_type = type(value)
    if value_type is dict:
        return dict.__sizeof__(value)
    if value_type is list:
        return list.__sizeof__(value)
    if issubclass(value_type, str):
        return str.__sizeof__(value)
    if issubclass(value_type, int):
        return int.__sizeof__(value)
    if issubclass(value_type, float):
        return float.__sizeof__(value)
    return object.__sizeof__(value)


def _bounded_deep_size(value: Any, limit: int) -> int:
    """Measure nested input without walking beyond the configured byte budget.

    Two properties callers must account for.

    The unit is the PYTHON OBJECT footprint (built-in ``__sizeof__`` descriptors, summed
    recursively), not serialized length, and the gap between them depends on SHAPE rather than
    size. A payload dominated by a few large strings — a conversation carrying inline media —
    measures within a percent of its JSON length, because a ``str``'s footprint is its bytes plus
    a small header. A payload of many small objects pays that header thousands of times over:
    10,000 short messages measure ~6x their JSON length, and a single tiny message ~12x.

    Shared immutable scalars are charged once, matching their live memory footprint. Container
    occurrences are charged every time they appear: validation and canonicalization materialize
    every occurrence, so identity-deduplicating a reused schema would hide their aggregate work
    and output memory.

    The result saturates: once the running total or reference-work budget passes ``limit`` this
    returns ``limit + 1`` and stops walking, so a return value above ``limit`` means "over budget"
    and is not a measurement of how far over. Iterator frames keep traversal memory proportional
    to nesting depth; the scalar identity set is bounded by the charged object footprint.
    """
    total = 0
    visits = 0
    max_visits = max(1, limit // _REFERENCE_BYTES + 1)
    seen_scalars: set[int] = set()
    active_containers: set[int] = set()
    frames: list[tuple[int | None, Any]] = [(None, iter((value,)))]
    while frames:
        try:
            current = next(frames[-1][1])
        except StopIteration:
            container_id, _ = frames.pop()
            if container_id is not None:
                active_containers.remove(container_id)
            continue

        visits += 1
        if visits > max_visits:
            return limit + 1

        current_id = id(current)
        current_type = type(current)
        is_container = current_type is dict or current_type is list
        if is_container:
            if current_id in active_containers:
                # Structural validation reports the cycle after this bounded preflight.
                continue
        elif issubclass(current_type, (dict, list)):
            raise ConversationValidationError(_EXACT_CONTAINER_ERROR)
        elif current_id in seen_scalars:
            continue
        else:
            seen_scalars.add(current_id)

        total += _base_object_size(current)
        if total > limit:
            return limit + 1

        if current_type is dict:
            active_containers.add(current_id)
            frames.append((current_id, _exact_dict_children(current)))
        elif current_type is list:
            active_containers.add(current_id)
            frames.append((current_id, list.__iter__(current)))
    return total


@dataclass(frozen=True)
class _TotalSizePreflight:
    """Identity-bound proof that one raw conversation passed its total-size budget."""

    messages: object
    tools: object
    limit: int

    def matches(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, limit: int
    ) -> bool:
        if type(messages) is not list or (tools is not None and type(tools) is not list):
            return False
        return (
            self.messages is messages
            and self.tools is (tools if tools else None)
            and self.limit == limit
        )


def preflight_conversation_total_size(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, limit: int
) -> _TotalSizePreflight:
    """Reject an oversized raw conversation before any full nested-data traversal."""

    if type(messages) is not list or (tools is not None and type(tools) is not list):
        raise ConversationValidationError(_EXACT_CONTAINER_ERROR)
    total_size = _bounded_deep_size({"messages": messages, "tools": tools or []}, limit)
    if total_size <= limit:
        return _TotalSizePreflight(messages, tools if tools else None, limit)

    # No measured size is reported. ``_bounded_deep_size`` stops walking the moment it
    # passes the budget and returns ``limit + 1``, so any number here would always be
    # ``limit + 1`` — a fake measurement that reads like a real one. Stating the budget
    # and the unit is the honest version.
    raise _limit_error(
        {
            "max_total_size": (
                f"input exceeds the {limit}-byte aggregate PYTHON OBJECT budget. This is not "
                "serialized length: repeated container references are charged per occurrence "
                "because validation materializes each occurrence, while shared immutable scalars "
                "are charged once. Large strings such as inline media measure near their JSON "
                "length, but per-object overhead makes many small messages measure roughly "
                "5-12x their JSON size. The exact size is not reported because measurement stops "
                "at the budget rather than walking an oversized payload in full"
            )
        }
    )


@dataclass
class ScanContext:
    """Context for rule evaluation."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        limits: SafetyLimits | None = None,
        resource_manager=None,
        function_cache=None,
        *,
        tool_call_types: ToolCallTypeRegistry | None = None,
        parameters=None,
        parameter_specs=None,
        _total_size_preflight: _TotalSizePreflight | None = None,
    ):
        self.resource_manager = resource_manager
        self.function_cache = function_cache
        self.tool_call_types = Messages.validate_tool_call_types(tool_call_types)
        self.parameters = parameters or {}
        self.parameter_specs = parameter_specs or {}
        # Scanner-local facts from trusted, registered extension schemas.
        self.extension_event_surfaces: dict[str, tuple[Any, ...]] = {}
        self.extension_event_sequence: tuple[Any, ...] = ()
        self.extension_state_values: dict[str, Any] = {}
        self.missing_extension_states: frozenset[str] = frozenset()
        self.input_faults: dict[str, str] = {}
        self.ordered_timeline: list[tuple[int, Any]] | None = None
        if limits is None and resource_manager and hasattr(resource_manager, "config"):
            self.limits = resource_manager.config.get_safety_limits()
        else:
            self.limits = limits or SafetyLimits()
        # Reporting channel for a scan that could not be evaluated in full. The
        # conversation-preparation layer below never sets these: it does not truncate, so it has
        # nothing to report here — an over-limit input raises instead. They stay for the
        # ``ScanResult`` plumbing that reads them (``scanner`` and ``cli``).
        self.was_truncated = False
        self.truncation_info = {}

        if self.limits.enforce_limits:
            # Validation only. ``messages`` and ``tools`` are handed to ``Conversation`` exactly
            # as the caller passed them — same objects, same bytes.
            self._enforce_limits(messages, tools, total_size_preflight=_total_size_preflight)

        self.conversation = Conversation(
            messages,
            tools,
            tool_call_types=self.tool_call_types,
            max_nesting_depth=self.limits.max_nesting_depth,
        )
        # A tool result is one canonical ``ToolMessage``. It occupies a
        # single timeline position that belongs to BOTH the ``messages`` and ``tool_responses``
        # surfaces (the surface index tags it on both) — there is no second synthesized event.
        self._timeline_cache: list[tuple[int, Message | ToolCall]] | None = None
        self.evaluation_warnings: list[str] = []
        # Map comprehension filters are memoized by item within one rule evaluation.
        self.map_filter_results: dict = {}
        self.message_value_memo = MessageValueMemo()

    def _enforce_limits(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        total_size_preflight: _TotalSizePreflight | None = None,
    ) -> None:
        """Reject any input that breaches a configured safety limit. Never reshape it.

        ``Conversation`` and ``Message`` are the canonical record of the original OpenAI
        conversation, so an over-limit input is REPORTED, not fitted: no message is skipped, no
        content is truncated, no tool call is dropped, and no object is rebuilt. Fitting the input
        would run rules against a conversation that differs from what the caller sent, and could
        report a clean result over it. Deciding how to fit oversized content to a
        particular model is ``ModelInferenceBackend``'s job at request-construction time, and it
        must not rebuild these source objects either.

        Breaches are collected and raised as ONE error, so an operator sees the whole list rather
        than fixing limits one scan at a time. ``max_total_size`` is the one exception and reports
        alone: it is what bounds the cost of measuring everything else, so a payload breaching it
        is rejected before the per-message and per-call walks run over it.
        """
        if type(messages) is not list:
            raise ConversationValidationError(
                f"messages must be a list, got {type(messages).__name__}"
            )
        if tools is not None and type(tools) is not list:
            raise ConversationValidationError(f"tools must be a list, got {type(tools).__name__}")

        # Scanner paths run this before full conversation validation so a wide tool schema
        # cannot consume work first. Direct ``ScanContext`` callers still get the same check
        # here; the private marker avoids measuring scanner input twice.
        if total_size_preflight is None or not total_size_preflight.matches(
            messages, tools, self.limits.max_total_size
        ):
            preflight_conversation_total_size(messages, tools, self.limits.max_total_size)

        violations: dict[str, str] = {}

        if len(messages) > self.limits.max_messages:
            violations["max_messages"] = (
                f"conversation has {len(messages)} messages; limit is {self.limits.max_messages}"
            )

        # Measured against the SERIALIZED form so multimodal content-part arrays are covered too,
        # not just plain string content (#6c: JSON escaping and multibyte characters make the raw
        # character count an undercount).
        oversized_content: list[str] = []
        for message_index, msg in enumerate(messages):
            # Structural guard, folded into this pass so it costs no extra walk. Every limit below
            # indexes into the message, so a shape that cannot be indexed has to fail as a
            # validation error rather than let an AttributeError or TypeError escape to the caller.
            # ``validate_conversation`` already rejects both shapes with a fuller message on the
            # scanner path; this covers callers that construct ``ScanContext`` directly and so
            # never reach that validator.
            if not isinstance(msg, dict):
                raise ConversationValidationError(
                    f"Message at index {message_index} is not a dict (got {type(msg).__name__}). "
                    "Each message must be an OpenAI message object."
                )
            raw_calls = msg.get("tool_calls")
            if raw_calls is not None and not isinstance(raw_calls, list):
                raise ConversationValidationError(
                    f"Message at index {message_index} has a non-list 'tool_calls' "
                    f"(got {type(raw_calls).__name__}). It must be a list of tool-call objects."
                )
            try:
                content_size = len(orjson.dumps(msg.get("content")))
            except (TypeError, orjson.JSONEncodeError) as exc:
                raise ConversationValidationError(
                    f"Message at index {message_index} cannot be measured for safety limits: {exc}"
                ) from exc
            if content_size > self.limits.max_message_size:
                oversized_content.append(f"index {message_index} is {content_size} bytes")
        if oversized_content:
            violations["max_message_size"] = (
                f"{len(oversized_content)} of {len(messages)} messages exceed the "
                f"{self.limits.max_message_size}-byte limit on serialized content "
                f"({_summarize(oversized_content)})"
            )

        # Every preserved call consumes preparation and timeline work, including calls on
        # unreadable cross-role senders. Apply call and argument limits before materialization.
        materialized_calls = [
            (message_index, call_index, call)
            for message_index, msg in enumerate(messages)
            for call_index, call in enumerate(msg.get("tool_calls") or [])
        ]
        if len(materialized_calls) > self.limits.max_tool_calls:
            violations["max_tool_calls"] = (
                f"conversation has {len(materialized_calls)} tool calls; limit is "
                f"{self.limits.max_tool_calls}"
            )

        # Bound each raw argument blob BEFORE ``Conversation`` parses any JSON,
        # so an oversized payload is never materialized to be measured.
        oversized_arguments: list[str] = []
        for message_index, call_index, tool_call in materialized_calls:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            argument_size = (
                len(str.encode(arguments, "utf-8"))
                if isinstance(arguments, str)
                else _bounded_deep_size(arguments, self.limits.max_tool_argument_size)
            )
            if argument_size > self.limits.max_tool_argument_size:
                oversized_arguments.append(
                    f"message {message_index} tool_calls[{call_index}] is {argument_size} bytes"
                )
        if oversized_arguments:
            violations["max_tool_argument_size"] = (
                f"{len(oversized_arguments)} of {len(materialized_calls)} tool calls exceed the "
                f"{self.limits.max_tool_argument_size}-byte argument limit "
                f"({_summarize(oversized_arguments)})"
            )

        # Bound the eventual timeline — the event word the DFA walks. Without a check
        # here max_timeline_items has no enforcement site.
        max_items = self.limits.max_timeline_items
        if max_items and max_items > 0:
            item_count = sum(_timeline_items(msg) for msg in messages)
            if item_count > max_items:
                violations["max_timeline_items"] = (
                    f"conversation produces {item_count} timeline items; limit is {max_items}"
                )

        if violations:
            raise _limit_error(violations)

    def copy(self) -> "ScanContext":
        """Create a copy with fresh mutable state for parallel rule execution."""
        new_ctx = object.__new__(ScanContext)
        new_ctx.resource_manager = self.resource_manager
        new_ctx.function_cache = self.function_cache
        new_ctx.limits = self.limits
        new_ctx.conversation = self.conversation
        new_ctx.was_truncated = self.was_truncated
        new_ctx.truncation_info = self.truncation_info
        new_ctx._timeline_cache = self._timeline_cache
        new_ctx.tool_call_types = self.tool_call_types
        new_ctx.parameters = self.parameters
        new_ctx.parameter_specs = self.parameter_specs
        new_ctx.extension_event_surfaces = self.extension_event_surfaces
        new_ctx.extension_event_sequence = self.extension_event_sequence
        new_ctx.extension_state_values = self.extension_state_values
        new_ctx.missing_extension_states = self.missing_extension_states
        new_ctx.input_faults = self.input_faults
        new_ctx.ordered_timeline = self.ordered_timeline
        new_ctx.evaluation_warnings = []
        new_ctx.map_filter_results = {}
        new_ctx.message_value_memo = MessageValueMemo()
        return new_ctx

    @property
    def timeline(self) -> list[tuple[int, Message | ToolCall]]:
        """Timeline of conversation events, cached per context."""

        if self._timeline_cache is None:
            self._timeline_cache = self._build_timeline()
        return self._timeline_cache

    def _build_timeline(self) -> list[tuple[int, Message | ToolCall]]:
        """Construct the evaluation timeline: one event per canonical message plus one per
        preserved tool call, in transcript order.

        A tool result is the canonical ``ToolMessage`` itself — it occupies a single position
        and is later tagged (by ``engine/events.py``) as belonging to BOTH the ``messages`` and
        ``tool_responses`` surfaces. There is no separate synthesized tool-response event, so a
        ``read_file`` result is: AssistantMessage → ToolCall → ToolMessage (one position each).

        Complexity: O(messages + tool_calls). Tool calls attach to their owning message
        by root ``message_index`` so an accepted-but-earlier role (a leading system message,
        etc.) can never shift the attachment.
        """
        timeline: list[tuple[int, Message | ToolCall]] = []
        position = 0

        tool_calls_by_msg: dict[int, list[ToolCall]] = {}
        for tc in self.conversation.tool_calls:
            tool_calls_by_msg.setdefault(tc.message_index, []).append(tc)

        for msg in self.conversation.messages:
            raw_idx = msg.message_index
            timeline.append((position, msg))
            position += 1

            # A tool call is emitted right after its owning message.
            if raw_idx in tool_calls_by_msg:
                for tc in tool_calls_by_msg[raw_idx]:
                    timeline.append((position, tc))
                    position += 1

        return timeline
