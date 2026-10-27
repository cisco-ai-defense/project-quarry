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

import sys
from dataclasses import dataclass
from typing import Any

import orjson

from quarry.core.conversation import Conversation
from quarry.safety.limits import SafetyLimits
from quarry.types.exceptions import ConversationValidationError
from quarry.types.messages import Message, Messages, ToolCall, ToolCallTypeRegistry

# Offending items named individually inside ONE limit's detail string before it collapses to a
# count. Bounds the error: a 10,000-message conversation that breaches a limit on every message
# must not build a 10,000-clause message. This caps items WITHIN a violation, not the number of
# violations reported — every breached limit is always named.
_MAX_REPORTED_ITEMS = 5


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

    One per message, plus one per assistant tool call. A tool result is one canonical
    ``ToolMessage`` occupying a single position (it belongs to both the ``messages`` and
    ``tool_responses`` surfaces there), so it is not counted twice."""

    if msg.get("role") == "assistant":
        return 1 + len(msg.get("tool_calls") or [])
    return 1


def _bounded_deep_size(value: Any, limit: int, seen: set[int] | None = None) -> int:
    """Measure nested input without walking beyond the configured byte budget.

    Two properties callers must account for.

    The unit is the PYTHON OBJECT footprint (``sys.getsizeof``, summed recursively), not
    serialized length, and the gap between them depends on SHAPE rather than size. A payload
    dominated by a few large strings — a conversation carrying inline media — measures within a
    percent of its JSON length, because a ``str``'s footprint is its bytes plus a small header.
    A payload of many small objects pays that header thousands of times over: 10,000 short
    messages measure ~6x their JSON length, and a single tiny message ~12x.

    And the result saturates: once the running total passes ``limit`` this returns ``limit + 1``
    and stops walking, so a return value above ``limit`` means "over budget" and is not a
    measurement of how far over.
    """
    if seen is None:
        seen = set()

    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)

    try:
        total = sys.getsizeof(value)
    except TypeError:
        total = 0
    if total > limit:
        return limit + 1

    if isinstance(value, dict):
        children = (child for pair in value.items() for child in pair)
    elif isinstance(value, (list, tuple, set, frozenset)):
        children = iter(value)
    else:
        return total

    for child in children:
        total += _bounded_deep_size(child, limit - total, seen)
        if total > limit:
            return limit + 1
    return total


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
    ):
        self.resource_manager = resource_manager
        self.function_cache = function_cache
        self.tool_call_types = Messages.validate_tool_call_types(tool_call_types)
        self.parameters = parameters or {}
        self.parameter_specs = parameter_specs or {}
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
            self._enforce_limits(messages, tools)

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

    def _enforce_limits(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
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
        if not isinstance(messages, list):
            raise ConversationValidationError(
                f"messages must be a list, got {type(messages).__name__}"
            )
        if tools is not None and not isinstance(tools, list):
            raise ConversationValidationError(f"tools must be a list, got {type(tools).__name__}")

        # Total size first and alone. It is the limit that bounds the cost of measuring the
        # others, so a payload breaching it is rejected before the per-message and per-call
        # walks below run over it. Measured recursively: a shallow top-level sizeof
        # lets arbitrarily large nested argument dicts bypass the budget.
        total_size = _bounded_deep_size(
            {"messages": messages, "tools": tools or []}, self.limits.max_total_size
        )
        if total_size > self.limits.max_total_size:
            # No measured size is reported. ``_bounded_deep_size`` stops walking the moment it
            # passes the budget and returns ``limit + 1``, so any number here would always be
            # ``limit + 1`` — a fake measurement that reads like a real one. Stating the budget
            # and the unit is the honest version.
            raise _limit_error(
                {
                    "max_total_size": (
                        f"input exceeds the {self.limits.max_total_size}-byte budget. This limit "
                        "counts the PYTHON OBJECT footprint in memory, not serialized length. The "
                        "two are within a percent for a conversation dominated by large strings "
                        "such as inline media, but per-object overhead makes a conversation of "
                        "many small messages measure roughly 5-12x its JSON size. The exact size "
                        "is not reported because measurement stops at the budget rather than "
                        "walking an oversized payload in full"
                    )
                }
            )

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

        # Count only assistant calls — the OpenAI contract does not assign tool-call semantics to
        # other roles, so they must not consume the assistant budget.
        assistant_calls = [
            (message_index, call_index, call)
            for message_index, msg in enumerate(messages)
            if msg.get("role") == "assistant"
            for call_index, call in enumerate(msg.get("tool_calls") or [])
        ]
        if len(assistant_calls) > self.limits.max_tool_calls:
            violations["max_tool_calls"] = (
                f"conversation has {len(assistant_calls)} assistant tool calls; limit is "
                f"{self.limits.max_tool_calls}"
            )

        # Bound each raw argument blob BEFORE ``Conversation`` parses any JSON,
        # so an oversized payload is never materialized to be measured.
        oversized_arguments: list[str] = []
        for message_index, call_index, tool_call in assistant_calls:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            argument_size = (
                len(arguments.encode("utf-8"))
                if isinstance(arguments, str)
                else _bounded_deep_size(arguments, self.limits.max_tool_argument_size)
            )
            if argument_size > self.limits.max_tool_argument_size:
                oversized_arguments.append(
                    f"message {message_index} tool_calls[{call_index}] is {argument_size} bytes"
                )
        if oversized_arguments:
            violations["max_tool_argument_size"] = (
                f"{len(oversized_arguments)} of {len(assistant_calls)} tool calls exceed the "
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
        new_ctx.evaluation_warnings = []
        new_ctx.map_filter_results = {}
        return new_ctx

    @property
    def timeline(self) -> list[tuple[int, Message | ToolCall]]:
        """Timeline of conversation events, cached per context."""

        if self._timeline_cache is None:
            self._timeline_cache = self._build_timeline()
        return self._timeline_cache

    def _build_timeline(self) -> list[tuple[int, Message | ToolCall]]:
        """Construct the evaluation timeline: one event per canonical message plus one per
        assistant tool call, in transcript order.

        A tool result is the canonical ``ToolMessage`` itself — it occupies a single position
        and is later tagged (by ``engine/events.py``) as belonging to BOTH the ``messages`` and
        ``tool_responses`` surfaces. There is no separate synthesized tool-response event, so a
        ``read_file`` result is: AssistantMessage → ToolCall → ToolMessage (one position each).

        Complexity: O(messages + tool_calls). Tool calls attach to their owning assistant turn
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

            # A tool call is emitted right after its owning assistant message.
            if raw_idx in tool_calls_by_msg:
                for tc in tool_calls_by_msg[raw_idx]:
                    timeline.append((position, tc))
                    position += 1

        return timeline
