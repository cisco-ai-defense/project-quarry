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

"""Canonical immutable conversation wrapper for scanning and model input.

``Conversation`` owns one ``Messages`` graph. Role and tool views derive from that graph,
and serialization rebuilds from its frozen values. A tool result is one canonical
``ToolMessage`` in the timeline that belongs to both the message and tool-response
surfaces; no duplicate event is synthesized.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import ValidationError

from quarry.language.metadata import DSLParameter, Use, dsl_function, dsl_surface
from quarry.safety.validator import validate_conversation
from quarry.types.exceptions import ConversationValidationError
from quarry.types.messages import (
    Message,
    Messages,
    Tool,
    ToolCall,
    ToolCallTypeRegistry,
    ToolMessage,
)


class Conversation:
    """Own request-level tools around one canonical ``Messages`` collection."""

    __slots__ = (
        "messages",
        "_tools",
        "_tools_by_name",
        "_tools_supplied",
        "_sealed",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject changes after the conversation graph has been constructed."""

        if getattr(self, "_sealed", False):
            raise AttributeError("Conversation is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Reject attribute deletion from an immutable conversation."""

        raise AttributeError("Conversation is immutable")

    def __init__(
        self,
        messages: list[dict[str, Any]] | Sequence[Message] | Messages,
        tools: list[dict[str, Any]] | None = None,
        *,
        tool_call_types: ToolCallTypeRegistry | None = None,
        max_nesting_depth: int | None = None,
    ) -> None:
        """Build the canonical graph from validated OpenAI data.

        The raw-list path *trusts its caller*: the scanner runs ``validate_conversation`` on
        the input before constructing ``ScanContext``/``Conversation`` (scanner.py), so the
        engine layer builds the canonical graph without re-validating. This also permits
        engine-level fixtures the strict OpenAI validator would reject. Callers with
        unvalidated data should use :meth:`from_openai`.
        """

        max_depth = (
            messages.max_nesting_depth
            if isinstance(messages, Messages)
            else 100
            if max_nesting_depth is None
            else max_nesting_depth
        )
        self._tools_supplied = tools is not None
        raw_tools = tools if tools is not None else []

        if isinstance(messages, Messages):
            if not messages._is_root_view():
                raise ConversationValidationError(
                    "Conversation requires a root Messages collection, not a filtered or sliced view"
                )
            if tool_call_types is not None or max_nesting_depth is not None:
                raise ConversationValidationError(
                    "Messages already owns tool_call_types and max_nesting_depth; "
                    "configure them with Messages.from_openai()"
                )
            self.messages = messages
        else:
            raw_messages = [
                message.to_openai_dict() if isinstance(message, Message) else message
                for message in messages
            ]
            # The scanner validates before reaching here, so well-formed inputs build cleanly.
            # A direct caller that bypasses validation with malformed data fails closed with a
            # typed ``ConversationValidationError`` instead of a raw pydantic/KeyError leak.
            try:
                self.messages = Messages._from_validated(
                    self._canonicalize_raw_messages(raw_messages),
                    tool_call_types,
                    max_depth,
                )
            except (ValidationError, KeyError, TypeError, ValueError) as exc:
                raise ConversationValidationError(f"Invalid conversation messages: {exc}") from exc
        self._set_tools(raw_tools)
        self._sealed = True

    # Canonical OpenAI wire keys per role — everything else on a trusted (post-scanner or
    # engine-direct) raw message is dropped. Notably a tool
    # message's legacy ``name`` (redundant with the linked call) is discarded so the strict
    # ``extra='forbid'`` ``ToolMessage`` accepts real-world traces that carry it.
    _ROLE_WIRE_KEYS = {
        "assistant": frozenset({"role", "content", "refusal", "tool_calls"}),
        "tool": frozenset({"role", "content", "tool_call_id"}),
    }
    _DEFAULT_WIRE_KEYS = frozenset({"role", "content"})

    @classmethod
    def _canonicalize_raw_messages(cls, raw_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Project trusted raw messages onto their canonical wire keys.

        Structure was already validated upstream (scanner ``validate_conversation``); this
        only strips non-canonical extras so the strict message models accept provider fields.
        Validated inputs never carry extras, so their build is
        unchanged. A malformed entry is passed through untouched for the model to reject with
        a precise error."""

        canonical: list[dict[str, Any]] = []
        for raw in raw_messages:
            if not isinstance(raw, dict) or "role" not in raw:
                canonical.append(raw)
                continue
            role = raw["role"]
            allowed = cls._ROLE_WIRE_KEYS.get(role, cls._DEFAULT_WIRE_KEYS)
            clean = {key: value for key, value in raw.items() if key in allowed}
            if role == "tool" and clean.get("content") is None:
                clean["content"] = ""
            canonical.append(clean)
        return canonical

    def _set_tools(self, raw_tools: list[dict[str, Any]]) -> None:
        """Build the canonical ordered tools list after structural validation."""

        try:
            canonical_tools = tuple(Tool.model_validate(tool, strict=True) for tool in raw_tools)
        except (ValueError, TypeError) as exc:
            raise ConversationValidationError(f"Invalid OpenAI tool definition: {exc}") from exc
        self._tools = canonical_tools
        self._tools_by_name = MappingProxyType({tool.name: tool for tool in self._tools})
        if len(self._tools_by_name) != len(self._tools):
            raise ConversationValidationError("Tool definitions must have unique function names")

    @property
    @dsl_surface(
        name="tools",
        public_type="collection<tool_definition>",
        domain="static",
        allowed_uses=(Use.COMPREHENSION_SOURCE,),
        fields_from=Tool,
    )
    def tools(self) -> list[Tool]:
        """Tools offered to the model in the inspected request.

        This static collection describes what the model was allowed to call. It does not
        indicate that the assistant actually called any of those tools.
        """

        return list(self._tools)

    @property
    def raw_messages(self) -> list[dict[str, Any]]:
        """Return a fresh compatibility projection of the source messages."""

        return self.messages.to_openai_messages()

    @property
    def raw_tools(self) -> list[dict[str, Any]]:
        """Return a fresh compatibility projection of the offered tools."""

        return [tool.to_openai_dict() for tool in self._tools]

    @property
    @dsl_surface(
        name="system_prompt",
        public_type="string",
        domain="static",
        allowed_uses=(),
    )
    def system_prompt(self) -> str | None:
        """Text content of the leading system message."""

        return self.messages.system_prompt

    @property
    def user_messages(self) -> Messages:
        """Compatibility view of user messages."""

        return self.messages.user_messages

    @property
    def assistant_messages(self) -> Messages:
        """Compatibility view of assistant messages."""

        return self.messages.assistant_messages

    @property
    def system_messages(self) -> Messages:
        """Compatibility view of system messages."""

        return self.messages.system_messages

    @property
    def developer_messages(self) -> Messages:
        """Compatibility view of developer messages."""

        return self.messages.developer_messages

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Compatibility projection of observed tool calls."""

        return self.messages.tool_calls

    @property
    def tool_responses(self) -> Messages:
        """Compatibility view of canonical tool-result messages."""

        return self.messages.tool_responses

    def to_openai_messages(self) -> list[dict[str, Any]]:
        """Return fresh OpenAI-compatible messages."""

        return self.messages.to_openai_messages()

    def to_openai_dict(self) -> dict[str, Any]:
        """Return the complete OpenAI-compatible conversation object."""

        result: dict[str, Any] = {"messages": self.to_openai_messages()}
        if self._tools_supplied:
            result["tools"] = self.raw_tools
        return result

    def to_json(self) -> str:
        """Return compact standards-compliant JSON for the conversation."""

        return json.dumps(
            self.to_openai_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    def tool_by_name(self, name: str) -> Tool | None:
        """Return the offered tool with this exact name, if present."""

        return self._tools_by_name.get(name)

    @dsl_function(
        allowed_uses=(Use.CONDITION,),
    )
    def has_tool(self, name: Annotated[str, DSLParameter(literal_only=True)]) -> bool:
        """Return whether a tool with this exact name was offered to the model.

        This checks the request's tool definitions, not the tool calls made by the assistant.
        """

        return name in self._tools_by_name

    @dsl_function(
        allowed_uses=(Use.CONDITION,),
    )
    def has_tools(
        self,
        *names: Annotated[str, DSLParameter(required=True, literal_only=True)],
    ) -> bool:
        """Return whether every named tool was offered to the model.

        At least one name is required. This checks availability, not whether the assistant used
        the tools.
        """

        return all(name in self._tools_by_name for name in names)

    def called_tool(self, name: str) -> bool:
        """Compatibility delegate to the root message collection."""

        return self.messages.called_tool(name)

    def called_tools(self, *names: str) -> bool:
        """Compatibility delegate to the root message collection."""

        return self.messages.called_tools(*names)

    def count_tool_calls(self, name: str) -> int:
        """Compatibility delegate to the root message collection."""

        return self.messages.count_tool_calls(name)

    def get_tool_calls(self, name: str) -> list[ToolCall]:
        """Compatibility delegate to the root message collection."""

        return self.messages.get_tool_calls(name)

    def get_tool_responses(self, name: str) -> list[ToolMessage]:
        """Compatibility delegate to the root message collection."""

        return self.messages.get_tool_responses(name)

    def user_said(self, text: str, case_sensitive: bool = False) -> bool:
        """Compatibility delegate to the root message collection."""

        return self.messages.user_said(text, case_sensitive)

    def assistant_said(self, text: str, case_sensitive: bool = False) -> bool:
        """Compatibility delegate to the root message collection."""

        return self.messages.assistant_said(text, case_sensitive)

    def call_sequence(self) -> list[str]:
        """Compatibility delegate to the root message collection."""

        return self.messages.call_sequence()

    def has_sequence(self, *names: str) -> bool:
        """Compatibility delegate to the root message collection."""

        return self.messages.has_sequence(*names)

    @classmethod
    def from_openai(
        cls,
        value: Mapping[str, Any],
        *,
        tool_call_types: ToolCallTypeRegistry | None = None,
        max_nesting_depth: int = 100,
    ) -> Conversation:
        """Build from a complete OpenAI-compatible conversation dictionary."""

        if not isinstance(value, Mapping):
            raise ConversationValidationError(f"Expected mapping, got {type(value).__name__}")
        payload = dict(value)
        validate_conversation(
            payload,
            parse_arguments=False,
            max_nesting_depth=max_nesting_depth,
        )

        conversation = cls.__new__(cls)
        conversation._tools_supplied = "tools" in payload
        conversation.messages = Messages._from_validated(
            payload["messages"],
            tool_call_types,
            max_nesting_depth,
        )
        conversation._set_tools(payload.get("tools", []))
        conversation._sealed = True
        return conversation

    @classmethod
    def _from_parts(
        cls,
        messages: Messages,
        tools: Sequence[Tool] = (),
        *,
        tools_supplied: bool | None = None,
    ) -> Conversation:
        """Wrap an existing ``Messages`` (root OR view) and shared ``Tool`` objects WITHOUT
        reparsing.

        Reference-backed: ``self.messages`` IS ``messages`` and the offered tools are the SAME
        ``Tool`` instances — no dict round-trip, no re-validation, no copy. This is how a
        Messages-only model-input selection is normalized to a tools-less conversation, and how
        token-bounded chunks reuse the transcript's canonical objects (so ``chunk.messages[i]``
        and ``chunk.tools[j]`` are identical to the source's). The caller owns the invariant
        that ``messages`` is a valid standalone sequence: chunking preserves call/result units,
        and a selection is checked with :meth:`Messages.assert_self_contained`."""

        conversation = cls.__new__(cls)
        conversation._tools_supplied = bool(tools) if tools_supplied is None else tools_supplied
        conversation.messages = messages
        conversation._tools = tuple(tools)
        conversation._tools_by_name = MappingProxyType(
            {tool.name: tool for tool in conversation._tools}
        )
        conversation._sealed = True
        return conversation
