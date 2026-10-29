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

"""Canonical OpenAI conversation objects exposed to Quarry rules."""

from __future__ import annotations

import json
import math
import re
import sys
from bisect import bisect_left
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from inspect import getattr_static
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Literal, Self, overload

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from quarry.language.metadata import (
    DSLParameter,
    SurfaceFieldExclusion,
    SurfaceFieldMetadata,
    Use,
    dsl_function,
    dsl_surface,
)
from quarry.types.exceptions import ConversationValidationError, EvaluationError
from quarry.types.results import Span


def wire_digest(wire: Mapping[str, Any]) -> bytes:
    """The digest of one message's wire form, as ``Message.digest`` computes it."""

    from quarry.cache.keys import input_key

    return input_key("message", wire)


def messages_digest(digests: Iterable[bytes]) -> bytes:
    """The digest of messages in order, from their own digests, as ``Messages.digest``
    computes it."""

    from quarry.cache.keys import combine_digests

    return combine_digests("messages", digests)


class _FrozenDict(Mapping[str, Any]):
    """Small immutable mapping used for nested source JSON."""

    __slots__ = ("_items", "_values", "_sealed")

    def __init__(self, items: Iterable[tuple[str, Any]]):
        self._items = tuple(items)
        self._values = MappingProxyType(dict(self._items))
        self._sealed = True

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("Frozen JSON object is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Frozen JSON object is immutable")

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._items)


class _ToolIntegerLimitError(ValueError):
    """A valid JSON integer exceeds the active decimal conversion limit."""


def _parse_tool_integer(raw: str) -> int:
    """Check decimal size explicitly before CPython's ``int`` conversion."""

    limit = sys.get_int_max_str_digits()
    digits = len(raw) - int(raw.startswith("-"))
    if limit and digits > limit:
        raise _ToolIntegerLimitError(
            f"JSON integer has {digits} decimal digits; active limit is {limit}"
        )
    return int(raw)


def _freeze_json(value: Any) -> Any:
    """Return an immutable JSON tree while rejecting non-JSON numeric values."""

    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        try:
            json.dumps(value)
        except ValueError as exc:
            raise ValueError("OpenAI integer exceeds the active JSON serializer limit") from exc
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("OpenAI data cannot contain non-finite numbers")
        return value
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("OpenAI JSON object keys must be strings")
        return _FrozenDict((key, _freeze_json(item)) for key, item in value.items())
    raise ValueError(f"Unsupported OpenAI JSON value: {type(value).__name__}")


def _project_json(value: Any) -> Any:
    """Project immutable source JSON into fresh ordinary containers."""

    if isinstance(value, BaseModel):
        return _project_json(value.model_dump(mode="python", by_alias=True, exclude_unset=True))
    if isinstance(value, Mapping):
        return {key: _project_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_project_json(item) for item in value]
    return value


def _hashable_json(value: Any) -> Any:
    """Return a hashable tree that follows Python's JSON-container equality."""

    if isinstance(value, Mapping):
        return ("object", tuple(sorted((key, _hashable_json(item)) for key, item in value.items())))
    if isinstance(value, list | tuple):
        return ("array", tuple(_hashable_json(item) for item in value))
    return ("scalar", value)


def _validate_depth(value: Any, max_depth: int) -> None:
    """Reject a decoded JSON container whose nesting exceeds ``max_depth``."""

    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if isinstance(current, Mapping):
            if depth > max_depth:
                raise EvaluationError(f"Tool arguments exceed max_nesting_depth {max_depth}")
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list | tuple):
            if depth > max_depth:
                raise EvaluationError(f"Tool arguments exceed max_nesting_depth {max_depth}")
            stack.extend((child, depth + 1) for child in current)


class OpenAIObject(BaseModel):
    """Immutable OpenAI-shaped object with lossless JSON-data serialization."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )
    DSL_FIELDS: ClassVar[frozenset[str]] = frozenset()
    DSL_FIELD_METADATA: ClassVar[tuple[SurfaceFieldMetadata, ...]] = ()
    DSL_FIELD_EXCLUSIONS: ClassVar[tuple[SurfaceFieldExclusion, ...]] = ()

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject observable mutation; construction and lazy caches use an internal bypass."""

        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        """Reject attribute deletion from an immutable OpenAI object."""

        raise AttributeError(f"{type(self).__name__} is immutable")

    def to_openai_dict(self) -> dict[str, Any]:
        """Return this object's OpenAI-compatible dictionary."""

        dumped = self.model_dump(mode="python", by_alias=True, exclude_unset=True)
        return _project_json(dumped)

    def to_json(self) -> str:
        """Return compact standards-compliant JSON for this object."""

        return json.dumps(
            self.to_openai_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    def __eq__(self, other: object) -> bool:
        """Compare only canonical wire data, never private graph links or caches."""

        if not isinstance(other, OpenAIObject):
            return NotImplemented
        return self.to_openai_dict() == other.to_openai_dict()

    def __hash__(self) -> int:
        """Hash canonical wire data consistently with value equality."""

        return hash(_hashable_json(self.to_openai_dict()))


class ContentPart(OpenAIObject):
    """Base class for one ordered multimodal message-content part."""

    # Subclasses narrow this discriminator; OpenAIObject rejects attribute writes.
    type: str
    DSL_FIELDS: ClassVar[frozenset[str]] = frozenset({"type"})


class TextPart(ContentPart):
    """OpenAI text content part."""

    type: Literal["text"]  # ty: ignore[invalid-mutable-override]
    text: str
    DSL_FIELDS: ClassVar[frozenset[str]] = ContentPart.DSL_FIELDS | {"text"}


class ImageURL(OpenAIObject):
    """OpenAI image URL payload."""

    url: str
    detail: Literal["auto", "low", "high"] | None = None


class ImagePart(ContentPart):
    """OpenAI image content part."""

    type: Literal["image_url"]  # ty: ignore[invalid-mutable-override]
    image_source: ImageURL = Field(alias="image_url")
    DSL_FIELDS: ClassVar[frozenset[str]] = ContentPart.DSL_FIELDS | {"image_url"}

    @property
    def image_url(self) -> str:
        """Return the image URL for DSL field access."""

        return self.image_source.url


class InputAudio(OpenAIObject):
    """OpenAI input-audio payload."""

    data: str
    format: Literal["wav", "mp3"]


class AudioPart(ContentPart):
    """OpenAI input-audio content part."""

    type: Literal["input_audio"]  # ty: ignore[invalid-mutable-override]
    audio_source: InputAudio = Field(alias="input_audio")
    DSL_FIELDS: ClassVar[frozenset[str]] = ContentPart.DSL_FIELDS | {"input_audio"}

    @property
    def input_audio(self) -> dict[str, Any]:
        """Return a defensive copy of the OpenAI audio payload."""

        return self.audio_source.to_openai_dict()


def _content_part(source: Mapping[str, Any]) -> ContentPart:
    """Materialize one already-verified content part."""

    payload = _project_json(source)
    part_type = payload.get("type")
    part_class: type[ContentPart]
    if part_type == "text":
        part_class = TextPart
    elif part_type == "image_url":
        part_class = ImagePart
    elif part_type == "input_audio":
        part_class = AudioPart
    else:
        raise ConversationValidationError(f"Unsupported OpenAI content part '{part_type}'")
    return part_class.model_validate(payload, strict=True)


class Message(OpenAIObject):
    """One role-bearing conversation entry."""

    # Subclasses narrow this discriminator; OpenAIObject rejects attribute writes.
    VALID_ROLES: ClassVar[frozenset[str]] = frozenset(
        {"system", "developer", "user", "assistant", "tool"}
    )
    DSL_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"role", "content", "content_parts", "openai_content", "message_index"}
    )
    DSL_FIELD_METADATA: ClassVar[tuple[SurfaceFieldMetadata, ...]] = (
        SurfaceFieldMetadata("role", "string", "OpenAI message role."),
        SurfaceFieldMetadata(
            "content",
            "string",
            "Flattened message text. For multimodal input, this contains text parts only.",
        ),
    )
    DSL_FIELD_EXCLUSIONS: ClassVar[tuple[SurfaceFieldExclusion, ...]] = (
        SurfaceFieldExclusion(
            "content_parts",
            "Typed content-part collections cannot be used as comprehension sources.",
        ),
        SurfaceFieldExclusion(
            "message_index",
            "Transcript positions cannot be used in compiled numeric comparisons.",
        ),
        SurfaceFieldExclusion(
            "openai_content",
            "The raw OpenAI value may be structured; use content for deterministic text rules.",
        ),
    )

    role: str
    source_content: Any = Field(default=None, alias="content")

    _content_cache: str | None = PrivateAttr(default=None)
    _content_parts_cache: tuple[ContentPart, ...] | None = PrivateAttr(default=None)
    _digest_cache: bytes | None = PrivateAttr(default=None)
    _message_index: int = PrivateAttr(default=-1)
    _segment: tuple[int, int] | None = PrivateAttr(default=None)

    @property
    def digest(self) -> bytes:
        """The digest of this message's wire form, computed on first use."""

        digest = self._digest_cache
        if digest is None:
            digest = wire_digest(self.to_openai_dict())
            object.__setattr__(self, "_digest_cache", digest)
        return digest

    @field_validator("role")
    @classmethod
    def _validate_role(cls, value: str) -> str:
        if value not in cls.VALID_ROLES:
            raise ValueError(f"invalid message role {value!r}")
        return value

    @field_validator("source_content", mode="before")
    @classmethod
    def _freeze_source_content(cls, value: Any) -> Any:
        if value is None or isinstance(value, str):
            return value
        if not isinstance(value, list | tuple):
            raise ValueError("message content must be a string, null, or content-part array")
        return _freeze_json(value)

    @property
    def content(self) -> str:
        """Return flattened text while retaining the source content separately."""

        if self._content_cache is not None:
            return self._content_cache
        source = self.source_content
        if isinstance(source, str):
            flattened = source
        elif source is None:
            flattened = ""
        else:
            flattened = "".join(
                str(part.get("text", ""))
                for part in source
                if isinstance(part, Mapping) and part.get("type") == "text"
            )
        object.__setattr__(self, "_content_cache", flattened)
        return flattened

    @property
    def content_parts(self) -> list[ContentPart]:
        """Return typed parts for array content, materialized only on first access."""

        parts = self._content_parts_cache
        if parts is None:
            source = self.source_content
            parts = (
                tuple(_content_part(part) for part in source) if isinstance(source, tuple) else ()
            )
            object.__setattr__(self, "_content_parts_cache", parts)
        return list(parts)

    @property
    def openai_content(self) -> Any:
        """Return a defensive projection of the original OpenAI content value."""

        return _project_json(self.source_content)

    @property
    def message_index(self) -> int:
        """Return this message's root transcript ordinal."""

        return self._message_index

    @property
    def _range(self) -> tuple[int, int]:
        """Return the ``[start, end)`` range of the root message's text a piece holds."""

        if self._segment is None:
            raise ValueError("message is whole, not a range of its root message")
        return self._segment

    def _segment_of(self, start: int, end: int) -> Self:
        """Return this message holding only ``content[start:end]``, at the same index.

        Tool calls and refusal stay with the range that ends the text, so a message cut
        into ranges carries them once.
        """

        if not isinstance(self.source_content, str):
            raise ValueError("only a message with string content can be divided")
        if not 0 <= start < end <= len(self.source_content):
            raise ValueError("message segment is outside the message content")
        update: dict[str, Any] = {"source_content": self.source_content[start:end]}
        if end < len(self.source_content) and isinstance(self, AssistantMessage):
            update |= {"tool_calls": (), "refusal": None}
        segment = self.model_copy(update=update)
        object.__setattr__(segment, "_content_cache", None)
        object.__setattr__(segment, "_content_parts_cache", None)
        object.__setattr__(segment, "_digest_cache", None)
        object.__setattr__(segment, "_segment", (start, end))
        return segment

    def contains(self, text: str, case_sensitive: bool = False) -> bool:
        """Return whether flattened text contains ``text``."""

        return text in self.content if case_sensitive else text.lower() in self.content.lower()

    def matches(self, pattern: str) -> bool:
        """Return whether flattened text matches ``pattern``."""

        # Import lazily so this foundational module does not close the safety
        # package's import cycle during Quarry startup.
        from quarry.cache.regex_cache import regex_cache

        return regex_cache.search(pattern, self.content, re.IGNORECASE)


class SystemMessage(Message):
    """System instruction message."""

    role: Literal["system"]  # ty: ignore[invalid-mutable-override]
    source_content: str | tuple[Any, ...] = Field(alias="content")


class DeveloperMessage(Message):
    """Developer instruction message."""

    role: Literal["developer"]  # ty: ignore[invalid-mutable-override]
    source_content: str | tuple[Any, ...] = Field(alias="content")


class UserMessage(Message):
    """User message."""

    role: Literal["user"]  # ty: ignore[invalid-mutable-override]
    source_content: str | tuple[Any, ...] = Field(alias="content")


class ToolCallFunction(OpenAIObject):
    """Function payload nested inside an OpenAI tool call."""

    name: str
    arguments: str


class ToolCall(OpenAIObject):
    """Lossless tool call whose argument object is decoded lazily."""

    dsl_fields: ClassVar[frozenset[str]] = frozenset()
    DSL_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"id", "type", "name", "raw_arguments", "arguments", "message_index", "response"}
    )
    DSL_FIELD_METADATA: ClassVar[tuple[SurfaceFieldMetadata, ...]] = (
        SurfaceFieldMetadata("id", "string", "Provider-assigned tool-call ID."),
        SurfaceFieldMetadata("type", "string", "Tool-call type; currently `function`."),
        SurfaceFieldMetadata("name", "string", "Exact called function name."),
        SurfaceFieldMetadata("raw_arguments", "string", "Original JSON argument text."),
        SurfaceFieldMetadata(
            "arguments",
            "object",
            "Decoded JSON arguments. Read a key with a path such as "
            "`call.arguments.command`, or all values with `call.arguments.*`.",
        ),
        SurfaceFieldMetadata(
            "response",
            "tool_response | null",
            "Linked tool response, or null when no response exists. Continue with a field such "
            "as `call.response.result`.",
        ),
    )
    DSL_FIELD_EXCLUSIONS: ClassVar[tuple[SurfaceFieldExclusion, ...]] = (
        SurfaceFieldExclusion(
            "message_index",
            "Transcript positions cannot be used in compiled numeric comparisons.",
        ),
    )

    id: str
    type: Literal["function"]
    function: ToolCallFunction

    _arguments_cache: _FrozenDict | None = PrivateAttr(default=None)
    _message_index: int = PrivateAttr(default=-1)
    _max_nesting_depth: int = PrivateAttr(default=100)
    _response: ToolMessage | None = PrivateAttr(default=None)

    @property
    def name(self) -> str:
        """Return the called function name."""

        return self.function.name

    @property
    def raw_arguments(self) -> str:
        """Return the original JSON argument string."""

        return self.function.arguments

    @property
    def arguments(self) -> dict[str, Any]:
        """Decode and return a defensive argument object on first access.

        JSON integers retain exact Python ``int`` values up to the interpreter's
        active decimal-digit limit; larger integers report that limit explicitly.
        """

        if self._arguments_cache is None:
            try:
                parsed = json.loads(self.raw_arguments, parse_int=_parse_tool_integer)
            except _ToolIntegerLimitError as exc:
                raise EvaluationError(
                    f"Tool call '{self.id}' arguments exceed the JSON integer digit limit: {exc}"
                ) from exc
            except (json.JSONDecodeError, ValueError, RecursionError) as exc:
                raise EvaluationError(
                    f"Tool call '{self.id}' has malformed JSON arguments: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise EvaluationError(
                    f"Tool call '{self.id}' arguments must decode to a JSON object"
                )
            _validate_depth(parsed, self._max_nesting_depth)
            try:
                frozen = _freeze_json(parsed)
            except ValueError as exc:
                raise EvaluationError(
                    f"Tool call '{self.id}' arguments contain invalid JSON data: {exc}"
                ) from exc
            object.__setattr__(self, "_arguments_cache", frozen)
        return _project_json(self._arguments_cache)

    def _discard_decoded_arguments(self) -> None:
        """Drop a tree decoded only for input preflight; rule reads stay lazy."""

        object.__setattr__(self, "_arguments_cache", None)

    @property
    def message_index(self) -> int:
        """Return the owning assistant message's root ordinal."""

        return self._message_index

    @property
    def response(self) -> ToolMessage | None:
        """Return the canonical tool-result message linked to this call."""

        return self._response


class AssistantMessage(Message):
    """Assistant message with optional refusal text and tool calls."""

    role: Literal["assistant"]  # ty: ignore[invalid-mutable-override]
    source_content: str | tuple[Any, ...] | None = Field(default=None, alias="content")
    refusal: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    DSL_FIELDS: ClassVar[frozenset[str]] = Message.DSL_FIELDS | {"refusal", "tool_calls"}
    DSL_FIELD_METADATA: ClassVar[tuple[SurfaceFieldMetadata, ...]] = (
        *Message.DSL_FIELD_METADATA,
        SurfaceFieldMetadata(
            "refusal",
            "string | null",
            "Assistant refusal text when supplied by the provider; otherwise null.",
        ),
        SurfaceFieldMetadata(
            "tool_calls",
            "collection<tool_call>",
            "Calls emitted with this assistant message. Use the `tool_calls` surface to iterate "
            "over calls; this field can be measured with `len(...)`.",
        ),
    )


class ToolMessage(Message):
    """Canonical tool-result message linked to the call it answers."""

    role: Literal["tool"]  # ty: ignore[invalid-mutable-override]
    source_content: str = Field(alias="content")
    tool_call_id: str
    DSL_FIELDS: ClassVar[frozenset[str]] = Message.DSL_FIELDS | {
        "tool_call_id",
        "call",
        "id",
        "name",
        "result",
    }
    DSL_FIELD_METADATA: ClassVar[tuple[SurfaceFieldMetadata, ...]] = (
        *Message.DSL_FIELD_METADATA,
        SurfaceFieldMetadata("tool_call_id", "string", "ID of the call this message answers."),
        SurfaceFieldMetadata(
            "call",
            "tool_call",
            "Originating call. Continue with a field such as `response.call.name`.",
        ),
        SurfaceFieldMetadata("id", "string", "Alias for `tool_call_id`."),
        SurfaceFieldMetadata("name", "string", "Exact name of the called function."),
        SurfaceFieldMetadata("result", "string", "Tool-result text; alias for `content`."),
    )

    _call: ToolCall | None = PrivateAttr(default=None)

    @property
    def call(self) -> ToolCall:
        """Return the exact canonical call answered by this message."""

        if self._call is None:
            raise ConversationValidationError(
                f"Tool message references unknown tool_call_id '{self.tool_call_id}'"
            )
        return self._call

    @property
    def id(self) -> str:
        """Compatibility alias for ``tool_call_id``."""

        return self.tool_call_id

    @property
    def name(self) -> str:
        """Return the answered function name."""

        return self.call.name

    @property
    def result(self) -> str:
        """Compatibility alias for tool-result content."""

        return self.source_content


class ToolFunction(OpenAIObject):
    """Function definition nested inside an OpenAI tool definition."""

    name: str
    source_description: str = Field(default="", alias="description")
    source_parameters: Any = Field(default_factory=lambda: _FrozenDict(()), alias="parameters")

    @field_validator("source_parameters", mode="before")
    @classmethod
    def _freeze_parameters(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            raise ValueError("tool parameters must be an object")
        return _freeze_json(value)


class Tool(OpenAIObject):
    """One function tool offered with a conversation."""

    type: Literal["function"]
    function: ToolFunction
    DSL_FIELDS: ClassVar[frozenset[str]] = frozenset({"type", "name", "description", "parameters"})
    DSL_FIELD_METADATA: ClassVar[tuple[SurfaceFieldMetadata, ...]] = (
        SurfaceFieldMetadata("type", "string", "Offered tool type; currently `function`."),
        SurfaceFieldMetadata("name", "string", "Exact offered function name."),
        SurfaceFieldMetadata("description", "string", "Description shown to the model."),
        SurfaceFieldMetadata("parameters", "object", "JSON Schema for the function arguments."),
    )

    @property
    def name(self) -> str:
        """Return the offered function name."""

        return self.function.name

    @property
    def description(self) -> str:
        """Return the offered function description or an empty string."""

        return self.function.source_description

    @property
    def parameters(self) -> dict[str, Any]:
        """Return a defensive tool-schema projection."""

        return _project_json(self.function.source_parameters)


MessageType = SystemMessage | DeveloperMessage | UserMessage | AssistantMessage | ToolMessage
ToolCallTypeRegistry = Mapping[tuple[str, str], type[ToolCall]]


@dsl_surface(
    name="messages",
    public_type="collection<message>",
    domain="event",
    allowed_uses=(Use.COMPREHENSION_SOURCE,),
    fields_from=Message,
    model_input=True,
)
class Messages(Sequence[MessageType]):
    """All messages in the inspected conversation, in chronological order.

    The collection contains system, developer, user, assistant, and tool-result messages. Use
    the bare ``messages`` surface in rule logic and as tools-free model input.
    """

    DSL_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "system_prompt",
            "system_messages",
            "developer_messages",
            "user_messages",
            "assistant_messages",
            "tool_calls",
            "tool_responses",
        }
    )

    __slots__ = (
        "_root",
        "_items",
        "_ordinals",
        "_max_nesting_depth",
        "_all_calls",
        "_ordinals_by_role",
        "_calls_by_name",
        "_responses_by_name",
        "_tool_call_types",
        "_segments",
        "_chunk",
        "_sealed",
    )

    def __init__(self) -> None:
        """Create an empty construction base."""

        self._root = self
        self._items: tuple[MessageType, ...] = ()
        self._ordinals: tuple[int, ...] = ()
        self._max_nesting_depth = 100
        self._all_calls: tuple[ToolCall, ...] = ()
        self._ordinals_by_role: Mapping[str, tuple[int, ...]] = MappingProxyType({})
        self._calls_by_name: Mapping[str, tuple[ToolCall, ...]] = MappingProxyType({})
        self._responses_by_name: Mapping[str, tuple[ToolMessage, ...]] = MappingProxyType({})
        self._tool_call_types: ToolCallTypeRegistry = MappingProxyType({})
        # Root ordinal -> the segment of that message this view holds instead of the whole.
        self._segments: Mapping[int, MessageType] = MappingProxyType({})
        # Whether this view is a ``split`` chunk (see ``_select_ranges``).
        self._chunk = False
        self._sealed = True

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject changes after the collection graph has been constructed."""

        if getattr(self, "_sealed", False):
            raise AttributeError("Messages is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Reject attribute deletion from an immutable collection."""

        raise AttributeError("Messages is immutable")

    @classmethod
    def from_openai(
        cls,
        raw_messages: list[dict[str, Any]],
        *,
        tool_call_types: ToolCallTypeRegistry | None = None,
        max_nesting_depth: int = 100,
    ) -> Messages:
        """Verify and build canonical messages from an OpenAI message array."""

        cls._validate_nesting_limit(max_nesting_depth)
        registry = cls.validate_tool_call_types(tool_call_types)
        from quarry.safety.validator import validate_conversation

        validate_conversation(
            {"messages": raw_messages},
            parse_arguments=False,
            max_nesting_depth=max_nesting_depth,
        )
        return cls._from_validated(raw_messages, registry, max_nesting_depth)

    @classmethod
    def _from_validated(
        cls,
        raw_messages: list[dict[str, Any]],
        tool_call_types: ToolCallTypeRegistry | None,
        max_nesting_depth: int,
    ) -> Messages:
        """Build objects after the scanner's structural verification."""

        cls._validate_nesting_limit(max_nesting_depth)
        registry = cls.validate_tool_call_types(tool_call_types)
        items: list[MessageType] = []
        for message_index, raw_message in enumerate(raw_messages):
            payload = dict(raw_message)
            role = raw_message["role"]
            if role == "assistant":
                calls: list[ToolCall] = []
                for raw_call in raw_message.get("tool_calls", []):
                    function = raw_call["function"]
                    call_class = registry.get(
                        (str(raw_call["type"]), str(function["name"])), ToolCall
                    )
                    calls.append(call_class.model_validate(raw_call, strict=True))
                if "tool_calls" in raw_message:
                    payload["tool_calls"] = tuple(calls)
                message_class: type[MessageType] = AssistantMessage
            else:
                message_class = {
                    "system": SystemMessage,
                    "developer": DeveloperMessage,
                    "user": UserMessage,
                    "tool": ToolMessage,
                }[role]
            message = message_class.model_validate(payload, strict=True)
            object.__setattr__(message, "_message_index", message_index)
            items.append(message)

        root = cls()
        object.__setattr__(root, "_items", tuple(items))
        object.__setattr__(root, "_ordinals", tuple(range(len(items))))
        object.__setattr__(root, "_max_nesting_depth", max_nesting_depth)
        object.__setattr__(root, "_tool_call_types", MappingProxyType(registry))
        cls._build_indices(root, max_nesting_depth)
        return root

    @classmethod
    def _from_items(
        cls,
        items: tuple[MessageType, ...],
        tool_call_types: ToolCallTypeRegistry | None,
        max_nesting_depth: int,
    ) -> Messages:
        """Build a root from EXISTING, already-validated ``Message`` objects, reusing them (and
        their cached ``ContentPart``s, ``ToolCall``s, and ``ToolMessage``s) BY IDENTITY.

        Only the aggregate ordinal/link indices are rebuilt — no message is re-validated or
        re-serialized. Each item's ``message_index`` must already be its position (0..n-1). This
        backs identity-preserving derivation such as :meth:`add_message`, where the existing
        turns must remain the same Python objects."""

        cls._validate_nesting_limit(max_nesting_depth)
        registry = cls.validate_tool_call_types(tool_call_types)
        root = cls()
        object.__setattr__(root, "_items", tuple(items))
        object.__setattr__(root, "_ordinals", tuple(range(len(items))))
        object.__setattr__(root, "_max_nesting_depth", max_nesting_depth)
        object.__setattr__(root, "_tool_call_types", MappingProxyType(registry))
        cls._build_indices(root, max_nesting_depth)
        return root

    @staticmethod
    def _build_indices(root: Messages, max_nesting_depth: int) -> None:
        """Build the aggregate ordinal/call/response indices and wire the call↔response links.

        Idempotent over the ``root._items`` objects: it (re)sets each call's message index and
        the bidirectional call/response links to the same values, so it is safe both for freshly
        validated messages and for reused ones (:meth:`_from_items`)."""

        call_by_id: dict[str, ToolCall] = {}
        response_by_id: dict[str, ToolMessage] = {}
        for message in root._items:
            if isinstance(message, AssistantMessage):
                for call in message.tool_calls:
                    object.__setattr__(call, "_message_index", message.message_index)
                    object.__setattr__(call, "_max_nesting_depth", max_nesting_depth)
                    call_by_id[call.id] = call
            elif isinstance(message, ToolMessage):
                response_by_id[message.tool_call_id] = message

        for call_id, response in response_by_id.items():
            call = call_by_id[call_id]
            object.__setattr__(call, "_response", response)
            object.__setattr__(response, "_call", call)

        ordinals_by_role: dict[str, list[int]] = {}
        calls_by_name: dict[str, list[ToolCall]] = {}
        responses_by_name: dict[str, list[ToolMessage]] = {}
        all_calls: list[ToolCall] = []
        for message in root._items:
            ordinals_by_role.setdefault(message.role, []).append(message.message_index)
            if isinstance(message, AssistantMessage):
                for call in message.tool_calls:
                    all_calls.append(call)
                    calls_by_name.setdefault(call.name, []).append(call)
            elif isinstance(message, ToolMessage):
                responses_by_name.setdefault(message.name, []).append(message)
        object.__setattr__(root, "_all_calls", tuple(all_calls))
        object.__setattr__(
            root,
            "_ordinals_by_role",
            MappingProxyType(
                {role: tuple(ordinals) for role, ordinals in ordinals_by_role.items()}
            ),
        )
        object.__setattr__(
            root,
            "_calls_by_name",
            MappingProxyType({name: tuple(calls) for name, calls in calls_by_name.items()}),
        )
        object.__setattr__(
            root,
            "_responses_by_name",
            MappingProxyType(
                {name: tuple(responses) for name, responses in responses_by_name.items()}
            ),
        )

    @staticmethod
    def _validate_nesting_limit(max_nesting_depth: int) -> None:
        if (
            isinstance(max_nesting_depth, bool)
            or not isinstance(max_nesting_depth, int)
            or not 1 <= max_nesting_depth <= 100
        ):
            raise ValueError("max_nesting_depth must be an integer from 1 to 100")

    @staticmethod
    def validate_tool_call_types(
        tool_call_types: ToolCallTypeRegistry | None,
    ) -> dict[tuple[str, str], type[ToolCall]]:
        """Validate and snapshot the optional tool-call subclass registry."""

        if tool_call_types is None:
            return {}
        if not isinstance(tool_call_types, Mapping):
            raise TypeError("tool_call_types must be a mapping")
        registry: dict[tuple[str, str], type[ToolCall]] = {}
        for key, value in tool_call_types.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not all(isinstance(part, str) and part for part in key)
            ):
                raise TypeError(
                    "tool_call_types keys must be non-empty (type, function_name) tuples"
                )
            if not isinstance(value, type) or not issubclass(value, ToolCall):
                raise TypeError("tool_call_types values must be ToolCall subclasses")
            if set(value.model_fields) != set(ToolCall.model_fields):
                raise TypeError("ToolCall subclasses must not add or remove OpenAI wire fields")
            redeclared_fields = (
                set().union(
                    *(
                        set(extension.__dict__.get("__annotations__", ()))
                        for extension in value.__mro__
                        if extension is not ToolCall and issubclass(extension, ToolCall)
                    )
                )
                & ToolCall.model_fields.keys()
            )
            if redeclared_fields:
                names = ", ".join(sorted(redeclared_fields))
                raise TypeError(
                    f"ToolCall subclasses must not redeclare OpenAI wire fields: {names}"
                )
            if value.model_config != ToolCall.model_config:
                raise TypeError("ToolCall subclasses must not change Pydantic model configuration")
            decorators = value.__pydantic_decorators__
            if any(
                getattr(decorators, category)
                for category in (
                    "validators",
                    "field_validators",
                    "root_validators",
                    "field_serializers",
                    "model_serializers",
                    "model_validators",
                    "computed_fields",
                )
            ):
                raise TypeError("ToolCall subclasses must not declare Pydantic validation hooks")
            if any(
                (hook := extension.__dict__.get("model_post_init")) is not None
                and getattr(hook, "__module__", None) != "pydantic._internal._model_construction"
                for extension in value.__mro__
                if extension is not ToolCall and issubclass(extension, ToolCall)
            ):
                raise TypeError("ToolCall subclasses must not declare Pydantic validation hooks")
            if not isinstance(value.dsl_fields, frozenset) or not all(
                isinstance(field, str) and field for field in value.dsl_fields
            ):
                raise TypeError(
                    "ToolCall subclasses must declare dsl_fields as a frozenset of names"
                )
            if value.dsl_fields & ToolCall.DSL_FIELDS:
                raise TypeError("ToolCall subclass DSL fields must not overlap canonical fields")
            canonical_overrides = (
                ToolCall.DSL_FIELDS
                | frozenset(ToolCall.model_fields)
                | {
                    "DSL_FIELDS",
                    "model_dump",
                    "model_validate",
                    "to_openai_dict",
                    "to_json",
                    "__eq__",
                    "__getattribute__",
                    "__hash__",
                    "__init__",
                    "__delattr__",
                    "__setattr__",
                }
            )
            missing = object()
            canonical_overrides = {
                name
                for name in canonical_overrides
                if getattr_static(value, name, missing)
                is not getattr_static(ToolCall, name, missing)
            }
            if canonical_overrides:
                names = ", ".join(sorted(canonical_overrides))
                raise TypeError(f"ToolCall subclasses must not override canonical members: {names}")
            if any(
                not isinstance(getattr_static(value, field, None), property)
                for field in value.dsl_fields
            ):
                raise TypeError(
                    "Every declared ToolCall DSL field must be a property on the subclass"
                )
            registry[key] = value
        return registry

    @property
    def max_nesting_depth(self) -> int:
        """Return the immutable decoded-argument depth bound."""

        return self._root._max_nesting_depth

    def __len__(self) -> int:
        return len(self._ordinals)

    def __iter__(self) -> Iterator[MessageType]:
        return (self._message_at(index) for index in self._ordinals)

    def _message_at(self, ordinal: int) -> MessageType:
        segment = self._segments.get(ordinal)
        return self._root._items[ordinal] if segment is None else segment

    @overload
    def __getitem__(self, index: int) -> MessageType: ...

    @overload
    def __getitem__(self, index: slice) -> Messages: ...

    def __getitem__(self, index: int | slice) -> MessageType | Messages:
        if isinstance(index, slice):
            return self._select_ordinals(self._ordinals[index])
        return self._message_at(self._ordinals[index])

    def __bool__(self) -> bool:
        return bool(self._ordinals)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Messages):
            return tuple(self) == tuple(other)
        if isinstance(other, Sequence) and not isinstance(other, str | bytes | bytearray):
            return list(self) == list(other)
        return NotImplemented

    def _select_ordinals(self, ordinals: Iterable[int]) -> Messages:
        """Return an identity-preserving view over validated root ordinals."""

        selected = tuple(ordinals)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in selected):
            raise TypeError("message ordinals must be integers")
        if any(left >= right for left, right in zip(selected, selected[1:], strict=False)):
            raise ValueError("message ordinals must be strictly increasing")
        if any(value < 0 or value >= len(self._root._items) for value in selected):
            raise IndexError("message ordinal is outside the root collection")

        view = object.__new__(Messages)
        view._root = self._root
        view._items = self._root._items
        view._ordinals = selected
        view._max_nesting_depth = self._root._max_nesting_depth
        view._all_calls = self._root._all_calls
        view._ordinals_by_role = self._root._ordinals_by_role
        view._calls_by_name = self._root._calls_by_name
        view._responses_by_name = self._root._responses_by_name
        view._tool_call_types = self._root._tool_call_types
        view._segments = MappingProxyType(
            {ordinal: self._segments[ordinal] for ordinal in selected if ordinal in self._segments}
        )
        view._chunk = False
        view._sealed = True
        return view

    def _select_ranges(
        self, ordinals: Iterable[int], ranges: Mapping[int, tuple[int, int]]
    ) -> Messages:
        """Return a view of ``ordinals`` holding only ``[start, end)`` of some messages' text.

        ``ranges`` maps a selected ordinal to a range of its whole text. The view stays a
        view of this conversation: every message keeps its root ``message_index``, and
        ``span`` reports the ranges. The view is a ``split`` chunk, a score input that may
        hold a tool response without its call.
        """

        view = self._select_ordinals(ordinals)
        segments = dict(view._segments)
        for ordinal, (start, end) in ranges.items():
            if not view._contains_root_ordinal(ordinal):
                raise ValueError("a divided message must be selected by the view")
            segments[ordinal] = self._root._items[ordinal]._segment_of(start, end)
        object.__setattr__(view, "_segments", MappingProxyType(segments))
        object.__setattr__(view, "_chunk", True)
        return view

    @property
    def span(self) -> Span:
        """Return where this view sits in its root conversation, first message to last.

        Messages inside the span that the view does not select, such as the assistant
        turns of ``user_messages``, are not part of it.
        """

        if not self._ordinals:
            raise ValueError("an empty message selection has no span")
        first, last = self._ordinals[0], self._ordinals[-1]
        first_piece = self._segments.get(first)
        last_piece = self._segments.get(last)
        start = 0 if first_piece is None else first_piece._range[0]
        end = len(self._root._items[last].content) if last_piece is None else last_piece._range[1]
        return Span(first, start, end, end_message_index=last)

    def _contains_root_ordinal(self, ordinal: int) -> bool:
        """Return whether this view selects one root-message ordinal."""

        position = bisect_left(self._ordinals, ordinal)
        return position < len(self._ordinals) and self._ordinals[position] == ordinal

    def _is_root_view(self) -> bool:
        return self is self._root

    def shares_root_with(self, other: Messages) -> bool:
        """Return whether both values are views of the same canonical conversation."""

        return isinstance(other, Messages) and self._root is other._root

    @dsl_function(
        return_type="string",
        allowed_uses=(Use.COMPREHENSION_BODY,),
        implicit_receiver_count=1,
        syntax="messages.extract(span[, left[, right]])",
    )
    def extract(
        self,
        span: Annotated[Span, DSLParameter(accepted_types=("span",))],
        left: Annotated[
            int,
            DSLParameter(accepted_types=("integer",), required=False, minimum=0),
        ] = 0,
        right: Annotated[
            int,
            DSLParameter(accepted_types=("integer",), required=False, minimum=0),
        ] = 0,
    ) -> str:
        """Return one span plus bounded context from its canonical source message.

        ``left`` and ``right`` default to zero and clamp at that message's boundaries.
        """

        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (left, right)
        ):
            raise ValueError("extract() context bounds must be non-negative integers")
        message = self._span_message(span, require_bound=True)
        content = message.content
        return content[max(0, span.start - left) : min(len(content), span.end + right)]

    def _bind_span(self, span: Span) -> Span:
        """Validate coordinates and bind a producer span to its canonical message."""

        segment = self._segments.get(span.message_index)
        if segment is not None:
            # A span over a segment is relative to the segment's text; report it in the
            # coordinates of the whole message.
            start, end = segment._range
            if span.end > end - start:
                raise ValueError("span is outside message content")
            span = replace(span, start=span.start + start, end=span.end + start)
        message = self._span_message(span, require_bound=False)
        return span._bind_to_message(message)

    def _span_message(self, span: Span, *, require_bound: bool) -> MessageType:
        """Return the selected canonical message after validating one span."""

        if not isinstance(span, Span):
            raise TypeError("extract() requires a Span")
        if span.message_index >= len(self._root._items):
            raise ValueError("span message index is outside the conversation")
        if not span.within_one_message:
            raise ValueError("a span across messages has no single text; use one within a message")
        if not self._contains_root_ordinal(span.message_index):
            raise ValueError("span message is not selected by this Messages value")
        message = self._root._items[span.message_index]
        if require_bound and span._message is None:
            raise ValueError("span is not bound to a source message")
        if require_bound and span._message is not message:
            raise ValueError("span source message is not present in this Messages value")
        if span.end > len(message.content):
            raise ValueError("span is outside message content")
        if span.end <= span.start and message.content:
            raise ValueError("span over message text must not be empty")
        return message

    def _select_indexed_ordinals(self, ordinals: tuple[int, ...]) -> Messages:
        """Intersect one root index with this view without scanning message data."""

        return self._select_ordinals(
            ordinal for ordinal in ordinals if self._contains_root_ordinal(ordinal)
        )

    def _select_positions(self, positions: Iterable[int]) -> Messages:
        """Return a reference-backed sub-view from LOCAL positions of this collection.

        ``positions`` index into this view's own sequence (0..len-1); they are mapped through
        to the underlying root ordinals so the result reuses the SAME canonical ``Message``
        objects — no copy, no re-index. Used by token-bounded chunking to build chunks that
        share the transcript's messages (and, wrapped in a ``Conversation``, its tools) by
        reference. Positions must be strictly increasing (``_select_ordinals`` enforces it)."""

        return self._select_ordinals(self._ordinals[position] for position in positions)

    def assert_self_contained(self) -> None:
        """Reject a selection that is not a valid standalone OpenAI sequence.

        An empty selection has nothing to score; a tool-result message whose originating
        assistant tool call is not also selected is an orphan no provider will accept. Either
        fails here — before any builder/adapter/provider dispatch — rather than being widened
        or scored as a malformed request. O(k) in the selection size (no message-text scan).

        A ``split`` chunk is exempt from the orphan check: it is only ever a score input, and
        the score builders flatten it into evidence text rather than send it as a request."""

        if not self._ordinals:
            raise ValueError("message selection is empty; nothing to score")
        if self._chunk:
            return
        for message in self:
            if isinstance(message, ToolMessage):
                call = message._call
                if call is None or not self._contains_root_ordinal(call.message_index):
                    raise ValueError(
                        "tool-result message selected without its originating assistant tool call"
                    )

    @property
    def system_prompt(self) -> str | None:
        """Return text when the first selected message is a system message."""

        if not self:
            return None
        first = self[0]
        return first.content if isinstance(first, SystemMessage) else None

    @property
    @dsl_surface(
        name="system_messages",
        public_type="collection<message>",
        domain="event",
        allowed_uses=(Use.COMPREHENSION_SOURCE,),
        fields_from=SystemMessage,
        model_input=True,
    )
    def system_messages(self) -> Messages:
        """System-role messages from the selected conversation, in source order."""

        return self._select_indexed_ordinals(self._root._ordinals_by_role.get("system", ()))

    @property
    @dsl_surface(
        name="developer_messages",
        public_type="collection<message>",
        domain="event",
        allowed_uses=(Use.COMPREHENSION_SOURCE,),
        fields_from=DeveloperMessage,
        model_input=True,
    )
    def developer_messages(self) -> Messages:
        """Developer-role messages from the selected conversation, in source order."""

        return self._select_indexed_ordinals(self._root._ordinals_by_role.get("developer", ()))

    @property
    @dsl_surface(
        name="user_messages",
        public_type="collection<message>",
        domain="event",
        allowed_uses=(Use.COMPREHENSION_SOURCE,),
        fields_from=UserMessage,
        model_input=True,
    )
    def user_messages(self) -> Messages:
        """User-role messages from the selected conversation, in source order."""

        return self._select_indexed_ordinals(self._root._ordinals_by_role.get("user", ()))

    @property
    @dsl_surface(
        name="assistant_messages",
        public_type="collection<message>",
        domain="event",
        allowed_uses=(Use.COMPREHENSION_SOURCE,),
        fields_from=AssistantMessage,
        model_input=True,
    )
    def assistant_messages(self) -> Messages:
        """Assistant-role messages from the selected conversation, in source order."""

        return self._select_indexed_ordinals(self._root._ordinals_by_role.get("assistant", ()))

    @property
    @dsl_surface(
        name="tool_responses",
        public_type="collection<tool_response>",
        domain="event",
        allowed_uses=(Use.COMPREHENSION_SOURCE,),
        fields_from=ToolMessage,
        model_input=True,
    )
    def tool_responses(self) -> Messages:
        """Tool-result messages linked to assistant calls by ``tool_call_id``."""

        return self._select_indexed_ordinals(self._root._ordinals_by_role.get("tool", ()))

    @property
    @dsl_surface(
        name="tool_calls",
        public_type="collection<tool_call>",
        domain="event",
        allowed_uses=(Use.COMPREHENSION_SOURCE,),
        fields_from=ToolCall,
    )
    def tool_calls(self) -> list[ToolCall]:
        """Assistant tool calls in chronological order.

        Each item exposes its call ID, exact function name, and decoded arguments.
        """

        if self._is_root_view():
            return list(self._root._all_calls)
        return [
            call
            for call in self._root._all_calls
            if self._contains_root_ordinal(call.message_index)
        ]

    @dsl_function(
        allowed_uses=(Use.CONDITION, Use.FLOW_OPERAND, Use.COMPREHENSION_BODY),
    )
    def called_tool(self, name: Annotated[str, DSLParameter(literal_only=True)]) -> bool:
        """Return whether the assistant called a tool with this exact name.

        This checks calls that occurred. It does not check tools that were merely offered in the
        request.
        """

        candidates = self._root._calls_by_name.get(name, ())
        if self._is_root_view():
            return bool(candidates)
        return any(self._contains_root_ordinal(call.message_index) for call in candidates)

    @dsl_function(
        allowed_uses=(Use.CONDITION, Use.FLOW_OPERAND, Use.COMPREHENSION_BODY),
    )
    def called_tools(
        self,
        *names: Annotated[str, DSLParameter(required=True, literal_only=True)],
    ) -> bool:
        """Return whether the assistant called every named tool.

        At least one exact tool name is required. This test does not impose an order.
        """

        return all(self.called_tool(name) for name in names)

    @dsl_function(
        allowed_uses=(),
    )
    def count_tool_calls(self, name: Annotated[str, DSLParameter(literal_only=True)]) -> int:
        """Count assistant calls to the tool with this exact name."""

        candidates = self._root._calls_by_name.get(name, ())
        if self._is_root_view():
            return len(candidates)
        return sum(self._contains_root_ordinal(call.message_index) for call in candidates)

    @dsl_function(
        return_type="collection<tool_call>",
        allowed_uses=(Use.COMPREHENSION_SOURCE,),
        accessor_surface="tool_calls",
        accessor_filter="name",
    )
    def get_tool_calls(
        self, name: Annotated[str, DSLParameter(literal_only=True)]
    ) -> list[ToolCall]:
        """Return assistant calls to the tool with this exact name.

        Use the result as a comprehension collection or count it.
        """

        candidates = self._root._calls_by_name.get(name, ())
        if self._is_root_view():
            return list(candidates)
        return [call for call in candidates if self._contains_root_ordinal(call.message_index)]

    @dsl_function(
        return_type="collection<tool_response>",
        allowed_uses=(Use.COMPREHENSION_SOURCE,),
        accessor_surface="tool_responses",
        accessor_filter="name",
    )
    def get_tool_responses(
        self, name: Annotated[str, DSLParameter(literal_only=True)]
    ) -> list[ToolMessage]:
        """Return results linked to calls of the tool with this exact name.

        The call/result relationship is resolved through ``tool_call_id``; unrelated tool
        results are excluded.
        """

        candidates = self._root._responses_by_name.get(name, ())
        if self._is_root_view():
            return list(candidates)
        return [
            message for message in candidates if self._contains_root_ordinal(message.message_index)
        ]

    @dsl_function(
        allowed_uses=(Use.CONDITION, Use.FLOW_OPERAND, Use.COMPREHENSION_BODY),
    )
    def user_said(
        self,
        text: Annotated[str, DSLParameter(literal_only=True)],
        case_sensitive: Annotated[bool, DSLParameter(literal_only=True)] = False,
    ) -> bool:
        """Return whether any user message contains the supplied text.

        Matching is case-insensitive by default. Pass ``true`` as the second argument to require
        the same letter case.
        """

        return any(message.contains(text, case_sensitive) for message in self.user_messages)

    @dsl_function(
        allowed_uses=(Use.CONDITION, Use.FLOW_OPERAND, Use.COMPREHENSION_BODY),
    )
    def assistant_said(
        self,
        text: Annotated[str, DSLParameter(literal_only=True)],
        case_sensitive: Annotated[bool, DSLParameter(literal_only=True)] = False,
    ) -> bool:
        """Return whether any assistant message contains the supplied text.

        Matching is case-insensitive by default. Pass ``true`` as the second argument to require
        the same letter case.
        """

        return any(message.contains(text, case_sensitive) for message in self.assistant_messages)

    @dsl_function(
        return_type="collection<string>",
        allowed_uses=(Use.COMPREHENSION_SOURCE,),
    )
    def call_sequence(self) -> list[str]:
        """Return assistant tool-call names in chronological order."""

        return [call.name for call in self.tool_calls]

    @dsl_function(
        allowed_uses=(Use.CONDITION, Use.COMPREHENSION_BODY),
    )
    def has_sequence(
        self,
        *names: Annotated[str, DSLParameter(required=True, literal_only=True)],
    ) -> bool:
        """Return whether these names occur contiguously in the tool-call order.

        Other messages and tool results may appear between the calls, but another tool call
        between two requested names breaks the sequence.
        """

        sequence = self.call_sequence()
        expected = list(names)
        return any(
            sequence[index : index + len(expected)] == expected
            for index in range(len(sequence) - len(expected) + 1)
        )

    @dsl_function(
        return_type="model_input",
        allowed_uses=(),
        syntax='.add_message("role", content)',
        implicit_receiver_count=1,
    )
    def add_message(
        self,
        role: Annotated[
            str,
            DSLParameter(
                literal_only=True,
                allowed_values=("system", "developer", "user", "assistant"),
            ),
        ],
        content: str,
    ) -> Messages:
        """Append one text message to a model input created with ``message(...)``.

        Use this only after a model-input value, for example
        ``message("user", "hello").add_message("assistant", "hi")``.

        Internal:

        Reference-preserving: only the NEW ``Message`` (and its lightweight index/link metadata)
        is created. Every existing message — and its ``ContentPart``s, ``ToolCall``s, and
        ``ToolMessage``s — is reused BY IDENTITY, so ``old[i] is new[i]`` for every prior turn.
        The prior implementation round-tripped through ``to_openai_messages()``/``from_openai()``,
        which re-validated and recreated every object; this does not."""

        if self._root is not self or self._ordinals != tuple(range(len(self._items))):
            raise ValueError("add_message() requires a root Messages collection")
        if not isinstance(role, str) or role not in {"system", "developer", "user", "assistant"}:
            raise ValueError("add_message() role must be system, developer, user, or assistant")
        if not isinstance(content, str):
            raise TypeError("add_message() content must be a string")
        message_class: type[MessageType] = {
            "system": SystemMessage,
            "developer": DeveloperMessage,
            "user": UserMessage,
            "assistant": AssistantMessage,
        }[role]
        new_message = message_class.model_validate({"role": role, "content": content}, strict=True)
        object.__setattr__(new_message, "_message_index", len(self._items))
        return self._from_items(
            (*self._items, new_message),
            self._root._tool_call_types,
            self._root._max_nesting_depth,
        )

    def to_openai_messages(self) -> list[dict[str, Any]]:
        """Return selected messages as fresh OpenAI-compatible dictionaries."""

        return [message.to_openai_dict() for message in self]

    @property
    def digest(self) -> bytes:
        """The digest of the selected messages, from their own digests, each computed once."""

        return messages_digest(message.digest for message in self)

    def to_json(self) -> str:
        """Return compact JSON for the selected OpenAI message array."""

        return json.dumps(
            self.to_openai_messages(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
