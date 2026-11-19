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

import asyncio
import json
import math
import re
import sys
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from inspect import getattr_static
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Literal, Self, cast, overload

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from quarry.language.metadata import (
    DSLParameter,
    SurfaceFieldExclusion,
    SurfaceFieldMetadata,
    Use,
    dsl_function,
    dsl_surface,
)
from quarry.types.exceptions import ConversationValidationError, EvaluationError, InvalidInputError
from quarry.types.results import Span
from quarry.types.transforms import (
    ContentSourcePart,
    ContentSourceRange,
    ProvenanceSegment,
    slice_provenance,
    source_mapping_for,
    source_part_mapping_for,
    source_ranges_for,
)


def wire_digest(wire: Mapping[str, Any]) -> bytes:
    """The digest of one message's wire form, as ``Message.digest`` computes it."""

    from quarry.cache.keys import input_key

    return input_key("message", wire)


def messages_digest(digests: Iterable[bytes]) -> bytes:
    """The digest of messages in order, from their own digests, as ``Messages.digest``
    computes it."""

    from quarry.cache.keys import combine_digests

    return combine_digests("messages", digests)


def _checked_rewrite_json(value: object, maximum_depth: int) -> None:
    """Bound adapter-owned JSON before equality, copying or message construction."""

    stack = [(value, 0, False)]
    active: set[int] = set()
    nodes = characters = 0
    while stack:
        item, depth, leaving = stack.pop()
        if leaving:
            active.remove(id(item))
            continue
        nodes += 1
        if nodes % 256 == 0:
            _raise_if_combine_expired()
        if nodes > 100_000 or depth > maximum_depth:
            raise ValueError("message rewrite exceeds JSON graph limits")
        if type(item) is str:
            characters += len(item)
            if characters > 2_000_000:
                raise ValueError("message rewrite exceeds its text limit")
        elif type(item) is int:
            if item.bit_length() > 64:
                raise ValueError("message rewrite integer exceeds JSON wire bounds")
        elif item is None or type(item) is bool:
            continue
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("message rewrite numbers must be finite")
        elif type(item) in {dict, list}:
            if id(item) in active:
                raise ValueError("message rewrite contains a JSON cycle")
            container = cast(dict | list, item)
            if len(container) + len(stack) + nodes > 100_000:
                raise ValueError("message rewrite exceeds JSON graph limits")
            active.add(id(item))
            stack.append((item, depth, True))
            if type(item) is dict:
                if 2 * len(container) + len(stack) + nodes > 100_000:
                    raise ValueError("message rewrite exceeds JSON graph limits")
                for key, child in cast(dict, item).items():
                    if type(key) is not str:
                        raise ValueError("rewritten message keys must be strings")
                    stack.append((key, depth + 1, False))
                    stack.append((child, depth + 1, False))
            else:
                stack.extend((child, depth + 1, False) for child in cast(list, item))
        else:
            raise ValueError("rewritten messages must contain only JSON values")


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
    visits = 0
    while stack:
        visits += 1
        if visits % 256 == 0:
            _raise_if_combine_expired()
        current, depth = stack.pop()
        if isinstance(current, Mapping):
            if depth > max_depth:
                raise EvaluationError(f"Tool arguments exceed max_nesting_depth {max_depth}")
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list | tuple):
            if depth > max_depth:
                raise EvaluationError(f"Tool arguments exceed max_nesting_depth {max_depth}")
            stack.extend((child, depth + 1) for child in current)


def load_tool_arguments(raw: str, max_depth: int, owner: str) -> Any:
    """Decode JSON with finite numbers, exact integers and a bounded container depth.

    JSON syntax errors remain JSONDecodeError so callers can treat freeform argument
    text as having no named fields. Resource/numeric failures are rule-local input errors.
    """

    def invalid(reason: str) -> InvalidInputError:
        return InvalidInputError(f"{owner}: {reason}", {"field": owner})

    def number(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            raise invalid("arguments contain a non-finite number")
        return value

    def constant(_text: str) -> None:
        raise invalid("arguments contain a non-finite number")

    _raise_if_combine_expired()
    try:
        value = json.loads(
            raw, parse_int=_parse_tool_integer, parse_float=number, parse_constant=constant
        )
    except _ToolIntegerLimitError as exc:
        raise invalid(f"arguments exceed the JSON integer digit limit: {exc}") from None
    except RecursionError:
        raise invalid(f"arguments exceed max_nesting_depth {max_depth}") from None
    _raise_if_combine_expired()
    try:
        _validate_depth(value, max_depth)
    except EvaluationError as exc:
        raise invalid(str(exc)) from None
    _raise_if_combine_expired()
    return value


def decode_tool_arguments(raw: str, max_depth: int, owner: str) -> _FrozenDict:
    """Return named JSON arguments; freeform text and other JSON shapes name none."""

    try:
        value = load_tool_arguments(raw, max_depth, owner)
    except json.JSONDecodeError:
        _raise_if_combine_expired()
        value = {}
    return _freeze_json(value if isinstance(value, dict) else {})


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
    _provenance: tuple[ProvenanceSegment, ...] | None = PrivateAttr(default=None)
    _synthetic_identity: object | None = PrivateAttr(default=None)
    _rewrite_source: Message | None = PrivateAttr(default=None)
    _rewrite_content_changed: bool = PrivateAttr(default=False)

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

        if not self._can_split_text():
            raise ValueError("only a message with text-only content can be divided")
        text = self.content
        if not 0 <= start < end <= len(text):
            raise ValueError("message segment is outside the message content")
        return self._copy_text_range(
            start, end, text, slice_provenance(self._content_provenance(), start, end)
        )

    def _copy_text_range(
        self, start: int, end: int, text: str, provenance: tuple[ProvenanceSegment, ...]
    ) -> Self:
        """Copy a range whose text and sliced map were checked by its range source."""

        update: dict[str, Any] = {"source_content": text[start:end]}
        if end < len(text) and isinstance(self, AssistantMessage):
            if self.tool_calls:
                update["tool_calls"] = ()
            if self.refusal is not None:
                update["refusal"] = None
        segment = self.model_copy(update=update)
        object.__setattr__(segment, "_content_cache", None)
        object.__setattr__(segment, "_content_parts_cache", None)
        object.__setattr__(segment, "_digest_cache", None)
        object.__setattr__(segment, "_segment", (start, end))
        object.__setattr__(segment, "_provenance", provenance)
        if not isinstance(self.source_content, str):
            object.__setattr__(segment, "_rewrite_content_changed", True)
        return segment

    def _can_split_text(self) -> bool:
        source = self.source_content
        return (
            isinstance(source, str)
            or isinstance(source, tuple)
            and bool(source)
            and all(isinstance(part, Mapping) and part.get("type") == "text" for part in source)
        )

    def _content_provenance(self) -> tuple[ProvenanceSegment, ...]:
        """Return this text's map to the root message's original content parts."""

        if self._provenance is not None:
            return self._provenance
        return tuple(segment for segment in self._provenance_steps() if segment is not None)

    def _provenance_steps(self) -> Iterator[ProvenanceSegment | None]:
        """Visit raw parts or an existing map without an uninterruptible projection."""

        if self._provenance is not None:
            yield from self._provenance
            return
        source = self.source_content
        if isinstance(source, str):
            if not source:
                return
            base = self._segment[0] if self._segment is not None else 0
            original = ContentSourceRange(
                self.message_index,
                None,
                base,
                base + len(source),
                base,
                base + len(source),
            )
            yield ProvenanceSegment(0, len(source), (original,))
            return
        if not isinstance(source, tuple):
            return
        output = 0
        for part_index, part in enumerate(source):
            if part_index % 256 == 0:
                _raise_if_combine_expired()
            if not isinstance(part, Mapping) or part.get("type") != "text":
                yield None
                continue
            text = part.get("text", "")
            if not isinstance(text, str) or not text:
                yield None
                continue
            original = ContentSourceRange(
                self.message_index,
                part_index,
                0,
                len(text),
                output,
                output + len(text),
            )
            yield ProvenanceSegment(output, output + len(text), (original,))
            output += len(text)

    def _source_ranges(self, start: int, end: int) -> tuple[ContentSourceRange, ...]:
        """Map one non-empty range of this text to its original content parts."""

        return source_ranges_for(
            self._content_provenance(),
            start,
            end,
            require_mappable=True,
            check_deadline=_raise_if_combine_expired,
        )

    def _with_transformed_content(
        self,
        replacements: Mapping[int | None, str],
        provenance: tuple[ProvenanceSegment, ...],
        *,
        replace_media: bool = False,
    ) -> Self:
        """Return an immutable content variant retaining role, index, and tool links."""

        source = self.source_content
        if isinstance(source, str):
            updated_source = replacements.get(None, source)
        elif isinstance(source, tuple):
            parts = []
            for part_index, part in enumerate(source):
                if (
                    isinstance(part, Mapping)
                    and (part.get("type") == "text" or replace_media)
                    and part_index in replacements
                ):
                    updated = (
                        {**dict(part), "text": replacements[part_index]}
                        if part.get("type") == "text"
                        else {"type": "text", "text": replacements[part_index]}
                    )
                    parts.append(updated)
                else:
                    parts.append(part)
            updated_source = _freeze_json(parts)
        else:
            updated_source = source
        transformed = self.model_copy(update={"source_content": updated_source})
        object.__setattr__(transformed, "_content_cache", None)
        object.__setattr__(transformed, "_content_parts_cache", None)
        object.__setattr__(transformed, "_digest_cache", None)
        object.__setattr__(transformed, "_message_index", self.message_index)
        object.__setattr__(transformed, "_segment", None)
        object.__setattr__(transformed, "_provenance", provenance)
        return transformed

    def _readable_for_rewrite(self) -> bool:
        """Whether a whole-message adapter may inspect this message's wire value."""

        return not isinstance(self, AssistantMessage) or all(
            call._readable_for_rewrite() for call in self.tool_calls
        )

    def _require_model_readable(self) -> None:
        """Check call values without interpreting their opaque argument strings."""

        if isinstance(self, AssistantMessage):
            for index, call in enumerate(self.tool_calls):
                if index % 256 == 0:
                    _raise_if_combine_expired()
                if isinstance(call, UnreadableToolCall):
                    raise InvalidInputError(call.fault)

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
        SurfaceFieldMetadata("type", "string", "Raw call type; unreadable calls fail field reads."),
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
    _arguments_error: str | None = PrivateAttr(default=None)
    _call_index: int = PrivateAttr(default=-1)

    def guard_read(self, field: str) -> None:
        """Hook for rule field access; structurally exact regular calls are readable."""

    def _readable_for_rewrite(self) -> bool:
        """Whether a containing message may be handed to a whole-message adapter."""

        return True

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
        Freeform text and non-object JSON have no named arguments. Numeric and depth
        limit failures affect the rule reading this property.
        """

        owner = f"messages[{self.message_index}].tool_calls[{self._call_index}].function.arguments"
        if self._arguments_error is not None:
            raise InvalidInputError(self._arguments_error, {"field": owner})
        if self._arguments_cache is None:
            try:
                frozen = decode_tool_arguments(self.raw_arguments, self._max_nesting_depth, owner)
            except InvalidInputError as exc:
                object.__setattr__(self, "_arguments_error", exc.message)
                raise
            object.__setattr__(self, "_arguments_cache", frozen)
        return _project_json(self._arguments_cache)

    def _discard_decoded_arguments(self) -> None:
        """Drop a tree decoded only for input preflight; rule reads stay lazy."""

        object.__setattr__(self, "_arguments_cache", None)
        object.__setattr__(self, "_arguments_error", None)

    @property
    def message_index(self) -> int:
        """Return the owning assistant message's root ordinal."""

        return self._message_index

    @property
    def response(self) -> ToolMessage | None:
        """Return the canonical tool-result message linked to this call."""

        return self._response


def call_fault(raw_call: Mapping[str, Any]) -> str | None:
    """Classify an otherwise structurally valid call without interpreting its arguments."""

    if raw_call.get("type") != "function":
        return "unsupported tool-call type"
    function = raw_call.get("function")
    if isinstance(function, Mapping) and not function.get("name"):
        return "tool call has no readable function name"
    return None


class UnreadableToolCall(ToolCall):
    """A preserved call whose rule fields are unreadable, except its transcript position."""

    type: str  # ty: ignore[invalid-attribute-override]
    _fault: str = PrivateAttr(default="unreadable tool call")

    @property
    def fault(self) -> str:
        return f"messages[{self.message_index}].tool_calls[{self._call_index}]: {self._fault}"

    def guard_read(self, field: str) -> None:
        if field != "message_index":
            raise InvalidInputError(self.fault)

    def _readable_for_rewrite(self) -> bool:
        return False

    @property
    def name(self) -> str:
        raise InvalidInputError(self.fault)

    @property
    def raw_arguments(self) -> str:
        raise InvalidInputError(self.fault)

    @property
    def arguments(self) -> dict[str, Any]:
        raise InvalidInputError(self.fault)

    @property
    def response(self) -> ToolMessage | None:
        raise InvalidInputError(self.fault)

    def texts(self) -> tuple[tuple[str, str], ...]:
        raise InvalidInputError(self.fault)

    def argument_texts(self, field: str) -> list[tuple[str, str]]:
        raise InvalidInputError(self.fault)

    def argument_text(self, field: str) -> str | None:
        raise InvalidInputError(self.fault)

    def place_texts(self) -> tuple[tuple[str, str], ...]:
        """Return opaque position text for evidence, without a semantic argument read."""

        return (("raw_arguments", self.function.arguments),)


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
    source_content: str | tuple[Any, ...] = Field(alias="content")
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
        """Compatibility alias for flattened tool-result text."""

        return self.content


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
    """One structurally valid offered tool, preserved even when rules cannot read it."""

    type: str
    function: ToolFunction
    DSL_FIELDS: ClassVar[frozenset[str]] = frozenset({"type", "name", "description", "parameters"})
    DSL_FIELD_METADATA: ClassVar[tuple[SurfaceFieldMetadata, ...]] = (
        SurfaceFieldMetadata(
            "type", "string", "Offered tool type; rules read named function tools."
        ),
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


def tools_fault(tools: Sequence[Tool]) -> str | None:
    """Return a bounded reason why the offered collection has no readable definitions."""

    names: set[str] = set()
    for index, tool in enumerate(tools):
        if index % 256 == 0:
            _raise_if_combine_expired()
        if tool.type != "function":
            return f"tools[{index}] has an unsupported tool type"
        if not tool.name:
            return f"tools[{index}].function has no readable name"
        if tool.name in names:
            return f"tools[{index}].function repeats an offered name"
        names.add(tool.name)
    return None


MessageType = SystemMessage | DeveloperMessage | UserMessage | AssistantMessage | ToolMessage
ToolCallTypeRegistry = Mapping[tuple[str, str], type[ToolCall]]
_MAX_COMBINED_MESSAGE_PARTS = 256
_MAX_COMBINED_MESSAGES = 10_000
_MAX_COMBINED_TOOL_IDENTITIES = 65_536
_MAX_COMBINED_WORK = 1_000_000
_MAX_SYNTHETIC_MESSAGES = 256
_MAX_SYNTHETIC_TEXT_CHARACTERS = 1_000_000
_MAX_SYNTHETIC_TEXT_BYTES = 4_000_000
_COMBINE_CHECK_INTERVAL = 256
_COMBINE_CANCELLATION_CHECK_INTERVAL = 16


def _validate_synthetic_text(value: str) -> int:
    """Validate one constructed text value without allocating an encoded copy."""

    if len(value) > _MAX_SYNTHETIC_TEXT_CHARACTERS:
        raise ValueError(
            f"constructed message text exceeds {_MAX_SYNTHETIC_TEXT_CHARACTERS} characters"
        )
    encoded = 0
    for index, character in enumerate(value):
        if index and index % 4_096 == 0:
            _raise_if_combine_expired()
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise ValueError("constructed message text must contain valid Unicode")
        encoded += (
            1 if codepoint <= 0x7F else 2 if codepoint <= 0x7FF else 3 if codepoint <= 0xFFFF else 4
        )
        if encoded > _MAX_SYNTHETIC_TEXT_BYTES:
            raise ValueError(
                f"constructed message text exceeds {_MAX_SYNTHETIC_TEXT_BYTES} UTF-8 bytes"
            )
    return encoded


@dataclass(slots=True)
class _CombineMetrics:
    """Deterministic counters for bounded message-combination regressions."""

    selection_visits: int = 0
    canonical_message_visits: int = 0
    canonical_call_visits: int = 0
    canonical_response_visits: int = 0
    canonical_link_checks: int = 0
    selected_message_visits: int = 0
    selected_call_visits: int = 0
    selected_response_visits: int = 0
    canonical_lookups: int = 0
    variant_projection_visits: int = 0
    variant_field_visits: int = 0
    variant_character_visits: int = 0
    variant_cache_hits: int = 0
    provenance_visits: int = 0
    segment_visits: int = 0
    call_clone_visits: int = 0
    response_clone_visits: int = 0
    relink_visits: int = 0
    assembly_visits: int = 0
    message_clone_visits: int = 0
    checkpoints: int = 0
    work_units: int = 0


class _CombineMeter:
    """Saturate combination work and request checkpoints before allocations."""

    __slots__ = ("_metrics", "_pending", "_work")

    def __init__(self, metrics: _CombineMetrics) -> None:
        self._metrics = metrics
        self._pending = 0
        self._work = 0

    def record(self, field: str, amount: int = 1) -> bool:
        charged = min(amount, max(0, _MAX_COMBINED_WORK + 1 - self._work))
        setattr(self._metrics, field, getattr(self._metrics, field) + charged)
        self._metrics.work_units += charged
        self._work += charged
        self._pending += charged
        if self._work > _MAX_COMBINED_WORK:
            raise ValueError(
                f"combined message selection exceeds {_MAX_COMBINED_WORK} relinking work units"
            )
        return self._pending >= _COMBINE_CHECK_INTERVAL

    def checkpoint(self, *, finish: bool = False) -> bool:
        if self._pending >= _COMBINE_CHECK_INTERVAL:
            self._pending -= _COMBINE_CHECK_INTERVAL
        elif finish and self._pending:
            self._pending = 0
        else:
            return False
        self._metrics.checkpoints += 1
        return True


@dataclass(slots=True)
class _CanonicalToolUnit:
    """One unambiguous canonical call and its optional canonical response."""

    call_ordinal: int
    call: ToolCall
    response_ordinal: int | None = None
    response: ToolMessage | None = None


def _require_tool_identity(value: object, owner: str) -> str:
    """Return one stable non-empty tool ID or reject a corrupted variant."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must have a non-empty string tool identity")
    return value


def _same_tool_call_identity(left: ToolCall, right: ToolCall) -> bool:
    """Compare stable tool-call wire identity without allocating a wire projection."""

    return (
        type(left) is type(right)
        and left.id == right.id
        and left.type == right.type
        and left.function.name == right.function.name
        and left.function.arguments == right.function.arguments
    )


def _synthetic_key(message: Message) -> tuple[str, object] | None:
    """Return one private identity key without exposing it through wire content."""

    identity = message._synthetic_identity
    if identity is None:
        return None
    if isinstance(identity, tuple) and identity and all(type(part) is int for part in identity):
        return "expression", identity
    return "instance", id(identity)


def _charge_combine_steps(
    meter: _CombineMeter,
    field: str,
    amount: int = 1,
) -> Generator[None, None, None]:
    """Charge work and yield every pending checkpoint before the caller allocates."""

    meter.record(field, amount)
    while meter.checkpoint():
        yield


def _variant_value_steps(
    value: object,
    meter: _CombineMeter,
    active: set[int],
) -> Generator[None, None, object]:
    """Build one exact immutable value representation with fully charged traversal."""

    yield from _charge_combine_steps(meter, "variant_projection_visits")
    if value is None:
        return (0, None)
    if type(value) is bool:
        return (1, value)
    if type(value) is int:
        return (2, value)
    if type(value) is float:
        return (3, value)
    if isinstance(value, str):
        yield from _charge_combine_steps(
            meter,
            "variant_character_visits",
            len(value),
        )
        return (4, value)
    if isinstance(value, type):
        return (5, value)

    identity = id(value)
    if identity in active:
        raise ValueError("combined content variant contains a structural cycle")
    active.add(identity)
    try:
        if isinstance(value, tuple | list):
            children = []
            for child in value:
                yield from _charge_combine_steps(meter, "variant_projection_visits")
                children.append((yield from _variant_value_steps(child, meter, active)))
            return (6, tuple(children))
        if isinstance(value, Mapping):
            fields = []
            for key, child in value.items():
                yield from _charge_combine_steps(meter, "variant_projection_visits")
                key_representation = yield from _variant_value_steps(key, meter, active)
                child_representation = yield from _variant_value_steps(child, meter, active)
                fields.append((key_representation, child_representation))
            return (7, tuple(fields))
        if isinstance(value, ContentSourceRange):
            fields = []
            for attribute in (
                "message_index",
                "part_index",
                "start",
                "end",
                "message_start",
                "message_end",
            ):
                yield from _charge_combine_steps(meter, "variant_projection_visits")
                child = getattr(value, attribute)
                fields.append((yield from _variant_value_steps(child, meter, active)))
            return (8, tuple(fields))
        if isinstance(value, ContentSourcePart):
            yield from _charge_combine_steps(meter, "variant_projection_visits", 2)
            return (11, value.message_index, value.part_index)
        if isinstance(value, ProvenanceSegment):
            fields = []
            for attribute in ("start", "end", "sources", "kind"):
                yield from _charge_combine_steps(meter, "variant_projection_visits")
                child = getattr(value, attribute)
                fields.append((yield from _variant_value_steps(child, meter, active)))
            if value.parts:
                yield from _charge_combine_steps(meter, "variant_projection_visits")
                fields.append((yield from _variant_value_steps(value.parts, meter, active)))
            return (9, tuple(fields))
    finally:
        active.remove(identity)
    raise ValueError(f"combined content variant contains unsupported {type(value).__name__} data")


def _tool_call_variant_steps(
    call: ToolCall,
    meter: _CombineMeter,
    active: set[int],
) -> Generator[None, None, object]:
    """Represent one call's exact stable fields and link identity without recursion."""

    fields = []
    for value in (type(call),):
        yield from _charge_combine_steps(meter, "variant_projection_visits")
        fields.append((yield from _variant_value_steps(value, meter, active)))
    for attribute in ("id", "type", "message_index"):
        yield from _charge_combine_steps(meter, "variant_projection_visits")
        value = getattr(call, attribute)
        fields.append((yield from _variant_value_steps(value, meter, active)))
    yield from _charge_combine_steps(meter, "variant_projection_visits")
    function = call.function
    for attribute in (
        "name",
        "arguments",
    ):
        yield from _charge_combine_steps(meter, "variant_projection_visits")
        value = getattr(function, attribute)
        fields.append((yield from _variant_value_steps(value, meter, active)))
    yield from _charge_combine_steps(meter, "variant_projection_visits")
    response = call._response
    if response is None:
        response_identity = None
    else:
        response_fields = []
        for attribute in ("message_index", "tool_call_id"):
            yield from _charge_combine_steps(meter, "variant_projection_visits")
            response_fields.append(getattr(response, attribute))
        response_identity = tuple(response_fields)
    fields.append((yield from _variant_value_steps(response_identity, meter, active)))
    return (10, tuple(fields))


def _message_variant_steps(
    message: Message,
    meter: _CombineMeter,
    *,
    message_index: int | None = None,
) -> Generator[None, None, object]:
    """Represent a variant, optionally normalizing its synthetic root-local index."""

    active: set[int] = set()
    fields = []
    yield from _charge_combine_steps(meter, "variant_projection_visits")
    fields.append((yield from _variant_value_steps(type(message), meter, active)))
    for attribute in (
        "role",
        "message_index",
        "source_content",
        "_segment",
        "_provenance",
    ):
        yield from _charge_combine_steps(meter, "variant_projection_visits")
        value = (
            message_index
            if attribute == "message_index" and message_index is not None
            else getattr(message, attribute)
        )
        fields.append((yield from _variant_value_steps(value, meter, active)))
    if message._rewrite_content_changed:
        yield from _charge_combine_steps(meter, "variant_projection_visits")
        fields.append(("rewritten_content", True))
    yield from _charge_combine_steps(meter, "variant_projection_visits")
    fields.append(
        (
            yield from _variant_value_steps(
                _synthetic_key(message),
                meter,
                active,
            )
        )
    )
    if isinstance(message, AssistantMessage):
        yield from _charge_combine_steps(meter, "variant_projection_visits")
        fields.append((yield from _variant_value_steps(message.refusal, meter, active)))
        calls = []
        yield from _charge_combine_steps(meter, "variant_projection_visits")
        tool_calls = message.tool_calls
        for call in tool_calls:
            yield from _charge_combine_steps(meter, "variant_projection_visits")
            calls.append((yield from _tool_call_variant_steps(call, meter, active)))
        fields.append((11, tuple(calls)))
    elif isinstance(message, ToolMessage):
        yield from _charge_combine_steps(meter, "variant_projection_visits")
        fields.append((yield from _variant_value_steps(message.tool_call_id, meter, active)))
        yield from _charge_combine_steps(meter, "variant_projection_visits")
        call = message._call
        if call is None:
            call_identity = None
        else:
            call_fields = []
            for attribute in ("message_index", "id"):
                yield from _charge_combine_steps(meter, "variant_projection_visits")
                call_fields.append(getattr(call, attribute))
            call_identity = tuple(call_fields)
        fields.append((yield from _variant_value_steps(call_identity, meter, active)))
    return (12, tuple(fields))


def _variant_representations_equal_steps(
    left: object,
    right: object,
    meter: _CombineMeter,
) -> Generator[None, None, bool]:
    """Compare exact representations while charging every node, edge, and character."""

    stack = [(left, right)]
    while stack:
        yield from _charge_combine_steps(meter, "variant_field_visits")
        left_value, right_value = stack.pop()
        if left_value is right_value:
            continue
        if type(left_value) is not type(right_value):
            return False
        if isinstance(left_value, tuple):
            right_tuple = cast(tuple[Any, ...], right_value)
            if len(left_value) != len(right_tuple):
                return False
            for left_child, right_child in zip(left_value, right_tuple, strict=True):
                yield from _charge_combine_steps(meter, "variant_field_visits")
                stack.append((left_child, right_child))
            continue
        if isinstance(left_value, str):
            right_text = cast(str, right_value)
            yield from _charge_combine_steps(
                meter,
                "variant_character_visits",
                max(len(left_value), len(right_text)),
            )
        if left_value != right_value:
            return False
    return True


def _memoized_message_variant_steps(
    message: Message,
    meter: _CombineMeter,
    cache: dict[int, tuple[Message, object]],
) -> Generator[None, None, object]:
    """Return one identity-memoized exact representation."""

    identity = id(message)
    existing = cache.get(identity)
    if existing is not None and existing[0] is message:
        yield from _charge_combine_steps(meter, "variant_cache_hits")
        return existing[1]
    representation = yield from _message_variant_steps(message, meter)
    cache[identity] = (message, representation)
    return representation


def _raise_if_combine_expired() -> None:
    """Check the task-local deadline without introducing a foundational import cycle."""

    from quarry.safety.timer import raise_if_evaluation_expired

    raise_if_evaluation_expired()


async def _yield_combine_cancellation() -> None:
    """Give cancellation a bounded opportunity to interrupt async relinking."""

    await asyncio.sleep(0)


def _same_message_variant(left: Message, right: Message) -> bool:
    """Return whether two independently built views have identical text provenance."""

    return left is right or (
        left.to_openai_dict() == right.to_openai_dict()
        and left._segment == right._segment
        and left._provenance == right._provenance
        and left._rewrite_content_changed == right._rewrite_content_changed
        and _synthetic_key(left) == _synthetic_key(right)
    )


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
        "_source_root",
        "_items",
        "_ordinals",
        "_ordinal_set",
        "_max_nesting_depth",
        "_all_calls",
        "_unreadable_calls",
        "_unreadable_responses",
        "_ordinals_by_role",
        "_calls_by_name",
        "_responses_by_name",
        "_tool_call_types",
        "_segments",
        "_position_variants",
        "_chunk",
        "_sealed",
    )

    def __init__(self) -> None:
        """Create an empty construction base."""

        self._root = self
        self._source_root: Messages | None = None
        self._items: tuple[MessageType, ...] = ()
        self._ordinals: tuple[int, ...] = ()
        self._ordinal_set: frozenset[int] = frozenset()
        self._max_nesting_depth = 100
        self._all_calls: tuple[ToolCall, ...] = ()
        self._unreadable_calls: tuple[UnreadableToolCall, ...] = ()
        self._unreadable_responses: tuple[ToolMessage, ...] = ()
        self._ordinals_by_role: Mapping[str, tuple[int, ...]] = MappingProxyType({})
        self._calls_by_name: Mapping[str, tuple[ToolCall, ...]] = MappingProxyType({})
        self._responses_by_name: Mapping[str, tuple[ToolMessage, ...]] = MappingProxyType({})
        self._tool_call_types: ToolCallTypeRegistry = MappingProxyType({})
        # Root ordinal -> the segment of that message this view holds instead of the whole.
        self._segments: Mapping[int, MessageType] = MappingProxyType({})
        # Local selection position -> a variant differing from the ordinal's first one.
        self._position_variants: Mapping[int, MessageType] = MappingProxyType({})
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
                    fault = call_fault(raw_call)
                    call_class = (
                        UnreadableToolCall
                        if fault is not None
                        else registry.get((str(raw_call["type"]), str(function["name"])), ToolCall)
                    )
                    call = call_class.model_validate(raw_call, strict=True)
                    if fault is not None:
                        object.__setattr__(call, "_fault", fault)
                    calls.append(call)
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
        object.__setattr__(root, "_ordinal_set", frozenset(root._ordinals))
        object.__setattr__(root, "_max_nesting_depth", max_nesting_depth)
        object.__setattr__(root, "_tool_call_types", MappingProxyType(registry))
        object.__setattr__(root, "_source_root", root)
        cls._build_indices(root, max_nesting_depth)
        return root

    @classmethod
    def _from_items(
        cls,
        items: tuple[MessageType, ...],
        tool_call_types: ToolCallTypeRegistry | None,
        max_nesting_depth: int,
        *,
        source_root: Messages | None,
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
        object.__setattr__(root, "_ordinal_set", frozenset(root._ordinals))
        object.__setattr__(root, "_max_nesting_depth", max_nesting_depth)
        object.__setattr__(root, "_tool_call_types", MappingProxyType(registry))
        object.__setattr__(root, "_source_root", source_root)
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
                for call_index, call in enumerate(message.tool_calls):
                    object.__setattr__(call, "_message_index", message.message_index)
                    object.__setattr__(call, "_call_index", call_index)
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
        unreadable_calls: list[UnreadableToolCall] = []
        unreadable_responses: list[ToolMessage] = []
        for message in root._items:
            ordinals_by_role.setdefault(message.role, []).append(message.message_index)
            if isinstance(message, AssistantMessage):
                for call in message.tool_calls:
                    all_calls.append(call)
                    if isinstance(call, UnreadableToolCall):
                        unreadable_calls.append(call)
                    else:
                        calls_by_name.setdefault(call.name, []).append(call)
            elif isinstance(message, ToolMessage):
                if isinstance(message._call, UnreadableToolCall):
                    unreadable_responses.append(message)
                else:
                    responses_by_name.setdefault(message.name, []).append(message)
        object.__setattr__(root, "_all_calls", tuple(all_calls))
        object.__setattr__(root, "_unreadable_calls", tuple(unreadable_calls))
        object.__setattr__(root, "_unreadable_responses", tuple(unreadable_responses))
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
        return (self._message_at_position(position) for position in range(len(self)))

    def _message_at_position(self, position: int) -> MessageType:
        variant = self._position_variants.get(position)
        return self._message_at(self._ordinals[position]) if variant is None else variant

    def _message_at(self, ordinal: int) -> MessageType:
        segment = self._segments.get(ordinal)
        return self._root._items[ordinal] if segment is None else segment

    @overload
    def __getitem__(self, index: int) -> MessageType: ...

    @overload
    def __getitem__(self, index: slice) -> Messages: ...

    def __getitem__(self, index: int | slice) -> MessageType | Messages:
        if isinstance(index, slice):
            return self._view_positions(range(len(self))[index])
        position = range(len(self))[index]
        return self._message_at_position(position)

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

        if self._position_variants:
            by_ordinal: dict[int, list[int]] = {}
            for position, ordinal in enumerate(self._ordinals):
                by_ordinal.setdefault(ordinal, []).append(position)
            if all(ordinal in by_ordinal for ordinal in selected):
                return self._view_positions(
                    position for ordinal in selected for position in by_ordinal[ordinal]
                )

        return self._view_ordinals(selected)

    def _view_ordinals(self, ordinals: Iterable[int]) -> Messages:
        """Return an ordered identity-preserving view over root message ordinals.

        Unlike ``_select_ordinals``, this helper permits reordered and repeated ordinals.
        It is used only after a public slice, selection, or bounded combination has already
        established deterministic ordering.
        """

        selected = tuple(ordinals)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in selected):
            raise TypeError("message ordinals must be integers")
        if any(value < 0 or value >= len(self._root._items) for value in selected):
            raise IndexError("message ordinal is outside the root collection")
        view = object.__new__(Messages)
        view._root = self._root
        view._source_root = self._root._source_root
        view._items = self._root._items
        view._ordinals = selected
        view._ordinal_set = frozenset(selected)
        view._max_nesting_depth = self._root._max_nesting_depth
        view._all_calls = self._root._all_calls
        view._unreadable_calls = self._root._unreadable_calls
        view._unreadable_responses = self._root._unreadable_responses
        view._ordinals_by_role = self._root._ordinals_by_role
        view._calls_by_name = self._root._calls_by_name
        view._responses_by_name = self._root._responses_by_name
        view._tool_call_types = self._root._tool_call_types
        view._segments = MappingProxyType(
            {ordinal: self._segments[ordinal] for ordinal in selected if ordinal in self._segments}
        )
        view._position_variants = MappingProxyType({})
        view._chunk = False
        view._sealed = True
        return view

    def _view_positions(self, positions: Iterable[int]) -> Messages:
        """Select occurrences rather than resolving a repeated source index again."""

        selected = tuple(positions)
        view = self._view_ordinals(self._ordinals[position] for position in selected)
        view._install_variants(tuple(self[position] for position in selected))
        return view

    def _install_variants(self, messages: Sequence[MessageType]) -> None:
        """Install already validated occurrence variants on a newly constructed view."""

        if len(messages) != len(self):
            raise ValueError("message variants must cover every selected occurrence")
        first: dict[int, MessageType] = {}
        positions: dict[int, MessageType] = {}
        for position, (ordinal, message) in enumerate(zip(self._ordinals, messages, strict=True)):
            if message.message_index != ordinal:
                raise ValueError("message variant does not match its source index")
            previous = first.setdefault(ordinal, message)
            if previous is not message:
                positions[position] = message
        object.__setattr__(
            self,
            "_segments",
            MappingProxyType(
                {
                    ordinal: message
                    for ordinal, message in first.items()
                    if message is not self._root._items[ordinal]
                }
            ),
        )
        object.__setattr__(self, "_position_variants", MappingProxyType(positions))

    def _with_overrides(self, overrides: Mapping[int, MessageType]) -> Messages:
        """Apply one variant to every occurrence of each named source ordinal."""

        if any(ordinal not in self._ordinal_set for ordinal in overrides):
            raise ValueError("transformed message override is outside its source selection")
        return self._with_position_overrides(
            {
                position: overrides[ordinal]
                for position, ordinal in enumerate(self._ordinals)
                if ordinal in overrides
            }
        )

    def _with_position_overrides(self, overrides: Mapping[int, MessageType]) -> Messages:
        """Apply content variants with the same graph validation as the async path."""

        steps = self._override_steps(overrides)
        count = 0
        _raise_if_combine_expired()
        while True:
            try:
                next(steps)
            except StopIteration as finished:
                _raise_if_combine_expired()
                return finished.value
            count += 1
            if count % 256 == 0:
                _raise_if_combine_expired()

    def rewritten(self, wires: Sequence[Mapping[str, Any]]) -> Messages:
        """Return text-preserving message variants, retaining original indexes and provenance.

        A changed entry and its source must make no tool calls. Reordered and repeated
        selections keep each occurrence, including independently rewritten variants.
        """

        steps = self._rewrite_steps(wires)
        while True:
            try:
                next(steps)
            except StopIteration as finished:
                return self if not finished.value else self._with_position_overrides(finished.value)
            _raise_if_combine_expired()

    async def _rewritten_async(
        self, wires: Sequence[Mapping[str, Any]], *, positions: tuple[int, ...] | None = None
    ) -> Messages:
        steps = self._rewrite_steps(wires, positions=positions)
        count = 0
        while True:
            try:
                next(steps)
            except StopIteration as finished:
                return (
                    self
                    if not finished.value
                    else await self._with_position_overrides_async(finished.value)
                )
            _raise_if_combine_expired()
            count += 1
            if count % 16 == 0:
                await _yield_combine_cancellation()

    def _rewrite_steps(
        self, wires: Sequence[Mapping[str, Any]], *, positions: tuple[int, ...] | None = None
    ) -> Generator[None, None, dict[int, MessageType]]:
        """Validate one complete result before publishing any of its content variants."""

        if type(wires) not in {list, tuple}:
            raise ValueError("message rewrite must return a message list")
        selected = tuple(range(len(self))) if positions is None else positions
        if len(wires) != len(selected):
            raise ValueError("message rewrite must return one message per input")
        overrides: dict[int, MessageType] = {}
        calls = {call.id: call for call in self._root._all_calls if call._readable_for_rewrite()}
        for position, wire in zip(selected, wires, strict=True):
            yield
            if type(wire) is not dict:
                raise ValueError("rewritten messages must be JSON objects")
            wire = cast(dict[str, Any], wire)
            # Validate bounded JSON before equality or Pydantic sees adapter-owned objects.
            _checked_rewrite_json(wire, self._root._max_nesting_depth)
            original = self[position]
            ordinal = self._ordinals[position]
            original_wire = original.to_openai_dict()
            if wire == original_wire:
                continue
            if wire.get("tool_calls") or (
                isinstance(original, AssistantMessage) and original.tool_calls
            ):
                raise ValueError("a changed message and its source must make no tool calls")
            role = wire.get("role")
            probe: list[dict[str, Any]] = [wire]
            if role == "tool":
                # Validate the standalone message shape without requiring this view to
                # include its call. Actual model dispatch still checks self-containment.
                probe = [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": wire.get("tool_call_id"),
                                "function": {"name": "rewrite_shape", "arguments": "{}"},
                            }
                        ],
                    },
                    wire,
                ]
            try:
                parsed = Messages.from_openai(
                    probe, max_nesting_depth=self._root._max_nesting_depth
                )
            except (ConversationValidationError, TypeError, ValueError) as exc:
                raise ValueError("rewritten message is not valid OpenAI data") from exc
            replacement = parsed[-1]
            if isinstance(replacement, AssistantMessage) and replacement.tool_calls:
                raise ValueError("a rewritten message must make no tool calls")
            if replacement.content != original.content:
                raise ValueError("a rewritten message must preserve its source text")
            object.__setattr__(replacement, "_message_index", ordinal)
            object.__setattr__(replacement, "_provenance", original._content_provenance())
            object.__setattr__(replacement, "_synthetic_identity", original._synthetic_identity)
            anchor = original._rewrite_source or self._root._items[ordinal]
            object.__setattr__(replacement, "_rewrite_source", anchor)
            object.__setattr__(
                replacement,
                "_rewrite_content_changed",
                original._rewrite_content_changed
                or replacement.source_content != original.source_content,
            )
            if isinstance(replacement, ToolMessage):
                object.__setattr__(replacement, "_call", calls.get(replacement.tool_call_id))
            overrides[position] = replacement
        return overrides

    async def _with_overrides_async(self, overrides: Mapping[int, MessageType]) -> Messages:
        if any(ordinal not in self._ordinal_set for ordinal in overrides):
            raise ValueError("transformed message override is outside its source selection")
        return await self._with_position_overrides_async(
            {
                position: overrides[ordinal]
                for position, ordinal in enumerate(self._ordinals)
                if ordinal in overrides
            }
        )

    async def _with_position_overrides_async(
        self, overrides: Mapping[int, MessageType]
    ) -> Messages:
        """Apply view-local tool relinking with deadline and cancellation checkpoints."""

        steps = self._override_steps(overrides)
        count = 0
        _raise_if_combine_expired()
        while True:
            try:
                next(steps)
            except StopIteration as finished:
                _raise_if_combine_expired()
                return finished.value
            count += 1
            if count % 256 == 0:
                _raise_if_combine_expired()
            if count % 2048 == 0:
                await _yield_combine_cancellation()

    def _override_steps(
        self, overrides: Mapping[int, MessageType]
    ) -> Generator[None, None, Messages]:
        """Return this view with content variants and a view-local linked tool graph."""

        if any(position < 0 or position >= len(self) for position in overrides):
            raise ValueError("transformed message override is outside its source selection")
        view = self._view_ordinals(self._ordinals)
        selected = {
            position: overrides.get(position, self[position]) for position in range(len(self))
        }
        selected_by_ordinal: dict[int, list[MessageType]] = {}
        for position, message in selected.items():
            selected_by_ordinal.setdefault(self._ordinals[position], []).append(message)

        # A transformed assistant/result must never traverse back into the canonical graph.
        # Prefer selected variants, then clone any linked counterpart outside this selection.
        call_sources: dict[str, ToolCall] = {}
        response_sources: dict[str, ToolMessage] = {}
        for position, message in selected.items():
            yield
            if not message._readable_for_rewrite() and not (
                isinstance(message, AssistantMessage)
                and (
                    position in overrides
                    or any(call._response is not None for call in message.tool_calls)
                )
            ):
                continue
            if isinstance(message, AssistantMessage):
                call_sources.update((call.id, call) for call in message.tool_calls)
            elif isinstance(message, ToolMessage):
                if message.tool_call_id in response_sources:
                    raise ValueError("message rewrite contains duplicate tool results")
                response_sources[message.tool_call_id] = message
        for response in tuple(response_sources.values()):
            yield
            if response._call is not None:
                call_sources.setdefault(response.tool_call_id, response._call)
        for call in tuple(call_sources.values()):
            yield
            selected_responses = (
                selected_by_ordinal.get(call._response.message_index, [])
                if call._response is not None
                else []
            )
            if call._response is not None and (
                not selected_responses
                or any(
                    isinstance(response, ToolMessage) and response.tool_call_id == call.id
                    for response in selected_responses
                )
            ):
                response_sources.setdefault(call.id, call._response)

        call_clones: dict[str, ToolCall] = {}
        for call_id, call in call_sources.items():
            yield
            clone = call.model_copy()
            object.__setattr__(clone, "_arguments_cache", None)
            object.__setattr__(clone, "_response", None)
            call_clones[call_id] = clone
        response_clones: dict[str, ToolMessage] = {}
        for call_id, response in response_sources.items():
            yield
            clone = response.model_copy()
            object.__setattr__(clone, "_call", None)
            response_clones[call_id] = clone
        for call_id, call in call_clones.items():
            yield
            response = response_clones.get(call_id)
            if response is not None:
                object.__setattr__(call, "_response", response)
                object.__setattr__(response, "_call", call)

        for position, message in selected.items():
            yield
            if not message._readable_for_rewrite():
                if isinstance(message, AssistantMessage) and any(
                    call.id in call_clones for call in message.tool_calls
                ):
                    selected[position] = message.model_copy(
                        update={
                            "tool_calls": tuple(
                                call_clones.get(call.id, call) for call in message.tool_calls
                            )
                        }
                    )
                continue
            if isinstance(message, AssistantMessage) and message.tool_calls:
                message = message.model_copy(
                    update={
                        "tool_calls": tuple(call_clones[call.id] for call in message.tool_calls)
                    }
                )
            elif isinstance(message, ToolMessage):
                message = response_clones[message.tool_call_id]
            selected[position] = message
        view._install_variants(tuple(selected.values()))
        object.__setattr__(view, "_chunk", self._chunk)
        return view

    @classmethod
    def _coalesce_combine_roots(
        cls,
        parts: tuple[Messages, ...],
        meter: _CombineMeter,
    ) -> Generator[None, None, tuple[Messages, ...]]:
        """Graft bounded synthetic units onto one source root before normal relinking."""

        first_root = parts[0]._root
        if all(part._root is first_root for part in parts[1:]):
            return parts

        source_root: Messages | None = None
        for part in parts:
            candidate = part._root._source_root
            if candidate is None:
                continue
            if source_root is not None and candidate is not source_root:
                raise ValueError(
                    "combined source-backed selections must share one canonical source"
                )
            source_root = candidate

        if source_root is not None and len(source_root._items) > _MAX_COMBINED_MESSAGES:
            raise ValueError(f"combined message source exceeds {_MAX_COMBINED_MESSAGES} messages")
        items: list[MessageType] = []
        if source_root is not None:
            for canonical in source_root._items:
                yield from _charge_combine_steps(
                    meter,
                    "canonical_message_visits",
                )
                items.append(canonical)
        registry = source_root._tool_call_types if source_root is not None else MappingProxyType({})
        nesting = source_root._max_nesting_depth if source_root is not None else 100
        synthetic_ordinals: dict[tuple[str, object], int] = {}
        synthetic_sources: dict[tuple[str, object], MessageType] = {}
        specifications: list[tuple[Messages, tuple[int, ...], list[MessageType]]] = []

        for part in parts:
            mapped_ordinals: list[int] = []
            mapped_items: list[MessageType] = []
            for position, ordinal in enumerate(part._ordinals):
                yield from _charge_combine_steps(meter, "selection_visits")
                selected = part[position]
                key = _synthetic_key(selected)
                if key is None:
                    if source_root is None or part._root._source_root is not source_root:
                        raise ValueError("a constructed message is missing synthetic identity")
                    mapped = selected.message_index
                    if (
                        mapped < 0
                        or mapped >= len(source_root._items)
                        or (
                            type(source_root._items[mapped]) is not type(selected)
                            and selected._rewrite_source is not source_root._items[mapped]
                        )
                    ):
                        raise ValueError(
                            "source-backed message does not match its canonical source"
                        )
                else:
                    mapped = synthetic_ordinals.get(key, -1)
                    canonical = part._root._items[ordinal]
                    if mapped >= 0 and synthetic_sources[key] is not canonical:
                        previous = yield from _message_variant_steps(
                            synthetic_sources[key], meter, message_index=mapped
                        )
                        current = yield from _message_variant_steps(
                            canonical, meter, message_index=mapped
                        )
                        if not (
                            yield from _variant_representations_equal_steps(
                                previous, current, meter
                            )
                        ):
                            raise ValueError("constructed identity has different content variants")
                    if mapped < 0:
                        if len(synthetic_ordinals) >= _MAX_SYNTHETIC_MESSAGES:
                            raise ValueError(
                                "combined input exceeds "
                                f"{_MAX_SYNTHETIC_MESSAGES} constructed messages"
                            )
                        if len(items) >= _MAX_COMBINED_MESSAGES:
                            raise ValueError(
                                "combined message selection exceeds "
                                f"{_MAX_COMBINED_MESSAGES} messages"
                            )
                        canonical = part._root._items[ordinal]
                        if (
                            _synthetic_key(canonical) != key
                            or isinstance(canonical, ToolMessage)
                            or isinstance(canonical, AssistantMessage)
                            and canonical.tool_calls
                        ):
                            raise ValueError("constructed messages cannot inject tool linkage")
                        yield from _charge_combine_steps(
                            meter,
                            "message_clone_visits",
                        )
                        clone = canonical.model_copy()
                        object.__setattr__(clone, "_message_index", len(items))
                        mapped = len(items)
                        synthetic_ordinals[key] = mapped
                        synthetic_sources[key] = canonical
                        items.append(clone)
                mapped_ordinals.append(mapped)
                piece = selected
                reanchor = (
                    key is not None
                    and piece._rewrite_source is part._root._items[ordinal]
                    and piece._rewrite_source is not items[mapped]
                )
                if key is not None and (piece.message_index != mapped or reanchor):
                    yield from _charge_combine_steps(meter, "message_clone_visits")
                    piece = piece.model_copy()
                    object.__setattr__(piece, "_message_index", mapped)
                    if reanchor:
                        object.__setattr__(piece, "_rewrite_source", items[mapped])
                mapped_items.append(piece)
            specifications.append((part, tuple(mapped_ordinals), mapped_items))

        aggregate = cls._from_items(
            tuple(items),
            registry,
            nesting,
            source_root=source_root,
        )
        normalized = []
        for original, ordinals, variants in specifications:
            view = aggregate._view_ordinals(ordinals)
            view._install_variants(variants)
            object.__setattr__(view, "_chunk", original._chunk)
            normalized.append(view)
        return tuple(normalized)

    @classmethod
    def _combine_steps(
        cls,
        parts: Sequence[Messages],
        metrics: _CombineMetrics,
        *,
        mapped: bool = False,
    ) -> Generator[None, None, Messages]:
        """Combine in bounded steps that sync and async callers checkpoint identically."""

        meter = _CombineMeter(metrics)
        if not isinstance(parts, Sequence) or isinstance(parts, str | bytes | bytearray):
            raise TypeError("parts must be a sequence of Messages values")
        part_limit = _MAX_COMBINED_MESSAGES if mapped else _MAX_COMBINED_MESSAGE_PARTS
        if len(parts) > part_limit:
            raise ValueError(f"cannot combine more than {part_limit} message selections")
        if any(not isinstance(part, Messages) for part in parts):
            raise TypeError("every combined value must be Messages")
        present = tuple(part for part in parts if part)
        if not present:
            return parts[0]._view_ordinals(()) if parts else cls()
        present = yield from cls._coalesce_combine_roots(present, meter)
        root = present[0]._root
        if any(part._root is not root for part in present[1:]):
            raise AssertionError("combined message roots were not normalized")
        count = sum(len(part) for part in present)
        if count > _MAX_COMBINED_MESSAGES:
            raise ValueError(
                f"combined message selection exceeds {_MAX_COMBINED_MESSAGES} messages"
            )
        combined = present[0]._view_ordinals(
            ordinal for part in present for ordinal in part._ordinals
        )
        whole = object()
        pieces: dict[int, MessageType | object] = {}
        ordered_pieces: list[MessageType | object] = []
        variant_representations: dict[int, tuple[Message, object]] = {}
        variant_compatibility: dict[tuple[int, int], tuple[Message, bool]] = {}
        for part in present:
            for position, ordinal in enumerate(part._ordinals):
                if meter.record("selection_visits") and meter.checkpoint():
                    yield
                piece = part._position_variants.get(position, part._segments.get(ordinal, whole))
                ordered_pieces.append(piece)
                if ordinal not in pieces:
                    pieces[ordinal] = piece
                    continue
                existing = pieces[ordinal]
                if existing is piece:
                    yield from _charge_combine_steps(meter, "variant_cache_hits")
                    continue
                if not isinstance(existing, Message) or not isinstance(piece, Message):
                    continue
                compatibility_key = (ordinal, id(piece))
                cached = variant_compatibility.get(compatibility_key)
                if cached is not None and cached[0] is piece:
                    yield from _charge_combine_steps(meter, "variant_cache_hits")
                    compatible = cached[1]
                else:
                    existing_representation = yield from _memoized_message_variant_steps(
                        existing,
                        meter,
                        variant_representations,
                    )
                    piece_representation = yield from _memoized_message_variant_steps(
                        piece,
                        meter,
                        variant_representations,
                    )
                    compatible = yield from _variant_representations_equal_steps(
                        existing_representation,
                        piece_representation,
                        meter,
                    )
                    variant_compatibility[compatibility_key] = (piece, compatible)
                if compatible:
                    ordered_pieces[-1] = existing

        canonical_tools: dict[str, _CanonicalToolUnit] = {}
        for ordinal, canonical in enumerate(root._items):
            if meter.record("canonical_message_visits") and meter.checkpoint():
                yield
            if isinstance(canonical, AssistantMessage):
                for call in canonical.tool_calls:
                    if meter.record("canonical_call_visits") and meter.checkpoint():
                        yield
                    call_id = _require_tool_identity(call.id, "canonical tool call")
                    if call_id in canonical_tools:
                        raise ValueError(
                            f"canonical source contains ambiguous tool-call ID {call_id!r}"
                        )
                    if len(canonical_tools) >= _MAX_COMBINED_TOOL_IDENTITIES:
                        raise ValueError(
                            "canonical source exceeds "
                            f"{_MAX_COMBINED_TOOL_IDENTITIES} tool identities"
                        )
                    if call.message_index != ordinal:
                        raise ValueError(
                            "canonical tool call does not match its source message identity"
                        )
                    canonical_tools[call_id] = _CanonicalToolUnit(ordinal, call)
                continue
            if isinstance(canonical, ToolMessage):
                if meter.record("canonical_response_visits") and meter.checkpoint():
                    yield
                call_id = _require_tool_identity(
                    canonical.tool_call_id,
                    "canonical tool result",
                )
                if meter.record("canonical_lookups") and meter.checkpoint():
                    yield
                unit = canonical_tools.get(call_id)
                if unit is None:
                    raise ValueError("canonical tool result has no earlier compatible tool call")
                if unit.response is not None:
                    raise ValueError(
                        f"canonical source contains ambiguous tool-result ID {call_id!r}"
                    )
                if canonical._call is not unit.call or unit.call._response is not canonical:
                    raise ValueError("canonical tool call and result have incompatible links")
                unit.response_ordinal = ordinal
                unit.response = canonical

        for unit in canonical_tools.values():
            if meter.record("canonical_link_checks") and meter.checkpoint():
                yield
            if (unit.call._response is None) != (unit.response is None):
                raise ValueError("canonical tool call has an incompatible hidden result")

        selected_calls: dict[str, tuple[int, ToolCall, _CanonicalToolUnit]] = {}
        selected_responses: dict[str, tuple[int, ToolMessage, _CanonicalToolUnit]] = {}
        selected_assistants: dict[int, AssistantMessage] = {}
        response_positions: dict[str, int] = {}
        relink_tool_variants = False
        for position, (ordinal, piece) in enumerate(
            zip(combined._ordinals, ordered_pieces, strict=True)
        ):
            if meter.record("selected_message_visits") and meter.checkpoint():
                yield
            message = root._items[ordinal] if piece is whole else piece
            if not isinstance(message, Message) or message.message_index != ordinal:
                raise ValueError("combined content variant does not match its source message")
            if message._provenance is not None:
                for segment in message._provenance:
                    if meter.record("provenance_visits") and meter.checkpoint():
                        yield
                    for source in (*segment.sources, *segment.parts):
                        if meter.record("provenance_visits") and meter.checkpoint():
                            yield
                        synthetic_source = _synthetic_key(message) is not None
                        if (
                            not synthetic_source
                            and source.message_index != ordinal
                            or synthetic_source
                            and (
                                root._source_root is None
                                or source.message_index >= len(root._source_root._items)
                            )
                        ):
                            raise ValueError(
                                "combined content variant has incompatible source provenance"
                            )
            if isinstance(message, AssistantMessage):
                canonical = root._items[ordinal]
                rewritten = message._rewrite_source is canonical
                if not isinstance(canonical, AssistantMessage) and not rewritten:
                    raise ValueError("combined tool-call variant does not match its source message")
                canonical_calls = (
                    canonical.tool_calls if isinstance(canonical, AssistantMessage) else ()
                )
                if message.tool_calls:
                    selected_assistants[position] = message
                if len(message.tool_calls) != len(canonical_calls):
                    raise ValueError(
                        "combined tool-call variant does not preserve its source tool identities"
                    )
                for call, canonical_call in zip(
                    message.tool_calls,
                    canonical_calls,
                    strict=True,
                ):
                    if meter.record("selected_call_visits") and meter.checkpoint():
                        yield
                    call_id = _require_tool_identity(call.id, "selected tool call")
                    if meter.record("canonical_lookups") and meter.checkpoint():
                        yield
                    unit = canonical_tools.get(call_id)
                    if (
                        unit is None
                        or unit.call is not canonical_call
                        or unit.call_ordinal != ordinal
                        or call.message_index != ordinal
                        or not _same_tool_call_identity(call, unit.call)
                    ):
                        raise ValueError(
                            "combined tool-call variant does not match its source tool identity"
                        )
                    if call_id in selected_calls:
                        raise ValueError(
                            f"message selection contains duplicate tool-call ID {call_id!r}"
                        )
                    if len(selected_calls) >= _MAX_COMBINED_TOOL_IDENTITIES:
                        raise ValueError(
                            "combined selection exceeds "
                            f"{_MAX_COMBINED_TOOL_IDENTITIES} tool identities"
                        )
                    selected_calls[call_id] = (ordinal, call, unit)
                relink_tool_variants = relink_tool_variants or (
                    piece is not whole and bool(message.tool_calls)
                )
            elif isinstance(message, ToolMessage):
                if meter.record("selected_response_visits") and meter.checkpoint():
                    yield
                call_id = _require_tool_identity(
                    message.tool_call_id,
                    "selected tool result",
                )
                if meter.record("canonical_lookups") and meter.checkpoint():
                    yield
                unit = canonical_tools.get(call_id)
                if unit is None or (
                    unit.response_ordinal != ordinal
                    and message._rewrite_source is not root._items[ordinal]
                ):
                    raise ValueError(
                        "combined tool-result variant does not match its source tool identity"
                    )
                if call_id in selected_responses:
                    raise ValueError(
                        "message selection contains duplicate tool-result for tool-call ID "
                        f"{call_id!r}"
                    )
                if len(selected_responses) >= _MAX_COMBINED_TOOL_IDENTITIES:
                    raise ValueError(
                        f"combined selection exceeds {_MAX_COMBINED_TOOL_IDENTITIES} tool results"
                    )
                selected_responses[call_id] = (ordinal, message, unit)
                response_positions[call_id] = position
                relink_tool_variants = relink_tool_variants or piece is not whole

        variants: list[MessageType] = [
            root._items[ordinal] if not isinstance(piece, Message) else cast(MessageType, piece)
            for ordinal, piece in zip(combined._ordinals, ordered_pieces, strict=True)
        ]
        for _ in pieces:
            if meter.record("segment_visits") and meter.checkpoint():
                yield

        if relink_tool_variants:
            call_clones: dict[str, ToolCall] = {}
            for call_id, (ordinal, call, _unit) in selected_calls.items():
                if meter.record("call_clone_visits") and meter.checkpoint():
                    yield
                clone = call.model_copy()
                object.__setattr__(clone, "_arguments_cache", None)
                object.__setattr__(clone, "_response", None)
                object.__setattr__(clone, "_message_index", ordinal)
                call_clones[call_id] = clone
            response_clones: dict[str, ToolMessage] = {}
            for call_id, (ordinal, response, _unit) in selected_responses.items():
                if meter.record("response_clone_visits") and meter.checkpoint():
                    yield
                clone = response.model_copy()
                object.__setattr__(clone, "_call", None)
                object.__setattr__(clone, "_message_index", ordinal)
                response_clones[call_id] = clone
            for call_id, call in call_clones.items():
                response = response_clones.get(call_id)
                if response is None:
                    continue
                if meter.record("relink_visits") and meter.checkpoint():
                    yield
                unit = selected_calls[call_id][2]
                response_unit = selected_responses[call_id][2]
                if unit is not response_unit:
                    raise ValueError(
                        "combined tool-call and tool-result variants have mismatched identities"
                    )
                object.__setattr__(call, "_response", response)
                object.__setattr__(response, "_call", call)
            for position, assistant in selected_assistants.items():
                cloned_calls: list[ToolCall] = []
                for call in assistant.tool_calls:
                    if meter.record("assembly_visits") and meter.checkpoint():
                        yield
                    cloned_calls.append(call_clones[call.id])
                if meter.record("message_clone_visits") and meter.checkpoint():
                    yield
                variants[position] = assistant.model_copy(
                    update={"tool_calls": tuple(cloned_calls)}
                )
            for call_id in selected_responses:
                if meter.record("relink_visits") and meter.checkpoint():
                    yield
                variants[response_positions[call_id]] = response_clones[call_id]

        combined._install_variants(variants)
        object.__setattr__(combined, "_chunk", all(part._chunk for part in present))
        if meter.checkpoint(finish=True):
            yield
        return combined

    @classmethod
    def combine(
        cls,
        parts: Sequence[Messages],
        *,
        _metrics: _CombineMetrics | None = None,
    ) -> Messages:
        """Combine source-backed and explicitly constructed selections in written order.

        Combination normally reuses immutable message objects. When independently evaluated
        tool views need a self-contained call/result graph, it returns immutable relinked
        variants instead; callers must not rely on object identity. Value, content, provenance,
        order, transcript indexes, and tool identities are preserved. Constructed messages keep
        separate scan-local identity and unmapped provenance. Empty selections are ignored.
        Source-backed selections must share one canonical source, and work is bounded.
        """

        _raise_if_combine_expired()
        steps = cls._combine_steps(parts, _metrics or _CombineMetrics())
        while True:
            try:
                next(steps)
            except StopIteration as finished:
                return finished.value
            _raise_if_combine_expired()

    @classmethod
    async def _combine_async(
        cls,
        parts: Sequence[Messages],
        *,
        _metrics: _CombineMetrics | None = None,
        _mapped: bool = False,
    ) -> Messages:
        """Async evaluator path with the same work accounting plus cancellation yields."""

        _raise_if_combine_expired()
        metrics = _metrics or _CombineMetrics()
        steps = cls._combine_steps(parts, metrics, mapped=_mapped)
        while True:
            try:
                next(steps)
            except StopIteration as finished:
                return finished.value
            _raise_if_combine_expired()
            if metrics.checkpoints % _COMBINE_CANCELLATION_CHECK_INTERVAL == 0:
                await _yield_combine_cancellation()

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
        for ordinal in ranges:
            if not view._contains_root_ordinal(ordinal):
                raise ValueError("a divided message must be selected by the view")
        view._install_variants(
            tuple(
                message._segment_of(*ranges[message.message_index])
                if message.message_index in ranges
                else message
                for message in view
            )
        )
        object.__setattr__(view, "_chunk", True)
        return view

    @property
    def span(self) -> Span:
        """Return the source envelope covered by this view.

        Messages inside the span that the view does not select, such as the assistant
        turns of ``user_messages``, are not part of it. A reordered or repeated composition
        reports the earliest through latest source message while preserving its own iteration
        order separately.
        """

        if not self._ordinals:
            raise ValueError("an empty message selection has no span")
        if self._position_variants:
            return self._occurrence_span()
        first, last = min(self._ordinals), max(self._ordinals)
        part_contexts = {
            ordinal: source_part_mapping_for(
                self._message_at(ordinal)._content_provenance(),
                0,
                len(self._message_at(ordinal).content),
            )[0]
            for ordinal in dict.fromkeys(self._ordinals)
            if self._message_at(ordinal)._provenance is not None
            and self._message_at(ordinal).content
        }
        first_piece = self._segments.get(first)
        last_piece = self._segments.get(last)
        first_ranges = (
            source_ranges_for(first_piece._provenance, 0, len(first_piece.content))
            if first_piece is not None
            and first_piece._provenance is not None
            and first_piece.content
            else ()
        )
        last_ranges = (
            source_ranges_for(last_piece._provenance, 0, len(last_piece.content))
            if last_piece is not None and last_piece._provenance is not None and last_piece.content
            else ()
        )
        start = (
            0
            if _synthetic_key(self._message_at(first)) is not None
            else min(source.message_start for source in first_ranges)
            if first_ranges
            else first_piece._range[0]
            if first_piece is not None and first_piece._segment is not None
            else 0
        )
        end = (
            len(self._message_at(last).content)
            if _synthetic_key(self._message_at(last)) is not None or part_contexts.get(last)
            else max(source.message_end for source in last_ranges)
            if last_ranges
            else last_piece._range[1]
            if last_piece is not None and last_piece._segment is not None
            else len(self._root._items[last].content)
        )
        contains_constructed = any(
            _synthetic_key(self._message_at(ordinal)) is not None or part_contexts.get(ordinal)
            for ordinal in self._ordinals
        )
        mapped: list[ContentSourceRange] = []
        mapped_seen: set[ContentSourceRange] = set()
        unmapped = False
        if contains_constructed:
            for ordinal in self._ordinals:
                message = self._message_at(ordinal)
                if not message.content:
                    unmapped = unmapped or _synthetic_key(message) is not None
                    continue
                ranges, has_unmapped = source_mapping_for(
                    message._content_provenance(),
                    0,
                    len(message.content),
                    check_deadline=_raise_if_combine_expired,
                )
                unmapped = unmapped or has_unmapped
                for source in ranges:
                    if source not in mapped_seen:
                        mapped_seen.add(source)
                        mapped.append(source)
        return Span(
            first,
            start,
            end,
            end_message_index=last,
            source_ranges=tuple(mapped),
            source_unmapped=unmapped,
            source_parts=tuple(
                dict.fromkeys(
                    part for ordinal in self._ordinals for part in part_contexts.get(ordinal, ())
                )
            ),
        )

    def _occurrence_span(self) -> Span:
        """Envelope all occurrence variants, including distinct ranges of one source."""

        spans = [self._view_positions((position,)).span for position in range(len(self))]
        first = min(span.message_index for span in spans)
        last = max(
            span.end_message_index if span.end_message_index is not None else span.message_index
            for span in spans
        )
        contextual = any(
            span.source_ranges or span.source_parts or span.source_unmapped for span in spans
        )
        ranges: dict[ContentSourceRange, None] = {}
        if contextual:
            for message in self:
                if message.content:
                    mapped, _ = source_mapping_for(
                        message._content_provenance(), 0, len(message.content)
                    )
                    ranges.update(dict.fromkeys(mapped))
        return Span(
            first,
            min(span.start for span in spans if span.message_index == first),
            max(span.end for span in spans if span.end_message_index == last),
            end_message_index=last,
            source_ranges=tuple(ranges),
            source_unmapped=any(span.source_unmapped for span in spans),
            source_parts=tuple(dict.fromkeys(part for span in spans for part in span.source_parts)),
        )

    def source_ranges(
        self, message_index: int, start: int, end: int
    ) -> tuple[ContentSourceRange, ...]:
        """Map one selected message range to original message/content-part offsets."""

        if (
            isinstance(message_index, bool)
            or not isinstance(message_index, int)
            or message_index not in self._ordinal_set
        ):
            raise ValueError("source range message_index must name a selected message")
        result = self._message_at(message_index)._source_ranges(start, end)
        for message in self._position_variants.values():
            if (
                message.message_index == message_index
                and message._source_ranges(start, end) != result
            ):
                raise ValueError("source range is ambiguous; bind a span to one occurrence")
        return result

    def _contains_root_ordinal(self, ordinal: int) -> bool:
        """Return whether this view selects one root-message ordinal."""

        return ordinal in self._ordinal_set

    def _is_root_view(self) -> bool:
        return self is self._root

    def shares_root_with(self, other: Messages) -> bool:
        """Return whether both values are views of the same canonical conversation."""

        return isinstance(other, Messages) and self._root is other._root

    def shares_source_with(self, other: Messages) -> bool:
        """Return whether both values retain one canonical conversation source."""

        return (
            isinstance(other, Messages)
            and self._root._source_root is not None
            and self._root._source_root is other._root._source_root
        )

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
        selected = self._segments.get(span.message_index)
        if isinstance(span._view, Message):
            selected = next(
                (
                    message
                    for message in self
                    if message.message_index == span.message_index
                    and (message is span._view or _same_message_variant(message, span._view))
                ),
                None,
            )
        if (
            span._view is not None
            and selected is not None
            and isinstance(span._view, Message)
            and _same_message_variant(selected, span._view)
        ):
            content = selected.content
            assert span._view_start is not None and span._view_end is not None
            return content[
                max(0, span._view_start - left) : min(len(content), span._view_end + right)
            ]
        content = message.content
        return content[max(0, span.start - left) : min(len(content), span.end + right)]

    def _bind_span(self, span: Span) -> Span:
        """Map a model-local span to its selected source or constructed message."""

        if not isinstance(span, Span):
            raise TypeError("model span must be a Span")
        if not span.within_one_message:
            raise ValueError("a model span must stay within one selected message")
        local_index = span.message_index
        if local_index >= len(self._ordinals):
            raise ValueError("span message index is outside the model input")
        ordinal = self._ordinals[local_index]
        selected = self[local_index]
        if span.end > len(selected.content):
            raise ValueError("span is outside message content")
        if span.end <= span.start and selected.content:
            raise ValueError("span over message text must not be empty")

        ranges: tuple[ContentSourceRange, ...] = ()
        unmapped = _synthetic_key(selected) is not None
        if span.end > span.start:
            ranges, unmapped = source_mapping_for(
                selected._content_provenance(),
                span.start,
                span.end,
                check_deadline=_raise_if_combine_expired,
            )
        synthetic = _synthetic_key(selected) is not None
        source_parts, context_known = source_part_mapping_for(
            selected._content_provenance(), span.start, span.end
        )
        if not synthetic and unmapped and not context_known:
            raise ValueError("transformed range includes inserted text with no source position")

        start, end = span.start, span.end
        if not synthetic and ranges and not unmapped:
            start = min(source.message_start for source in ranges)
            end = max(source.message_end for source in ranges)
        span = replace(
            span,
            message_index=ordinal,
            end_message_index=ordinal,
            start=start,
            end=end,
            source_ranges=ranges if selected._provenance is not None or synthetic else (),
            source_unmapped=unmapped,
            source_parts=source_parts,
            _view=selected,
            _view_start=span.start,
            _view_end=span.end,
        )
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
        bounds_message = (
            span._view
            if isinstance(span._view, Message)
            and (_synthetic_key(span._view) is not None or span.source_parts)
            else message
        )
        if require_bound and span._message is None:
            raise ValueError("span is not bound to a source message")
        if require_bound and span._message is not message:
            raise ValueError("span source message is not present in this Messages value")
        if span.end > len(bounds_message.content):
            raise ValueError("span is outside message content")
        if span.end <= span.start and bounds_message.content:
            raise ValueError("span over message text must not be empty")
        return message

    def _select_indexed_ordinals(self, ordinals: tuple[int, ...]) -> Messages:
        """Intersect one root index with this view without scanning message data."""

        return self._select_ordinals(
            ordinal for ordinal in ordinals if self._contains_root_ordinal(ordinal)
        )

    def _select_role(self, role: str) -> Messages:
        """Keep messages of ``role`` in this value's current order."""

        return self._view_positions(
            position for position, message in enumerate(self) if message.role == role
        )

    def _select_positions(self, positions: Iterable[int]) -> Messages:
        """Return a reference-backed sub-view from LOCAL positions of this collection.

        ``positions`` index into this view's own sequence (0..len-1). Repeated and reordered
        positions retain their matching message variants without copying or re-indexing the
        source. Token-bounded chunks therefore share their selected occurrences by reference."""

        return self._view_positions(positions)

    def assert_self_contained(self) -> None:
        """Reject a selection that is not a valid standalone OpenAI sequence.

        An empty selection has nothing to score; a tool-result message whose originating
        assistant tool call is not also selected is an orphan no provider will accept. Repeated
        calls or results are invalid even when composition selected the same source turn more
        than once. These cases fail here — before any builder/adapter/provider dispatch — rather
        than being widened or scored as a malformed request. O(k) in the selection size (no
        message-text scan).

        A ``split`` chunk is exempt from the orphan check: it is only ever a score input, and
        the score builders flatten it into evidence text rather than send it as a request."""

        if not self._ordinals:
            raise ValueError("message selection is empty; nothing to score")
        selected_call_ids: set[str] = set()
        selected_call_links: set[int] = set()
        selected_response_ids: set[str] = set()
        selected_response_links: set[int] = set()
        for message in self:
            if isinstance(message, AssistantMessage):
                for call in message.tool_calls:
                    link = id(call)
                    if call.id in selected_call_ids or link in selected_call_links:
                        raise ValueError(
                            f"message selection contains duplicate tool-call ID {call.id!r}"
                        )
                    selected_call_ids.add(call.id)
                    selected_call_links.add(link)
                continue
            if isinstance(message, ToolMessage):
                call = message._call
                link = id(call) if call is not None else None
                if (
                    message.tool_call_id in selected_response_ids
                    or link is not None
                    and link in selected_response_links
                ):
                    raise ValueError(
                        "message selection contains duplicate tool-result for tool-call ID "
                        f"{message.tool_call_id!r}"
                    )
                selected_response_ids.add(message.tool_call_id)
                if link is not None:
                    selected_response_links.add(link)
                if not self._chunk and (
                    call is None
                    or call.id not in selected_call_ids
                    or link not in selected_call_links
                ):
                    raise ValueError(
                        "tool-result message selected before its originating assistant tool call"
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
        """System-role messages from this value, in its current order."""

        return self._select_role("system")

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
        """Developer-role messages from this value, in its current order."""

        return self._select_role("developer")

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
        """User-role messages from this value, in its current order."""

        return self._select_role("user")

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
        """Assistant-role messages from this value, in its current order."""

        return self._select_role("assistant")

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

        return self._select_role("tool")

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
            for message in self
            if isinstance(message, AssistantMessage)
            for call in message.tool_calls
        ]

    def _name_counts(self, *, responses: bool = False) -> Mapping[str, int]:
        """Compute a requested name aggregate within the canonical value's fault boundary."""

        if responses:
            self._root._read_call_names()
        items = self.tool_responses if responses else self.tool_calls
        counts: dict[str, int] = {}
        for index, item in enumerate(items):
            if index % 256 == 0:
                _raise_if_combine_expired()
            name = item.name
            counts[name] = counts.get(name, 0) + 1
        return MappingProxyType(counts)

    def _read_call_names(self) -> None:
        for call in self._root._unreadable_calls:
            _raise_if_combine_expired()
            if self._contains_root_ordinal(call.message_index):
                raise InvalidInputError(call.fault)

    def _read_response_names(self) -> None:
        for response in self._root._unreadable_responses:
            _raise_if_combine_expired()
            if self._contains_root_ordinal(response.message_index):
                _ = response.name

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
            found = bool(candidates)
        else:
            found = any(self._contains_root_ordinal(call.message_index) for call in candidates)
        if found:
            return True
        self._read_call_names()
        return False

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

        self._read_call_names()
        candidates = self._root._calls_by_name.get(name, ())
        if self._is_root_view():
            return len(candidates)
        return sum(call.name == name for call in self.tool_calls)

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

        self._read_call_names()
        candidates = self._root._calls_by_name.get(name, ())
        if self._is_root_view():
            return list(candidates)
        return [call for call in self.tool_calls if call.name == name]

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

        # Canonical result-by-name lookup first resolves the root's call names,
        # even when this view contains no result for an unreadable call.
        self._root._read_call_names()
        self._read_response_names()
        candidates = self._root._responses_by_name.get(name, ())
        if self._is_root_view():
            return list(candidates)
        return [
            message for message in self if isinstance(message, ToolMessage) and message.name == name
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
        """Append one allowlisted text message to an immutable model input.

        In rules, use this after a ``message(...)`` constructor, for example
        ``message("user", "hello").add_message("assistant", "hi")``.
        Python integrations may also append to a source selection; its source messages retain
        identity and the new message has separate scan-local identity with unmapped text.

        Internal:

        Reference-preserving: only the NEW ``Message`` (and its lightweight index/link metadata)
        is created. Every existing message — and its ``ContentPart``s, ``ToolCall``s, and
        ``ToolMessage``s — is reused BY IDENTITY, so ``old[i] is new[i]`` for every prior turn.
        The prior implementation round-tripped through ``to_openai_messages()``/``from_openai()``,
        which re-validated and recreated every object; this does not."""

        return self._add_synthetic_message(role, content, identity=object())

    def _add_synthetic_message(
        self,
        role: str,
        content: str,
        *,
        identity: object,
        provenance: tuple[ProvenanceSegment, ...] | None = None,
    ) -> Messages:
        """Append one validated synthetic message with private scan-local identity."""

        return self._add_synthetic_messages(((role, content, identity, provenance),))

    def _empty_constructor_context(self) -> Messages:
        """Keep source provenance for constructed text without copying source indices."""

        root = Messages()
        object.__setattr__(root, "_source_root", self._root._source_root)
        object.__setattr__(root, "_max_nesting_depth", self._root._max_nesting_depth)
        return root

    def _add_synthetic_messages(
        self,
        specifications: tuple[
            tuple[
                str,
                str,
                object,
                tuple[ProvenanceSegment, ...] | None,
            ],
            ...,
        ],
    ) -> Messages:
        """Append one bounded constructor batch while rebuilding aggregate indices once."""

        if not specifications:
            return self
        existing = [
            message for message in self._root._items if message._synthetic_identity is not None
        ]
        if len(existing) + len(specifications) > _MAX_SYNTHETIC_MESSAGES:
            raise ValueError(f"constructed input exceeds {_MAX_SYNTHETIC_MESSAGES} messages")
        total_characters = sum(len(message.content) for message in existing)
        total_bytes = sum(_validate_synthetic_text(message.content) for message in existing)
        additions: list[MessageType] = []
        for role, content, identity, provenance in specifications:
            if not isinstance(role, str) or role not in {
                "system",
                "developer",
                "user",
                "assistant",
            }:
                raise ValueError("add_message() role must be system, developer, user, or assistant")
            if not isinstance(content, str):
                raise TypeError("add_message() content must be a string")
            if identity is None:
                raise ValueError("constructed message identity must not be null")
            total_characters += len(content)
            total_bytes += _validate_synthetic_text(content)
            if total_characters > _MAX_SYNTHETIC_TEXT_CHARACTERS:
                raise ValueError(
                    f"constructed input text exceeds {_MAX_SYNTHETIC_TEXT_CHARACTERS} characters"
                )
            if total_bytes > _MAX_SYNTHETIC_TEXT_BYTES:
                raise ValueError(
                    f"constructed input text exceeds {_MAX_SYNTHETIC_TEXT_BYTES} UTF-8 bytes"
                )
            message_class: type[MessageType] = {
                "system": SystemMessage,
                "developer": DeveloperMessage,
                "user": UserMessage,
                "assistant": AssistantMessage,
            }[role]
            new_message = message_class.model_validate(
                {"role": role, "content": content},
                strict=True,
            )
            object.__setattr__(
                new_message,
                "_message_index",
                len(self._root._items) + len(additions),
            )
            object.__setattr__(new_message, "_synthetic_identity", identity)
            if provenance is None:
                provenance = (ProvenanceSegment(0, len(content), (), "whole"),) if content else ()
            object.__setattr__(new_message, "_provenance", provenance)
            additions.append(new_message)
        root = self._from_items(
            (*self._root._items, *additions),
            self._root._tool_call_types,
            self._root._max_nesting_depth,
            source_root=self._root._source_root,
        )
        result = root._view_ordinals(
            (*self._ordinals, *(message.message_index for message in additions))
        )
        object.__setattr__(result, "_segments", self._segments)
        object.__setattr__(result, "_position_variants", self._position_variants)
        object.__setattr__(result, "_chunk", self._chunk)
        return result

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
