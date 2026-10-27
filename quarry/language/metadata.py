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

"""Typed annotations used to generate Quarry's public language catalog."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, TypeAlias


class Use(StrEnum):
    """Places where a Boolean or collection-valued item may appear."""

    CONDITION = "condition"
    FLOW_OPERAND = "flow_operand"
    COMPREHENSION_BODY = "comprehension_body"
    COMPREHENSION_SOURCE = "comprehension_source"


PublicType: TypeAlias = Literal[
    "boolean",
    "integer",
    "number",
    "string",
    "boolean | number | string",
    "string_variable",
    "text_reference",
    "model_reference",
    "model_input",
    "message",
    "span",
    "collection",
    "collection<message>",
    "collection<model_input>",
    "collection<span>",
    "collection<string>",
    "collection<tool_call>",
    "collection<tool_response>",
    "collection<tool_definition>",
]
ResourceGroup: TypeAlias = Literal[
    "model",
    "score",
    "completion",
    "token",
    "topic_prober",
    "capability_classifier",
]
SurfaceFieldType: TypeAlias = Literal[
    "string",
    "string | null",
    "object",
    "tool_call",
    "tool_response | null",
    "collection<tool_call>",
]
SURFACE_FIELD_TYPES: tuple[SurfaceFieldType, ...] = (
    "string",
    "string | null",
    "object",
    "tool_call",
    "tool_response | null",
    "collection<tool_call>",
)
CompilerTag: TypeAlias = Literal[
    "field_first_external",
    "score_external",
    "whole_conversation_inference",
    "chunk_scalar",
    "complete_comparison",
    "chunk_source",
    "chunk_count",
]

PUBLIC_TYPES: tuple[PublicType, ...] = (
    "boolean",
    "integer",
    "number",
    "string",
    "boolean | number | string",
    "string_variable",
    "text_reference",
    "model_reference",
    "model_input",
    "message",
    "span",
    "collection",
    "collection<message>",
    "collection<model_input>",
    "collection<span>",
    "collection<string>",
    "collection<tool_call>",
    "collection<tool_response>",
    "collection<tool_definition>",
)
# Implicit conversions: a value of the first type is accepted where the second is expected.
TYPE_CONVERSIONS: tuple[tuple[PublicType, PublicType, str], ...] = (
    (
        "message",
        "model_input",
        "A message of the conversation, such as `user_messages[-1]` or the element `m` of "
        "`any m in messages`, is the model input that holds only that message. A message the "
        "conversation does not have is an empty model input, which makes no model call.",
    ),
)
RESOURCE_GROUPS: tuple[ResourceGroup, ...] = (
    "model",
    "score",
    "completion",
    "token",
    "topic_prober",
    "capability_classifier",
)
COMPILER_TAGS: tuple[CompilerTag, ...] = (
    "field_first_external",
    "score_external",
    "whole_conversation_inference",
    "chunk_scalar",
    "complete_comparison",
    "chunk_source",
    "chunk_count",
)


@dataclass(frozen=True)
class DSLParameter:
    """Public parameter facts that Python's annotation cannot express."""

    accepted_types: tuple[PublicType, ...] = ()
    name: str | None = None
    required: bool | None = None
    literal_only: bool = False
    field_only: bool = False
    hidden: bool = False
    minimum: int | float | None = None
    maximum: int | float | None = None
    allowed_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class FunctionMetadata:
    """Facts for an ordinary Python function exposed to Quarry rules."""

    name: str
    return_type: PublicType | None
    allowed_uses: tuple[Use, ...]
    resource_group: ResourceGroup | None = None
    compiler_tags: tuple[CompilerTag, ...] = ()
    accessor_surface: str | None = None
    accessor_filter: str | None = None
    implicit_receiver_count: int = 0
    syntax: str | None = None
    deprecated: bool = False
    replacement: str | None = None

    @property
    def id(self) -> str:
        """Return the ID of the function."""
        return f"function:{self.name}"


@dataclass(frozen=True)
class CompilerFormMetadata:
    """Facts for public syntax implemented directly by a compiler helper or owned type."""

    name: str
    return_type: PublicType
    allowed_uses: tuple[Use, ...]
    resource_group: ResourceGroup | None = None
    compiler_tags: tuple[CompilerTag, ...] = ()
    syntax: str | None = None
    implicit_receiver_count: int = 0
    deprecated: bool = False
    replacement: str | None = None

    @property
    def id(self) -> str:
        """Return the ID of the compiler form."""
        return f"compiler_form:{self.name}"


@dataclass(frozen=True)
class SurfaceMetadata:
    """Facts for a bare conversation value such as ``messages`` or ``tools``."""

    name: str
    public_type: PublicType
    domain: Literal["event", "static"]
    allowed_uses: tuple[Use, ...]
    fields_from: type[Any] | None = None
    model_input: bool = False

    @property
    def id(self) -> str:
        """Return the ID of the surface."""
        return f"surface:{self.name}"


@dataclass(frozen=True)
class SurfaceFieldMetadata:
    """Agent-facing type and usage text for one runtime-backed surface field."""

    name: str
    public_type: SurfaceFieldType
    description: str


@dataclass(frozen=True)
class SurfaceFieldExclusion:
    """Explain why one runtime field is not safe to advertise for rule authoring."""

    name: str
    reason: str


@dataclass(frozen=True)
class ConstructMetadata:
    """Facts for a complete grammar-backed form implemented by a transformer handler."""

    name: str
    grammar_rule: str | None = None

    @property
    def id(self) -> str:
        """Return the ID of the construct."""
        return f"construct:{self.name}"


def _register(target: Any, metadata: Any) -> Any:
    target.__quarry_language_metadata__ = metadata
    return target


def dsl_function(
    *,
    name: str | None = None,
    return_type: PublicType | None = None,
    allowed_uses: tuple[Use, ...] = (),
    resource_group: ResourceGroup | None = None,
    compiler_tags: tuple[CompilerTag, ...] = (),
    accessor_surface: str | None = None,
    accessor_filter: str | None = None,
    implicit_receiver_count: int = 0,
    syntax: str | None = None,
    deprecated: bool = False,
    replacement: str | None = None,
):
    """Mark a real Python function as part of Quarry's public rule language."""

    def decorate(target: Any) -> Any:
        public_name = name or target.__name__
        return _register(
            target,
            FunctionMetadata(
                name=public_name,
                return_type=return_type,
                allowed_uses=allowed_uses,
                resource_group=resource_group,
                compiler_tags=compiler_tags,
                accessor_surface=accessor_surface,
                accessor_filter=accessor_filter,
                implicit_receiver_count=implicit_receiver_count,
                syntax=syntax,
                deprecated=deprecated,
                replacement=replacement,
            ),
        )

    return decorate


def dsl_compiler_form(
    *,
    name: str,
    return_type: PublicType,
    allowed_uses: tuple[Use, ...] = (),
    resource_group: ResourceGroup | None = None,
    compiler_tags: tuple[CompilerTag, ...] = (),
    syntax: str | None = None,
    implicit_receiver_count: int = 0,
    deprecated: bool = False,
    replacement: str | None = None,
):
    """Mark a compiler helper or owned type as a public Quarry form."""

    metadata = CompilerFormMetadata(
        name=name,
        return_type=return_type,
        allowed_uses=allowed_uses,
        resource_group=resource_group,
        compiler_tags=compiler_tags,
        syntax=syntax,
        implicit_receiver_count=implicit_receiver_count,
        deprecated=deprecated,
        replacement=replacement,
    )

    def decorate(target: Any) -> Any:
        return _register(target, metadata)

    return decorate


def dsl_surface(
    *,
    name: str,
    public_type: PublicType,
    domain: Literal["event", "static"],
    allowed_uses: tuple[Use, ...],
    fields_from: type[Any] | None = None,
    model_input: bool = False,
):
    """Mark the property or class that owns a bare Quarry surface."""

    metadata = SurfaceMetadata(
        name=name,
        public_type=public_type,
        domain=domain,
        allowed_uses=allowed_uses,
        fields_from=fields_from,
        model_input=model_input,
    )

    def decorate(target: Any) -> Any:
        return _register(target, metadata)

    return decorate


def dsl_construct(*, name: str, grammar_rule: str | None = None):
    """Mark a transformer handler as the owner of a public grammar-backed form."""

    metadata = ConstructMetadata(name=name, grammar_rule=grammar_rule)

    def decorate(target: Any) -> Any:
        return _register(target, metadata)

    return decorate


__all__ = [
    "CompilerFormMetadata",
    "CompilerTag",
    "COMPILER_TAGS",
    "ConstructMetadata",
    "DSLParameter",
    "FunctionMetadata",
    "PublicType",
    "PUBLIC_TYPES",
    "ResourceGroup",
    "RESOURCE_GROUPS",
    "SURFACE_FIELD_TYPES",
    "SurfaceFieldExclusion",
    "SurfaceFieldMetadata",
    "SurfaceFieldType",
    "SurfaceMetadata",
    "Use",
    "dsl_compiler_form",
    "dsl_construct",
    "dsl_function",
    "dsl_surface",
]
