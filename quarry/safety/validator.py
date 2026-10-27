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

"""Static rule validation to detect potentially expensive patterns."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, cast

import orjson

from quarry.cache.regex_cache import regex_cache
from quarry.language.runtime import (
    accessor_functions,
    external_function_costs,
    function_by_name,
    function_signatures,
    surface_by_name,
)
from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.dsl import (
    AllComprehension,
    AnyComprehension,
    BinaryOp,
    ConditionNode,
    ContainsAnyOp,
    FieldAccess,
    FilterComprehension,
    FlowOp,
    FunctionCall,
    Identifier,
    Literal,
    MapComprehension,
    ModelCall,
    PatternMatch,
    StringCount,
    StringPosition,
    StringSetQuantifier,
    walk_condition_visits,
)
from quarry.types.dsl.string_ops import StringModifiers
from quarry.types.exceptions import ConversationValidationError, EvaluationError
from quarry.types.messages import Messages
from quarry.types.rules import DSLRule
from quarry.types.text_resources import TEXT_REF_FUNCTION

OPENAI_MESSAGE_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
OPENAI_TOP_LEVEL_KEYS = frozenset({"messages", "tools"})
OPENAI_TOOL_KEYS = frozenset({"type", "function"})
OPENAI_FUNCTION_KEYS = frozenset({"name", "description", "parameters"})
OPENAI_TOOL_CALL_KEYS = frozenset({"id", "type", "function"})
OPENAI_TOOL_RESPONSE_KEYS = frozenset({"role", "content", "tool_call_id"})
OPENAI_TEXT_PART_KEYS = frozenset({"type", "text"})
OPENAI_IMAGE_PART_KEYS = frozenset({"type", "image_url"})
OPENAI_IMAGE_URL_KEYS = frozenset({"url", "detail"})
OPENAI_AUDIO_PART_KEYS = frozenset({"type", "input_audio"})
OPENAI_AUDIO_KEYS = frozenset({"data", "format"})


class Severity(Enum):
    """Severity of a validation warning."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class Complexity(Enum):
    """Complexity of a validation warning."""

    # Keep members ordered from low to extreme; the plugin ranks estimates by this order.
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class ValidationWarning:
    """Advisory warning about rule cost or detection quality."""

    severity: Severity
    message: str
    suggestion: str


@dataclass
class RegexError:
    """Error for an invalid regex pattern."""

    variable: str
    pattern: str
    error: str


@dataclass
class SemanticError:
    """Error for semantic issues like undefined variables or wrong function arity."""

    error_type: str
    message: str
    location: str | None = None


@dataclass
class ValidationResult:
    """Result of rule validation."""

    rule_name: str
    warnings: list[ValidationWarning]
    estimated_complexity: Complexity
    regex_errors: list[RegexError] = field(default_factory=list)
    semantic_errors: list[SemanticError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True if no fatal errors (regex/semantic errors are fatal)."""
        return len(self.regex_errors) == 0 and len(self.semantic_errors) == 0


@dataclass(frozen=True)
class _ValueShape:
    """Known fields for one authored value, derived from the language catalog."""

    label: str
    fields: dict[str, str] | None = None
    element: _ValueShape | None = None
    open_fields: bool = False


def _surface_shape(name: str) -> _ValueShape | None:
    item = surface_by_name(name)
    if item is None:
        return None
    public_type = str(item["public_type"])
    if public_type.startswith("collection<"):
        fields = {str(field["name"]): str(field["type"]) for field in item["field_details"]}
        return _ValueShape(
            public_type,
            element=_ValueShape(f"{name} item", fields=fields),
        )
    return _shape_from_public_type(public_type)


def _shape_from_public_type(public_type: str) -> _ValueShape | None:
    normalized = public_type.removesuffix(" | null")
    if normalized == "object":
        return _ValueShape("object", open_fields=True)
    surface_for_type = {
        "message": "messages",
        "tool_call": "tool_calls",
        "tool_response": "tool_responses",
        "tool_definition": "tools",
    }.get(normalized)
    if surface_for_type is not None:
        surface = _surface_shape(surface_for_type)
        return surface.element if surface is not None else None
    if normalized.startswith("collection<") and normalized.endswith(">"):
        element_type = normalized.removeprefix("collection<").removesuffix(">")
        return _ValueShape(normalized, element=_shape_from_public_type(element_type))
    if normalized in {"model_input", "collection"}:
        return None
    return _ValueShape(normalized)


def _expression_shape(node: ConditionNode, scopes: dict[str, _ValueShape]) -> _ValueShape | None:
    if isinstance(node, Identifier):
        return scopes.get(node.name) or _surface_shape(node.name)
    if isinstance(node, FunctionCall):
        if node.name == "_slice" and node.args:
            return _expression_shape(node.args[0], scopes)
        if node.name == "_index" and node.args:
            collection = _expression_shape(node.args[0], scopes)
            return collection.element if collection is not None else None
        accessor = accessor_functions().get(node.name)
        if accessor is not None:
            return _surface_shape(accessor[0])
        item = function_by_name(node.name)
        if item is not None and item.get("return_type") is not None:
            return _shape_from_public_type(str(item["return_type"]))
        return None
    if isinstance(node, FieldAccess):
        shape, fields = _field_root(node, scopes)
        resolved, _ = _follow_fields(shape, fields)
        return resolved
    return None


def _field_root(
    node: FieldAccess,
    scopes: dict[str, _ValueShape],
) -> tuple[_ValueShape | None, list[str]]:
    if node.base is not None:
        return _expression_shape(node.base, scopes), list(node.fields)
    if not node.fields:
        return None, []
    root = str(node.fields[0])
    return scopes.get(root) or _surface_shape(root), [str(field) for field in node.fields[1:]]


def _follow_fields(
    shape: _ValueShape | None,
    fields: list[str],
) -> tuple[_ValueShape | None, tuple[str, _ValueShape] | None]:
    current = shape
    for field_name in fields:
        if current is None:
            return None, None
        if current.open_fields:
            return current, None
        if current.fields is None or field_name not in current.fields:
            return None, (field_name, current)
        current = _shape_from_public_type(current.fields[field_name])
    return current, None


def _scopes_from_ancestors(ancestors: tuple[ConditionNode, ...]) -> dict[str, _ValueShape]:
    scopes: dict[str, _ValueShape] = {}
    comprehension_types = (
        AnyComprehension,
        AllComprehension,
        FilterComprehension,
        MapComprehension,
    )
    for ancestor in ancestors:
        if not isinstance(ancestor, comprehension_types):
            continue
        collection = _expression_shape(ancestor.collection, scopes)
        if collection is not None and collection.element is not None:
            scopes[ancestor.variable] = collection.element
    return scopes


def _is_model_expression_root(node: ConditionNode) -> bool:
    """Return whether the compiler types the fields read below ``node``. Only a model
    expression reads ``join()`` and a map comprehension, so the compiler also types their
    fields, and it rejects every other use of them with a dedicated message."""

    return isinstance(node, ModelCall | MapComprehension) or (
        isinstance(node, FunctionCall) and node.name == "join"
    )


def _function_parameter_owns_field_validation(
    node: FieldAccess,
    ancestors: tuple[ConditionNode, ...],
) -> bool:
    """Return whether a catalogued function accepts this path as model input.

    Legacy inference functions represent model-input selectors as ordinary
    ``FunctionCall`` arguments. Their compiler validates selector pipelines such as
    ``messages.user_messages``. Applying object-field validation to the same path would
    reject valid syntax before it reaches that compiler check.
    """

    for ancestor in reversed(ancestors):
        if not isinstance(ancestor, FunctionCall):
            continue
        item = function_by_name(ancestor.name)
        if item is None:
            continue
        parameters = item["parameters"]
        for index, argument in enumerate(ancestor.args):
            if index >= len(parameters):
                break
            if "model_input" not in parameters[index]["accepted_types"]:
                continue
            if _is_model_input_selector_field(argument, node):
                return True
    return False


def _is_model_input_selector_field(argument: ConditionNode, node: FieldAccess) -> bool:
    """Return whether ``node`` is part of the selector itself, including a slice base."""

    if argument is node:
        return True
    return bool(
        isinstance(argument, FunctionCall)
        and argument.name == "_slice"
        and argument.args
        and _is_model_input_selector_field(argument.args[0], node)
    )


def _unexpected_keys(value: dict, allowed: set[str] | frozenset[str]) -> list[str]:
    """Return keys outside the accepted OpenAI conversation shape."""

    return sorted(set(value) - set(allowed))


def _validate_json_arguments(
    arguments: str,
    message_index: int,
    call_index: int,
    max_nesting_depth: int = 100,
) -> None:
    """Validate a tool-call argument string after raw-size preflight."""

    try:
        parsed = orjson.loads(arguments)
    except (ValueError, RecursionError) as exc:
        raise ConversationValidationError(
            f"Assistant message at index {message_index} has malformed JSON in "
            f"tool_calls[{call_index}].function.arguments: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ConversationValidationError(
            f"Assistant message at index {message_index} has non-object JSON in "
            f"tool_calls[{call_index}].function.arguments; expected a JSON object"
        )
    # Bound the decoded argument tree's nesting to the same limit the lazy ToolCall.arguments
    # decoder enforces, so an eager (parse_arguments=True) validation cannot accept arguments
    # deeper than the caller's configured max_nesting_depth.
    _validate_json_tree(
        parsed,
        f"messages[{message_index}].tool_calls[{call_index}].function.arguments",
        max_nesting_depth,
    )


def _validate_json_tree(value: object, path: str, max_nesting_depth: int) -> None:
    """Validate a JSON tree and bound container depth from its value root."""

    stack: list[tuple[object, int, bool]] = [(value, 1, False)]
    active_containers: set[int] = set()
    while stack:
        current, depth, exiting = stack.pop()
        if exiting:
            active_containers.remove(id(current))
            continue
        if current is None or isinstance(current, str | bool | int):
            continue
        if isinstance(current, float):
            if isfinite(current):
                continue
            raise ConversationValidationError(f"{path} contains a non-finite number")
        if isinstance(current, list):
            current_id = id(current)
            if current_id in active_containers:
                raise ConversationValidationError(f"{path} contains a cyclic reference")
            if depth > max_nesting_depth:
                raise ConversationValidationError(
                    f"{path} exceeds max_nesting_depth {max_nesting_depth}"
                )
            active_containers.add(current_id)
            stack.append((current, depth, True))
            stack.extend((child, depth + 1, False) for child in current)
            continue
        if isinstance(current, dict):
            if not all(isinstance(key, str) for key in current):
                raise ConversationValidationError(f"{path} contains a non-string object key")
            current_id = id(current)
            if current_id in active_containers:
                raise ConversationValidationError(f"{path} contains a cyclic reference")
            if depth > max_nesting_depth:
                raise ConversationValidationError(
                    f"{path} exceeds max_nesting_depth {max_nesting_depth}"
                )
            active_containers.add(current_id)
            stack.append((current, depth, True))
            stack.extend((child, depth + 1, False) for child in current.values())
            continue
        raise ConversationValidationError(
            f"{path} contains unsupported JSON value {type(current).__name__}"
        )


def _validate_content_part(part: object, message_index: int, part_index: int) -> None:
    """Validate one supported OpenAI multimodal content part."""
    if not isinstance(part, dict):
        raise ConversationValidationError(
            f"Message at index {message_index} content[{part_index}] must be an object"
        )
    part = cast(dict[str, Any], part)

    part_type = part.get("type")
    if part_type == "text":
        unexpected = _unexpected_keys(part, OPENAI_TEXT_PART_KEYS)
        if unexpected or not isinstance(part.get("text"), str):
            raise ConversationValidationError(
                f"Message at index {message_index} content[{part_index}] must be an OpenAI "
                "text part with string 'text'"
            )
        return

    if part_type == "image_url":
        image_url = part.get("image_url")
        if (
            _unexpected_keys(part, OPENAI_IMAGE_PART_KEYS)
            or not isinstance(image_url, dict)
            or _unexpected_keys(image_url, OPENAI_IMAGE_URL_KEYS)
            or not isinstance(image_url.get("url"), str)
            or ("detail" in image_url and image_url["detail"] not in {"auto", "low", "high"})
        ):
            raise ConversationValidationError(
                f"Message at index {message_index} content[{part_index}] must be an OpenAI "
                "image_url part with a string URL"
            )
        return

    if part_type == "input_audio":
        input_audio = part.get("input_audio")
        if (
            _unexpected_keys(part, OPENAI_AUDIO_PART_KEYS)
            or not isinstance(input_audio, dict)
            or _unexpected_keys(input_audio, OPENAI_AUDIO_KEYS)
            or not isinstance(input_audio.get("data"), str)
            or input_audio.get("format") not in {"wav", "mp3"}
        ):
            raise ConversationValidationError(
                f"Message at index {message_index} content[{part_index}] must be an OpenAI "
                "input_audio part"
            )
        return

    raise ConversationValidationError(
        f"Message at index {message_index} content[{part_index}] has unsupported OpenAI "
        f"content type '{part_type}'"
    )


def _validate_message_content(
    content: object, message_index: int, max_nesting_depth: int = 100, *, allow_null: bool = False
) -> None:
    """Validate string or ordered OpenAI content-part-array message content."""
    if content is None and allow_null:
        return
    if isinstance(content, str):
        return
    if isinstance(content, list):
        if not content:
            raise ConversationValidationError(
                f"Message at index {message_index} content-part arrays must not be empty"
            )
        for part_index, part in enumerate(content):
            _validate_content_part(part, message_index, part_index)
        # Bound nesting depth of the whole content-part array (JSON tree).
        _validate_json_tree(content, f"messages[{message_index}].content", max_nesting_depth)
        return
    raise ConversationValidationError(
        f"Message at index {message_index} content must be a string or OpenAI content-part list"
    )


def validate_conversation_arguments(messages: Messages) -> None:
    """Validate every canonical tool call after the scanner's raw-size preflight.

    Touching ``ToolCall.arguments`` uses the same decoder as rule evaluation.
    Remember raw strings to skip repeated validation, then discard decoded
    trees so a large name-only scan does not retain them until rule evaluation.
    """

    validated: set[str] = set()
    for message in messages.assistant_messages:
        for call_index, call in enumerate(message.tool_calls):
            # The inspection deadline covers this eager pass too. With safety
            # limits enabled, max_tool_argument_size bounds one decode; check
            # between calls so a collection cannot outlive the request budget.
            raise_if_evaluation_expired()
            arguments = call.raw_arguments
            # The wire contract accepts str subclasses. Avoid their custom
            # hash/equality methods when looking up repeated arguments.
            cacheable = type(arguments) is str
            if cacheable and arguments in validated:
                call._discard_decoded_arguments()
                raise_if_evaluation_expired()
                continue
            try:
                decoded = call.arguments
                del decoded
            except EvaluationError as exc:
                raise_if_evaluation_expired()
                raise ConversationValidationError(
                    f"Assistant message at index {call.message_index} has invalid JSON in "
                    f"tool_calls[{call_index}].function.arguments: {exc}"
                ) from exc
            finally:
                call._discard_decoded_arguments()
            raise_if_evaluation_expired()
            if cacheable:
                validated.add(arguments)


def validate_conversation(
    data: dict,
    *,
    parse_arguments: bool = True,
    allowed_roles: set[str] | frozenset[str] | None = None,
    max_nesting_depth: int = 100,
) -> None:
    """Validate the strict OpenAI-compatible conversation contract.

    ``parse_arguments=False`` is used by the scanner before safety-limit
    preflight. It validates the argument field's wire type without parsing a
    potentially oversized blob; the scanner validates every argument object
    after limits have been applied. ``max_nesting_depth`` bounds the JSON
    nesting of message content (and, when ``parse_arguments`` is set, tool-call
    arguments) — the same limit the lazy ``ToolCall.arguments`` decoder enforces.
    """

    if (
        isinstance(max_nesting_depth, bool)
        or not isinstance(max_nesting_depth, int)
        or not 1 <= max_nesting_depth <= 100
    ):
        raise ConversationValidationError("max_nesting_depth must be an integer from 1 to 100")

    if allowed_roles is None:
        allowed_roles = OPENAI_MESSAGE_ROLES

    if not isinstance(data, dict):
        raise ConversationValidationError(
            f"Expected dict, got {type(data).__name__}. "
            "Conversation must be a JSON object with 'messages' key."
        )

    unexpected = _unexpected_keys(data, OPENAI_TOP_LEVEL_KEYS)
    if unexpected:
        raise ConversationValidationError(
            f"Conversation contains unsupported top-level keys: {unexpected}. "
            "Expected only 'messages' and optional 'tools'."
        )

    if "messages" not in data:
        available_keys = list(data.keys())
        raise ConversationValidationError(
            f"Missing required 'messages' key. "
            f"Available keys: {available_keys}. "
            "Did you mean to wrap your data in {'messages': [...]}?"
        )

    messages = data["messages"]
    if not isinstance(messages, list):
        raise ConversationValidationError(
            f"'messages' must be a list, got {type(messages).__name__}"
        )

    if not messages:
        raise ConversationValidationError("'messages' list is empty. Nothing to scan.")

    seen_call_ids: set[str] = set()

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ConversationValidationError(
                f"Message at index {i} is not a dict (got {type(msg).__name__}). "
                "Each message must be an object with 'role' and 'content' keys."
            )

        if "role" not in msg:
            raise ConversationValidationError(
                f"Message at index {i} missing 'role' key. Available keys: {list(msg.keys())}"
            )

        role = msg["role"]
        if not isinstance(role, str) or role not in allowed_roles:
            raise ConversationValidationError(
                f"Message at index {i} has invalid role '{role}'. Valid roles: "
                f"{sorted(allowed_roles)}"
            )

        allowed_keys = (
            OPENAI_TOOL_RESPONSE_KEYS
            if role == "tool"
            else frozenset({"role", "content", "tool_calls", "refusal"})
            if role == "assistant"
            else frozenset({"role", "content"})
        )
        unexpected = _unexpected_keys(msg, allowed_keys)
        if unexpected:
            raise ConversationValidationError(
                f"Message at index {i} has unsupported keys: {unexpected}"
            )

        if role == "tool":
            if "content" not in msg or not isinstance(msg["content"], str):
                raise ConversationValidationError(
                    f"Tool message at index {i} must have string 'content'"
                )
            if not isinstance(msg.get("tool_call_id"), str) or not msg["tool_call_id"]:
                raise ConversationValidationError(
                    f"Tool message at index {i} must have a non-empty string 'tool_call_id'"
                )
        elif "content" not in msg and "tool_calls" not in msg and "refusal" not in msg:
            raise ConversationValidationError(
                f"Message at index {i} (role={role}) missing both 'content' and 'tool_calls'. "
                "Messages must have at least one of these fields."
            )

        if role in {"system", "developer", "user"} and "content" not in msg:
            raise ConversationValidationError(
                f"{role.title()} message at index {i} must have 'content'"
            )
        if role in {"system", "developer", "user"}:
            _validate_message_content(msg["content"], i, max_nesting_depth)

        if role == "assistant":
            if (
                "refusal" in msg
                and msg["refusal"] is not None
                and not isinstance(msg["refusal"], str)
            ):
                raise ConversationValidationError(
                    f"Assistant message at index {i} 'refusal' must be a string or null"
                )
            if "content" in msg:
                _validate_message_content(msg["content"], i, max_nesting_depth, allow_null=True)
            elif "tool_calls" not in msg and "refusal" not in msg:
                raise ConversationValidationError(
                    f"Assistant message at index {i} must have 'content', 'tool_calls', "
                    "or 'refusal'"
                )
            if "tool_calls" in msg:
                tool_calls = msg["tool_calls"]
                if not isinstance(tool_calls, list) or not tool_calls:
                    raise ConversationValidationError(
                        f"Assistant message at index {i} 'tool_calls' must be a non-empty list"
                    )
                for call_index, tool_call in enumerate(tool_calls):
                    _validate_tool_call(
                        tool_call,
                        message_index=i,
                        call_index=call_index,
                        seen_call_ids=seen_call_ids,
                        parse_arguments=parse_arguments,
                        max_nesting_depth=max_nesting_depth,
                    )

    # Validate tool responses after calls are seen so references are unambiguous.
    call_message_indices = {
        tool_call["id"]: message_index
        for message_index, message in enumerate(messages)
        if message["role"] == "assistant"
        for tool_call in message.get("tool_calls", [])
    }
    seen_response_ids: set[str] = set()
    for i, msg in enumerate(messages):
        if msg["role"] != "tool":
            continue
        tool_call_id = msg["tool_call_id"]
        if tool_call_id not in seen_call_ids:
            raise ConversationValidationError(
                f"Tool message at index {i} references unknown tool_call_id '{tool_call_id}'"
            )
        if call_message_indices[tool_call_id] >= i:
            raise ConversationValidationError(
                f"Tool message at index {i} references tool_call_id '{tool_call_id}' "
                "before its assistant tool call"
            )
        if tool_call_id in seen_response_ids:
            raise ConversationValidationError(
                f"Tool message at index {i} duplicates response for tool_call_id '{tool_call_id}'"
            )
        seen_response_ids.add(tool_call_id)

    if "tools" in data:
        tools = data["tools"]
        if not isinstance(tools, list):
            raise ConversationValidationError(f"'tools' must be a list, got {type(tools).__name__}")

        tool_names: set[str] = set()
        for i, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise ConversationValidationError(f"Tool at index {i} is not a dict")

            if _unexpected_keys(tool, OPENAI_TOOL_KEYS):
                raise ConversationValidationError(
                    f"Tool at index {i} has unsupported keys: "
                    f"{_unexpected_keys(tool, OPENAI_TOOL_KEYS)}. Expected OpenAI tool fields."
                )
            if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
                raise ConversationValidationError(
                    f"Tool at index {i} has invalid format. "
                    "Expected OpenAI format with type='function' and a function object. "
                    f"Got keys: {list(tool.keys())}"
                )
            function = tool["function"]
            unexpected_function_keys = _unexpected_keys(function, OPENAI_FUNCTION_KEYS)
            if unexpected_function_keys:
                raise ConversationValidationError(
                    f"Tool at index {i} function has unsupported keys: {unexpected_function_keys}"
                )
            if not isinstance(function.get("name"), str) or not function["name"]:
                raise ConversationValidationError(
                    f"Tool at index {i} function must have a non-empty string 'name'"
                )
            if "description" in function and not isinstance(function["description"], str):
                raise ConversationValidationError(
                    f"Tool at index {i} function 'description' must be a string"
                )
            if "parameters" in function and not isinstance(function["parameters"], dict):
                raise ConversationValidationError(
                    f"Tool at index {i} function 'parameters' must be an object"
                )

            tool_name = function["name"]
            if tool_name in tool_names:
                raise ConversationValidationError(
                    f"Tool definitions contain duplicate function name '{tool_name}'"
                )
            tool_names.add(tool_name)


def _validate_tool_call(
    tool_call: object,
    *,
    message_index: int,
    call_index: int,
    seen_call_ids: set[str],
    parse_arguments: bool,
    max_nesting_depth: int = 100,
) -> None:
    """Validate one OpenAI assistant tool call without parsing its arguments."""

    if not isinstance(tool_call, dict):
        raise ConversationValidationError(
            f"Assistant message at index {message_index} tool_calls[{call_index}] "
            f"is not an object (got {type(tool_call).__name__})"
        )
    tool_call = cast(dict[str, Any], tool_call)
    unexpected = _unexpected_keys(tool_call, OPENAI_TOOL_CALL_KEYS)
    if unexpected:
        raise ConversationValidationError(
            f"Assistant message at index {message_index} tool_calls[{call_index}] "
            f"has unsupported keys: {unexpected}"
        )
    call_id = tool_call.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise ConversationValidationError(
            f"Assistant message at index {message_index} tool_calls[{call_index}] "
            "must have a non-empty string 'id'"
        )
    if call_id in seen_call_ids:
        raise ConversationValidationError(
            f"Duplicate tool-call id '{call_id}' at assistant message index {message_index}"
        )
    if tool_call.get("type") != "function" or not isinstance(tool_call.get("function"), dict):
        raise ConversationValidationError(
            f"Assistant message at index {message_index} tool_calls[{call_index}] "
            "must use type='function' with a function object"
        )
    function = tool_call["function"]
    unexpected_function_keys = _unexpected_keys(function, frozenset({"name", "arguments"}))
    if unexpected_function_keys:
        raise ConversationValidationError(
            f"Assistant message at index {message_index} tool_calls[{call_index}].function "
            f"has unsupported keys: {unexpected_function_keys}"
        )
    if not isinstance(function.get("name"), str) or not function["name"]:
        raise ConversationValidationError(
            f"Assistant message at index {message_index} tool_calls[{call_index}] "
            "function must have a non-empty string 'name'"
        )
    if not isinstance(function.get("arguments"), str):
        raise ConversationValidationError(
            f"Assistant message at index {message_index} tool_calls[{call_index}] "
            "function 'arguments' must be a JSON string"
        )

    seen_call_ids.add(call_id)
    if parse_arguments:
        _validate_json_arguments(
            function["arguments"], message_index, call_index, max_nesting_depth
        )


class RuleValidator:
    """Validates rules for performance and correctness issues."""

    def __init__(self):
        self.max_flow_ops = 5
        self.max_nested_comprehensions = 3
        self.max_total_comprehensions = 10

    def validate(self, rule: DSLRule) -> ValidationResult:
        """Validate a rule and return warnings."""
        warnings = []
        regex_errors = []
        semantic_errors = []
        stats = {
            "flow_ops": 0,
            "comprehensions": 0,
            "nested_depth": 0,
            "regex_ops": 0,
            "external_calls": 0,
        }

        if rule.strings:
            regex_errors = self._validate_regex_patterns(rule.strings)
            semantic_errors = self._validate_string_arrays(rule.strings)

        executable_nodes = rule.executable_nodes()
        regex_errors.extend(self._validate_regex_operands(executable_nodes, rule.strings or {}))
        semantic_errors.extend(self._validate_list_operands(executable_nodes))
        for node in executable_nodes:
            self._analyze_node(node, stats, warnings, depth=0)
        if executable_nodes:
            semantic_errors.extend(self._validate_semantics(rule))

        complexity = self._estimate_complexity(stats)

        return ValidationResult(
            rule_name=rule.name,
            warnings=warnings,
            estimated_complexity=complexity,
            regex_errors=regex_errors,
            semantic_errors=semantic_errors,
        )

    def _validate_regex_patterns(self, strings: dict) -> list[RegexError]:
        """Validate all regex patterns in the strings block."""
        errors = []

        for var_name, var_value in strings.items():
            if isinstance(var_value, tuple) and len(var_value) == 2:
                pattern, modifiers = var_value
            else:
                pattern = var_value
                modifiers = None

            if not self._is_regex_pattern(modifiers):
                continue

            patterns_to_check = pattern if isinstance(pattern, list) else [pattern]

            for p in patterns_to_check:
                error = self._validate_single_regex(str(p))
                if error:
                    errors.append(RegexError(variable=var_name, pattern=str(p), error=error))

        return errors

    @staticmethod
    def _validate_string_arrays(strings: dict) -> list[SemanticError]:
        """Reject a strings definition that nests an array, which no operator matches."""
        errors = []
        for var_name, var_value in strings.items():
            value = var_value[0] if isinstance(var_value, tuple) else var_value
            if isinstance(value, list) and any(isinstance(item, list) for item in value):
                errors.append(
                    SemanticError(
                        error_type="nested_string_array",
                        message=(
                            f"String '{var_name}' nests an array inside an array; a strings "
                            "definition holds one string or a flat array of strings"
                        ),
                    )
                )
        return errors

    @staticmethod
    def _validate_list_operands(nodes) -> list[SemanticError]:
        """Reject a literal list after ``in`` or ``contains_any`` that nests a list, which no
        text can match, as the strings block is checked."""
        errors = []
        for node in nodes:
            for visit in walk_condition_visits(node):
                current = visit.node
                if isinstance(current, ContainsAnyOp):
                    operator, values = "contains_any", current.values
                elif (
                    isinstance(current, BinaryOp)
                    and current.operator == "in"
                    and isinstance(current.right, Literal)
                    and isinstance(current.right.value, list)
                ):
                    operator, values = "in", current.right.value
                else:
                    continue
                if any(isinstance(item, list) for item in values):
                    errors.append(
                        SemanticError(
                            error_type="nested_list_operand",
                            message=(
                                f"the list after `{operator}` nests a list inside a list; "
                                "write one flat list of strings"
                            ),
                        )
                    )
        return errors

    def _validate_regex_operands(self, nodes, strings: dict) -> list[RegexError]:
        """Validate every pattern on the right of ``~``, as the strings block is, so a pattern
        RE2 cannot compile fails the rule at load instead of every scan.

        The operand is a literal pattern or a ``strings:`` name. ``~`` compiles a named
        string's own patterns as a regex, with or without the ``regex`` or ``base64``
        modifiers, so they are checked too unless the strings block already checked them."""
        errors = []
        checked_names: set[str] = set()
        for node in nodes:
            for visit in walk_condition_visits(node):
                current = visit.node
                if not (isinstance(current, BinaryOp) and current.operator == "~"):
                    continue
                operand = current.right
                if isinstance(operand, Literal) and isinstance(operand.value, str):
                    error = self._validate_single_regex(operand.value)
                    if error:
                        errors.append(
                            RegexError(variable="~ operand", pattern=operand.value, error=error)
                        )
                    continue
                name = operand.name if isinstance(operand, Identifier) else None
                if name is None or name not in strings or name in checked_names:
                    continue
                checked_names.add(name)
                value = strings[name]
                pattern, modifiers = value if isinstance(value, tuple) else (value, None)
                if self._is_regex_pattern(modifiers):
                    continue
                for item in pattern if isinstance(pattern, list) else [pattern]:
                    error = self._validate_single_regex(str(item))
                    if error:
                        errors.append(RegexError(variable=name, pattern=str(item), error=error))
        return errors

    def _is_regex_pattern(self, modifiers) -> bool:
        """Check if pattern uses regex modifier."""
        if modifiers is None:
            return False
        if isinstance(modifiers, StringModifiers):
            return modifiers.regex or modifiers.fullword
        return False

    def _validate_single_regex(self, pattern: str) -> str | None:
        """Validate a single regex pattern. Returns error message or None if valid."""
        try:
            regex_cache.compile(pattern)
            return None
        except (re.error, ValueError, OverflowError, RecursionError) as e:
            return str(e)

    def _analyze_node(
        self,
        node: ConditionNode,
        stats: dict,
        warnings: list[ValidationWarning],
        depth: int,
    ) -> None:
        """Analyze a complete condition AST using the shared typed visitor."""

        comprehension_types = (
            AnyComprehension,
            AllComprehension,
            FilterComprehension,
            MapComprehension,
        )

        for visit in walk_condition_visits(node):
            current = visit.node

            if isinstance(current, FlowOp):
                stats["flow_ops"] += 1
                if stats["flow_ops"] > self.max_flow_ops:
                    warnings.append(
                        ValidationWarning(
                            severity=Severity.HIGH,
                            message=(
                                f"Rule contains {stats['flow_ops']} flow operators "
                                "(after/right_after)"
                            ),
                            suggestion=(
                                "Consider consolidating flow checks or splitting into multiple rules"
                            ),
                        )
                    )

                if current.operator == "after" and current.max_distance is None:
                    warnings.append(
                        ValidationWarning(
                            severity=Severity.MEDIUM,
                            message="Unbounded 'after' operator detected",
                            suggestion="Use 'after(within: N)' to limit timeline scan distance",
                        )
                    )

            if isinstance(current, comprehension_types):
                comprehension_depth = depth + sum(
                    isinstance(ancestor, comprehension_types) for ancestor in visit.ancestors
                )
                stats["comprehensions"] += 1
                stats["nested_depth"] = max(stats["nested_depth"], comprehension_depth)

                if stats["comprehensions"] > self.max_total_comprehensions:
                    warnings.append(
                        ValidationWarning(
                            severity=Severity.HIGH,
                            message=f"Rule contains {stats['comprehensions']} comprehensions",
                            suggestion="Break complex rules into smaller, focused rules",
                        )
                    )

                if comprehension_depth > self.max_nested_comprehensions:
                    warnings.append(
                        ValidationWarning(
                            severity=Severity.HIGH,
                            message=(
                                f"Nested comprehension depth {comprehension_depth} "
                                "exceeds recommended limit"
                            ),
                            suggestion=(
                                "Flatten nested comprehensions or use intermediate variables"
                            ),
                        )
                    )

            if isinstance(current, BinaryOp) and current.operator == "~":
                stats["regex_ops"] += 1

            if isinstance(current, ModelCall) or (
                isinstance(current, FunctionCall) and current.name in external_function_costs()
            ):
                stats["external_calls"] += 1

    def _estimate_complexity(self, stats: dict) -> Complexity:
        """Estimate rule complexity from statistics."""
        score = 0

        score += stats["flow_ops"] * 3
        score += stats["comprehensions"] * 2
        score += stats["nested_depth"] * 2
        score += stats["regex_ops"] * 1
        score += stats["external_calls"] * 2

        if score <= 5:
            return Complexity.LOW
        if score <= 15:
            return Complexity.MEDIUM
        if score <= 30:
            return Complexity.HIGH
        return Complexity.EXTREME

    def _validate_semantics(self, rule: DSLRule) -> list[SemanticError]:
        """Validate semantic correctness: variable references and function arities."""
        errors: list[SemanticError] = []
        defined_vars = set(rule.strings.keys()) if rule.strings else set()

        for node in rule.executable_nodes():
            self._check_node_semantics(node, defined_vars, errors)

        return errors

    def _check_node_semantics(
        self,
        node: ConditionNode,
        defined_vars: set[str],
        errors: list[SemanticError],
    ) -> None:
        """Check node-local semantics across the complete typed condition AST."""

        for visit in walk_condition_visits(node):
            current = visit.node

            if isinstance(current, FunctionCall):
                self._check_function_arity(current, errors)
                self._check_function_arguments(current, errors)
                self._check_text_ref_name(current, errors)

            if isinstance(current, Identifier) and current.name.startswith("$"):
                self._check_variable_reference(current.name, defined_vars, errors)

            if isinstance(current, StringPosition):
                self._check_string_var_reference(current.variable, defined_vars, errors)

            if isinstance(current, StringCount):
                self._check_string_var_reference(current.variable, defined_vars, errors)

            if isinstance(current, StringSetQuantifier):
                for pattern in current.patterns:
                    self._check_variable_reference(pattern, defined_vars, errors)

            # Typed model expressions perform their own field validation in the compiler.
            # Ordinary conversation fields use the same generated surface records that power
            # the public catalog, so documentation and compile-time acceptance cannot diverge.
            if (
                isinstance(current, FieldAccess)
                and not any(_is_model_expression_root(ancestor) for ancestor in visit.ancestors)
                and not _function_parameter_owns_field_validation(current, visit.ancestors)
            ):
                self._check_field_access(
                    current,
                    _scopes_from_ancestors(visit.ancestors),
                    errors,
                )

            if isinstance(current, PatternMatch):
                self._check_pattern_fields(current, visit.ancestors, errors)

    @staticmethod
    def _append_field_error(
        field_name: str,
        owner: _ValueShape,
        path: str,
        errors: list[SemanticError],
    ) -> None:
        if owner.fields is None:
            message = f"Field '{field_name}' cannot be read from {owner.label}"
        else:
            available = ", ".join(sorted(owner.fields))
            message = (
                f"Unknown field '{field_name}' on {owner.label}. Available fields: {available}"
            )
        errors.append(
            SemanticError(
                error_type="unknown_field",
                message=message,
                location=path,
            )
        )

    def _check_field_access(
        self,
        node: FieldAccess,
        scopes: dict[str, _ValueShape],
        errors: list[SemanticError],
    ) -> None:
        shape, fields = _field_root(node, scopes)
        if shape is None or not fields:
            return
        _, failure = _follow_fields(shape, fields)
        if failure is None:
            return
        field_name, owner = failure
        self._append_field_error(field_name, owner, ".".join(fields), errors)

    def _check_pattern_fields(
        self,
        pattern: PatternMatch,
        ancestors: tuple[ConditionNode, ...],
        errors: list[SemanticError],
    ) -> None:
        if not ancestors or not isinstance(ancestors[-1], BinaryOp):
            return
        parent = ancestors[-1]
        if parent.operator != "matches" or parent.right is not pattern:
            return
        scopes = _scopes_from_ancestors(ancestors)
        subject = _expression_shape(parent.left, scopes)
        if subject is None:
            return
        for predicate in pattern.patterns:
            if not isinstance(predicate, BinaryOp):
                continue
            field = predicate.left
            if isinstance(field, Identifier):
                fields = [field.name]
            elif isinstance(field, FieldAccess) and field.base is None:
                fields = [str(name) for name in field.fields]
            else:
                continue
            _, failure = _follow_fields(subject, fields)
            if failure is None:
                continue
            field_name, owner = failure
            self._append_field_error(field_name, owner, ".".join(fields), errors)

    def _check_string_var_reference(
        self, var_name: str, defined_vars: set[str], errors: list[SemanticError]
    ) -> None:
        """Check a string variable reference from StringPosition/StringCount nodes."""
        full_name = f"${var_name}" if not var_name.startswith("$") else var_name
        self._check_variable_reference(full_name, defined_vars, errors)

    def _check_variable_reference(
        self, var_name: str, defined_vars: set[str], errors: list[SemanticError]
    ) -> None:
        """Check that a variable reference resolves to a defined variable."""
        if var_name.endswith("*"):
            prefix = var_name[:-1]
            if not any(v.startswith(prefix) for v in defined_vars):
                errors.append(
                    SemanticError(
                        error_type="undefined_variable",
                        message=f"No variables match wildcard pattern '{var_name}'. "
                        f"Defined variables: {sorted(defined_vars) if defined_vars else 'none'}",
                    )
                )
        else:
            if var_name not in defined_vars:
                errors.append(
                    SemanticError(
                        error_type="undefined_variable",
                        message=f"Undefined variable '{var_name}'. "
                        f"Defined variables: {sorted(defined_vars) if defined_vars else 'none'}",
                    )
                )

    def _check_text_ref_name(self, node: FunctionCall, errors: list[SemanticError]) -> None:
        """Require a literal name in text_ref(), so the reference is checkable when rules load.

        A computed name cannot be looked up against the config until the rule runs, and by then a
        name that resolves to nothing is a scan against an empty policy on live traffic.
        """
        if node.name != TEXT_REF_FUNCTION or len(node.args) != 1:
            return

        argument = node.args[0]
        if (
            isinstance(argument, Literal)
            and isinstance(argument.value, str)
            and argument.value.strip()
        ):
            return

        errors.append(
            SemanticError(
                error_type="dynamic_text_ref",
                message=(
                    f"{TEXT_REF_FUNCTION}() requires a literal resource name, so that a missing "
                    f"name is reported when rules load rather than per scan on live traffic"
                ),
            )
        )

    def _check_function_arity(self, node: FunctionCall, errors: list[SemanticError]) -> None:
        """Check that a function is called with the correct number of arguments."""
        func_name = node.name
        num_args = len(node.args)

        if func_name.startswith("_"):
            return

        signatures = function_signatures()
        if func_name not in signatures:
            return

        min_args, max_args = signatures[func_name]

        if num_args < min_args:
            errors.append(
                SemanticError(
                    error_type="function_arity",
                    message=f"Function '{func_name}' requires at least {min_args} argument(s), "
                    f"but was called with {num_args}",
                )
            )
        elif max_args is not None and num_args > max_args:
            errors.append(
                SemanticError(
                    error_type="function_arity",
                    message=f"Function '{func_name}' accepts at most {max_args} argument(s), "
                    f"but was called with {num_args}",
                )
            )

    def _check_function_arguments(self, node: FunctionCall, errors: list[SemanticError]) -> None:
        """Apply catalog constraints that can be decided from the authored argument nodes."""

        item = function_by_name(node.name)
        if item is None:
            return
        parameters = item["parameters"]
        implicit = int(item.get("implicit_receiver_count", 0))
        public_args = node.args[implicit:]
        variadic = next((parameter for parameter in parameters if parameter["variadic"]), None)

        for index, argument in enumerate(public_args):
            parameter = parameters[index] if index < len(parameters) else variadic
            if parameter is None:
                break
            accepted_types = set(parameter["accepted_types"])
            is_text_reference = (
                "text_reference" in accepted_types
                and isinstance(argument, FunctionCall)
                and argument.name == TEXT_REF_FUNCTION
            )
            if (
                parameter["literal_only"]
                and not isinstance(argument, Literal)
                and not is_text_reference
            ):
                errors.append(
                    SemanticError(
                        error_type="function_argument",
                        message=(
                            f"Function '{node.name}' argument '{parameter['name']}' must be "
                            "written as a literal value"
                        ),
                    )
                )
                continue
            if parameter["field_only"] and not isinstance(argument, Identifier | FieldAccess):
                errors.append(
                    SemanticError(
                        error_type="function_argument",
                        message=(
                            f"Function '{node.name}' argument '{parameter['name']}' must be "
                            "a conversation field or surface"
                        ),
                    )
                )
                continue
            if not isinstance(argument, Literal):
                continue
            value = argument.value
            literal_type = (
                "null"
                if value is None
                else "boolean"
                if isinstance(value, bool)
                else "integer"
                if isinstance(value, int)
                else "number"
                if isinstance(value, float)
                else "string"
                if isinstance(value, str)
                else None
            )
            type_matches = literal_type in accepted_types or (
                literal_type == "integer" and "number" in accepted_types
            )
            if not type_matches:
                errors.append(
                    SemanticError(
                        error_type="function_argument",
                        message=(
                            f"Function '{node.name}' argument '{parameter['name']}' accepts "
                            f"{sorted(accepted_types)}, got {literal_type}"
                        ),
                    )
                )
            allowed_values = parameter["allowed_values"]
            if allowed_values and value not in allowed_values:
                errors.append(
                    SemanticError(
                        error_type="function_argument",
                        message=(
                            f"Function '{node.name}' argument '{parameter['name']}' must be one "
                            f"of {allowed_values}, got {value!r}"
                        ),
                    )
                )
            minimum = parameter["minimum"]
            maximum = parameter["maximum"]
            if minimum is not None and (
                isinstance(value, bool) or not isinstance(value, int | float) or value < minimum
            ):
                errors.append(
                    SemanticError(
                        error_type="function_argument",
                        message=(
                            f"Function '{node.name}' argument '{parameter['name']}' must be at "
                            f"least {minimum}"
                        ),
                    )
                )
            if maximum is not None and (
                isinstance(value, bool) or not isinstance(value, int | float) or value > maximum
            ):
                errors.append(
                    SemanticError(
                        error_type="function_argument",
                        message=(
                            f"Function '{node.name}' argument '{parameter['name']}' must be at "
                            f"most {maximum}"
                        ),
                    )
                )


def validate_rules(rules: dict) -> dict:
    """Validate multiple rules and return results."""
    validator = RuleValidator()
    results = {}

    for rule_name, rule in rules.items():
        results[rule_name] = validator.validate(rule)

    return results
