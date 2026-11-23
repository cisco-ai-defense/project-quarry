# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Strict bounded native snapshots for whole-request local token counts."""

from math import isfinite

from quarry.integrations.local_tokenizer_worker import _Job
from quarry.types.exceptions import LocalTokenizerInvalidInputError, LocalTokenizerLimitError
from quarry.types.local_tokenizer import LocalTokenizerConfig


def _invalid():
    raise LocalTokenizerInvalidInputError(
        "Local tokenizer input is unsupported or invalid", {"code": "local_tokenizer_input"}
    )


def _fields(value, allowed, required=None):
    return (
        type(value) is dict
        and len(value) <= len(allowed)
        and all(type(key) is str for key in value)
        and (allowed if required is None else required) <= set(value) <= allowed
    )


def snapshot_request(messages, tools, config: LocalTokenizerConfig, job: _Job) -> tuple[list, list]:
    """Detach exactly the supported wire fields; never coerce host objects/media."""
    limits = config.limits
    nodes = characters = byte_count = 0

    def clone(value, depth=0):
        nonlocal nodes, characters, byte_count
        job.checkpoint()
        nodes += 1
        if nodes > 32_768 or depth > 16:
            raise LocalTokenizerLimitError(
                "Local tokenizer input work limit was reached", {"code": "local_tokenizer_limit"}
            )
        kind = type(value)
        if value is None or kind is bool:
            return value
        if kind in (int, float):
            if kind is int and value.bit_length() > 63 or kind is float and not isfinite(value):
                _invalid()
            return value
        if kind is str:
            characters += len(value)
            if characters > limits.max_input_characters:
                raise LocalTokenizerLimitError(
                    "Local tokenizer input size limit was reached",
                    {"code": "local_tokenizer_limit"},
                )
            try:
                byte_count += len(value.encode("utf-8"))
            except UnicodeError:
                _invalid()
            if byte_count > limits.max_input_bytes:
                raise LocalTokenizerLimitError(
                    "Local tokenizer input byte limit was reached",
                    {"code": "local_tokenizer_limit"},
                )
            return value
        if kind in (tuple, list):
            if len(value) > 32_768:
                _invalid()
            return [clone(item, depth + 1) for item in value]
        if kind is dict:
            if len(value) > 4096 or any(type(key) is not str for key in value):
                _invalid()
            return {clone(key, depth + 1): clone(item, depth + 1) for key, item in value.items()}
        _invalid()

    if type(messages) not in (tuple, list) or len(messages) > limits.max_messages:
        _invalid()
    if type(tools) not in (tuple, list) or len(tools) > limits.max_tools:
        _invalid()
    kept_messages = []
    for message in messages:
        job.checkpoint()
        if type(message) is not dict or not 2 <= len(message) <= 6:
            _invalid()
        if (
            any(type(key) is not str for key in message)
            or not set(message)
            <= {"role", "content", "name", "tool_calls", "tool_call_id", "function_call"}
            or not {"role", "content"} <= set(message)
        ):
            _invalid()
        role, content = message["role"], message["content"]
        if type(role) is not str or role not in {
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
            "function",
        }:
            _invalid()
        if type(content) is not str and not (
            content is None and ("tool_calls" in message or "function_call" in message)
        ):
            # Image/audio/video and opaque content parts have no local text-count shortcut.
            _invalid()
        if "name" in message and (
            type(message["name"]) is not str or not 1 <= len(message["name"]) <= 256
        ):
            _invalid()
        if role == "tool" and (
            type(message.get("tool_call_id")) is not str or not message["tool_call_id"]
        ):
            _invalid()
        if "tool_calls" in message:
            calls = message["tool_calls"]
            if type(calls) not in (tuple, list) or len(calls) > 64:
                _invalid()
            for call in calls:
                if (
                    not _fields(call, {"id", "type", "function"})
                    or type(call["type"]) is not str
                    or call["type"] != "function"
                ):
                    _invalid()
                if type(call["id"]) is not str or not 1 <= len(call["id"]) <= 256:
                    _invalid()
                function = call["function"]
                if not _fields(function, {"name", "arguments"}):
                    _invalid()
                if (
                    type(function["name"]) is not str
                    or not 1 <= len(function["name"]) <= 256
                    or type(function["arguments"]) is not str
                ):
                    _invalid()
        if "function_call" in message:
            function = message["function_call"]
            if not _fields(function, {"name", "arguments"}) or any(
                type(value) is not str for value in function.values()
            ):
                _invalid()
        kept_messages.append(clone(message))
    kept_tools = []
    for tool in tools:
        if (
            not _fields(tool, {"type", "function"})
            or type(tool["type"]) is not str
            or tool["type"] != "function"
        ):
            _invalid()
        function = tool["function"]
        if not _fields(function, {"name", "description", "parameters", "strict"}, {"name"}):
            _invalid()
        if type(function["name"]) is not str or not 1 <= len(function["name"]) <= 256:
            _invalid()
        if "description" in function and type(function["description"]) is not str:
            _invalid()
        if "parameters" in function and type(function["parameters"]) is not dict:
            _invalid()
        if "strict" in function and type(function["strict"]) is not bool:
            _invalid()
        kept_tools.append(clone(tool))
    return kept_messages, kept_tools
