# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""An allowlisted, bounded Jinja AST interpreter; no template Python is executed.

Jinja supplies parsing/whitespace syntax only. Values are native detached request
data; calls, attributes, operators and filters have explicit implementations here.
Full tool serialization must reach output, not merely an unused expression.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from typing import Any, NoReturn

from quarry.integrations.local_tokenizer_worker import _Job
from quarry.types.exceptions import (
    LocalTokenizerInvalidInputError,
    LocalTokenizerLimitError,
    LocalTokenizerUnavailableError,
)
from quarry.types.local_tokenizer import LocalTokenizerConfig

_MISSING = object()
_RESERVED = frozenset({"messages", "tools", "add_generation_prompt", "loop"})
_NODES = frozenset(
    {
        "Template",
        "Output",
        "TemplateData",
        "Const",
        "Name",
        "Getitem",
        "Getattr",
        "Filter",
        "For",
        "If",
        "Assign",
        "Compare",
        "Operand",
        "And",
        "Or",
        "Not",
        "Concat",
        "Add",
        "List",
        "Tuple",
        "Dict",
        "Pair",
        "CondExpr",
        "Slice",
        "Neg",
        "Pos",
        "Call",
    }
)
_FILTERS = frozenset(
    {"tojson", "length", "trim", "lower", "upper", "replace", "join", "string", "default"}
)


def _unsupported() -> NoReturn:
    raise LocalTokenizerUnavailableError(
        "Local chat template syntax is unsupported", {"code": "local_tokenizer_unavailable"}
    )


def _limit() -> NoReturn:
    raise LocalTokenizerLimitError(
        "Local chat template work limit was reached", {"code": "local_tokenizer_limit"}
    )


@dataclass(frozen=True, slots=True)
class _Text:
    text: str
    tools: int = 0
    structured: int = 0


class _Renderer:
    def __init__(self, config: LocalTokenizerConfig, job: _Job, messages: list, tools: list):
        self.config, self.job, self.messages, self.tools = config, job, messages, tools
        self.steps = 0
        self.parts: list[str] = []
        self.characters = 0
        self.tools_rendered = self.structured_rendered = 0
        self.structured_required = sum(
            1 << index
            for index, message in enumerate(messages)
            if set(message) != {"role", "content"} or message["content"] is None
        )

    def step(self):
        self.job.checkpoint()
        self.steps += 1
        if self.steps > self.config.limits.max_template_steps:
            _limit()

    def bounded(self, text: str, tools: int = 0, structured: int = 0) -> _Text:
        if len(text) > self.config.limits.max_output_characters:
            _limit()
        return _Text(text, tools, structured)

    def text(self, value: Any) -> _Text:
        if type(value) is _Text:
            return value
        if type(value) is str:
            return self.bounded(value)
        if value is None or type(value) in (bool, int, float):
            return self.bounded(str(value))
        _unsupported()

    def truth(self, value: Any) -> bool:
        if value is _MISSING:
            return False
        if type(value) is _Text:
            return bool(value.text)
        if value is None or type(value) in (str, bool, int, float, tuple, list, dict, range):
            return bool(value)
        _unsupported()

    def sequence(self, value: Any):
        if type(value) in (str, tuple, list, dict, range):
            return value
        _unsupported()

    def append(self, value):
        part = self.text(value)
        if self.characters + len(part.text) > self.config.limits.max_output_characters:
            _limit()
        self.parts.append(part.text)
        self.characters += len(part.text)
        self.tools_rendered |= part.tools
        self.structured_rendered |= part.structured

    def combine(self, values) -> _Text:
        parts, total, tools, structured = [], 0, 0, 0
        for value in values:
            self.step()
            part = self.text(value)
            total += len(part.text)
            if total > self.config.limits.max_output_characters:
                _limit()
            parts.append(part.text)
            tools |= part.tools
            structured |= part.structured
        return _Text("".join(parts), tools, structured)

    def lookup(self, receiver, key):
        if type(key) is str and key.startswith("_"):
            _unsupported()
        if type(receiver) is dict and type(key) in (str, int):
            return receiver.get(key, _MISSING)
        if type(receiver) in (tuple, list, str) and type(key) in (int, slice):
            try:
                return receiver[key]
            except IndexError:
                return _MISSING
        if type(receiver) is _Text and type(key) in (int, slice):
            try:
                value = receiver.text[key]
            except IndexError:
                return _MISSING
            return self.bounded(
                value,
                receiver.tools if value == receiver.text else 0,
                receiver.structured if value == receiver.text else 0,
            )
        _unsupported()

    def bind(self, target, value, context):
        if type(target).__name__ == "Name":
            if target.name in _RESERVED or target.name.startswith("_"):
                _unsupported()
            context[target.name] = value
            return
        if (
            type(target).__name__ == "Tuple"
            and type(value) in (tuple, list)
            and len(target.items) == len(value)
        ):
            for child, item in zip(target.items, value, strict=True):
                self.bind(child, item, context)
            return
        _unsupported()

    def filter(self, node, context):
        value = self.eval(node.node, context)
        args = [self.eval(arg, context) for arg in node.args]
        if node.kwargs or node.dyn_args is not None or node.dyn_kwargs is not None:
            _unsupported()
        name = node.name
        if name == "default" and len(args) <= 2:
            if (
                value is _MISSING
                or len(args) == 2
                and self.truth(args[1])
                and not self.truth(value)
            ):
                return args[0] if args else ""
            return value
        if name == "length" and not args:
            return len(value.text) if type(value) is _Text else len(self.sequence(value))
        if name == "tojson" and not args:
            # Request validation has already detached and bounded this native graph.
            if value is _MISSING or type(value) is _Text:
                _unsupported()
            tool_mask = message_mask = 0
            pending = [(value, 0)]
            while pending:
                item, depth = pending.pop()
                self.step()
                if depth > 32:
                    _limit()
                tool_mask |= sum(
                    1 << index for index, tool in enumerate(self.tools) if item is tool
                )
                message_mask |= (
                    sum(
                        1 << index for index, message in enumerate(self.messages) if item is message
                    )
                    & self.structured_required
                )
                if type(item) in (tuple, list):
                    if len(item) > self.config.limits.max_template_steps:
                        _limit()
                    pending.extend((child, depth + 1) for child in item)
                elif type(item) is dict:
                    if len(item) > self.config.limits.max_template_steps or any(
                        type(key) not in (str, int) for key in item
                    ):
                        _unsupported()
                    pending.extend((child, depth + 1) for pair in item.items() for child in pair)
                elif item is not None and type(item) not in (str, int, float, bool):
                    _unsupported()
            chunks, total = [], 0
            try:
                # Jinja's default JSON policy, streamed before whole-value allocation.
                encoder = json.JSONEncoder(ensure_ascii=True, allow_nan=False, sort_keys=True)
                for chunk in encoder.iterencode(value):
                    self.step()
                    chunk = (
                        chunk.replace("<", "\\u003c")
                        .replace(">", "\\u003e")
                        .replace("&", "\\u0026")
                        .replace("'", "\\u0027")
                    )
                    total += len(chunk)
                    if total > self.config.limits.max_output_characters:
                        _limit()
                    chunks.append(chunk)
            except (TypeError, ValueError, OverflowError, RecursionError):
                _unsupported()
            encoded = "".join(chunks)
            return self.bounded(encoded, tool_mask, message_mask)
        if name == "join" and len(args) <= 1:
            separator = self.text(args[0] if args else "")
            values = []
            for ordinal, item in enumerate(self.sequence(value)):
                self.step()
                if ordinal:
                    values.append(separator)
                values.append(item)
                if len(values) > self.config.limits.max_template_steps:
                    _limit()
            return self.combine(values)
        before = self.text(value)
        if name == "string" and not args:
            return before
        if name in {"trim", "lower", "upper"} and not args:
            after = {"trim": str.strip, "lower": str.lower, "upper": str.upper}[name](before.text)
            return self.bounded(
                after,
                before.tools if after == before.text else 0,
                before.structured if after == before.text else 0,
            )
        if name == "replace" and 2 <= len(args) <= 3:
            old, new = self.text(args[0]).text, self.text(args[1]).text
            count = args[2] if len(args) == 3 else -1
            if type(count) is not int or not -1 <= count <= self.config.limits.max_template_steps:
                _unsupported()
            occurrences = before.text.count(old)
            if count >= 0:
                occurrences = min(count, occurrences)
            if (
                len(before.text) + occurrences * (len(new) - len(old))
                > self.config.limits.max_output_characters
            ):
                _limit()
            after = before.text.replace(old, new, count)
            return self.bounded(
                after,
                before.tools if after == before.text else 0,
                before.structured if after == before.text else 0,
            )
        _unsupported()

    def call(self, node, context):
        if node.dyn_args is not None or node.dyn_kwargs is not None or node.kwargs:
            _unsupported()
        args = [self.eval(arg, context) for arg in node.args]
        callee = node.node
        if type(callee).__name__ == "Name":
            if callee.name == "raise_exception":
                _unsupported()
            if (
                callee.name == "range"
                and 1 <= len(args) <= 3
                and all(type(value) is int and value.bit_length() <= 63 for value in args)
            ):
                try:
                    value = range(*args)
                    if len(value) > self.config.limits.max_template_steps:
                        _limit()
                except (ValueError, OverflowError):
                    _unsupported()
                return value
        if type(callee).__name__ == "Getattr":
            receiver = self.eval(callee.node, context)
            if type(receiver) is dict:
                if callee.attr == "get" and 1 <= len(args) <= 2 and type(args[0]) in (str, int):
                    return receiver.get(args[0], args[1] if len(args) == 2 else None)
                if callee.attr in {"items", "keys", "values"} and not args:
                    return tuple(
                        {"items": dict.items, "keys": dict.keys, "values": dict.values}[
                            callee.attr
                        ](receiver)
                    )
        _unsupported()

    def eval(self, node, context):
        self.step()
        kind = type(node).__name__
        if kind == "Const":
            value = node.value
            if (
                type(value) is int
                and value.bit_length() > 63
                or type(value) is float
                and not isfinite(value)
            ):
                _unsupported()
            if value is None or type(value) in (str, int, float, bool):
                return value
            _unsupported()
        if kind == "TemplateData":
            return node.data
        if kind == "Name":
            return context.get(node.name, _MISSING)
        if kind in {"Getitem", "Getattr"}:
            return self.lookup(
                self.eval(node.node, context),
                self.eval(node.arg, context) if kind == "Getitem" else node.attr,
            )
        if kind == "Slice":
            parts = [
                self.eval(value, context) if value is not None else None
                for value in (node.start, node.stop, node.step)
            ]
            if (
                any(value is not None and type(value) is not int for value in parts)
                or parts[2] == 0
            ):
                _unsupported()
            return slice(*parts)
        if kind in {"List", "Tuple"}:
            return tuple(self.eval(value, context) for value in node.items)
        if kind == "Dict":
            values = {}
            for pair in node.items:
                key = self.eval(pair.key, context)
                if type(key) not in (str, int):
                    _unsupported()
                values[key] = self.eval(pair.value, context)
            return values
        if kind == "Filter":
            return self.filter(node, context)
        if kind == "Call":
            return self.call(node, context)
        if kind == "Concat":
            return self.combine(self.eval(value, context) for value in node.nodes)
        if kind == "Add":
            left, right = self.eval(node.left, context), self.eval(node.right, context)
            if type(left) is int and type(right) is int:
                result = left + right
                if result.bit_length() > 63:
                    _limit()
                return result
            if type(left) in (str, _Text) and type(right) in (str, _Text):
                return self.combine((left, right))
            if type(left) in (tuple, list) and type(right) in (tuple, list):
                if len(left) + len(right) > self.config.limits.max_template_steps:
                    _limit()
                return (*left, *right)
            _unsupported()
        if kind in {"Pos", "Neg"}:
            value = self.eval(node.node, context)
            if type(value) not in (int, float):
                _unsupported()
            return value if kind == "Pos" else -value
        if kind == "Not":
            return not self.truth(self.eval(node.node, context))
        if kind in {"And", "Or"}:
            left = self.eval(node.left, context)
            return self.eval(node.right, context) if self.truth(left) == (kind == "And") else left
        if kind == "CondExpr":
            chosen = node.expr1 if self.truth(self.eval(node.test, context)) else node.expr2
            return self.eval(chosen, context) if chosen is not None else _MISSING
        if kind == "Compare":
            left = self.eval(node.expr, context)
            for operand in node.ops:
                right = self.eval(operand.expr, context)
                a = left.text if type(left) is _Text else left
                b = right.text if type(right) is _Text else right
                if operand.op in {"in", "notin"}:
                    self.sequence(b)
                    if type(a) not in (str, int, bool, type(None)):
                        _unsupported()
                    if type(b) is str and type(a) is not str:
                        _unsupported()
                    result = a in b
                    if operand.op == "notin":
                        result = not result
                elif (a is None or type(a) in (str, int, float, bool)) and (
                    b is None or type(b) in (str, int, float, bool)
                ):
                    if operand.op == "eq":
                        result = a == b
                    elif operand.op == "ne":
                        result = a != b
                    elif (
                        type(a) in (int, float)
                        and type(b) in (int, float)
                        or type(a) is type(b) is str
                    ):
                        if operand.op == "gt":
                            result = a > b
                        elif operand.op == "gteq":
                            result = a >= b
                        elif operand.op == "lt":
                            result = a < b
                        elif operand.op == "lteq":
                            result = a <= b
                        else:
                            _unsupported()
                    else:
                        _unsupported()
                else:
                    _unsupported()
                if not result:
                    return False
                left = right
            return True
        _unsupported()

    def body(self, body, context):
        for node in body:
            self.step()
            kind = type(node).__name__
            if kind == "Output":
                for value in node.nodes:
                    self.append(self.eval(value, context))
            elif kind == "Assign":
                self.bind(node.target, self.eval(node.node, context), context)
            elif kind == "If":
                chosen = next(
                    (
                        branch.body
                        for branch in (node, *node.elif_)
                        if self.truth(self.eval(branch.test, context))
                    ),
                    node.else_,
                )
                self.body(chosen, context)
            elif kind == "For":
                if node.recursive:
                    _unsupported()
                iterable = self.sequence(self.eval(node.iter, context))
                selected = []
                for value in iterable:
                    self.step()
                    child = dict(context)
                    self.bind(node.target, value, child)
                    if node.test is None or self.truth(self.eval(node.test, child)):
                        selected.append(value)
                for index, value in enumerate(selected):
                    child = dict(context)
                    self.bind(node.target, value, child)
                    child["loop"] = {
                        "index0": index,
                        "index": index + 1,
                        "first": index == 0,
                        "last": index == len(selected) - 1,
                        "length": len(selected),
                    }
                    self.body(node.body, child)
                if not selected:
                    self.body(node.else_, context)
            else:
                _unsupported()


def render_local_request(
    config: LocalTokenizerConfig, messages: list, tools: list, job: _Job
) -> str:
    """Render validated native request snapshots; never load a remote template."""
    if config.template is None:
        _unsupported()
    try:
        from jinja2 import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        _unsupported()
    try:
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=config.trim_blocks, lstrip_blocks=config.lstrip_blocks
        )
        environment.globals.clear()
        for ordinal, _token in enumerate(environment.lex(config.template)):
            job.checkpoint()
            if ordinal >= config.limits.max_template_nodes * 8:
                _limit()
        tree = environment.parse(config.template)
    except (TemplateError, ValueError, RecursionError):
        _unsupported()
    pending: list[tuple[Any, int]] = [(tree, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        job.checkpoint()
        count += 1
        if count > config.limits.max_template_nodes or depth > 32:
            _limit()
        kind = type(node).__name__
        if kind not in _NODES or kind == "Filter" and node.name not in _FILTERS:
            _unsupported()
        if kind in {"Name", "Getattr"} and (node.name if kind == "Name" else node.attr).startswith(
            "_"
        ):
            _unsupported()
        pending.extend((child, depth + 1) for child in node.iter_child_nodes())
    renderer = _Renderer(config, job, messages, tools)
    context = {
        **dict(config.template_variables),
        "messages": messages,
        "tools": tools,
        "add_generation_prompt": config.add_generation_prompt,
    }
    renderer.body(tree.body, context)
    job.checkpoint()
    if renderer.tools_rendered != (1 << len(tools)) - 1:
        raise LocalTokenizerInvalidInputError(
            "Local template did not serialize the offered tools", {"code": "local_tokenizer_input"}
        )
    if renderer.structured_rendered != renderer.structured_required:
        raise LocalTokenizerInvalidInputError(
            "Local template did not serialize structured message fields",
            {"code": "local_tokenizer_input"},
        )
    return "".join(renderer.parts)
