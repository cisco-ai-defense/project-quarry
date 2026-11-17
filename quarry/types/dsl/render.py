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

"""Render parsed DSL conditions back to rule syntax.

The compiler labels a rule's decision-tree nodes with these renderings, and the plugin's
source graph labels its nodes with them, so both read the same way.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

from quarry.dsl.grammar import OperatorPrecedence, operator_precedence
from quarry.types.dsl import (
    MODEL_OPERATION_TYPES,
    AllComprehension,
    AnyComprehension,
    Assignment,
    BinaryOp,
    ConditionalValue,
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
    ParameterRef,
    PatternMatch,
    StringCount,
    StringPosition,
    StringSetQuantifier,
    UnaryOp,
)

MAX_RENDER_NODES = 50_000
MAX_RENDER_DEPTH = 128
MAX_RENDER_OUTPUT = 1_000_000

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_STRING_VARIABLE = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*\*?\Z")
_POSITION_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\*?\Z")
_NON_VALUE_IDENTIFIERS = {"any", "all", "not", "true", "false", "null", "right_after"}
_BINARY_OPERATORS = {
    "or",
    "and",
    "=",
    "!=",
    ">",
    "<",
    ">=",
    "<=",
    "contains",
    "~",
    "matches",
    "in",
    "inside",
    "+",
    "-",
    "*",
}
_PATTERN_OPERATORS = {"=", "!=", ">", "<", ">=", "<=", "contains"}
_FLOW_OPERATORS = {"after", "until"}
_PRECEDENCE_BY_FAMILY = {record.family: record for record in operator_precedence()}
_CONDITIONAL_PRECEDENCE = _PRECEDENCE_BY_FAMILY["conditional value"]
_NOT_PRECEDENCE = _PRECEDENCE_BY_FAMILY["boolean not"]
_WEAK_PRECEDENCE = _PRECEDENCE_BY_FAMILY["text and membership"]
_POSTFIX_PRECEDENCE = _PRECEDENCE_BY_FAMILY["postfix access"]
_PRIMARY_PRECEDENCE = _POSTFIX_PRECEDENCE.precedence + 1
_OPERATOR_FAMILIES = {
    "or": "boolean or",
    "and": "boolean and",
    "after": "flow",
    "until": "flow",
    "=": "comparison",
    "!=": "comparison",
    ">": "comparison",
    "<": "comparison",
    ">=": "comparison",
    "<=": "comparison",
    "contains": "text and membership",
    "~": "text and membership",
    "matches": "text and membership",
    "in": "text and membership",
    "inside": "text and membership",
    "+": "addition",
    "-": "addition",
    "*": "multiplication",
}
_JSON_SHORT_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}
_ESCAPE_CHUNK_SIZE = 1_024
# Strict rational bounds around log2(10), used to reject oversized integers without
# copying or decimal conversion. Only the one-bit boundary case needs a power-of-ten
# comparison, whose size is proportional to the caller's permitted output.
_LOG2_10_LOWER_NUMERATOR = 3_321_928_094_887_362
_LOG2_10_UPPER_NUMERATOR = 3_321_928_094_887_363
_LOG2_10_DENOMINATOR = 1_000_000_000_000_000


class ConditionRenderError(ValueError):
    """A condition cannot be rendered safely as equivalent public DSL syntax."""


def _json_escape_chunks(value: str) -> Iterator[str]:
    """Yield escaped DSL text without building it all at once.

    DSL strings decode Python Unicode escapes. A supplementary character therefore needs
    one eight-digit escape: JSON's surrogate pair would reparse as two separate characters.
    """

    parts: list[str] = []
    size = 0
    for character in value:
        escaped = _JSON_SHORT_ESCAPES.get(character)
        if escaped is None:
            codepoint = ord(character)
            if codepoint < 0x20 or codepoint > 0x7E:
                escaped = f"\\u{codepoint:04x}" if codepoint <= 0xFFFF else f"\\U{codepoint:08x}"
            else:
                escaped = character
        parts.append(escaped)
        size += len(escaped)
        if size >= _ESCAPE_CHUNK_SIZE:
            yield "".join(parts)
            parts.clear()
            size = 0
    if parts:
        yield "".join(parts)


def _integer_text(value: int) -> str:
    """Conversion seam used only after the integer is proven to fit the output budget."""

    return str(value)


@dataclass
class _PreflightFrame:
    """One iterative graph-validation frame."""

    node: ConditionNode
    children: Iterator[ConditionNode]
    incoming_depth: int
    path_depth: int
    height: int = 0


def json_value(value: Any) -> Any:
    """Project a DSL value into a bounded, JSON-compatible representation."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return str(value)


@dataclass(frozen=True)
class _KeywordArgument:
    """Output-only keyword spelling for the public parser's positional internal calls.

    This is never an input AST node. The parser stores extract bounds as positional
    FunctionCall arguments; rendering restores their authored keyword syntax.
    """

    name: str
    value: ConditionNode


class _Renderer:
    """Bounded writer for one condition AST."""

    def __init__(self, *, max_nodes: int, max_depth: int, max_output: int):
        limits = (max_nodes, max_depth, max_output)
        if any(type(limit) is not int or limit <= 0 for limit in limits):
            raise ValueError("condition render limits must be positive non-Boolean integers")
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_output = max_output
        self.output_size = 0
        self.parts: list[str] = []
        self.greedy_tails: set[int] = set()

    def append(self, text: str) -> None:
        if type(text) is not str:
            raise ConditionRenderError("condition renderer received malformed text")
        size = len(text)
        if self.output_size + size > self.max_output:
            raise ConditionRenderError(
                f"condition rendering exceeds {self.max_output} output characters"
            )
        self.parts.append(text)
        self.output_size += size

    def ensure_raw_capacity(self, text: str, *, overhead: int = 0) -> None:
        """Reject raw dynamic text before any scan or escaped copy can scale past the cap."""

        if type(text) is not str:
            raise ConditionRenderError("condition renderer received malformed text")
        if len(text) + overhead > self.max_output - self.output_size:
            raise ConditionRenderError(
                f"condition rendering exceeds {self.max_output} output characters"
            )

    def append_identifier(self, value: object, what: str) -> str:
        if type(value) is not str:
            raise ConditionRenderError(f"{what} is not a valid DSL identifier")
        text = value
        self.ensure_raw_capacity(text)
        if not _IDENTIFIER.fullmatch(text):
            raise ConditionRenderError(f"{what} is not a valid DSL identifier")
        self.append(text)
        return text

    @staticmethod
    def _edge_depth(parent: ConditionNode, child: ConditionNode) -> int:
        """Treat a flattened left Boolean spine as one grammar level."""

        parent = cast(Any, parent)
        child = cast(Any, child)
        if (
            type(parent) is BinaryOp
            and type(parent.operator) is str
            and parent.operator in {"and", "or"}
            and child is parent.left
            and type(child) is BinaryOp
            and type(child.operator) is str
            and child.operator == parent.operator
        ):
            return 0
        return 1

    @staticmethod
    def _node(value: object, what: str) -> ConditionNode:
        if not isinstance(value, ConditionNode):
            raise ConditionRenderError(f"{what} must be a condition node")
        return value

    @staticmethod
    def _node_list(value: object, what: str) -> list[ConditionNode]:
        if type(value) is not list:
            raise ConditionRenderError(f"{what} must be a condition-node list")
        value = cast(list[object], value)
        for index in range(len(value)):
            if not isinstance(value[index], ConditionNode):
                raise ConditionRenderError(f"{what}[{index}] must be a condition node")
        return cast(list[ConditionNode], value)

    @staticmethod
    def _text_list(value: object, what: str) -> list[str]:
        if type(value) is not list:
            raise ConditionRenderError(f"{what} must be a text list")
        value = cast(list[object], value)
        for index in range(len(value)):
            if type(value[index]) is not str:
                raise ConditionRenderError(f"{what}[{index}] must be plain text")
        return cast(list[str], value)

    @staticmethod
    def _plain_text(value: object, what: str) -> str:
        if type(value) is not str:
            raise ConditionRenderError(f"{what} must be plain text")
        return value

    @staticmethod
    def _optional_plain_text(value: object, what: str) -> str | None:
        if value is not None and type(value) is not str:
            raise ConditionRenderError(f"{what} must be plain text or null")
        return value

    def _children(self, node: ConditionNode) -> Iterator[ConditionNode]:
        """Yield supported executable edges without invoking subclass behavior."""

        node = cast(Any, node)
        node_type = type(node)
        if node_type is Literal:
            self._optional_plain_text(node.raw_value, "literal raw spelling")
            return
        if node_type is Identifier:
            self._plain_text(node.name, "identifier")
            return
        if node_type is ParameterRef:
            self._plain_text(node.name, "parameter name")
            return
        if node_type is FieldAccess:
            self._text_list(node.fields, "field path")
            if node.base is not None:
                yield self._node(node.base, "field base")
            return
        if node_type is FunctionCall:
            self._plain_text(node.name, "function name")
            yield from self._node_list(node.args, "function arguments")
            return
        if node_type is ModelCall:
            self._plain_text(node.alias, "model alias")
            self._plain_text(node.operation, "model operation")
            self._plain_text(node.result_type, "model result type")
            yield from self._node_list(node.args, "model arguments")
            return
        if node_type is UnaryOp:
            self._plain_text(node.operator, "unary operator")
            yield self._node(node.operand, "unary operand")
            return
        if node_type is ConditionalValue:
            yield self._node(node.chosen, "conditional chosen value")
            yield self._node(node.predicate, "conditional predicate")
            yield self._node(node.fallback, "conditional fallback value")
            return
        if node_type is FlowOp:
            self._plain_text(node.operator, "flow operator")
            if node.max_distance is not None and type(node.max_distance) is not int:
                raise ConditionRenderError("bounded flow distance must be a plain integer")
            yield self._node(node.left, "flow left operand")
            yield self._node(node.right, "flow right operand")
            return
        if node_type is BinaryOp:
            self._plain_text(node.operator, "binary operator")
            yield self._node(node.left, "binary left operand")
            yield self._node(node.right, "binary right operand")
            return
        if node_type in {AnyComprehension, AllComprehension}:
            self._plain_text(node.variable, "comprehension variable")
            yield self._node(node.collection, "comprehension collection")
            yield self._node(node.condition, "comprehension condition")
            return
        if node_type is FilterComprehension:
            self._plain_text(node.variable, "filter variable")
            self._optional_plain_text(node.output_variable, "filter output variable")
            yield self._node(node.collection, "filter collection")
            yield self._node(node.condition, "filter condition")
            return
        if node_type is MapComprehension:
            self._plain_text(node.variable, "map variable")
            yield self._node(node.expression, "map expression")
            yield self._node(node.collection, "map collection")
            if node.condition is not None:
                yield self._node(node.condition, "map condition")
            return
        if node_type is Assignment:
            self._plain_text(node.variable, "alias")
            yield self._node(node.expression, "alias expression")
            return
        if node_type is PatternMatch:
            yield from self._node_list(node.patterns, "pattern predicates")
            return
        if node_type in {StringPosition, StringCount}:
            self._plain_text(node.variable, "string position/count name")
            self._optional_plain_text(node.text_to_search, "string position/count search text")
            return
        if node_type is StringSetQuantifier:
            if type(node.quantifier) not in {str, int}:
                raise ConditionRenderError("string-set quantifier must be plain text or integer")
            self._text_list(node.patterns, "string-set members")
            self._optional_plain_text(node.text_to_search, "string-set search text")
            return
        if node_type is ContainsAnyOp:
            if type(node.values) is not list:
                raise ConditionRenderError("contains_any values must be a literal list")
            yield self._node(node.left, "contains_any left operand")
            return
        raise ConditionRenderError(f"unsupported condition node {node_type.__name__}")

    def preflight(self, root: ConditionNode) -> None:
        """Validate every graph edge once, including cycles, depth, and shared-DAG work."""

        if not isinstance(root, ConditionNode):
            raise ConditionRenderError(f"expected a condition node, got {type(root).__name__}")
        active = {id(root)}
        heights: dict[int, int] = {}
        work = 1
        try:
            root_children = self._children(root)
        except ConditionRenderError:
            raise
        except (TypeError, ValueError) as exc:
            raise ConditionRenderError("condition node has malformed children") from exc
        stack = [_PreflightFrame(root, root_children, 0, 0)]

        while stack:
            frame = stack[-1]
            try:
                child = next(frame.children)
            except StopIteration:
                child = None
            except ConditionRenderError:
                raise
            except (TypeError, ValueError) as exc:
                raise ConditionRenderError("condition node has malformed children") from exc

            if child is None:
                identity = id(frame.node)
                if frame.height > self.max_depth:
                    raise ConditionRenderError(
                        f"condition rendering exceeds {self.max_depth} nested levels"
                    )
                frame_node = cast(Any, frame.node)
                if type(frame_node) in {AnyComprehension, AllComprehension} or (
                    type(frame_node) is UnaryOp and id(frame_node.operand) in self.greedy_tails
                ):
                    self.greedy_tails.add(identity)
                heights[identity] = frame.height
                active.remove(identity)
                stack.pop()
                if stack:
                    parent = stack[-1]
                    parent.height = max(parent.height, frame.height + frame.incoming_depth)
                continue

            work += 1
            if work > self.max_nodes:
                raise ConditionRenderError(
                    f"condition rendering exceeds {self.max_nodes} condition nodes or edges"
                )
            identity = id(child)
            edge_depth = self._edge_depth(frame.node, child)
            path_depth = frame.path_depth + edge_depth
            if path_depth > self.max_depth:
                raise ConditionRenderError(
                    f"condition rendering exceeds {self.max_depth} nested levels"
                )
            if identity in active:
                raise ConditionRenderError("condition rendering found a cycle")
            if identity in heights:
                frame.height = max(frame.height, heights[identity] + edge_depth)
                continue
            active.add(identity)
            try:
                children = self._children(child)
            except ConditionRenderError:
                raise
            except (TypeError, ValueError) as exc:
                raise ConditionRenderError("condition node has malformed children") from exc
            stack.append(_PreflightFrame(child, children, edge_depth, path_depth))

    @staticmethod
    def precedence(node: ConditionNode) -> int:
        node = cast(Any, node)
        if type(node) is ConditionalValue:
            return _CONDITIONAL_PRECEDENCE.precedence
        if type(node) is UnaryOp:
            return _NOT_PRECEDENCE.precedence
        if type(node) is FlowOp:
            if type(node.operator) is not str:
                raise ConditionRenderError("unsupported flow operator")
            family = _OPERATOR_FAMILIES.get(node.operator)
            if family != "flow":
                raise ConditionRenderError("unsupported flow operator")
            return _PRECEDENCE_BY_FAMILY[family].precedence
        if type(node) is BinaryOp:
            if type(node.operator) is not str:
                raise ConditionRenderError("unsupported binary operator")
            family = _OPERATOR_FAMILIES.get(node.operator)
            if family is None:
                raise ConditionRenderError("unsupported binary operator")
            return _PRECEDENCE_BY_FAMILY[family].precedence
        if type(node) in {Assignment, ContainsAnyOp}:
            return _WEAK_PRECEDENCE.precedence
        if type(node) in {FieldAccess, FunctionCall, ModelCall}:
            return _POSTFIX_PRECEDENCE.precedence
        return _PRIMARY_PRECEDENCE

    @staticmethod
    def _writes_own_group(node: ConditionNode) -> bool:
        return type(node) in {BinaryOp, FlowOp, ConditionalValue}

    def write(self, node: ConditionNode, depth: int = 0) -> None:
        if not isinstance(node, ConditionNode):
            raise ConditionRenderError(f"expected a condition node, got {type(node).__name__}")
        self._run([("node", node, depth)])

    def _run(self, pending: list[tuple[Any, ...]]) -> None:
        """Execute output actions without using the Python call stack."""

        while pending:
            task = pending.pop()
            kind = task[0]
            if kind == "text":
                self.append(task[1])
                continue
            if kind == "identifier":
                self.append_identifier(task[1], task[2])
                continue
            if kind == "node":
                self._schedule_node(pending, task[1], task[2])
                continue
            if kind == "operand":
                _, node, depth, minimum, followed, force_group = task
                grouped = self._writes_own_group(node)
                needs_group = (
                    force_group
                    or self.precedence(node) < minimum
                    or (followed and id(node) in self.greedy_tails)
                )
                if needs_group and not grouped:
                    pending.append(("text", ")"))
                pending.append(("node", node, depth))
                if needs_group and not grouped:
                    pending.append(("text", "("))
                continue
            if kind == "postfix":
                _, node, depth = task
                if type(node) is Literal:
                    pending.extend(
                        [
                            ("text", ")"),
                            ("node", node, depth),
                            ("text", "("),
                        ]
                    )
                else:
                    left_minimum, _ = self.operand_minima(_POSTFIX_PRECEDENCE)
                    pending.append(("operand", node, depth, left_minimum, True, False))
                continue
            if kind == "arguments":
                _, arguments, index, depth = task
                if index >= len(arguments):
                    continue
                pending.append(("arguments", arguments, index + 1, depth))
                argument = arguments[index]
                if type(argument) is _KeywordArgument:
                    pending.append(("node", argument.value, depth + 1))
                    pending.append(("text", "="))
                    pending.append(("identifier", argument.name, "argument name"))
                else:
                    pending.append(("node", argument, depth + 1))
                if index:
                    pending.append(("text", ", "))
                continue
            if kind == "fields":
                _, fields, index = task
                if index >= len(fields):
                    continue
                field = fields[index]
                if field == "*" and index != len(fields) - 1:
                    raise ConditionRenderError("wildcard must be the last field")
                if index:
                    self.append(".")
                if field == "*":
                    self.append("*")
                else:
                    self.append_identifier(field, "field name")
                pending.append(("fields", fields, index + 1))
                continue
            if kind == "slice":
                _, node, depth = task
                if type(node) is Literal and node.value is None:
                    continue
                if not (type(node) is Literal and type(node.value) is int):
                    raise ConditionRenderError("slice bounds must be integer literals or null")
                pending.append(("node", node, depth))
                continue
            if kind == "flat_boolean":
                _, operands, index, operator, depth, minimum = task
                if index >= len(operands):
                    continue
                pending.append(
                    (
                        "flat_boolean",
                        operands,
                        index + 1,
                        operator,
                        depth,
                        minimum,
                    )
                )
                pending.append(
                    (
                        "operand",
                        operands[index],
                        depth + 1,
                        minimum,
                        index < len(operands) - 1,
                        False,
                    )
                )
                if index:
                    pending.append(("text", f" {operator} "))
                continue
            if kind == "pattern_entries":
                _, patterns, index, depth = task
                if index >= len(patterns):
                    continue
                pattern = patterns[index]
                if type(pattern) is not BinaryOp:
                    raise ConditionRenderError("pattern block entries must be binary predicates")
                operator = pattern.operator
                if type(operator) is not str or operator not in _PATTERN_OPERATORS:
                    raise ConditionRenderError("unsupported pattern-block operator")
                if type(pattern.right) is not Literal:
                    raise ConditionRenderError("pattern-block values must be literals")
                pending.append(("pattern_entries", patterns, index + 1, depth))
                pending.append(("node", pattern.right, depth + 2))
                pending.append(("text", f" {'==' if operator == '=' else operator} "))
                pending.append(("postfix", pattern.left, depth + 2))
                if index:
                    pending.append(("text", " or "))
                continue
            if kind == "pattern_names":
                _, patterns, index = task
                if index >= len(patterns):
                    continue
                pending.append(("pattern_names", patterns, index + 1))
                pending.append(("pattern_name", patterns[index]))
                if index:
                    pending.append(("text", ", "))
                continue
            if kind == "pattern_name":
                self.append_pattern_name(task[1])
                continue
            if kind == "literal":
                _, value, depth = task
                self._schedule_literal(pending, value, depth)
                continue
            if kind == "literal_items":
                _, values, index, depth = task
                if index >= len(values):
                    continue
                pending.append(("literal_items", values, index + 1, depth))
                pending.append(("literal", values[index], depth + 1))
                if index:
                    pending.append(("text", ", "))
                continue
            if kind == "regex":
                self._append_regex(task[1])
                continue
            if kind == "integer":
                self.append_integer(task[1])
                continue
            raise ConditionRenderError("condition renderer encountered an invalid action")

    def _schedule_node(
        self, pending: list[tuple[Any, ...]], node: ConditionNode, depth: int
    ) -> None:
        """Schedule one supported AST node in reverse output order."""

        node = cast(Any, node)
        node_type = type(node)
        if node_type is Literal:
            pending.append(("literal", node.value, 0))
            return
        if node_type is Identifier:
            name = self._plain_text(node.name, "identifier")
            if name.startswith("$"):
                self.ensure_raw_capacity(name)
                if not _STRING_VARIABLE.fullmatch(name):
                    raise ConditionRenderError("string variable is not valid DSL syntax")
                self.append(name)
            else:
                if name in _NON_VALUE_IDENTIFIERS:
                    raise ConditionRenderError("identifier is reserved by the DSL")
                for index, part in enumerate(name.split(".")):
                    if index:
                        self.append(".")
                    self.append_identifier(part, "identifier")
            return
        if node_type is ParameterRef:
            self.append("params.")
            self.append_identifier(node.name, "parameter name")
            return
        if node_type is FieldAccess:
            fields = self._text_list(node.fields, "field path")
            if not fields:
                raise ConditionRenderError("field access has no field")
            pending.append(("fields", fields, 0))
            if node.base is not None:
                pending.append(("text", "."))
                pending.append(("postfix", node.base, depth + 1))
            return
        if node_type is FunctionCall:
            name = self._plain_text(node.name, "function name")
            arguments = self._node_list(node.args, "function arguments")
            if name in _NON_VALUE_IDENTIFIERS:
                raise ConditionRenderError("function name is reserved by the DSL grammar")
            if name == "_string_spans":
                if not arguments:
                    raise ConditionRenderError("string spans require a receiver")
                pending.extend(
                    [
                        ("text", ")"),
                        ("arguments", arguments[1:], 0, depth),
                        ("text", ".spans("),
                        ("postfix", arguments[0], depth + 1),
                    ]
                )
                return
            if name == "_extract_windows":
                if len(arguments) < 2 or len(arguments) > 5:
                    raise ConditionRenderError(
                        "text extraction requires spans and up to three bounds"
                    )
                context = [
                    arguments[1],
                    *(
                        _KeywordArgument(name, value)
                        for name, value in zip(
                            ("left", "right", "head"), arguments[2:], strict=False
                        )
                    ),
                ]
                if (
                    len(arguments) > 2
                    and type(arguments[-1]) is Literal
                    and arguments[-1].value is None
                ):
                    # Named bounds omit trailing null defaults when parsed. Positional
                    # spelling retains an explicit bound and therefore the same AST.
                    context = arguments[1:]
                pending.extend(
                    [
                        ("text", ")"),
                        ("arguments", context, 0, depth),
                        ("text", ".extract("),
                        ("postfix", arguments[0], depth + 1),
                    ]
                )
                return
            if (
                name == "extract"
                and 2 <= len(arguments) <= 4
                and type(arguments[0]) is Identifier
                and arguments[0].name in {"messages", "conversation"}
            ):
                context = [
                    arguments[1],
                    *(
                        _KeywordArgument(name, value)
                        for name, value in zip(("left", "right"), arguments[2:], strict=False)
                    ),
                ]
                if (
                    len(arguments) > 2
                    and type(arguments[-1]) is Literal
                    and arguments[-1].value is None
                ):
                    context = arguments[1:]
                pending.extend(
                    [
                        ("text", ")"),
                        ("arguments", context, 0, depth),
                        ("text", ".extract("),
                        ("postfix", arguments[0], depth + 1),
                    ]
                )
                return
            if name == "_transform":
                receiver = arguments[0] if arguments else Identifier("messages")
                pending.extend(
                    [
                        ("text", ")"),
                        ("arguments", arguments[1:], 0, depth),
                        ("text", ".transform("),
                        ("postfix", receiver, depth + 1),
                    ]
                )
                return
            if name == "_index":
                if len(arguments) != 2:
                    raise ConditionRenderError("index node requires two arguments")
                pending.extend(
                    [
                        ("text", "]"),
                        ("node", arguments[1], depth + 1),
                        ("text", "["),
                        ("postfix", arguments[0], depth + 1),
                    ]
                )
                return
            if name == "_slice":
                if len(arguments) != 3:
                    raise ConditionRenderError("slice node requires three arguments")
                pending.extend(
                    [
                        ("text", "]"),
                        ("slice", arguments[2], depth + 1),
                        ("text", ":"),
                        ("slice", arguments[1], depth + 1),
                        ("text", "["),
                        ("postfix", arguments[0], depth + 1),
                    ]
                )
                return
            self.append_identifier(name, "function name")
            self.append("(")
            pending.extend(
                [
                    ("text", ")"),
                    ("arguments", arguments, 0, depth),
                ]
            )
            return
        if node_type is ModelCall:
            alias = self._plain_text(node.alias, "model alias")
            operation = self._plain_text(node.operation, "model operation")
            result_type = self._plain_text(node.result_type, "model result type")
            arguments = self._node_list(node.args, "model arguments")
            if not alias:
                raise ConditionRenderError("model alias must not be empty")
            expected_type = MODEL_OPERATION_TYPES.get(operation)
            if expected_type is None or result_type != expected_type:
                raise ConditionRenderError("unsupported or malformed model operation")
            self.append("model(")
            self.append_quoted_string(alias)
            self.append(").")
            self.append_identifier(operation, "model operation")
            self.append("(")
            pending.extend(
                [
                    ("text", ")"),
                    ("arguments", arguments, 0, depth),
                ]
            )
            return
        if node_type is UnaryOp:
            operator = self._plain_text(node.operator, "unary operator")
            if operator != "not":
                raise ConditionRenderError("unsupported unary operator")
            self.append("not ")
            _, minimum = self.operand_minima(_NOT_PRECEDENCE)
            pending.append(("operand", node.operand, depth + 1, minimum, False, False))
            return
        if node_type is ConditionalValue:
            left_minimum, right_minimum = self.operand_minima(_CONDITIONAL_PRECEDENCE)
            self.append("(")
            pending.extend(
                [
                    ("text", ")"),
                    (
                        "operand",
                        node.fallback,
                        depth + 1,
                        right_minimum,
                        False,
                        False,
                    ),
                    ("text", " else "),
                    (
                        "operand",
                        node.predicate,
                        depth + 1,
                        left_minimum,
                        True,
                        False,
                    ),
                    ("text", " if "),
                    (
                        "operand",
                        node.chosen,
                        depth + 1,
                        left_minimum,
                        True,
                        False,
                    ),
                ]
            )
            return
        if node_type in {BinaryOp, FlowOp}:
            self._schedule_binary(pending, node, depth)
            return
        if node_type is AnyComprehension and node.variable.startswith("\x00composed"):
            # Linked checks are internal ASTs. Keep their condition in compiler labels
            # so the linker can attribute every string reference to its called rule.
            self.append("any ")
            self.append_identifier(node.variable.removeprefix("\x00composed"), "called rule alias")
            self.append(" in ")
            pending.extend(
                [
                    ("node", node.condition, depth + 1),
                    ("text", " where "),
                    (
                        "operand",
                        node.collection,
                        depth + 1,
                        _CONDITIONAL_PRECEDENCE.precedence,
                        True,
                        False,
                    ),
                ]
            )
            return
        if node_type in {AnyComprehension, AllComprehension}:
            keyword = "any" if node_type is AnyComprehension else "all"
            self.append(f"{keyword} ")
            self.append_identifier(node.variable, "comprehension variable")
            self.append(" in ")
            pending.extend(
                [
                    ("node", node.condition, depth + 1),
                    ("text", " where "),
                    (
                        "operand",
                        node.collection,
                        depth + 1,
                        _CONDITIONAL_PRECEDENCE.precedence,
                        True,
                        False,
                    ),
                ]
            )
            return
        if node_type is FilterComprehension:
            variable = self._plain_text(node.variable, "filter variable")
            output = self._optional_plain_text(node.output_variable, "filter output variable")
            self.append("[")
            self.append_identifier(
                output if output is not None else variable, "filter output variable"
            )
            self.append(" for ")
            self.append_identifier(variable, "filter variable")
            self.append(" in ")
            minimum = _PRECEDENCE_BY_FAMILY["boolean or"].precedence
            pending.extend(
                [
                    ("text", "]"),
                    (
                        "operand",
                        node.condition,
                        depth + 1,
                        minimum,
                        False,
                        False,
                    ),
                    ("text", " if "),
                    (
                        "operand",
                        node.collection,
                        depth + 1,
                        minimum,
                        True,
                        False,
                    ),
                ]
            )
            return
        if node_type is MapComprehension:
            variable = self._plain_text(node.variable, "map variable")
            self.append("[")
            minimum = _PRECEDENCE_BY_FAMILY["boolean or"].precedence
            pending.append(("text", "]"))
            if node.condition is not None:
                pending.extend(
                    [
                        (
                            "operand",
                            node.condition,
                            depth + 1,
                            minimum,
                            False,
                            False,
                        ),
                        ("text", " if "),
                    ]
                )
            pending.extend(
                [
                    (
                        "operand",
                        node.collection,
                        depth + 1,
                        minimum,
                        node.condition is not None,
                        False,
                    ),
                    ("text", " in "),
                    ("identifier", variable, "map variable"),
                    ("text", " for "),
                    (
                        "operand",
                        node.expression,
                        depth + 1,
                        _CONDITIONAL_PRECEDENCE.precedence,
                        True,
                        False,
                    ),
                ]
            )
            return
        if node_type is Assignment:
            self._plain_text(node.variable, "alias")
            _, minimum = self.operand_minima(_WEAK_PRECEDENCE)
            pending.extend(
                [
                    ("identifier", node.variable, "alias"),
                    ("text", " as "),
                    (
                        "operand",
                        node.expression,
                        depth + 1,
                        minimum,
                        True,
                        False,
                    ),
                ]
            )
            return
        if node_type is PatternMatch:
            patterns = self._node_list(node.patterns, "pattern predicates")
            if not patterns:
                raise ConditionRenderError("pattern block must contain at least one predicate")
            self.append("{")
            pending.extend(
                [
                    ("text", "}"),
                    ("pattern_entries", patterns, 0, depth),
                ]
            )
            return
        if node_type in {StringPosition, StringCount}:
            variable = self._plain_text(node.variable, "string position/count name")
            self.ensure_raw_capacity(variable, overhead=1)
            if not _POSITION_VARIABLE.fullmatch(variable):
                raise ConditionRenderError("string position/count name is not valid DSL syntax")
            self.append("@" if node_type is StringPosition else "#")
            self.append(variable)
            return
        if node_type is StringSetQuantifier:
            quantifier = node.quantifier
            patterns = self._text_list(node.patterns, "string-set members")
            if not (
                type(quantifier) is str
                and quantifier in {"any", "all"}
                or type(quantifier) is int
                and quantifier >= 0
            ):
                raise ConditionRenderError("unsupported string-set quantifier")
            if not patterns:
                raise ConditionRenderError("string set must contain at least one member")
            if type(quantifier) is int:
                self.append_integer(quantifier)
            else:
                self.append(quantifier)
            self.append(" of (")
            pending.extend(
                [
                    ("text", ")"),
                    ("pattern_names", patterns, 0),
                ]
            )
            return
        if node_type is ContainsAnyOp:
            if type(node.values) is not list:
                raise ConditionRenderError("contains_any values must be a literal list")
            if not node.values:
                raise ConditionRenderError("contains_any must contain at least one value")
            _, minimum = self.operand_minima(_WEAK_PRECEDENCE)
            pending.extend(
                [
                    ("text", "]"),
                    ("literal_items", node.values, 0, 0),
                    ("text", " contains_any ["),
                    (
                        "operand",
                        node.left,
                        depth + 1,
                        minimum,
                        True,
                        False,
                    ),
                ]
            )
            return
        raise ConditionRenderError(f"unsupported condition node {node_type.__name__}")

    def _schedule_binary(
        self,
        pending: list[tuple[Any, ...]],
        node: BinaryOp | FlowOp,
        depth: int,
    ) -> None:
        operator = self._plain_text(node.operator, "binary operator")
        if type(node) is FlowOp:
            if operator not in _FLOW_OPERATORS:
                raise ConditionRenderError("unsupported flow operator")
            if node.max_distance is not None and (
                operator != "after" or type(node.max_distance) is not int or node.max_distance < 1
            ):
                raise ConditionRenderError("invalid bounded flow distance")
        elif operator not in _BINARY_OPERATORS:
            raise ConditionRenderError("unsupported binary operator")

        record = self.binary_precedence(operator)
        left_minimum, right_minimum = self.operand_minima(record)
        self.append("(")
        pending.append(("text", ")"))
        if type(node) is BinaryOp and operator in {"and", "or"}:
            rights: list[ConditionNode] = []
            current = node
            while (
                type(current.left) is BinaryOp
                and type(current.left.operator) is str
                and current.left.operator == operator
            ):
                rights.append(current.right)
                current = current.left
            operands = [current.left, current.right, *reversed(rights)]
            pending.append(
                (
                    "flat_boolean",
                    operands,
                    0,
                    operator,
                    depth,
                    record.precedence + 1,
                )
            )
            return

        if operator == "~" and type(node.right) is Literal:
            pending.append(("regex", node.right))
        else:
            pending.append(
                (
                    "operand",
                    node.right,
                    depth + 1,
                    right_minimum,
                    False,
                    False,
                )
            )
        pending.append(("text", " "))
        if type(node) is FlowOp and node.max_distance is not None:
            pending.extend(
                [
                    ("text", ")"),
                    ("integer", node.max_distance),
                    ("text", "(within: "),
                ]
            )
        pending.append(("text", "==" if operator == "=" else operator))
        pending.append(("text", " "))
        pending.append(
            (
                "operand",
                node.left,
                depth + 1,
                left_minimum,
                True,
                operator == "*" and type(node.left) in {StringPosition, StringCount},
            )
        )

    def _schedule_literal(self, pending: list[tuple[Any, ...]], value: Any, depth: int) -> None:
        if depth > self.max_depth:
            raise ConditionRenderError(f"literal rendering exceeds {self.max_depth} nested levels")
        value_type = type(value)
        if value is None:
            self.append("null")
            return
        if value_type is str:
            self.append_quoted_string(value)
            return
        if value_type is bool:
            self.append("true" if value else "false")
            return
        if value_type is int:
            self.append_integer(value)
            return
        if value_type is float:
            if not math.isfinite(value):
                raise ConditionRenderError("numeric literals must be finite")
            self.append(str(value))
            return
        if value_type is list:
            self.append("[")
            pending.extend(
                [
                    ("text", "]"),
                    ("literal_items", value, 0, depth),
                ]
            )
            return
        raise ConditionRenderError(
            f"literal type {value_type.__name__} is not supported by the DSL"
        )

    def _append_regex(self, node: Literal) -> None:
        if type(node) is not Literal or type(node.value) is not str:
            raise ConditionRenderError("regex operand must be text")
        raw = node.value
        self.ensure_raw_capacity(raw, overhead=2)
        escaped = False
        for character in raw:
            if character in "\r\n":
                raise ConditionRenderError("regex spelling cannot contain a line break")
            if character == '"' and not escaped:
                raise ConditionRenderError("regex spelling contains an unescaped quote")
            escaped = not escaped if character == "\\" else False
        if escaped:
            raise ConditionRenderError("regex spelling ends in an unpaired backslash")
        self.append('"')
        self.append(raw)
        self.append('"')

    def binary_precedence(self, operator: str) -> OperatorPrecedence:
        if type(operator) is not str:
            raise ConditionRenderError("unsupported binary operator")
        family = _OPERATOR_FAMILIES.get(operator)
        if family is None:
            raise ConditionRenderError("unsupported binary operator")
        return _PRECEDENCE_BY_FAMILY[family]

    @staticmethod
    def operand_minima(record: OperatorPrecedence) -> tuple[int, int]:
        if record.associativity == "left":
            return record.precedence, record.precedence + 1
        if record.associativity == "right":
            return record.precedence + 1, record.precedence
        return record.precedence + 1, record.precedence + 1

    def append_quoted_string(self, value: object) -> None:
        if type(value) is not str:
            raise ConditionRenderError("string literal must be text")
        self.ensure_raw_capacity(value, overhead=2)
        self.append('"')
        for chunk in _json_escape_chunks(value):
            self.append(chunk)
        self.append('"')

    def append_pattern_name(self, value: object) -> None:
        if type(value) is not str:
            raise ConditionRenderError("string-set member is not valid DSL syntax")
        self.ensure_raw_capacity(value)
        if not _STRING_VARIABLE.fullmatch(value):
            raise ConditionRenderError("string-set member is not valid DSL syntax")
        self.append(value)

    def append_integer(self, value: int) -> None:
        if type(value) is not int:
            raise ConditionRenderError("integer literal must be a plain integer")
        remaining = self.max_output - self.output_size
        sign_size = int(value < 0)
        digit_capacity = remaining - sign_size
        if digit_capacity < 1:
            raise ConditionRenderError(
                f"condition rendering exceeds {self.max_output} output characters"
            )

        bits = value.bit_length()
        definitely_fits = digit_capacity * _LOG2_10_LOWER_NUMERATOR // _LOG2_10_DENOMINATOR
        definitely_exceeds = (
            digit_capacity * _LOG2_10_UPPER_NUMERATOR + _LOG2_10_DENOMINATOR - 1
        ) // _LOG2_10_DENOMINATOR
        if bits > definitely_exceeds:
            raise ConditionRenderError(
                f"condition rendering exceeds {self.max_output} output characters"
            )
        if bits > definitely_fits:
            limit = 10**digit_capacity
            if value >= limit or value <= -limit:
                raise ConditionRenderError(
                    f"condition rendering exceeds {self.max_output} output characters"
                )
        try:
            text = _integer_text(value)
        except ValueError as exc:
            raise ConditionRenderError("integer literal is too large to render") from exc
        self.append(text)


def render_condition(
    node: ConditionNode,
    *,
    max_nodes: int = MAX_RENDER_NODES,
    max_depth: int = MAX_RENDER_DEPTH,
    max_output: int = MAX_RENDER_OUTPUT,
) -> str:
    """Render one condition as deterministic, round-trippable public DSL syntax.

    Rendering is bounded by node, nesting, and output limits. Unsupported or malformed
    nodes fail closed with :class:`ConditionRenderError`.
    """

    renderer = _Renderer(
        max_nodes=max_nodes,
        max_depth=max_depth,
        max_output=max_output,
    )
    renderer.preflight(node)
    renderer.write(node)
    return "".join(renderer.parts)
