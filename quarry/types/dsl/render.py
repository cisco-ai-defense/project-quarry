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

import json
from typing import Any

from quarry.types.dsl import (
    AllComprehension,
    AnyComprehension,
    Assignment,
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
    ParameterRef,
    PatternMatch,
    StringCount,
    StringPosition,
    StringSetQuantifier,
    UnaryOp,
)


def json_value(value: Any) -> Any:
    """Project a DSL value into a bounded, JSON-compatible representation."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return str(value)


def _operand(node: ConditionNode) -> str:
    """Render a binary operand; a comprehension's `where` would otherwise absorb the rest."""

    rendered = render_condition(node)
    return f"({rendered})" if isinstance(node, AnyComprehension | AllComprehension) else rendered


def render_condition(node: ConditionNode) -> str:
    """Render a condition as rule syntax, without depending on the source text.

    Index and slice access print as written, and a comprehension used as an operand is
    parenthesized, so a rendered condition parses back to the same condition.
    """

    if isinstance(node, Literal):
        return json.dumps(json_value(node.value), sort_keys=True)
    if isinstance(node, Identifier):
        return node.name
    if isinstance(node, ParameterRef):
        return f"params.{node.name}"
    if isinstance(node, FieldAccess):
        base = node.base
        prefix = render_condition(base) + "." if isinstance(base, ConditionNode) else ""
        return prefix + ".".join(node.fields)
    if isinstance(node, FunctionCall):
        # The parser desugars `x[i]` and `x[a:b]` into these calls; print the syntax.
        if node.name == "_index":
            base, index = node.args
            return f"{render_condition(base)}[{render_condition(index)}]"
        if node.name == "_slice":
            base, start, stop = node.args
            bounds = [
                ""
                if isinstance(bound, Literal) and bound.value is None
                else render_condition(bound)
                for bound in (start, stop)
            ]
            return f"{render_condition(base)}[{bounds[0]}:{bounds[1]}]"
        args = ", ".join(render_condition(arg) for arg in node.args)
        return f"{node.name}({args})"
    if isinstance(node, ModelCall):
        args = ", ".join(render_condition(arg) for arg in node.args)
        return f'model("{node.alias}").{node.operation}({args})'
    if isinstance(node, UnaryOp):
        return f"{node.operator} {render_condition(node.operand)}"
    if isinstance(node, BinaryOp):
        operator = "==" if node.operator == "=" else node.operator
        if isinstance(node, FlowOp) and node.max_distance is not None:
            operator += f"(within: {node.max_distance})"
        return f"({_operand(node.left)} {operator} {_operand(node.right)})"
    if isinstance(node, AnyComprehension):
        return f"any {node.variable} in {render_condition(node.collection)} where {render_condition(node.condition)}"
    if isinstance(node, AllComprehension):
        return f"all {node.variable} in {render_condition(node.collection)} where {render_condition(node.condition)}"
    if isinstance(node, FilterComprehension):
        return f"[{node.variable} for {render_condition(node.collection)} if {render_condition(node.condition)}]"
    if isinstance(node, MapComprehension):
        suffix = f" if {render_condition(node.condition)}" if node.condition is not None else ""
        return f"[{render_condition(node.expression)} for {node.variable} in {render_condition(node.collection)}{suffix}]"
    if isinstance(node, Assignment):
        return f"{render_condition(node.expression)} as {node.variable}"
    if isinstance(node, PatternMatch):
        return "{" + " or ".join(render_condition(pattern) for pattern in node.patterns) + "}"
    if isinstance(node, StringPosition):
        return f"@{node.variable}"
    if isinstance(node, StringCount):
        return f"#{node.variable}"
    if isinstance(node, StringSetQuantifier):
        return f"{node.quantifier} of ({', '.join(node.patterns)})"
    if isinstance(node, ContainsAnyOp):
        values = ", ".join(json.dumps(json_value(value)) for value in node.values)
        return f"{render_condition(node.left)} contains_any [{values}]"
    return type(node).__name__
