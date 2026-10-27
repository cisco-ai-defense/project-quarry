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

"""Deterministic traversal for executable DSL condition nodes."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields, is_dataclass
from functools import cache
from typing import Any, get_args, get_type_hints

from quarry.types.dsl.base import ConditionNode


@dataclass(frozen=True, slots=True)
class ConditionVisit:
    """A condition node and the nodes on its active ancestor path."""

    node: ConditionNode
    ancestors: tuple[ConditionNode, ...]


def _annotation_contains_condition_node(annotation: Any) -> bool:
    """Return whether a type annotation can contain executable condition nodes."""

    try:
        if issubclass(annotation, ConditionNode):
            return True
    except TypeError:
        pass

    return any(_annotation_contains_condition_node(arg) for arg in get_args(annotation))


@cache
def _condition_child_fields(node_type: type[ConditionNode]) -> tuple[str, ...]:
    """Find typed executable child fields in deterministic declaration order."""

    if not is_dataclass(node_type) or "__dataclass_fields__" not in node_type.__dict__:
        raise TypeError(f"Condition node {node_type.__name__} must directly declare @dataclass")

    type_hints = get_type_hints(node_type)
    child_fields = tuple(
        field.name
        for field in fields(node_type)
        if _annotation_contains_condition_node(type_hints.get(field.name))
    )
    undeclared_child_fields = {
        name
        for name, annotation in type_hints.items()
        if _annotation_contains_condition_node(annotation) and name not in child_fields
    }
    if undeclared_child_fields:
        fields_text = ", ".join(sorted(undeclared_child_fields))
        raise TypeError(
            f"Condition node {node_type.__name__} has executable field(s) not declared "
            f"as dataclass fields: {fields_text}"
        )
    return child_fields


def condition_child_fields(node: ConditionNode) -> tuple[str, ...]:
    """Names of ``node``'s typed executable child fields, in declaration order.

    The single source of truth for AST child structure. ``iter_condition_children`` reads
    these to walk the tree; callers that must WRITE children back (variable substitution)
    read the names directly, so no separate helper re-hardcodes the field list.
    """

    if not isinstance(node, ConditionNode):
        raise TypeError(f"Expected ConditionNode, got {type(node).__name__}")
    return _condition_child_fields(type(node))


def iter_condition_children(node: ConditionNode) -> Iterator[ConditionNode]:
    """Yield only typed executable children of a condition node."""

    if not isinstance(node, ConditionNode):
        raise TypeError(f"Expected ConditionNode, got {type(node).__name__}")

    for field_name in _condition_child_fields(type(node)):
        value = getattr(node, field_name)
        if value is None:
            continue
        if isinstance(value, ConditionNode):
            yield value
            continue
        if isinstance(value, list | tuple):
            for index, child in enumerate(value):
                if not isinstance(child, ConditionNode):
                    raise TypeError(
                        f"{type(node).__name__}.{field_name}[{index}] must be a ConditionNode"
                    )
                yield child
            continue
        raise TypeError(
            f"{type(node).__name__}.{field_name} must be a ConditionNode or node sequence"
        )


def walk_condition_visits(root: ConditionNode | None) -> Iterator[ConditionVisit]:
    """Walk a condition AST in deterministic preorder.

    Cycles on the active ancestor path are rejected. Shared nodes in an acyclic
    graph are visited once for each executable occurrence.
    """

    if root is None:
        return
    if not isinstance(root, ConditionNode):
        raise TypeError(f"Expected ConditionNode root, got {type(root).__name__}")

    stack: list[tuple[ConditionNode, tuple[ConditionNode, ...]]] = [(root, ())]
    while stack:
        node, ancestors = stack.pop()
        if any(node is ancestor for ancestor in ancestors):
            raise ValueError(f"Cycle detected in condition AST at {type(node).__name__}")

        yield ConditionVisit(node=node, ancestors=ancestors)

        child_ancestors = (*ancestors, node)
        children = tuple(iter_condition_children(node))
        stack.extend((child, child_ancestors) for child in reversed(children))


def walk_condition_nodes(root: ConditionNode | None) -> Iterator[ConditionNode]:
    """Yield condition nodes in deterministic preorder."""

    for visit in walk_condition_visits(root):
        yield visit.node
