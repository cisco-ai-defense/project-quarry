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

"""Tests for exhaustive, deterministic DSL condition traversal."""

from dataclasses import is_dataclass

import pytest

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
    walk_condition_nodes,
    walk_condition_visits,
)


def _call(name: str) -> FunctionCall:
    return FunctionCall(name, [])


def test_walk_is_deterministic_preorder_with_ancestor_paths() -> None:
    expression_arg = _call("expression_arg")
    root = MapComprehension(
        FunctionCall("expression", [expression_arg]),
        "item",
        _call("collection"),
        _call("condition"),
    )

    visits = list(walk_condition_visits(root))

    assert [
        visit.node.name if isinstance(visit.node, FunctionCall) else type(visit.node).__name__
        for visit in visits
    ] == [
        "MapComprehension",
        "expression",
        "expression_arg",
        "collection",
        "condition",
    ]
    assert visits[2].ancestors == (root, root.expression)


def test_walk_rejects_active_path_cycle() -> None:
    root = UnaryOp("not", Literal(True))
    root.operand = root

    with pytest.raises(ValueError, match="Cycle detected"):
        list(walk_condition_nodes(root))


def test_walk_rejects_invalid_values_in_typed_child_collection() -> None:
    root = FunctionCall("outer", [Literal(True)])
    root.args.append("not-a-node")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"FunctionCall\.args\[1\]"):
        list(walk_condition_nodes(root))


def _all_subclasses(base: type[ConditionNode]) -> set[type[ConditionNode]]:
    subclasses: set[type[ConditionNode]] = set()
    for child in base.__subclasses__():
        subclasses.add(child)
        subclasses.update(_all_subclasses(child))
    return subclasses


def test_concrete_condition_node_inventory_requires_traversal_review() -> None:
    expected = {
        AllComprehension,
        AnyComprehension,
        Assignment,
        BinaryOp,
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
    }

    production_subclasses = {
        node_type
        for node_type in _all_subclasses(ConditionNode)
        if node_type.__module__.startswith("quarry.")
    }

    assert production_subclasses == expected
    assert all(is_dataclass(node_type) for node_type in expected)


def test_walk_rejects_subclass_that_inherits_dataclass_without_declaring_it() -> None:
    class UndeclaredChild(BinaryOp):
        hidden: ConditionNode

    root = UndeclaredChild("and", Literal(True), Literal(True))
    root.hidden = _call("hidden")

    with pytest.raises(TypeError, match="must directly declare @dataclass"):
        list(walk_condition_nodes(root))
