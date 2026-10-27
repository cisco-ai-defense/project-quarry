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

"""DSL node types."""

from quarry.types.dsl.access import FieldAccess
from quarry.types.dsl.base import ConditionNode
from quarry.types.dsl.comprehensions import (
    AllComprehension,
    AnyComprehension,
    FilterComprehension,
    MapComprehension,
)
from quarry.types.dsl.functions import FunctionCall
from quarry.types.dsl.literals import Identifier, Literal, ParameterRef
from quarry.types.dsl.model_calls import MODEL_OPERATION_TYPES, ModelCall
from quarry.types.dsl.operators import (
    BinaryOp,
    ContainsAnyOp,
    FlowOp,
    StringCount,
    StringPosition,
    StringSetQuantifier,
    UnaryOp,
)
from quarry.types.dsl.patterns import Assignment, PatternMatch
from quarry.types.dsl.string_ops import StringModifiers
from quarry.types.dsl.traversal import (
    ConditionVisit,
    condition_child_fields,
    iter_condition_children,
    walk_condition_nodes,
    walk_condition_visits,
)

__all__ = [
    "ConditionNode",
    "ConditionVisit",
    "condition_child_fields",
    "iter_condition_children",
    "walk_condition_nodes",
    "walk_condition_visits",
    "BinaryOp",
    "UnaryOp",
    "ContainsAnyOp",
    "FlowOp",
    "FieldAccess",
    "Literal",
    "Identifier",
    "Assignment",
    "PatternMatch",
    "ParameterRef",
    "FunctionCall",
    "AnyComprehension",
    "AllComprehension",
    "FilterComprehension",
    "MapComprehension",
    "MODEL_OPERATION_TYPES",
    "ModelCall",
    "StringPosition",
    "StringCount",
    "StringSetQuantifier",
    "StringModifiers",
    "ConditionVisit",
    "iter_condition_children",
    "walk_condition_nodes",
    "walk_condition_visits",
]
