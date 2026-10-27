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

"""Data-only DSL comprehension nodes.

The compiler lowers ``any x in S where φ`` to an existential temporal condition and
``all x in S where φ`` to a universal condition, preserving scoped access and captures.
"""

from __future__ import annotations

from dataclasses import dataclass

from quarry.types.dsl.base import ConditionNode


@dataclass
class AnyComprehension(ConditionNode):
    """``any variable in collection where condition``."""

    variable: str
    collection: ConditionNode
    condition: ConditionNode


@dataclass
class AllComprehension(ConditionNode):
    """``all variable in collection where condition``."""

    variable: str
    collection: ConditionNode
    condition: ConditionNode


@dataclass
class FilterComprehension(ConditionNode):
    """Filter comprehension for the DSL."""

    variable: str
    collection: ConditionNode
    condition: ConditionNode
    output_variable: str | None = None


@dataclass
class MapComprehension(ConditionNode):
    """Map comprehension for the DSL."""

    expression: ConditionNode
    variable: str
    collection: ConditionNode
    condition: ConditionNode | None = None
