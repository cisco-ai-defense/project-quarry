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

"""Data-only pattern matching and assignment nodes for parsed DSL rules."""

from __future__ import annotations

from dataclasses import dataclass

from quarry.types.dsl.base import ConditionNode


@dataclass
class Assignment(ConditionNode):
    """Assignment for the DSL.

    For text-matching operations (``pattern in text``) the compiler treats this as a
    value capture (``... as v``); for other expressions it is an alias binding. Both are
    lowered by ``quarry.engine.compiler``.
    """

    expression: ConditionNode
    variable: str


@dataclass
class PatternMatch(ConditionNode):
    """Pattern match for the DSL."""

    patterns: list[ConditionNode]
