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

"""Data-only binary operator nodes for parsed rules.

The compiler lowers boolean, comparison, and string operators into temporal formulas
and grounded atoms.
"""

from dataclasses import dataclass

from quarry.types.dsl.base import ConditionNode


@dataclass
class BinaryOp(ConditionNode):
    """Binary operation: ``left <operator> right``."""

    operator: str
    left: ConditionNode
    right: ConditionNode
