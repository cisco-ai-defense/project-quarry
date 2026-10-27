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

"""Function call AST node and load-time name validation.

``FunctionCall`` stores a name and argument nodes. The compiler lowers supported calls
to atoms; ``link`` rejects unknown names before a rule is published.
"""

from __future__ import annotations

from dataclasses import dataclass

from quarry.language.runtime import INTERNAL_FUNCTION_NAMES, supported_function_names
from quarry.types.dsl.base import ConditionNode


@dataclass
class FunctionCall(ConditionNode):
    """Function call for the DSL."""

    name: str
    args: list[ConditionNode]

    def link(self):
        """Validate the function name at load time; reject unknown names.

        Name-only check (no callable resolution) that fails fast on unknown names.
        """
        if self.name not in supported_function_names() | INTERNAL_FUNCTION_NAMES:
            raise ValueError(f"Unknown function: {self.name}")
