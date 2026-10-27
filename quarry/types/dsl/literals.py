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

"""Data-only literal, identifier, and parameter nodes for parsed DSL rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quarry.types.dsl.base import ConditionNode


@dataclass
class Literal(ConditionNode):
    """Literal for the DSL."""

    value: Any
    # Keep the token contents available for contexts, such as regex patterns,
    # where Python string escape semantics must not be applied.
    raw_value: str | None = field(default=None, repr=False, compare=False)


@dataclass
class Identifier(ConditionNode):
    """Identifier for the DSL."""

    name: str


@dataclass(frozen=True)
class ParameterRef(ConditionNode):
    """A reference to one value declared by the rule's ``parameters:`` block."""

    name: str
