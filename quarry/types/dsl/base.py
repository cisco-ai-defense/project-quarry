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

"""Structural base for parsed DSL condition nodes.

The parser builds data-only AST nodes and the compiled engine lowers them into rule
artifacts. Evaluation runs through the DFA executor rather than methods on AST nodes.
"""

from __future__ import annotations

from abc import ABC


class ConditionNode(ABC):  # noqa: B024 — intentional data-marker base
    """Structural base for a parsed DSL condition node.

    Carries no behavior: subclasses are dataclasses holding the fields the compiler reads.
    It remains an ``ABC`` purely as a not-directly-instantiable marker / common type for the
    AST; execution runs through the compiled engine.
    """
