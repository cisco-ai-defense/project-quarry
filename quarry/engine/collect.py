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

"""Closed collection-execution plan carried beside a rule's Boolean DFA."""

from __future__ import annotations

from dataclasses import dataclass

from quarry.engine.atoms import ModelExpression


@dataclass(frozen=True, slots=True)
class CollectPlan:
    """One locally filtered collection with an optional model post-filter."""

    variable: str
    item_type: str
    source: ModelExpression
    local_predicate: ModelExpression
    model_predicate: ModelExpression | None
    has_external_work: bool
