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

"""Evaluate a compiler-validated collection plan against one scan context."""

from __future__ import annotations

from dataclasses import replace

from quarry.engine.collect import CollectPlan
from quarry.engine.matchers import MISSING, truth
from quarry.integrations.model_expression import evaluate_model_expression
from quarry.types.results import CollectionResult


async def evaluate_collect_plan(
    plan: CollectPlan,
    context,
    *,
    max_items: int | None,
    regex_backend=None,
) -> CollectionResult | None:
    """Materialize once, exhaustively filter, and return every surviving item."""

    source = await evaluate_model_expression(
        plan.source,
        context,
        regex_backend=regex_backend,
    )
    if not isinstance(source, CollectionResult):
        raise TypeError("collect producer must return a CollectionResult")
    if max_items is not None and len(source.items) > max_items:
        raise ValueError(
            f"collect producer returned {len(source.items)} items, exceeding "
            f"max_comprehension_size={max_items}"
        )

    local_survivors = []
    for item in source.items:
        keep = await evaluate_model_expression(
            plan.local_predicate,
            context,
            bindings={plan.variable: item.value},
            regex_backend=regex_backend,
        )
        if not isinstance(keep, bool):
            raise TypeError("collect filter must return a boolean")
        if keep:
            local_survivors.append(item)

    survivors = local_survivors
    if plan.model_predicate is not None:
        survivors = []
        for item in local_survivors:
            keep = await evaluate_model_expression(
                plan.model_predicate,
                context,
                bindings={plan.variable: item.value},
                regex_backend=regex_backend,
            )
            if keep is not MISSING and not isinstance(keep, bool):
                raise TypeError("collect model filter must return a boolean")
            if truth(keep):
                survivors.append(item)

    if not survivors:
        return None
    return replace(source, items=tuple(survivors))
