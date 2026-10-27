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

"""Preflight checks for authored bounds that expand before DFA construction."""

import pytest

from quarry.dsl.parser import DSLParser
from quarry.engine import builders, desugar, lower, temporal
from quarry.engine.budget import DFAConstructionLimit, WorkBudget, spending
from quarry.engine.compiler import _Compiler


def test_filtered_count_reserves_work_before_formula_allocation(monkeypatch) -> None:
    allocations = 0
    original = builders.Eventually

    def counted_eventually(inner):
        nonlocal allocations
        allocations += 1
        return original(inner)

    monkeypatch.setattr(builders, "Eventually", counted_eventually)
    budget = WorkBudget(4)
    with spending(budget), pytest.raises(DFAConstructionLimit, match="work budget"):
        builders.at_least(temporal.Atom("p"), 10_000_000)

    assert allocations == 0
    assert budget.used > budget.allowed


def test_budgeted_small_count_keeps_its_formula() -> None:
    predicate = temporal.Atom("p")
    expected = builders.at_least(predicate, 3)
    budget = WorkBudget(100)
    with spending(budget):
        actual = builders.at_least(predicate, 3)

    assert actual == expected
    assert budget.used == 9


def test_variadic_flow_reserves_work_before_pair_expansion(monkeypatch) -> None:
    names = ", ".join(f'"tool_{i}"' for i in range(100))
    rule = DSLParser().parse(
        f'rule "bounded-flow" condition: called_tools({names}) after called_tools({names})'
    )
    calls = 0
    original = desugar.Desugarer.flow

    def counted_flow(self, node, scope):
        nonlocal calls
        calls += 1
        return original(self, node, scope)

    monkeypatch.setattr(desugar.Desugarer, "flow", counted_flow)
    budget = WorkBudget(4)
    with spending(budget), pytest.raises(DFAConstructionLimit, match="work budget"):
        lower._desugar(rule.condition, _Compiler(rule, None))

    assert calls == 1
    assert budget.used > budget.allowed


@pytest.mark.parametrize(
    "condition",
    [
        'called_tools("a", "b", "c") after '
        '(called_tool("w") or called_tool("x") or called_tool("y") or called_tool("z"))',
        '(called_tool("w") or called_tool("x") or called_tool("y") or called_tool("z")) '
        'after called_tools("a", "b", "c")',
    ],
)
def test_variadic_flow_counts_top_level_or_alternatives_before_expansion(
    monkeypatch, condition
) -> None:
    rule = DSLParser().parse(f'rule "or-flow" condition: {condition}')
    calls = 0
    original = desugar.Desugarer.flow

    def counted_flow(self, node, scope):
        nonlocal calls
        calls += 1
        return original(self, node, scope)

    monkeypatch.setattr(desugar.Desugarer, "flow", counted_flow)
    budget = WorkBudget(10)  # 3 names alone cost 9; four alternatives must exceed it.
    with spending(budget), pytest.raises(DFAConstructionLimit, match="work budget"):
        lower._desugar(rule.condition, _Compiler(rule, None))

    assert calls == 1
    assert budget.used == 84


def test_variadic_flow_reserves_work_for_a_large_single_operand(monkeypatch) -> None:
    predicates = " and ".join(f'm.content contains "x{i}"' for i in range(24))
    rule = DSLParser().parse(
        'rule "large-operand" condition: called_tools("a", "b", "c") after '
        f"(any m in user_messages where ({predicates}))"
    )
    calls = 0
    original = desugar.Desugarer.flow

    def counted_flow(self, node, scope):
        nonlocal calls
        calls += 1
        return original(self, node, scope)

    monkeypatch.setattr(desugar.Desugarer, "flow", counted_flow)
    budget = WorkBudget(20)  # Tool names alone cost 9; the body is repeated three times.
    with spending(budget), pytest.raises(DFAConstructionLimit, match="work budget"):
        lower._desugar(rule.condition, _Compiler(rule, None))

    assert calls == 1
    assert budget.used > budget.allowed


def test_budgeted_small_flow_keeps_its_core_formula() -> None:
    rule = DSLParser().parse(
        'rule "small-flow" condition: called_tools("a", "b") after called_tools("c", "d")'
    )
    _, expected = lower._desugar(rule.condition, _Compiler(rule, None))
    budget = WorkBudget(100)
    with spending(budget):
        _, actual = lower._desugar(rule.condition, _Compiler(rule, None))

    assert actual == expected
    assert budget.used == 20


def test_budgeted_small_flow_with_a_body_keeps_its_core_formula() -> None:
    rule = DSLParser().parse(
        'rule "small-body" condition: called_tools("a", "b") after '
        '(any m in user_messages where m.content contains "x")'
    )
    _, expected = lower._desugar(rule.condition, _Compiler(rule, None))
    budget = WorkBudget(100)
    with spending(budget):
        _, actual = lower._desugar(rule.condition, _Compiler(rule, None))

    assert actual == expected
    assert budget.used <= budget.allowed
