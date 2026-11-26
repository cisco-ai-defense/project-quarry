# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Rule composition is bounded before copying and shares its caller's compile work."""

import pytest

from quarry.dsl import composition
from quarry.dsl.parser import DSLParser
from quarry.engine.budget import WorkBudget, spending
from quarry.engine.lower import compile_core
from quarry.types.exceptions import RuleValidationError


def module(source):
    rules = DSLParser().parse_many(source)
    return {rule.name: rule for rule in rules}


def test_expansion_is_rejected_before_copying_a_branching_call_graph(monkeypatch):
    source = 'rule "Leaf" condition: len(messages) > 0'
    previous = "Leaf"
    for number in range(5):
        name = f"Caller{number}"
        source += (
            f'\nrule "{name}" imports: leaf="{previous}" '
            "condition: leaf(messages) and leaf(messages[1:])"
        )
        previous = name
    rules = module(source)
    monkeypatch.setattr(composition, "MAX_LINKED_NODES", 100, raising=False)
    copies = 0
    original = composition.copy.deepcopy

    def counted_copy(*args, **kwargs):
        nonlocal copies
        copies += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(composition.copy, "deepcopy", counted_copy)
    with pytest.raises(RuleValidationError, match="expansion") as failure:
        composition.link_public_rule_calls(rules, {})
    assert failure.value.details["reason"] == "expansion_limit"
    assert copies == 0


def test_string_copies_also_consume_the_composition_budget(monkeypatch):
    strings = " ".join(f'$p{i}="needle{i}"' for i in range(8))
    source = (
        f'rule "Leaf" strings: {strings} condition: true\n'
        'rule "Caller" imports: leaf="Leaf" condition: leaf(messages)'
    )
    rules = module(source)
    monkeypatch.setattr(composition, "MAX_LINKED_NODES", 15, raising=False)
    with pytest.raises(RuleValidationError, match="expansion"):
        composition.link_public_rule_calls(rules, {})


def test_compilation_counts_work_in_its_enclosing_budget():
    rule = DSLParser().parse('rule "R" condition: user_said("needle")')
    budget = WorkBudget(None)
    with spending(budget):
        result = compile_core(rule)
    assert result.ok, result.diagnostics.errors
    assert budget.used > 0


def test_small_composition_retains_a_valid_compiled_check(monkeypatch):
    rules = module(
        'rule "Leaf" condition: len(messages)>0\n'
        'rule "Caller" imports: leaf="Leaf" condition: leaf(messages)'
    )
    monkeypatch.setattr(composition, "MAX_LINKED_NODES", 100, raising=False)
    linked = composition.link_public_rule_calls(rules, {})
    result = compile_core(linked["Caller"])
    assert result.ok, result.diagnostics.errors
    assert result.artifact.split_checks
