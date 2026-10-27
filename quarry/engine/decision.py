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

"""The compile-side shape of a rule's decision tree: what its checks read and whether its
root automaton's boot step is the tree.

The compiler closes every operand of a rule's root ``and``/``or``/``not`` into an atom of
its own (``desugar.hoist(split_root=True)``), so the rule's automaton reads only those atoms,
at the boot step. That step's guard is a reduced ordered decision diagram over them, tested
cheapest first: it is the tree, and a scan that follows it grounds only the checks on its
path. Reports of the tree and of a scan's path through it are ``quarry.decision_report``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from quarry.engine import temporal as T
from quarry.engine.contracts import CompiledRuleArtifact
from quarry.engine.symbolic_dfa import DFA, GuardLeaf, GuardNode

MATCH = "match"
NO_MATCH = "no_match"
COLLECT = "collect"


@dataclass(frozen=True)
class Parts:
    """What a check reads through: its closed parts' automata and its tools bodies."""

    closed: Mapping[str, DFA]
    tool_bodies: Mapping[str, Iterable[str]]


def artifact_parts(artifact: CompiledRuleArtifact) -> Parts:
    """The closed parts and tools bodies of a compiled rule."""

    return Parts(
        {atom_id: plan.dfa for atom_id, plan in artifact.closed_conditions.items()},
        {atom_id: T.atoms_of(plan.body) for atom_id, plan in artifact.tool_quantifiers.items()},
    )


def reads(atom_id: str, parts: Parts) -> set[str]:
    """Every atom a node's check reads, through its closed parts and tools bodies."""

    seen: set[str] = set()
    pending = [atom_id]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        closed = parts.closed.get(current)
        if closed is not None:
            pending.extend(closed.atoms)
        body = parts.tool_bodies.get(current)
        if body is not None:
            pending.extend(body)
    return seen


def reads_all(root: DFA, parts: Parts) -> set[str]:
    """Every atom a whole-rule node reads."""

    return set().union(*(reads(atom_id, parts) for atom_id in root.atoms))


def boot_leaf(dfa: DFA, target: int) -> str | None:
    """The verdict a decomposed root reaches after its boot step, or None if undecided."""

    if dfa.is_rejecting(target):
        return MATCH
    if dfa.is_accepting(target):
        return NO_MATCH
    return None


def boot_diagram_is_decided(dfa: DFA) -> bool:
    """Whether the root's boot guard is the same diagram with or without events after it,
    and every leaf decides the verdict: the condition for walking it as the tree."""

    initial = dfa.initial_state

    def shape(guard) -> object:
        if isinstance(guard, GuardLeaf):
            return boot_leaf(dfa, guard.target)
        return (guard.atom, shape(guard.hi), shape(guard.lo))

    if not isinstance(dfa.normal_guards[initial], GuardNode):
        return False
    normal = shape(dfa.normal_guards[initial])
    final = shape(dfa.final_guards[initial])

    def decided(value: object) -> bool:
        if isinstance(value, tuple):
            return decided(value[1]) and decided(value[2])
        return value is not None

    return normal == final and decided(normal)
