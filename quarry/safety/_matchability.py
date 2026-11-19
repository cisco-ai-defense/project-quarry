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

"""Bounded formal search for impossible string-message rules.

This first verifier slice is deliberately one-sided: exhausting every feasible DFA path
proves that a supported rule cannot match, while finding a feasible path returns UNKNOWN
until a later slice can construct, validate, and replay a concrete witness.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from time import monotonic
from typing import Any

from quarry.engine.atoms import AtomDomain, AtomSpec, StringSemantics
from quarry.engine.contracts import CompiledRuleArtifact
from quarry.engine.symbolic_dfa import Guard, GuardLeaf
from quarry.safety.quality import MatchabilityAnswer, MatchabilityBasis, MatchabilityResult

_SUPPORTED_SELECTORS = frozenset({"messages", "user_messages", "assistant_messages"})
_SUPPORTED_PREDICATES = frozenset({"on_surface", "equals", "contains_any"})
_ROLES = ("system", "developer", "user", "assistant", "tool")
_MAX_GUARD_VISITS = 65_536
_MAX_CONSTRUCTION_ATOMS = 256
_MAX_CONSTRUCTION_PATTERNS = 64
_MAX_LITERAL_CHARS = 4096
_MAX_TOTAL_LITERAL_CHARS = 16_384
_CONSTRUCTION_LIMIT_REASON = "formal verification constraint-construction limit exceeded"


class _SearchIncomplete(RuntimeError):
    """The bounded search could not establish a complete answer."""


@dataclass(slots=True)
class _Budget:
    max_solver_checks: int
    deadline: float
    solver_checks: int = 0
    guard_visits: int = 0

    def check_guard_visit(self) -> None:
        """Bound guard traversal independently from solver work."""
        if self.guard_visits >= _MAX_GUARD_VISITS:
            raise _SearchIncomplete("formal verification guard-traversal budget exhausted")
        if monotonic() >= self.deadline:
            raise _SearchIncomplete("formal verification time budget exhausted")
        self.guard_visits += 1

    def next_solver_timeout_ms(self) -> int:
        """Consume one solver check and return its remaining wall-clock budget."""
        if self.solver_checks >= self.max_solver_checks:
            raise _SearchIncomplete("formal verification solver-check budget exhausted")
        remaining_ms = int((self.deadline - monotonic()) * 1000)
        if remaining_ms <= 0:
            raise _SearchIncomplete("formal verification time budget exhausted")
        self.solver_checks += 1
        return max(1, remaining_ms)


def _unsupported_reason(artifact: CompiledRuleArtifact) -> str | None:
    if artifact.capture_plan is not None:
        return "capture bindings"
    if artifact.collect_plan is not None:
        return "collection output"
    if artifact.parameter_schema:
        return "rule parameters"
    if artifact.closed_conditions:
        return "decomposed decision-tree checks"
    if artifact.tool_quantifiers:
        return "tool quantifiers"

    default_semantics = StringSemantics()
    for atom_id in artifact.dfa.atoms:
        spec = artifact.atom_specs.get(atom_id)
        if spec is None:
            return "an atom without a grounding specification"
        if spec.domain is not AtomDomain.EVENT:
            return f"static atom '{spec.predicate}'"
        if spec.selector not in _SUPPORTED_SELECTORS:
            return f"surface '{spec.selector}'"
        if spec.predicate not in _SUPPORTED_PREDICATES:
            return f"predicate '{spec.predicate}'"
        if spec.parameter_names or spec.ordinal is not None or spec.is_slice:
            return "binding or positional constraints"
        if spec.predicate != "on_surface" and spec.value_expr_key != "field:content":
            return f"field '{spec.value_expr_key}'"
        if spec.string_semantics != default_semantics:
            return "string modifiers or transforms"
        if spec.predicate == "equals" and len(spec.patterns) != 1:
            return "an equality atom without one literal"
    return None


def _literal_construction_limit_exceeded(artifact: CompiledRuleArtifact) -> bool:
    """Check literal work using O(1) string lengths before constructing solver terms."""
    pattern_count = 0
    total_chars = 0
    for atom_id in artifact.dfa.atoms:
        spec = artifact.atom_specs[atom_id]
        pattern_count += len(spec.patterns)
        if pattern_count > _MAX_CONSTRUCTION_PATTERNS:
            return True
        for pattern in spec.patterns:
            chars = len(pattern)
            if chars > _MAX_LITERAL_CHARS:
                return True
            total_chars += chars
            if total_chars > _MAX_TOTAL_LITERAL_CHARS:
                return True
    return False


def _guard_paths(
    guard: Guard,
    budget: _Budget,
) -> Iterator[tuple[int, dict[str, bool]]]:
    """Yield each guard leaf with the partial atom valuation that reaches it."""
    pending: list[tuple[Guard, dict[str, bool]]] = [(guard, {})]
    while pending:
        budget.check_guard_visit()
        current, required = pending.pop()
        if isinstance(current, GuardLeaf):
            yield current.target, required
            continue
        pending.append((current.lo, {**required, current.atom: False}))
        pending.append((current.hi, {**required, current.atom: True}))


class _Verifier:
    def __init__(
        self,
        z3: Any,
        *,
        max_solver_checks: int,
        max_states: int,
        timeout_ms: int,
    ) -> None:
        self._z3 = z3
        self._max_states = max_states
        self._budget = _Budget(
            max_solver_checks=max_solver_checks,
            deadline=monotonic() + timeout_ms / 1000,
        )
        self._ctx: Any | None = None
        self._literal_alphabet: dict[int, str] = {}

    def run(self, artifact: CompiledRuleArtifact) -> MatchabilityResult:
        if artifact.dfa.num_states > self._max_states:
            return MatchabilityResult(
                MatchabilityAnswer.UNKNOWN,
                reason=(
                    f"formal verification state budget exceeded "
                    f"({artifact.dfa.num_states} > {self._max_states})"
                ),
            )
        if len(artifact.dfa.atoms) > _MAX_CONSTRUCTION_ATOMS:
            return MatchabilityResult(
                MatchabilityAnswer.UNKNOWN,
                reason=_CONSTRUCTION_LIMIT_REASON,
            )
        try:
            abstract_match = self._has_abstract_match(artifact)
        except _SearchIncomplete as exc:
            return MatchabilityResult(MatchabilityAnswer.UNKNOWN, reason=str(exc))
        if not abstract_match:
            return MatchabilityResult(
                MatchabilityAnswer.NO,
                basis=MatchabilityBasis.ABSTRACT_DFA,
                reason="no matching DFA verdict is reachable under any atom valuation",
            )

        if reason := _unsupported_reason(artifact):
            return MatchabilityResult(
                MatchabilityAnswer.UNKNOWN,
                reason=f"formal verification does not support {reason}",
            )
        if _literal_construction_limit_exceeded(artifact):
            return MatchabilityResult(
                MatchabilityAnswer.UNKNOWN,
                reason=_CONSTRUCTION_LIMIT_REASON,
            )

        # Equality and substring predicates are invariant under a one-to-one
        # renaming of the characters appearing in their literals. Use a bounded,
        # controlled alphabet so StringVal cannot reinterpret a literal backslash
        # escape or an out-of-alphabet Unicode scalar as solver syntax. Characters
        # absent from every literal may map to any fresh solver character: they
        # cannot make a literal equality or substring predicate true.
        characters = dict.fromkeys(
            character
            for atom_id in artifact.dfa.atoms
            for pattern in artifact.atom_specs[atom_id].patterns
            for character in pattern
        )
        self._literal_alphabet = {
            ord(character): chr(0x100 + index) for index, character in enumerate(characters)
        }

        try:
            feasible_match = self._has_feasible_match(artifact)
        except _SearchIncomplete as exc:
            return MatchabilityResult(MatchabilityAnswer.UNKNOWN, reason=str(exc))

        if feasible_match:
            return MatchabilityResult(
                MatchabilityAnswer.UNKNOWN,
                reason=(
                    "a feasible model path exists; concrete witness replay is outside "
                    "this verifier slice"
                ),
            )
        return MatchabilityResult(
            MatchabilityAnswer.NO,
            basis=MatchabilityBasis.CONTENT_MODEL,
            reason=(
                "no feasible matching trace exists for the supported string-message predicates"
            ),
        )

    def _has_abstract_match(self, artifact: CompiledRuleArtifact) -> bool:
        """Search DFA reachability while charging every guard node to the shared budget."""
        dfa = artifact.dfa
        if dfa.is_rejecting(dfa.initial_state):
            return True

        pending = deque([dfa.initial_state])
        visited = {dfa.initial_state}
        while pending:
            state = pending.popleft()
            for is_last in (True, False):
                guard = (dfa.final_guards if is_last else dfa.normal_guards)[state]
                for target, _ in _guard_paths(guard, self._budget):
                    if dfa.is_rejecting(target):
                        return True
                    if not is_last and target not in visited:
                        visited.add(target)
                        pending.append(target)
        return False

    def _has_feasible_match(self, artifact: CompiledRuleArtifact) -> bool:
        dfa = artifact.dfa
        if dfa.is_rejecting(dfa.initial_state):
            return True

        pending = deque([dfa.initial_state])
        visited = {dfa.initial_state}
        while pending:
            state = pending.popleft()
            for is_last in (True, False):
                guard = (dfa.final_guards if is_last else dfa.normal_guards)[state]
                for target, required in _guard_paths(guard, self._budget):
                    if not self._is_feasible(artifact, required):
                        continue
                    if dfa.is_rejecting(target):
                        return True
                    if not is_last and target not in visited:
                        visited.add(target)
                        pending.append(target)
        return False

    def _is_feasible(
        self,
        artifact: CompiledRuleArtifact,
        required: dict[str, bool],
    ) -> bool:
        z3 = self._z3
        if self._ctx is None:
            self._ctx = z3.Context()
        ctx = self._ctx
        solver = z3.Solver(ctx=ctx)
        content = z3.String("content", ctx=ctx)
        role = z3.String("role", ctx=ctx)
        solver.add(z3.Or(*(role == z3.StringVal(item, ctx=ctx) for item in _ROLES)))

        for atom_id, wanted in required.items():
            spec = artifact.atom_specs[atom_id]
            constraint = self._atom_constraint(spec, content, role, ctx)
            solver.add(constraint if wanted else z3.Not(constraint))

        solver.set(timeout=self._budget.next_solver_timeout_ms())
        result = solver.check()
        if result == z3.sat:
            return True
        if result == z3.unsat:
            return False
        raise _SearchIncomplete("formal verification solver returned an unknown result")

    def _atom_constraint(self, spec: AtomSpec, content: Any, role: Any, ctx: Any) -> Any:
        z3 = self._z3
        if spec.selector == "user_messages":
            surface = role == z3.StringVal("user", ctx=ctx)
        elif spec.selector == "assistant_messages":
            surface = role == z3.StringVal("assistant", ctx=ctx)
        else:
            surface = z3.BoolVal(True, ctx=ctx)

        if spec.predicate == "on_surface":
            predicate = z3.BoolVal(True, ctx=ctx)
        elif spec.predicate == "equals":
            predicate = content == z3.StringVal(
                spec.patterns[0].translate(self._literal_alphabet), ctx=ctx
            )
        else:
            alternatives = [
                z3.Contains(
                    content, z3.StringVal(pattern.translate(self._literal_alphabet), ctx=ctx)
                )
                for pattern in spec.patterns
                if pattern
            ]
            predicate = z3.Or(*alternatives) if alternatives else z3.BoolVal(False, ctx=ctx)
        return z3.And(surface, predicate)


def verify(
    artifact: CompiledRuleArtifact,
    z3: Any,
    *,
    max_solver_checks: int,
    max_states: int,
    timeout_ms: int,
) -> MatchabilityResult:
    """Run the bounded formal verifier with an already imported solver module."""
    return _Verifier(
        z3,
        max_solver_checks=max_solver_checks,
        max_states=max_states,
        timeout_ms=timeout_ms,
    ).run(artifact)
