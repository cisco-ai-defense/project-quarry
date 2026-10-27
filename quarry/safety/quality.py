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

"""Conservative, advisory detection-quality checks for compiled Quarry rules."""

from __future__ import annotations

import math

from quarry.engine.atoms import AtomDomain, AtomSpec, Cost
from quarry.engine.contracts import CompiledRuleArtifact
from quarry.engine.matchers import compare
from quarry.engine.symbolic_dfa import DFA
from quarry.engine.temporal import atoms_of
from quarry.safety.validator import Severity, ValidationWarning
from quarry.types.rules import DSLRule

_MAX_COUNT_CASES = 32
_COUNT_OPS = frozenset({"=", "==", "!=", ">", "<", ">=", "<="})


def _constant_count_value(spec: AtomSpec) -> bool | None:
    """Prove a comparison from the fact that every collection size is nonnegative."""
    if spec.predicate != "surface_count" or spec.number is None:
        return None
    bound = spec.number
    if not math.isfinite(bound):
        return None
    if spec.op == ">=":
        return True if bound <= 0 else None
    if spec.op == ">":
        return True if bound < 0 else None
    if spec.op == "<=":
        return False if bound < 0 else None
    if spec.op == "<":
        return False if bound <= 0 else None
    if spec.op in {"=", "=="}:
        return False if bound < 0 else None
    if spec.op == "!=":
        return True if bound < 0 else None
    return None


def _count_cases(artifact: CompiledRuleArtifact) -> list[dict[str, bool]]:
    """Enumerate bounded, sound count valuations over nonnegative integer sizes.

    Comparisons of one collection size share an integer. Their truth values can change
    only at a comparison bound or the next integer. Distinct collections are treated
    independently, which can miss a proof but cannot create a false one. If the product
    grows too large, leave remaining groups free instead of dropping possibilities.
    """
    base = {
        atom_id: value
        for atom_id, spec in artifact.atom_specs.items()
        if (value := _constant_count_value(spec)) is not None
    }
    groups: dict[tuple, list[tuple[str, AtomSpec]]] = {}
    for atom_id, spec in artifact.atom_specs.items():
        if (
            spec.predicate != "surface_count"
            or spec.number is None
            or not math.isfinite(spec.number)
            or spec.op not in _COUNT_OPS
        ):
            continue
        key = (
            spec.selector,
            spec.patterns,
            spec.value_expr_key,
            spec.parameter_names,
            spec.is_slice,
            spec.slice_start,
            spec.slice_stop,
        )
        groups.setdefault(key, []).append((atom_id, spec))

    alternatives = []
    for key, atoms in groups.items():
        representatives = {0}
        for _, spec in atoms:
            assert spec.number is not None
            lower = math.floor(spec.number)
            representatives.update(n for n in (lower, lower + 1) if n >= 0)
        assignments = {
            tuple(compare(size, spec.op, spec.number) for _, spec in atoms)
            for size in representatives
        }
        alternatives.append(
            (
                key,
                [
                    dict(zip((atom_id for atom_id, _ in atoms), values, strict=True))
                    for values in sorted(assignments)
                ],
            )
        )

    cases = [base]
    for _, assignments in sorted(alternatives, key=lambda item: (len(item[1]), repr(item[0]))):
        if len(cases) * len(assignments) > _MAX_COUNT_CASES:
            continue
        cases = [{**case, **assignment} for case in cases for assignment in assignments]
    return cases


def _with_constant_closed_parts(
    artifact: CompiledRuleArtifact, fixed: dict[str, bool]
) -> dict[str, bool]:
    """Propagate proven constant closed checks into the root DFA."""
    fixed = dict(fixed)
    pending = dict(artifact.closed_conditions)
    changed = True
    while changed:
        changed = False
        for atom_id, plan in list(pending.items()):
            can_match = plan.dfa.can_reject_with(fixed)
            can_be_clean = plan.dfa.can_accept_with(fixed)
            if can_match == can_be_clean:
                continue
            fixed[atom_id] = can_match
            del pending[atom_id]
            changed = True
    return fixed


def _used_dfa_atoms(dfa: DFA) -> set[str]:
    """Atoms that can affect a transition in any compiled state."""
    return {
        atom_id
        for state_id in range(dfa.num_states)
        for is_last in (False, True)
        for atom_id in dfa.relevant_atoms(state_id, is_last=is_last)
    }


def _potentially_relevant_atoms(artifact: CompiledRuleArtifact) -> set[str]:
    """Follow only checks the compiled verdict can read, including derived checks."""
    # Candidate enumeration can supply its own signal outside the DFA transitions.
    if artifact.capture_plan is not None:
        return set(artifact.atom_specs)
    pending = list(_used_dfa_atoms(artifact.dfa))
    relevant: set[str] = set()
    while pending:
        atom_id = pending.pop()
        if atom_id in relevant:
            continue
        relevant.add(atom_id)
        if plan := artifact.closed_conditions.get(atom_id):
            pending.extend(_used_dfa_atoms(plan.dfa))
        if plan := artifact.tool_quantifiers.get(atom_id):
            pending.extend(atoms_of(plan.body))
    return relevant


def _has_event_anchor(artifact: CompiledRuleArtifact) -> bool:
    # Bare on_surface and at_candidate atoms only prove that an event exists. Named
    # tool counts are static aggregates, but identify a particular call or response.
    # Model and system-prompt content checks also inspect a detection signal.
    relevant = _potentially_relevant_atoms(artifact)
    return any(
        (spec.domain is AtomDomain.EVENT and spec.predicate not in {"on_surface", "at_candidate"})
        or (
            spec.predicate == "surface_count"
            and spec.selector in {"tool_calls", "tool_responses"}
            and bool(spec.patterns)
        )
        or spec.cost in {Cost.MODEL, Cost.LLM, Cost.YARA}
        or (spec.selector == "system_prompt" and spec.predicate != "surface_count")
        or (
            spec.selector == "tools"
            and spec.predicate not in {"surface_count", "tool_quantifier", "tool_names_all"}
            and spec.value_expr_key != "field:name"
        )
        for atom_id, spec in artifact.atom_specs.items()
        if atom_id in relevant
    )


def quality_warnings(rule: DSLRule, artifact: CompiledRuleArtifact) -> list[ValidationWarning]:
    """Report proven constant verdicts and rules without an event/content anchor.

    Free atoms may take either value, so reachability can miss a constant verdict but
    cannot invent one. Warnings do not change compilation or scan decisions.
    """
    cases = [_with_constant_closed_parts(artifact, case) for case in _count_cases(artifact)]
    if not any(artifact.dfa.can_reject_with(case) for case in cases):
        return [
            ValidationWarning(
                Severity.HIGH,
                f"Rule '{rule.name}' has a condition that is always false for complete scans "
                "(vacuous condition).",
                "Remove contradictory conditions or correct the comparison bounds.",
            )
        ]
    # A broad condition can intentionally admit candidates for a collect clause; the
    # collector may still yield no output. Likewise, a per-binding DFA may reject for
    # every binding even when a conversation supplies no candidate binding at all.
    if (
        rule.collect is None
        and artifact.capture_plan is None
        and not any(artifact.dfa.can_accept_with(case) for case in cases)
    ):
        return [
            ValidationWarning(
                Severity.HIGH,
                f"Rule '{rule.name}' has a condition that is always true for complete scans "
                "(tautology).",
                "Add a condition that checks a message, tool call, or other detection signal.",
            )
        ]
    # A collect rule's signal can live entirely in the collector, outside the condition
    # DFA and its atom specs. A broad condition is intentional for such extraction.
    if rule.collect is None and not _has_event_anchor(artifact):
        return [
            ValidationWarning(
                Severity.MEDIUM,
                f"Rule '{rule.name}' has no specific event or message-content anchor.",
                "Add a tool-call, flow, or message-content predicate to identify the behavior.",
            )
        ]
    return []
