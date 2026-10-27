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

"""Compiled-artifact contracts and internal result types.

The stable data contracts the rest of the pipeline builds on.
A loaded rule carries a ``CompiledRuleArtifact`` (built at scanner init), and the executor
returns ``ExecutionResult`` objects. Keeping these here — dependency-free except for
the engine's own DFA type — lets the compiler, executor, and scanner agree on shapes
without importing each other.

``COMPILER_VERSION`` is stamped into every artifact and folded into its hash, so a
change in compilation strategy invalidates cached/serialized artifacts rather than
silently reusing a stale automaton.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from quarry.engine.symbolic_dfa import DFA
from quarry.engine.temporal import Formula

if TYPE_CHECKING:
    from quarry.engine.witness import ModelResultEvidence

# Bump when compilation or admission semantics change in a way that must invalidate
# previously produced artifacts. Compiler-6 admitted colliding `variables:`/`as`
# bindings and referenced captures on the sustained side of `until`.
COMPILER_VERSION = "compiler-7"


@dataclass(frozen=True, slots=True)
class ToolQuantifierPlan:
    """An any/all Boolean body evaluated coherently against each tool at boot."""

    body: Formula
    kind: Literal["any", "all"] = "any"
    # Authored preorder node for each Boolean atom leaf, in formula preorder.
    body_source_nodes: tuple[int | None, ...] = ()


@dataclass(frozen=True, slots=True)
class ClosedPlan:
    """The automaton deciding a closed part's atom: its value in a run is whether this
    automaton matches over that run's events, oracle and binding."""

    dfa: DFA


@dataclass(frozen=True, slots=True)
class TreeNode:
    """One check of a rule's decision tree: an atom the rule's automaton reads at boot.

    ``atom_id`` is a closed part's atom or a static atom, or None for a rule the compiler
    could not decompose, whose one node is its whole automaton. The atom id is a content
    hash of the check, so it is the node's identity across compiles. What the check scores,
    measures and calls is read from the atom (``quarry.decision_report``), not stored here.
    """

    atom_id: str | None
    label: str
    # The authored condition's preorder index in the rule's source graph, if one matches.
    source_node: int | None = None


@dataclass(frozen=True, slots=True)
class DecisionTree:
    """A rule's checks. Their true and false edges are the root automaton's boot guard.

    A decomposed tree's root automaton reads only its nodes' atoms, at the boot step, so the
    reduced decision diagram of that step's guard is the tree; a scan walks it and grounds
    only the checks on its path. A tree with one whole-rule node is not decomposed.
    """

    nodes: tuple[TreeNode, ...]

    @property
    def decomposed(self) -> bool:
        """Return True if the decision tree is decomposed."""
        return self.nodes[0].atom_id is not None


ObservationOrigin = Literal["verdict", "explored"]


@dataclass(frozen=True, slots=True)
class Observation:
    """One value a scan computed: an atom's value at one position.

    ``position`` is None at the boot step, where static atoms are decided. ``origin`` is
    ``"verdict"`` for a value the verdict run computed and ``"explored"`` for one computed
    only to report it; an explored value never decides the verdict or the path. ``value`` is
    None when it could not be computed, and ``reason`` says why. ``check`` marks a
    decision-tree check, whose ``evidence`` is a model's result or the size a size check
    compared.
    """

    atom_id: str
    position: int | None
    value: bool | None
    origin: ObservationOrigin = "verdict"
    reason: str | None = None
    check: bool = False
    evidence: ModelResultEvidence | int | None = None


@dataclass
class ScanRecord:
    """Every value one rule's scan computed, in the order it computed them.

    The caller owns it, so a check that fails or times out leaves the values computed
    before it. The verdict's decision path and the scan's other outputs are views of it.
    It holds no conversation text, so a host can store it. Every scan records, so an entry
    is kept as a plain tuple of the ``Observation`` fields, in field order, and becomes an
    ``Observation`` only when an output reads it.
    """

    entries: list[tuple] = field(default_factory=list)
    # The condition held but ``collect:`` kept nothing, so the rule did not match.
    empty_collection: bool = False

    def add(
        self,
        atom_id: str,
        position: int | None,
        value: bool | None,
        origin: ObservationOrigin = "verdict",
        reason: str | None = None,
        check: bool = False,
        evidence: ModelResultEvidence | int | None = None,
    ) -> None:
        """Add an observation to the scan record."""
        self.entries.append((atom_id, position, value, origin, reason, check, evidence))

    @property
    def observations(self) -> list[Observation]:
        """Return the observations of the scan record."""
        return [Observation(*entry) for entry in self.entries]

    @property
    def checks(self) -> list[Observation]:
        """The decision-tree checks, in the order the scan grounded them."""
        return [Observation(*entry) for entry in self.entries if entry[5]]


@dataclass(frozen=True)
class CompiledRuleArtifact:
    """The immutable compiled form of one rule.

    Carries the DFA, its identity hash, and the compiler version, plus the atom-grounding
    specs (how each atom is evaluated against a conversation) and any capture/register
    plan. The optional fields keep this contract stable for artifacts that omit them.
    """

    rule_name: str
    dfa: DFA
    dfa_hash: str
    compiler_version: str = COMPILER_VERSION
    atom_specs: dict = field(default_factory=dict)
    # AST-preorder node index -> the atom ids lowered from that node's subtree. The index
    # is deliberately structural rather than an AST object reference so compiled artifacts
    # remain independent of the mutable parser objects and can be serialized later.
    node_atom_ids: dict[int, tuple[str, ...]] = field(default_factory=dict)
    temporal: bool = False
    notes: tuple[str, ...] = ()
    # Value capture (``... as v`` / cross-field join): the recipe the executor uses
    # to enumerate candidate values and run the *same* DFA once per value, grounding the
    # ``capture_bind``/``capture_ref`` parameterized atoms as ``contains <value>``. None
    # for ordinary rules. Its concrete type is ``quarry.engine.captures.CapturePlan``
    # (kept as ``object`` here so contracts stays free of a captures import).
    capture_plan: object | None = None
    # Concrete items are ``quarry.types.parameters.ParameterSpec``. Kept untyped here so the
    # engine core remains dependency-free; the compiler adapter owns that boundary.
    parameter_schema: tuple = ()
    # Exhaustive typed-collection work executed after a true Boolean gate. Its concrete
    # type is ``quarry.engine.collect.CollectPlan``; kept loose for the same isolation reason.
    collect_plan: object | None = None
    tool_quantifiers: dict[str, ToolQuantifierPlan] = field(default_factory=dict)
    # Each closed part's atom (a part of a body that reads no element) and its automaton.
    closed_conditions: dict[str, ClosedPlan] = field(default_factory=dict)
    # A universal/absence claim must be explained over the original complete input.
    full_trace_proof: bool = False
    decision_tree: DecisionTree | None = None

    @property
    def num_states(self) -> int:
        """Return the number of states in the DFA."""
        return self.dfa.num_states

    @property
    def atoms(self) -> list[str]:
        """Return the atoms in the DFA."""
        return self.dfa.atoms

    @property
    def automaton_atoms(self) -> set[str]:
        """The atoms a run reads: the rule automaton's and each closed part automaton's."""
        return set(self.dfa.atoms).union(
            *(plan.dfa.atoms for plan in self.closed_conditions.values())
        )


AtomEvaluationStatus = Literal["evaluated_true", "evaluated_false", "skipped"]
AttemptKind = Literal["default", "tool", "capture", "capture_tool"]


@dataclass(frozen=True, slots=True)
class AtomEvaluation:
    """One atom's disposition in one attempt.

    ``witness`` marks clear evidence from a selected transition. An unmarked evaluation
    may still matter when the selected path is too ambiguous to attribute locally.
    """

    atom_id: str
    position: int
    status: AtomEvaluationStatus
    witness: bool = False
    source_node_index: int | None = None
    body_occurrence: int | None = None

    @property
    def value(self) -> bool | None:
        """Return the value of the atom evaluation."""
        if self.status == "evaluated_true":
            return True
        if self.status == "evaluated_false":
            return False
        return None

    def to_dict(self) -> dict[str, object]:
        """Return the dictionary representation of the atom evaluation."""
        return {
            "atom_id": self.atom_id,
            "position": self.position,
            "status": self.status,
            "value": self.value,
            "witness": self.witness,
            "source_node_index": self.source_node_index,
            "body_occurrence": self.body_occurrence,
        }


@dataclass(frozen=True, slots=True)
class ExecutionAttempt:
    """One coherent rule run under one default, tool, or capture binding.

    For top-level attempts, ``matched`` is the final full-rule verdict, including post-DFA
    gates such as ``collect:``. A nested ``tool_attempts`` entry is a tool-body check:
    its ``matched`` describes that body's result, while ``selected`` means that check
    contributed to the enclosing quantifier. An existential uses one success or every
    failure; a universal uses every success or one failure.
    """

    attempt_id: str
    kind: AttemptKind
    binding_index: int | None
    matched: bool
    evaluations: tuple[AtomEvaluation, ...] = ()
    selected: bool = False
    capture_binding_index: int | None = None
    tool_binding_index: int | None = None
    tool_attempts: tuple[ExecutionAttempt, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return the dictionary representation of the execution attempt."""
        return {
            "attempt_id": self.attempt_id,
            "kind": self.kind,
            "binding_index": self.binding_index,
            "capture_binding_index": self.capture_binding_index,
            "tool_binding_index": self.tool_binding_index,
            "matched": self.matched,
            "selected": self.selected,
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "tool_attempts": [attempt.to_dict() for attempt in self.tool_attempts],
        }


@dataclass(frozen=True, slots=True)
class WitnessTrace:
    """The canonical runtime trace used to explain one rule verdict.

    Alternative bindings remain separate attempts. ``attempts`` records the original
    execution; ``proof_attempt`` records the minimized replay when one was run. The replay
    may choose a different binding, so ``selected_attempt`` prefers its proven path. In that
    case ``selected_attempt_id`` names ``proof_attempt``, which may not appear in
    ``attempts``. Without a replay it names the matched entry of ``attempts``.
    Universal and absence claims use the original input, because deleting elements
    changes the domain being claimed over. ``witness`` marks locally established checks;
    an ambiguous path can have cited positions without any flagged atoms. With explanation
    disabled, a raw path is observable history without proof flags. Skipped atoms are
    recorded explicitly but can never be proof evidence.
    """

    positions: tuple[int, ...] = ()
    attempts: tuple[ExecutionAttempt, ...] = ()
    selected_attempt_id: str | None = None
    minimized: bool = False
    proof_attempt: ExecutionAttempt | None = None
    # Unmatched attempts past the retention limit are counted instead of kept.
    omitted_attempts: int = 0

    @property
    def selected_attempt(self) -> ExecutionAttempt | None:
        """Return the selected attempt."""
        if self.proof_attempt is not None:
            return self.proof_attempt
        if self.selected_attempt_id is not None:
            return next(
                (
                    attempt
                    for attempt in self.attempts
                    if attempt.attempt_id == self.selected_attempt_id
                ),
                None,
            )
        # A single non-matching/default attempt is still an unambiguous observation path.
        return self.attempts[0] if len(self.attempts) == 1 else None

    def to_dict(self) -> dict[str, object]:
        """Return the dictionary representation of the witness trace."""
        return {
            "positions": list(self.positions),
            "selected_attempt_id": self.selected_attempt_id,
            "minimized": self.minimized,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "omitted_attempts": self.omitted_attempts,
            "proof_attempt": self.proof_attempt.to_dict() if self.proof_attempt else None,
        }


@dataclass(frozen=True)
class ExecutionResult:
    """Internal result of running a compiled artifact over one conversation.

    ``matched`` mirrors the DSL semantics: a rule matches when its *safety* formula is
    violated (the DFA reaches a rejecting state) — i.e. ``matched = not satisfied``.
    ``witness.positions`` are selected evidence positions for a match. Existential paths
    may be minimized by replay; universal and absence paths cite visited events in the
    original domain. With explanation disabled, positions are the raw
    matched path. Positions are empty when the rule does not match.
    ``captured_value`` is the winning ``... as v`` value for a capture rule (the substring
    that satisfied the bind→test), else ``None``. It follows the proof: with witness
    minimization on, it is the replay's binding, which can differ from the first binding
    the original run matched; with minimization off, it is that first binding.
    """

    rule_name: str
    matched: bool
    satisfied: bool
    witness: WitnessTrace = field(default_factory=WitnessTrace)
    external_calls: int = 0
    warnings: tuple[str, ...] = ()
    error: str | None = None
    captured_value: str | None = None
    model_results: tuple[ModelResultEvidence, ...] = ()
    output: object | None = None
    # With deferred minimization: ``await refine(time_left)`` returns this result with a
    # reduced witness, reusing the verdict run's grounding. None when nothing is deferred.
    refine: Callable[[Callable[[], float | None] | None], Awaitable[ExecutionResult]] | None = (
        field(default=None, compare=False, repr=False)
    )
    # With a scan record: ``await explore(full, time_left)`` is the second pass, which
    # fills the record with the values the verdict did not need. None without a record.
    explore: Callable[[bool, Callable[[], float | None] | None], Awaitable[None]] | None = field(
        default=None, compare=False, repr=False
    )
