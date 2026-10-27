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

"""Batch DFA executor.

Runs a :class:`CompiledRuleArtifact` over an :class:`EvaluationInput`, driving the
guard-graph DFA and grounding atoms **lazily** through the :class:`Oracle`. One path
honors every executor contract:

- **Lazy grounding (guard hooks):** at each state only the atoms on the active guard path
  are grounded (``dfa.step`` walks the reduced guard graph; an atom off the path is
  never requested). Cheap-first ordering falls out of the guard atom order (PURE atoms
  are minted first) — the oracle caps external calls further via caching.
- **Boot step:** static atoms are grounded at a synthetic boot step
  (position −1) that is **not** counted in event distances; the DFA steps boot → events.
  Compiled tools quantifiers check coherent tool bindings at boot, then feed one Boolean
  result into the timeline DFA.
- **Capture instantiation:** a rule with a ``capture_plan`` runs the *same* DFA
  once per candidate value (candidates from ``captures.iter_capture_values``),
  first-accepting wins.

This is the batch executor with an injectable oracle, which makes the whole corpus
differential-testable end-to-end.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import NamedTuple

from quarry.engine import captures as _cap
from quarry.engine import temporal as T
from quarry.engine import witness as _witness
from quarry.engine.atoms import AtomDomain, AtomSpec, Cost
from quarry.engine.atoms import is_per_tool_static as _is_per_tool_static
from quarry.engine.contracts import (
    AtomEvaluation,
    AttemptKind,
    CompiledRuleArtifact,
    ExecutionAttempt,
    ExecutionResult,
    ObservationOrigin,
    ScanRecord,
    WitnessTrace,
)
from quarry.engine.decision import artifact_parts
from quarry.engine.exceptions import UnsupportedExecution
from quarry.engine.oracle import BOOT, Oracle
from quarry.engine.pacing import Pacer
from quarry.engine.symbolic_dfa import GuardNode
from quarry.types.exceptions import EvaluationError, EvaluationTimeoutError

_Transition = tuple[int, int, bool, dict[str, bool], int]
# A nested tool check and the body occurrences its Boolean proof used.
_ToolCheck = tuple[ExecutionAttempt, frozenset[int]]
# A closed part's run within one attempt: its atom, its value and the transitions it took.
_ClosedRun = tuple[str, bool, tuple[_Transition, ...]]


@dataclass
class _CausalEvaluations:
    """Keep timeline evidence separate from each coherent tool check."""

    timeline: set[tuple[str, int]] = field(default_factory=set)
    tool_timeline: set[tuple[str, int | None, int]] = field(default_factory=set)
    tool_checks: set[tuple[str, int]] = field(default_factory=set)
    selected_quantifiers: set[str] = field(default_factory=set)


_UNMATCHED_ATTEMPTS_LIMIT = 16


@dataclass
class _TraceBuilder:
    """Mutable executor-local builder for the immutable canonical witness trace."""

    attempts: list[ExecutionAttempt] = field(default_factory=list)
    selected_attempt_id: str | None = None
    transitions: dict[str, tuple[_Transition, ...]] = field(default_factory=dict)
    # Per matched attempt: the runs of the closed parts it read.
    closed_runs: dict[str, tuple[_ClosedRun, ...]] = field(default_factory=dict)
    # Per matched attempt: each nested tool check's proof occurrences, by check id.
    tool_proofs: dict[str, dict[str, frozenset[int]]] = field(default_factory=dict)
    unmatched_attempts: int = 0
    omitted_attempts: int = 0

    def add_attempt(
        self,
        *,
        kind: AttemptKind,
        binding_index: int | None,
        matched: bool,
        evaluations: list[AtomEvaluation],
        transitions: list[_Transition] | None = None,
        tool_attempts: list[_ToolCheck] | None = None,
        closed_runs: list[_ClosedRun] | None = None,
        capture_binding_index: int | None = None,
        tool_binding_index: int | None = None,
    ) -> str:
        """Add an attempt to the trace builder."""
        attempt_id = (
            f"capture:{capture_binding_index}/tool:{tool_binding_index}"
            if kind == "capture_tool"
            else f"{kind}:{0 if binding_index is None else binding_index}"
        )
        if not matched:
            if self.unmatched_attempts >= _UNMATCHED_ATTEMPTS_LIMIT:
                self.omitted_attempts += 1
                return attempt_id
            self.unmatched_attempts += 1
        self.attempts.append(
            ExecutionAttempt(
                attempt_id=attempt_id,
                kind=kind,
                binding_index=binding_index,
                matched=matched,
                evaluations=tuple(evaluations),
                capture_binding_index=capture_binding_index,
                tool_binding_index=tool_binding_index,
                tool_attempts=tuple(check for check, _ in tool_attempts or ()),
            )
        )
        # Only a matched attempt can be selected, so only its path feeds the proof.
        if matched:
            if transitions is not None:
                self.transitions[attempt_id] = tuple(transitions)
            self.closed_runs[attempt_id] = tuple(closed_runs or ())
            self.tool_proofs[attempt_id] = {
                check.attempt_id: proof for check, proof in tool_attempts or ()
            }
        return attempt_id

    def select(self, attempt_id: str) -> None:
        """Select an attempt."""
        self.selected_attempt_id = attempt_id

    def build(
        self,
        positions: list[int],
        *,
        matched: bool,
        minimized: bool,
        causal_evaluations: _CausalEvaluations | None = None,
        proof_builder: _TraceBuilder | None = None,
    ) -> WitnessTrace:
        """Build a witness trace."""
        selected_id = (
            (
                proof_builder.selected_attempt_id
                if proof_builder is not None
                else self.selected_attempt_id
            )
            if matched
            else None
        )
        # The selected proof may use a minimized replay or the complete original run
        # when deletion would change a universal/absence claim's input domain.
        causal = causal_evaluations
        selected_tool_prefixes = tuple(
            f"quantifier:{atom_id}/tool:"
            for atom_id in (causal.selected_quantifiers if causal else ())
        )

        def mark_tool_checks(attempt: ExecutionAttempt, parent_selected: bool):
            checks = []
            for check in attempt.tool_attempts:
                active = (
                    parent_selected
                    and check.selected
                    and check.attempt_id.startswith(selected_tool_prefixes)
                )
                checks.append(
                    replace(
                        check,
                        selected=active,
                        evaluations=check.evaluations
                        if causal is None
                        else tuple(
                            replace(
                                evaluation,
                                witness=(
                                    active
                                    and evaluation.status != "skipped"
                                    and (
                                        check.attempt_id,
                                        evaluation.body_occurrence,
                                    )
                                    in causal.tool_checks
                                ),
                            )
                            for evaluation in check.evaluations
                        ),
                    )
                )
            return tuple(checks)

        attempts = []
        for attempt in self.attempts:
            # The proof replay may select a binding that was tried but did not match on
            # the original input. Such an observation must not become a selected match.
            selected = attempt.attempt_id == selected_id and attempt.matched
            # Recorded evaluations start unflagged, so without a proof there is nothing to mark.
            evaluations = (
                attempt.evaluations
                if causal is None
                else tuple(
                    replace(
                        evaluation,
                        witness=(
                            selected
                            and proof_builder is None
                            and evaluation.status != "skipped"
                            and (
                                (evaluation.atom_id, evaluation.position) in causal.timeline
                                if evaluation.body_occurrence is None
                                else (
                                    evaluation.atom_id,
                                    evaluation.source_node_index,
                                    evaluation.body_occurrence,
                                )
                                in causal.tool_timeline
                            )
                        ),
                    )
                    for evaluation in attempt.evaluations
                )
            )
            attempts.append(
                replace(
                    attempt,
                    # An attempt is part of the full rule execution, so a later gate such as
                    # an empty ``collect:`` result must veto its provisional DFA match too.
                    matched=matched and attempt.matched,
                    selected=selected,
                    evaluations=evaluations,
                    tool_attempts=mark_tool_checks(attempt, selected and proof_builder is None),
                )
            )
        proof_attempt = None
        if proof_builder is not None and selected_id is not None:
            replay = next(
                attempt for attempt in proof_builder.attempts if attempt.attempt_id == selected_id
            )
            proof_attempt = replace(
                replay,
                selected=True,
                evaluations=tuple(
                    replace(
                        evaluation,
                        witness=(
                            evaluation.status != "skipped"
                            and causal is not None
                            and (
                                (evaluation.atom_id, evaluation.position) in causal.timeline
                                if evaluation.body_occurrence is None
                                else (
                                    evaluation.atom_id,
                                    evaluation.source_node_index,
                                    evaluation.body_occurrence,
                                )
                                in causal.tool_timeline
                            )
                        ),
                    )
                    for evaluation in replay.evaluations
                ),
                tool_attempts=mark_tool_checks(replay, True),
            )
        return WitnessTrace(
            positions=tuple(positions),
            attempts=tuple(attempts),
            selected_attempt_id=selected_id,
            minimized=minimized,
            proof_attempt=proof_attempt,
            omitted_attempts=self.omitted_attempts,
        )


def _domain_positions(artifact, trace_builder: _TraceBuilder, events) -> list[int]:
    """Cite every visited event in a universal or absence claim's authored domains."""

    if trace_builder.selected_attempt_id is None:
        return []
    selectors = {
        spec.selector
        for spec in artifact.atom_specs.values()
        if spec.domain is AtomDomain.EVENT and spec.selector is not None
    }
    visited = {
        position for _, position, _, _, _ in _attempt_transitions(trace_builder) if position >= 0
    }
    return [
        event.position
        for event in events
        if event.position in visited and selectors.intersection(event.surface_tags)
    ]


def _attempt_transitions(trace_builder: _TraceBuilder) -> list[_Transition]:
    """The selected attempt's transitions: its own and those of the closed parts it read."""
    attempt_id = trace_builder.selected_attempt_id
    if attempt_id is None:
        return []
    return [
        *trace_builder.transitions.get(attempt_id, ()),
        *(
            transition
            for _, _, path in trace_builder.closed_runs.get(attempt_id, ())
            for transition in path
        ),
    ]


def _transition_literals(dfa, state, is_last, valuation, target) -> dict[str, bool]:
    """Keep only values that establish this transition under its recorded guard."""

    selected = dict(valuation)
    for atom_id in valuation:
        without = {key: value for key, value in selected.items() if key != atom_id}
        if dfa.partial_step(state, without, is_last=is_last) == target:
            selected = without
    return selected


def _causal_atom_evaluations(
    artifact,
    trace_builder: _TraceBuilder,
    events,
    positions: set[int],
    *,
    full_domain: bool = False,
) -> _CausalEvaluations:
    """Mark clear transition evidence; leave ambiguous observations unflagged."""

    if trace_builder.selected_attempt_id is None:
        return _CausalEvaluations()
    causal = _CausalEvaluations()
    selected = next(
        attempt
        for attempt in trace_builder.attempts
        if attempt.attempt_id == trace_builder.selected_attempt_id
    )
    by_position = {event.position: event for event in events}
    transitions = trace_builder.transitions.get(trace_builder.selected_attempt_id, ())
    closed_runs = trace_builder.closed_runs.get(trace_builder.selected_attempt_id, ())
    tool_proofs = trace_builder.tool_proofs.get(trace_builder.selected_attempt_id, {})
    event_atoms = {
        atom_id
        for atom_id in artifact.automaton_atoms
        if (spec := artifact.atom_specs.get(atom_id)) is not None
        and spec.domain is AtomDomain.EVENT
        and spec.predicate != "on_surface"
    }
    # A local transition cannot distinguish the winning branch of a compound absence
    # or universal claim. Keep its complete domain as explanation positions, but make
    # no per-atom causal claim about the branch that eventually won.
    if full_domain and len(event_atoms) != 1:
        return causal
    single_event_atom = next(iter(event_atoms)) if full_domain else None
    domain_values = (
        {
            valuation[single_event_atom]
            for _, position, _, valuation, _ in _attempt_transitions(trace_builder)
            if single_event_atom in valuation and position in positions
        }
        if single_event_atom is not None
        else set()
    )
    # The closed parts the evidence relies on: cited on the timeline or in the proof of a
    # selected tool check, whose body may be universal and so is not copied to the timeline.
    relied_on: set[str] = set()

    def mark(dfa, path, decides, progress: bool) -> None:
        """Mark the literals of ``path`` that establish its outcome, which ``decides``
        recognizes in a target state; with ``progress``, also the true literals of every
        step that moves the automaton towards it."""
        for state, position, is_last, valuation, target in path:
            if position >= 0 and position not in positions:
                continue
            if not full_domain and (target == state or not (progress or decides(target))):
                continue
            if full_domain and len(domain_values) > 1 and not decides(target):
                # A later counterexample decides this claim; earlier checks are observations.
                continue
            for atom_id, value in _transition_literals(
                dfa, state, is_last, valuation, target
            ).items():
                if not full_domain and not value and not decides(target):
                    # A failed alternative can move the DFA into a waiting state, yet
                    # contribute nothing to the later match.
                    continue
                spec = artifact.atom_specs.get(atom_id)
                if spec is not None and spec.predicate == "on_surface":
                    continue
                if (
                    spec is not None
                    and spec.domain is AtomDomain.EVENT
                    and (
                        position not in by_position
                        or spec.selector not in by_position[position].surface_tags
                    )
                ):
                    continue
                causal.timeline.add((atom_id, position))
                relied_on.add(atom_id)
                if atom_id in artifact.tool_quantifiers:
                    mark_tool_checks(atom_id, value)

    def mark_tool_checks(atom_id: str, value: bool) -> None:
        causal.selected_quantifiers.add(atom_id)
        for check in selected.tool_attempts:
            if not check.attempt_id.startswith(f"quantifier:{atom_id}/tool:"):
                continue
            if not check.selected:
                continue
            proof = tool_proofs.get(check.attempt_id, frozenset())
            for evaluation in check.evaluations:
                body_occurrence = evaluation.body_occurrence
                if body_occurrence is None or body_occurrence not in proof:
                    continue
                body_atom_id = evaluation.atom_id
                body_position = evaluation.position
                causal.tool_checks.add((check.attempt_id, body_occurrence))
                relied_on.add(body_atom_id)
                if value and artifact.tool_quantifiers[atom_id].kind == "any":
                    causal.timeline.add((body_atom_id, body_position))
                    causal.tool_timeline.add(
                        (
                            body_atom_id,
                            evaluation.source_node_index,
                            body_occurrence,
                        )
                    )

    dfa = artifact.dfa
    mark(dfa, transitions, dfa.is_rejecting, True)
    # A closed part's run explains its atom wherever the evidence above relies on it: a
    # match by the steps towards its automaton's rejection, a failure by the steps that
    # settle its acceptance. A part read inside another is reached once that one is marked.
    pending = list(closed_runs)
    marked = True
    while marked:
        marked = False
        for run in list(pending):
            atom_id, value, path = run
            if atom_id not in relied_on:
                continue
            pending.remove(run)
            marked = True
            dfa = artifact.closed_conditions[atom_id].dfa
            mark(dfa, path, dfa.is_rejecting if value else dfa.is_accepting, value)
    return causal


def _model_results_for_witness(oracle, artifact, events, trace, captured_value) -> tuple:
    """Keep only model evidence proven causal by the final selected trace."""

    attempt = trace.selected_attempt
    if attempt is None:
        return ()
    by_position = {event.position: event for event in events}
    tools = list(oracle._static.tools.values()) if oracle._static else []
    seen: set[int] = set()
    results = []

    def add_evidence(evaluation: AtomEvaluation, tool_index: int | None) -> None:
        if not evaluation.witness:
            return
        spec = artifact.atom_specs.get(evaluation.atom_id)
        if spec is None:
            return
        where = BOOT if evaluation.position == -1 else by_position.get(evaluation.position)
        if where is None:
            return
        binding = captured_value if spec.parameter_names else None
        if _is_per_tool_static(spec) and tool_index is not None and tool_index < len(tools):
            evidence = oracle.evidence_for_tool(spec, tools[tool_index], binding)
        else:
            evidence = oracle.evidence_for(spec, where, binding)
        if isinstance(evidence, _witness.ModelResultEvidence) and id(evidence) not in seen:
            seen.add(id(evidence))
            results.append(evidence)

    # The same winning tool observations are copied into the enclosing timeline attempt.
    # Read the nested check first so the lookup always uses its own coherent tool binding.
    nested_witnesses = set()
    for check in attempt.tool_attempts:
        if not check.selected:
            continue
        for evaluation in check.evaluations:
            if evaluation.witness:
                nested_witnesses.add((evaluation.atom_id, evaluation.position))
                add_evidence(evaluation, check.tool_binding_index)

    for evaluation in attempt.evaluations:
        if (evaluation.atom_id, evaluation.position) not in nested_witnesses:
            add_evidence(evaluation, attempt.tool_binding_index)
    return tuple(results)


async def evaluate_async(
    artifact: CompiledRuleArtifact,
    evaluation_input,
    *,
    async_detector=None,
    async_collector=None,
    warnings: list[str] | None = None,
    regex_backend=None,
    minimize_witness: bool = True,
    record_trace: bool = True,
    time_left: Callable[[], float | None] | None = None,
    defer_minimization: bool = False,
    record: ScanRecord | None = None,
) -> ExecutionResult:
    """Evaluate a compiled rule over ``evaluation_input`` (from
    ``events.build_evaluation_input``): the scanner's execution path. The result's ``matched``
    is True iff the rule fired (the safety DFA rejected).

    External (MODEL/LLM/YARA) atoms are grounded by awaiting
    ``async_detector(spec, text) -> Awaitable[bool]``, so live detectors that need the event
    loop can be driven; omitted, external atoms raise (never silently False). Because external
    atoms carry the highest cost rank, cheap-first ordering grounds them last and the
    ``partial_step`` short-circuit still skips them whenever a cheaper atom already decides
    the step — the cheap-first gating benefit is preserved end-to-end.

    ``warnings`` (if provided) collects execution-time notices
    such as capture cap-exhaustion. ``regex_backend`` (a ``matchers.RegexBackend``) grounds
    local regex/contains atoms; omitted → the pure stdlib backend.

    ``minimize_witness`` (default True) selects explanation evidence. Replay can reduce
    existential paths; universal and absence claims retain the original domain. Set it
    False when the caller does not need explanation evidence. The verdict is unaffected.

    ``record_trace`` (default True) returns every atom evaluation in ``witness.attempts``.
    Explanation evidence records them internally either way; with both off, the witness
    carries positions only.

    ``time_left() -> seconds | None`` is the caller's remaining deadline. Minimization stops
    before a replay that could outlast it and keeps the witness reduced so far, so evidence
    polish can never turn a decided match into a timeout.

    ``record`` (a ``ScanRecord`` the caller owns) receives every value the verdict run
    computes, in order, each decision-tree check with its model evidence; it keeps the values
    computed so far when a check raises. With a record, ``result.explore(full, time_left)``
    is the second pass, which fills in the values the verdict did not need (see
    :func:`_explore_async`); the verdict and the path are unchanged.

    ``defer_minimization`` returns the decided verdict with its raw witness at once and puts
    the reduction in ``result.refine(time_left)``, so a caller can run the verdict inside its
    deadline and the evidence polish after it.
    """
    oracle = Oracle(
        static_env=evaluation_input.static,
        async_detector=async_detector,
        regex_backend=regex_backend,
    )
    events = evaluation_input.events
    trace_builder = _TraceBuilder()
    recorder = trace_builder if record_trace or minimize_witness else None

    started = time.monotonic()
    candidates = _CandidateLog(_candidate_stream(artifact, events, oracle))
    if artifact.capture_plan is not None:
        matched, raw_witness, first_value = await _run_with_capture_async(
            artifact,
            events,
            oracle,
            candidates,
            trace_builder=recorder,
            warnings=warnings,
            record=record,
        )
    else:
        matched, raw_witness = await _run_once_async(
            artifact,
            events,
            oracle,
            trace_builder=recorder,
            record=record,
        )
        first_value = None
    verdict_cost = time.monotonic() - started

    output = None
    if matched and artifact.collect_plan is not None:
        if async_collector is None:
            raise UnsupportedExecution("async evaluation of a collect plan requires a collector")
        output = await async_collector(artifact.collect_plan)
        if output is None:
            matched = False
            raw_witness = []
            first_value = None
            if record is not None:
                record.empty_collection = True

    async def evidence(
        minimize: bool, deadline: Callable[[], float | None] | None, refine=None
    ) -> ExecutionResult:
        """The result with its explanation evidence; ``minimize`` reduces the witness by
        replays that stop at ``deadline``. ``refine`` is the deferred minimization, if any."""
        witness = raw_witness
        captured_value = first_value
        trial_proof: tuple[tuple[int, ...], _TraceBuilder, str | None] | None = None

        async def run(evs, should_stop=None) -> bool:
            nonlocal trial_proof
            candidate_builder = _TraceBuilder()
            if artifact.capture_plan is not None:
                reproduced, _, value = await _run_with_capture_async(
                    artifact,
                    evs,
                    oracle,
                    _replay_candidates(artifact, evs, oracle, candidates),
                    trace_builder=candidate_builder,
                    should_stop=should_stop,
                )
            else:
                reproduced, _ = await _run_once_async(
                    artifact, evs, oracle, trace_builder=candidate_builder, should_stop=should_stop
                )
                value = None
            if reproduced:
                trial_proof = (tuple(event.position for event in evs), candidate_builder, value)
            return reproduced

        use_original_trace = bool(
            matched and minimize and (artifact.full_trace_proof or not artifact.dfa.atoms)
        )
        witness_minimized = bool(matched and minimize and not use_original_trace)
        proof_builder = None
        if use_original_trace:
            witness = _domain_positions(artifact, trace_builder, events)
        elif matched and witness and minimize:
            # The verdict above is sound and final; minimization is best-effort
            # evidence polish and must never discard it. A replay can ground (atom, event)
            # pairs the verdict run never grounded, so the replay budget does not bound an
            # awaited model call: minimization also runs under its own timeout that expires
            # before ``deadline``. Any failure degrades to the raw (sound, non-minimal)
            # witness. We catch broadly rather than name the safety layer's timeout
            # exception so the engine stays isolated (test_engine_isolation).
            replay_budget = _ReplayBudget(deadline, verdict_cost)

            def out_of_time() -> bool:
                remaining = deadline() if deadline is not None else None
                return remaining is not None and remaining <= 0

            try:
                async with asyncio.timeout(_minimization_timeout(deadline)):
                    seed = (
                        await _state_change_seed_async(
                            artifact, events, oracle, trace_builder, out_of_time
                        )
                        if replay_budget.allows_replay()
                        else None
                    )
                    witness = await _minimize_witness_async(
                        events,
                        witness,
                        run,
                        seed=seed,
                        budget=replay_budget,
                        out_of_time=out_of_time,
                    )
                    if trial_proof is not None and trial_proof[0] == tuple(witness):
                        proof_builder, captured_value = trial_proof[1:]
                    elif tuple(event.position for event in events) == tuple(witness):
                        proof_builder = trace_builder
                    else:
                        subset = tuple(e for e in events if e.position in set(witness))
                        if not await run(subset, out_of_time):
                            raise RuntimeError("minimized witness did not reproduce the rule match")
                        assert trial_proof is not None
                        proof_builder, captured_value = trial_proof[1:]
            except Exception:
                # Evidence polish only: the witness stays sound, just not minimal, and the
                # scan's completeness is decided by the verdict alone, so nothing is
                # recorded as an evaluation warning here.
                witness = _raw_witness(events, raw_witness)
                captured_value = first_value
                witness_minimized = False
                proof_builder = None
        elif matched and witness:
            witness = _raw_witness(events, witness)
        else:
            witness = []

        final_trace = (
            WitnessTrace(positions=tuple(witness))
            if recorder is None
            else trace_builder.build(
                witness,
                matched=matched,
                minimized=witness_minimized,
                causal_evaluations=(
                    _causal_atom_evaluations(
                        artifact,
                        proof_builder or trace_builder,
                        events,
                        set(witness),
                        full_domain=artifact.full_trace_proof,
                    )
                    if use_original_trace or witness_minimized
                    else None
                ),
                proof_builder=proof_builder,
            )
        )
        return ExecutionResult(
            rule_name=artifact.rule_name,
            matched=matched,
            satisfied=not matched,
            witness=final_trace,
            external_calls=oracle.external_calls,
            captured_value=captured_value,
            model_results=_model_results_for_witness(
                oracle, artifact, events, final_trace, captured_value
            ),
            output=output,
            refine=refine,
            explore=explore,
        )

    explore = None
    if record is not None:
        scan_record = record

        async def explore(full: bool, deadline: Callable[[], float | None] | None) -> None:
            """Explore the artifact."""
            await _explore_async(
                artifact, evaluation_input, oracle, scan_record, full=full, time_left=deadline
            )

    if not (defer_minimization and minimize_witness and matched and raw_witness):
        return await evidence(minimize_witness, time_left)

    async def refine(deadline: Callable[[], float | None] | None) -> ExecutionResult:
        return await evidence(True, deadline)

    return await evidence(False, None, refine)


# The verdict short-circuits, so its record holds only the values it needed. The second
# pass fills in the rest for the outputs that read every value: the checks a decomposed
# rule's path skipped (``off_path``) and, for the full explanation, every leaf atom on
# every event of its surface. It runs after every verdict of the request and reuses the
# verdict's oracle, so a value the verdict computed is never computed again. A value it
# cannot compute is recorded as unknown with the reason, never as false.

_MODEL_COSTS = (Cost.MODEL, Cost.LLM)


class _Target(NamedTuple):
    atom_id: str
    where: object
    position: int | None
    check: bool = False
    tool: bool = False


def _explore_targets(artifact, evaluation_input, known, held, *, full: bool) -> list[_Target]:
    """What the second pass grounds, in order: the tree's unrecorded checks, then with
    ``full`` each leaf atom's unrecorded candidates, in authored order. ``held`` is the
    recorded values that were true."""

    targets: list[_Target] = []
    tree = artifact.decision_tree
    if tree is not None and tree.decomposed:
        targets += [
            _Target(node.atom_id, BOOT, None, check=True)
            for node in tree.nodes
            if (node.atom_id, None) not in known
        ]
    if not full:
        return targets
    parts = artifact_parts(artifact)
    leaves = _witness.explained_atoms(artifact, parts)
    tool_body_atoms = {atom_id for body in parts.tool_bodies.values() for atom_id in body}
    tools = list((evaluation_input.static.tools or {}).values())
    for atom_id in leaves:
        spec = artifact.atom_specs[atom_id]
        if spec.domain is AtomDomain.EVENT:
            targets += [
                _Target(atom_id, event, event.position)
                for event in evaluation_input.in_domain(spec)
                if (atom_id, event.position) not in known
            ]
        elif atom_id in tool_body_atoms or _is_per_tool_static(spec):
            # A recorded tool value is one tool's, and the verdict stops at the tool that
            # decides it, so only a tool that held settles the atom.
            if (atom_id, None) not in held:
                targets += [_Target(atom_id, tool, None, tool=True) for tool in tools]
        elif (atom_id, None) in known:
            continue
        else:
            targets.append(_Target(atom_id, BOOT, None))
    return targets


def _failure_reason(error: BaseException, out_of_time: Callable[[], bool]) -> str:
    details = getattr(error, "details", None) or {}
    if out_of_time() or details.get("code") == "model_call_limit":
        return "budget_exhausted"
    return "model_failed"


async def _explore_async(
    artifact, evaluation_input, oracle, record: ScanRecord, *, full: bool, time_left
) -> None:
    """Fill ``record`` with the values the verdict did not compute, marked explored.

    ``time_left() -> seconds | None`` is the pass's own budget. When it runs out, every
    value not yet computed is recorded unknown with ``budget_exhausted``. A value that
    fails to compute is unknown with ``model_failed``, and one that tests a captured
    value, which has no value outside a binding, with ``captured_value``. A per-tool atom
    is grounded tool by tool until one holds.
    """

    def remaining() -> float | None:
        return time_left() if time_left is not None else None

    def out_of_time() -> bool:
        seconds = remaining()
        return seconds is not None and seconds <= 0

    def unknown(target: _Target, reason: str) -> None:
        record.add(target.atom_id, target.position, None, "explored", reason, target.check)

    if full:
        _attach_model_evidence(artifact, evaluation_input, oracle, record)
    known = {(entry[0], entry[1]) for entry in record.entries if entry[2] is not None}
    held = {(entry[0], entry[1]) for entry in record.entries if entry[2]}
    targets = _explore_targets(artifact, evaluation_input, known, held, full=full)
    run = _Run(
        artifact,
        evaluation_input.events,
        None,
        should_stop=out_of_time,
        record=record,
        origin="explored",
    )
    held: set[str] = set()
    for index, target in enumerate(targets):
        if target.tool and target.atom_id in held:
            continue
        spec = artifact.atom_specs.get(target.atom_id)
        if out_of_time():
            for rest in targets[index:]:
                unknown(rest, "budget_exhausted")
            return
        if spec is not None and spec.parameter_names:
            unknown(target, "captured_value")
            continue
        try:
            async with asyncio.timeout(remaining()):
                if target.tool:
                    value = (await oracle.ground_tool_async(spec, target.where, None)).value
                else:
                    value = await _ground_atom_async(run, target.atom_id, target.where, oracle)
        except (TimeoutError, _ReplayStopped):
            for rest in targets[index:]:
                unknown(rest, "budget_exhausted")
            return
        except Exception as error:
            unknown(target, _failure_reason(error, out_of_time))
            continue
        if target.check:
            _record_check(run, target.atom_id, target.where, value, oracle)
        else:
            record.add(
                target.atom_id,
                target.position,
                value,
                "explored",
                evidence=_model_evidence(oracle, spec, target.where, tool=target.tool),
            )
        if target.tool and value:
            held.add(target.atom_id)


def _model_evidence(oracle, spec, where, *, tool: bool = False):
    """A model atom's result at ``where``, from the oracle's cache, else None."""
    if spec is None or spec.cost not in _MODEL_COSTS:
        return None
    evidence = (
        oracle.evidence_for_tool(spec, where, None) if tool else oracle.evidence_for(spec, where)
    )
    return evidence if isinstance(evidence, _witness.ModelResultEvidence) else None


def _attach_model_evidence(artifact, evaluation_input, oracle, record: ScanRecord) -> None:
    """Give each model value the verdict recorded without its result the result, so the full
    explanation can report the score. The value itself is unchanged."""
    by_position = {event.position: event for event in evaluation_input.events}
    for index, entry in enumerate(record.entries):
        atom_id, position, value, _origin, _reason, check, evidence = entry
        if check or value is None or evidence is not None:
            continue
        where = BOOT if position is None else by_position.get(position)
        found = _model_evidence(oracle, artifact.atom_specs.get(atom_id), where)
        if found is not None:
            record.entries[index] = (*entry[:6], found)


def _raw_witness(events, raw_positions) -> list[int]:
    """The un-minimized witness: the raw path filtered to real event positions.

    Sound but not minimal — it's the path the DFA actually walked to reach the match, so
    every position is a real event and the set still justifies the verdict. Used when
    minimization is skipped (``minimize_witness=False``) or bails out (e.g. timeout)."""
    valid = {ev.position for ev in events}
    return [p for p in raw_positions if p in valid]


def _minimization_timeout(time_left: Callable[[], float | None] | None) -> float | None:
    """Seconds minimization may run: the caller's remaining time minus a reserve, so the
    decided verdict is returned before the caller's own deadline cancels the rule."""
    remaining = time_left() if time_left is not None else None
    if remaining is None:
        return None
    return max(0.0, remaining - max(0.05, remaining * 0.1))


class _ReplayStopped(Exception):
    """A minimization replay stopped before the deadline; keep the witness reduced so far."""


class _ReplayBudget:
    """Decides whether one more minimization replay fits before the caller's deadline.

    A replay runs the rule over a subset of the events the verdict run covered, so the
    verdict run's duration bounds the first replay; later replays update the estimate."""

    def __init__(self, time_left: Callable[[], float | None] | None, first_cost: float):
        self._time_left = time_left
        self._max_cost = first_cost
        self.stopped = False

    def allows_replay(self) -> bool:
        """Return whether a replay is allowed."""
        remaining = self._time_left() if self._time_left is not None else None
        if remaining is not None and remaining <= 2 * self._max_cost:
            self.stopped = True
        return not self.stopped

    def record(self, cost: float) -> None:
        """Record the cost of a replay."""
        self._max_cost = max(self._max_cost, cost)


async def _minimize_witness_async(
    events, raw_positions, run, seed=None, budget=None, out_of_time=None
) -> list[int]:
    """Reduce the matched run's raw witness to the load-bearing timeline positions.

    Dropping a filler event makes the remaining steps adjacent, which is exactly the
    semantics ``within:N`` flow distance needs, so the surviving positions are a sound,
    minimal justification of the match. ``seed`` is a smaller derived starting set (see
    :func:`_state_change_seed_async`); it is verified before use, so soundness never rests on
    it being optimal.

    ``run(events_subset, should_stop)`` replays the rule; a replay stops once
    ``out_of_time`` says so. With a ``budget``, stops before a replay that would not fit and
    returns the current candidate, which always reproduces the match."""
    by_pos = {ev.position: ev for ev in events}
    candidate = [p for p in raw_positions if p in by_pos]
    budget = budget or _ReplayBudget(None, 0.0)

    async def reproduces(subset_positions: list[int]) -> bool:
        keep = set(subset_positions)
        subset = tuple(ev for ev in events if ev.position in keep)
        started = time.monotonic()
        matched = await run(subset, out_of_time)
        budget.record(time.monotonic() - started)
        await asyncio.sleep(0)
        return matched

    # Seed greedy reduction with the derived state-change positions when they reproduce
    # (see the docstring). Verified before use; falls back to the full prefix.
    if seed is not None:
        seed = [p for p in seed if p in by_pos]
        try:
            if (
                seed
                and len(seed) < len(candidate)
                and budget.allows_replay()
                and await reproduces(seed)
            ):
                candidate = seed
        except _ReplayStopped:
            return candidate

    # ``minimize`` needs a sync predicate; drive the async re-run through a tiny loop-free
    # shim that mirrors witness.minimize's passes.
    cur = list(candidate)
    changed = True
    while changed and len(cur) > 1:
        changed = False
        for pos in list(cur):
            if len(cur) == 1:
                break
            if not budget.allows_replay():
                return cur
            trial = [p for p in cur if p != pos]
            try:
                reproduced = await reproduces(trial)
            except _ReplayStopped:
                return cur
            if reproduced:
                cur = trial
                changed = True
    return cur


def _uses_tools_static(artifact) -> bool:
    return any(
        _is_per_tool_static(artifact.atom_specs[atom_id])
        for atom_id in artifact.automaton_atoms
        if atom_id in artifact.atom_specs
    )


def _tool_bindings(artifact, oracle):
    """Return coherent tool choices for one DFA run; an empty existential has none."""

    if not _uses_tools_static(artifact):
        return [(None, None)]
    tools = list(oracle._static.tools.values()) if oracle._static else []
    return list(enumerate(tools))


# Yielded by _candidate_stream for an element outside its collection, and as a pause inside
# a long enumeration, so a runner can yield.
_SKIPPED = object()


# Pruned candidates between two yields of the async capture runner.
_PRUNED_PER_PAUSE = 64


@dataclass(frozen=True)
class _Pruned:
    """A candidate whose run cannot match, or can only as the run with no candidate bound
    does: a value no reference site holds for, or an element whose event fails the plan's
    own-step test. It keeps its candidate index and is shown as an unmatched attempt carrying
    the evaluations that ruled it out, but it runs no DFA and does not count toward the run
    cap."""

    evaluations: tuple[AtomEvaluation, ...]


class _CandidateLog:
    """The verdict run's candidate stream, kept as it is read, so a minimization replay reads
    the same candidates at the same indices without enumerating or prefiltering them again.
    A replay that reads past the verdict run continues the one stream."""

    def __init__(self, stream: Iterator[object]):
        self._stream = stream
        self._items: list[object] = []

    def __iter__(self) -> Iterator[object]:
        index = 0
        while True:
            if index < len(self._items):
                yield self._items[index]
                index += 1
                continue
            item = next(self._stream, self)
            if item is self:
                return
            if item is not _SKIPPED:
                # A pause or an element outside its collection takes no index.
                self._items.append(item)
                index += 1
            yield item


def _candidate_stream(artifact, events, oracle) -> Iterator[object]:
    """The candidates to instantiate, lazily and in timeline order: each value or element
    binding to run, a :class:`_Pruned` for a value no reference site holds for or an element
    whose event fails the plan's own-step test, ``_SKIPPED`` for an element outside its
    collection and as a pause, and ``captures.MORE`` when the
    enumeration bound cut values off. Every verdict run prunes the same values, so an
    explanation or trace never changes which values reach the run cap.

    Value enumeration runs through the oracle's regex backend (RE2 on the scanner
    path), so a catastrophic capture regex is answered in linear time."""
    plan = artifact.capture_plan
    if isinstance(plan, _cap.ElementPlan):
        # A pinned element's collection inside its window; the window resolves against the
        # surface size as the oracle's position gate does, and the accessor filter decides
        # which events are the collection's own.
        size = oracle._static.surface_sizes.get(plan.surface, 0) if oracle._static else 0
        lo, hi, _ = slice(plan.window_start, plan.window_stop).indices(size)
        accessor = artifact.atom_specs.get(plan.filter_atom_id)
        for ev in events:
            ordinal = ev.ordinals.get(plan.surface)
            if ordinal is None or not lo <= ordinal < hi:
                continue
            if accessor is not None and not _may_hold(oracle, accessor, ev):
                yield _SKIPPED
                continue
            grounded: dict[str, bool | None] = {}
            if _step_value(oracle, artifact.atom_specs, plan.own_step, ev, grounded) is False:
                yield _Pruned(
                    tuple(
                        AtomEvaluation(
                            atom_id, ev.position, "evaluated_true" if value else "evaluated_false"
                        )
                        for atom_id, value in grounded.items()
                        if value is not None
                    )
                )
                continue
            yield _cap.ElementBinding(ev.position, ev.source_path, ev.source)
        return
    prefilter = _reference_prefilter(artifact, events, oracle)
    values = _cap.iter_capture_values(
        plan,
        events,
        regex_backend=oracle._regex,
        pause=_SKIPPED,
        surface_sizes=oracle._static.surface_sizes if oracle._static else None,
    )
    for value in values:
        if isinstance(value, str) and prefilter is not None:
            keep, pruned = prefilter
            verdict = None
            for verdict in keep(value):
                if verdict is None:
                    yield _SKIPPED
            if not verdict:
                yield pruned
                continue
        yield value


def _may_hold(oracle, spec, event, binding=None, *, retain: bool = True) -> bool:
    """Whether ``spec`` holds at ``event``. A grounding error counts as "may hold", so the DFA
    run decides whether that atom is needed; a deadline expiry (not an EvaluationError) and an
    evaluation timeout propagate. ``retain=False`` leaves the oracle's cache untouched."""
    try:
        if retain:
            return oracle.value(spec, event, binding)
        return oracle._ground_uncached(spec, event, binding).value
    except EvaluationTimeoutError:
        raise
    except EvaluationError:
        return True


def _step_value(oracle, specs, formula, event, grounded: dict[str, bool | None]) -> bool | None:
    """The value of the Boolean ``formula`` at ``event``, grounding each atom once into
    ``grounded``, or ``None`` when it depends on an atom whose grounding failed (the run
    decides whether that atom is needed). A deadline expiry and a timeout propagate."""
    match formula:
        case T.TrueConst():
            return True
        case T.FalseConst():
            return False
        case T.Atom(name):
            if name not in grounded:
                try:
                    grounded[name] = oracle.value(specs[name], event)
                except EvaluationTimeoutError:
                    raise
                except EvaluationError:
                    grounded[name] = None
            return grounded[name]
        case T.Not(child):
            value = _step_value(oracle, specs, child, event, grounded)
            return None if value is None else not value
        case T.And(left, right) | T.Or(left, right):
            decisive = isinstance(formula, T.Or)
            first = _step_value(oracle, specs, left, event, grounded)
            if first is decisive:
                return decisive
            second = _step_value(oracle, specs, right, event, grounded)
            if second is decisive:
                return decisive
            return None if None in (first, second) else not decisive
        case T.Implies(left, right):
            return _step_value(oracle, specs, T.Or(T.Not(left), right), event, grounded)
    raise UnsupportedExecution("an element's own-step test is not a Boolean formula")


def _evidence_value(value) -> str | None:
    """The captured value reported as evidence: a pinned element is named by its path."""
    return value.label if isinstance(value, _cap.ElementBinding) else value


def _reference_prefilter(
    artifact, events, oracle
) -> tuple[Callable[[str], Iterator[bool | None]], _Pruned] | None:
    """A test that keeps only values some reference atom can hold for, with the
    :class:`_Pruned` marker for a value it rules out, or ``None``.

    Used only when the DFA cannot reject while every reference atom is False, so a value no
    reference site holds for cannot match. The test grounds the reference atoms themselves
    (every modifier applies) without retaining the probes. A grounding error keeps the
    value: the run decides whether that atom is needed."""
    plan = artifact.capture_plan
    refs = [
        artifact.atom_specs[atom_id]
        for atom_id in sorted(plan.ref_atom_ids or {plan.ref_atom_id})
        if atom_id in artifact.atom_specs
    ]
    if not refs or _can_reject_with(artifact, {spec.atom_id: False for spec in refs}):
        return None
    # A static reference (a test on the value itself) is read once, at the boot step.
    sites: list[tuple[AtomSpec, object]] = [
        (spec, BOOT) for spec in refs if spec.domain is AtomDomain.STATIC
    ]
    sites += [
        (spec, ev)
        for spec in refs
        if spec.domain is not AtomDomain.STATIC
        for ev in events
        if spec.selector in ev.surface_tags
    ]

    def keep(value: str) -> Iterator[bool | None]:
        """``True`` once some site may hold for ``value``, else ``False``; ``None`` between
        sites whenever a pause is due, so the caller can yield."""
        pacer = Pacer(every=64)
        for spec, ev in sites:
            if pacer.due():
                yield None
            if _may_hold(oracle, spec, ev, value, retain=False):
                yield True
                return
        yield False

    # A ruled-out value was tested False at every reference site.
    pruned = _Pruned(
        tuple(
            AtomEvaluation(spec.atom_id, _hit_position(where), "evaluated_false")
            for spec, where in sites
        )
    )
    return keep, pruned


def _cap_warning(plan) -> str:
    if isinstance(plan, _cap.ElementPlan):
        limit = f"{_cap.MAX_CAPTURE_VALUES} {plan.surface} items"
    else:
        limit = f"{_cap.MAX_CAPTURE_VALUES} runs of {_cap.MAX_CAPTURE_CANDIDATES} values"
    return f"'{plan.var}' hit the candidate cap ({limit}) without a match; result may be incomplete"


def _replay_candidates(artifact, events, oracle, candidates: _CandidateLog) -> Iterator[object]:
    """The verdict run's candidates, in their original order, for a replay over ``events``:
    each one the replayed events still produce, and ``None`` in the place of one they do not,
    so every candidate keeps its index and a replay may prove a later one.

    Collecting the replayed events' values yields ``_SKIPPED`` whenever a pause is due, so
    a field of many repeated matches cannot hold the replay past its deadline.
    """
    plan = artifact.capture_plan
    if isinstance(plan, _cap.ElementPlan):
        positions = {ev.position for ev in events}

        def present(item) -> bool:
            return item.position in positions

        kind: type = _cap.ElementBinding
    else:
        values: set[str] = set()
        for value in _cap.iter_capture_values(
            plan,
            events,
            regex_backend=oracle._regex,
            pause=_SKIPPED,
            surface_sizes=oracle._static.surface_sizes if oracle._static else None,
        ):
            if value is _SKIPPED:
                yield _SKIPPED
            elif isinstance(value, str):
                values.add(value)
        present = values.__contains__
        kind = str
    for item in candidates:
        yield (item if present(item) else None) if isinstance(item, kind) else item


async def _state_change_seed_async(
    artifact, events, oracle, trace_builder: _TraceBuilder | None = None, should_stop=None
):
    """Derived minimization seed for the plain temporal path: the timeline positions where
    the DFA state changed on the matched walk, or on the walk of a closed part it read.

    Dropping the self-loop steps between two state changes cannot alter the verdict (the DFA
    arrives at each change in the same state either way), so the state-change set is itself a
    reproducing witness, and seeding minimization with it avoids the O(N²) replay cost on long
    conversations. Returns ``None`` for capture and per-tool rules, where one trajectory is
    ill-defined."""
    if artifact.capture_plan is not None or _uses_tools_static(artifact):
        return None
    if trace_builder is not None and trace_builder.selected_attempt_id is not None:
        return _state_changes(_attempt_transitions(trace_builder))
    return await _trace_state_changes_async(artifact, events, oracle, should_stop)


def _state_changes(transitions) -> list[int]:
    """The event positions, in timeline order, where a transition changed the state."""
    return sorted(
        {
            position
            for state, position, _, _, target in transitions
            if position != -1 and state != target
        }
    )


async def _trace_state_changes_async(artifact, events, oracle, should_stop=None) -> list[int]:
    """Replay the matched walk (plain path, no binding) and return the positions where
    ``dfa.step`` changed the state, in it or in a closed part it read. One cache-warm pass."""
    run = _Run(artifact, events, None, closed_runs=[], should_stop=should_stop)
    transitions: list[_Transition] = []
    await _walk_async(run, artifact.dfa, oracle, transitions)
    return _state_changes(
        [*transitions, *(step for _, _, path in run.closed_runs or () for step in path)]
    )


def _trace_steps(artifact, events):
    """The ordered (where, is_last) steps of a run: optional boot, then each event. Shared
    by every run helper and the state-change seed so their stepping is identical. With no
    events the boot step is the last one, so the DFA's end-of-trace transition still runs."""
    steps: list[tuple[object, bool]] = []
    if _has_static(artifact):
        steps.append((BOOT, not events))
    for i, ev in enumerate(events):
        steps.append((ev, i == len(events) - 1))
    return steps


def _has_static(artifact) -> bool:
    return any(s.domain is AtomDomain.STATIC for s in artifact.atom_specs.values())


def _empty_trace_matched(dfa) -> bool:
    """Verdict for the EMPTY word (a conversation with no timeline events).

    The DFA encodes the *safety* formula (``matched = safety violated``). The empty-word
    answer is the DFA's own: take one FINAL step from the initial state with an empty
    valuation (no atom is true — there is no event to read) and read the terminal state —
    ``matched`` iff it rejects. This yields the correct verdict per formula shape without
    special-casing: an ``any``/Eventually condition is unsatisfiable on the empty word so
    its safety formula holds → not matched; an ``all``/Globally condition is vacuously true
    so its safety formula is violated → matched.
    A non-terminal result (neither accept nor reject) means the safety formula still holds
    over the empty word → not matched.

    NOTE — layering: this is the *rule-matching* verdict, which follows the DSL's semantics
    (``all`` over an empty collection is vacuously true → the rule matches). It
    deliberately differs from ``symbolic_dfa.run_trace([])``, which returns the raw-LTLf
    "empty word vacuously satisfies any formula" convention (True). The two layers ask different questions: ``run_trace`` = "does this
    formula hold on the empty word?"; this = "does the RULE match an empty conversation?".
    They only ever differ on the fully-empty trace, and only for ``all``/Globally-shaped
    rules."""
    final_state = dfa.step(dfa.initial_state, {}, is_last=True)
    return dfa.is_rejecting(final_state)


async def _ground_value_async(spec, where, oracle, binding) -> bool:
    """Ground one atom at ``where``. A tool result is one ``ToolMessage`` event tagged on both
    the ``messages`` and ``tool_responses`` surfaces, so a ``tool_responses`` atom grounds
    directly at that single position."""
    return await oracle.value_async(spec, where, binding)


def _witness_position(where) -> int:
    """Position to cite in the witness for this step — the step's own position. A tool result
    is a single event, so there is no phantom message/response split to remap around."""
    return where.position


def _hit_position(where) -> int:
    """Return the stable position used by runtime attribution (-1 means boot)."""

    return -1 if where is BOOT else _witness_position(where)


_RunResult = tuple[bool, list[int]]


@dataclass
class _Run:
    """One run of the rule under one binding (a captured value or candidate element, and
    for legacy artifacts one tool), and the atoms decided by an evaluation of their own over
    it: a tools quantifier over the tool definitions, a closed part by its automaton over
    the run's events. Each is decided once per run, never in the oracle's cache, since a
    replay over fewer events decides it again; what those evaluations read is recorded with
    the run's own evaluations."""

    artifact: CompiledRuleArtifact
    events: tuple
    binding: object
    single_tool: object = None
    evaluations: list[AtomEvaluation] | None = None
    tool_attempts: list[_ToolCheck] | None = None
    closed_runs: list[_ClosedRun] | None = None
    should_stop: Callable[[], bool] | None = None
    derived: dict[str, bool] = field(default_factory=dict)
    # The positions the closed parts' runs walked, which join the run's witness.
    positions: set[int] = field(default_factory=set)
    # The verdict run records every value it computes; a replay records none.
    record: ScanRecord | None = None
    origin: ObservationOrigin = "verdict"

    def is_derived(self, atom_id: str) -> bool:
        """Return whether an atom is derived."""
        return (
            atom_id in self.artifact.tool_quantifiers or atom_id in self.artifact.closed_conditions
        )

    def witness(self, walked: list[int]) -> list[int]:
        """Return the witness positions."""
        return sorted(self.positions.union(walked))


def _ground_order(dfa, state, is_last, run: _Run) -> list[str]:
    """The atoms that can change this state's transition, cheapest first; an atom decided by
    an evaluation of its own after the plain atoms of its cost. Missing specs sort as cheap
    (they ground to False)."""
    specs = run.artifact.atom_specs
    return sorted(
        dfa.relevant_atoms(state, is_last=is_last),
        key=lambda a: (specs[a].cost.rank, run.is_derived(a)) if a in specs else (-1, False),
    )


def _record(run: _Run, atom_id: str, where, value: bool, *, check=False, evidence=None) -> None:
    position = _hit_position(where)
    if run.evaluations is not None:
        run.evaluations.append(
            AtomEvaluation(
                atom_id=atom_id,
                position=position,
                status=("evaluated_true" if value else "evaluated_false"),
            )
        )
    if run.record is not None:
        # The hot path: one tuple per value, in ``Observation`` field order.
        run.record.entries.append(
            (
                atom_id,
                None if position == -1 else position,
                value,
                run.origin,
                None,
                check,
                evidence,
            )
        )


def _record_skipped(run: _Run, atoms: list[str], where) -> None:
    if run.evaluations is None:
        return
    plans = run.artifact.tool_quantifiers
    for skipped_atom_id in atoms:
        run.evaluations.append(AtomEvaluation(skipped_atom_id, _hit_position(where), "skipped"))
        if skipped_atom_id in plans:
            run.evaluations.extend(
                _skipped_tool_body_evaluations(plans[skipped_atom_id], run.artifact.atom_specs)
            )


def _walks_tree(dfa, state, where, run: _Run) -> bool:
    """Whether this step is a decomposed rule's boot step, which walks its decision tree."""

    tree = run.artifact.decision_tree
    return (
        tree is not None
        and tree.decomposed
        and dfa is run.artifact.dfa
        and where is BOOT
        and state == dfa.initial_state
    )


def _record_check(run: _Run, atom_id: str, where, value: bool, oracle) -> None:
    spec = run.artifact.atom_specs.get(atom_id)
    if run.record is None or spec is None or run.is_derived(atom_id):
        evidence = None
    elif spec.predicate == "surface_count":
        # A size check's evidence is the size it compared, so its bound can be moved later.
        evidence = oracle.surface_size(spec)
    else:
        evidence = oracle.evidence_for(spec, where, run.binding)
    _record(run, atom_id, where, value, check=True, evidence=evidence)


def _tool_body_layout(body, specs):
    """Index authored leaves and choose cheap-first order in one formula walk."""

    leaves = []
    indices = {}
    orders = {}

    def visit(formula, path):
        match formula:
            case T.Atom(name):
                indices[path] = len(leaves)
                leaves.append(name)
                return specs[name].cost.rank
            case T.Not(child):
                return visit(child, (*path, 0))
            case T.And(left, right) | T.Or(left, right) | T.Implies(left, right):
                left_cost = visit(left, (*path, 0))
                right_cost = visit(right, (*path, 1))
                orders[path] = (0, 1) if left_cost <= right_cost else (1, 0)
                return min(left_cost, right_cost)
            case _:
                return -1

    visit(body, ())
    return leaves, indices, orders


def _skipped_tool_body_evaluations(plan, specs):
    leaves, _, _ = _tool_body_layout(plan.body, specs)
    return [
        AtomEvaluation(
            atom_id=atom_id,
            position=-1,
            status="skipped",
            source_node_index=(
                plan.body_source_nodes[occurrence]
                if occurrence < len(plan.body_source_nodes)
                else None
            ),
            body_occurrence=occurrence,
        )
        for occurrence, atom_id in enumerate(leaves)
    ]


async def _eval_tool_body_async(plan, run: _Run, oracle, tool, layout):
    specs = run.artifact.atom_specs
    body = plan.body
    leaves, indices, orders = layout
    observed: dict[str, bool] = {}
    visited: dict[int, bool] = {}

    async def value(formula, path=()):
        match formula:
            case T.TrueConst():
                return True, set()
            case T.FalseConst():
                return False, set()
            case T.Atom(name):
                if name not in observed:
                    spec = specs[name]
                    if run.is_derived(name):
                        observed[name] = await _ground_derived_async(run, name, oracle)
                    elif _is_per_tool_static(spec):
                        grounded = await oracle.ground_tool_async(spec, tool, run.binding)
                        observed[name] = grounded.value
                    else:
                        observed[name] = await oracle.value_async(spec, BOOT, run.binding)
                    if run.record is not None:
                        # A body value is static, so it is kept without a position.
                        run.record.add(name, None, observed[name], run.origin)
                occurrence = indices[path]
                visited[occurrence] = observed[name]
                return observed[name], {occurrence}
            case T.Not(child):
                result, proof = await value(child, (*path, 0))
                return not result, proof
            case T.And():
                children = (formula.left, formula.right)
                first, second = orders[path]
                first_value, first_proof = await value(children[first], (*path, first))
                if not first_value:
                    return False, first_proof
                second_value, second_proof = await value(children[second], (*path, second))
                if second_value:
                    first_proof.update(second_proof)
                    return True, first_proof
                return False, second_proof
            case T.Or():
                children = (formula.left, formula.right)
                first, second = orders[path]
                first_value, first_proof = await value(children[first], (*path, first))
                if first_value:
                    return True, first_proof
                second_value, second_proof = await value(children[second], (*path, second))
                if second_value:
                    return True, second_proof
                first_proof.update(second_proof)
                return False, first_proof
            case T.Implies(left, right):
                left_value, left_proof = await value(left, (*path, 0))
                if not left_value:
                    return True, left_proof
                right_value, right_proof = await value(right, (*path, 1))
                if right_value:
                    return True, right_proof
                left_proof.update(right_proof)
                return False, left_proof
            case _:
                raise UnsupportedExecution("non-Boolean tool quantifier body")

    matched, proof_occurrences = await value(body)
    evaluations = [
        AtomEvaluation(
            atom_id=atom_id,
            position=-1,
            status=(
                "skipped"
                if occurrence not in visited
                else "evaluated_true"
                if visited[occurrence]
                else "evaluated_false"
            ),
            source_node_index=(
                plan.body_source_nodes[occurrence]
                if occurrence < len(plan.body_source_nodes)
                else None
            ),
            body_occurrence=occurrence,
        )
        for occurrence, atom_id in enumerate(leaves)
    ]
    return matched, evaluations, frozenset(proof_occurrences)


async def _ground_tool_quantifier_async(atom_id, plan, run: _Run, oracle):
    evaluations, tool_attempts = run.evaluations, run.tool_attempts
    first_check = len(tool_attempts) if tool_attempts is not None else 0
    layout = None
    for tool_index, tool in enumerate(oracle._static.tools.values() if oracle._static else ()):
        if layout is None:
            layout = _tool_body_layout(plan.body, run.artifact.atom_specs)
        matched, observed, proof_occurrences = await _eval_tool_body_async(
            plan, run, oracle, tool, layout
        )
        decisive = matched if plan.kind == "any" else not matched
        if tool_attempts is not None:
            tool_attempts.append(
                (
                    ExecutionAttempt(
                        attempt_id=f"quantifier:{atom_id}/tool:{tool_index}",
                        kind="tool",
                        binding_index=tool_index,
                        tool_binding_index=tool_index,
                        matched=matched,
                        selected=decisive,
                        evaluations=tuple(observed),
                    ),
                    proof_occurrences,
                )
            )
        if decisive:
            if evaluations is not None and plan.kind == "any":
                evaluations.extend(observed)
            return matched
    if tool_attempts is not None:
        tool_attempts[first_check:] = [
            (replace(check, selected=True), proof) for check, proof in tool_attempts[first_check:]
        ]
    return plan.kind == "all"


def _stop_if(should_stop) -> None:
    """End a minimization replay once ``should_stop`` says its time is up: the replay is
    synchronous work that a timer cannot interrupt."""
    if should_stop is not None and should_stop():
        raise _ReplayStopped


# Atom grounding awaits ``oracle.value_async`` (external atoms are coroutines).


async def _run_once_async(
    artifact,
    events,
    oracle,
    binding: str | None = None,
    *,
    trace_builder: _TraceBuilder | None = None,
    should_stop=None,
    record: ScanRecord | None = None,
) -> _RunResult:
    uses_tools_static = _uses_tools_static(artifact)
    for tool_index, tool in _tool_bindings(artifact, oracle):
        evaluations: list[AtomEvaluation] | None = [] if trace_builder is not None else None
        tool_attempts: list[_ToolCheck] | None = [] if trace_builder is not None else None
        transitions: list[_Transition] | None = [] if trace_builder is not None else None
        closed_runs: list[_ClosedRun] | None = [] if trace_builder is not None else None
        matched, witness = await _run_trace_async(
            artifact,
            events,
            oracle,
            binding,
            single_tool=tool,
            evaluations=evaluations,
            transitions=transitions,
            tool_attempts=tool_attempts,
            closed_runs=closed_runs,
            should_stop=should_stop,
            record=record,
        )
        attempt_id = None
        if trace_builder is not None:
            attempt_id = trace_builder.add_attempt(
                kind="tool" if uses_tools_static else "default",
                binding_index=tool_index,
                tool_binding_index=tool_index,
                matched=matched,
                evaluations=evaluations or [],
                transitions=transitions,
                tool_attempts=tool_attempts,
                closed_runs=closed_runs,
            )
        if matched:
            if trace_builder is not None and attempt_id is not None:
                trace_builder.select(attempt_id)
            return True, witness
    return False, []


async def _run_trace_async(
    artifact,
    events,
    oracle,
    binding,
    single_tool,
    evaluations=None,
    transitions=None,
    tool_attempts=None,
    closed_runs=None,
    should_stop=None,
    record=None,
) -> tuple[bool, list[int]]:
    run = _Run(
        artifact,
        events,
        binding,
        single_tool,
        evaluations,
        tool_attempts,
        closed_runs,
        should_stop,
        record=record,
    )
    matched, walked = await _walk_async(run, artifact.dfa, oracle, transitions)
    return matched, run.witness(walked)


async def _walk_async(run: _Run, dfa, oracle, transitions=None) -> tuple[bool, list[int]]:
    state = dfa.initial_state
    witness: list[int] = []

    steps = _trace_steps(run.artifact, run.events)

    if not steps:
        return _empty_trace_matched(dfa), []

    pacer = Pacer()
    for where, is_last in steps:
        _stop_if(run.should_stop)
        if pacer.due():
            await asyncio.sleep(0)
        valuation = await _ground_relevant_async(dfa, state, oracle, where, is_last, run)
        next_state = dfa.step(state, valuation, is_last=is_last)
        if transitions is not None:
            transitions.append((state, _hit_position(where), is_last, valuation, next_state))
        state = next_state
        if where is not BOOT:
            witness.append(_witness_position(where))
        if dfa.is_rejecting(state):
            return True, witness
    return (not dfa.is_accepting(state)), witness


async def _ground_atom_async(run: _Run, atom_id: str, where, oracle) -> bool:
    spec = run.artifact.atom_specs.get(atom_id)
    if spec is None:
        return False
    if run.is_derived(atom_id):
        return await _ground_derived_async(run, atom_id, oracle)
    if run.single_tool is not None and _is_per_tool_static(spec):
        grounded = await oracle.ground_tool_async(spec, run.single_tool, run.binding)
        return grounded.value
    return await _ground_value_async(spec, where, oracle, run.binding)


async def _walk_tree_async(dfa, state, oracle, where, is_last, run: _Run) -> dict:
    guard = (dfa.final_guards if is_last else dfa.normal_guards)[state]
    valuation: dict[str, bool] = {}
    while isinstance(guard, GuardNode):
        value = valuation[guard.atom] = await _ground_atom_async(run, guard.atom, where, oracle)
        _record_check(run, guard.atom, where, value, oracle)
        guard = guard.hi if value else guard.lo
    skipped = [a for a in _ground_order(dfa, state, is_last, run) if a not in valuation]
    _record_skipped(run, skipped, where)
    return valuation


async def _ground_relevant_async(dfa, state, oracle, where, is_last, run: _Run) -> dict:
    if _walks_tree(dfa, state, where, run):
        return await _walk_tree_async(dfa, state, oracle, where, is_last, run)
    ordered = _ground_order(dfa, state, is_last, run)
    valuation: dict[str, bool] = {}
    for index, atom_id in enumerate(ordered):
        valuation[atom_id] = await _ground_atom_async(run, atom_id, where, oracle)
        _record(run, atom_id, where, valuation[atom_id])
        if dfa.partial_step(state, valuation, is_last=is_last) is not None:
            _record_skipped(run, ordered[index + 1 :], where)
            break
    return valuation


async def _ground_derived_async(run: _Run, atom_id: str, oracle) -> bool:
    value = run.derived.get(atom_id)
    if value is None:
        plan = run.artifact.tool_quantifiers.get(atom_id)
        if plan is not None:
            value = await _ground_tool_quantifier_async(atom_id, plan, run, oracle)
        else:
            value = await _ground_closed_async(atom_id, run, oracle)
        run.derived[atom_id] = value
    return value


async def _ground_closed_async(atom_id: str, run: _Run, oracle) -> bool:
    path: list[_Transition] | None = [] if run.closed_runs is not None else None
    dfa = run.artifact.closed_conditions[atom_id].dfa
    matched, walked = await _walk_async(run, dfa, oracle, path)
    run.positions.update(walked)
    if run.closed_runs is not None and path is not None:
        run.closed_runs.append((atom_id, matched, tuple(path)))
    return matched


def _record_pruned(
    artifact, oracle, trace_builder, candidate_index, pruned: _Pruned, record=None
) -> None:
    """Record a pruned candidate as the unmatched attempt of each tool binding, and the
    values that ruled it out in ``record``."""
    if record is not None:
        for evaluation in pruned.evaluations:
            if evaluation.status in ("evaluated_true", "evaluated_false"):
                position = None if evaluation.position == -1 else evaluation.position
                record.add(evaluation.atom_id, position, evaluation.status == "evaluated_true")
    if trace_builder is None:
        return
    uses_tools_static = _uses_tools_static(artifact)
    for tool_index, _ in _tool_bindings(artifact, oracle):
        trace_builder.add_attempt(
            kind="capture_tool" if uses_tools_static else "capture",
            binding_index=candidate_index,
            capture_binding_index=candidate_index,
            tool_binding_index=tool_index,
            matched=False,
            evaluations=list(pruned.evaluations),
        )


def _unbound_run_may_match(artifact) -> bool:
    """Whether the rule can match with every capture atom False. Only then can the run with
    no bound value decide anything; with a required bind it never can."""
    fixed: dict[str, bool] = dict.fromkeys(artifact.capture_plan.all_capture_atom_ids(), False)
    return _can_reject_with(artifact, fixed)


def _can_reject_with(artifact, fixed: dict[str, bool]) -> bool:
    """Whether the rule can match while every atom of ``fixed`` has its given value at every
    step. A closed part whose automaton cannot match then is false throughout, which the
    rule's automaton, or an enclosing part's, reads as another fixed atom."""
    fixed = dict(fixed)
    open_parts = dict(artifact.closed_conditions)
    settled = True
    while settled:
        settled = False
        for atom_id, plan in list(open_parts.items()):
            if not plan.dfa.can_reject_with(fixed):
                fixed[atom_id] = False
                del open_parts[atom_id]
                settled = True
    return artifact.dfa.can_reject_with(fixed)


async def _run_with_capture_async(
    artifact,
    events,
    oracle,
    candidates,
    trace_builder: _TraceBuilder | None = None,
    should_stop=None,
    warnings: list[str] | None = None,
    record: ScanRecord | None = None,
) -> tuple[bool, list[int], str | None]:
    """Run the DFA once per candidate capture value (or pinned element); the first accepting
    run wins. Returns ``(matched, witness, winning_value)``, where the value is the
    ``... as v`` substring (or the element's path) that satisfied the rule.

    ``candidates`` yields values, ``_SKIPPED`` for a pause or an element outside its
    collection, ``captures.MORE`` when enumeration was cut off, a :class:`_Pruned` for a
    candidate ruled out before its run, and ``None`` in place of a candidate a replay no
    longer has. Only a candidate that runs counts toward the cap. Candidate work is
    synchronous, so the run
    yields to the event loop between candidates and at enumeration pauses, and
    ``should_stop``, checked at both, lets a minimization replay give up in time."""
    uses_tools_static = _uses_tools_static(artifact)
    runs = 0
    truncated = False
    candidate_index = -1
    for value in candidates:
        if value is _SKIPPED:
            _stop_if(should_stop)
            await asyncio.sleep(0)
            continue
        if isinstance(value, _Pruned):
            candidate_index += 1
            _record_pruned(artifact, oracle, trace_builder, candidate_index, value, record)
            # Pruning is cheap and its checks pause on their own (``_SKIPPED``); yield now and
            # then so a long run of pruned candidates still lets the loop and a stop in.
            if candidate_index % _PRUNED_PER_PAUSE == 0:
                _stop_if(should_stop)
                await asyncio.sleep(0)
            continue
        if value is _cap.MORE or runs == _cap.MAX_CAPTURE_VALUES:
            truncated = True
            break
        candidate_index += 1
        if value is None:
            continue
        _stop_if(should_stop)
        await asyncio.sleep(0)
        runs += 1
        for tool_index, tool in _tool_bindings(artifact, oracle):
            evaluations: list[AtomEvaluation] | None = [] if trace_builder is not None else None
            tool_attempts: list[_ToolCheck] | None = [] if trace_builder is not None else None
            transitions: list[_Transition] | None = [] if trace_builder is not None else None
            closed_runs: list[_ClosedRun] | None = [] if trace_builder is not None else None
            matched, witness = await _run_trace_async(
                artifact,
                events,
                oracle,
                value,
                single_tool=tool,
                evaluations=evaluations,
                transitions=transitions,
                tool_attempts=tool_attempts,
                closed_runs=closed_runs,
                should_stop=should_stop,
                record=record,
            )
            attempt_id = None
            if trace_builder is not None:
                attempt_id = trace_builder.add_attempt(
                    kind="capture_tool" if uses_tools_static else "capture",
                    binding_index=candidate_index,
                    capture_binding_index=candidate_index,
                    tool_binding_index=tool_index,
                    matched=matched,
                    evaluations=evaluations or [],
                    transitions=transitions,
                    tool_attempts=tool_attempts,
                    closed_runs=closed_runs,
                )
            if matched:
                if trace_builder is not None and attempt_id is not None:
                    trace_builder.select(attempt_id)
                return True, witness, _evidence_value(value)
    if _unbound_run_may_match(artifact):
        matched, witness = await _run_once_async(
            artifact,
            events,
            oracle,
            trace_builder=trace_builder,
            should_stop=should_stop,
            record=record,
        )
        if matched:
            return True, witness, None
    # A cap that cut candidates off leaves a non-match undecided.
    if truncated and warnings is not None:
        warnings.append(_cap_warning(artifact.capture_plan))
    return False, [], None
