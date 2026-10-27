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

"""Symbolic DFA over a reduced guard decision graph.

Transitions are not enumerated 2ⁿ complete-valuation tables. Each state's transition is a
*reduced ordered decision graph* (BDD-like) over the atoms the state actually branches on:

    guard node = test one atom -> (hi: atom true, lo: atom false)
    guard leaf = a next-state id

Reduction: a node whose ``hi`` and ``lo`` are identical collapses to that child, and
structurally identical nodes are interned. So a state that only depends on 2 of 50
atoms has a 2-deep guard, not a 2⁵⁰ table. This makes the query hooks the lazy /
cheap-first / witness executor needs *first-class* rather than a scan:

    relevant_atoms(state)      — atoms that can change this state's transition
    partial_step(state, known) — next state if already determined by ``known``, else None

The DFA is the runtime decision engine: ``step`` walks the guard graph and never calls
``temporal.progress``. Progression is used only at *build* time, in ``compile``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from quarry.engine.budget import check
from quarry.engine.temporal import (
    FALSE,
    TRUE,
    And,
    Atom,
    Formula,
    Implies,
    Not,
    Or,
    atoms_of,
    branch_atoms,
    canonical,
    is_const_false,
    is_const_true,
    is_propositional,
    merge_next,
    progress,
    restrict,
    simplify,
    step_skeleton,
)

# Entries kept per DFA by the partial-step memo before it is cleared.
_PARTIAL_MEMO_LIMIT = 65_536
_MISS = object()


@dataclass(frozen=True)
class GuardLeaf:
    """Terminal of a guard graph: the next DFA state id."""

    target: int


@dataclass(frozen=True)
class GuardNode:
    """Internal decision node: test ``atom``; go ``hi`` if true, ``lo`` if false.

    ``node_id`` is a stable interning id (assigned by the builder). Frozen + hashable
    so structurally identical guards share one object.
    """

    node_id: int
    atom: str
    hi: Guard
    lo: Guard


Guard = GuardLeaf | GuardNode


# A past subformula is stood for in the formula by a reserved token atom, so the temporal
# walkers never see a past operator. The builder keeps, per state, one memory value per
# definition and computes each token's value at a step from that memory and the step's
# atoms, one update equation per operator.

PAST_PREFIX = "\x00past"


@dataclass(frozen=True)
class PastDef:
    """The past-time subformula that the atom ``token`` stands for.

    ``op`` is ``"Y"`` (``operand`` held at the previous step; false at the first step) or
    ``"O"`` (``operand`` held at some step from ``bound`` steps before now up to now;
    ``bound=None`` is unbounded). ``operand`` is propositional over rule atoms and the tokens
    of earlier definitions."""

    token: str
    op: str
    operand: Formula
    bound: int | None = None


# Memory per definition: Y keeps the operand's value at the previous step; unbounded O
# keeps whether the operand has held; bounded O keeps the steps since the operand last
# held, or None once that is beyond the bound.
Memory = tuple


def _check_past(formula: Formula, past: tuple[PastDef, ...]) -> None:
    """Every token the formula reads is defined, every operand is propositional and reads
    only tokens defined before it (a token's value is computed from inner tokens first), and
    every O bound is ``None`` or a step count."""
    defined: set[str] = set()
    for d in past:
        if not d.token.startswith(PAST_PREFIX) or d.token in defined:
            raise ValueError(f"past definition token {d.token!r} is not a fresh past token")
        if d.op not in ("Y", "O") or (d.op == "Y" and d.bound is not None):
            raise ValueError(f"past definition {d.token!r} has an unknown operator {d.op!r}")
        if d.bound is not None and (type(d.bound) is not int or d.bound < 0):
            raise ValueError(f"past definition {d.token!r} has an invalid bound {d.bound!r}")
        if not is_propositional(d.operand):
            raise ValueError(f"past definition {d.token!r} has a non-propositional operand")
        unknown = {a for a in atoms_of(d.operand) if a.startswith(PAST_PREFIX)} - defined
        if unknown:
            raise ValueError(
                f"past definition {d.token!r} reads {sorted(unknown)} before they are defined"
            )
        defined.add(d.token)
    undefined = {a for a in atoms_of(formula) if a.startswith(PAST_PREFIX)} - defined
    if undefined:
        raise ValueError(f"formula reads undefined past tokens {sorted(undefined)}")


def _initial_memory(past: tuple[PastDef, ...]) -> Memory:
    return tuple(None if d.op == "O" and d.bound is not None else False for d in past)


def _substitute(formula: Formula, values: dict[str, Formula]) -> Formula:
    """``formula`` with the atoms named in ``values`` replaced; propositional formulas only."""
    match formula:
        case Atom(name):
            return values.get(name, formula)
        case Not(child):
            return Not(_substitute(child, values))
        case And(left, right):
            return And(_substitute(left, values), _substitute(right, values))
        case Or(left, right):
            return Or(_substitute(left, values), _substitute(right, values))
        case Implies(left, right):
            return Implies(_substitute(left, values), _substitute(right, values))
    return formula


def _step_equations(
    past: tuple[PastDef, ...], memory: Memory
) -> tuple[dict[str, Formula], list[Formula]]:
    """Each token's value at this step, and each definition's memory update, as formulas
    over the step's rule atoms (inner tokens already substituted)."""
    now: dict[str, Formula] = {}
    updates: list[Formula] = []
    for d, mem in zip(past, memory, strict=True):
        operand = simplify(_substitute(d.operand, now))
        if d.op == "Y":
            now[d.token] = TRUE if mem else FALSE
            updates.append(operand)
        elif d.bound is None:
            now[d.token] = TRUE if mem else operand
            updates.append(now[d.token])
        else:
            within = mem is not None and mem + 1 <= d.bound
            now[d.token] = TRUE if within else operand
            updates.append(operand)
    return now, updates


def _step_values(
    past: tuple[PastDef, ...], memory: Memory, valuation: dict[str, bool]
) -> tuple[dict[str, bool], Memory]:
    """Token values at a step with atom ``valuation``, and the memory after it."""
    now: dict[str, bool] = {}
    new: list = []
    for d, mem in zip(past, memory, strict=True):
        held = is_const_true(simplify(progress(d.operand, {**valuation, **now}, False)))
        if d.op == "Y":
            now[d.token] = bool(mem)
            new.append(held)
        elif d.bound is None:
            now[d.token] = bool(mem) or held
            new.append(now[d.token])
        else:
            age = 0 if held else (mem + 1 if mem is not None and mem + 1 <= d.bound else None)
            now[d.token] = age is not None
            new.append(age)
    return now, tuple(new)


def _live_past(past: tuple[PastDef, ...], residual: Formula) -> set[int]:
    """Indexes of definitions whose memory can still matter: tokens the residual reads, and
    the tokens their operands read."""
    by_token = {d.token: i for i, d in enumerate(past)}
    live: set[int] = set()
    pending = [a for a in atoms_of(residual) if a in by_token]
    while pending:
        i = by_token[pending.pop()]
        if i in live:
            continue
        live.add(i)
        pending.extend(a for a in atoms_of(past[i].operand) if a in by_token)
    return live


def _settle(past: tuple[PastDef, ...], residual: Formula, memory: Memory):
    """Fold tokens whose value can no longer change into the residual, and reset the memory
    of definitions nothing reads any more.

    A token is fixed when an unbounded O has held, or when its operand is fixed (by earlier
    tokens) and its memory already agrees: ``Y`` of a fixed operand once its memory holds that
    value, ``O`` of an operand fixed true. Definitions read only earlier tokens, so one pass in
    order finds every fixed token."""
    fixed: dict[str, Formula] = {}
    for d, mem in zip(past, memory, strict=True):
        operand = simplify(_substitute(d.operand, fixed))
        if d.op == "Y":
            if bool(mem) and is_const_true(operand):
                fixed[d.token] = TRUE
            elif not mem and is_const_false(operand):
                fixed[d.token] = FALSE
        elif d.bound is None:
            if mem or is_const_true(operand):
                fixed[d.token] = TRUE
            elif is_const_false(operand):
                fixed[d.token] = FALSE
        elif is_const_true(operand):
            fixed[d.token] = TRUE
        elif mem is None and is_const_false(operand):
            fixed[d.token] = FALSE
    if fixed:
        residual = simplify(_substitute_temporal(residual, fixed))
    live = _live_past(past, residual)
    initial = _initial_memory(past)
    memory = tuple(mem if i in live else initial[i] for i, mem in enumerate(memory))
    return residual, memory


def _substitute_temporal(formula: Formula, values: dict[str, Formula]) -> Formula:
    """``formula`` with the named atoms replaced, under temporal operators too."""
    if isinstance(formula, Atom):
        return values.get(formula.name, formula)
    fields = getattr(formula, "__dataclass_fields__", {})
    if not fields:
        return formula
    changed: dict[str, Any] = {
        name: _substitute_temporal(getattr(formula, name), values)
        if not isinstance(getattr(formula, name), int)
        else getattr(formula, name)
        for name in fields
    }
    return type(formula)(**changed)


@dataclass
class DFAState:
    """A state in the compiled DFA (an interned formula residual)."""

    id: int
    key: str
    is_accepting: bool
    is_rejecting: bool


class _GuardBuilder:
    """Builds reduced, interned guard graphs for one DFA compilation.

    Shannon-expands a state's residual over its branch atoms (in a fixed order), then
    reduces: identical hi/lo collapse, identical subgraphs are shared. ``next_state``
    maps a fully-progressed residual to its interned state id.

    Cofactor pruning. Deciding k branch atoms is 2^k in the worst case, but many real
    shapes collapse far earlier: once a partial assignment fixes the transition, the
    remaining atoms cannot change it. We detect that by carrying the step's boolean
    *skeleton* (``temporal.step_skeleton`` — the current-step transition as a pure
    boolean function, temporal carries opaque) alongside the recursion and cofactoring
    it (``temporal.restrict``) as atoms are decided; an atom no longer in the cofactored
    skeleton's support cannot change the transition and is skipped. A wide single-step
    disjunction (the lexical-prefilter shape: `any call where $a or $b or …`) lowers to
    `¬F(a₀∨…∨aₖ)`; one true atom makes `¬(a₀∨…)` false now and collapses the whole step,
    so its guard builds in O(k) leaves, not 2^k. The *leaf* is still computed by
    ``_progress_full`` over the accumulated assignment, so the guard produced is
    identical to the un-pruned Shannon expansion (pruning only skips branches the
    `hi == lo` reduction would have merged anyway).

    Genuinely branchy steps do not prune — e.g. a nested-F safety chain
    `¬F(a₀∧F(a₁∧…))` re-reads every atom each step, so no single assignment collapses
    it. Those stay 2^k, and every leaf and branch checks the compile's work budget
    (``quarry.engine.budget``): construction fails fast with :class:`DFAConstructionLimit`
    rather than hanging. (A fully general lift needs a symbolic BDD/MONA backend — a later
    effort.)
    """

    def __init__(
        self,
        next_state,
        is_last: bool,
        past: tuple[PastDef, ...] = (),
        memory: Memory = (),
    ):
        self._next_state = next_state
        self._is_last = is_last
        self._past = past
        self._memory = memory
        self._nodes: dict[tuple, GuardNode] = {}
        # This state's subgraph for each (cofactored parts, next live atom): two assignments
        # that leave the same parts decide the rest of the step alike.
        self._built: dict[tuple, Guard] = {}
        self._counter = 0

    def build(self, residual: Formula, atoms: list[str]) -> Guard:
        """Build the guard."""
        self._built = {}
        skeleton = step_skeleton(residual, self._is_last)
        if not self._past:
            read = set(branch_atoms(residual))
            live = [a for a in atoms if a in read]
            return self._expand(residual, (skeleton,), live, 0, {})
        # A token's value and a live definition's memory update are functions of this step's
        # atoms: the skeleton reads the former, and every live update is kept beside it, so an
        # atom that only changes the next memory is never pruned.
        now, updates = _step_equations(self._past, self._memory)
        live_defs = _live_past(self._past, residual)
        parts = (
            simplify(_substitute(skeleton, now)),
            *(update for i, update in enumerate(updates) if i in live_defs),
        )
        read = set(branch_atoms(residual)) | {
            a for i in live_defs for a in atoms_of(self._past[i].operand)
        }
        live = [a for a in atoms if a in read]
        return self._expand(residual, parts, live, 0, {})

    def _leaf(self, residual: Formula, assigned: dict) -> GuardLeaf:
        check()
        if not self._past:
            return GuardLeaf(self._next_state(_progress_full(residual, assigned, self._is_last)))
        tokens, memory = _step_values(self._past, self._memory, assigned)
        successor = _progress_full(residual, {**assigned, **tokens}, self._is_last)
        return GuardLeaf(self._next_state(successor, memory))

    def _expand(
        self,
        residual: Formula,
        parts: tuple[Formula, ...],
        live: list[str],
        i: int,
        assigned: dict,
    ) -> Guard:
        # Skip atoms the accumulated assignment has already made irrelevant: an atom in the
        # support of no cofactored part cannot change this step's outcome. (Opaque carry
        # tokens may appear in the support but never match a live atom.)
        relevant = {a for part in parts for a in branch_atoms(part)}
        while i < len(live) and live[i] not in relevant:
            i += 1
        key = (parts, i)
        built = self._built.get(key)
        if built is not None:
            return built
        if i >= len(live):
            guard = self._leaf(residual, assigned)
            self._built[key] = guard
            return guard
        check()
        atom = live[i]
        hi = self._expand(
            residual,
            tuple(simplify(restrict(part, {atom: True})) for part in parts),
            live,
            i + 1,
            {**assigned, atom: True},
        )
        lo = self._expand(
            residual,
            tuple(simplify(restrict(part, {atom: False})) for part in parts),
            live,
            i + 1,
            {**assigned, atom: False},
        )
        # Reduction: identical hi/lo means the atom is irrelevant along this path.
        guard = hi if hi == lo else self._intern(atom, hi, lo)
        self._built[key] = guard
        return guard

    def _intern(self, atom: str, hi: Guard, lo: Guard) -> GuardNode:
        sig = (atom, _guard_sig(hi), _guard_sig(lo))
        existing = self._nodes.get(sig)
        if existing is not None:
            return existing
        node = GuardNode(node_id=self._counter, atom=atom, hi=hi, lo=lo)
        self._counter += 1
        self._nodes[sig] = node
        return node


def _guard_sig(g: Guard):
    if isinstance(g, GuardLeaf):
        return ("leaf", g.target)
    return ("node", g.node_id)


def _progress_full(residual: Formula, assigned: dict, is_last: bool) -> Formula:
    """Progress ``residual`` under a full assignment of its branch atoms. Atoms not in
    ``assigned`` are treated as False (they are, by construction, not branch atoms of
    ``residual`` so their value cannot change the result)."""
    return simplify(progress(residual, assigned, is_last))


@dataclass
class DFA:
    """Compiled DFA from an LTLf formula, using reduced guard decision graphs."""

    formula_name: str
    atoms: list[str]
    states: list[DFAState]
    initial_state: int
    # per state: the guard root for a normal step and for the final (last) step
    normal_guards: dict[int, Guard]
    final_guards: dict[int, Guard]

    @property
    def num_states(self) -> int:
        """Return the number of states."""
        return len(self.states)

    def is_accepting(self, state_id: int) -> bool:
        """Return whether the state is accepting."""
        return self.states[state_id].is_accepting

    def is_rejecting(self, state_id: int) -> bool:
        """Return whether the state is rejecting."""
        return self.states[state_id].is_rejecting

    def step(self, state_id: int, valuation: dict[str, bool], is_last: bool = False) -> int:
        """Advance one step by walking the guard graph. O(depth) ≤ O(#branch atoms)."""
        guard = (self.final_guards if is_last else self.normal_guards)[state_id]
        while isinstance(guard, GuardNode):
            guard = guard.hi if valuation.get(guard.atom, False) else guard.lo
        return guard.target

    def relevant_atoms(self, state_id: int, is_last: bool = False) -> list[str]:
        """Atoms whose value can change this state's transition, in guard order.

        These are exactly the atoms appearing on some path of the reduced guard graph;
        any atom not here is irrelevant at this state and need not be grounded (lazy
        grounding). Order follows the DFA's global atom order, so a cheap-first
        executor can reorder within it.
        """
        # A pure function of the fixed guard graph, asked at every step: memoized outside the
        # dataclass fields, which the compiled-policy format pins.
        memo = self.__dict__.setdefault("_relevant_memo", {})
        key = (state_id, is_last)
        atoms = memo.get(key)
        if atoms is None:
            guard = (self.final_guards if is_last else self.normal_guards)[state_id]
            found: list[str] = []
            self._collect_guard_atoms(guard, found, set())
            atoms = memo[key] = tuple(a for a in self.atoms if a in found)
        return list(atoms)

    def _collect_guard_atoms(self, guard: Guard, out: list[str], seen: set) -> None:
        if isinstance(guard, GuardLeaf):
            return
        if guard.node_id in seen:
            return
        seen.add(guard.node_id)
        if guard.atom not in out:
            out.append(guard.atom)
        self._collect_guard_atoms(guard.hi, out, seen)
        self._collect_guard_atoms(guard.lo, out, seen)

    def partial_step(
        self, state_id: int, known: dict[str, bool], is_last: bool = False
    ) -> int | None:
        """Next state if it is already determined by ``known`` alone, else ``None``.

        Walks the guard graph using only known atoms; if it reaches a node whose test
        atom is unknown, it returns ``None`` *unless both branches lead to the same
        target* (in which case the unknown atom cannot change the outcome). This is the
        cheap-first short-circuit: ground atoms incrementally and stop as soon as the
        transition is pinned.
        """
        # Asked after every grounded atom, and events repeat the same partial valuations, so
        # answers are memoized (bounded) outside the dataclass fields.
        memo = self.__dict__.setdefault("_partial_memo", {})
        key = (state_id, is_last, tuple(known.items()))
        # A miss sentinel, since None is a valid answer; the result stays local so a
        # concurrent clear() cannot remove it between the store and the return.
        target = memo.get(key, _MISS)
        if target is _MISS:
            if len(memo) >= _PARTIAL_MEMO_LIMIT:
                memo.clear()
            guard = (self.final_guards if is_last else self.normal_guards)[state_id]
            target = self._resolve_partial(guard, known)
            memo[key] = target
        return target

    def _resolve_partial(self, guard: Guard, known: dict[str, bool]) -> int | None:
        if isinstance(guard, GuardLeaf):
            return guard.target
        if guard.atom in known:
            branch = guard.hi if known[guard.atom] else guard.lo
            return self._resolve_partial(branch, known)
        # Unknown atom: determined only if both branches agree on a target, so an undetermined
        # `hi` already decides the answer.
        hi = self._resolve_partial(guard.hi, known)
        if hi is None:
            return None
        return hi if self._resolve_partial(guard.lo, known) == hi else None

    def can_reject_with(self, fixed: dict[str, bool]) -> bool:
        """True if some run in which every atom of ``fixed`` has its given value at every
        step reaches a rejecting state, through normal steps or the final step."""
        return self._can_reach_with(fixed, self.is_rejecting)

    def can_accept_with(self, fixed: dict[str, bool]) -> bool:
        """True if some run compatible with ``fixed`` reaches an accepting state."""
        return self._can_reach_with(fixed, self.is_accepting)

    def _can_reach_with(self, fixed: dict[str, bool], terminal: Callable[[int], bool]) -> bool:
        if terminal(self.initial_state):
            return True
        seen = {self.initial_state}
        frontier = [self.initial_state]
        while frontier:
            state = frontier.pop()
            if any(terminal(t) for t in _guard_targets(self.final_guards[state], fixed)):
                return True
            for target in _guard_targets(self.normal_guards[state], fixed):
                if terminal(target):
                    return True
                if target not in seen:
                    seen.add(target)
                    frontier.append(target)
        return False

    def validate(self) -> None:
        """Assert the DFA is total and deterministic: every state has both guard roots,
        and every guard leaf targets a real state. Raises ``ValueError`` otherwise.
        (Determinism is structural — a guard graph yields exactly one leaf per
        valuation — so validation checks totality and target integrity.)"""
        n = self.num_states
        if not (0 <= self.initial_state < n):
            raise ValueError(f"initial_state {self.initial_state} out of range [0,{n})")
        for sid in range(n):
            if sid not in self.normal_guards or sid not in self.final_guards:
                raise ValueError(f"state {sid} missing a guard root")
            for guards in (self.normal_guards, self.final_guards):
                for target in _guard_targets(guards[sid]):
                    if not (0 <= target < n):
                        raise ValueError(f"guard leaf targets missing state {target}")

    def summary(self) -> str:
        """Return the summary of the DFA."""
        accepting = sum(1 for s in self.states if s.is_accepting)
        rejecting = sum(1 for s in self.states if s.is_rejecting)
        return (
            f"DFA({self.formula_name}): {self.num_states} states "
            f"({accepting} accepting, {rejecting} rejecting), {len(self.atoms)} atoms"
        )


def _guard_targets(guard: Guard, fixed: dict[str, bool] | None = None) -> set[int]:
    """All next-state ids reachable through a guard graph, following only the branch of each
    atom whose value ``fixed`` gives."""
    out: set[int] = set()
    stack = [guard]
    seen_nodes: set[int] = set()
    while stack:
        g = stack.pop()
        if isinstance(g, GuardLeaf):
            out.add(g.target)
        elif g.node_id not in seen_nodes:
            seen_nodes.add(g.node_id)
            if fixed is not None and g.atom in fixed:
                stack.append(g.hi if fixed[g.atom] else g.lo)
            else:
                stack.append(g.hi)
                stack.append(g.lo)
    return out


@dataclass
class _CompileState:
    key_to_id: dict[str, int] = field(default_factory=dict)
    states: list[DFAState] = field(default_factory=list)
    residuals: dict[int, Formula] = field(default_factory=dict)
    memories: dict[int, Memory] = field(default_factory=dict)


def compile_dfa(
    formula: Formula,
    formula_name: str = "",
    past: tuple[PastDef, ...] = (),
    *,
    atom_order: Callable[[str], Any] | None = None,
) -> DFA:
    """Compile an LTLf formula into a guard-graph DFA.

    ``atom_order`` sorts the atoms the guards test, first to last; by default they are
    tested in the order they first appear in ``formula``.

    Construction is bounded by the work budget of the compile running in this context
    (``quarry.engine.budget``), which the automata of one rule share: the formula walkers
    count their visits, and every guard leaf and branch and every new state checks the
    count, failing with :class:`DFAConstructionLimit` past the budget. Outside a budgeted
    compile, construction is unbounded.

    ``past`` defines the past-time subformulas that ``PAST_PREFIX`` token atoms in
    ``formula`` stand for (see :class:`PastDef`); a state is then its residual together with
    the memory of the definitions it still reads.

    Explores reachable formula residuals via compile-time progression; each unique
    canonical residual is an interned state. Transitions are stored as reduced guard
    graphs, not enumerated tables — so *runtime* stepping is O(guard depth) and never
    2^n. Building a guard is 2^k in the atoms a step reads only when nothing prunes;
    cofactor pruning collapses the common wide-but-decidable shapes (e.g. a prefilter's
    `$a or $b or …` disjunction) to O(k) (see ``_GuardBuilder``).
    """
    _check_past(formula, past)
    tokens = {d.token for d in past}
    # Operand atoms in token order, so the order of independent definitions does not change
    # the guards' atom order.
    by_token = sorted(past, key=lambda d: d.token)
    atoms = [
        a
        for a in dict.fromkeys(
            [*atoms_of(formula), *(a for d in by_token for a in atoms_of(d.operand))]
        )
        if a not in tokens
    ]
    if atom_order is not None:
        # Guards test atoms in this order: a stable sort keeps first appearance among ties.
        atoms.sort(key=atom_order)
    cs = _CompileState()
    # Many leaves of a state's guard progress to the same successor: normalize each
    # formula object once. Hashing a structural Formula key recursively descends through
    # every And/Or node, which can exceed Python's recursion limit for a valid wide rule.
    # Keep the objects alive so their ids cannot be reused during this compilation.
    interned: dict[tuple[int, Memory], tuple[Formula, int]] = {}

    def intern(residual: Formula, memory: Memory = ()) -> int:
        raw = (id(residual), memory)
        cached = interned.get(raw)
        if cached is not None and cached[0] is residual:
            return cached[1]
        sid = _intern(residual, memory)
        interned[raw] = (residual, sid)
        return sid

    def _intern(residual: Formula, memory: Memory) -> int:
        residual = simplify(merge_next(simplify(residual)))
        key = canonical(residual)
        if past:
            residual, memory = _settle(past, residual, memory)
            residual = simplify(merge_next(residual))
            key = canonical(residual)
            live = _live_past(past, residual)
            if live:
                key += "\x00memory" + repr(sorted((past[i].token, memory[i]) for i in live))
        existing = cs.key_to_id.get(key)
        if existing is not None:
            return existing
        # A new state checks the budget as it is minted, so one state's guard build,
        # which interns a successor per branch valuation, stops as soon as the budget is spent.
        check()
        sid = len(cs.states)
        cs.key_to_id[key] = sid
        cs.states.append(
            DFAState(
                id=sid,
                key=key,
                is_accepting=is_const_true(residual),
                is_rejecting=is_const_false(residual),
            )
        )
        cs.residuals[sid] = residual
        cs.memories[sid] = memory
        return sid

    initial = intern(simplify(formula), _initial_memory(past))

    normal_guards: dict[int, Guard] = {}
    final_guards: dict[int, Guard] = {}

    sid = 0
    while sid < len(cs.states):
        residual = cs.residuals[sid]
        memory = cs.memories[sid]
        normal_guards[sid] = _GuardBuilder(intern, False, past, memory).build(residual, atoms)
        final_guards[sid] = _GuardBuilder(intern, True, past, memory).build(residual, atoms)
        sid += 1

    dfa = DFA(
        formula_name=formula_name,
        atoms=atoms,
        states=cs.states,
        initial_state=initial,
        normal_guards=normal_guards,
        final_guards=final_guards,
    )
    dfa.validate()
    return dfa


def run_trace(dfa: DFA, trace: list[dict[str, bool]]) -> bool:
    """Return True if the formula is *satisfied* over the finite ``trace`` (empty trace
    is vacuously satisfied). Pure guard-graph stepping — no progression."""
    if not trace:
        return True
    state = dfa.initial_state
    last = len(trace) - 1
    for i, step_vals in enumerate(trace):
        state = dfa.step(state, step_vals, is_last=(i == last))
        if dfa.is_rejecting(state):
            return False
    return dfa.is_accepting(state)
