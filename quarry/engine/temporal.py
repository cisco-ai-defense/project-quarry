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

"""LTLf (Linear Temporal Logic over finite traces) formula layer.

Self-contained temporal-logic core for the Quarry DFA backend. This
module has **no imports from `quarry.types.dsl`** — it knows nothing about the rule
AST. The compiler lowers rules into these formulas; this layer only
provides:

- an immutable formula AST (constants, atoms, boolean and temporal operators),
- ``simplify`` — constant-folding normalization,
- ``progress`` — the one-step symbolic derivative used **at compile time** to build
  the DFA (never called at runtime; the DFA is the runtime decision engine),
- ``branch_atoms`` — the atoms a formula's *current* step depends on (everything not
  strictly under a ``Next``); this is what makes the guard decision graph reduced and
  lazy grounding first-class,
- ``canonical`` — an order-independent string key for deterministic hashing / state
  interning.

Finite-trace semantics:
- ``F(phi)`` at the last step: ``phi`` must hold now (no future steps).
- ``G(phi)`` at the last step: ``phi`` must hold now (trivially).
- ``X(phi)`` at the last step: FALSE (no next step exists).

Excluded by design: complete-valuation transition tables, atom caps, runtime progression
fallback, trace/analyzer types.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from quarry.engine.budget import current as _work_budget

# Each call of a walker below that building an automaton repeats is one formula-node visit of
# the compile running in this context (``quarry.engine.budget``).
_budget_now = _work_budget.get


def _charge_walk(visits: int) -> None:
    """Charge a whole walk's node visits at once (the recursive walkers count per call)."""
    budget = _budget_now()
    if budget is not None:
        budget.used += visits


@dataclass(frozen=True)
class TrueConst:
    """True constant."""

    pass


@dataclass(frozen=True)
class FalseConst:
    """False constant."""

    pass


@dataclass(frozen=True)
class Atom:
    """Atom."""

    name: str


@dataclass(frozen=True)
class Not:
    """Not."""

    child: Formula


@dataclass(frozen=True)
class And:
    """And."""

    left: Formula
    right: Formula


@dataclass(frozen=True)
class Or:
    """Or."""

    left: Formula
    right: Formula


@dataclass(frozen=True)
class Implies:
    """Implies."""

    left: Formula
    right: Formula


@dataclass(frozen=True)
class Next:
    """X(phi) — phi must hold at the next time step."""

    child: Formula


@dataclass(frozen=True)
class Eventually:
    """F(phi) — phi must hold at some point from now until end of trace."""

    child: Formula


@dataclass(frozen=True)
class BoundedEventually:
    """F≤k(phi) — phi holds now or within the next ``bound`` steps.

    One node whatever the bound, so a long ``within:N`` window does not nest ``N`` levels of
    ``Next`` that every formula walker would have to recurse through."""

    child: Formula
    bound: int


@dataclass(frozen=True)
class Globally:
    """G(phi) — phi must hold at every point from now until end of trace."""

    child: Formula


@dataclass(frozen=True)
class Until:
    """left U right — right must hold at some point, and left must hold at every
    step until then (finite-trace semantics)."""

    left: Formula
    right: Formula


Formula = (
    TrueConst
    | FalseConst
    | Atom
    | Not
    | And
    | Or
    | Implies
    | Next
    | Eventually
    | BoundedEventually
    | Globally
    | Until
)

TRUE = TrueConst()
FALSE = FalseConst()


def simplify(f: Formula) -> Formula:
    """Simplify a formula by eliminating trivial constants and double negation."""
    budget = _budget_now()
    if budget is not None:
        budget.used += 1
    match f:
        case TrueConst() | FalseConst() | Atom():
            return f
        case Not(TrueConst()):
            return FALSE
        case Not(FalseConst()):
            return TRUE
        case Not(Not(inner)):
            return simplify(inner)
        case Not(child):
            sc = simplify(child)
            if sc is child:
                return f
            return _not_of(sc)
        case And(TrueConst(), r):
            return simplify(r)
        case And(l, TrueConst()):
            return simplify(l)
        case And(FalseConst(), _) | And(_, FalseConst()):
            return FALSE
        case And(l, r):
            sl, sr = simplify(l), simplify(r)
            if sl is l and sr is r:
                return f
            return _and_of(sl, sr)
        case Or(TrueConst(), _) | Or(_, TrueConst()):
            return TRUE
        case Or(FalseConst(), r):
            return simplify(r)
        case Or(l, FalseConst()):
            return simplify(l)
        case Or(l, r):
            sl, sr = simplify(l), simplify(r)
            if sl is l and sr is r:
                return f
            return _or_of(sl, sr)
        case Implies(l, r):
            return simplify(Or(Not(l), r))
        case Next(child):
            sc = simplify(child)
            if sc is child:
                return f
            return Next(sc)
        case Eventually(TrueConst()):
            return TRUE
        case Eventually(FalseConst()):
            return FALSE
        case Eventually(child):
            sc = simplify(child)
            if sc is child:
                return f
            return sc if isinstance(sc, (TrueConst, FalseConst)) else Eventually(sc)
        case BoundedEventually(child, 0):
            return simplify(child)
        case BoundedEventually(TrueConst(), _):
            return TRUE
        case BoundedEventually(FalseConst(), _):
            return FALSE
        case BoundedEventually(child, bound):
            sc = simplify(child)
            if sc is child:
                return f
            return sc if isinstance(sc, (TrueConst, FalseConst)) else BoundedEventually(sc, bound)
        case Globally(TrueConst()):
            return TRUE
        case Globally(FalseConst()):
            return FALSE
        case Globally(child):
            sc = simplify(child)
            if sc is child:
                return f
            return sc if isinstance(sc, (TrueConst, FalseConst)) else Globally(sc)
        case Until(l, r):
            sl, sr = simplify(l), simplify(r)
            # (l U TRUE) = TRUE; (l U FALSE) = FALSE; (FALSE U r) = r
            if isinstance(sr, TrueConst):
                return TRUE
            if isinstance(sr, FalseConst):
                return FALSE
            if isinstance(sl, FalseConst):
                return sr
            if sl is l and sr is r:
                return f
            return Until(sl, sr)
    return f


# One-level rebuilds of a node whose children are already simplified: only the node's own
# constant rules can still apply, so its children are not walked again.


def _not_of(sc: Formula) -> Formula:
    match sc:
        case TrueConst():
            return FALSE
        case FalseConst():
            return TRUE
        case Not(inner):
            return inner
    return Not(sc)


def _and_of(sl: Formula, sr: Formula) -> Formula:
    if isinstance(sl, FalseConst) or isinstance(sr, FalseConst):
        return FALSE
    if isinstance(sl, TrueConst):
        return sr
    if isinstance(sr, TrueConst):
        return sl
    return And(sl, sr)


def _or_of(sl: Formula, sr: Formula) -> Formula:
    if isinstance(sl, TrueConst) or isinstance(sr, TrueConst):
        return TRUE
    if isinstance(sl, FalseConst):
        return sr
    if isinstance(sr, FalseConst):
        return sl
    return Or(sl, sr)


def merge_next(f: Formula) -> Formula:
    """Merge sibling ``Next`` operands: ``X a ∨ X b = X(a ∨ b)`` and ``X a ∧ X b = X(a ∧ b)``,
    plus their negated forms ``¬X a ∨ ¬X b = ¬X(a ∧ b)`` and ``¬X a ∧ ¬X b = ¬X(a ∨ b)``.

    Each identity holds in LTLf for strong and weak next. Progressing a bounded window
    ``F≤k φ`` leaves a disjunction of shorter windows over the same ``φ``; merging their
    ``Next`` tails folds it back into the longest window, so the residual count stays linear
    in ``k`` instead of growing with every combination of pending windows."""
    return _merge_next(f, {})


def _merge_next(f: Formula, memo: dict[int, tuple[Formula, Formula]]) -> Formula:
    # Memoized per call on object identity (the cached input is kept alive with its result),
    # so a subtree already merged is not walked again when it is folded under a new Next.
    hit = memo.get(id(f))
    if hit is not None and hit[0] is f:
        return hit[1]
    result = _merge_next_uncached(f, memo)
    memo[id(f)] = (f, result)
    memo[id(result)] = (result, result)
    return result


def _merge_next_uncached(f: Formula, memo: dict[int, tuple[Formula, Formula]]) -> Formula:
    budget = _budget_now()
    if budget is not None:
        budget.used += 1

    def merge_next(child: Formula) -> Formula:
        """Merge the next child."""
        return _merge_next(child, memo)

    match f:
        case Or() | And():
            is_or = isinstance(f, Or)
            parts = [merge_next(p) for p in (_flatten_or(f) if is_or else _flatten_and(f))]
            nexts = [p.child for p in parts if isinstance(p, Next)]
            negated = [
                p.child.child for p in parts if isinstance(p, Not) and isinstance(p.child, Next)
            ]
            rest: list[Formula] = [
                p
                for p in parts
                if not isinstance(p, Next)
                and not (isinstance(p, Not) and isinstance(p.child, Next))
            ]
            join, dual = (Or, And) if is_or else (And, Or)
            if nexts:
                rest.append(Next(merge_next(_fold(join, nexts))))
            if negated:
                rest.append(Not(Next(merge_next(_fold(dual, negated)))))
            return _fold(join, _absorb(join, _merge_windows(is_or, rest)))
        case Not(child):
            return Not(merge_next(child))
        case Next(child):
            return Next(merge_next(child))
        case Eventually(child):
            return Eventually(merge_next(child))
        case BoundedEventually(child, bound):
            return BoundedEventually(merge_next(child), bound)
        case Globally(child):
            return Globally(merge_next(child))
        case Until(left, right):
            return Until(merge_next(left), merge_next(right))
    return f


def _merge_windows(is_or: bool, parts: list[Formula]) -> list[Formula]:
    """Keep the longest of several bounded windows over the same condition.

    ``F≤a φ ∨ F≤b φ = F≤max(a,b) φ``, and ``F φ`` subsumes both; under ``And`` the negated
    forms ``¬F≤a φ ∧ ¬F≤b φ = ¬F≤max(a,b) φ`` merge the same way. Progressing a ``within:N``
    window leaves one shorter window per pending start, so this keeps the residual count
    linear in ``N``."""

    def window(part: Formula) -> BoundedEventually | None:
        inner = part if is_or else (part.child if isinstance(part, Not) else None)
        return inner if isinstance(inner, BoundedEventually) else None

    unbounded = {_norm_repr(p.child) for p in parts if is_or and isinstance(p, Eventually)}
    longest: dict[str, int] = {}
    for part in parts:
        w = window(part)
        if w is not None:
            key = _norm_repr(w.child)
            longest[key] = max(longest.get(key, 0), w.bound)
    merged: list[Formula] = []
    emitted: set[str] = set()
    for part in parts:
        w = window(part)
        if w is None:
            merged.append(part)
            continue
        key = _norm_repr(w.child)
        if key in unbounded or key in emitted:
            continue
        emitted.add(key)
        widest = BoundedEventually(w.child, longest[key])
        merged.append(widest if is_or else Not(widest))
    return merged


def _absorb(join, parts: list[Formula]) -> list[Formula]:
    """Absorption against sibling operands of one ``join`` (``Or`` or ``And``).

    ``x ∨ (x ∧ y) = x`` and ``x ∨ ((x ∨ y) ∧ z) = x ∨ (y ∧ z)``, with the dual forms for
    ``And``. These hold for any Boolean-valued operands, temporal ones included. Progressing
    a nested ``Until`` repeats ``(R ∧ P) ∨ Q`` around its previous residual; absorption folds
    that back into the earlier state instead of minting a longer one each step."""
    dual = And if join is Or else Or
    flatten_join = _flatten_or if join is Or else _flatten_and
    flatten_dual = _flatten_and if join is Or else _flatten_or
    keyed: dict[str, Formula] = {}
    for part in parts:
        keyed.setdefault(_norm_repr(part), part)
    kept_parts: list[Formula] = []
    for key, part in keyed.items():
        others = keyed.keys() - {key}
        if not isinstance(part, dual):
            kept_parts.append(part)
            continue
        operands: list[Formula] = []
        absorbed = False
        for operand in flatten_dual(part):
            if _norm_repr(operand) in others:
                absorbed = True
                break
            if isinstance(operand, join):
                remaining = [q for q in flatten_join(operand) if _norm_repr(q) not in others]
                if not remaining:
                    absorbed = True
                    break
                operand = _fold(join, remaining)
            operands.append(operand)
        if not absorbed:
            kept_parts.append(_fold(dual, operands))
    return kept_parts


def _fold(op, parts: list[Formula]) -> Formula:
    result = parts[0]
    for part in parts[1:]:
        result = op(result, part)
    return result


# Used ONLY by the DFA builder to enumerate reachable states. The DFA itself never
# progresses at runtime — see symbolic_dfa.py.


def progress(formula: Formula, valuation: dict[str, bool], is_last: bool) -> Formula:
    """Advance ``formula`` by one step given the current predicate values.

    Args:
        formula: the LTLf formula to progress.
        valuation: current step's predicate values ``{name: bool}``.
        is_last: whether this is the last step in the (finite) trace.
    """
    budget = _budget_now()
    if budget is not None:
        budget.used += 1
    match formula:
        case TrueConst():
            return TRUE
        case FalseConst():
            return FALSE
        case Atom(name):
            return TRUE if valuation.get(name, False) else FALSE
        case Not(child):
            return Not(progress(child, valuation, is_last))
        case And(l, r):
            return And(progress(l, valuation, is_last), progress(r, valuation, is_last))
        case Or(l, r):
            return Or(progress(l, valuation, is_last), progress(r, valuation, is_last))
        case Implies(l, r):
            return Or(Not(progress(l, valuation, is_last)), progress(r, valuation, is_last))
        case Next(child):
            if is_last:
                return FALSE
            return child
        case Eventually(child):
            # F(phi) = phi OR (not_last AND X(F(phi)))
            now = progress(child, valuation, is_last)
            if is_last:
                return now
            return Or(now, Eventually(child))
        case BoundedEventually(child, bound):
            # F≤k(phi) = phi OR X F≤k-1(phi); at the last step or k = 0 only phi remains.
            now = progress(child, valuation, is_last)
            if is_last or bound == 0:
                return now
            return Or(now, BoundedEventually(child, bound - 1))
        case Globally(child):
            # G(phi) = phi AND (last OR X(G(phi)))
            now = progress(child, valuation, is_last)
            if is_last:
                return now
            return And(now, Globally(child))
        case Until(l, r):
            # (l U r) = r OR (l AND not_last AND X(l U r))
            now_r = progress(r, valuation, is_last)
            if is_last:
                return now_r  # must satisfy r at the final step
            now_l = progress(l, valuation, is_last)
            return Or(now_r, And(now_l, Until(l, r)))
    raise ValueError(f"Unknown formula type: {type(formula)}")


# Used ONLY by the DFA builder, to prune guard atoms a partial assignment has already
# made irrelevant. ``restrict`` is a plain boolean cofactor (fold assigned atoms to their
# truth values); it is sound ONLY on a formula with no temporal operators — folding an
# atom under ``Eventually``/``Globally``/``Until`` is wrong, because e.g. ``F(a)`` with
# ``a`` false *now* is still ``F(a)`` (``a`` may hold later), not ``FALSE``.
#
# ``step_skeleton`` bridges that gap: it turns a residual into the pure-boolean function
# of the *current* step's transition, mirroring ``progress`` exactly — real atoms keep
# their current-step reading, and each temporal *carry* (the part ``progress`` defers to
# the next step) becomes a fresh opaque token that is never assigned. Cofactoring the
# skeleton with ``restrict`` is then sound: ``branch_atoms(simplify(restrict(skeleton,
# P)))`` (ignoring the opaque tokens, which never match a real atom) is exactly the set
# of atoms whose current value can still change the transition given ``P``.


def restrict(formula: Formula, assignment: dict[str, bool]) -> Formula:
    """Boolean cofactor: fold assigned atoms to their truth values. SOUND ONLY on a
    temporal-operator-free formula (e.g. a :func:`step_skeleton`)."""
    budget = _budget_now()
    if budget is not None:
        budget.used += 1
    match formula:
        case TrueConst() | FalseConst():
            return formula
        case Atom(name):
            if name in assignment:
                return TRUE if assignment[name] else FALSE
            return formula
        case Not(child):
            return Not(restrict(child, assignment))
        case And(l, r):
            return And(restrict(l, assignment), restrict(r, assignment))
        case Or(l, r):
            return Or(restrict(l, assignment), restrict(r, assignment))
        case Implies(l, r):
            return Implies(restrict(l, assignment), restrict(r, assignment))
    # A step_skeleton has no Next/Eventually/Globally/Until nodes; anything else is left
    # as-is (defensive — restrict must never be handed a temporal formula).
    return formula


_CARRY_PREFIX = "\x00carry"


def step_skeleton(formula: Formula, is_last: bool) -> Formula:
    """Pure-boolean function of the current step's transition, mirroring ``progress``.

    Real atoms keep their current-step reading; each temporal carry becomes a fresh
    opaque token (an ``Atom`` with a reserved ``\\x00carry`` prefix that no rule atom can
    collide with, and that the builder never assigns). The support of this skeleton — the
    real atoms it mentions — is exactly the set of atoms whose *current* value can affect
    the transition, which is what makes cofactor pruning sound.
    """
    counter = [0]

    def token() -> Atom:
        counter[0] += 1
        return Atom(f"{_CARRY_PREFIX}{counter[0]}")

    def now(f: Formula) -> Formula:
        match f:
            case TrueConst() | FalseConst() | Atom():
                return f
            case Not(c):
                return Not(now(c))
            case And(l, r):
                return And(now(l), now(r))
            case Or(l, r):
                return Or(now(l), now(r))
            case Implies(l, r):
                return Implies(now(l), now(r))
            case Next(_):
                # X(phi): defers phi untouched to the next step; FALSE at the last step.
                return FALSE if is_last else token()
            case Eventually(c):
                # F(phi) = phi OR (not_last AND X F(phi))
                return now(c) if is_last else Or(now(c), token())
            case BoundedEventually(c, bound):
                # F≤k(phi) = phi OR (not_last AND X F≤k-1(phi)); F≤0(phi) = phi
                return now(c) if is_last or bound == 0 else Or(now(c), token())
            case Globally(c):
                # G(phi) = phi AND (last OR X G(phi))
                return now(c) if is_last else And(now(c), token())
            case Until(l, r):
                # (l U r) = r OR (l AND not_last AND X(l U r))
                return now(r) if is_last else Or(now(r), And(now(l), token()))
        return f

    return now(formula)


def is_const_true(f: Formula) -> bool:
    """Return whether the formula is a true constant."""
    return isinstance(f, TrueConst)


def is_const_false(f: Formula) -> bool:
    """Return whether the formula is a false constant."""
    return isinstance(f, FalseConst)


def is_propositional(formula: Formula) -> bool:
    """True if ``formula`` has no temporal operator: its value at a step is decided by that
    step's atoms alone."""
    match formula:
        case TrueConst() | FalseConst() | Atom():
            return True
        case Not(child):
            return is_propositional(child)
        case And(left, right) | Or(left, right) | Implies(left, right):
            return is_propositional(left) and is_propositional(right)
    return False


def atoms_of(formula: Formula) -> list[str]:
    """All atom names in the formula, in order of first appearance."""
    seen: dict[str, None] = {}
    visits = 0

    def walk(f: Formula) -> None:
        nonlocal visits
        visits += 1
        match f:
            case Atom(name):
                seen[name] = None
            case (
                Not(child)
                | Next(child)
                | Eventually(child)
                | BoundedEventually(child, _)
                | Globally(child)
            ):
                walk(child)
            case And(l, r) | Or(l, r) | Implies(l, r) | Until(l, r):
                walk(l)
                walk(r)

    walk(formula)
    _charge_walk(visits)
    return list(seen)


def branch_atoms(formula: Formula) -> list[str]:
    """Atoms whose value at the *current* step can affect this step's progression.

    An atom matters *now* unless it sits strictly under a ``Next`` (which defers its
    child to the next step untouched). Everything else — including atoms under
    ``Eventually``/``Globally``/``Until`` — is progressed at the current step, so its
    value is read now. This is the set the DFA's guard decision graph branches on,
    and exactly the set lazy grounding needs to consult per state (in first-appearance
    order, so cheap-first ordering can be layered on top).
    """
    seen: dict[str, None] = {}
    visits = 0

    def walk(f: Formula) -> None:
        nonlocal visits
        visits += 1
        match f:
            case Atom(name):
                seen[name] = None
            case Next(_):
                return
            case Not(child) | Eventually(child) | BoundedEventually(child, _) | Globally(child):
                walk(child)
            case And(l, r) | Or(l, r) | Implies(l, r) | Until(l, r):
                walk(l)
                walk(r)

    walk(formula)
    _charge_walk(visits)
    return list(seen)


def canonical(f: Formula) -> str:
    """Canonical string key for a formula, normalizing structurally equivalent forms
    to the same key: constants folded, commutative And/Or operands sorted, nested
    And/Or of the same type flattened. Used for deterministic hashing and for
    interning DFA states (two residuals with the same key are the same state)."""
    return _norm_repr(simplify(f))


def _norm_repr(f: Formula) -> str:
    budget = _budget_now()
    if budget is not None:
        budget.used += 1
    match f:
        case TrueConst():
            return "T"
        case FalseConst():
            return "F"
        case Atom(name):
            # Quoted, so an atom name can never read as an operator form such as F(a) or T.
            return json.dumps(name)
        case Not(child):
            return f"!{_norm_repr(child)}"
        case And():
            # dedup for idempotence: (x & x) == x, so canonically identical operands
            # collapse — this is what makes residual exploration converge.
            parts = sorted({_norm_repr(p) for p in _flatten_and(f)})
            return parts[0] if len(parts) == 1 else f"({' & '.join(parts)})"
        case Or():
            parts = sorted({_norm_repr(p) for p in _flatten_or(f)})
            return parts[0] if len(parts) == 1 else f"({' | '.join(parts)})"
        case Implies(l, r):
            return f"({_norm_repr(l)} -> {_norm_repr(r)})"
        case Next(child):
            return f"X({_norm_repr(child)})"
        case Eventually(child):
            return f"F({_norm_repr(child)})"
        case BoundedEventually(child, bound):
            return f"F<={bound}({_norm_repr(child)})"
        case Globally(child):
            return f"G({_norm_repr(child)})"
        case Until(l, r):
            return f"({_norm_repr(l)} U {_norm_repr(r)})"
    return repr(f)


def _flatten_and(f: Formula) -> list[Formula]:
    match f:
        case And(l, r):
            return _flatten_and(l) + _flatten_and(r)
        case _:
            return [f]


def _flatten_or(f: Formula) -> list[Formula]:
    match f:
        case Or(l, r):
            return _flatten_or(l) + _flatten_or(r)
        case _:
            return [f]
