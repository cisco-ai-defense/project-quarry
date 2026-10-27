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

"""The core calculus rules are desugared into before lowering.

Every rule becomes a formula over the positions of one conversation timeline. The
constructors below are the whole semantic language. Lowering first chooses an execution
form for semantic joins and value quantifiers, then gives every remaining constructor one
lowering rule (``quarry.engine.lower``). A leaf test (``Pred``) keeps the rule's syntax
subtree, which the atom factory turns into an atom.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Events:
    """The events of a timeline surface, optionally only tool events named ``name`` (an
    accessor such as ``get_tool_calls("x")``) and optionally a signed slice ``window``
    ``(start, stop)`` resolved against the surface size, or only the one event at the signed
    ``index`` (``messages[-1]``)."""

    surface: str
    name: str | None = None
    window: tuple[int | None, int | None] | None = None
    index: int | None = None
    # The rule syntax these events come from (a comprehension or a function call), for
    # attributing the atoms they gate to it; not part of the collection's identity.
    src: object = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class AllEvents:
    """Every timeline event; used only to lift a static condition onto events."""


@dataclass(frozen=True)
class Tools:
    """The offered tool definitions (static)."""


Collection = Events | AllEvents | Tools


@dataclass(frozen=True)
class Const:
    """A constant boolean value."""

    value: bool


TRUE = Const(True)
FALSE = Const(False)


@dataclass(frozen=True)
class Pred:
    """A test read at one event: ``node`` is the rule's syntax subtree, which reads fields of
    the element ``var`` (``None`` for a static test). ``surface`` is ``var``'s surface."""

    node: object
    var: str | None
    surface: str | None
    # How ``node`` names the element when that differs from ``var`` (a leaf moved to another
    # variable standing for the same event); ``None`` when ``node`` names it ``var``.
    name: str | None = None


@dataclass(frozen=True)
class Join:
    """The semantic relation ``test_field contains bound_field`` between two elements.

    Lowering may implement the relation by running once per value or once per candidate
    element, but neither choice is represented here. The surfaces make both field reads
    explicit without consulting the source scopes.
    """

    test_var: str
    test_surface: str
    test_field: object
    bound_var: str
    bound_surface: str
    bound_field: object


@dataclass(frozen=True)
class ValueQuantifier:
    """There exists a captured value ``var`` for which ``body`` holds.

    Candidate values are the matches produced at the assignment leaves for ``var`` in
    ``body``. The quantifier is placed at the rule root, which is the documented scope of an
    explicit capture. A fresh value that no field contains is also in the domain, preserving
    the language's behavior when no assignment matches.
    """

    var: str
    body: Formula


@dataclass(frozen=True)
class Not:
    """A negation of a formula."""

    body: Formula


@dataclass(frozen=True)
class And:
    """A conjunction of formulas."""

    parts: tuple[Formula, ...]


@dataclass(frozen=True)
class Or:
    """A disjunction of formulas."""

    parts: tuple[Formula, ...]


@dataclass(frozen=True)
class Rel:
    """Positions strictly on ``side`` (``"after"`` or ``"before"``) of ``of_var``'s
    position, at a distance of ``lo`` to ``hi`` steps (``hi=None`` is unbounded)."""

    of_var: str
    side: str
    lo: int = 1
    hi: int | None = None


@dataclass(frozen=True)
class Quant:
    """``q`` (``"exists"`` or ``"forall"``) over the events ``var`` of ``coll``, restricted
    to ``anchor`` when given."""

    q: str
    var: str
    coll: Collection
    anchor: Rel | None
    body: Formula


@dataclass(frozen=True)
class Count:
    """The number of events ``var`` of ``coll`` satisfying ``body``, compared ``op n``."""

    var: str
    coll: Events
    body: Formula
    op: str
    n: int


@dataclass(frozen=True)
class Size:
    """A collection size comparison (``len``, ``count``, ``count_tool_calls``), kept as the
    rule's syntax subtree: a static atom compared with the bound as written."""

    node: object


@dataclass(frozen=True)
class Member:
    """``var``'s event belongs to ``coll`` (a gate written out, as on the sustained side of
    ``until``, whose events are the union of its alternatives' collections)."""

    var: str
    coll: Events


@dataclass(frozen=True)
class Same:
    """``var``'s event is ``of_var``'s own event (``of_var`` bound as a candidate)."""

    var: str
    of_var: str


@dataclass(frozen=True)
class AtCandidate:
    """``var``'s event is the candidate's own event."""

    var: str


@dataclass(frozen=True)
class ToolsExists:
    """Some offered tool definition (``q == "exists"``) or every one (``"forall"``), ``var``,
    satisfies ``body`` (tests on ``var`` only, once conversation-wide parts are hoisted)."""

    var: str
    body: Formula
    q: str = "exists"
    # The comprehension this stands for, for attributing its atoms; not part of its identity.
    src: object = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class ChunkQuant:
    """``q`` over the model chunks of ``collection`` (the rule's split call), with ``body``
    tests on the chunk ``var``: one folded model atom."""

    q: str
    var: str
    collection: object
    body: Formula


@dataclass(frozen=True)
class Static:
    """A closed test with its own atom: a system-prompt test, a model or chunked check, or
    ``has_sequence``; ``node`` is the rule's syntax subtree."""

    node: object


@dataclass(frozen=True)
class Closed:
    """A part of a quantifier or tools body that reads no element, or a temporal operand of
    the rule's root beside another, made by hoisting (``quarry.engine.desugar.hoist``): one
    value per run, read as one static atom that its own automaton decides over the run's
    events. Hoisting alone makes it; the unhoisted core never contains one."""

    body: Formula


Formula = (
    Const
    | Pred
    | Join
    | ValueQuantifier
    | Not
    | And
    | Or
    | Quant
    | Count
    | Size
    | Member
    | Same
    | AtCandidate
    | ToolsExists
    | ChunkQuant
    | Static
    | Closed
)


def conj(parts) -> Formula:
    """``And`` with constants folded, nested conjunctions flattened and repeats dropped."""
    flat: list[Formula] = []
    for part in parts:
        if part == FALSE:
            return FALSE
        if part == TRUE:
            continue
        for p in part.parts if isinstance(part, And) else (part,):
            if p not in flat:
                flat.append(p)
    if not flat:
        return TRUE
    return flat[0] if len(flat) == 1 else And(tuple(flat))


def disj(parts) -> Formula:
    """``Or`` with constants folded, nested disjunctions flattened and repeats dropped."""
    flat: list[Formula] = []
    for part in parts:
        if part == TRUE:
            return TRUE
        if part == FALSE:
            continue
        for p in part.parts if isinstance(part, Or) else (part,):
            if p not in flat:
                flat.append(p)
    if not flat:
        return FALSE
    return flat[0] if len(flat) == 1 else Or(tuple(flat))


def quant(q: str, var: str, coll: Collection, anchor: Rel | None, body: Formula) -> Formula:
    """``Quant`` with a constant body folded: no event satisfies ``False``, and every event
    satisfies ``True``."""
    if q == "exists" and body == FALSE:
        return FALSE
    if q == "forall" and body == TRUE:
        return TRUE
    return Quant(q, var, coll, anchor, body)


def neg(body: Formula) -> Formula:
    """Negate a formula."""
    if isinstance(body, Const):
        return Const(not body.value)
    if isinstance(body, Not):
        return body.body
    return Not(body)


def children(formula: Formula) -> tuple[Formula, ...]:
    """The formulas directly inside ``formula``."""
    match formula:
        case Not(body) | Quant(body=body) | Count(body=body) | ToolsExists(body=body):
            return (body,)
        case ChunkQuant(body=body) | Closed(body) | ValueQuantifier(body=body):
            return (body,)
        case And(parts) | Or(parts):
            return tuple(parts)
    return ()


def free_vars(formula: Formula) -> set[str]:
    """Element variables ``formula`` reads without binding them: through a ``Pred``, a
    ``Rel`` anchor or ``Same``."""
    match formula:
        case Const() | Size() | Static() | Closed():
            return set()
        case Join(test_var=test_var, bound_var=bound_var):
            return {test_var, bound_var}
        case ValueQuantifier(body=body):
            return free_vars(body)
        case ToolsExists(var=var, body=body) | ChunkQuant(var=var, body=body):
            return free_vars(body) - {var}
        case Pred(var=var):
            return {var} if var is not None else set()
        case Not(body):
            return free_vars(body)
        case And(parts) | Or(parts):
            return set().union(*(free_vars(p) for p in parts))
        case Quant(var=var, anchor=anchor, body=body):
            inner = free_vars(body) - {var}
            return inner | ({anchor.of_var} if anchor is not None else set())
        case Count(var=var, body=body):
            return free_vars(body) - {var}
        case Member(var=var) | AtCandidate(var=var):
            return {var}
        case Same(var=var, of_var=of_var):
            return {var, of_var}
    raise TypeError(formula)
