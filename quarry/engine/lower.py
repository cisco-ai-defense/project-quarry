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

"""Lower the core calculus to LTLf with past tokens.

``compile_core`` is the compiler: desugar the rule (``quarry.engine.desugar``), choose a
lowering-only execution form for semantic joins and value quantifiers, hoist
conversation-wide parts, lower the result, and build the DFA. Leaf tests are turned into
atoms by the atom factory (``quarry.engine.compiler._Compiler``), exactly as before; this
module decides only structure.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field, fields, replace
from typing import cast

from quarry.engine import builders
from quarry.engine import core as C
from quarry.engine import temporal as T
from quarry.engine.artifacts import dfa_hash
from quarry.engine.atoms import CAPTURE_VALUE, AtomDomain, Cost, make_atom
from quarry.engine.budget import (
    DEFAULT_MAX_WORK,
    DFAConstructionLimit,
    WorkBudget,
    charge,
    spending,
)
from quarry.engine.captures import CaptureBind, CapturePlan, CaptureTest, ElementPlan
from quarry.engine.contracts import (
    COMPILER_VERSION,
    ClosedPlan,
    CompiledRuleArtifact,
    DecisionTree,
    TreeNode,
)
from quarry.engine.decision import Parts, boot_diagram_is_decided, reads
from quarry.engine.desugar import (
    Desugarer,
    ValueTest,
    bind_candidate,
    described,
    hoist,
    source_name,
)
from quarry.engine.diagnostics import Diagnostics, ErrorCategory, SourceRef, UnsupportedConstruct
from quarry.engine.symbolic_dfa import DFA, PAST_PREFIX, PastDef, compile_dfa
from quarry.types.dsl import (
    AllComprehension,
    AnyComprehension,
    Assignment,
    BinaryOp,
    ConditionNode,
    FieldAccess,
    FilterComprehension,
    FlowOp,
    FunctionCall,
    Identifier,
    Literal,
    MapComprehension,
    ModelCall,
    UnaryOp,
    condition_child_fields,
)
from quarry.types.dsl.render import render_condition
from quarry.types.dsl.traversal import walk_condition_nodes


class _Tables:
    """Collection and syntax questions the desugarer asks, answered from the compiler's
    tables and helpers."""

    def __init__(self, factory):
        from quarry.engine import compiler as K

        self._f = factory
        self._k = K

    def value_test(self, node):
        """``node`` as a test on the explicit captured value, or ``None``."""

        return _value_test(node, self._f._capture_var, ValueTest)

    def contains_model_call(self, node) -> bool:
        """Return whether the node contains a model call."""
        return self._k._Compiler._contains_model_call(node)

    def model_calls(self, node):
        """Return the model calls in the node."""
        return self._k._Compiler._model_calls(node)

    def note(self, text: str) -> None:
        """Add a note to the tables."""
        self._f.notes.append(text)

    def is_chunked(self, node) -> bool:
        """``node`` uses a model split (a whole-conversation chunked check)."""

        source = self._k._chunk_source_func()
        return any(
            (isinstance(n, FunctionCall) and n.name in (source, "model_split"))
            or (isinstance(n, ModelCall) and n.operation == "split")
            for n in walk_condition_nodes(node)
        )

    def indexed_read(self, node):
        """``(surface, index)`` when every event field ``node`` reads is one indexed event
        (``messages[-1].content``), else ``None``."""

        found = set()
        for n in walk_condition_nodes(node):
            if (
                isinstance(n, FieldAccess)
                and isinstance(n.base, FunctionCall)
                and n.base.name == "_index"
            ):
                coll, idx = n.base.args
                if not (
                    isinstance(coll, Identifier)
                    and coll.name in self._k._EVENT_SURFACES
                    and isinstance(idx, Literal)
                    and isinstance(idx.value, int)
                ):
                    return None
                found.add((coll.name, idx.value))
        return next(iter(found)) if len(found) == 1 else None

    def drop_index(self, node, name: str):
        """``node`` with each indexed read ``S[i].f`` read as ``name.f``, a field of the
        element ``name`` standing for that event, as a copy (see ``substitute_identifier``)."""
        aliases = self._f._node_aliases

        def rewrite(current):
            if (
                isinstance(current, FieldAccess)
                and isinstance(current.base, FunctionCall)
                and current.base.name == "_index"
            ):
                return aliases.derive(replace(current, base=Identifier(name)), current)
            changes = {}
            for field_name in condition_child_fields(current):
                value = getattr(current, field_name)
                if isinstance(value, ConditionNode):
                    new = rewrite(value)
                elif isinstance(value, list | tuple):
                    items = [rewrite(v) if isinstance(v, ConditionNode) else v for v in value]
                    changed = any(a is not b for a, b in zip(items, value, strict=True))
                    new = (items if isinstance(value, list) else tuple(items)) if changed else value
                else:
                    continue
                if new is not value:
                    changes[field_name] = new
            if not changes:
                return current
            return aliases.derive(replace(current, **changes), current)

        return rewrite(node)

    def lit(self, node) -> str:
        """Return the literal of the node."""
        return self._f._lit(node)

    def scalar_alias(self, node):
        """Return the scalar alias of the node."""
        return self._f._scalar_alias(node)

    def substitute_identifier(self, node, var, expr):
        """``node`` with every ``Identifier(var)`` replaced by ``expr``, as a copy: the rule's
        own syntax is left as written, and each rebuilt node is recorded as standing for the
        authored node it copies."""
        aliases = self._f._node_aliases

        def substitute(current):
            if isinstance(current, Identifier) and current.name == var:
                return expr
            changes = {}
            for field_name in condition_child_fields(current):
                value = getattr(current, field_name)
                if isinstance(value, ConditionNode):
                    new = substitute(value)
                elif isinstance(value, list | tuple):
                    items = [substitute(v) if isinstance(v, ConditionNode) else v for v in value]
                    changed = any(a is not b for a, b in zip(items, value, strict=True))
                    new = (items if isinstance(value, list) else tuple(items)) if changed else value
                else:
                    continue
                if new is not value:
                    changes[field_name] = new
            if not changes:
                return current
            return aliases.derive(replace(current, **changes), current)

        return substitute(node)

    def name_alias(self, comp):
        """For ``any x in call_sequence() where …``, the body with a bare ``x`` read as
        ``x.name`` (each element is a tool call's name); other bodies as written."""
        coll = comp.collection
        if not (isinstance(coll, FunctionCall) and coll.name == "call_sequence"):
            return comp.condition
        name = FieldAccess(base=Identifier(comp.variable), fields=["name"])
        return self.substitute_identifier(comp.condition, comp.variable, name)

    def is_tools(self, coll) -> bool:
        """Return whether the collection is the tools collection."""
        return (
            isinstance(coll, Identifier)
            and coll.name in self._k._STATIC_SURFACES
            and (coll.name == "tools")
        )

    def is_special_collection(self, comp) -> bool:
        """Return whether the collection is a special collection."""
        coll = comp.collection
        if isinstance(coll, ModelCall):
            return True
        return isinstance(coll, FunctionCall) and coll.name in (
            "model_split",
            self._k._chunk_source_func(),
        )

    def resolve_collection(self, coll):
        """``(surface, name, window, name_alias)`` of a comprehension collection."""
        k = self._k
        if isinstance(coll, FunctionCall) and coll.name == "_slice" and len(coll.args) == 3:
            base, start, stop = coll.args
            if not (isinstance(base, Identifier) and base.name in k._EVENT_SURFACES):
                raise UnsupportedConstruct(
                    f"slice is only supported directly on an event surface, not {base!r}"
                )
            lo = start.value if isinstance(start, Literal) else None
            hi = stop.value if isinstance(stop, Literal) else None
            return base.name, None, None if lo is None and hi is None else (lo, hi), False
        if isinstance(coll, Identifier) and coll.name in k._EVENT_SURFACES:
            return coll.name, None, None, False
        if isinstance(coll, FunctionCall):
            if coll.name in k._EVENT_SURFACES:
                return coll.name, None, None, False
            if coll.name in k._ACCESSORS:
                surface, filter_field = k._ACCESSORS[coll.name]
                name = self.lit(coll.args[0]) if filter_field is not None and coll.args else None
                return surface, name, None, False
            if coll.name == "call_sequence":
                return "tool_calls", None, None, True
        if isinstance(coll, FunctionCall) and coll.name == "join":
            raise UnsupportedConstruct(k._JOIN_FORM)
        if isinstance(coll, MapComprehension):
            raise UnsupportedConstruct(k._MAP_COMPREHENSION_FORM)
        raise UnsupportedConstruct(f"comprehension over non-surface collection {coll!r}")

    def is_size(self, node) -> bool:
        """Return whether the node is a size comparison."""
        k = self._k
        if not (hasattr(node, "operator") and node.operator in k._COMPARISONS):
            return False
        fc, number, _ = k._Compiler._split_scalar(node.left, node.right)
        if fc is None:
            return False
        if fc.name == "count_tool_calls":
            return bool(fc.args)
        if fc.name not in ("count", "len") or len(fc.args) != 1:
            return False
        arg = fc.args[0]
        if isinstance(arg, FilterComprehension):
            return False
        if isinstance(arg, FunctionCall) and arg.name == "_slice":
            arg = arg.args[0]
        if isinstance(arg, Identifier):
            return arg.name in k._SURFACES
        return isinstance(arg, FunctionCall) and arg.name in k._ACCESSORS

    def filter_count_shape(self, node):
        """``(comprehension, op, n)`` for ``count([x for x in S if φ]) op n`` over an event
        surface, or ``None`` for any other node."""
        k = self._k
        if not (hasattr(node, "operator") and node.operator in k._COMPARISONS):
            return None
        fc, number, flipped = k._Compiler._split_scalar(node.left, node.right)
        if fc is None or number is None or fc.name not in k._CHUNK_COUNT_FUNCS:
            return None
        if len(fc.args) != 1 or not isinstance(fc.args[0], FilterComprehension):
            return None
        comp = fc.args[0]
        if (
            isinstance(comp.collection, FunctionCall)
            and comp.collection.name == k._chunk_source_func()
        ):
            return None
        op = k._FLIP.get(node.operator, node.operator) if flipped else node.operator
        if number != int(number):
            raise UnsupportedConstruct(
                f"count() compares against an event count, so {number} is not a meaningful "
                "bound — use an integer"
            )
        n = int(number)
        if n < 0:
            raise UnsupportedConstruct(f"count() bound must be >= 0, got {n}")
        if isinstance(comp.collection, Identifier) and comp.collection.name in k._STATIC_SURFACES:
            raise UnsupportedConstruct(
                f"count() over the static surface {comp.collection.name!r} is not supported: a "
                "static surface has no timeline positions for the temporal counter to range over"
            )
        if self._f._find_flow(comp.condition) is not None:
            raise UnsupportedConstruct(
                "a temporal flow inside count([...]) is not supported: the filter must be a "
                "predicate on ONE element, but a flow relates two timeline positions"
            )
        if self._f._capture_var and self._f._references_var(comp.condition, self._f._capture_var):
            raise UnsupportedConstruct(
                "a value capture inside count([...]) is not supported: the counter needs one "
                "fixed element predicate, but a capture re-binds it per candidate value"
            )
        return comp, op, n


@dataclass
class _Past:
    """Past definitions minted while lowering, one token per distinct definition."""

    defs: list[PastDef] = field(default_factory=list)
    by_key: dict[tuple, str] = field(default_factory=dict)

    def token(self, op: str, operand: T.Formula, bound: int | None = None) -> T.Formula:
        """Return the token for the operation."""
        operand = T.simplify(operand)
        if T.is_const_false(operand):
            return T.FALSE
        key = (op, T.canonical(operand), bound)
        token = self.by_key.get(key)
        if token is None:
            token = f"{PAST_PREFIX}{len(self.defs)}"
            self.by_key[key] = token
            self.defs.append(PastDef(token, op, operand, bound))
        return T.Atom(token)


class NeedsBinding(Exception):
    """Lowering needs ``var`` decided one candidate event per run."""

    def __init__(self, var: str):
        super().__init__(var)
        self.var = var


@dataclass(frozen=True)
class _JoinSide:
    """A lowering-only implementation of a semantic :class:`core.Join`.

    ``bind``/``ref`` implement a per-value run. ``pinned``/``within`` implement a run with
    one of the join's elements fixed to a candidate event. These roles never occur in the
    semantic core produced by :class:`desugar.Desugarer`.
    """

    role: str
    field: object
    bound_field: object
    bound_var: str


_PINNED_ROLES = ("pinned", "within")


@dataclass
class _Bound:
    """The candidate element of this lowering: its atom and plan fields, the conjuncts that
    test its own event alone (``bind_candidate``), and each test on it as lowered."""

    var: str
    coll: C.Events
    at: T.Formula
    at_atom_id: str
    filter_atom_id: str = ""
    filter_name: str = ""
    ref_ids: set[str] = field(default_factory=set)
    own_step: tuple[C.Formula, ...] = ()
    preds: list[tuple[C.Pred, T.Formula]] = field(default_factory=list)

    def own_step_formula(self, specs: dict) -> T.Formula:
        """The own-step conjuncts a candidate's event can be tested on before its run: those
        that read, as lowered, only unparameterized event atoms of its surface below model
        cost (a cheaper test that decides the run leaves a model call unmade)."""

        def lowered(f: C.Formula) -> T.Formula | None:
            match f:
                case C.Const(value):
                    return T.TRUE if value else T.FALSE
                case C.Pred():
                    return next((t for p, t in self.preds if p == f), None)
                case C.Not(body):
                    inner = lowered(body)
                    return None if inner is None else T.Not(inner)
                case C.And(parts) | C.Or(parts):
                    children = [lowered(p) for p in parts]
                    if any(c is None for c in children):
                        return None
                    join, unit = (T.And, T.TRUE) if isinstance(f, C.And) else (T.Or, T.FALSE)
                    return _fold(join, [c for c in children if c is not None], unit)
            return None

        def testable(t: T.Formula) -> bool:
            return T.is_propositional(t) and all(
                a in specs
                and specs[a].domain is AtomDomain.EVENT
                and specs[a].selector == self.coll.surface
                and not specs[a].parameter_names
                and specs[a].cost.rank < Cost.MODEL.rank
                for a in T.atoms_of(t)
            )

        parts = [t for t in map(lowered, self.own_step) if t is not None and testable(t)]
        return T.simplify(_fold(T.And, parts, T.TRUE))


class _Lowerer:
    def __init__(self, factory):
        self._f = factory
        self._past = _Past()
        self.bound: _Bound | None = None
        # Each closed part's atom: the formula its automaton decides, with its past tokens.
        self.closed: dict[str, tuple[T.Formula, tuple[PastDef, ...]]] = {}
        # Each closed body lowered so far, by object: expansion repeats one body in every
        # branch, and it is lowered once.
        self._closed_atoms: dict[int, tuple[C.Formula, T.Formula]] = {}

    def root(self, formula: C.Formula) -> T.Formula:
        """Return the root formula."""
        return self._statics_last(self.lower(formula, None), self._past.defs)

    def lower(self, f: C.Formula, at: str | None) -> T.Formula:
        """Lower the formula."""
        charge(1)
        match f:
            case C.Const(value):
                return T.TRUE if value else T.FALSE
            case C.Not(body):
                return T.Not(self.lower(body, at))
            case C.And(parts):
                return _fold(T.And, [self.lower(p, at) for p in parts], T.TRUE)
            case C.Or(parts):
                return _fold(T.Or, [self.lower(p, at) for p in parts], T.FALSE)
            case C.Pred(node):
                with self._f.activate_source(_source_of(node)):
                    lowered = self.pred(f, at)
                if self.bound is not None and at == self.bound.var:
                    self.bound.preds.append((f, lowered))
                return lowered
            case C.Member(var, coll):
                self._require_current(var, at)
                with self._f.activate_source(coll.src):
                    return self.gate(coll)
            case C.Quant(coll=coll) | C.Count(coll=coll):
                # The comprehension (or call) the quantifier comes from owns its gate, filter
                # and body atoms, as when lowering walked the rule's syntax.
                with self._f.activate_source(getattr(coll, "src", None)):
                    return self.quant(f, at) if isinstance(f, C.Quant) else self.count(f, at)
            case C.Size(node):
                with self._f.activate_source(node):
                    return self._f._count_collection_macro(node)
            case C.ToolsExists(var, body, q, src):
                self._f.has_static = True
                with self._f.activate_source(src):
                    source_start = len(self._f._atom_source_events)
                    atom = self._f._tool_quantifier_atom(
                        self.lower(body, var),
                        kind="any" if q == "exists" else "all",
                        source_start=source_start,
                    )
                    return atom
            case C.ChunkQuant(q, var, collection, body):
                syntax = cast(ConditionNode, collection)
                comp = (AnyComprehension if q == "exists" else AllComprehension)(
                    var, syntax, _to_ast(body)
                )
                with self._f.activate_source(collection):
                    return self._f.compile_closed(comp)
            case C.Static(node):
                self._require_root(at, "a conversation-wide condition")
                with self._f.activate_source(node):
                    return self._f.compile_closed(node)
            case C.Same(of_var=of_var):
                raise NeedsBinding(of_var)
            case C.AtCandidate(var):
                self._require_current(var, at)
                assert self.bound is not None
                return self.bound.at
            case C.Closed(body):
                return self.closed_atom(body)
        raise TypeError(f)

    def closed_atom(self, body: C.Formula) -> T.Formula:
        """A part that reads no element: ``body`` lowered as a rule of its own (its own past
        tokens, the same candidate), which its own automaton decides over the run's events,
        read as one static atom. Its identity is that formula and its past definitions."""
        known = self._closed_atoms.get(id(body))
        if known is not None and known[0] is body:
            return known[1]
        atom = self._lower_closed(body)
        self._closed_atoms[id(body)] = (body, atom)
        return atom

    def _lower_closed(self, body: C.Formula) -> T.Formula:
        outer, self._past = self._past, _Past()
        try:
            with self._f._in_window("", None):
                formula = self._statics_last(self.lower(body, None), self._past.defs)
        finally:
            past, self._past = self._past, outer
        defs = tuple(past.defs)
        by_id = {s.atom_id: s for s in self._f.atoms.specs}
        reads = set(T.atoms_of(formula)).union(*(T.atoms_of(d.operand) for d in defs))
        cost = max(
            (by_id[a].cost for a in reads if a in by_id), key=lambda c: c.rank, default=Cost.PURE
        )
        identity = T.canonical(formula) + "".join(
            f"|{d.token}={d.op}{d.bound}:{T.canonical(d.operand)}" for d in defs
        )
        spec = self._f._intern_atom(
            make_atom(
                domain=AtomDomain.STATIC,
                selector="",
                value_expr_key=f"closed:{identity}",
                predicate="closed",
                patterns=(),
                cost=cost,
            )
        )
        self._f.has_static = True
        self.closed[spec.atom_id] = (formula, defs)
        return T.Atom(spec.atom_id)

    def pred(self, f: C.Pred, at: str | None) -> T.Formula:
        """Lower the predicate."""
        from quarry.engine.compiler import Scope

        var, surface = f.var, f.surface
        if isinstance(f.node, ValueTest):
            return value_test_atom(self._f, f.node)
        assert var is not None and surface is not None, "only a value test is static"
        self._require_current(var, at)
        node = f.node
        if isinstance(node, _JoinSide) and node.role in _PINNED_ROLES:
            return self.pinned_join(node, surface)
        if isinstance(node, _JoinSide):
            path = self._f._field_path(node.field)
            if node.role == "bind":
                return self._f._mint_capture(surface, path, "capture_bind", predicate="equals")
            return self._f._mint_capture(surface, path, "capture_ref")
        if isinstance(node, FunctionCall) and node.name in ("user_said", "assistant_said"):
            return self._f._said_atom(node.name, node, surface)
        if isinstance(node, FunctionCall) and node.name in ("called_tool", "called_tools"):
            return self._f._mint_atom("tool_calls", "name", "equals", (self._f._lit(node.args[0]),))
        name = f.name or source_name(var)
        return self._f.compile_step(node, surface, Scope().bind(name, surface))

    def pinned_join(self, side: _JoinSide, surface: str) -> T.Formula:
        """A join to the candidate element, parameterized by the candidate's field:
        ``field contains candidate.bound_field`` (``pinned``) or ``candidate.bound_field
        contains field`` (``within``)."""

        # A tools comprehension is answered once over the static tool definitions.  Even
        # after the enclosing event is selected, that per-candidate value is unavailable
        # inside the static question.  Keep this as ``NeedsBinding`` so the retry reports
        # the same precise "inside a comprehension over tools" diagnostic as other
        # candidate values that cannot reach their use site.
        if surface == "tools" or self.bound is None or self.bound.var != side.bound_var:
            raise NeedsBinding(str(side.bound_var))
        spec = self._f.atoms.intern(
            make_atom(
                domain=AtomDomain.EVENT,
                selector=surface,
                value_expr_key=f"field:{self._f._field_path(side.field)}",
                predicate="contains_any" if side.role == "pinned" else "contained_in",
                patterns=(),
                parameter_names=(f"{side.bound_var}.{self._f._field_path(side.bound_field)}",),
            )
        )
        self.bound.ref_ids.add(spec.atom_id)
        self._f.temporal = True
        return T.Atom(spec.atom_id)

    def gate(self, coll: C.Events | C.AllEvents) -> T.Formula:
        """Return the gate for the collection."""
        if isinstance(coll, C.AllEvents):
            return T.Or(
                self._f._on_surface_atom("messages"), self._f._on_surface_atom("tool_calls")
            )
        with self._f._in_window(coll.surface, coll.window, coll.index):
            if coll.name is None:
                return self._f._on_surface_atom(coll.surface)
            return self._f._mint_atom(coll.surface, "name", "equals", (coll.name,))

    def gated(self, coll: C.Events | C.AllEvents, body: T.Formula) -> T.Formula:
        """``gate ∧ body``, without the gate when ``body`` can only hold at an event of the
        collection: every atom it reads is an event atom on that surface and it is false
        where they all are (the gate would not change the formula's meaning)."""
        if isinstance(coll, C.AllEvents):
            if self._false_at_boot(body):
                return body
        elif (
            coll.window is None
            and coll.index is None
            and coll.name is None
            and self._implies_gate(body, coll.surface)
        ):
            return body
        return T.And(self.gate(coll), body)

    def _statics_last(self, f: T.Formula, past: list[PastDef]) -> T.Formula:
        """``f``, an automaton's formula, with each condition of its root ``and``/``or``/``not``
        that reads static atoms only (a model or chunked check, a tools test, a closed part)
        read once at the final step (``F(last ∧ s)``), after every event test the automaton
        makes: a cheaper test that already decides it leaves the condition, and its model call,
        unread. A static condition has one value at every step, so this is exact. An automaton
        that makes no event test reads its conditions at the boot step, where nothing cheaper
        comes first."""
        by_id = {s.atom_id: s for s in self._f.atoms.specs}

        def static(g: T.Formula) -> bool:
            atoms = T.atoms_of(g)
            return bool(atoms) and all(
                a in by_id and by_id[a].domain is AtomDomain.STATIC for a in atoms
            )

        reads = [T.Atom(a) for a in T.atoms_of(f)]
        reads += [T.Atom(a) for d in past for a in T.atoms_of(d.operand)]
        if all(static(r) for r in reads):
            return f

        def last(g: T.Formula) -> T.Formula:
            if T.is_propositional(g) and static(g):
                return T.Eventually(T.And(_weak_next(T.FALSE), g))
            match g:
                case T.Not(child):
                    return T.Not(last(child))
                case T.And(left, right):
                    return T.And(last(left), last(right))
                case T.Or(left, right):
                    return T.Or(last(left), last(right))
            return g

        return last(f)

    def _false_at_boot(self, body: T.Formula) -> bool:
        """``body`` is false at the boot step, the one step that is not an event: it reads
        only event atoms (false there) and is false when they all are."""
        if not T.is_propositional(body):
            return False
        by_id = {s.atom_id: s for s in self._f.atoms.specs}
        atoms = T.atoms_of(body)
        if any(a not in by_id or by_id[a].domain is not AtomDomain.EVENT for a in atoms):
            return False
        return T.is_const_false(T.simplify(T.restrict(body, dict.fromkeys(atoms, False))))

    def _implies_gate(self, body: T.Formula, surface: str) -> bool:
        """``body`` is false at every event off ``surface``: a propositional part whose atoms
        are all event atoms on it and that is false where they all are, conjoined with
        anything, or such parts on every side of a disjunction."""
        if T.is_propositional(body):
            return self._f._body_pinned_to_surface(
                body, surface
            ) and self._f._body_false_off_surface(body)
        match body:
            case T.And(left, right):
                return self._implies_gate(left, surface) or self._implies_gate(right, surface)
            case T.Or(left, right):
                return self._implies_gate(left, surface) and self._implies_gate(right, surface)
        return False

    def quant(self, f: C.Quant, at: str | None) -> T.Formula:
        """Return the quantifier for the formula."""
        if isinstance(f.coll, C.Tools):
            raise TypeError("tools quantifiers are ToolsExists")
        self._f.temporal = True
        events = f.coll if isinstance(f.coll, C.Events) else None
        # An element fixed by index (`messages[-1]`): every atom read on it holds only at
        # that event, so the atoms carry its ordinal.
        with (
            self._f._in_window(events.surface, None, events.index)
            if events is not None and events.index is not None
            else nullcontext()
        ):
            body = self.lower(f.body, f.var)
        if f.q == "exists":
            inner = self.gated(f.coll, body)
        else:
            gate = self.gated(f.coll, T.Not(body)) if isinstance(f.coll, C.AllEvents) else None
            inner = T.Not(gate) if gate is not None else T.Or(T.Not(self.gate(f.coll)), body)
        if f.anchor is None:
            if at is None:
                return T.Eventually(inner) if f.q == "exists" else T.Globally(inner)
            raise UnsupportedConstruct(
                "internal: a closed quantifier reached lowering inside another element's "
                "condition (hoisting moves it out)"
            )
        self._require_current(f.anchor.of_var, at)
        lo, hi = f.anchor.lo, f.anchor.hi
        width = None if hi is None else hi - lo
        if f.anchor.side == "after":
            if f.q == "exists":
                reach = T.Eventually(inner) if width is None else T.BoundedEventually(inner, width)
            else:
                reach = T.Globally(inner) if width is None else _bounded_globally(inner, width)
            for _ in range(lo):
                reach = T.Next(reach) if f.q == "exists" else _weak_next(reach)
            return reach
        # Before the anchor: a past token over a propositional body.
        if not T.is_propositional(inner):
            raise NeedsBinding(f.anchor.of_var)
        if f.q == "exists":
            reach = self._past.token("O", inner, width)
        else:
            reach = T.Not(self._past.token("O", T.Not(inner), width))
        for _ in range(lo):
            reach = (
                self._past.token("Y", reach)
                if f.q == "exists"
                else T.Not(self._past.token("Y", T.Not(reach)))
            )
        return reach

    def bind(self, var: str, coll: C.Events, own_step: tuple[C.Formula, ...]) -> None:
        """Decide ``var`` one candidate event of ``coll`` per run; ``own_step`` are the
        conjuncts that test its event alone."""

        spec = self._f.atoms.intern(
            make_atom(
                domain=AtomDomain.EVENT,
                selector=coll.surface,
                value_expr_key="",
                predicate="at_candidate",
                patterns=(),
                parameter_names=(var,),
            )
        )
        at = T.Atom(spec.atom_id)
        filter_atom = ""
        if coll.name is not None:
            name = self._f._mint_atom(coll.surface, "name", "equals", (coll.name,))
            filter_atom = name.name if isinstance(name, T.Atom) else ""
        filter_name = coll.name if filter_atom and coll.name is not None else ""
        self.bound = _Bound(
            var, coll, at, spec.atom_id, filter_atom, filter_name, own_step=own_step
        )

    def count(self, f: C.Count, at: str | None) -> T.Formula:
        """Return the count for the formula."""
        self._require_root(at, "a filtered count")
        self._f.temporal = True
        body = self.gated(f.coll, self.lower(f.body, f.var))
        at_least_n = builders.at_least(body, f.n)
        at_least_n1 = builders.at_least(body, f.n + 1)
        if f.op == ">=":
            return at_least_n
        if f.op == ">":
            return at_least_n1
        if f.op == "<":
            return T.Not(at_least_n)
        if f.op == "<=":
            return T.Not(at_least_n1)
        exactly = T.And(at_least_n, T.Not(at_least_n1))
        return exactly if f.op == "=" else T.Not(exactly)

    @staticmethod
    def _require_current(var: str, at: str | None) -> None:
        if var != at:
            raise UnsupportedConstruct(
                f"internal: a condition on `{var}` reached lowering at `{at}`"
            )

    @staticmethod
    def _require_root(at: str | None, what: str) -> None:
        if at is not None:
            raise UnsupportedConstruct(f"internal: {what} reached lowering inside a comprehension")


def _explicit_capture_var(factory, condition) -> str | None:
    """The variable of an explicit ``<pattern> in <field> as v`` capture that is referenced
    elsewhere (a field test on it, or a test on the value itself); an unreferenced bind is
    the plain match."""

    assign = factory._find_assignment(condition)
    if assign is None or factory._scalar_alias(assign) is not None:
        return None
    var = str(getattr(assign, "variable", "") or "")
    if not var:
        return None
    referenced = factory._find_reference(condition, var) is not None or any(
        _value_test(n, var, lambda *parts: parts) is not None
        for n in walk_condition_nodes(condition)
    )
    return var if referenced else None


def _value_test(node, var: str, make):
    """``make(predicate, pattern)`` when ``node`` tests the captured value ``var`` itself."""

    if not var or var == "__join__" or not isinstance(node, BinaryOp):
        return None

    def is_var(n) -> bool:
        return isinstance(n, Identifier) and n.name == var

    op, left, right = node.operator, node.left, node.right
    if op == "in" and is_var(right) and not is_var(left):
        return make("contains_any", left)
    if op == "contains" and is_var(left) and not is_var(right):
        return make("contains_any", right)
    if op in ("~", "matches") and is_var(left):
        return make("regex", right)
    if op == "=" and (is_var(left) or is_var(right)):
        return make("equals", right if is_var(left) else left)
    return None


def value_test_atom(factory, test) -> T.Formula:
    """The static atom of a test on the captured value: bound per run like a reference."""

    patterns, sem = factory._resolve_patterns(test.pattern, force_regex=test.predicate == "regex")
    spec = factory.atoms.intern(
        make_atom(
            domain=AtomDomain.STATIC,
            selector=CAPTURE_VALUE,
            value_expr_key="field:",
            predicate=test.predicate,
            patterns=tuple(patterns),
            parameter_names=(factory._capture_var,),
            string_semantics=sem,
        )
    )
    factory._capture_ref_ids.add(spec.atom_id)
    factory._capture_ref_id = factory._capture_ref_id or spec.atom_id
    factory.has_static = True
    return T.Atom(spec.atom_id)


def _capture_plan(factory, formula: C.Formula):
    """The value-capture plan read off lowering's execution formula."""

    var = factory._capture_var
    binds, refs = [], []
    for pred in _preds(formula):
        node = pred.node
        if isinstance(node, _JoinSide):
            (binds if node.role == "bind" else refs).append((pred, node.field))
        elif isinstance(node, Assignment) and str(getattr(node, "variable", "")) == var:
            binds.append((pred, node.expression))
        elif isinstance(node, ValueTest):
            refs.append((pred, None))
        elif _references(node, var):
            refs.append((pred, _reference_field(node, var)))
    if not binds or not refs:
        raise UnsupportedConstruct("a captured value must be bound and referenced")
    bind_pred, bind_node = binds[0]
    from quarry.engine.compiler import _reject_static_bind

    _reject_static_bind(bind_pred.surface)
    ref_pred, ref_field = next((r for r in refs if r[1] is not None), (bind_pred, None))
    coll = _collection_of(formula, bind_pred.var)
    start, stop = coll.window if coll is not None and coll.window is not None else (None, None)
    name = coll.name if coll is not None and coll.name is not None else ""
    if var == "__join__":
        bind = CaptureBind(
            bind_pred.surface,
            factory._field_path(bind_node),
            (),
            is_regex=False,
            window_start=start,
            window_stop=stop,
            name=name,
        )
        test = CaptureTest(ref_pred.surface, factory._field_path(ref_field))
    else:
        patterns, sem = factory._resolve_patterns(bind_node.left)
        bind = CaptureBind(
            bind_pred.surface,
            factory._field_path(bind_node.right),
            tuple(patterns),
            is_regex=sem.is_regex,
            case_insensitive=sem.case_insensitive,
            fullword=sem.fullword,
            base64=sem.base64,
            window_start=start,
            window_stop=stop,
            name=name,
        )
        test = CaptureTest(
            ref_pred.surface or bind_pred.surface,
            factory._field_path(ref_field if ref_field is not None else bind_node.right),
            case_insensitive=sem.case_insensitive,
        )
    return CapturePlan(
        var=var,
        bind=bind,
        test=test,
        bind_atom_id=factory._capture_bind_id,
        ref_atom_id=factory._capture_ref_id,
        bind_atom_ids=frozenset(factory._capture_bind_ids),
        ref_atom_ids=frozenset(factory._capture_ref_ids),
    )


def _join_values(f: C.Formula) -> dict[tuple[str, str], tuple]:
    """Each captured value a join reads, with its candidate domain: the bound field and the
    bound element's collection (surface, window, accessor name)."""
    found: dict[tuple[str, str], tuple] = {}
    for join in _joins(f):
        parts = _join_field_parts(join.bound_field)
        path = ".".join(str(part) for part in parts)
        value = (join.bound_var, path)
        coll = _collection_of(f, join.bound_var)
        window = coll.window if coll is not None else None
        name = coll.name if coll is not None else None
        surface = coll.surface if coll is not None else None
        found.setdefault(value, (surface, parts, window, name))
    return found


def _one_value_reason(f: C.Formula) -> str | None:
    """Why running the rule once per captured value would not decide its joins exactly, or
    ``None``. One value serves the rule when the joins read values over one candidate domain,
    no ``and`` needs two different values at once (alternatives under ``or`` may share one),
    and every value is read on a positive path (``_values_positive``)."""
    values = _join_values(f)
    if len(set(values.values())) > 1 or _conjoins_two_values(f):
        fields = ", ".join(_field_of(var, path, f) for var, path in sorted(values))
        return f"the joins in this rule bind different values ({fields})"
    if not _values_positive(f):
        return (
            "a join under `not`, inside `all`, in a count or on the sustained side of `until` "
            "would need its captured value chosen per element, not once for the rule"
        )
    return None


def _join_element(f: C.Formula, reason: str) -> str:
    """The element decided one candidate at a time when one captured value cannot decide the
    rule's joins (``reason`` says why): the outermost element that takes part in every join,
    on either side, and whose ``any`` is reached through ``and``, ``or`` and ``any`` only,
    where running per candidate and OR-ing the runs is exact. Each join is then a test at its
    other element's event, parameterized by the candidate's field."""
    joins = [{join.test_var, join.bound_var} for join in _joins(f)]
    shared = set.intersection(*joins)
    for var in _positive_binders(f):
        if var in shared:
            return var
    if not shared:
        raise UnsupportedConstruct(
            f"{reason}, and no one element takes part in every join; a rule is decided one "
            "captured value or one element at a time, so it cannot need two at once"
        )
    names = ", ".join(described(var) for var in sorted(shared))
    raise UnsupportedConstruct(
        f"{reason}, and the elements that take part in every join ({names}) are not reached "
        "through `and`, `or` and `any` only, where one could be decided one candidate at a time"
    )


def _positive_binders(f: C.Formula) -> list[str]:
    """The elements whose ``any`` over events is reached through ``and``, ``or`` and ``any``
    only (``bind_candidate`` can decide them per candidate), outermost first."""

    def walk(g: C.Formula, depth: int) -> list[tuple[int, str]]:
        match g:
            case C.And(parts) | C.Or(parts):
                return [b for p in parts for b in walk(p, depth)]
            case C.Quant("exists", var, coll, _, body):
                own = [(depth, var)] if isinstance(coll, C.Events) else []
                return own + walk(body, depth + 1)
        return []

    return [var for _, var in sorted(walk(f, 0), key=lambda b: b[0])]


def _values_positive(f: C.Formula) -> bool:
    """Every join side that reads a value is reached through ``and``, ``or`` and ``any``
    only. Under ``not``, ``all`` or a count the value would be chosen per element, not once
    for the rule."""

    def ok(g: C.Formula, positive: bool) -> bool:
        charge(1)
        if isinstance(g, C.Join):
            return positive
        existential = isinstance(g, (C.And, C.Or)) or (
            isinstance(g, (C.Quant, C.ToolsExists)) and g.q == "exists"
        )
        return all(ok(child, positive and existential) for child in C.children(g))

    return ok(f, True)


def _conjoins_two_values(f: C.Formula) -> bool:
    """Whether two parts of some ``and`` read different values: the rule would need both at
    once. Values that meet only under ``or`` do not count."""
    conflict = False

    def values_in(g: C.Formula) -> set:
        nonlocal conflict
        charge(1)
        parts = [values_in(child) for child in C.children(g)]
        found = {_join_value(g)} if isinstance(g, C.Join) else set()
        found = found.union(*parts)
        if isinstance(g, C.And) and sum(1 for p in parts if p) > 1 and len(found) > 1:
            conflict = True
        return found

    values_in(f)
    return conflict


def _join_value(join: C.Join) -> tuple[str, str]:
    return join.bound_var, ".".join(str(part) for part in _join_field_parts(join.bound_field))


def _join_field_parts(field: object) -> tuple:
    if not isinstance(field, FieldAccess):
        raise UnsupportedConstruct("internal: a join field is not a field access")
    return tuple(field.fields)


def _joins(f: C.Formula):
    charge(1)
    if isinstance(f, C.Join):
        yield f
    for child in C.children(f):
        yield from _joins(child)


def _collection_of(f: C.Formula, var: str) -> C.Events | None:
    """The collection of the quantifier that binds ``var``."""
    match f:
        case C.Quant(var=v, coll=C.Events() as coll) if v == var:
            return coll
        case C.Quant(body=body) | C.Count(body=body) | C.Not(body):
            return _collection_of(body, var)
        case C.And(parts) | C.Or(parts):
            for p in parts:
                found = _collection_of(p, var)
                if found is not None:
                    return found
    return None


def _preds(f: C.Formula):
    match f:
        case C.Pred():
            yield f
        case C.Not(body):
            yield from _preds(body)
        case C.And(parts) | C.Or(parts):
            for p in parts:
                yield from _preds(p)
        case C.Quant(body=body) | C.Count(body=body) | C.ToolsExists(body=body):
            yield from _preds(body)


def _references(node, var: str) -> bool:
    return _reference_field(node, var) is not None


def _reference_field(node, var: str):
    """The field a ``var in field`` / ``field contains var`` reference tests, if any."""

    if isinstance(node, BinaryOp):
        if node.operator == "in" and isinstance(node.left, Identifier) and node.left.name == var:
            return node.right
        if (
            node.operator == "contains"
            and isinstance(node.right, Identifier)
            and node.right.name == var
        ):
            return node.left
        for child in (node.left, node.right):
            found = _reference_field(child, var)
            if found is not None:
                return found
    operand = getattr(node, "operand", None)
    return _reference_field(operand, var) if operand is not None else None


def _to_ast(f: C.Formula):
    """A chunk body back as syntax, for the chunked-check lowering."""

    match f:
        case C.Const(value):
            return Literal(value)
        case C.Pred(node=node):
            return node
        case C.Not(body):
            return UnaryOp("not", _to_ast(body))
        case C.And(parts) | C.Or(parts):
            op = "and" if isinstance(f, C.And) else "or"
            out = _to_ast(parts[0])
            for part in parts[1:]:
                out = BinaryOp(op, out, _to_ast(part))
            return out
    raise UnsupportedConstruct("a chunked body can only test the chunk")


def _fold(op, parts: list[T.Formula], unit: T.Formula) -> T.Formula:
    if not parts:
        return unit
    out = parts[0]
    for part in parts[1:]:
        out = op(out, part)
    return out


def _weak_next(f: T.Formula) -> T.Formula:
    """``X_w f``: ``f`` at the next step, or true at the last step."""
    return T.Not(T.Next(T.Not(f)))


def _bounded_globally(f: T.Formula, bound: int) -> T.Formula:
    return T.Not(T.BoundedEventually(T.Not(f), bound))


def _desugar(condition, c):
    """The rule's strategy-neutral semantic core and the factory that will lower it."""
    explicit = _explicit_capture_var(c, condition)
    if explicit is not None:
        c._reject_capture_outside_positive_existentials(condition, explicit)
        c._capture_var = explicit
    return c, Desugarer(_Tables(c), value_var=explicit).condition(condition)


def _execution_core(c, core: C.Formula) -> C.Formula:
    """Translate semantic joins and a value quantifier into lowering-only execution forms.

    The returned formula is private to lowering. The semantic ``core`` itself is left
    unchanged.
    """
    if isinstance(core, C.ValueQuantifier):
        explicit, body = core.var, core.body
    else:
        explicit, body = None, core
    joins = tuple(_joins(body))
    if explicit is not None:
        if joins:
            raise UnsupportedConstruct(
                "a join between two elements cannot be combined with a value capture "
                "(`... as v`): lowering can run one captured value or one join strategy"
            )
        return body
    if not joins:
        return body
    reason = _one_value_reason(body)
    if reason is None:
        c._capture_var = "__join__"
        return _plan_joins(body, None)
    return _plan_joins(body, _join_element(body, reason))


def _plan_joins(f: C.Formula, candidate: str | None) -> C.Formula:
    """``f`` with semantic joins replaced by lowering's selected private execution form."""
    charge(1)
    match f:
        case C.Join(test_var, test_surface, test_field, bound_var, bound_surface, bound_field):
            if candidate is None:
                return C.conj(
                    [
                        C.Pred(
                            _JoinSide("ref", test_field, bound_field, bound_var),
                            test_var,
                            test_surface,
                        ),
                        C.Pred(
                            _JoinSide("bind", bound_field, bound_field, bound_var),
                            bound_var,
                            bound_surface,
                        ),
                    ]
                )
            if candidate == bound_var:
                return C.Pred(
                    _JoinSide("pinned", test_field, bound_field, bound_var),
                    test_var,
                    test_surface,
                )
            if candidate == test_var:
                return C.Pred(
                    _JoinSide("within", bound_field, test_field, test_var),
                    bound_var,
                    bound_surface,
                )
            raise UnsupportedConstruct("internal: selected element does not take part in a join")
        case C.Not(body):
            return C.Not(_plan_joins(body, candidate))
        case C.And(parts):
            return C.And(tuple(_plan_joins(p, candidate) for p in parts))
        case C.Or(parts):
            return C.Or(tuple(_plan_joins(p, candidate) for p in parts))
        case C.Quant(q, var, coll, anchor, body):
            return C.Quant(q, var, coll, anchor, _plan_joins(body, candidate))
        case C.Count(var, coll, body, op, n):
            return C.Count(var, coll, _plan_joins(body, candidate), op, n)
        case C.ToolsExists(var, body, q, src):
            return C.ToolsExists(var, _plan_joins(body, candidate), q, src)
        case C.ChunkQuant(q, var, collection, body):
            return C.ChunkQuant(q, var, collection, _plan_joins(body, candidate))
        case C.Closed(body):
            return C.Closed(_plan_joins(body, candidate))
    return f


def compile_core(
    rule,
    *,
    max_work: int | None = DEFAULT_MAX_WORK,
    decompose: bool = True,
    response_schemas: Mapping[str, str] | None = None,
):
    """Compile a parsed rule through the core calculus into a :class:`CompileResult`.

    Compiling the rule directly (its automaton and its closed parts') may make ``max_work``
    formula-node visits (``quarry.engine.budget``; ``None`` lifts the budget). When the rule
    fails at that budget or another construction limit, it is compiled again with one of its
    positive ``any`` elements decided per candidate event, outermost first: each run then tests that element at one event, which can
    make the automaton far smaller. The retries together may make another ``max_work`` visits.
    The first retry that compiles is kept; otherwise the original diagnostic is reported.

    ``decompose`` compiles each operand of the rule's root ``and``/``or``/``not`` as a check
    of its own, so the artifact carries the rule's decision tree (``quarry.engine.decision``).
    A rule that fails that way at a construction limit is compiled whole, with a fresh
    budget, before any candidate retry, so a compile does at most three times ``max_work``."""
    result, limited, candidates = _compile(
        rule, WorkBudget(max_work), None, decompose, response_schemas
    )
    if limited and decompose:
        result, limited, candidates = _compile(
            rule, WorkBudget(max_work), None, False, response_schemas
        )
    if not limited:
        return result
    retries = WorkBudget(max_work)
    for var in candidates:
        retry, _, _ = _compile(rule, retries, var, False, response_schemas)
        if retry.ok:
            return retry
    return result


def _compile(
    rule,
    budget: WorkBudget,
    bind: str | None,
    decompose: bool,
    response_schemas: Mapping[str, str] | None,
):
    """:func:`_attempt` spending ``budget``, with a rule nested deeper than the recursive
    walkers allow failing to load with a named reason. It is not a construction limit: a retry
    would nest as deeply."""
    from quarry.engine.compiler import CompileResult

    try:
        with spending(budget):
            return _attempt(rule, bind, decompose, response_schemas)
    except RecursionError:
        name = str(getattr(rule, "name", "") or "")
        diagnostics = Diagnostics()
        diagnostics.error(
            "the rule is nested too deeply to compile (for example, a very long chain of "
            "`and`/`or`); split it into several rules",
            ErrorCategory.UNSUPPORTED,
            SourceRef(rule_name=name),
        )
        return CompileResult(name, None, diagnostics), False, []


def _attempt(rule, bind: str | None, decompose: bool, response_schemas: Mapping[str, str] | None):
    """One compilation, with ``bind`` decided per candidate when given. Returns the result,
    whether it failed at a construction limit, and the elements a retry may decide per
    candidate (none once an element or a value is bound).

    ``decompose`` splits the root into checks unless a candidate or a captured value is
    decided per run, which binds across the whole root."""
    from quarry.engine.compiler import CompileResult, _Compiler, reject_unguarded_refusal_reads

    c = _Compiler(rule, response_schemas)
    source = SourceRef(rule_name=c.rule_name)
    condition = getattr(rule, "condition", None)
    if condition is None:
        c.diagnostics.error("rule has no condition", ErrorCategory.UNSUPPORTED, source)
        return CompileResult(c.rule_name, None, c.diagnostics), False, []
    semantic_core: C.Formula | None = None
    execution: C.Formula | None = None
    lowerer: _Lowerer | None = None

    def at_limit(exc: DFAConstructionLimit):
        c.diagnostics.error(str(exc), ErrorCategory.UNSUPPORTED, source)
        per_candidate = (
            execution is not None
            and lowerer is not None
            and lowerer.bound is None
            and not c._capture_var
        )
        # A flow operand's own event has no name in the rule and is not retried.
        retry = (
            [v for v in _positive_binders(execution) if not v.startswith("\x00")]
            if per_candidate
            else []
        )
        return CompileResult(c.rule_name, None, c.diagnostics), True, retry

    try:
        reject_unguarded_refusal_reads(condition)
        c, semantic_core = _desugar(condition, c)
        execution = _execution_core(c, semantic_core)
        split_root = decompose and bind is None and not c._capture_var
        collect_plan = c.compile_collect(rule.collect) if getattr(rule, "collect", None) else None
        lowerer = _Lowerer(c)
        try:
            if bind is not None:
                rebound, coll, own_step = bind_candidate(execution, bind)
                lowerer.bind(bind, coll, own_step)
                pattern = lowerer.root(hoist(rebound))
            else:
                pattern = lowerer.root(hoist(execution, split_root=split_root))
        except NeedsBinding as need:
            if bind is not None:
                raise UnsupportedConstruct(
                    f"{_described_pair(bind, need.var, execution)} would both be decided one "
                    "candidate at a time; split the rule"
                ) from None
            if c._capture_var:
                raise UnsupportedConstruct(
                    f"{described(need.var)} would be decided one candidate at a time, which "
                    "cannot be combined with a value capture or a join"
                ) from None
            split_root = False
            c = _Compiler(rule, response_schemas)
            if collect_plan is not None:
                collect_plan = c.compile_collect(rule.collect)
            lowerer = _Lowerer(c)
            rebound, coll, own_step = bind_candidate(execution, need.var)
            lowerer.bind(need.var, coll, own_step)
            try:
                pattern = lowerer.root(hoist(rebound))
            except NeedsBinding as again:
                if again.var == need.var:
                    # The candidate is decided, yet a test still needs it: it is read where one
                    # run's candidate cannot reach, as in a comprehension over tools.
                    raise UnsupportedConstruct(
                        f"{described(need.var)} is read where it cannot be decided one candidate "
                        "at a time (for example, inside a comprehension over tools); split the "
                        "rule"
                    ) from None
                raise UnsupportedConstruct(
                    f"{_described_pair(need.var, again.var, execution)} would both be decided "
                    "one candidate at a time; a rule decides at most one element that way, so "
                    "split the rule"
                ) from None
        capture_plan = (
            _capture_plan(c, execution) if c._capture_var and c._capture_bind_id else None
        )
    except DFAConstructionLimit as exc:
        # Hoisting or an `N of` expansion needs more than the budget.
        return at_limit(exc)
    except UnsupportedConstruct as exc:
        c.diagnostics.error(str(exc), ErrorCategory.UNSUPPORTED, source)
        return CompileResult(c.rule_name, None, c.diagnostics), False, []
    except (IndexError, AttributeError, ValueError, TypeError) as exc:
        c.diagnostics.error(
            f"{type(exc).__name__} while lowering: {exc}", ErrorCategory.INTERNAL, source
        )
        return CompileResult(c.rule_name, None, c.diagnostics), False, []

    unused_parameters = sorted(set(c.parameters) - c.used_parameters)
    if unused_parameters:
        c.diagnostics.error(
            "declared parameters are not used: " + ", ".join(unused_parameters),
            ErrorCategory.REFERENCE,
            source,
        )
        return CompileResult(c.rule_name, None, c.diagnostics), False, []

    # The rule's automaton and each closed part's spend one construction budget. A rule
    # nested too deeply for the walkers is reported by ``_compile``, with no retry.
    def automaton(formula: T.Formula, past: tuple[PastDef, ...], atom_order=None):
        return compile_dfa(T.simplify(T.Not(formula)), c.rule_name, past, atom_order=atom_order)

    by_id = {s.atom_id: s for s in c.atoms.specs}

    def cheapest_first(atom_id: str) -> tuple[int, bool]:
        # The executor's grounding order: by cost, a check decided by an evaluation of its
        # own after the plain checks of its cost. The root's guard tests its checks in this
        # order, so walking it grounds only checks a cheapest-first loop would ground.
        spec = by_id[atom_id]
        return spec.cost.rank, atom_id in lowerer.closed or atom_id in c.tool_quantifiers

    try:
        dfa = automaton(pattern, tuple(lowerer._past.defs), cheapest_first if split_root else None)
        closed_conditions = {
            atom_id: ClosedPlan(automaton(formula, past))
            for atom_id, (formula, past) in lowerer.closed.items()
        }
    except DFAConstructionLimit as exc:
        return at_limit(exc)

    notes = list(c.notes)
    if c.has_static:
        notes.append("has static (boot-step) atoms")
    specs = {s.atom_id: s for s in c.atoms.specs}
    bound = lowerer.bound
    if bound is not None:
        window = bound.coll.window or (None, None)
        capture_plan = ElementPlan(
            var=bound.var,
            surface=bound.coll.surface,
            at_atom_id=bound.at_atom_id,
            ref_atom_ids=frozenset(bound.ref_ids),
            window_start=window[0],
            window_stop=window[1],
            filter_atom_id=bound.filter_atom_id,
            filter_name=bound.filter_name,
            own_step=bound.own_step_formula(specs),
        )
        notes.append(f"candidate element `{bound.var}` (run DFA per candidate event)")
    elif capture_plan is not None:
        notes.append(f"value capture `{capture_plan.var}` (run DFA per candidate value)")
    tool_body_atoms = {a for plan in c.tool_quantifiers.values() for a in T.atoms_of(plan.body)}
    closed_atoms = {a for plan in closed_conditions.values() for a in plan.dfa.atoms}
    # A candidate or captured value is run once per binding, so its proof keeps the whole
    # domain of a negated event check.
    full_trace_proof = _needs_full_trace_proof(
        _for_domain_analysis(pattern, {d.token: d for d in lowerer._past.defs}),
        specs,
        keep_negated_event_domain=capture_plan is not None,
        derived={
            **{atom_id: plan.body for atom_id, plan in c.tool_quantifiers.items()},
            **{
                atom_id: _for_domain_analysis(formula, {d.token: d for d in past})
                for atom_id, (formula, past) in lowerer.closed.items()
            },
        },
    )
    node_atom_ids = c.final_node_atom_ids(set(dfa.atoms) | tool_body_atoms | closed_atoms)
    decomposed = (
        split_root
        and bound is None
        and capture_plan is None
        and bool(dfa.atoms)
        and all(specs[a].domain is AtomDomain.STATIC for a in dfa.atoms)
        and boot_diagram_is_decided(dfa)
    )
    decision_tree = _build_tree(
        condition=condition,
        root=dfa,
        decomposed=decomposed,
        closed={atom_id: plan.dfa for atom_id, plan in closed_conditions.items()},
        tool_bodies={
            atom_id: T.atoms_of(plan.body) for atom_id, plan in c.tool_quantifiers.items()
        },
        node_atom_ids=node_atom_ids,
    )
    artifact = CompiledRuleArtifact(
        rule_name=c.rule_name,
        dfa=dfa,
        dfa_hash=dfa_hash(dfa, COMPILER_VERSION),
        atom_specs=specs,
        node_atom_ids=node_atom_ids,
        temporal=c.temporal,
        notes=tuple(notes),
        capture_plan=capture_plan,
        parameter_schema=tuple(getattr(rule, "parameters", ())),
        collect_plan=collect_plan,
        tool_quantifiers=c.tool_quantifiers,
        closed_conditions=closed_conditions,
        full_trace_proof=full_trace_proof,
        decision_tree=decision_tree,
    )
    return CompileResult(c.rule_name, artifact, c.diagnostics), False, []


def _top_operands(condition: ConditionNode) -> list[ConditionNode]:
    """The authored conditions under the root ``and``/``or``/``not``, in authored order."""

    if isinstance(condition, UnaryOp) and condition.operator == "not":
        return _top_operands(condition.operand)
    if (
        isinstance(condition, BinaryOp)
        and not isinstance(condition, FlowOp)
        and condition.operator in {"and", "or"}
    ):
        return [*_top_operands(condition.left), *_top_operands(condition.right)]
    return [condition]


def _build_tree(
    *,
    condition: ConditionNode,
    root: DFA,
    decomposed: bool,
    closed: Mapping[str, DFA],
    tool_bodies: Mapping[str, Iterable[str]],
    node_atom_ids: Mapping[int, tuple[str, ...]],
) -> DecisionTree:
    """Build a rule's node table.

    A decomposed tree has one node per atom its root automaton reads; each is labeled with
    the authored root operand whose lowered atoms it reads most. An undecomposed tree has
    one node: the rule.
    """

    preorder = {id(node): index for index, node in enumerate(walk_condition_nodes(condition))}
    if not decomposed:
        return DecisionTree((TreeNode(None, render_condition(condition), preorder[id(condition)]),))
    parts = Parts(closed, tool_bodies)
    operands = [(operand, preorder[id(operand)]) for operand in _top_operands(condition)]
    nodes = []
    for atom_id in root.atoms:
        read = reads(atom_id, parts)
        best = max(
            operands,
            key=lambda item: len(read.intersection(node_atom_ids.get(item[1], ()))),
        )
        overlap = read.intersection(node_atom_ids.get(best[1], ()))
        label_node, source = best if overlap else (condition, preorder[id(condition)])
        nodes.append(TreeNode(atom_id, render_condition(label_node), source))
    return DecisionTree(tuple(nodes))


def _described_pair(a: str, b: str, core: C.Formula) -> str:
    """``a`` and ``b`` as a diagnostic names them together. Two elements the rule names alike
    are told apart by their order among its elements of that name (the order they are bound
    in, which is the rule's reading order)."""
    if described(a) != described(b):
        return f"{described(a)} and {described(b)}"
    if a.startswith("\x00"):
        return "the events of two flow operands"
    alike = _alike(a, core)
    first, second = sorted((alike.index(a) + 1, alike.index(b) + 1))
    return f"the {_ordinal(first)} and {_ordinal(second)} `{source_name(a)}`"


def _alike(var: str, core: C.Formula) -> list[str]:
    """The rule's elements named like ``var``, in the order they are bound (its reading
    order)."""
    name = source_name(var)
    return sorted(
        {v for v in _vars(core) if not v.startswith("\x00") and source_name(v) == name},
        key=lambda v: int(v.split("\x00")[1]) if "\x00" in v else -1,
    )


def _field_of(var: str, path: str, core: C.Formula) -> str:
    """``var``'s field ``path`` as a diagnostic names it: an element named like another is
    told apart by its order among them, and a flow operand's event, unnamed in the rule, by
    what it is."""
    if var.startswith("\x00"):
        return f"`{path}` of {described(var)}"
    alike = _alike(var, core)
    if len(alike) > 1:
        return f"`{path}` of the {_ordinal(alike.index(var) + 1)} `{source_name(var)}`"
    return f"`{source_name(var)}.{path}`"


def _vars(f: C.Formula) -> set[str]:
    """The element variables ``f`` binds or reads."""
    own = getattr(f, "var", None)
    found = {own} if isinstance(own, str) else set()
    return found.union(*(_vars(child) for child in C.children(f)))


def _ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _needs_full_trace_proof(
    pattern: T.Formula,
    specs: dict,
    *,
    keep_negated_event_domain: bool = False,
    derived: dict[str, T.Formula] | None = None,
) -> bool:
    """Whether a match must be explained over the complete event word.

    A positive ``Globally`` or negative ``Eventually`` proves a claim over a domain;
    deleting elements could make it true on a different domain. ``Until`` similarly
    needs its sustained prefix. A negated predicate inside an existential can shrink
    when its execution path has a small replay seed. Capture and legacy per-tool rules
    retain their conservative full-domain evidence for that case. An atom decided by a
    formula of its own (``derived``: a tools body, a closed part's formula) is read through
    that formula, at the polarity where the atom occurs.
    """
    derived = derived or {}

    def visit(formula: T.Formula, negative: bool = False) -> bool:
        match formula:
            case T.Atom(name) if name in derived:
                return visit(derived[name], negative)
            case T.Atom(name):
                return (
                    keep_negated_event_domain
                    and negative
                    and specs[name].domain is AtomDomain.EVENT
                )
            case T.Not(child):
                return visit(child, not negative)
            case T.Implies(left, right):
                return visit(left, not negative) or visit(right, negative)
            case T.And(left, right) | T.Or(left, right):
                return visit(left, negative) or visit(right, negative)
            case T.Until():
                return True
            case T.Globally(child):
                return not negative or visit(child, negative)
            case T.Eventually(child):
                return negative or visit(child, negative)
            case T.Next(child):
                return visit(child, negative)
        return False

    return visit(pattern)


def _for_domain_analysis(f: T.Formula, past: dict[str, PastDef]) -> T.Formula:
    """``f`` with each past token replaced by its meaning in future operators of the same
    polarity (``O φ`` as ``F φ``, ``Y φ`` as ``X φ``) and bounded windows read as ``F``: the
    shape proof-domain classification reads. It is not evaluated."""
    # A left-associated but otherwise valid And/Or chain can be hundreds of nodes deep.
    # Rebuilding it recursively uses two Python frames per node (the function and its
    # children comprehension), so walk it in postorder instead. Hold each source object
    # with its result to keep identity keys stable throughout this walk.
    rewritten: dict[int, tuple[T.Formula, T.Formula]] = {}
    pending = [(f, False)]
    while pending:
        node, ready = pending.pop()
        key = id(node)
        if key in rewritten:
            continue
        if not ready:
            pending.append((node, True))
            if isinstance(node, T.Atom) and node.name in past:
                pending.append((past[node.name].operand, False))
            else:
                for fl in fields(node):
                    child = getattr(node, fl.name)
                    if isinstance(child, T.Formula):
                        pending.append((child, False))
            continue
        if isinstance(node, T.Atom) and node.name in past:
            definition = past[node.name]
            operand = rewritten[id(definition.operand)][1]
            result = T.Eventually(operand) if definition.op == "O" else T.Next(operand)
        elif isinstance(node, T.BoundedEventually):
            result = T.Eventually(rewritten[id(node.child)][1])
        else:
            children = {
                fl.name: rewritten[id(child)][1]
                for fl in fields(node)
                if isinstance(child := getattr(node, fl.name), T.Formula)
            }
            result = replace(node, **children) if children else node
        rewritten[key] = (node, result)
    return rewritten[id(f)][1]


def _source_of(node):
    """The authored syntax a lowering leaf stands for."""
    if isinstance(node, _JoinSide):
        return node.field
    if isinstance(node, ValueTest):
        return node.pattern
    return node
