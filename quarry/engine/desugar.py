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

"""Desugar a rule's syntax tree into the core calculus (``quarry.engine.core``).

Each surface family has one rule here; every name and field is resolved once. Leaf tests
keep their syntax subtree (``Pred``); closed special forms that have their own atom
(system-prompt tests, model and chunked checks, ``has_sequence``, collection sizes) become
``Static`` or ``Size`` nodes. Shapes the reference does not accept raise
``UnsupportedConstruct``. :func:`hoist` then moves every part of a quantifier body that does
not read the quantified element out of it, which is the reference's rule that such a part
"is checked over the whole conversation", and gives each of the root's temporal operands an
automaton of its own when there are several.
"""

from __future__ import annotations

import itertools
from contextvars import ContextVar
from dataclasses import dataclass, replace

from quarry.engine import core as C
from quarry.engine.budget import charge
from quarry.engine.diagnostics import UnsupportedConstruct
from quarry.types.dsl import (
    AllComprehension,
    AnyComprehension,
    BinaryOp,
    FieldAccess,
    FilterComprehension,
    FunctionCall,
    Identifier,
    Literal,
    MapComprehension,
    PatternMatch,
    StringCount,
    StringPosition,
    StringSetQuantifier,
    UnaryOp,
)
from quarry.types.dsl.operators.flow import FlowOp
from quarry.types.dsl.traversal import condition_child_fields, iter_condition_children

# Closed groups (parts that read an enclosing element) one quantifier body may move out by
# expansion (``_expand``), which doubles the body per group. The cap, not the work budget, bounds
# it: an expansion step copies and walks the whole body outside the budgeted walkers. Eight
# groups load in under 3 seconds; ten spend 15 seconds before the budget would reject them.
_MAX_HOISTED_PARTS = 8
_NESTED_CHUNKS = (
    "chunked scoring is a whole-conversation question and cannot be nested inside another "
    "comprehension; write it as its own top-level term"
)
_FLOW_FUNCTIONS = ("called_tool", "called_tools", "user_said", "assistant_said")


@dataclass(frozen=True)
class _Binding:
    """An element variable in scope: the rule's ``name`` for it, its core variable ``var``
    (unique in the rule, so two comprehensions over one name stay two binders), its surface
    and collection."""

    name: str
    var: str
    surface: str
    coll: C.Events
    name_alias: bool = False  # ``call_sequence()``: the element stands for its ``.name``


@dataclass(frozen=True)
class ValueTest:
    """A test on the captured value itself: it contains ``pattern`` (``"x" in v``), equals
    it, or matches it as a regex (``predicate`` is ``contains_any``, ``equals`` or
    ``regex``)."""

    predicate: str
    pattern: object


@dataclass(frozen=True)
class _EventAlt:
    """A flow operand alternative that is an event quantifier: ``var`` over ``coll``."""

    var: str
    coll: C.Events
    body: C.Formula


@dataclass(frozen=True)
class _ElementAlt:
    """A flow operand alternative that is the enclosing element ``element``'s condition."""

    element: str
    body: C.Formula


@dataclass(frozen=True)
class _ToolsAlt:
    """A flow operand alternative that is a static tools condition, true from boot on."""

    body: C.Formula


_Alternative = _EventAlt | _ElementAlt | _ToolsAlt


class Desugarer:
    """Surface syntax to core. ``tables`` answers surface and accessor questions (the
    compiler's collection tables), so this module owns structure only."""

    def __init__(self, tables, *, value_var: str | None = None):
        self._t = tables
        self._value_var = value_var
        self._fresh = itertools.count()
        self._bound: set[str] = set()

    def fresh(self, stem: str = "e") -> str:
        return f"\x00{stem}{next(self._fresh)}"

    def _bind(self, name: str, surface: str, coll: C.Events, alias: bool = False) -> _Binding:
        """A binder for ``name``: the name itself the first time, a unique variant after."""
        var = name if name not in self._bound else f"{name}\x00{next(self._fresh)}"
        self._bound.add(var)
        return _Binding(name, var, surface, coll, alias)

    def condition(self, node) -> C.Formula:
        """Desugar a condition node into a core formula."""
        body = self.cond(node, ())
        return C.ValueQuantifier(self._value_var, body) if self._value_var else body

    def cond(self, node, scope: tuple[_Binding, ...]) -> C.Formula:
        """Desugar a condition node into a core formula."""
        if isinstance(node, Literal) and isinstance(node.value, bool):
            return C.Const(node.value)
        if (
            isinstance(node, BinaryOp)
            and not isinstance(node, FlowOp)
            and node.operator in ("and", "or")
        ):
            alias = self._t.scalar_alias(node.left)
            if alias is not None and node.operator == "and":
                var, expr = alias
                return self.cond(self._t.substitute_identifier(node.right, var, expr), scope)
            left, right = self.cond(node.left, scope), self.cond(node.right, scope)
            return C.conj([left, right]) if node.operator == "and" else C.disj([left, right])
        if isinstance(node, UnaryOp) and node.operator == "not":
            return C.neg(self.cond(node.operand, scope))
        if isinstance(node, FlowOp):
            return self.flow(node, scope)
        if isinstance(node, (AnyComprehension, AllComprehension)):
            return self.comprehension(node, scope)
        counted = self.filter_count(node, scope)
        if counted is not None:
            return counted
        if isinstance(node, FunctionCall) and node.name in _FLOW_FUNCTIONS:
            return self.fact(node)
        if self._t.is_size(node):
            return C.Size(node)
        test = self._t.value_test(node)
        if test is not None:
            return C.Pred(test, None, None)
        reads = self.reads(node, scope)
        if not reads:
            if scope and self._t.is_chunked(node):
                raise UnsupportedConstruct(_NESTED_CHUNKS)
            if self._t.contains_model_call(node):
                # A model input reads `X[i]` as a message that may not exist, not as an event
                # to pin the check to: a model check that reads no element in scope is scored
                # once per scan against the conversation, as `messages[-1:]` is.
                return C.Static(node)
            indexed = self._t.indexed_read(node)
            if indexed is not None:
                # A test on one indexed event (`messages[-1].content`): its atom holds only at
                # that event of the surface.
                surface, index = indexed
                var = self.fresh()
                events = C.Events(surface, index=index, src=node)
                read = self._t.drop_index(node, source_name(var))
                return C.Quant("exists", var, events, None, C.Pred(read, var, surface))
            return C.Static(node)
        join = self.join(node, scope)
        if join is not None:
            return join
        if len(reads) > 1:
            raise UnsupportedConstruct(
                f"a condition reads fields of {', '.join(f'`{r}`' for r in sorted(reads))} at "
                "one event; a condition can read one element (join with `... contains x.f`)"
            )
        (var,) = reads
        binding = _lookup(scope, var)
        return C.Pred(node, binding.var, binding.surface)

    def join(self, node, scope) -> C.Formula | None:
        """``test contains bound`` as one strategy-neutral relation between scoped elements."""
        if not (
            isinstance(node, BinaryOp)
            and node.operator == "contains"
            and isinstance(node.left, FieldAccess)
            and isinstance(node.right, FieldAccess)
        ):
            return None
        test_name, bound_name = field_root(node.left), field_root(node.right)
        names = {b.name for b in scope}
        if test_name is None or bound_name is None or not {test_name, bound_name} <= names:
            return None
        test, bound = _lookup(scope, test_name), _lookup(scope, bound_name)
        return C.Join(
            test.var,
            test.surface,
            node.left,
            bound.var,
            bound.surface,
            node.right,
        )

    def fact(self, node: FunctionCall) -> C.Formula:
        """Desugar a fact node into a core formula."""
        if node.name in ("called_tool", "called_tools"):
            if not node.args:
                raise UnsupportedConstruct(f"{node.name}() needs a tool name")
            return C.conj(
                C.Quant(
                    "exists",
                    self.fresh(),
                    C.Events("tool_calls", self._t.lit(a), src=node),
                    None,
                    C.TRUE,
                )
                for a in node.args
            )
        surface = "user_messages" if node.name == "user_said" else "assistant_messages"
        var = self.fresh()
        events = C.Events(surface, src=node)
        return C.Quant("exists", var, events, None, C.Pred(node, var, surface))

    def comprehension(self, node, scope) -> C.Formula:
        """Desugar a comprehension node into a core formula."""
        q = "exists" if isinstance(node, AnyComprehension) else "forall"
        if self._t.is_special_collection(node):
            if scope:
                raise UnsupportedConstruct(_NESTED_CHUNKS)
            chunk = self._bind(node.variable, "\x00chunks", C.Events("\x00chunks"))
            return C.ChunkQuant(q, chunk.var, node.collection, self.cond(node.condition, (chunk,)))
        if self._t.contains_model_call(node.condition):
            self._check_model_comprehension(node, scope)
        if self._t.is_tools(node.collection):
            binding = self._bind(node.variable, "tools", C.Events("tools"))
            body = self.cond(node.condition, (*scope, binding))
            return C.ToolsExists(binding.var, body, q, src=node)
        binding = self.binding(node.variable, node.collection, src=node)
        body = self.cond(self._aliased(node, binding), (*scope, binding))
        return C.Quant(q, binding.var, binding.coll, None, body)

    def _check_model_comprehension(self, node, scope) -> None:
        """Model calls are read at one item of one non-nested ``any`` comprehension, and each
        reads exactly that item."""
        if isinstance(node, AllComprehension):
            raise UnsupportedConstruct("model calls are not supported inside all comprehensions")
        if contains_comprehension(node.condition):
            raise UnsupportedConstruct(
                "model calls are supported in one non-nested any comprehension"
            )
        if self._t.is_tools(node.collection):
            raise UnsupportedConstruct(
                "model calls are not supported inside a static tools comprehension"
            )
        names = {b.name for b in scope} | {node.variable}
        item = _Binding(node.variable, node.variable, "", C.Events(""))
        for call in self._t.model_calls(node.condition):
            if self.reads(call, (*scope, item)) & names != {node.variable}:
                raise UnsupportedConstruct(
                    "every model call in an any comprehension must depend on its bound item"
                )

    def binding(self, var: str, collection, src=None) -> _Binding:
        """``var`` over ``collection``; ``src`` is the syntax that binds it (its
        comprehension), for attributing the collection's atoms."""
        surface, name, window, alias = self._t.resolve_collection(collection)
        return self._bind(var, surface, C.Events(surface, name, window, src=src), alias)

    def _aliased(self, node, binding: _Binding):
        return self._t.name_alias(node) if binding.name_alias else node.condition

    def filter_count(self, node, scope) -> C.Formula | None:
        """Desugar a filter count node into a core formula."""
        shape = self._t.filter_count_shape(node)
        if shape is None:
            return None
        comp, op, n = shape
        binding = self.binding(comp.variable, comp.collection, src=comp)
        body = self.cond(self._aliased(comp, binding), (*scope, binding))
        if not free_only(body, binding.var):
            raise UnsupportedConstruct(
                f"count([x for x in {binding.surface} if ...]) needs a filter that only reads "
                "the element itself"
            )
        return C.Count(binding.var, binding.coll, body, op, n)

    def flow(self, node: FlowOp, scope) -> C.Formula:
        """Desugar a flow node into a core formula."""
        if self._t.contains_model_call(node):
            raise UnsupportedConstruct("model calls cannot be operands of temporal flow")
        if node.operator == "until":
            return self.until(node, scope)
        if node.operator != "after":
            raise UnsupportedConstruct(f"flow operator {node.operator!r}")
        lefts, rights = _name_variants(node.left), _name_variants(node.right)
        if len(lefts) > 1 or len(rights) > 1:
            # `called_tools(a, b)` is every named tool called: one flow per name, all holding.
            # A top-level `or` gives each named flow several event alternatives. A
            # single alternative can also contain a large Boolean body, desugared once
            # per flow. Reserve both repeated body work and the `disj`/`conj` duplicate
            # scans before constructing any expanded flow.
            flows = len(lefts) * len(rights)
            left_count, left_extra, left_square = _flow_operand_profile(
                node.left, split=len(lefts) > 1
            )
            right_count, right_extra, right_square = _flow_operand_profile(
                node.right, split=len(rights) > 1
            )
            alternatives = left_count * right_count
            pair_extra = left_extra * right_count + right_extra * left_count
            charge(
                flows * alternatives * (flows + alternatives)
                + flows * (left_square + right_square)
                + flows * (flows + alternatives) * pair_extra
            )
            return C.conj(
                self.flow(FlowOp("after", left, right, max_distance=node.max_distance), scope)
                for left in lefts
                for right in rights
            )
        later = self.alternatives(node.left, scope, sustained=False)
        earlier = self.alternatives(node.right, scope, sustained=False)
        within = node.max_distance
        self._t.note("unbounded flow" if within is None else f"bounded flow within:{within}")
        pairs = []
        for late in later:
            for early in earlier:
                if (
                    isinstance(late, _ElementAlt)
                    and isinstance(early, _ElementAlt)
                    and late.element == early.element
                ):
                    continue  # one event cannot precede itself
                pairs.append(self._after(late, early, within))
        if not pairs:
            raise UnsupportedConstruct(
                "a flow cannot relate an element to itself: both operands read the same "
                "element; use a second comprehension variable for the other event"
            )
        return C.disj(pairs)

    def _after(self, late: _Alternative, early: _Alternative, within) -> C.Formula:
        if isinstance(early, _ToolsAlt):
            # A static condition holds at every step from the boot step on, so some step
            # right before every event satisfies it, within any bound.
            return C.conj([early.body, self._event_or_element(late, None)])
        if isinstance(late, _ToolsAlt):
            if isinstance(early, _ElementAlt):
                return C.conj(
                    [
                        late.body,
                        early.body,
                        C.Quant(
                            "exists",
                            self.fresh(),
                            C.AllEvents(),
                            C.Rel(early.element, "after", 1, within),
                            C.TRUE,
                        ),
                    ]
                )
            nxt = self.fresh()
            return C.Quant(
                "exists",
                early.var,
                early.coll,
                None,
                C.conj(
                    [
                        early.body,
                        late.body,
                        C.Quant(
                            "exists",
                            nxt,
                            C.AllEvents(),
                            C.Rel(early.var, "after", 1, within),
                            C.TRUE,
                        ),
                    ]
                ),
            )
        if isinstance(late, _ElementAlt):
            if isinstance(early, _ElementAlt):
                at = self.fresh()
                anchor = C.Rel(early.element, "after", 1, within)
                later = C.Quant("exists", at, C.AllEvents(), anchor, C.Same(at, late.element))
                return C.conj([late.body, early.body, later])
            return C.conj(
                [
                    late.body,
                    C.Quant(
                        "exists",
                        early.var,
                        early.coll,
                        C.Rel(late.element, "before", 1, within),
                        early.body,
                    ),
                ]
            )
        if isinstance(early, _ElementAlt):
            return C.conj(
                [
                    early.body,
                    C.Quant(
                        "exists",
                        late.var,
                        late.coll,
                        C.Rel(early.element, "after", 1, within),
                        late.body,
                    ),
                ]
            )
        return C.Quant(
            "exists",
            early.var,
            early.coll,
            None,
            C.conj(
                [
                    early.body,
                    C.Quant(
                        "exists",
                        late.var,
                        late.coll,
                        C.Rel(early.var, "after", 1, within),
                        late.body,
                    ),
                ]
            ),
        )

    def _event_or_element(self, alt: _Alternative, anchor) -> C.Formula:
        if isinstance(alt, _ElementAlt):
            return alt.body
        if isinstance(alt, _ToolsAlt):
            return C.conj(
                [alt.body, C.Quant("exists", self.fresh(), C.AllEvents(), anchor, C.TRUE)]
            )
        return C.Quant("exists", alt.var, alt.coll, anchor, alt.body)

    def until(self, node: FlowOp, scope) -> C.Formula:
        """Desugar a until node into a core formula."""
        if node.max_distance is not None:
            raise UnsupportedConstruct("`until` takes no `within` bound")
        self._t.note("until")
        sustained = self.alternatives(node.left, scope, sustained=True)
        releases = self.alternatives(node.right, scope, sustained=False)
        out = []
        for release in releases:
            if isinstance(release, _ToolsAlt):
                # A static condition that holds, holds at the boot step, before any event:
                # it releases at once, and nothing needs sustaining.
                out.append(release.body)
            elif isinstance(release, _ElementAlt):
                out.append(C.conj([release.body, self._sustain(sustained, release.element)]))
            else:
                out.append(
                    C.Quant(
                        "exists",
                        release.var,
                        release.coll,
                        None,
                        C.conj([release.body, self._sustain(sustained, release.var)]),
                    )
                )
        return C.disj(out)

    def _sustain(self, alternatives: list[_Alternative], release: str) -> C.Formula:
        """Every event of the sustained side's kinds before ``release`` satisfies one of its
        alternatives there."""
        a = self.fresh("s")
        gates, holds = [], []
        for alt in alternatives:
            if isinstance(alt, _ElementAlt):
                gate = C.Same(a, alt.element)
                gates.append(gate)
                holds.append(C.conj([gate, rename(alt.body, alt.element, a)]))
            elif isinstance(alt, _ToolsAlt):
                gates.append(C.TRUE)
                holds.append(alt.body)
            else:
                gate = C.Member(a, alt.coll)
                gates.append(gate)
                holds.append(C.conj([gate, rename(alt.body, alt.var, a)]))
        body = C.disj([C.neg(C.disj(gates)), C.disj(holds)])
        return C.Quant("forall", a, C.AllEvents(), C.Rel(release, "before", 1, None), body)

    def alternatives(self, node, scope, *, sustained: bool) -> list[_Alternative]:
        """The alternatives an operand holds at: split at top-level ``or``."""
        if isinstance(node, BinaryOp) and not isinstance(node, FlowOp) and node.operator == "or":
            return [
                *self.alternatives(node.left, scope, sustained=sustained),
                *self.alternatives(node.right, scope, sustained=sustained),
            ]
        if isinstance(node, (AnyComprehension, AllComprehension)):
            if isinstance(node, AllComprehension) and not sustained:
                raise UnsupportedConstruct(
                    "`all` cannot be a flow operand except on the sustained side of `until`; "
                    "use `any` for one event"
                )
            if self._t.is_tools(node.collection):
                return [_ToolsAlt(self.comprehension(node, scope))]
            if self._t.is_special_collection(node):
                raise UnsupportedConstruct("this collection cannot be a flow operand")
            binding = self.binding(node.variable, node.collection, src=node)
            body = self.cond(self._aliased(node, binding), (*scope, binding))
            return [_EventAlt(binding.var, binding.coll, body)]
        if isinstance(node, FunctionCall) and node.name in _FLOW_FUNCTIONS:
            if node.name == "called_tools" and len(node.args) != 1:
                raise UnsupportedConstruct(
                    "`called_tools` with several names cannot be one flow operand alternative; "
                    "write one flow per name"
                )
            if node.name in ("called_tool", "called_tools"):
                # Its events are every tool call (the kind a sustained side is read at); the
                # name is a test on the call.
                var = self.fresh()
                events = C.Events("tool_calls", src=node)
                return [_EventAlt(var, events, C.Pred(node, var, "tool_calls"))]
            surface = "user_messages" if node.name == "user_said" else "assistant_messages"
            var = self.fresh()
            return [_EventAlt(var, C.Events(surface, src=node), C.Pred(node, var, surface))]
        reads = self.reads(node, scope)
        if len(reads) == 1 and not contains_comprehension(node):
            (name,) = reads
            var = _lookup(scope, name).var
            body = self.cond(node, scope)
            if _groups(body, var, keep_static=False):
                raise UnsupportedConstruct(
                    f"a boolean flow operand can only test the element `{name}`; move other "
                    "conditions beside the flow"
                )
            return [_ElementAlt(var, body)]
        raise UnsupportedConstruct(
            "a flow operand is one event: a condition on the enclosing element, an `any` "
            "comprehension, a fact such as called_tool(...), or an `or` of these"
        )

    def reads(self, node, scope) -> set[str]:
        """The scope variables a leaf test reads, explicitly or implicitly. The variable of a
        map comprehension in the leaf hides a scope variable with the same name."""
        innermost = scope[-1].name if scope else None
        found: set[str] = set()

        def visit(n, nested: bool, names: set[str]) -> None:
            if isinstance(n, Identifier) and n.name in names:
                found.add(n.name)
            elif isinstance(n, FieldAccess):
                root = field_root(n)
                if root is not None and root in names:
                    found.add(root)
            elif (
                not nested
                and innermost is not None
                and isinstance(n, (StringCount, StringPosition, StringSetQuantifier, PatternMatch))
            ):
                found.add(innermost)
            if (
                isinstance(n, BinaryOp)
                and n.operator == "in"
                and isinstance(n.left, (StringCount, StringPosition, StringSetQuantifier))
                and isinstance(n.right, FieldAccess)
            ):
                visit(n.right, nested, names)
                return
            comprehension = hasattr(n, "variable") and hasattr(n, "collection")
            body = names - {n.variable} if isinstance(n, MapComprehension) else names
            for field_name in condition_child_fields(n):
                value = getattr(n, field_name)
                inner = comprehension and field_name != "collection"
                for child in value if isinstance(value, list | tuple) else (value,):
                    if child is not None and hasattr(child, "__dataclass_fields__"):
                        visit(child, nested or inner, body if inner else names)

        visit(node, False, {b.name for b in scope})
        return found


def _name_variants(node) -> list:
    """``called_tools("a", "b")`` as the single-name operands it stands for."""
    if (
        isinstance(node, FunctionCall)
        and node.name in ("called_tool", "called_tools")
        and len(node.args) > 1
    ):
        return [FunctionCall(name="called_tool", args=[arg]) for arg in node.args]
    return [node]


def _flow_operand_profile(node, *, split: bool) -> tuple[int, int, int]:
    """Count top-level `or` leaves and their extra AST work.

    A split `called_tools` becomes a one-name fact in every expanded flow. For other
    leaves, the first two syntax nodes are the ordinary fact baseline already covered by
    the flow-pair charge. Larger bodies are re-desugared for every named flow, and nested
    conjunction/disjunction deduplication can compare their nodes quadratically.
    """
    if split:
        return 1, 0, 0
    leaves = extra = square = 0
    pending = [node]
    while pending:
        current = pending.pop()
        if (
            isinstance(current, BinaryOp)
            and not isinstance(current, FlowOp)
            and current.operator == "or"
        ):
            pending.extend((current.left, current.right))
        else:
            leaves += 1
            body = max(0, _condition_node_count(current) - 2)
            extra += body
            square += body * body
    return leaves, extra, square


def _condition_node_count(node) -> int:
    """Count executable syntax nodes without recursion or materializing a copied tree."""
    count = 0
    active: set[int] = set()
    pending = [(node, False)]
    while pending:
        current, leaving = pending.pop()
        identity = id(current)
        if leaving:
            active.remove(identity)
            continue
        if identity in active:
            raise UnsupportedConstruct("a cyclic condition cannot be a flow operand")
        active.add(identity)
        count += 1
        pending.append((current, True))
        pending.extend((child, False) for child in iter_condition_children(current))
    return count


def field_root(node) -> str | None:
    """Return the root of a field access node."""
    root = node
    while isinstance(root, FieldAccess) and root.base is not None:
        root = root.base
    if isinstance(root, FieldAccess) and root.base is None and root.fields:
        return str(root.fields[0])
    if isinstance(root, Identifier):
        return root.name
    return None


def contains_comprehension(node) -> bool:
    """Return whether a node contains a comprehension."""
    if isinstance(node, (AnyComprehension, AllComprehension, FilterComprehension)):
        return True
    for field_name in condition_child_fields(node):
        value = getattr(node, field_name)
        for child in value if isinstance(value, list | tuple) else (value,):
            if (
                child is not None
                and hasattr(child, "__dataclass_fields__")
                and contains_comprehension(child)
            ):
                return True
    return False


def free_only(formula: C.Formula, var: str) -> bool:
    """True if every leaf of ``formula`` reads ``var`` and nothing else."""
    return C.free_vars(formula) <= {var} and not _has_closed_part(formula, var)


def _has_closed_part(formula: C.Formula, var: str) -> bool:
    return any(var not in C.free_vars(part) for part in _parts(formula))


def rename(formula: C.Formula, old: str, new: str) -> C.Formula:
    """``formula`` with the element variable ``old`` renamed to ``new``."""
    match formula:
        case C.Pred(node=node, var=var, surface=surface, name=name):
            if old != var:
                return formula
            if var == old and name is None:
                name = source_name(old)
            return C.Pred(
                node,
                new if var == old else var,
                surface,
                name,
            )
        case C.Join(
            test_var,
            test_surface,
            test_field,
            bound_var,
            bound_surface,
            bound_field,
        ):
            return C.Join(
                new if test_var == old else test_var,
                test_surface,
                test_field,
                new if bound_var == old else bound_var,
                bound_surface,
                bound_field,
            )
        case C.Not(body):
            return C.Not(rename(body, old, new))
        case C.And(parts):
            return C.And(tuple(rename(p, old, new) for p in parts))
        case C.Or(parts):
            return C.Or(tuple(rename(p, old, new) for p in parts))
        case C.Quant(q, var, coll, anchor, body):
            if var == old:
                return formula
            anchor = (
                C.Rel(new, anchor.side, anchor.lo, anchor.hi)
                if anchor is not None and anchor.of_var == old
                else anchor
            )
            return C.Quant(q, var, coll, anchor, rename(body, old, new))
        case C.Member(var=var, coll=coll):
            return C.Member(new if var == old else var, coll)
        case C.Same(var=var, of_var=of_var):
            return C.Same(new if var == old else var, new if of_var == old else of_var)
        case C.Count(var, coll, body, op, n) if var != old:
            return C.Count(var, coll, rename(body, old, new), op, n)
        case C.ToolsExists(var, body) if var != old:
            return replace(formula, body=rename(body, old, new))
        case C.ChunkQuant(q, var, collection, body) if var != old:
            return C.ChunkQuant(q, var, collection, rename(body, old, new))
        case C.AtCandidate(var) if var == old:
            return C.AtCandidate(new)
    return formula


def _parts(formula: C.Formula) -> list[C.Formula]:
    """Maximal subformulas reached through ``And``/``Or``/``Not``."""
    match formula:
        case C.And(parts) | C.Or(parts):
            return [q for p in parts for q in _parts(p)]
        case C.Not(body):
            return _parts(body)
    return [formula]


def hoist(formula: C.Formula, *, split_root: bool = False) -> C.Formula:
    """``formula`` hoisted (``_hoist``), with its root split into closed parts where two or
    more of them would multiply in one automaton (``_split``).

    ``split_root`` splits every root operand that is not a static leaf into a closed part
    of its own, whenever the root joins two or more with ``and``, ``or`` or ``not``: the
    rule's automaton then decides only static atoms, each one check of its decision tree."""
    token = _keys.set({})
    try:
        return _split(_hoist(formula), force=split_root)
    finally:
        _keys.reset(token)


def _split(formula: C.Formula, *, force: bool = False) -> C.Formula:
    """A closed formula (the rule's root, or a body's closed members) with each temporal
    operand of its ``and``/``or``/``not`` a ``Closed`` part of its own, when there are at
    least two: one automaton would be the product of theirs. Each has one value per run, the
    value at its root, so this is exact.

    ``force`` closes every operand that is not a static leaf, whenever ``formula`` has two
    or more operands under its root ``and``/``or``/``not``."""
    if force:
        # One operand is one check: the tree has one node either way, so keep it whole.
        if len(_parts(formula)) < 2:
            return formula
    elif sum(map(_temporal, _parts(formula))) < 2:
        return formula

    def go(f: C.Formula) -> C.Formula:
        match f:
            case C.Not(body):
                return C.neg(go(body))
            case C.And(parts):
                return C.conj(go(p) for p in parts)
            case C.Or(parts):
                return C.disj(go(p) for p in parts)
        if force:
            # A static check (a model score, a size) already has one value per run and
            # lowers to one static atom: it is a node as it stands.
            static = is_static(f) or isinstance(f, C.Const | C.Static)
            return f if static else C.Closed(f)
        return C.Closed(f) if _temporal(f) else f

    return go(formula)


def _temporal(f: C.Formula) -> bool:
    """``f`` reads events under a temporal operator in the automaton it lowers into: it has
    an event quantifier or count outside a static leaf, whose events (a closed part's) its own
    automaton reads."""
    if is_static(f):
        return False
    return isinstance(f, (C.Quant, C.Count)) or any(_temporal(p) for p in C.children(f))


def _hoist(formula: C.Formula) -> C.Formula:
    """Take every part of a quantifier or tools body that does not read the bound element out
    of that element's reach, bottom-up: a part that reads no element at all becomes one
    ``Closed`` atom where it stands (``_close``), and a part that reads an enclosing element
    moves out of the quantifier by Shannon expansion (``_expand``)."""
    match formula:
        case C.Not(body):
            return C.neg(_hoist(body))
        case C.And(parts):
            return C.conj(_hoist(p) for p in parts)
        case C.Or(parts):
            return C.disj(_hoist(p) for p in parts)
        case C.Quant(q, var, coll, anchor, body):
            body = _close(_hoist(body))

            def build(b: C.Formula) -> C.Formula:
                return C.quant(q, var, coll, anchor, b)

            # `all` distributes over `and` and `any` over `or`: each part is then a quantifier
            # of its own, whose closed parts move out without splitting the others.
            if q == "forall" and isinstance(body, C.And):
                return C.conj(_expand(p, var, build) for p in body.parts)
            if q == "exists" and isinstance(body, C.Or):
                return C.disj(_expand(p, var, build) for p in body.parts)
            return _expand(body, var, build)
        case C.Count(var, coll, body, op, n):
            return C.Count(var, coll, _hoist(body), op, n)
        case C.ChunkQuant(q, var, collection, body):
            body = _hoist(body)
            return _expand(
                body,
                var,
                lambda b: (
                    b
                    if (q == "exists" and b == C.FALSE) or (q == "forall" and b == C.TRUE)
                    else C.ChunkQuant(q, var, collection, b)
                ),
                keep_static=False,
            )
        case C.ToolsExists(var, body, q):
            body = _close(_hoist(body))
            return _expand(
                body,
                var,
                lambda b: (
                    b
                    if (q == "exists" and b == C.FALSE) or (q == "forall" and b == C.TRUE)
                    else replace(formula, body=b)
                ),
                keep_static=False,
            )
    return formula


def _close(body: C.Formula) -> C.Formula:
    """``body`` with the members of each ``And``/``Or`` in it that read no element gathered
    into one ``Closed`` part, or ``body`` closed whole when it reads none. Members that are
    all static leaves stay as they are (``is_static``): each already has one value per run.
    It walks through ``And``, ``Or`` and ``Not``; a quantifier inside closed its own body when
    it was hoisted."""
    if not C.free_vars(body):
        return body if is_static(body) or isinstance(body, C.Const) else _closed_part(body)
    match body:
        case C.Not(inner):
            return C.neg(_close(inner))
        case C.And(parts) | C.Or(parts):
            join = C.conj if isinstance(body, C.And) else C.disj
            reading = [_close(p) for p in parts if C.free_vars(p)]
            closed = [p for p in parts if not C.free_vars(p)]
            if closed and not all(is_static(p) for p in closed):
                closed = [_closed_part(join(closed))]
            return join([*reading, *closed])
    return body


def _closed_part(f: C.Formula) -> C.Formula:
    """``f`` as ``Closed`` atoms: one per temporal operand when it has two or more
    (``_split``), else one for the whole, a negated one as the negated atom of what it
    negates."""
    split = _split(f)
    if split is not f:
        return split
    return C.neg(C.Closed(f.body)) if isinstance(f, C.Not) else C.Closed(f)


def bind_candidate(
    formula: C.Formula, var: str
) -> tuple[C.Formula, C.Events, tuple[C.Formula, ...]]:
    """``formula`` with ``var`` decided one candidate event per run, ``var``'s collection, and
    the conjuncts of its body (lowering keeps, as the candidate's prefilter, those that are
    tests on its own event).

    The quantifier over ``var`` gains the conjunct that its event is the candidate's, so it
    ranges over that one event and distributes over the conjuncts of its body. It must be a
    positive existential (reached through ``and``, ``or`` and existential bodies only), where
    running per candidate and OR-ing the runs is exact. Its element's own event becomes a test (``Same(·, var)``), a join to its field a
    parameter, and an event before it whose condition relates later events is related to the
    candidate instead of to ``var``, which makes it closed and hoistable.

    Every candidate atom sits inside what replaces the quantifier, and each own-step conjunct
    is conjoined there with the candidate's event, so a candidate whose event fails one makes
    that replacement false: its run is the run with no candidate bound."""
    found: list[C.Events] = []
    own: list[C.Formula] = []

    def positive(f: C.Formula) -> C.Formula:
        match f:
            case C.And(parts):
                return C.And(tuple(positive(p) for p in parts))
            case C.Or(parts):
                return C.Or(tuple(positive(p) for p in parts))
            case C.Quant("exists", v, C.Events() as coll, anchor, body) if v == var:
                found.append(coll)
                # One event is the candidate, so the quantifier distributes over the body's
                # conjuncts; each is then decided (and hoisted) on its own.
                parts = body.parts if isinstance(body, C.And) else (body,)
                own.extend(parts)
                at = C.AtCandidate(v)
                return C.conj(
                    [C.Quant("exists", v, coll, anchor, at)]
                    + [C.Quant("exists", v, coll, anchor, C.conj([at, p])) for p in parts]
                )
            case C.Quant("exists", v, coll, anchor, body):
                return C.Quant("exists", v, coll, anchor, positive(body))
        return f

    bound = positive(formula)
    if len(found) != 1:
        raise UnsupportedConstruct(
            f"{described(var)} would be decided one candidate at a time, which needs a single "
            "positive `any` over it, reached through `and`, `or` and `any` only"
        )
    return _rebind(bound, var), found[0], tuple(own)


def _lookup(scope: tuple[_Binding, ...], name: str) -> _Binding:
    """The innermost binder of ``name`` in ``scope``."""
    return next(b for b in reversed(scope) if b.name == name)


def source_name(var: str) -> str:
    """The rule's own name for the binder ``var`` (a unique variant drops its suffix)."""
    return var if var.startswith("\x00") else var.split("\x00")[0]


def described(var: str) -> str:
    """``var`` as a diagnostic names it: a fresh variable stands for a flow operand's event."""
    if var.startswith("\x00"):
        return "an event of a flow operand"
    return f"`{source_name(var)}`"


def _rebind(f: C.Formula, var: str) -> C.Formula:
    match f:
        case C.Same(a, of_var) if of_var == var:
            return C.AtCandidate(a)
        case C.Quant(q, y, coll, C.Rel(of_var, "before", lo, hi), body) if (
            of_var == var and not _local(body, y)
        ):
            # `y` before the candidate, related to it from `y`'s side: closed in `var`.
            z = f"\x00at{y}"
            later = C.Quant("exists", z, C.AllEvents(), C.Rel(y, "after", lo, hi), C.AtCandidate(z))
            inner = _rebind(body, var)
            body = C.conj([later, inner]) if q == "exists" else C.disj([C.neg(later), inner])
            return C.Quant(q, y, coll, None, body)
        case C.Not(body):
            return C.Not(_rebind(body, var))
        case C.And(parts):
            return C.And(tuple(_rebind(p, var) for p in parts))
        case C.Or(parts):
            return C.Or(tuple(_rebind(p, var) for p in parts))
        case C.Quant(q, v, coll, anchor, body):
            return C.Quant(q, v, coll, anchor, _rebind(body, var))
        case C.Count(v, coll, body, op, n):
            return C.Count(v, coll, _rebind(body, var), op, n)
    return f


def is_static(f: C.Formula) -> bool:
    """A closed leaf read off static atoms (a size, a tools or chunk test, a static
    predicate, a closed part): it has one value at every step, so it lowers where it
    stands."""
    return isinstance(f, (C.Size, C.ToolsExists, C.ChunkQuant, C.Closed)) or (
        isinstance(f, C.Pred) and f.var is None
    )


def _local(f: C.Formula, var: str) -> bool:
    match f:
        case C.Const():
            return True
        case C.Join(test_var=test_var, bound_var=bound_var):
            return test_var == var and bound_var == var
        case C.Pred(var=v) | C.Member(var=v) | C.AtCandidate(var=v):
            return v == var
        case C.Not(inner):
            return _local(inner, var)
        case C.And(parts) | C.Or(parts):
            return all(_local(p, var) for p in parts)
    return is_static(f)


def _expand(body: C.Formula, var: str, build, keep_static: bool = True) -> C.Formula:
    """``build(body)`` with the closed groups of ``body`` moved out by Shannon expansion, one
    group at a time, found again in each branch so that none is left behind; with
    ``keep_static``, groups of static leaves stay (``is_static``). At most
    ``_MAX_HOISTED_PARTS`` groups move out, and each step is charged to the compile's work
    budget by the size of the body it copies."""

    def go(b: C.Formula, depth: int) -> C.Formula:
        # A step copies and walks the whole body: charge it by the body's size.
        charge(_size(b))
        found = _groups(b, var, keep_static)
        if not found:
            return build(b)
        if depth == _MAX_HOISTED_PARTS:
            _too_many(var)
        group = found[0]
        holds = go(_substitute(b, group, C.TRUE), depth + 1)
        fails = go(_substitute(b, group, C.FALSE), depth + 1)
        polarities = _polarities(b, group, True)
        # Where the group occurs with one polarity the formula is monotone in it, and the
        # branch that cannot be the weaker one needs no test of the group.
        if polarities == {True}:
            return C.disj([C.conj([group.formula, holds]), fails])
        if polarities == {False}:
            return C.disj([holds, C.conj([C.neg(group.formula), fails])])
        return C.disj([C.conj([group.formula, holds]), C.conj([C.neg(group.formula), fails])])

    return go(body, 0)


def _size(f: C.Formula) -> int:
    """The number of core nodes in ``f``."""
    return 1 + sum(_size(child) for child in C.children(f))


def _too_many(var: str) -> None:
    raise UnsupportedConstruct(
        f"a comprehension body with more than {_MAX_HOISTED_PARTS} conditions that do not "
        f"read {described(var)} is not supported; move them outside the comprehension"
    )


@dataclass(frozen=True)
class _Group:
    """The closed siblings ``members`` of one ``And`` (``conjunctive``) or ``Or`` node, as one
    condition on the conversation; ``keys`` identify them up to bound-variable names."""

    members: tuple[C.Formula, ...]
    conjunctive: bool
    keys: frozenset[str]

    @property
    def formula(self) -> C.Formula:
        """Return the formula of the group."""
        return C.conj(self.members) if self.conjunctive else C.disj(self.members)


def _groups(body: C.Formula, var: str, keep_static: bool) -> list[_Group]:
    """The distinct closed groups of ``body`` that must move out of ``var``'s quantifier: with
    ``keep_static``, not a group of static leaves only."""
    found: list[_Group] = []

    def add(members: list[C.Formula], conjunctive: bool) -> None:
        # A closed part is one static atom where it stands, whatever the keep_static mode.
        members = [m for m in members if not isinstance(_unnegated(m), C.Closed)]
        if not members:
            return
        if len(members) == 1 and isinstance(members[0], C.Not):
            add([members[0].body], conjunctive)
            return
        if keep_static and all(is_static(m) for m in members):
            return
        group = _Group(tuple(members), conjunctive, frozenset(_key(m) for m in members))
        if all(g.keys != group.keys for g in found):
            found.append(group)

    def walk(f: C.Formula) -> None:
        match f:
            case C.Const():
                return
            case C.Not(inner):
                walk(inner)
                return
            case C.And(parts) | C.Or(parts):
                closed = [p for p in parts if var not in C.free_vars(p)]
                if closed:
                    add(closed, isinstance(f, C.And))
                for p in parts:
                    if var in C.free_vars(p):
                        walk(p)
                return
        if var not in C.free_vars(f):
            add([f], True)

    walk(body)
    return found


def _unnegated(f: C.Formula) -> C.Formula:
    return f.body if isinstance(f, C.Not) else f


def _substitute(formula: C.Formula, group: _Group, value: C.Const) -> C.Formula:
    """``formula`` with every occurrence of ``group`` replaced by ``value``: a node of the
    group's kind holding all its members, or its one member anywhere."""
    if len(group.members) == 1 and _may_be(formula, group) and _key(formula) in group.keys:
        return value
    match formula:
        case C.And(parts) | C.Or(parts):
            parts = tuple(_substitute(p, group, value) for p in parts)
            if len(group.members) > 1 and isinstance(formula, C.And) == group.conjunctive:
                keys = [_key(p) for p in parts]
                if group.keys <= set(keys):
                    parts = (
                        *(p for p, k in zip(parts, keys, strict=True) if k not in group.keys),
                        value,
                    )
            return C.conj(parts) if isinstance(formula, C.And) else C.disj(parts)
        case C.Not(body):
            return C.neg(_substitute(body, group, value))
    return formula


def _polarities(formula: C.Formula, group: _Group, positive: bool) -> set[bool]:
    """The polarities at which ``group`` occurs in ``formula`` (as ``_substitute`` finds it);
    both inside a count, which is not monotone in its body."""
    if len(group.members) == 1 and _may_be(formula, group) and _key(formula) in group.keys:
        return {positive}
    match formula:
        case C.Not(body):
            return _polarities(body, group, not positive)
        case C.And(parts) | C.Or(parts):
            found: set[bool] = set()
            if (
                len(group.members) > 1
                and isinstance(formula, C.And) == group.conjunctive
                and group.keys <= {_key(p) for p in parts}
            ):
                found.add(positive)
            for p in parts:
                found |= _polarities(p, group, positive)
            return found
        case C.Quant(body=body) | C.ToolsExists(body=body):
            return _polarities(body, group, positive)
        case C.ChunkQuant(body=body) | C.Closed(body):
            return _polarities(body, group, positive)
        case C.Count(body=body):
            return {True, False} if _polarities(body, group, positive) else set()
    return set()


def _key(f: C.Formula) -> str:
    """``f`` identified up to the names of the variables it binds (fresh names differ between
    two copies of one condition). Expansion asks again for the same parts in every branch, so
    each object's key is kept for the ``hoist`` call (``_keys``)."""
    keys = _keys.get()
    known = keys.get(id(f)) if keys is not None else None
    if known is not None and known[0] is f:
        return known[1]
    key = repr(_canonical(f, 0))
    if keys is not None:
        keys[id(f)] = (f, key)
    return key


# ``_key`` of each part seen during the ``hoist`` call running in this context, by object (the
# part is kept alive with its key, so its id is not reused).
_keys = ContextVar[dict[int, tuple[C.Formula, str]] | None]("quarry_hoist_keys", default=None)


def _may_be(formula: C.Formula, group: _Group) -> bool:
    """Whether ``formula`` has the constructor of one of ``group``'s members: only then can it
    be one of them, so only then is its key worth computing."""
    return any(type(formula) is type(m) for m in group.members)


def _canonical(f: C.Formula, level: int) -> C.Formula:
    charge(1)
    name = f"\x00b{level}"
    match f:
        case C.Quant(q, var, coll, anchor, body):
            return C.Quant(q, name, coll, anchor, _canonical(rename(body, var, name), level + 1))
        case C.Count(var, coll, body, op, n):
            return C.Count(name, coll, _canonical(rename(body, var, name), level + 1), op, n)
        case C.ToolsExists(var, body, q):
            return C.ToolsExists(name, _canonical(rename(body, var, name), level + 1), q)
        case C.ChunkQuant(q, var, collection, body):
            return C.ChunkQuant(q, name, collection, _canonical(rename(body, var, name), level + 1))
        case C.Not(body):
            return C.Not(_canonical(body, level))
        case C.Closed(body):
            return C.Closed(_canonical(body, level))
        case C.And(parts):
            return C.And(tuple(_canonical(p, level) for p in parts))
        case C.Or(parts):
            return C.Or(tuple(_canonical(p, level) for p in parts))
    return f
