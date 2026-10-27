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

"""The atom factory and leaf compilation for rules; ``compile_rule`` is the entry point.

``compile_rule`` compiles a parsed ``DSLRule`` through the core calculus
(``quarry.engine.desugar`` desugars it, ``quarry.engine.lower`` lowers each constructor into
LTLf with past operators, and ``quarry.engine.symbolic_dfa`` builds the DFA). This module
owns what those passes ask of the rule's syntax: resolving collections and accessors,
minting interned atoms for a leaf test at one event (``compile_step``), and compiling
closed conditions that read no element (``compile_closed``: chunked scoring, model checks,
system-prompt tests, top-level functions). It, ``desugar`` and ``lower`` are the engine
modules that read the rule AST (``quarry.types.dsl``); none may import the scanner or the DSL
layer (``quarry.scanner`` / ``quarry.dsl.*``), enforced by ``tests/engine/test_async_runtime.py``.

A rule's condition describes an *unsafe* pattern; the artifact's formula is its
**safety** negation, so ``run_trace(dfa, ...) == True`` means "did not match".
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping, Set
from dataclasses import dataclass, field, replace
from itertools import combinations
from math import comb
from typing import Annotated, TypeVar, cast
from typing import Literal as TypeLiteral

from quarry.engine import temporal as T
from quarry.engine.atoms import (
    POSITION_FIELD_SEP,
    POSITION_GROUP_SEP,
    AtomDomain,
    AtomInterner,
    Cost,
    ModelExpression,
    StringSemantics,
    make_atom,
)
from quarry.engine.budget import DEFAULT_MAX_WORK, charge
from quarry.engine.collect import CollectPlan
from quarry.engine.contracts import CompiledRuleArtifact, ToolQuantifierPlan
from quarry.engine.diagnostics import (
    Diagnostics,
    ErrorCategory,
    SourceRef,
    UnsupportedConstruct,
)
from quarry.language.metadata import TYPE_CONVERSIONS, DSLParameter, Use, dsl_compiler_form
from quarry.language.runtime import (
    LanguageSpecError,
    accessor_functions,
    external_function_costs,
    function_by_name,
    function_signatures,
    functions_with_tag,
    role_view_surface_names,
    surface_names,
)
from quarry.rule_versioning import check_rule_compatibility

# Adapter/compiler tier: importing the rule AST is allowed and expected here.
from quarry.types.dsl import (
    AllComprehension,
    AnyComprehension,
    Assignment,
    BinaryOp,
    ConditionNode,
    ContainsAnyOp,
    FieldAccess,
    FilterComprehension,
    FunctionCall,
    Identifier,
    Literal,
    MapComprehension,
    ModelCall,
    ParameterRef,
    PatternMatch,
    UnaryOp,
    condition_child_fields,
    iter_condition_children,
    walk_condition_nodes,
)
from quarry.types.dsl.operators.flow import FlowOp
from quarry.types.dsl.operators.string import (
    StringCount,
    StringPosition,
    StringSetQuantifier,
)
from quarry.types.response_schema import response_schema_types

_T = TypeVar("_T")


class _LazySet(Set[str]):
    """Set projection whose catalog is not read until the compiler actually uses it."""

    def __init__(self, load: Callable[[], Set[str] | frozenset[str]]) -> None:
        self._load = load

    def _value(self) -> frozenset[str]:
        return frozenset(self._load())

    def __contains__(self, value: object) -> bool:
        return value in self._value()

    def __iter__(self) -> Iterator[str]:
        return iter(self._value())

    def __len__(self) -> int:
        return len(self._value())


class _LazyMapping(Mapping[str, _T]):
    """Mapping projection whose catalog is not read until the compiler actually uses it."""

    def __init__(self, load: Callable[[], Mapping[str, _T]]) -> None:
        self._load = load

    def _value(self) -> Mapping[str, _T]:
        return self._load()

    def __getitem__(self, key: str) -> _T:
        return self._value()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._value())

    def __len__(self) -> int:
        return len(self._value())


def _one_function_with_tag(tag: str) -> str:
    names = functions_with_tag(tag)
    if len(names) != 1:
        raise LanguageSpecError(
            f"Quarry's language catalog must declare exactly one {tag!r} function; "
            f"found {len(names)}"
        )
    return next(iter(names))


def _complete_func() -> str:
    return _one_function_with_tag("complete_comparison")


def _chunk_source_func() -> str:
    return _one_function_with_tag("chunk_source")


def _chunk_scalar_shapes() -> dict[str, dict[str, int | bool]]:
    signatures = function_signatures()
    shapes: dict[str, dict[str, int | bool]] = {}
    for name in functions_with_tag("chunk_scalar"):
        item = function_by_name(name)
        if item is None:
            raise LanguageSpecError(f"Missing catalog record for chunk scalar {name!r}")
        model_inputs = [
            index
            for index, parameter in enumerate(item["parameters"])
            if "model_input" in parameter["accepted_types"]
        ]
        if len(model_inputs) != 1:
            raise LanguageSpecError(
                f"Chunk scalar {name!r} must have exactly one model_input parameter"
            )
        shapes[name] = {
            "arity": signatures[name][0],
            "chunk_arg": model_inputs[0],
            "constitution": any(
                parameter["name"] == "constitution" for parameter in item["parameters"]
            ),
        }
    return shapes


_EVENT_SURFACES = _LazySet(lambda: surface_names(domain="event"))
_STATIC_SURFACES = _LazySet(lambda: surface_names(domain="static"))
# Surfaces valid as a Messages-typed model-input selection (no tools). ``tool_calls`` is
# excluded — it is a collection of ToolCall objects, not a message transcript.
_MODEL_INPUT_SURFACES = _LazySet(lambda: surface_names(model_input=True))
_SURFACES = _LazySet(surface_names)

# Role-view accessors valid as a model-input selection filter step (everything in
# _MODEL_INPUT_SURFACES except the whole-transcript ``messages`` root).
_ROLE_VIEW_SURFACES = _LazySet(role_view_surface_names)

# Scoped collection accessors → (surface, optional name-filter field).
_ACCESSORS = _LazyMapping(accessor_functions)

_COMPARISONS = {"=", "!=", ">", "<", ">=", "<="}
_CONTENT_OPS = {"contains", "in", "~", "matches"}

_MODEL_FIELD_TYPES: dict[str, dict[str, str]] = {
    "AssistantMessage": {
        "role": "string",
        "content": "string",
        "refusal": "string?",
        "tool_calls": "ToolCall[]",
    },
    "ToolCall": {
        "id": "string",
        "type": "string",
        "name": "string",
        "raw_arguments": "string",
        "arguments": "object",
        "message_index": "int",
        "function": "ToolCallFunction",
    },
    "ToolCallFunction": {"name": "string", "arguments": "string"},
    # `tool_calls` reads as missing on a message that is not an assistant message.
    "Message": {
        "role": "string",
        "content": "string",
        "message_index": "int",
        "tool_calls": "ToolCall[]",
    },
    "Score": {"value": "float"},
    "Span": {
        "message_index": "int",
        "start": "int",
        "end": "int",
        "score": "Score",
    },
}

# The model-expression type of one element of a bound surface.
_SURFACE_ELEMENT_TYPES: dict[str, str] = {
    "model_chunks": "Messages",
    "messages": "Message",
    "system_messages": "Message",
    "developer_messages": "Message",
    "user_messages": "Message",
    "assistant_messages": "AssistantMessage",
    "tool_calls": "ToolCall",
}

# Fields whose value, read as text (``message()`` content), is another field's text: a tool
# call's decoded ``arguments`` object is the JSON argument text the call carries.
_MODEL_TEXT_FIELDS: dict[str, dict[str, str]] = {"ToolCall": {"arguments": "raw_arguments"}}

# The scope surface of a map comprehension item, which is not a conversation surface.
_MAP_ITEM_SURFACE = "map_items"
# The types of a map comprehension item field that a text test reads as a rule step reads it.
_STEP_TEXT_TYPES = frozenset({"object", "object[]", "string", "string?"})

# `join()` makes the text of a model input from a map comprehension. The compiler rejects every
# other use of `join()` and of a map comprehension with one of these messages.
_JOIN_FORM = (
    "join() is supported only as message() or add_message() content in a model input, e.g. "
    'message("user", join([c.name for c in messages[-1].tool_calls], "\\n"))'
)
_MAP_COMPREHENSION_FORM = "a map comprehension is supported only as the first argument of join()"
_LEGACY_COMPUTED_CONTENT = (
    "message() content computed with join() is supported only in a version 2 model expression, "
    'such as model("alias").score(...) or model("alias").complete(...), with config_version: 2'
)

# Resource-backed functions and their compiler cost. Allowed-use checks decide which of these
# may become a bare Boolean atom; comparison-only and collection-only forms take dedicated paths.
_EXTERNAL_FUNCS = _LazyMapping(
    lambda: {name: Cost(cost) for name, cost in external_function_costs().items()}
)

# Inference scorers whose config is ``(model[, constitution], conversation())`` and which are
# grounded once over the whole conversation (STATIC), reading ``ctx.conversation`` in the
# detector rather than a per-event field. ``llm_score`` also folds a second literal (the
# constitution) onto the atom.
_INFERENCE_SCORE_EXTERNAL = _LazySet(lambda: functions_with_tag("whole_conversation_inference"))

# String-valued whole-conversation inference external: ``llm_complete(model, conversation())``
# generates a completion (a string), so a rule uses it ONLY through a folded comparison —
# ``llm_complete(...) = "text"`` (exact) or ``llm_complete(...) ~ "regex"`` (match) — which the
# compiler lowers to one boolean STATIC atom (detector ``llm_complete``, the literal folded onto
# ``patterns`` and the operator onto ``op``). Its empty ``allowed_uses`` declaration rejects a
# bare call before the ordinary external path can treat it as a truthy value.
# ``llm_split_conversation`` is the comprehension SOURCE of the chunked-scoring shape. It
# returns a list of conversations, which no atom can bind, so it is never lowered on its own —
# it exists in a rule only as the collection of the shape below.
# The synthesized detector the shape desugars to. It is absent from the generated public
# catalog, so a rule cannot call it directly. Only this comprehension rewrite can create it.
_CHUNK_REDUCE_DETECTOR = "llm_chunk_reduce"
# The whole-chunk SCALARS a chunked body may compare. To qualify, a function must map one entire
# chunk to one number with every other operand a compile-time literal — that is what lets the
# chunk variable be erased and the whole comprehension fold into a single atom. Adding another
# such function requires a ``chunk_scalar`` annotation plus a dispatch arm in
# ``llm_chunk_reduce``.
#   arity       — exact DSL arg count
#   chunk_arg   — which argument must be the chunk variable (all others must be literals)
#   constitution— whether arg[1] is a constitution/text_ref that folds onto the atom
_CHUNK_SCALARS = _LazyMapping(_chunk_scalar_shapes)
# Cardinality form: `count([c for c in llm_split_conversation(...) if <scalar> <op> t]) >= N`.
# ``len`` is accepted as a synonym since both are the length of the filtered list.
_CHUNK_COUNT_FUNCS = _LazySet(lambda: functions_with_tag("chunk_count"))
# Which reduction makes the quantifier-elimination EXACT, per (comparator, quantifier). The
# reduction has to agree with the direction the comparator is monotone in, so it is decided by
# BOTH — `any` is not always ``max``:
#
#     ∃c. s(c) > t ≡ max{s} > t     ∀c. s(c) > t ≡ min{s} > t
#     ∃c. s(c) < t ≡ min{s} < t     ∀c. s(c) < t ≡ max{s} < t
#
# ``=`` / ``!=`` are deliberately absent: they are not threshold tests, so no single max/min
# reduction can express them and the desugar must reject them instead of guessing.
_CHUNK_COMBINE = {
    ">": {"any": "max", "all": "min"},
    ">=": {"any": "max", "all": "min"},
    "<": {"any": "min", "all": "max"},
    "<=": {"any": "min", "all": "max"},
}

# Probers whose *first* argument is the scanned field (no literal config arg), unlike
# ``model_score(model, field)`` where arg[0] is config. ``topic_drift(msg.content)`` /
# ``off_topic(msg.content)`` / ``unrequested_action(call.name)`` — the intent baseline is
# read from the conversation, not passed.
_FIELD_FIRST_EXTERNAL = _LazySet(lambda: functions_with_tag("field_first_external"))

# Score-returning externals whose comparison (`f(...) <op> n`) folds into the atom.
_SCORE_EXTERNAL = _LazySet(lambda: functions_with_tag("score_external"))


@dataclass
class Scope:
    """Comprehension element variables → their surface; captured names in scope."""

    element_surfaces: dict[str, str] = field(default_factory=dict)
    element_types: dict[str, str] = field(default_factory=dict)
    captures: set[str] = field(default_factory=set)
    parent: Scope | None = None

    def bind(self, var: str, surface: str, element_type: str | None = None) -> Scope:
        """Bind the given variable to the given surface and element type."""
        s = Scope(parent=self)
        s.element_surfaces[var] = surface
        if element_type is not None:
            s.element_types[var] = element_type
        return s

    def resolve_element(self, name: str) -> str | None:
        """Resolve the element name to its surface."""
        if name in self.element_surfaces:
            return self.element_surfaces[name]
        return self.parent.resolve_element(name) if self.parent else None

    def resolve_element_type(self, name: str) -> str | None:
        """Resolve the element name to its type."""
        if name in self.element_types:
            return self.element_types[name]
        return self.parent.resolve_element_type(name) if self.parent else None


@dataclass
class CompileResult:
    """Outcome of compiling one rule. ``artifact`` is None if any error occurred."""

    rule_name: str
    artifact: CompiledRuleArtifact | None
    diagnostics: Diagnostics

    @property
    def ok(self) -> bool:
        """Return True if the compile result is ok."""
        return self.artifact is not None and not self.diagnostics.has_errors


class _Compiler:
    def __init__(self, rule, response_schemas: Mapping[str, str] | None = None):
        self.rule = rule
        self.rule_name = getattr(rule, "name", "")
        # An alias with a response schema completes to a message whose `.json` the schema
        # types, so a rule reading a field the schema lacks fails at load.
        self._field_types = dict(_MODEL_FIELD_TYPES)
        self._completion_types: dict[str, str] = {}
        for alias, schema_text in (response_schemas or {}).items():
            root, objects = response_schema_types(alias, schema_text)
            completion = f"completion of model {alias!r}"
            self._field_types[completion] = {**_MODEL_FIELD_TYPES["AssistantMessage"], "json": root}
            self._field_types.update(objects)
            self._completion_types[alias] = completion
        self.strings = getattr(rule, "strings", None) or {}
        self.parameters = {spec.name: spec for spec in getattr(rule, "parameters", ())}
        self.used_parameters: set[str] = set()
        self.atoms = AtomInterner()
        # Provenance is recorded against authored object occurrences while lowering runs. Keep
        # their original preorder positions so compiler-only normalization can never change the
        # source graph exposed to callers.
        self._source_node_occurrence_counts: dict[int, int] = {}
        self._source_node_indices: dict[tuple[int, int], int] = {}
        for node_index, node in enumerate(walk_condition_nodes(getattr(rule, "condition", None))):
            node_id = id(node)
            occurrence = self._source_node_occurrence_counts.get(node_id, 0)
            self._source_node_indices[(node_id, occurrence)] = node_index
            self._source_node_occurrence_counts[node_id] = occurrence + 1
        self._ast_node_visit_counts: dict[int, int] = {}
        # id(authored node) -> its authored parent, for attributing a leaf's atoms to every
        # enclosing section of the rule (see activate_source).
        self._authored_parents: dict[int, ConditionNode] = {}
        for parent in walk_condition_nodes(getattr(rule, "condition", None)):
            for field_name in condition_child_fields(parent):
                value = getattr(parent, field_name)
                for child in value if isinstance(value, list | tuple) else (value,):
                    if isinstance(child, ConditionNode):
                        self._authored_parents.setdefault(id(child), parent)
        # The authored nodes each rewritten node stands for (see alias_rewritten_nodes).
        self._node_aliases = _NodeAliases()
        self._active_node_occurrences: list[tuple[int, int]] = []
        self._node_atom_ids: dict[tuple[int, int], set[str]] = {
            (node_id, occurrence): set()
            for node_id, count in self._source_node_occurrence_counts.items()
            for occurrence in range(count)
        }
        self._atom_source_events: list[tuple[str, int | None]] = []
        self.diagnostics = Diagnostics()
        self.temporal = False
        self.has_static = False
        self.tool_quantifiers: dict[str, ToolQuantifierPlan] = {}
        self.notes: list[str] = []
        # Active surface window for a sliced comprehension (``any m in messages[-4:] …``):
        # (surface, start, stop). While set, EVENT atoms minted on ``surface`` are restricted
        # to the [start:stop) window (resolved against the surface size at grounding). Scoped
        # by save/restore around each comprehension body so nested comprehensions don't leak.
        # The position EVENT atoms minted on a surface are restricted to: (surface, slice
        # window, fixed index), set by ``_in_window``.
        self._active_window: tuple[str, tuple[int | None, int | None] | None, int | None] | None = (
            None
        )
        self._n = 0
        # Value capture. While a capture var is active, the bind site mints a
        # `capture_bind` atom and each reference site a `capture_ref` atom (both with an
        # empty pattern placeholder the executor substitutes per candidate value).
        # A captured value may be referenced at MORE THAN ONE site (e.g. the token
        # appears in both a tool response AND a tool-call argument). Each distinct reference
        # mints its own parameterized atom, and the executor must bind the candidate value
        # into ALL of them — so we collect every ref (and bind) atom id, not just the last.
        # ``_capture_ref_id``/``_capture_bind_id`` keep the FIRST id (evidence + back-compat);
        # the executor grounds against the full ``*_ids`` sets.
        self._capture_var = ""
        self._capture_bind_id = ""
        self._capture_ref_id = ""
        # Join atoms on an enclosing element's field, waiting for that element's step:
        # (element variable, placeholder atom name left at the join site, the atom).
        self._capture_bind_ids: set[str] = set()
        self._capture_ref_ids: set[str] = set()

    @contextlib.contextmanager
    def _activate_ast_node(self, node):
        """Attribute atoms minted by a lowering helper to the authored AST occurrences that
        ``node`` is or stands for."""

        with contextlib.ExitStack() as stack:
            for authored in self._node_aliases.authored(node):
                stack.enter_context(self._activate_authored_node(authored))
            yield

    @contextlib.contextmanager
    def activate_source(self, node):
        """Attribute atoms minted while lowering a leaf of the core to its authored syntax and
        every authored node enclosing it, as a lowering that walks the rule would."""

        with contextlib.ExitStack() as stack:
            for authored in self._node_aliases.authored(node) if node is not None else ():
                chain = [authored]
                while id(chain[-1]) in self._authored_parents:
                    chain.append(self._authored_parents[id(chain[-1])])
                for enclosing in reversed(chain):
                    stack.enter_context(self._activate_authored_node(enclosing))
            yield

    @contextlib.contextmanager
    def _activate_authored_node(self, node):
        """Attribute atoms minted by a lowering helper to one authored AST occurrence."""

        node_id = id(node)
        occurrence_count = self._source_node_occurrence_counts.get(node_id, 0)
        if occurrence_count == 0:
            yield
            return
        visit = self._ast_node_visit_counts.get(node_id, 0)
        self._ast_node_visit_counts[node_id] = visit + 1
        occurrence = visit if visit < occurrence_count else occurrence_count - 1
        self._active_node_occurrences.append((node_id, occurrence))
        try:
            yield
        finally:
            self._active_node_occurrences.pop()

    def _intern_atom(self, spec):
        atom = self.atoms.intern(self._apply_window(spec))
        # Attribute a lowered atom to every active authored ancestor. This makes a
        # visualization node represent the coverage of its whole rule section, while the
        # atom remains available for precise leaf-level inspection.
        for occurrence in self._active_node_occurrences:
            self._node_atom_ids[occurrence].add(atom.atom_id)
        source = (
            self._source_node_indices.get(self._active_node_occurrences[-1])
            if self._active_node_occurrences
            else None
        )
        self._atom_source_events.append((atom.atom_id, source))
        return atom

    def final_node_atom_ids(self, retained_atom_ids: set[str]) -> dict[int, tuple[str, ...]]:
        """Resolve stable authored occurrences to their original preorder positions."""

        result: dict[int, tuple[str, ...]] = {}
        for occurrence, node_index in self._source_node_indices.items():
            atom_ids = self._node_atom_ids.get(occurrence, set())
            retained = tuple(
                sorted(atom_id for atom_id in atom_ids if atom_id in retained_atom_ids)
            )
            if retained:
                result[node_index] = retained
        return result

    def alias_rewritten_nodes(self, aliases: _NodeAliases) -> None:
        """Attribute nodes created by an AST rewrite to the authored nodes they replace."""
        self._node_aliases = aliases

    def _tool_quantifier_atom(
        self,
        body: T.Formula,
        *,
        kind: TypeLiteral["any", "all"] = "any",
        source_start: int,
    ) -> T.Formula:
        """Fold one tools comprehension into a static, binding-coherent question."""

        def boolean_only(formula: T.Formula) -> bool:
            match formula:
                case T.TrueConst() | T.FalseConst() | T.Atom():
                    return True
                case T.Not(child):
                    return boolean_only(child)
                case T.And(left, right) | T.Or(left, right) | T.Implies(left, right):
                    return boolean_only(left) and boolean_only(right)
                case _:
                    return False

        # Every part of the body that does not read the tool is hoisted out of it before
        # lowering, so what is left must be a Boolean test of the tool definition. Running
        # the whole rule once per tool instead would be wrong under `not` and beside a
        # second tools quantifier.
        if not boolean_only(body):
            if kind == "all":
                raise UnsupportedConstruct("all tools requires a Boolean body over the bound tool")
            raise UnsupportedConstruct(
                "a comprehension over tools can only test the tool definition itself"
            )
        body_atoms = T.atoms_of(body)
        specs_by_id = {spec.atom_id: spec for spec in self.atoms.specs}
        if any(atom_id not in specs_by_id for atom_id in body_atoms):
            # A join to an enclosing element waits for that element's step, which a question
            # answered once over the tool definitions does not have.
            raise UnsupportedConstruct(
                "a comprehension over tools cannot relate a tool to an enclosing element"
            )
        if any(specs_by_id[atom_id].predicate == "tool_quantifier" for atom_id in body_atoms):
            if kind == "all":
                raise UnsupportedConstruct(
                    "all tools with a nested tools comprehension is not supported"
                )
            # The inner quantifier is an independent static question, but the outer
            # existential still requires at least one tool to bind.
            nonempty = self._intern_atom(
                make_atom(
                    domain=AtomDomain.STATIC,
                    selector="tools",
                    value_expr_key="",
                    predicate="surface_count",
                    patterns=(),
                    op=">",
                    number=0.0,
                    cost=Cost.PURE,
                )
            )
            return T.And(T.Atom(nonempty.atom_id), body)
        # A test of the tool reads the tool definitions; a closed part of the body is a
        # static atom with one value per run.
        if any(specs_by_id[atom_id].domain is not AtomDomain.STATIC for atom_id in body_atoms):
            if kind == "all":
                raise UnsupportedConstruct("all tools requires predicates over the bound tool")
            raise UnsupportedConstruct(
                "a comprehension over tools can only test the tool definition itself"
            )
        cost = max(
            (specs_by_id[atom_id].cost for atom_id in body_atoms),
            key=lambda item: item.rank,
            default=Cost.PURE,
        )
        spec = self._intern_atom(
            make_atom(
                domain=AtomDomain.STATIC,
                selector="tools",
                value_expr_key=f"quantifier:{kind}:{T.canonical(body)}",
                predicate="tool_quantifier",
                patterns=(),
                cost=cost,
            )
        )
        sources_by_atom: dict[str, list[int | None]] = {}
        for body_atom_id, source in self._atom_source_events[source_start:-1]:
            if body_atom_id in body_atoms:
                sources_by_atom.setdefault(body_atom_id, []).append(source)
        for sources in sources_by_atom.values():
            sources.sort(key=lambda source: source if source is not None else -1)
        source_cursor: dict[str, int] = {}
        body_sources: list[int | None] = []

        def assign_sources(formula: T.Formula) -> None:
            match formula:
                case T.Atom(name):
                    cursor = source_cursor.get(name, 0)
                    candidates = sources_by_atom.get(name, ())
                    body_sources.append(candidates[cursor] if cursor < len(candidates) else None)
                    source_cursor[name] = cursor + 1
                case T.Not(child):
                    assign_sources(child)
                case T.And(left, right) | T.Or(left, right) | T.Implies(left, right):
                    assign_sources(left)
                    assign_sources(right)

        assign_sources(body)
        if spec.atom_id in self.tool_quantifiers:
            # The DFA interns identical whole quantifiers too. Its single boot
            # observation cannot identify which authored quantifier occurrence it used.
            body_sources = [None] * len(body_sources)
        self.tool_quantifiers[spec.atom_id] = ToolQuantifierPlan(body, kind, tuple(body_sources))
        return T.Atom(spec.atom_id)

    def compile_closed(self, node) -> T.Formula:
        """A closed condition, one that reads no element in scope (the core's ``Static`` and
        ``ChunkQuant`` leaves): a chunked-scoring comprehension or count, a model check, a
        system-prompt test, a collection-count macro or a top-level function."""
        scope = Scope()
        if not self._contains_model_call(node):
            self._reject_model_text_forms(node)
        # Chunked scoring, both forms. The collection is a function result rather than a
        # timeline surface. Each desugars to ONE folded atom.
        #   quantifier:  any/all <c> in llm_split_conversation(...) where <scalar>(…, <c>) <op> n
        #   cardinality: count([<c> for <c> in llm_split_conversation(...) if …]) <op> N
        if (
            isinstance(node, AnyComprehension)
            and isinstance(node.collection, FunctionCall)
            and node.collection.name == "model_split"
        ):
            try:
                split_call = self._compile_transitional_model_split(*node.collection.args)
            except TypeError as exc:
                raise UnsupportedConstruct(
                    "model_split() takes a model alias, messages, token limit, and optional tools"
                ) from exc
            node = AnyComprehension(
                variable=node.variable,
                collection=split_call,
                condition=node.condition,
            )
        if isinstance(node, AnyComprehension) and isinstance(node.collection, ModelCall):
            model_split = self._compile_model_split(node, scope)
            if model_split is not None:
                return model_split
        if isinstance(node, (AnyComprehension, AllComprehension)):
            chunked = self._desugar_chunked_score(node, scope)
            if chunked is not None:
                return chunked
        if isinstance(node, BinaryOp) and node.operator in _COMPARISONS:
            chunked = self._desugar_chunked_count(node, scope)
            if chunked is not None:
                return chunked
        # `llm_complete(model, <input>) =/!=/~/matches "literal"` — generate a completion and
        # compare it. Folds into ONE boolean STATIC atom (detector reads ctx.conversation);
        # intercepted before the generic content/equality/comparison routes below, which do not
        # recognize a function-call subject.
        if isinstance(node, BinaryOp) and node.operator in ("=", "!=", "~", "matches"):
            complete = self._complete_comparison(node)
            if complete is not None:
                return complete
        if self._contains_model_call(node):
            return self._model_predicate_atom(node, scope)
        # `system_prompt contains/~ X` — a zero-width static content check on the
        # system prompt. Bare static atom (holds at boot, false on real events).
        if isinstance(node, BinaryOp) and node.operator in _CONTENT_OPS:
            root = self._content_root_name(node)
            if root == "system_prompt":
                self.has_static = True
                # A string-set quantifier over the system prompt (`N of ($a,$b) in
                # system_prompt`) isn't a plain content match, so `_static_system_prompt`
                # can't lower it — route it through the shared step path instead.
                if isinstance(node.left, StringSetQuantifier):
                    return self._static_surface_step(node, "system_prompt", scope)
                return self._static_system_prompt(node)
        if isinstance(node, BinaryOp) and node.operator in ("=", "!="):
            for a, b in ((node.left, node.right), (node.right, node.left)):
                if (
                    isinstance(a, Identifier)
                    and a.name == "system_prompt"
                    and isinstance(b, Literal)
                ):
                    self.has_static = True
                    atom = self._mint_atom(
                        "system_prompt",
                        "",
                        "equals",
                        (str(b.value if b.value is not None else ""),),
                        domain=AtomDomain.STATIC,
                    )
                    return atom if node.operator == "=" else T.Not(atom)
        # `count(surface) <op> n` — a collection-count macro (number of events on a
        # surface), lowered to a temporal AtLeast over a surface-membership atom.
        if isinstance(node, BinaryOp) and node.operator in _COMPARISONS:
            macro = self._count_collection_macro(node)
            if macro is not None:
                return macro
            # A top-level scored comparison whose model input is a constructed value with a
            # direct indexed message field is a per-event EVENT atom → Eventually-wrapped.
            dyn_surface = self._toplevel_dynamic_scored_surface(node)
            if dyn_surface is not None:
                self.temporal = True
                return T.Eventually(self._comparison_atom(node, dyn_surface))
            # A comparison whose subject is a STATIC surface (`model_score(…,
            # system_prompt) > n`, `count(system_prompt, "p") > n`, `@a in system_prompt >
            # n`, `#a in system_prompt > n`). These are zero-width static checks that
            # compile inside a comprehension; at top level they go through the same shared
            # step machinery, grounded at the boot step.
            static_surface = self._toplevel_static_surface(node)
            if static_surface is not None:
                self.has_static = True
                return self._static_surface_step(node, static_surface, scope)
        if isinstance(node, FunctionCall):
            return self._compile_toplevel_function(node, scope)
        unknown = _unknown_root_name(node, scope)
        if unknown is not None:
            raise UnsupportedConstruct(
                f"undefined reference {unknown!r} (not a field, element, or declared variable)"
            )
        raise UnsupportedConstruct(f"unsupported top-level condition {type(node).__name__}")

    @staticmethod
    def _scope_has_elements(scope: Scope) -> bool:
        """True if any enclosing scope binds a comprehension element (i.e. we are lowering
        inside another comprehension's body)."""
        s: Scope | None = scope
        while s is not None:
            if s.element_surfaces:
                return True
            s = s.parent
        return False

    def _desugar_chunked_score(self, node, scope: Scope) -> T.Formula | None:
        """Lower the chunked-scoring comprehension to ONE folded STATIC atom; return None when
        ``node`` is an ordinary comprehension over a timeline surface.

        Recognised shape ("split → score each chunk → combine")::

            any chunk in llm_split_conversation("m", conversation(), 200)
                where llm_score("m", text_ref("policy"), chunk) > 0.96

        The body may compare any whole-chunk scalar in ``_CHUNK_SCALARS``, not just ``llm_score``;
        see :meth:`_desugar_chunked_count` for the cardinality form.

        **Why a rewrite is needed at all.** The DFA grounds boolean atoms over an alphabet fixed
        at compile time, and a comprehension's quantifier is a *temporal* one: the automaton
        advances over timeline events, and the loop variable binds an event at a position. Chunks
        are neither — they are values the inference backend only produces mid-scan, in a count
        nobody knows until then, with no position on the timeline. So ordinary comprehension
        lowering rejects the collection, and rightly.

        **Why the rewrite is exact.** Over a FINITE chunk set the quantifier is just a reduction,
        which the *comparator* can absorb::

            ∃c. s(c) > t   ≡   max{s(c)} > t          (`any` → chunk_combine="max")
            ∀c. s(c) > t   ≡   min{s(c)} > t          (`all` → chunk_combine="min")

        So the iteration moves BELOW the atom boundary into the detector (ordinary host Python,
        which may loop and await freely), and the automaton receives the same single boolean leaf
        it already receives for a plain ``llm_score(...) > t``. Nothing is approximated and the
        compiled DFA is unchanged in shape — only the atom's interpretation is richer.

        **Why the shape is narrow.** Every operand except the chunk must be a compile-time literal
        so it can fold into the atom's identity, and the chunk variable must occur exactly once,
        in the scalar's designated chunk argument. Anything else is rejected loudly: a chunk has no
        timeline position, so there is no sound fallback lowering to degrade to.
        """
        coll = getattr(node, "collection", None)
        if not (isinstance(coll, FunctionCall) and coll.name == _chunk_source_func()):
            return None
        self._reject_nested_chunking(scope)
        var = str(getattr(node, "variable", "") or "")
        split_model, max_tokens, selector_extra = self._chunk_split_args(coll)
        scalar, op, number, extra = self._chunk_body(node.condition, var, split_model)
        extra = {**extra, **selector_extra}

        # Which reduction makes the rewrite exact depends on the quantifier AND the comparator's
        # direction — NOT on the quantifier alone. The threshold test is a Boolean lattice
        # homomorphism only when it is monotone in the same direction as the reduction:
        #
        #     ∃c. s(c) > t  ≡  max{s(c)} > t        ∃c. s(c) < t  ≡  min{s(c)} < t
        #     ∀c. s(c) > t  ≡  min{s(c)} > t        ∀c. s(c) < t  ≡  max{s(c)} < t
        #
        # So `any` with `<` needs MIN, not max. Deriving combine from `isinstance(node,
        # AnyComprehension)` alone silently computes the opposite quantifier.
        if op not in _CHUNK_COMBINE:
            # `=` / `!=` are not threshold tests: ∃c. s(c) = t asks whether SOME chunk hits a
            # value, which no single max/min can answer. Equality on a float score is a rule bug
            # regardless, so reject rather than pick a reduction that is merely plausible. The
            # cardinality form below is the way to ask a counting question.
            raise UnsupportedConstruct(
                f"chunked scoring needs an ordering comparison ({', '.join(sorted(_CHUNK_COMBINE))})"
                f", not {op!r} — a max/min reduction cannot express an equality over chunks"
            )
        combine = _CHUNK_COMBINE[op]["any" if isinstance(node, AnyComprehension) else "all"]
        return self._mint_chunk_atom(split_model, max_tokens, scalar, combine, op, number, extra)

    def _desugar_chunked_count(self, node: BinaryOp, scope: Scope) -> T.Formula | None:
        """Lower the CARDINALITY chunked form to one folded atom; return None when ``node`` is
        not a count over a chunk split.

        Recognised shape::

            count([c for c in llm_split_conversation("m", conversation(), 200)
                   if llm_score("m", "policy", c) > 0.9]) >= 2

        This is strictly more expressive than ``any``/``all``, which are its N>=1 and N=K special
        cases, and it is the form that actually carries signal: "two separate chunks look like
        injection" is much harder to trigger by accident than "some chunk does".

        It needs a reduction max/min cannot express, so the atom carries TWO comparisons — the
        inner per-chunk threshold (``chunk_inner_op``/``chunk_inner_number``, applied inside the
        detector) and the outer count comparison (the atom's own ``op``/``number``). Both fold at
        compile time, so this is still one boolean leaf and the DFA is still unchanged.
        """
        fc, number, flipped = self._split_scalar(node.left, node.right)
        if fc is None or number is None or fc.name not in _CHUNK_COUNT_FUNCS:
            return None
        if len(fc.args) != 1 or not isinstance(fc.args[0], FilterComprehension):
            return None
        inner = (
            self._bind_count_arguments(*fc.args)[0]
            if fc.name == "count"
            else self._bind_len_argument(*fc.args)
        )
        coll = inner.collection
        if not (isinstance(coll, FunctionCall) and coll.name == _chunk_source_func()):
            return None
        self._reject_nested_chunking(scope)
        # NOTE: `=` / `!=` are allowed on the OUTER comparison, unlike the max/min form. An exact
        # chunk count ("exactly one chunk trips") is a well-defined integer question; equality
        # against a float score is not.
        op = _FLIP.get(node.operator, node.operator) if flipped else node.operator
        var = str(inner.variable or "")
        split_model, max_tokens, selector_extra = self._chunk_split_args(coll)
        scalar, inner_op, inner_number, extra = self._chunk_body(inner.condition, var, split_model)
        extra = {**extra, **selector_extra}
        if inner_op not in _CHUNK_COMBINE:
            raise UnsupportedConstruct(
                "the per-chunk filter needs an ordering comparison "
                f"({', '.join(sorted(_CHUNK_COMBINE))}), not {inner_op!r}"
            )
        return self._mint_chunk_atom(
            split_model,
            max_tokens,
            scalar,
            "count",
            op,
            number,
            extra,
            inner_op=inner_op,
            inner_number=inner_number,
        )

    def _reject_nested_chunking(self, scope: Scope) -> None:
        """Chunked forms are whole-conversation questions, so they cannot sit inside another
        comprehension: the atom is STATIC and would be co-located at an EVENT step, where a
        STATIC atom grounds False — a silent never-match. Reject instead of mis-compiling."""
        if self._scope_has_elements(scope):
            raise UnsupportedConstruct(
                "chunked scoring is a whole-conversation question and cannot be nested inside "
                "another comprehension; write it as its own top-level term"
            )

    def _chunk_split_args(self, coll: FunctionCall) -> tuple[str, int, dict]:
        """Validate ``llm_split_conversation(model, <input>, max_input_tokens)`` and return the
        folded ``(model, budget, selector_extra)``. The source is the same typed model-input
        selector as every other op (default conversation() → whole transcript + tools; a
        Messages selection → that transcript, no tools), so chunked scoring splits exactly the
        SELECTED input, not always the whole request."""
        if len(coll.args) != 3:
            raise UnsupportedConstruct(
                f"{_chunk_source_func()}() takes (model, <input>, max_input_tokens)"
            )
        return self._compile_llm_split_conversation(*coll.args)

    @dsl_compiler_form(
        name="llm_split_conversation",
        return_type="collection<model_input>",
        allowed_uses=(Use.COMPREHENSION_SOURCE,),
        resource_group="token",
        compiler_tags=("chunk_source",),
    )
    def _compile_llm_split_conversation(
        self,
        model: Annotated[
            object,
            DSLParameter(accepted_types=("string",), literal_only=True),
        ],
        model_input: Annotated[
            object,
            DSLParameter(accepted_types=("model_input",), name="input"),
        ],
        max_input_tokens: Annotated[
            object,
            DSLParameter(accepted_types=("integer",), literal_only=True, minimum=1),
        ],
    ) -> tuple[str, int, dict]:
        """Split model input into chunks no larger than ``max_input_tokens``.

        Use this only after ``in`` in a documented ``any`` or ``all`` chunk check, or in a
        filtered chunk list passed to ``count`` or ``len``.

        Internal:

        The compiler folds this form into a chunk-reduction atom; it does not create a general
        runtime list.
        """
        split_model = self._lit(model)
        selector_extra = self._score_model_input(model_input)
        if "_dynamic_content_refs" in selector_extra:
            raise UnsupportedConstruct(
                f"{_chunk_source_func()}() does not support dynamic constructed content as a split "
                "source; use a static selection"
            )
        max_tokens = self._lit_index(max_input_tokens)
        if max_tokens <= 0:
            raise UnsupportedConstruct("max_input_tokens must be a positive integer literal")
        return split_model, max_tokens, selector_extra

    @dsl_compiler_form(
        name="count",
        return_type="integer",
        compiler_tags=("chunk_count",),
    )
    def _bind_count_arguments(
        self,
        value: Annotated[
            object,
            DSLParameter(accepted_types=("collection", "string")),
        ],
        pattern: Annotated[
            object | None,
            DSLParameter(accepted_types=("string",), required=False, literal_only=True),
        ] = None,
    ) -> tuple[object, object | None]:
        """Return the size of a collection or text, or count a pattern in text.

        `count(value)` measures a supported collection or text value. `count(text, pattern)`
        counts occurrences of literal pattern text. When `value` is a filtered timeline
        collection, compare the result with an integer from 0 through 8.

        Internal:

        This helper records the public call shape and returns the arguments to the compiler.
        """
        return value, pattern

    @dsl_compiler_form(
        name="len",
        return_type="integer",
        compiler_tags=("chunk_count",),
    )
    def _bind_len_argument(
        self,
        value: Annotated[
            object,
            DSLParameter(accepted_types=("collection", "string")),
        ],
    ) -> object:
        """Return the size of a supported collection or the length of text.

        When `value` is a filtered timeline collection, compare the result with an integer
        from 0 through 8.
        """

        return value

    def _chunk_body(self, body, var: str, split_model: str) -> tuple[str, str, float, dict]:
        """Validate a per-chunk body — ``<scalar>(model, …, <chunk>) <op> n`` — and return
        (scalar name, normalized operator, threshold, folded extras).

        Shared by the quantifier and cardinality forms, which differ only in how the returned
        comparison is used: as the whole test, or as the inner filter of a count.
        """
        if not (isinstance(body, BinaryOp) and body.operator in _COMPARISONS):
            raise UnsupportedConstruct(
                "a chunked body must be a single `<scalar>(...) <op> n` comparison over "
                f"one of {', '.join(sorted(_CHUNK_SCALARS))}"
            )
        fc, number, flipped = self._split_scalar(body.left, body.right)
        if fc is None or number is None or fc.name not in _CHUNK_SCALARS:
            raise UnsupportedConstruct(
                "a chunked body must compare one of "
                f"{', '.join(sorted(_CHUNK_SCALARS))} against a numeric literal"
            )
        op = _FLIP.get(body.operator, body.operator) if flipped else body.operator
        shape = _CHUNK_SCALARS[fc.name]
        if len(fc.args) != shape["arity"]:
            raise UnsupportedConstruct(f"{fc.name}() takes {shape['arity']} arguments")
        scalar_model = self._lit(fc.args[0])
        if scalar_model != split_model:
            # The token budget only means something relative to the model that will consume the
            # chunks, so splitting with one model and measuring with another is a rule bug.
            raise UnsupportedConstruct(
                f"chunked scoring splits and measures with ONE model: {_chunk_source_func()} uses "
                f"{split_model!r} but {fc.name} uses {scalar_model!r}"
            )
        chunk_arg = shape["chunk_arg"]
        if not (isinstance(fc.args[chunk_arg], Identifier) and fc.args[chunk_arg].name == var):
            raise UnsupportedConstruct(
                f"{fc.name}()'s argument {chunk_arg} must be the chunk variable {var!r}"
            )
        for position, other in enumerate(fc.args):
            if position != chunk_arg and self._references_var(other, var):
                raise UnsupportedConstruct(
                    f"the chunk variable {var!r} may only be {fc.name}()'s chunk argument"
                )

        extra: dict = {}
        if shape["constitution"]:
            # The constitution folds exactly as it does for a plain llm_score atom: literal
            # policy text, or a text_ref NAME the detector resolves from config.texts at scan
            # time.
            const = fc.args[1]
            if isinstance(const, FunctionCall) and const.name == "text_ref":
                extra["constitution"] = self._lit(const.args[0]) if const.args else ""
                extra["constitution_is_ref"] = True
            else:
                extra["constitution"] = self._lit(const)
        return fc.name, op, number, extra

    def _mint_chunk_atom(
        self,
        split_model: str,
        max_tokens: int,
        scalar: str,
        combine: str,
        op: str,
        number: float,
        extra: dict,
        *,
        inner_op: str = "",
        inner_number: float = 0.0,
    ) -> T.Formula:
        """Intern the folded chunked atom. STATIC, anchored on the system_prompt boot step
        (always groundable; the detector ignores the anchor text and reads ctx.conversation) —
        identical treatment to the whole-conversation llm_score atom in `_comparison_atom`."""
        self.has_static = True
        spec = make_atom(
            domain=AtomDomain.STATIC,
            selector="system_prompt",
            value_expr_key="field:",
            predicate="external",
            patterns=(),
            cost=Cost.LLM,
            op=op,
            number=number,
            detector=_CHUNK_REDUCE_DETECTOR,
            detector_arg=split_model,
            chunk_max_tokens=max_tokens,
            chunk_combine=combine,
            chunk_scalar=scalar,
            chunk_inner_op=inner_op,
            chunk_inner_number=inner_number,
            **extra,
        )
        inner = f" inner {inner_op} {inner_number}" if combine == "count" else ""
        self.notes.append(
            f"chunked {scalar}({split_model}, max_tokens={max_tokens}, "
            f"combine={combine}){inner} {op} {number}"
        )
        return T.Atom(self._intern_atom(spec).atom_id)

    def _toplevel_static_surface(self, node: BinaryOp) -> str | None:
        """The STATIC surface (``system_prompt``/``tools``) a top-level comparison is over,
        or None if it isn't a static-surface comparison. Recognises the four shapes that
        lower through :meth:`compile_step` at top level:

          - ``model_score(…, system_prompt) <op> n`` — external classifier on the prompt;
          - ``count(system_prompt, "p") <op> n`` — occurrence count on the prompt text;
          - ``@a in system_prompt <op> n`` / ``#a in system_prompt <op> n`` — match
            position / occurrence count on the prompt text.
        """
        for a in (node.left, node.right):
            if isinstance(a, FunctionCall) and a.name in _INFERENCE_SCORE_EXTERNAL:
                return "system_prompt"
            if isinstance(a, FunctionCall) and a.name in _EXTERNAL_FUNCS and len(a.args) > 1:
                surf = self._static_field_surface(a.args[1])
                if surf is not None:
                    return surf
            if isinstance(a, FunctionCall) and a.name == "count" and len(a.args) == 2:
                surf = self._static_field_surface(a.args[0])
                if surf is not None:
                    return surf
            if (
                isinstance(a, BinaryOp)
                and a.operator == "in"
                and isinstance(a.left, (StringPosition, StringCount))
            ):
                surf = self._static_field_surface(a.right)
                if surf is not None:
                    return surf
        return None

    def _constructed_dynamic_ref(self, node):
        """The FIRST dynamic field-access node of a ``message(...)[.add_message(...)]`` builder,
        or None (all-literal, or not a builder). All dynamic slots must reference the same event
        surface, so the first is enough for top-level surface routing. Never raises."""
        if not (isinstance(node, FunctionCall) and node.name in ("message", "add_message")):
            return None
        try:
            _, dynamic_refs = self._flatten_constructed_input(node)
        except UnsupportedConstruct:
            return None
        return dynamic_refs[0] if dynamic_refs else None

    def _toplevel_dynamic_scored_surface(self, node) -> str | None:
        """The event surface for a top-level scored comparison whose model input is a constructed
        value with a DIRECT INDEXED message field (``llm_score(m, p, message("user",
        messages[0].content)) <op> n``), or None. Such a comparison lowers to an
        Eventually-wrapped EVENT atom (a bound comprehension value only occurs inside a
        comprehension, which routes itself)."""
        if not (isinstance(node, BinaryOp) and node.operator in _COMPARISONS):
            return None
        input_index = {"llm_score": 2, "llm_input_tokens": 1}
        for a in (node.left, node.right):
            if (
                isinstance(a, FunctionCall)
                and a.name in input_index
                and len(a.args) > input_index[a.name]
            ):
                ref = self._constructed_dynamic_ref(a.args[input_index[a.name]])
                if ref is not None:
                    resolved = self._resolve_field_target(ref)
                    if resolved is not None and resolved[0] in _EVENT_SURFACES:
                        return resolved[0]
        return None

    @staticmethod
    def _static_field_surface(node) -> str | None:
        """If ``node`` names the ``system_prompt`` static surface (bare Identifier or the
        equivalent base-less FieldAccess), return ``"system_prompt"``; else None.

        Restricted to ``system_prompt`` deliberately: it is the static surface with plain
        *text* semantics, so count/position/external predicates over it are well-defined.
        ``tools`` is a collection of objects (not a text field), so ``count(tools, …)`` /
        ``model_score(…, tools)`` have no meaning through this path — they keep failing to
        compile (loudly) rather than grounding on dubious text."""
        if isinstance(node, Identifier) and node.name == "system_prompt":
            return node.name
        if (
            isinstance(node, FieldAccess)
            and node.base is None
            and node.fields
            and str(node.fields[0]) == "system_prompt"
        ):
            return "system_prompt"
        return None

    def _static_surface_step(self, node, surface: str, scope: Scope) -> T.Formula:
        """Lower a static-surface predicate (comparison or string-set quantifier over
        ``system_prompt``/``tools``) through the shared :meth:`compile_step` machinery. This
        reuses the exact position/count/external/quantifier lowering the comprehension path
        uses, so a construct that works inside `any … where …` works at top level too.

        The atoms come out STATIC automatically: ``system_prompt``/``tools`` are in
        ``_STATIC_SURFACES``, and every minting helper (`_mint_scalar`, `_external_atom`,
        `_mint_atom`) tags an atom STATIC when its surface is not an event surface — so the
        executor grounds it once at the boot step (never on the timeline, so ``within:N`` is
        unaffected).

        A bare surface Identifier (`system_prompt`) is normalised to a base-less
        `FieldAccess` so the step matchers recognise it as the field to read (their field
        operands are always `FieldAccess`, never a bare Identifier)."""
        normalised = self._normalise_static_operands(node, surface)
        return self.compile_step(normalised, surface, scope)

    def _normalise_static_operands(self, node, surface: str):
        """Rewrite a bare surface ``Identifier`` (e.g. ``system_prompt``) into a base-less
        ``FieldAccess`` wherever it appears as a field operand, so the step matchers (which
        expect a ``FieldAccess`` field) accept it. Structural, in place; returns the node."""
        for attr in ("left", "right", "operand", "condition"):
            child = getattr(node, attr, None)
            if isinstance(child, Identifier) and child.name == surface:
                setattr(node, attr, FieldAccess(base=None, fields=[surface]))
            elif child is not None and type(child).__module__.startswith("quarry.types.dsl"):
                self._normalise_static_operands(child, surface)
        return node

    def _count_collection_macro(self, node: BinaryOp) -> T.Formula | None:
        """`count(<surface>) <op> n` / `len(<surface>) <op> n` → a zero-width **static
        surface-count atom** (grounded once at the boot step against
        ``StaticEnvironment.surface_sizes``).

        Collection size is a property of the whole word, not a temporal count, so it is
        one bare static atom regardless of ``n`` — scaling to any N (e.g. ``len(messages)
        > 5000``) with a single DFA state, unlike the O(n) temporal counter. Returns None
        if this isn't a ``count/len(surface)`` comparison (falls through, e.g.
        ``count(field, "pat")`` / ``len(field)`` over a scalar field).
        """
        fc, number, flipped = self._split_scalar(node.left, node.right)
        if fc is None:
            return None
        # Resolve (surface, optional name-filter) for the size expression:
        #   count(S) / len(S)                    → whole surface S
        #   count_tool_calls("x")                → tool_calls named "x"
        #   len(get_tool_calls("x"))             → tool_calls named "x" (accessor)
        surface = None
        name_filter = ""
        window: tuple[int | None, int | None] | None = None
        if fc.name in ("count", "len") and len(fc.args) == 1:
            coll = (
                self._bind_count_arguments(*fc.args)[0]
                if fc.name == "count"
                else self._bind_len_argument(*fc.args)
            )
            if isinstance(coll, FunctionCall) and coll.name == "_slice" and len(coll.args) == 3:
                # `len(S[a:b])`: the size of the slice, resolved against the surface size.
                base, start_node, stop_node = coll.args
                if not (isinstance(base, Identifier) and base.name in _EVENT_SURFACES):
                    raise UnsupportedConstruct(
                        f"slice is only supported directly on an event surface, not {base!r}"
                    )
                surface = base.name
                start = start_node.value if isinstance(start_node, Literal) else None
                stop = stop_node.value if isinstance(stop_node, Literal) else None
                window = None if start is None and stop is None else (start, stop)
            elif isinstance(coll, Identifier) and coll.name in _SURFACES:
                surface = coll.name
            elif isinstance(coll, FunctionCall) and coll.name in _ACCESSORS:
                surface, filter_field = _ACCESSORS[coll.name]
                if filter_field is not None and coll.args:
                    name_filter = self._lit(coll.args[0])
        elif fc.name == "count_tool_calls" and fc.args:
            surface, name_filter = "tool_calls", self._lit(fc.args[0])
        if surface is None:
            return None
        op = _FLIP.get(node.operator, node.operator) if flipped else node.operator
        if op not in _COMPARISONS:
            raise UnsupportedConstruct(f"count comparison {op!r} not expressible")
        # Bare static atom: domain STATIC → the executor grounds it once at the boot step
        # (not counted in within:N). The oracle compares the surface count `op` the bound as
        # written (a fractional bound is compared, not truncated); a name-filter (tool name)
        # narrows the count to matching tool calls.
        self.has_static = True
        spec = make_atom(
            domain=AtomDomain.STATIC,
            selector=surface,
            value_expr_key="",
            predicate="surface_count",
            patterns=((name_filter,) if name_filter else ()),
            op=op,
            number=float(number),
            cost=Cost.PURE,
        )
        if window is not None:
            spec = replace(
                spec, atom_id="", slice_start=window[0], slice_stop=window[1], is_slice=True
            )
        return T.Atom(self._intern_atom(spec).atom_id)

    def compile_step(self, node, surface: str, scope: Scope) -> T.Formula:
        self._reject_foreign_reads(node, scope)
        if not self._contains_model_call(node):
            self._reject_model_text_forms(node)
        if isinstance(node, BinaryOp):
            op = node.operator
            if op == "matches" and isinstance(node.right, PatternMatch):
                return self._pattern_match(node.left, node.right, surface, scope)
            if op in ("and", "or"):
                left = self.compile_step(node.left, surface, scope)
                right = self.compile_step(node.right, surface, scope)
                return T.And(left, right) if op == "and" else T.Or(left, right)
            if self._contains_model_call(node):
                return self._model_predicate_atom(node, scope, surface, event_bound=True)
            if op == "contains":
                if self._is_capture_ref(node.right):
                    return self._mint_capture(surface, self._field_path(node.left), "capture_ref")
                return self._content_atom(node.left, node.right, surface, "contains_any")
            if op == "in":
                # `N of (...) in field`, `@x in field`, `#x in field`, list membership
                # (`field in [lits]`), pattern-in-field, or `v in field` (capture ref).
                if isinstance(node.left, StringSetQuantifier):
                    return self._string_set_quantifier(node.left, node.right, surface)
                if self._is_capture_ref(node.left) and isinstance(node.right, FieldAccess):
                    return self._mint_capture(surface, self._field_path(node.right), "capture_ref")
                lit_list = self._as_literal_list(node.right)
                if lit_list is not None and isinstance(node.left, FieldAccess):
                    # `field in [a, b, …]` — the field's value equals ANY listed literal.
                    fld = self._field_path(node.left)
                    return _or_all(
                        [
                            self._mint_scalar(surface, fld, "equals", "=", None, patterns=(str(v),))
                            for v in lit_list
                        ]
                    )
                return self._content_atom(node.right, node.left, surface, "contains_any")
            if op in ("=", "!=", "~", "matches"):
                complete = self._complete_comparison(node)
                if complete is not None:
                    return complete
            if op in ("~", "matches"):
                return self._content_atom(node.left, node.right, surface, "regex")
            if op in _COMPARISONS:
                special = self._position_or_count(node, surface)
                if special is not None:
                    return special
                return self._comparison_atom(node, surface)
            raise UnsupportedConstruct(f"operator {op!r} in step predicate")
        if isinstance(node, ContainsAnyOp):
            fld = self._field_path(node.left)
            patterns = tuple(str(v) for v in node.values)
            return self._mint_atom(surface, fld, "contains_any", patterns)
        if isinstance(node, UnaryOp) and node.operator == "not":
            return T.Not(self.compile_step(node.operand, surface, scope))
        if isinstance(node, Assignment):
            expr = getattr(node, "expression", None)
            # Explicit capture bind: `<pattern> in field as v` — mint a capture_bind atom on
            # the bind field. The value domain is enumerated by the registry per candidate.
            if (
                self._capture_var
                and str(getattr(node, "variable", "") or "") == self._capture_var
                and isinstance(expr, BinaryOp)
                and expr.operator == "in"
                and isinstance(expr.right, FieldAccess)
            ):
                patterns, sem = self._resolve_patterns(expr.left)
                return self._mint_capture(
                    surface,
                    self._field_path(expr.right),
                    "capture_bind",
                    predicate="capture_match",
                    patterns=tuple(patterns),
                    sem=sem,
                )
            # Bind-only (var never referenced → no active capture): `$pat in field as v`
            # means the plain boolean `$pat in field`, so lower the inner
            # `in` expression directly (the capture is evidence-only).
            if isinstance(expr, BinaryOp):
                return self.compile_step(expr, surface, scope)
            raise UnsupportedConstruct("value capture `... as v` shape not supported")
        if isinstance(node, StringSetQuantifier):
            # Bare `N of ($a, $b, …)` (no `in field`) → the sets are matched against the
            # element's content (the default field when none is named).
            return self._string_set_quantifier(node, None, surface)
        if isinstance(node, FunctionCall) and node.name in _REFUSAL_GUARDS:
            # A bare `refused(...)` is a model predicate, not a step function.
            return self._model_predicate_atom(node, scope, surface, event_bound=True)
        if isinstance(node, FunctionCall):
            return self._compile_step_function(node, surface, scope)
        if isinstance(node, Literal) and isinstance(node.value, bool):
            return T.TRUE if node.value else T.FALSE
        # A bare identifier here is not a surface/element and (after variable inlining) not a
        # declared variable — most often a typo'd or undefined `variables:` reference.
        if isinstance(node, Identifier):
            raise UnsupportedConstruct(
                f"undefined reference {node.name!r} (not a field, element, or declared variable)"
            )
        raise UnsupportedConstruct(f"step node {type(node).__name__}")

    def _pattern_match(
        self,
        subject,
        pattern: PatternMatch,
        surface: str,
        scope: Scope,
    ) -> T.Formula:
        """Lower ``element matches { field op value or ... }`` as alternative field checks."""

        if not isinstance(subject, Identifier) or scope.resolve_element(subject.name) != surface:
            raise UnsupportedConstruct(
                "the value left of `matches { ... }` must be the whole comprehension element"
            )
        alternatives: list[T.Formula] = []
        with self._activate_ast_node(pattern):
            for predicate in pattern.patterns:
                if not isinstance(predicate, BinaryOp):
                    raise UnsupportedConstruct("each pattern entry must be a field comparison")
                field = predicate.left
                if isinstance(field, Identifier):
                    rooted = FieldAccess(subject, [field.name])
                elif isinstance(field, FieldAccess):
                    root = field.base
                    if isinstance(root, Identifier):
                        rooted = FieldAccess(subject, [root.name, *field.fields])
                    elif root is None:
                        rooted = FieldAccess(subject, list(field.fields))
                    else:
                        raise UnsupportedConstruct("pattern fields must be relative to the element")
                else:
                    raise UnsupportedConstruct("pattern entries must start with a field name")
                # The rooted comparison is compiler-generated. Keep its atom attached to the
                # pattern block and the specific predicate the author actually wrote.
                with self._activate_ast_node(predicate):
                    alternatives.append(
                        self.compile_step(
                            BinaryOp(predicate.operator, rooted, predicate.right),
                            surface,
                            scope,
                        )
                    )
        return _or_all(alternatives)

    def _is_capture_ref(self, node) -> bool:
        return (
            bool(self._capture_var)
            and isinstance(node, Identifier)
            and node.name == self._capture_var
        )

    @staticmethod
    def _contains_comprehension(node) -> bool:
        if isinstance(node, (AnyComprehension, AllComprehension)):
            return True
        for attr in ("left", "right", "operand", "condition"):
            child = getattr(node, attr, None)
            if child is not None and _Compiler._contains_comprehension(child):
                return True
        return False

    @staticmethod
    def _reject_model_text_forms(node) -> None:
        """Reject ``join()`` and a map comprehension in a test that no model expression
        reads. The message builder checks the content of a version 1 ``message()`` and gives
        its own reason."""

        pending = [node]
        while pending:
            current = pending.pop()
            if isinstance(current, FunctionCall) and current.name in ("message", "add_message"):
                continue
            if isinstance(current, MapComprehension):
                raise UnsupportedConstruct(_MAP_COMPREHENSION_FORM)
            if isinstance(current, FunctionCall) and current.name == "join":
                raise UnsupportedConstruct(_JOIN_FORM)
            pending.extend(iter_condition_children(current))

    @staticmethod
    def _contains_model_call(node) -> bool:
        """Return whether ``node`` contains a typed model operation.

        A refusal guard counts as one: it must read a model call, so lowering it as a model
        expression reports a misplaced argument instead of an unsupported condition.
        """

        return any(
            isinstance(current, ModelCall)
            or (isinstance(current, FunctionCall) and current.name in _REFUSAL_GUARDS)
            for current in walk_condition_nodes(node)
        )

    @staticmethod
    def _model_calls(node) -> tuple[ModelCall, ...]:
        """Return typed model operations in deterministic source order."""

        return tuple(
            current for current in walk_condition_nodes(node) if isinstance(current, ModelCall)
        )

    @staticmethod
    def _model_binding_dependencies(node, scope: Scope) -> set[str]:
        """Return comprehension variables read by a model expression. The variable of a map
        comprehension in ``node`` hides an enclosing variable with the same name."""

        dependencies: set[str] = set()
        pending = [(node, frozenset())]
        while pending:
            current, hidden = pending.pop()
            if (
                isinstance(current, Identifier)
                and current.name not in hidden
                and scope.resolve_element(current.name) is not None
            ):
                dependencies.add(current.name)
            body = hidden | {current.variable} if isinstance(current, MapComprehension) else hidden
            for child in iter_condition_children(current):
                # The collection of a map comprehension is outside the scope of its variable.
                outside = isinstance(current, MapComprehension) and child is current.collection
                pending.append((child, hidden if outside else body))
        return dependencies

    def _compile_model_split(
        self,
        node: AnyComprehension,
        scope: Scope,
        source_node: AnyComprehension | None = None,
    ) -> T.Formula | None:
        """Lower the closed ``any chunk in model(...).split(...)`` form to one atom."""

        split_call = node.collection
        if not isinstance(split_call, ModelCall):
            return None
        if split_call.operation != "split":
            raise UnsupportedConstruct("only model().split() can be a comprehension collection")
        if self._contains_comprehension(node.condition):
            raise UnsupportedConstruct("model split supports one non-nested any comprehension")
        score_calls = self._model_calls(node.condition)
        if not score_calls or any(call.operation != "score" for call in score_calls):
            raise UnsupportedConstruct("model().split() must be consumed by a score predicate")
        for current in walk_condition_nodes(node.condition):
            # A split predicate is summarized by its chunk scores, so it can guard a score
            # with a fallback score but not read the refusal as a boolean.
            if isinstance(current, FunctionCall) and (
                current.name == "refused"
                or (
                    current.name == "if_refused"
                    and not (current.args and isinstance(current.args[0], ModelCall))
                )
            ):
                raise UnsupportedConstruct(
                    "a model().split() score predicate may use only if_refused(score, fallback)"
                )

        chunk_scope = scope.bind(node.variable, "model_chunks")
        for score_call in score_calls:
            if score_call.alias != split_call.alias:
                raise UnsupportedConstruct("model split and score must use the same model")
            score_input = score_call.args[0] if score_call.args else None
            if not isinstance(score_input, Identifier) or score_input.name != node.variable:
                raise UnsupportedConstruct(
                    "each split score must use the bound chunk as its messages argument"
                )
            if any(
                self._model_binding_dependencies(argument, chunk_scope)
                for argument in score_call.args[1:]
            ):
                raise UnsupportedConstruct(
                    "the bound chunk may only be used as model().score()'s messages argument"
                )

        source_node = source_node or node
        source_collection = (
            source_node.collection if isinstance(source_node.collection, ConditionNode) else None
        )
        with self._activate_ast_node(source_collection):
            split_expression, split_type = self._lower_model_expression(
                split_call, scope, allow_collection_source=True
            )
        with self._activate_ast_node(source_node.condition):
            predicate, predicate_type = self._lower_model_expression(node.condition, chunk_scope)
        if split_type != "Messages[]" or predicate_type != "bool":
            raise UnsupportedConstruct(
                "model split comprehension requires a boolean score predicate"
            )
        expression = ModelExpression(
            "split_any",
            node.variable,
            (split_expression, predicate),
        )
        self.has_static = True
        # The split form is lowered through a synthetic comprehension. Keep both authored
        # operands active while the single folded atom is interned so the source graph can
        # attribute it to the collection and predicate as well as to the comprehension.
        with (
            self._activate_ast_node(source_collection),
            self._activate_ast_node(source_node.condition),
        ):
            return self._mint_model_expression(expression, "messages")

    @dsl_compiler_form(
        name="model_split",
        return_type="collection<model_input>",
        allowed_uses=(Use.COMPREHENSION_SOURCE,),
        resource_group="token",
        syntax='model_split("alias", messages, max_input_tokens)',
        deprecated=True,
        replacement='model("alias").split(messages, max_input_tokens)',
    )
    def _compile_transitional_model_split(
        self,
        model: Annotated[
            ConditionNode,
            DSLParameter(accepted_types=("string",), literal_only=True),
        ],
        messages: Annotated[ConditionNode, DSLParameter(accepted_types=("model_input",))],
        max_input_tokens: Annotated[
            ConditionNode,
            DSLParameter(
                accepted_types=("integer",),
                literal_only=True,
                minimum=1,
            ),
        ],
    ) -> ModelCall:
        """Compatibility spelling for `model("alias").split(...)` in version 2 rules.

        This spelling remains available during migration. New rules should use the model method.
        """

        return ModelCall(
            alias=self._lit(model),
            operation="split",
            args=[messages, max_input_tokens],
            result_type="Messages[]",
        )

    @staticmethod
    def _model_tools_argument(call: ModelCall):
        """Return the normalized optional tools argument for a model call."""

        index = 2 if call.operation == "score" else 1
        if len(call.args) <= index:
            return None
        tools = call.args[index]
        if not isinstance(tools, Identifier) or tools.name != "tools":
            raise UnsupportedConstruct("model tools input must be the root tools value")
        return "tools"

    def _model_predicate_atom(
        self,
        node,
        scope: Scope,
        surface: str = "messages",
        *,
        event_bound: bool = False,
    ) -> T.Formula:
        """Lower one closed model-derived boolean expression to an external atom."""

        if any(call.operation == "split" for call in self._model_calls(node)):
            raise UnsupportedConstruct(
                "model().split() is supported only as an any-comprehension collection"
            )
        expression, result_type = self._lower_model_expression(node, scope)
        if result_type != "bool":
            raise UnsupportedConstruct("a model expression must produce a boolean rule condition")
        if surface == "messages" and not event_bound:
            self.has_static = True
        return self._mint_model_expression(expression, surface, event_bound=event_bound)

    def _mint_model_expression(
        self,
        expression: ModelExpression,
        surface: str,
        *,
        event_bound: bool = False,
    ) -> T.Formula:
        """Intern an engine-native model predicate without overloading legacy atom fields."""

        domain = AtomDomain.EVENT if event_bound else AtomDomain.STATIC
        spec = make_atom(
            domain=domain,
            selector=surface,
            value_expr_key="model_expression",
            predicate="external",
            cost=Cost.LLM,
            detector="model_expression",
            model_expression=expression,
            evidence_policy="model",
        )
        spec = self._intern_atom(spec)
        self.notes.append("model expression")
        return T.Atom(spec.atom_id)

    def _lower_refusal_guard(self, node: FunctionCall, scope: Scope) -> tuple[ModelExpression, str]:
        """Lower ``refused(value)`` or ``if_refused(value, fallback)`` to one model expression."""

        bind = self._bind_refused if node.name == "refused" else self._bind_if_refused
        try:
            arguments = bind(*node.args)
        except TypeError as exc:
            raise UnsupportedConstruct(
                f"{node.name}() received the wrong number of arguments"
            ) from exc
        if not any(
            call.operation in _REFUSABLE_OPERATIONS for call in self._model_calls(arguments[0])
        ):
            raise UnsupportedConstruct(
                f"{node.name}() must read a model().score() or model().complete() call"
            )
        guarded, guarded_type = self._lower_model_expression(arguments[0], scope)
        if node.name == "refused":
            return ModelExpression("refused", None, (guarded,)), "bool"
        fallback, fallback_type = self._lower_model_expression(arguments[1], scope)
        if fallback.kind not in {"literal", "parameter"}:
            raise UnsupportedConstruct("if_refused() fallback must be a literal or a parameter")
        return (
            ModelExpression("if_refused", None, (guarded, fallback)),
            _fallback_type(guarded_type, fallback_type),
        )

    def _lower_model_expression(
        self,
        node,
        scope: Scope,
        *,
        allow_collection_source: bool = False,
    ) -> tuple[ModelExpression, str]:
        """Convert the supported model-expression subset to a typed immutable payload."""

        if isinstance(node, ParameterRef):
            spec = self.parameters.get(node.name)
            if spec is None:
                raise UnsupportedConstruct(f"undeclared parameter {node.name!r}")
            self.used_parameters.add(node.name)
            return ModelExpression("parameter", node.name), {
                "number": "float",
                "text": "string",
            }[spec.type]

        if isinstance(node, Literal):
            value = node.value
            if value is not None and not isinstance(value, str | int | float | bool):
                raise UnsupportedConstruct("model expressions accept only scalar literals")
            result_type = (
                "null"
                if value is None
                else "bool"
                if isinstance(value, bool)
                else "int"
                if isinstance(value, int)
                else "float"
                if isinstance(value, float)
                else "string"
            )
            return ModelExpression("literal", value), result_type

        if isinstance(node, Identifier):
            if node.name.startswith("$"):
                patterns, semantics = self._resolve_string_set(node.name)
                return (
                    ModelExpression(
                        "string_patterns",
                        "".join(
                            "1" if enabled else "0"
                            for enabled in (
                                semantics.case_insensitive,
                                semantics.fullword,
                                semantics.base64,
                                semantics.is_regex,
                            )
                        ),
                        tuple(ModelExpression("literal", pattern) for pattern in patterns),
                    ),
                    "string_pattern_set",
                )
            if node.name == "messages":
                return ModelExpression("messages"), "Messages"
            if node.name in {
                "system_messages",
                "developer_messages",
                "user_messages",
                "assistant_messages",
            }:
                return ModelExpression("message_view", node.name), "Messages"
            if node.name == "tools":
                return ModelExpression("tools"), "Tools"
            bound_surface = scope.resolve_element(node.name)
            if bound_surface is not None:
                result_type = scope.resolve_element_type(node.name) or _SURFACE_ELEMENT_TYPES.get(
                    bound_surface, "object"
                )
                return ModelExpression("binding", node.name), result_type
            raise UnsupportedConstruct(f"unsupported model-expression reference {node.name!r}")

        if isinstance(node, FieldAccess):
            base, base_type, fields = self._model_field_parts(node, scope)
            if not fields:
                return base, base_type
            result_type = self._model_field_type(base_type, fields)
            return ModelExpression("field", ".".join(fields), (base,)), result_type

        if isinstance(node, FunctionCall):
            if node.name == "extract" and 2 <= len(node.args) <= 4:
                messages, messages_type = self._lower_model_expression(node.args[0], scope)
                span, span_type = self._lower_model_expression(node.args[1], scope)
                if messages_type != "Messages" or span_type != "Span":
                    raise UnsupportedConstruct("messages.extract() requires Messages and a span")
                bounds = [ModelExpression("literal", 0), ModelExpression("literal", 0)]
                for index, bound_node in enumerate(node.args[2:]):
                    bound, bound_type = self._lower_model_expression(bound_node, scope)
                    if bound_type != "int":
                        raise UnsupportedConstruct(
                            "messages.extract() context bounds must be integers"
                        )
                    if bound.kind == "literal" and (
                        isinstance(bound.value, bool) or int(cast(int, bound.value)) < 0
                    ):
                        raise UnsupportedConstruct(
                            "messages.extract() context bounds must be non-negative"
                        )
                    bounds[index] = bound
                return (
                    ModelExpression("extract_span", None, (messages, span, *bounds)),
                    "string",
                )
            if node.name == "text_ref":
                if len(node.args) != 1:
                    raise UnsupportedConstruct("text_ref() takes one literal name")
                name = self._lit(node.args[0])
                if not name:
                    raise UnsupportedConstruct("text_ref() name must not be empty")
                return ModelExpression("text_ref", name), "TextRef"
            if node.name == "_slice" and len(node.args) == 3:
                base, base_type = self._lower_model_expression(node.args[0], scope)
                if base_type != "Messages":
                    raise UnsupportedConstruct("only Messages values can be sliced for a model")
                start = self._model_slice_bound(node.args[1])
                stop = self._model_slice_bound(node.args[2])
                return (
                    ModelExpression(
                        "slice",
                        None,
                        (base, ModelExpression("literal", start), ModelExpression("literal", stop)),
                    ),
                    "Messages",
                )
            if node.name == "_index" and len(node.args) == 2:
                base, base_type = self._lower_model_expression(node.args[0], scope)
                if base_type == "Messages":
                    result_type = "Message"
                elif base_type.endswith("[]"):
                    result_type = base_type[:-2]
                else:
                    raise UnsupportedConstruct(
                        "only Messages or structured collection values can be indexed for a model"
                    )
                index = self._lit_index_signed(node.args[1])
                return ModelExpression("index", index, (base,)), result_type
            if node.name in {"message", "add_message"}:
                expected = 2 if node.name == "message" else 3
                if len(node.args) != expected:
                    raise UnsupportedConstruct(f"{node.name}() takes {expected} arguments")
                offset = 0
                children: list[ModelExpression] = []
                if node.name == "add_message":
                    base, base_type = self._lower_model_expression(node.args[0], scope)
                    if base_type != "Messages":
                        raise UnsupportedConstruct("add_message() receiver must be Messages")
                    if base.kind not in {"messages", "message", "add_message"}:
                        raise UnsupportedConstruct(
                            "add_message() receiver must be a root Messages collection"
                        )
                    children.append(base)
                    offset = 1
                role = self._lit(node.args[offset])
                self._require_builder_role(role)
                content, content_type = self._lower_model_content(node.args[offset + 1], scope)
                if content_type not in {"string", "TextRef"}:
                    raise UnsupportedConstruct(
                        f"message content must resolve to a string or text_ref, not {content_type}"
                    )
                children.extend((ModelExpression("literal", role), content))
                return ModelExpression(node.name, None, tuple(children)), "Messages"
            if node.name in {"refused", "if_refused"}:
                return self._lower_refusal_guard(node, scope)
            if node.name == "join":
                raise UnsupportedConstruct(_JOIN_FORM)
            raise UnsupportedConstruct(f"function {node.name}() is not a model-expression input")

        if isinstance(node, ModelCall):
            if node.result_type.endswith("[]") and not allow_collection_source:
                if node.operation == "spans":
                    raise UnsupportedConstruct(
                        "model().spans() is supported only as the direct collect source"
                    )
                raise UnsupportedConstruct(
                    "collection-valued model operations are supported only as comprehension sources"
                )
            handler = getattr(self, f"_bind_model_{node.operation}", None)
            if handler is None:
                raise UnsupportedConstruct(f"unknown model operation {node.operation!r}")
            try:
                arguments = handler(*node.args)
            except TypeError as exc:
                raise UnsupportedConstruct(
                    f"model().{node.operation}() received the wrong number of arguments"
                ) from exc
            # ``split`` binds its optional arguments to fixed positions; an absent one is
            # None. Trailing absent arguments are dropped, so the written form decides identity.
            while arguments and arguments[-1] is None:
                arguments = arguments[:-1]
            children = tuple(
                (ModelExpression("absent"), None)
                if arg is None
                else self._lower_model_expression(arg, scope)
                for arg in arguments
            )
            expressions = self._validate_model_call(node, arguments, children)
            result_type = (
                self._completion_types.get(node.alias, node.result_type)
                if node.operation == "complete"
                else node.result_type
            )
            return (
                ModelExpression("model_call", f"{node.alias}\x1f{node.operation}", expressions),
                result_type,
            )

        if isinstance(node, UnaryOp) and node.operator == "not":
            operand, operand_type = self._lower_model_expression(node.operand, scope)
            if operand_type != "bool":
                raise UnsupportedConstruct("not requires a boolean model expression")
            return ModelExpression("not", None, (operand,)), "bool"

        if isinstance(node, BinaryOp):
            equality = self._lower_item_equality(node, scope)
            if equality is not None:
                return equality, "bool"
            lower = (
                self._lower_text_operand
                if node.operator in _CONTENT_OPS
                else self._lower_model_expression
            )
            left, left_type = lower(node.left, scope)
            right, right_type = lower(node.right, scope)
            if "string_pattern_set" in {left_type, right_type}:
                if node.operator == "in" and left_type == "string_pattern_set":
                    if right_type == "string":
                        return ModelExpression("string_compare", "in", (left, right)), "bool"
                    if right_type == "object[]":
                        text = ModelExpression("field_text", None, (right,))
                        return ModelExpression("string_compare", "in", (left, text)), "bool"
                raise UnsupportedConstruct("a string set is supported only as '$set in text'")
            if node.operator in {"and", "or"}:
                if left_type != "bool" or right_type != "bool":
                    raise UnsupportedConstruct(
                        f"{node.operator} requires boolean model expressions"
                    )
                return ModelExpression(node.operator, None, (left, right)), "bool"
            if node.operator in {"+", "-", "*"}:
                if left_type not in {"int", "float"} or right_type not in {"int", "float"}:
                    raise UnsupportedConstruct("model arithmetic requires numeric operands")
                result_type = "float" if "float" in {left_type, right_type} else "int"
                return ModelExpression(node.operator, None, (left, right)), result_type
            if node.operator in _COMPARISONS:
                if node.operator in {"=", "!="}:
                    if not self._model_types_equalable(left_type, right_type):
                        raise UnsupportedConstruct("model equality requires compatible operands")
                elif left_type not in {"int", "float"} or right_type not in {"int", "float"}:
                    raise UnsupportedConstruct("ordered model comparisons require numeric operands")
                return ModelExpression("compare", node.operator, (left, right)), "bool"
            if node.operator in _CONTENT_OPS:
                return ModelExpression("string_compare", node.operator, (left, right)), "bool"
            raise UnsupportedConstruct(f"operator {node.operator!r} is not supported for models")

        if isinstance(node, MapComprehension):
            raise UnsupportedConstruct(_MAP_COMPREHENSION_FORM)
        raise UnsupportedConstruct(f"unsupported model expression {type(node).__name__}")

    def compile_collect(self, node) -> CollectPlan:
        """Lower one identity filter comprehension to an AST-free collection plan."""

        if not isinstance(node, FilterComprehension):
            raise UnsupportedConstruct("collect: requires one filter comprehension")

        if not isinstance(node.collection, ModelCall) or node.collection.operation != "spans":
            raise UnsupportedConstruct("collect: source must be a direct model().spans() call")

        output_variable = node.output_variable or node.variable
        if output_variable != node.variable:
            raise UnsupportedConstruct(
                "collect: must use the same item name before and after 'for'"
            )
        if self._find_flow(node.condition) is not None:
            raise UnsupportedConstruct("temporal expressions are not supported in collect:")

        source, source_type = self._lower_model_expression(
            node.collection, Scope(), allow_collection_source=True
        )
        if not source_type.endswith("[]"):
            raise UnsupportedConstruct("collect: source must produce a collection")

        item_type = source_type[:-2]
        item_scope = Scope().bind(node.variable, "collect_items", item_type)
        local_terms: list[ModelExpression] = []
        model_terms: list[ModelExpression] = []
        for term in self._collect_conjuncts(node.condition):
            has_model_call = self._contains_model_call(term)
            if has_model_call and any(
                (isinstance(current, UnaryOp) and current.operator == "not")
                or (isinstance(current, BinaryOp) and current.operator == "or")
                for current in walk_condition_nodes(term)
            ):
                raise UnsupportedConstruct(
                    "collect: model filters must be positive and may be combined only "
                    "with top-level 'and'"
                )
            predicate, predicate_type = self._lower_model_expression(term, item_scope)
            if predicate_type != "bool":
                raise UnsupportedConstruct("collect: filter must produce a boolean value")
            (model_terms if has_model_call else local_terms).append(predicate)

        return CollectPlan(
            variable=node.variable,
            item_type=item_type,
            source=source,
            local_predicate=self._join_collect_predicates(local_terms),
            model_predicate=(self._join_collect_predicates(model_terms) if model_terms else None),
            has_external_work=self._contains_model_call(node),
        )

    @staticmethod
    def _collect_conjuncts(node) -> tuple[ConditionNode, ...]:
        """Flatten only top-level ``and`` terms, preserving source order."""

        if isinstance(node, BinaryOp) and node.operator == "and":
            return (
                *_Compiler._collect_conjuncts(node.left),
                *_Compiler._collect_conjuncts(node.right),
            )
        return (node,)

    @staticmethod
    def _join_collect_predicates(expressions: list[ModelExpression]) -> ModelExpression:
        """Join validated predicates into one left-associated Boolean expression."""

        if not expressions:
            return ModelExpression("literal", True)
        result = expressions[0]
        for expression in expressions[1:]:
            result = ModelExpression("and", None, (result, expression))
        return result

    def _model_field_parts(
        self, node: FieldAccess, scope: Scope
    ) -> tuple[ModelExpression, str, list[str]]:
        """A model-expression field access as its lowered base, the base's type, and the
        field path read from it."""

        if node.base is None:
            if not node.fields:
                raise UnsupportedConstruct("empty model-expression field access")
            base, base_type = self._lower_model_expression(Identifier(str(node.fields[0])), scope)
            return base, base_type, [str(name) for name in node.fields[1:]]
        base, base_type = self._lower_model_expression(node.base, scope)
        return base, base_type, [str(name) for name in node.fields]

    def _lower_model_text(self, node, scope: Scope) -> tuple[ModelExpression, str]:
        """Lower ``message()`` content, which is text. A field with a text form
        (``_MODEL_TEXT_FIELDS``) is read as that text: a tool call's ``arguments`` is the JSON
        argument text the call carries (``raw_arguments``), the same text a model sees in the
        assistant turn."""

        if not isinstance(node, FieldAccess):
            return self._lower_model_expression(node, scope)
        base, base_type, fields = self._model_field_parts(node, scope)
        if not fields:
            return base, base_type
        owner = self._model_field_type(base_type, fields[:-1])
        text_field = _MODEL_TEXT_FIELDS.get(owner, {}).get(fields[-1])
        if text_field is not None:
            fields = [*fields[:-1], text_field]
        return (
            ModelExpression("field", ".".join(fields), (base,)),
            self._model_field_type(base_type, fields),
        )

    def _lower_item_equality(self, node: BinaryOp, scope: Scope) -> ModelExpression | None:
        """Lower ``field = literal`` or ``field != literal`` on a map comprehension item as a
        rule step reads it, or return None for a different test.

        The test compares the text of the field value with the text of the literal, so the
        number 5 equals "5". A missing or null field equals only ``null``. Thus ``!=`` with a
        text is true for a call that does not have the field."""

        if node.operator not in {"=", "!="}:
            return None
        for field_node, value_node in ((node.left, node.right), (node.right, node.left)):
            if not (isinstance(field_node, FieldAccess) and isinstance(value_node, Literal)):
                continue
            root = _field_root_name(field_node)
            if root is None or scope.resolve_element(root) != _MAP_ITEM_SURFACE:
                continue
            value, _ = self._lower_model_expression(field_node, scope)
            wanted = None if value_node.value is None else str(value_node.value)
            test = ModelExpression("text_equals", wanted, (value,))
            return test if node.operator == "=" else ModelExpression("not", None, (test,))
        return None

    def _lower_text_operand(self, node, scope: Scope) -> tuple[ModelExpression, str]:
        """Lower an operand of a text test. A field of a map comprehension item reads the text
        that the same test in a rule step reads (``field_text``): ``c.arguments`` is the
        decoded object, and ``c.arguments.*`` or a list value is one value on each line."""

        lowered, lowered_type = self._lower_model_expression(node, scope)
        root = _field_root_name(node) if isinstance(node, FieldAccess) else None
        if root is None or scope.resolve_element(root) != _MAP_ITEM_SURFACE:
            return lowered, lowered_type
        if lowered_type not in _STEP_TEXT_TYPES:
            return lowered, lowered_type
        return ModelExpression("field_text", None, (lowered,)), "string"

    def _model_field_type(self, base_type: str, fields: list[str]) -> str:
        """Return the public type reached through a structured model-result path."""

        result_type = base_type
        for field_name in fields:
            if result_type == "object":
                # A JSON value read by key is expected to be a string (checked where it is
                # used); ``*`` reads every value of the object, which is a list.
                return "object[]" if fields[-1] == "*" else "string"
            field_types = self._field_types.get(result_type)
            if field_types is None or field_name not in field_types:
                if result_type == "AssistantMessage" and field_name == "json":
                    raise UnsupportedConstruct(
                        ".json requires a model alias with complete.response_schema"
                    )
                if field_types is not None and result_type.startswith("json object"):
                    raise UnsupportedConstruct(
                        f"{result_type} has no field {field_name!r} "
                        f"(fields: {', '.join(field_types)})"
                    )
                raise UnsupportedConstruct(
                    f"model result type {result_type} has no field {field_name!r}"
                )
            result_type = field_types[field_name]
        return result_type

    @staticmethod
    def _model_types_equalable(left_type: str, right_type: str) -> bool:
        """Return whether two model-expression values support equality comparison."""

        if left_type == right_type or {left_type, right_type} <= {"int", "float"}:
            return True
        if "object" in {left_type, right_type}:
            return True
        if left_type == "null":
            return right_type.endswith("?")
        if right_type == "null":
            return left_type.endswith("?")
        return left_type.removesuffix("?") == right_type.removesuffix("?")

    @staticmethod
    def _model_slice_bound(node) -> int | None:
        if isinstance(node, Literal) and (
            node.value is None or (isinstance(node.value, int) and not isinstance(node.value, bool))
        ):
            return node.value
        raise UnsupportedConstruct("message slice bounds must be integer literals")

    @staticmethod
    def _lit_index_signed(node) -> int:
        if (
            isinstance(node, Literal)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            return node.value
        raise UnsupportedConstruct("message indexes must be integer literals")

    @staticmethod
    def _validate_model_call(
        call: ModelCall,
        arguments: tuple[object, ...],
        children: tuple[tuple[ModelExpression, str | None], ...],
    ) -> tuple[ModelExpression, ...]:
        """Validate model-method arguments against their generated catalog entry, and return
        their expressions with the language's implicit conversions applied."""

        item = function_by_name(f"model.{call.operation}")
        if item is None:
            raise LanguageSpecError(
                f"Missing catalog entry for model().{call.operation}(); regenerate the language spec"
            )
        public_to_compiler_types = {
            "model_input": {"Messages"},
            "number": {"int", "float"},
            "string": {"string"},
            "text_reference": {"TextRef"},
            "collection<tool_definition>": {"Tools"},
            "integer": {"int"},
        }
        parameters = item["parameters"]
        expressions = [expression for expression, _ in children]
        for position, (argument, (expression, argument_type), parameter) in enumerate(
            zip(arguments, children, parameters, strict=False)
        ):
            if argument is None:
                continue
            accepted = set().union(
                *(
                    public_to_compiler_types.get(value, {value})
                    for value in parameter["accepted_types"]
                )
            )
            expressions[position], argument_type = _convert(expression, argument_type, accepted)
            if argument_type not in accepted:
                names = [*parameter["accepted_types"]] + [
                    f"a {source} of the conversation"
                    for source, target, _ in TYPE_CONVERSIONS
                    if target in parameter["accepted_types"]
                ]
                raise UnsupportedConstruct(
                    f"model().{call.operation}() argument {position + 1} must be "
                    + " or ".join(names)
                )
            if parameter["literal_only"] and not isinstance(argument, Literal):
                raise UnsupportedConstruct(
                    f"model().{call.operation}() {parameter['name']} must be written directly"
                )
            minimum = parameter["minimum"]
            if minimum is not None and (
                not isinstance(argument, Literal) or argument.value < minimum
            ):
                raise UnsupportedConstruct(
                    f"model().{call.operation}() {parameter['name']} must be at least {minimum}"
                )
            maximum = parameter["maximum"]
            if maximum is not None and (
                not isinstance(argument, Literal) or argument.value > maximum
            ):
                raise UnsupportedConstruct(
                    f"model().{call.operation}() {parameter['name']} must be at most {maximum}"
                )
            if parameter["name"] == "tools" and (
                not isinstance(argument, Identifier) or argument.name != "tools"
            ):
                raise UnsupportedConstruct("optional model tools argument must be root tools")
        return tuple(expressions)

    @dsl_compiler_form(
        name="model.score",
        return_type="number",
        resource_group="score",
        syntax='model("alias").score(messages, constitution[, tools])',
        implicit_receiver_count=1,
    )
    def _bind_model_score(
        self,
        messages: Annotated[object, DSLParameter(accepted_types=("model_input",))],
        constitution: Annotated[
            object,
            DSLParameter(accepted_types=("string", "text_reference")),
        ],
        tools: Annotated[
            object | None,
            DSLParameter(
                accepted_types=("collection<tool_definition>",),
                required=False,
            ),
        ] = None,
    ) -> tuple[object, ...]:
        """Score messages against policy text and return a number from 0 through 1.

        The model alias comes from the surrounding `model("alias")` reference. `constitution`
        may be literal policy text or a `text_ref("name")`. Pass the bare `tools` value only
        when the model should also see the request's offered tools.
        """

        return (messages, constitution) + ((tools,) if tools is not None else ())

    @dsl_compiler_form(
        name="model.classify",
        return_type="number",
        resource_group="model",
        syntax='model("alias").classify(messages[, tools])',
        implicit_receiver_count=1,
    )
    def _bind_model_classify(
        self,
        messages: Annotated[object, DSLParameter(accepted_types=("model_input",))],
        tools: Annotated[
            object | None,
            DSLParameter(
                accepted_types=("collection<tool_definition>",),
                required=False,
            ),
        ] = None,
    ) -> tuple[object, ...]:
        """Return the configured classifier head's probability for these messages.

        The model alias comes from the surrounding `model("alias")` reference. Pass the bare
        `tools` value only when the classifier should also see offered tool definitions.
        """

        return (messages,) + ((tools,) if tools is not None else ())

    @dsl_compiler_form(
        name="model.spans",
        return_type="collection<span>",
        allowed_uses=(Use.COMPREHENSION_SOURCE,),
        resource_group="model",
        syntax='model("alias").spans(messages, threshold)',
        implicit_receiver_count=1,
    )
    def _bind_model_spans(
        self,
        messages: Annotated[object, DSLParameter(accepted_types=("model_input",))],
        threshold: Annotated[
            object,
            DSLParameter(
                accepted_types=("number",),
                literal_only=True,
                minimum=0.0,
                maximum=1.0,
            ),
        ],
    ) -> tuple[object, ...]:
        """Return original-conversation spans from the alias-configured span operation.

        `messages` must be the root conversation or one of its views. Quarry reuses the raw
        classifier response cached for the same model input. The adapter owns span decoding and
        how the required `threshold` is applied.
        """

        if not self._is_conversation_backed_messages(messages):
            raise UnsupportedConstruct(
                "model().spans() input must be a view of the original conversation"
            )
        return messages, threshold

    @staticmethod
    def _is_conversation_backed_messages(node) -> bool:
        """Return whether a span input is visibly derived from the conversation root."""

        if isinstance(node, Identifier):
            return node.name in {
                "messages",
                "system_messages",
                "developer_messages",
                "user_messages",
                "assistant_messages",
            }
        return (
            isinstance(node, FunctionCall)
            and node.name in {"_slice", "_index"}
            and _Compiler._is_conversation_backed_messages(node.args[0])
        )

    @dsl_compiler_form(
        name="model.complete",
        return_type="message",
        resource_group="completion",
        syntax='model("alias").complete(messages[, tools])',
        implicit_receiver_count=1,
    )
    def _bind_model_complete(
        self,
        messages: Annotated[object, DSLParameter(accepted_types=("model_input",))],
        tools: Annotated[
            object | None,
            DSLParameter(
                accepted_types=("collection<tool_definition>",),
                required=False,
            ),
        ] = None,
    ) -> tuple[object, ...]:
        """Generate one assistant message from the supplied messages.

        Read fields such as `.content`, `.refusal`, or `.tool_calls` from the result. Pass the
        bare `tools` value only when the model may call offered tools. When the alias declares
        `complete.response_schema`, read the reply's fields by name with `.json.<field>`; a
        reply that does not match the schema is a refused call.
        """

        return (messages,) + ((tools,) if tools is not None else ())

    @dsl_compiler_form(name="refused", return_type="boolean")
    def _bind_refused(
        self,
        value: Annotated[
            object, DSLParameter(accepted_types=("boolean", "number", "string", "message"))
        ],
    ) -> tuple[object]:
        """Return whether a `model("alias").score(...)` or `.complete(...)` call in `value` was refused.

        A call is refused when a guardrail fired, so the provider stopped the reply or
        rejected the request, or when the reply is not in the format the operation asked for,
        such as a score reply that is not a configured verdict token. A completion that says
        no in its own words is an answer, not a refusal; read it from `.refusal` or
        `.content`. `refused()` reads the call's own request, so it sends none of its own.

        Without `refused()` or `if_refused()`, a refused call fails the rule. A rule that reads
        a call inside them must read it there everywhere, so no unguarded read can fail first.
        """

        return (value,)

    @dsl_compiler_form(name="if_refused", return_type="boolean | number | string")
    def _bind_if_refused(
        self,
        value: Annotated[object, DSLParameter(accepted_types=("boolean", "number", "string"))],
        fallback: Annotated[object, DSLParameter(accepted_types=("boolean", "number", "string"))],
    ) -> tuple[object, object]:
        """Return `value`, or `fallback` when a model call in `value` was refused.

        `if_refused(model("guard").score(messages, text_ref("policy")), 1) > 0.5` counts a
        refusal as a score of 1; `if_refused(model("guard").complete(messages).content, "")`
        reads a refused completion as empty text. `fallback` is a literal or a parameter of
        the same kind as `value`. In a `model().split()` score predicate it may wrap only the
        chunk's score call.

        Quarry does not make a call whose model input has no content, so that call is not
        refused. Then a boolean `value` is false, and a `not` around it is true.
        """

        return value, fallback

    @dsl_compiler_form(
        name="model.input_tokens",
        return_type="integer",
        resource_group="token",
        syntax='model("alias").input_tokens(messages[, tools])',
        implicit_receiver_count=1,
    )
    def _bind_model_input_tokens(
        self,
        messages: Annotated[object, DSLParameter(accepted_types=("model_input",))],
        tools: Annotated[
            object | None,
            DSLParameter(
                accepted_types=("collection<tool_definition>",),
                required=False,
            ),
        ] = None,
    ) -> tuple[object, ...]:
        """Return the model-specific token count for the supplied messages and optional tools."""

        return (messages,) + ((tools,) if tools is not None else ())

    @dsl_compiler_form(
        name="model.split",
        return_type="collection<model_input>",
        allowed_uses=(Use.COMPREHENSION_SOURCE,),
        resource_group="token",
        syntax='model("alias").split(messages, max_input_tokens[, overlap][, context])',
        implicit_receiver_count=1,
    )
    def _bind_model_split(
        self,
        messages: Annotated[object, DSLParameter(accepted_types=("model_input",))],
        max_input_tokens: Annotated[
            object,
            DSLParameter(
                accepted_types=("integer",),
                literal_only=True,
                minimum=1,
            ),
        ],
        overlap: Annotated[
            object | None,
            DSLParameter(
                accepted_types=("integer",),
                required=False,
                literal_only=True,
                minimum=0,
            ),
        ] = None,
        context: Annotated[
            object | None,
            DSLParameter(accepted_types=("model_input",), required=False),
        ] = None,
    ) -> tuple[object, ...]:
        """Partition messages into model-sized chunks for one non-nested `any` score check.

        Each chunk is one span of the conversation within `max_input_tokens`, and nothing is
        repeated by default; a message longer than the limit is cut into ranges. The optional
        arguments keep the order `overlap`, `context` and either may be left out: `overlap` is
        the most tokens repeated from the end of the previous chunk, and `context` a view
        such as `system_messages` repeated at the start of every chunk. Tools are not
        messages: pass them to the chunk's score, outside the limit.
        """

        slots: dict[str, object] = {}
        order = ("overlap", "context")
        for option in (overlap, context):
            if option is None:
                continue
            if isinstance(option, Identifier) and option.name == "tools":
                raise UnsupportedConstruct(
                    "model().split() takes no tools: tools are not messages; "
                    "pass them to the chunk's score"
                )
            name = (
                "overlap"
                if isinstance(option, Literal) and not isinstance(option.value, str | bool)
                else "context"
            )
            if slots and order.index(name) <= order.index(next(reversed(slots))):
                raise UnsupportedConstruct(
                    "model().split() optional arguments are overlap, then context, "
                    "each at most once"
                )
            slots[name] = option
        if "context" in slots and not self._is_conversation_backed_messages(slots["context"]):
            raise UnsupportedConstruct("model().split() context must be a view of the conversation")
        return (messages, max_input_tokens, *(slots.get(name) for name in order))

    def _compile_toplevel_function(self, fc: FunctionCall, scope: Scope) -> T.Formula:
        """Boolean-valued builtins used at top level (not inside a comprehension)."""
        name = fc.name
        item = function_by_name(name)
        if item is not None and Use.CONDITION.value not in item["allowed_uses"]:
            raise UnsupportedConstruct(
                f"{name}() returns {item['return_type']} and cannot be used as a bare condition"
            )
        if name in ("called_tool", "called_tools"):
            names = [self._lit(a) for a in fc.args]
            # called_tools(a,b) => all must be called => AND of per-tool F(tool_call∧name).
            self.temporal = True
            terms = [self._tool_call_named(n) for n in names]
            return _and_all(terms)
        if name == "has_tool":
            # `any t in tools where t.name = X`: one question over every tool definition,
            # so conjoined and negated uses each keep their own meaning.
            source_start = len(self._atom_source_events)
            defined = self._tool_defined(self._lit(fc.args[0]))
            return self._tool_quantifier_atom(defined, source_start=source_start)
        if name == "has_tools":
            names = [self._lit(a) for a in fc.args]
            self.has_static = True
            return self._mint_atom(
                "tools", "name", "tool_names_all", names, domain=AtomDomain.STATIC
            )
        if name in ("user_said", "assistant_said"):
            surface = "user_messages" if name == "user_said" else "assistant_messages"
            self.temporal = True
            return T.Eventually(self._said_atom(name, fc, surface))
        if name in _EXTERNAL_FUNCS:
            self.temporal = True
            return T.Eventually(self._external_atom(fc, "messages"))
        if name == "has_sequence":
            self.temporal = True
            return self._has_sequence_formula([self._lit(a) for a in fc.args])
        raise UnsupportedConstruct(f"top-level function {name}()")

    def _has_sequence_formula(self, names: list[str]) -> T.Formula:
        """`has_sequence(a, b, …)` → the named tools called in that order and **contiguous
        within the tool-call subsequence** (``sequence[i:i+k] == names`` over the tool-call
        names: no *other tool call* between consecutive names; intervening messages/responses
        are fine).

        Encoded in LTLf as ``F( callₐ ∧ X(¬is_call U (call_b ∧ X(¬is_call U …))) )`` where
        ``call_x`` is a `tool_calls.name` atom (true only on that tool call) and
        ``is_call`` is the zero-cost `on_surface(tool_calls)` atom. "the NEXT tool call is
        b" = at the step after callₐ, no tool call occurs until a call named b."""
        if not names:
            raise UnsupportedConstruct("has_sequence() needs at least one tool name")
        call = lambda nm: self._mint_atom("tool_calls", "name", "equals", (nm,))  # noqa: E731
        is_call = self._on_surface_atom("tool_calls")
        acc = call(names[-1])
        for nm in reversed(names[:-1]):
            acc = T.And(call(nm), T.Next(T.Until(T.Not(is_call), acc)))
        return T.Eventually(acc)

    def _said_atom(self, name: str, fc: FunctionCall, surface: str) -> T.Formula:
        """`user_said(text[, case_sensitive])` / `assistant_said(...)` → a content atom.

        Quarry's ``user_said``/``assistant_said`` default to **case-insensitive**
        (``case_sensitive=False``); an optional 2nd arg can force case-sensitivity. The
        atom must honor that default, else e.g. `user_said("urgent")` misses "URGENT"."""
        text = self._lit(fc.args[0])
        case_sensitive = False
        if len(fc.args) > 1 and isinstance(fc.args[1], Literal):
            case_sensitive = bool(fc.args[1].value)
        sem = StringSemantics(case_insensitive=not case_sensitive)
        return self._mint_atom(surface, "content", "contains_any", (text,), sem=sem)

    def _compile_step_function(self, fc: FunctionCall, surface: str, scope: Scope) -> T.Formula:
        name = fc.name
        item = function_by_name(name)
        if item is not None and Use.COMPREHENSION_BODY.value not in item["allowed_uses"]:
            raise UnsupportedConstruct(
                f"{name}() returns {item['return_type']} and cannot be used as a bare "
                "comprehension condition"
            )
        if name in _EXTERNAL_FUNCS:
            return self._external_atom(fc, surface)
        if name in ("called_tool", "called_tools"):
            # A tool-call predicate inside a comprehension body: the agent called all the
            # named tools somewhere. Each is F(tool_calls.name equals X), with exact names.
            self.temporal = True
            return _and_all([self._tool_call_named(self._lit(a)) for a in fc.args])
        if name in ("user_said", "assistant_said"):
            self.temporal = True
            s = "user_messages" if name == "user_said" else "assistant_messages"
            return T.Eventually(self._said_atom(name, fc, s))
        if name == "has_sequence":
            self.temporal = True
            return self._has_sequence_formula([self._lit(a) for a in fc.args])
        raise UnsupportedConstruct(f"function {name}() in step predicate")

    def _tool_call_named(self, tool: str) -> T.Formula:
        return T.Eventually(self._mint_atom("tool_calls", "name", "equals", (tool,)))

    def _tool_defined(self, tool: str) -> T.Formula:
        # Static: a tool of this name is defined — a zero-width fact about tool
        # definitions. Lowers to a BARE static atom (true at the boot step, false on every
        # real event), so it consumes no timeline position and can't distort within:N.
        self.has_static = True
        return self._mint_atom("tools", "name", "equals", (tool,), domain=AtomDomain.STATIC)

    def _body_pinned_to_surface(self, body: T.Formula, surface: str) -> bool:
        """True if every atom the ``any`` body reads is an EVENT atom on ``surface`` (so the
        body can only be satisfied at an event of S, making the on_surface gate redundant).

        False if the body reads a static atom, an event atom on a DIFFERENT surface, a
        surface-count/on_surface aggregate, or a nested-comprehension lift — any of which can
        be true without an S-element existing, so the gate is needed. An empty body
        (T.TRUE/T.FALSE, no atoms) is NOT pinned: `any e in S where true` must still require
        an S element, so it needs the gate."""
        atom_ids = T.atoms_of(body)
        if not atom_ids:
            return False
        by_id = {s.atom_id: s for s in self.atoms.specs}
        for aid in atom_ids:
            spec = by_id.get(aid)
            if spec is None:
                return False
            if spec.domain is not AtomDomain.EVENT or spec.selector != surface:
                return False
            if spec.predicate in ("on_surface", "surface_count"):
                return False
        return True

    @staticmethod
    def _body_false_off_surface(body: T.Formula) -> bool:
        """True if a pinned body is False wherever all of its atoms ground False.

        Off the surface (and outside an active slice window) every atom of a pinned body grounds
        False, so this decides whether ``F(body)`` can only be witnessed at an element of S. A
        body with temporal operators is not a single-step predicate and keeps its lowering."""
        if not T.is_propositional(body):
            return True
        all_false = {atom_id: False for atom_id in T.atoms_of(body)}
        return T.is_const_false(T.simplify(T.restrict(body, all_false)))

    @staticmethod
    def _references_var(node, var: str) -> bool:
        """True if ``node`` names the element bound to ``var`` (a bare Identifier or the root of
        a field access like ``var.field``). A nested comprehension that rebinds ``var`` hides
        its body, so a shadowed name refers to the nested element instead."""
        if not isinstance(node, ConditionNode):
            return False
        if isinstance(node, Identifier) and node.name == var:
            return True
        if isinstance(node, FieldAccess) and _field_root_name(node) == var:
            return True
        rebinds = getattr(node, "variable", None) == var and hasattr(node, "collection")
        for field_name in condition_child_fields(node):
            if rebinds and field_name != "collection":
                continue
            value = getattr(node, field_name)
            for child in value if isinstance(value, list | tuple) else (value,):
                if _Compiler._references_var(child, var):
                    return True
        return False

    def _comparison_atom(self, node: BinaryOp, surface: str) -> T.Formula:
        """A scalar comparison in step position. Handles ``count(field,"pat") <op> n``
        and ``len(field) <op> n`` as per-step scalar-gate atoms; a bare collection count
        macro is a top-level concern (see _count_macro)."""
        left, right, op = node.left, node.right, node.operator
        # Field vs literal (=/!=). `field = null` is a NULL-CHECK (present-vs-absent),
        # NOT a string compare to "None" — quarry's FieldAccess returns None for a null
        # value or a missing key, and `= null` is true in both cases. `field = "lit"` is
        # an exact-match atom. Both operand orders.
        for fnode, vnode in ((left, right), (right, left)):
            if isinstance(fnode, FieldAccess) and isinstance(vnode, Literal) and op in ("=", "!="):
                if vnode.value is None:
                    atom = self._mint_scalar(surface, self._field_path(fnode), "is_null", "=", None)
                    return atom if op == "=" else T.Not(atom)
                atom = self._mint_scalar(
                    surface,
                    self._field_path(fnode),
                    "equals",
                    "=",
                    None,
                    patterns=(str(vnode.value),),
                )
                return atom if op == "=" else T.Not(atom)
        # Same-field additive count: `count(f,"a") + count(f,"b") [+ …] <op> n`. The
        # oracle's `count` predicate already sums occurrences over ALL its patterns on one
        # field, so a `+`-chain of same-field counts folds into ONE multi-pattern count atom
        # — no new atom kind, benefits intact. (Different-field sums are not foldable this
        # way and remain unsupported; a count feeding further arithmetic likewise.)
        for cnode, vnode, flip in ((left, right, False), (right, left, True)):
            if isinstance(cnode, BinaryOp) and cnode.operator == "+" and isinstance(vnode, Literal):
                folded = self._fold_additive_count(cnode)
                if folded is not None:
                    field_path, patterns = folded
                    cop = _FLIP.get(op, op) if flip else op
                    return self._mint_scalar(
                        surface, field_path, "count", cop, float(vnode.value), patterns=patterns
                    )
        fc, number, flipped = self._split_scalar(left, right)
        if fc is None:
            raise UnsupportedConstruct(f"comparison {op!r} not expressible in a step")
        if flipped:
            op = _FLIP.get(op, op)
        if fc.name == "len":
            value = self._bind_len_argument(*fc.args)
            return self._mint_scalar(surface, self._field_path(value), "length", op, number)
        if fc.name == "count":
            value, pattern = self._bind_count_arguments(*fc.args)
            if pattern is None:
                return self._mint_scalar(surface, self._field_path(value), "length", op, number)
            pat = self._lit(pattern)
            return self._mint_scalar(
                surface, self._field_path(value), "count", op, number, patterns=(pat,)
            )
        if fc.name in _SCORE_EXTERNAL:
            # Field-first probers (topic_drift) carry the field at arg[0] and no config arg;
            # config-first scorers (model_score/model_scan) carry config at arg[0]; inference
            # scorers (llm_score/llm_input_tokens) are whole-conversation and read ctx.
            extra: dict = {}
            atom_surface = surface
            if fc.name in _INFERENCE_SCORE_EXTERNAL:
                # (model[, constitution], conversation()) — the model is the detector arg, the
                # constitution (llm_score) folds onto the atom, and the conversation() argument
                # is a whole-conversation marker (the detector reads ctx.conversation). Ground
                # STATIC at the boot step; anchor on the system_prompt static surface (always
                # groundable) since the detector ignores the anchor text.
                detector_arg = self._lit(fc.args[0]) if fc.args else ""
                field_node = None
                if fc.name == "llm_score" and len(fc.args) > 1:
                    const = fc.args[1]
                    if isinstance(const, FunctionCall) and const.name == "text_ref":
                        # Named constitution: fold the text_ref name; the detector resolves it
                        # from config.texts at scan time via ResourceManager.resolve_text.
                        extra["constitution"] = self._lit(const.args[0]) if const.args else ""
                        extra["constitution_is_ref"] = True
                    else:
                        extra["constitution"] = self._lit(const)
                # The conversation-valued input arg selects WHICH messages the op sees, via one
                # typed selector: llm_score(model, constitution, <input>) and
                # llm_input_tokens(model, <input>). Default conversation() → whole + tools (no
                # fields → historical identity); messages / role view / slice → that Messages
                # selection (no tools); message()[.add_message()] → constructed.
                input_arg_index = 2 if fc.name == "llm_score" else 1
                if len(fc.args) > input_arg_index:
                    extra.update(self._score_model_input(fc.args[input_arg_index]))
                # A constructed input with a DYNAMIC turn — a bound comprehension value
                # (message("user", call.arguments.command)) or a direct indexed message field
                # (message("user", messages[0].content)) — makes the atom EVENT on the referenced
                # surface reading that field, so the oracle resolves the value and the detector
                # fills the dynamic turn. Otherwise llm_score/llm_input_tokens are
                # whole-conversation STATIC atoms anchored on system_prompt.
                dyn_refs = extra.pop("_dynamic_content_refs", None)
                if dyn_refs:
                    atom_surface, field_path_dynamic, dyn_paths, ordinal_val = (
                        self._resolve_dynamic_refs(dyn_refs, surface)
                    )
                    field_node = None
                    if ordinal_val is not None:
                        extra["ordinal"] = ordinal_val
                    # A single dynamic slot keeps its historical single-field representation
                    # (via value_expr_key) so its atom identity/dfa_hash is unchanged; two or
                    # more slots carry every field path, resolved off the same event source.
                    if len(dyn_paths) > 1:
                        extra["constructed_field_paths"] = dyn_paths
                else:
                    atom_surface = "system_prompt"
                    field_node = None
                    field_path_dynamic = None
            elif fc.name in _FIELD_FIRST_EXTERNAL:
                detector_arg = ""
                field_node = fc.args[0] if fc.args else None
                field_path_dynamic = None
            else:
                detector_arg = self._lit(fc.args[0]) if fc.args else ""
                field_node = fc.args[1] if len(fc.args) > 1 else None
                field_path_dynamic = None
            ordinal = None
            if fc.name in _INFERENCE_SCORE_EXTERNAL:
                field_path = (
                    self._field_path(field_node) if isinstance(field_node, FieldAccess) else ""
                )
                if field_path_dynamic is not None:
                    field_path = field_path_dynamic
            else:
                resolved = self._resolve_field_target(field_node)
                if resolved is None:
                    raise UnsupportedConstruct(
                        f"{fc.name}() text argument must be a conversation field or surface"
                    )
                resolved_surface, field_path, ordinal = resolved
                if resolved_surface is not None:
                    atom_surface = resolved_surface
            cost = _EXTERNAL_FUNCS[fc.name]
            # Fold the literal output index (and, for model_scan, threshold) onto the atom.
            # For model_score the DSL comparator (op/number) is the threshold; class_index
            # still selects which output the score is read from.
            threshold, class_index = self._external_scoring_args(fc)
            if threshold is not None:
                extra["threshold"] = threshold
            if class_index is not None:
                extra["class_index"] = class_index
            if ordinal is not None:
                extra["ordinal"] = ordinal
            spec = make_atom(
                domain=AtomDomain.EVENT if atom_surface in _EVENT_SURFACES else AtomDomain.STATIC,
                selector=atom_surface,
                value_expr_key=f"field:{field_path}" if field_path else "field:",
                predicate="external",
                patterns=(),
                cost=cost,
                op=op,
                number=number,
                detector=fc.name,
                detector_arg=detector_arg,
                **extra,
            )
            self.notes.append(f"external {fc.name}({detector_arg}) {op} {number}")
            return T.Atom(self._intern_atom(spec).atom_id)
        raise UnsupportedConstruct(f"comparison over {fc.name}() not expressible")

    def _string_set_quantifier(
        self, node: StringSetQuantifier, field_node, surface: str
    ) -> T.Formula:
        """``N of ($a,$b,…) in field`` → a boolean guard over per-pattern content atoms
        at one step: ``any`` → OR, ``all`` → AND, integer N → at-least-N (OR over N-subsets,
        bounded). Each named set is one content atom on ``field``."""
        field_path = (
            self._field_path(field_node) if isinstance(field_node, FieldAccess) else "content"
        )
        names = self._expand_set_names(node.patterns)
        if not names:
            raise UnsupportedConstruct("string-set quantifier references no known $vars")
        groups = []
        for v in names:
            patterns, sem = self._resolve_string_set(v)
            groups.append(self._mint_atom(surface, field_path, "contains_any", patterns, sem=sem))
        q = node.quantifier
        if q == "all":
            return _and_all(groups)
        if q == "any" or q == 1:
            return _or_all(groups)
        try:
            n = int(q)
        except (TypeError, ValueError) as exc:
            raise UnsupportedConstruct(f"unsupported set quantifier {q!r}") from exc
        return self._at_least_n(groups, n)

    def _expand_set_names(self, patterns) -> list[str]:
        names: list[str] = []
        keys = list(self.strings.keys())
        for pat in patterns:
            p = str(pat)
            if p.endswith("*"):
                prefix = p[:-1]
                names.extend(k for k in keys if k.startswith(prefix))
            elif p in self.strings:
                names.append(p)
            else:
                raise UnsupportedConstruct(f"undefined string set {p}")
        return list(dict.fromkeys(names))

    def _at_least_n(self, groups: list[T.Formula], n: int) -> T.Formula:
        m = len(groups)
        if n <= 0:
            return T.TRUE
        if n > m:
            return T.FALSE
        if n == m:
            return _and_all(groups)
        # One conjunction of n atoms per subset: charged before it is built, so an expansion
        # larger than the compile's work budget fails at once.
        charge(comb(m, n) * n)
        return _or_all([_and_all(list(c)) for c in combinations(groups, n)])

    def _position_or_count(self, node: BinaryOp, surface: str) -> T.Formula | None:
        """Recognise the ``@``/``#`` operator forms in a comparison, else None so ordinary
        comparisons fall through. Intra-field, single-step atoms.

        ``@a in f < @b in f`` → position-ordering atom; ``(@a - @b) <op> N`` → position
        arithmetic; ``#set in f <op> N`` → occurrence-count atom."""
        op, left, right = node.operator, node.left, node.right
        la, lb = self._as_position(left), self._as_position(right)
        if la is not None and lb is not None:
            fa, fb = la[1], lb[1]
            pa, pb = self._position_group(la[0]), self._position_group(lb[0])
            sem = self._position_semantics(la[0], lb[0])
            if fa != fb:
                # Positions from two different fields of the same event become a
                # `position_cross` atom carrying "field\x1fgroup" per operand. Still one
                # per-event boolean atom (both fields read off the same source), so lazy
                # grounding / cheap-first / minimality are unaffected.
                return self._mint_scalar(
                    surface,
                    "",
                    "position_cross",
                    op,
                    None,
                    patterns=(f"{fa}{POSITION_FIELD_SEP}{pa}", f"{fb}{POSITION_FIELD_SEP}{pb}"),
                    semantics=sem,
                )
            return self._mint_scalar(
                surface, fa, "position", op, None, patterns=(pa, pb), semantics=sem
            )
        diff = self._as_position_diff(left)
        if diff is not None and isinstance(right, Literal):
            px, py, fld, fld2, sem = diff
            if fld != fld2:
                return self._mint_scalar(
                    surface,
                    "",
                    "position_cross",
                    op,
                    float(right.value),
                    patterns=(f"{fld}{POSITION_FIELD_SEP}{px}", f"{fld2}{POSITION_FIELD_SEP}{py}"),
                    semantics=sem,
                )
            return self._mint_scalar(
                surface, fld, "position", op, float(right.value), patterns=(px, py), semantics=sem
            )
        diff = self._as_position_diff(right)
        if diff is not None and isinstance(left, Literal):
            px, py, fld, fld2, sem = diff
            if fld != fld2:
                return self._mint_scalar(
                    surface,
                    "",
                    "position_cross",
                    _FLIP.get(op, op),
                    float(left.value),
                    patterns=(f"{fld}{POSITION_FIELD_SEP}{px}", f"{fld2}{POSITION_FIELD_SEP}{py}"),
                    semantics=sem,
                )
            return self._mint_scalar(
                surface,
                fld,
                "position",
                _FLIP.get(op, op),
                float(left.value),
                patterns=(px, py),
                semantics=sem,
            )
        # Single match-position vs a literal offset: `@a <op> N` → pos(a) <op> N.
        if la is not None and isinstance(right, Literal):
            pa = self._position_group(la[0])
            return self._mint_scalar(
                surface,
                la[1],
                "position_abs",
                op,
                float(right.value),
                patterns=(pa,),
                semantics=self._position_semantics(la[0]),
            )
        if lb is not None and isinstance(left, Literal):
            pb = self._position_group(lb[0])
            return self._mint_scalar(
                surface,
                lb[1],
                "position_abs",
                _FLIP.get(op, op),
                float(left.value),
                patterns=(pb,),
                semantics=self._position_semantics(lb[0]),
            )
        ca = self._as_count(left)
        if ca is not None and isinstance(right, Literal):
            patterns, sem = self._count_group(ca[0])
            return self._mint_scalar(
                surface, ca[1], "count", op, float(right.value), patterns=patterns, semantics=sem
            )
        cb = self._as_count(right)
        if cb is not None and isinstance(left, Literal):
            patterns, sem = self._count_group(cb[0])
            return self._mint_scalar(
                surface,
                cb[1],
                "count",
                _FLIP.get(op, op),
                float(left.value),
                patterns=patterns,
                semantics=sem,
            )
        return None

    def _as_position(self, n):
        """``@var in field`` → (var, field_path); bare ``@var`` → (var, "content")
        (the element's content, the default field when none is named).
        Else None."""
        if (
            isinstance(n, BinaryOp)
            and n.operator == "in"
            and isinstance(n.left, StringPosition)
            and isinstance(n.right, FieldAccess)
        ):
            return n.left.variable, self._field_path(n.right)
        if isinstance(n, StringPosition):
            return n.variable, "content"
        return None

    @staticmethod
    def _reject_foreign_reads(node, scope: Scope) -> None:
        """A condition lowered at the current element's event may read only that element's
        fields, except in a field join (``a.f contains b.g``), which the capture or pinned
        runner evaluates across elements. Any other read of an enclosing element's field would
        be taken from the current event, so it is rejected."""
        current = scope.element_surfaces
        if not current:
            return

        def foreign(name: str | None) -> bool:
            return (
                name is not None and name not in current and scope.resolve_element(name) is not None
            )

        def visit(n) -> None:
            if isinstance(n, (AnyComprehension, AllComprehension, FilterComprehension)):
                return
            if (
                isinstance(n, BinaryOp)
                and n.operator == "contains"
                and isinstance(n.left, FieldAccess)
                and isinstance(n.right, FieldAccess)
            ):
                return
            root = None
            if isinstance(n, FieldAccess):
                root = _field_root_name(n)
            elif (
                isinstance(n, BinaryOp)
                and n.operator == "matches"
                and isinstance(n.left, Identifier)
            ):
                root = n.left.name
            if foreign(root):
                raise UnsupportedConstruct(
                    f"a field of `{root}` is read inside a comprehension over another element, "
                    "where it would be read at that element's event; move the condition on "
                    f"`{root}` outside the comprehension, or join with `... contains {root}.field`"
                )
            for name in condition_child_fields(n):
                value = getattr(n, name)
                for child in value if isinstance(value, list | tuple) else (value,):
                    if isinstance(child, ConditionNode):
                        visit(child)

        visit(node)

    def _as_count(self, n):
        """``#var in field`` → (var, field_path); bare ``#var`` → (var, "content").
        Else None."""
        if (
            isinstance(n, BinaryOp)
            and n.operator == "in"
            and isinstance(n.left, StringCount)
            and isinstance(n.right, FieldAccess)
        ):
            return n.left.variable, self._field_path(n.right)
        if isinstance(n, StringCount):
            return n.variable, "content"
        return None

    def _as_position_diff(self, n):
        """``(@x in fX) - (@y in fY)`` → (group_x, group_y, field_x, field_y), else None.
        ``field_x == field_y`` is the intra-field case (`position` atom); differing fields
        are the cross-field case (`position_cross`), handled by the caller."""
        if not (isinstance(n, BinaryOp) and n.operator == "-"):
            return None
        lx, ly = self._as_position(n.left), self._as_position(n.right)
        if lx is None or ly is None:
            return None
        return (
            self._position_group(lx[0]),
            self._position_group(ly[0]),
            lx[1],
            ly[1],
            self._position_semantics(lx[0], ly[0]),
        )

    def _position_semantics(self, *variables: str) -> StringSemantics:
        """The one set of string modifiers shared by every pattern behind ``@`` operands.

        A position atom carries a single ``StringSemantics``, so operands whose declared
        modifiers differ (``@a`` nocase against ``@b`` case-sensitive) cannot be expressed."""
        semantics = set()
        for variable in variables:
            key = variable if variable.startswith("$") else "$" + variable
            names = self._expand_set_names((key,)) if key.endswith("*") else (key,)
            semantics.update(self._resolve_string_set(name)[1] for name in names)
        if len(semantics) > 1:
            raise UnsupportedConstruct(
                "match positions of strings with different modifiers cannot be compared"
            )
        return semantics.pop() if semantics else StringSemantics()

    def _position_group(self, var: str) -> str:
        """A match-position operand as a NUL-joined pattern *group* (see
        ``POSITION_GROUP_SEP``). ``@a`` → the single pattern; ``@a*`` → every ``$a*``
        string's pattern, whose match position the oracle takes the EARLIEST of. Each
        contributing var must
        be single-pattern (a multi-pattern *list* var has list-order, not earliest,
        semantics — still rejected)."""
        key = var if var.startswith("$") else "$" + var
        if key.endswith("*"):
            names = self._expand_set_names((key,))
            if not names:
                raise UnsupportedConstruct(f"@{var} matches no string set")
            group: list[str] = []
            for name in names:
                patterns, _ = self._resolve_string_set(name)
                if len(patterns) != 1:
                    raise UnsupportedConstruct(
                        f"@{var}: wildcard member {name} is multi-pattern (unsupported)"
                    )
                group.append(patterns[0])
            return POSITION_GROUP_SEP.join(group)
        patterns, _ = self._resolve_string_set(key)
        if len(patterns) != 1:
            raise UnsupportedConstruct(f"@{var} needs a single-pattern string set")
        return patterns[0]

    def _count_group(self, var: str) -> tuple[tuple[str, ...], StringSemantics]:
        """Resolve ``#name`` or ``#prefix*`` to one occurrence-count pattern group.

        One count atom carries one ``StringSemantics``, so wildcard members must use the
        same modifiers. Occurrences of every resolved pattern are summed by the oracle.
        """

        key = var if var.startswith("$") else "$" + var
        names = self._expand_set_names((key,)) if key.endswith("*") else (key,)
        if not names:
            raise UnsupportedConstruct(f"#{var} matches no string set")
        patterns: list[str] = []
        semantics: set[StringSemantics] = set()
        for name in names:
            member_patterns, member_semantics = self._resolve_string_set(name)
            patterns.extend(member_patterns)
            semantics.add(member_semantics)
        if len(semantics) > 1:
            raise UnsupportedConstruct(
                "occurrence counts of strings with different modifiers cannot be combined"
            )
        return tuple(patterns), semantics.pop() if semantics else StringSemantics()

    def _fold_additive_count(self, node) -> tuple[str, tuple[str, ...]] | None:
        """A `+`-chain of `count(field, "pat")` terms all on the SAME field →
        ``(field_path, (pat, …))``, else None. Sound because the oracle's ``count``
        predicate sums occurrences over all patterns on one field, so
        ``count(f,a)+count(f,b)`` == a single count atom with patterns ``(a, b)`` (repeats
        included → the term is counted twice, as the arithmetic intends)."""
        if isinstance(node, BinaryOp) and node.operator == "+":
            lhs = self._fold_additive_count(node.left)
            rhs = self._fold_additive_count(node.right)
            if lhs is None or rhs is None or lhs[0] != rhs[0]:
                return None
            return (lhs[0], lhs[1] + rhs[1])
        if isinstance(node, FunctionCall) and node.name == "count" and len(node.args) == 2:
            if not isinstance(node.args[0], FieldAccess):
                return None
            value, pattern = self._bind_count_arguments(*node.args)
            return (self._field_path(value), (self._lit(pattern),))
        return None

    @staticmethod
    def _split_scalar(left, right):
        # A scalar comparison peels a FunctionCall off one side and a NUMERIC literal off the
        # other. A non-numeric literal (a string, e.g. ``llm_complete(...) = "unsafe"``) is NOT
        # a scalar comparison — return no split so the caller falls through to its string/other
        # handling rather than crashing on ``float("unsafe")``.
        def _numeric(node) -> bool:
            return (
                isinstance(node, Literal)
                and not isinstance(node.value, bool)
                and isinstance(node.value, (int, float))
            )

        if isinstance(left, FunctionCall) and _numeric(right):
            return left, float(right.value), False
        if isinstance(right, FunctionCall) and _numeric(left):
            return right, float(left.value), True
        return None, None, False

    def _complete_comparison(self, node: BinaryOp) -> T.Formula | None:
        """Route ``llm_complete(model, <input>) =/!=/~/matches "literal"`` to a folded boolean
        atom. Returns ``None`` when neither operand is an ``llm_complete(...)`` call, so the
        caller falls through to its normal handling.

        Operand direction matters for the match operators. ``=``/``!=`` are commutative, so the
        completion may be on either side. ``~``/``matches`` are NOT: ``lhs ~ rhs`` matches the
        regex ``rhs`` against text ``lhs``. So ``llm_complete(...)`` must be on the LEFT; the
        reversed ``"regex" ~ llm_complete(...)`` is rejected at compile rather than silently
        swapping operands (which would treat the completion as a regex over the literal)."""

        op = node.operator
        left_call = (
            node.left
            if isinstance(node.left, FunctionCall) and node.left.name == _complete_func()
            else None
        )
        right_call = (
            node.right
            if isinstance(node.right, FunctionCall) and node.right.name == _complete_func()
            else None
        )
        if op in ("~", "matches"):
            if left_call is not None:
                return self._complete_predicate(left_call, op, node.right)
            if right_call is not None:
                raise UnsupportedConstruct(
                    f'{_complete_func()}(...) {op} "regex" requires {_complete_func()}() on the '
                    "LEFT; the reversed form is not supported (it would match the completion as "
                    "a regex against the literal)"
                )
            return None
        if left_call is not None:
            atom = self._complete_predicate(left_call, op, node.right)
        elif right_call is not None:
            atom = self._complete_predicate(right_call, op, node.left)
        else:
            return None
        return T.Not(atom) if op == "!=" else atom

    def _complete_predicate(self, fc: FunctionCall, op: str, literal_node) -> T.Formula:
        """``llm_complete(model, <input>) <op> "literal"`` → one boolean external atom.

        Generate-then-compare: the configured backend produces a completion and the folded
        literal is applied as exact equality (``=``/``!=``) or a regex match (``~``/``matches``);
        ``!=`` is the caller wrapping this atom in ``Not``. Whole-conversation STATIC (the
        detector reads the selected conversation, not a per-event field), grounded at the boot
        step. The operator is folded onto ``op`` (``=`` exact vs ``~`` match) and the literal
        onto ``patterns``. Dynamic constructed content is rejected — there is no per-event value
        to fill for a whole-conversation completion."""

        if not (isinstance(literal_node, Literal) and isinstance(literal_node.value, str)):
            raise UnsupportedConstruct(
                f"{_complete_func()}(...) may only be compared to a string literal, "
                f"not {literal_node!r}"
            )
        detector_arg = self._lit(fc.args[0]) if fc.args else ""
        input_extra = self._score_model_input(fc.args[1]) if len(fc.args) > 1 else {}
        if "_dynamic_content_refs" in input_extra:
            raise UnsupportedConstruct(
                f"{_complete_func()}() does not support dynamic constructed content; its model "
                "input must be a static selection"
            )
        spec = make_atom(
            domain=AtomDomain.STATIC,
            selector="system_prompt",
            value_expr_key="field:",
            predicate="external",
            patterns=(literal_node.value,),
            cost=Cost.LLM,
            op="~" if op in ("~", "matches") else "=",
            detector=_complete_func(),
            detector_arg=detector_arg,
            **input_extra,
        )
        self.has_static = True
        self.notes.append(f"external {_complete_func()}() {op} literal")
        return T.Atom(self._intern_atom(spec).atom_id)

    @dsl_compiler_form(
        name="conversation",
        return_type="model_input",
    )
    def _compile_conversation_selector(self) -> tuple[str]:
        """Select the complete conversation, including offered tool definitions."""

        return ("conversation",)

    def _resolve_selection(self, node):
        """Normalize a model-input expression to an ORDERED selection pipeline.

        Returns one of:
          - ``("conversation",)`` — the whole transcript WITH offered tools (``conversation()``,
            unmodified); the default, which folds to the empty selector.
          - ``("pipeline", steps)`` — a Messages selection (NO tools), where ``steps`` is a tuple
            of ordered operations over the root ``Messages``: ``("role", name)`` filters and
            ``("slice", start, stop)`` windows, applied left-to-right at grounding time.
          - ``None`` — not a recognized selection (a field, ``tool_calls``, a literal, a builder);
            the caller routes or rejects it. It is NEVER widened to the whole conversation.

        Composition is preserved exactly: ``messages[:3].user_messages`` (slice then filter) and
        ``messages.user_messages[:3]`` (filter then slice) produce different pipelines, and a
        cross-role chain (``messages.user_messages.assistant_messages``) keeps both filters (an
        empty result at runtime). ``.messages`` is the Conversation→Messages boundary only — it is
        rejected on an already-Messages value, so ``user_messages.messages`` does not widen."""

        if isinstance(node, FunctionCall) and node.name == "_slice" and len(node.args) == 3:
            base, start_node, stop_node = node.args
            inner = self._resolve_selection(base)
            if inner is None or inner[0] == "conversation":
                return None
            start = start_node.value if isinstance(start_node, Literal) else None
            stop = stop_node.value if isinstance(stop_node, Literal) else None
            return ("pipeline", (*inner[1], ("slice", start, stop)))
        if isinstance(node, FunctionCall) and node.name == "conversation" and not node.args:
            return self._compile_conversation_selector()
        if isinstance(node, Identifier) and node.name in _MODEL_INPUT_SURFACES:
            if node.name == "messages":
                return ("pipeline", ())
            return ("pipeline", (("role", node.name),))
        # A field-access chain. ``.messages`` is valid ONLY as the first hop off ``conversation()``
        # (the Conversation→Messages boundary); every other field must be a role view, appended as
        # an ordered filter. ``.messages`` on an already-Messages value is rejected (no widen).
        if isinstance(node, FieldAccess) and node.fields:
            inner = self._resolve_selection(node.base)
            if inner is None:
                return None
            from_conversation = inner[0] == "conversation"
            steps = () if from_conversation else inner[1]
            for position, field in enumerate(node.fields):
                if field == "messages":
                    if not (from_conversation and steps == () and position == 0):
                        return None
                    continue
                if field in _ROLE_VIEW_SURFACES:
                    steps = (*steps, ("role", field))
                else:
                    return None
            return ("pipeline", steps)
        return None

    def _score_model_input(self, node) -> dict:
        """Resolve an inference op's conversation-valued input arg to folded atom extras.

        One typed selector, shared by llm_score / llm_input_tokens / llm_complete. Every
        supported message expression normalizes to the SAME reference-backed Messages selection,
        regardless of spelling:

        - ``conversation()`` → ``{}`` = the whole transcript WITH the request's offered tools
          (the default; no extra fields → identical historical atom identity/dfa_hash).
        - a Messages selection — any of ``messages`` / ``conversation().messages``
          / a role view (``user_messages`` …) / ``messages.user_messages`` /
          ``conversation().user_messages`` — → that Messages selection, carrying NO tools.
        - any of the above sliced (``messages[a:b]`` / ``messages.user_messages[-1:]`` /
          ``conversation().messages[-4:]``) → exactly the sliced canonical messages (no tools).
        - ``message(role, content)[.add_message(...)]*`` → a constructed Messages value.

        Any other shape (``tool_calls``, a field, a bare literal) is not a conversation/Messages
        value and is REJECTED — never widened to the full conversation — so the rule fails to
        compile rather than silently scoring the wrong thing."""

        selection = self._resolve_selection(node)
        if selection is not None:
            if selection[0] == "conversation":
                return {}
            return {"input_pipeline": selection[1]}
        if isinstance(node, FunctionCall) and node.name in ("message", "add_message"):
            turns, dynamic_refs = self._flatten_constructed_input(node)
            extra = {"constructed_input": turns}
            if dynamic_refs:
                # Transient key (popped by the caller, not an atom field): the field-access nodes
                # whose values fill the dynamic turns, in slot order. Only llm_score can serve
                # them (as a per-element EVENT atom); the whole-conversation ops reject them.
                extra["_dynamic_content_refs"] = dynamic_refs
            return extra
        raise UnsupportedConstruct(
            "model input must be conversation(), messages / a role view (optionally sliced), "
            f"or a message(...) builder, not {node!r}"
        )

    def _resolve_dynamic_refs(self, dyn_refs, surface: str):
        """Resolve the dynamic constructed-content refs to ``(surface, first_path, paths,
        ordinal)``.

        Every ref must resolve to a message field, and ALL refs must reference the SAME event
        source — the same surface and the same ordinal — because one EVENT atom grounds at one
        event and reads every field off that single source. A bound reference (no explicit
        surface, e.g. ``call.arguments.command``) rides the enclosing comprehension surface; an
        indexed reference (``messages[0].content``) names its own surface + ordinal. Mixed
        sources (e.g. ``messages[0].content`` and ``messages[1].content``) are rejected."""

        resolved = []
        for ref in dyn_refs:
            target = self._resolve_field_target(ref)
            if target is None:
                raise UnsupportedConstruct(
                    f"dynamic message() content must be a message field, not {ref!r}"
                )
            resolved.append(target)
        surfaces = {(t[0] if t[0] is not None else surface) for t in resolved}
        ordinals = {t[2] for t in resolved}
        if len(surfaces) != 1 or len(ordinals) != 1:
            raise UnsupportedConstruct(
                "all dynamic message() content fields must reference the SAME message or bound "
                'element (one event source); e.g. message("user", call.arguments.command)'
                '.add_message("user", call.arguments.args), not fields of different messages'
            )
        atom_surface = surfaces.pop()
        if atom_surface not in _EVENT_SURFACES:
            raise UnsupportedConstruct(
                "dynamic message() content must reference an event-surface field — a bound "
                "comprehension element or a direct indexed message field, e.g. "
                'message("user", call.arguments.command) or message("user", messages[0].content)'
            )
        element_type = _SURFACE_ELEMENT_TYPES.get(atom_surface, "object")
        paths = tuple(self._dynamic_text_path(element_type, t[1]) for t in resolved)
        return atom_surface, paths[0], paths, ordinals.pop()

    @staticmethod
    def _dynamic_text_path(element_type: str, path: str) -> str:
        """The field path a dynamic ``message()`` content reads, as text: ``*`` (every value of
        an object, a list) is rejected, and a field with a text form (``_MODEL_TEXT_FIELDS``)
        reads that text, as in a model expression's ``message()``."""

        *owner_path, last = path.split(".")
        if last == "*":
            raise UnsupportedConstruct(
                "message content must resolve to a string or text_ref, not object[]"
            )
        owner = element_type
        for name in owner_path:
            owner = _MODEL_FIELD_TYPES.get(owner, {}).get(name, "object")
        text_field = _MODEL_TEXT_FIELDS.get(owner, {}).get(last)
        return ".".join([*owner_path, text_field]) if text_field is not None else path

    def _flatten_constructed_input(self, node):
        """Flatten a ``message(...)[.add_message(...)]*`` chain to turns + its dynamic refs.

        Returns ``(turns, dynamic_refs)``. Each turn is ``(role, content)`` where content is a
        STRING literal (preserved exactly — a numeric/boolean/null value is rejected, never
        ``str()``-coerced) OR ``None`` marking a dynamic slot whose string content is a field
        access resolved at scan time: a direct indexed message field (``messages[0].content``)
        or a bound comprehension value (``call.arguments.command``). ``dynamic_refs`` is the
        tuple of those field-access nodes, in slot order (empty when the input is all-literal).
        Several dynamic slots are allowed; the caller requires that they all reference the SAME
        event source (checked when the atom is built). Roles follow ``message()``'s contract: a
        valid non-``tool`` OpenAI role."""

        if isinstance(node, FunctionCall) and node.name == "message":
            if len(node.args) != 2:
                raise UnsupportedConstruct("message() takes exactly (role, content)")
            return self._compile_message(*node.args)
        if isinstance(node, FunctionCall) and node.name == "add_message" and len(node.args) == 3:
            base, role_node, content_node = node.args
            prev, prev_dyn = self._flatten_constructed_input(base)
            role = self._string_literal(role_node, "add_message() role")
            content, dyn = self._builder_content(content_node, "add_message() content")
            dynamic_refs = (*prev_dyn, dyn) if dyn else prev_dyn
            return (*prev, (self._require_builder_role(role), content)), dynamic_refs
        raise UnsupportedConstruct(
            "constructed model input must be message(role, content) optionally chained with "
            f".add_message(role, content), not {node!r}"
        )

    def _lower_model_content(self, node, scope: Scope) -> tuple[ModelExpression, str]:
        """Lower ``message()`` content: text, or the text ``join()`` makes."""

        if isinstance(node, FunctionCall) and node.name == "join":
            return self._lower_join(node, scope)
        return self._lower_model_text(node, scope)

    def _lower_join(self, node: FunctionCall, scope: Scope) -> tuple[ModelExpression, str]:
        """Lower ``join([value for item in source if filter], separator)``."""

        try:
            values, separator = self._bind_join(*node.args)
        except TypeError as exc:
            raise UnsupportedConstruct("join() takes a map comprehension and a separator") from exc
        if not isinstance(values, MapComprehension):
            raise UnsupportedConstruct(
                "join() values must be a map comprehension, e.g. "
                "[c.name for c in messages[-1].tool_calls]"
            )
        if not (isinstance(separator, Literal) and isinstance(separator.value, str)):
            raise UnsupportedConstruct("join() separator must be a string literal")
        for part in (values.expression, values.condition):
            if part is not None and self._contains_model_call(part):
                # One model input holds every value, so a value cannot need its own model call.
                raise UnsupportedConstruct("a map comprehension cannot call a model")
        source, source_type = self._lower_model_expression(values.collection, scope)
        if not source_type.endswith("[]"):
            raise UnsupportedConstruct(
                "a map comprehension needs a collection of structured items, such as a "
                "message's tool_calls (messages[-1].tool_calls, assistant_messages[-1]."
                f"tool_calls, …), not {source_type}"
            )
        item_scope = scope.bind(values.variable, _MAP_ITEM_SURFACE, source_type[:-2])
        predicate = ModelExpression("literal", True)
        if values.condition is not None:
            predicate, predicate_type = self._lower_model_expression(values.condition, item_scope)
            if predicate_type != "bool":
                raise UnsupportedConstruct("a map comprehension filter must be a condition")
        projection, projection_type = self._lower_model_text(values.expression, item_scope)
        root = (
            _field_root_name(values.expression)
            if isinstance(values.expression, FieldAccess)
            else None
        )
        if (
            root is not None
            and item_scope.resolve_element(root) == _MAP_ITEM_SURFACE
            and projection_type in _STEP_TEXT_TYPES
        ):
            # The value a filter test on this item reads is field_text: a list (an argv, or any
            # field whose JSON value is an array), a number, or a bool renders as the text the
            # filter matched, instead of being silently left out as "not text".
            projection, projection_type = (
                ModelExpression("field_text", None, (projection,)),
                "string",
            )
        if projection_type not in {"string", "string?"}:
            raise UnsupportedConstruct(f"join() values must be text, not {projection_type}")
        mapped = ModelExpression("map", values.variable, (source, predicate, projection))
        return ModelExpression("join", separator.value, (mapped,)), "string"

    @dsl_compiler_form(name="join", return_type="string")
    def _bind_join(
        self,
        values: Annotated[object, DSLParameter(accepted_types=("collection<string>",))],
        separator: Annotated[object, DSLParameter(accepted_types=("string",), literal_only=True)],
    ) -> tuple[object, object]:
        """Combine (concatenate) the text values of a map comprehension with a separator.

        `separator` goes between two values. Use `join()` as `message()` content in a version 2
        model input. For example, `message("user", join([c.arguments.command for c in
        messages[-1].tool_calls if $risky in c.arguments.*], "\\n"))` gives the commands of the
        risky calls in the newest message to the model as one input.

        `join()` combines the values in the order of the collection, each rendered as the same
        text the same test in a rule step reads from it: a list is its elements joined with
        `"\\n"`, and a number or a boolean is its text. It leaves out a value that is missing,
        null, or empty, and keeps a value that is only whitespace. One value gives that value
        unchanged.

        When no value stays, the model input has no content. Quarry does not call the model, and
        the comparison that reads the result is false.

        When Quarry enforces its limits, `join()` counts the encoded bytes of the text it makes
        as it builds it. A text at or under `max_tool_argument_size` bytes does not change. A
        longer text fails the rule, rather than truncating it: a benign value must not be able to
        push a risky value out of the judge's input unnoticed.

        """

        return values, separator

    @dsl_compiler_form(
        name="message",
        return_type="model_input",
    )
    def _compile_message(
        self,
        role_node: Annotated[
            object,
            DSLParameter(
                accepted_types=("string",),
                name="role",
                literal_only=True,
                allowed_values=("system", "developer", "user", "assistant"),
            ),
        ],
        content_node: Annotated[
            object,
            DSLParameter(accepted_types=("string",), name="content"),
        ],
    ) -> tuple[tuple[tuple[str, str | None], ...], tuple[object, ...]]:
        """Build a one-message model input from a role and literal or field-backed content.

        In a version 2 model expression, `content` can also be the text that `join()` gives.
        """

        role = self._string_literal(role_node, "message() role")
        content, dynamic_ref = self._builder_content(content_node, "message() content")
        turns = ((self._require_builder_role(role), content),)
        return turns, ((dynamic_ref,) if dynamic_ref else ())

    def _builder_content(self, node, what: str):
        """A constructed-turn content: a string literal → ``(literal, None)``; a field access
        (indexed message field or bound comprehension value) → ``(None, node)`` marking a
        dynamic slot. Any other value (number/bool/null/collection) is rejected — never
        coerced."""

        if isinstance(node, Literal) and isinstance(node.value, str):
            return node.value, None
        if isinstance(node, FieldAccess):
            return None, node
        if isinstance(node, FunctionCall) and node.name == "join":
            raise UnsupportedConstruct(_LEGACY_COMPUTED_CONTENT)
        if isinstance(node, MapComprehension):
            raise UnsupportedConstruct(_MAP_COMPREHENSION_FORM)
        raise UnsupportedConstruct(
            f"{what} must be a string literal or a string field (an indexed message field or a "
            f"bound comprehension value), not {type(node).__name__}"
        )

    @staticmethod
    def _string_literal(node, what: str) -> str:
        """A string-literal argument, preserved exactly. A non-string value (number, bool,
        null, collection) is rejected — never coerced with ``str()``."""

        if isinstance(node, Literal) and isinstance(node.value, str):
            return node.value
        raise UnsupportedConstruct(f"{what} must be a string literal, not {node!r}")

    @staticmethod
    def _require_builder_role(role: str) -> str:
        """A message() builder role: a valid non-``tool`` OpenAI role."""

        allowed = {"system", "developer", "user", "assistant"}
        if role not in allowed:
            raise UnsupportedConstruct(
                f"message() role {role!r} must be one of {sorted(allowed)} (not 'tool')"
            )
        return role

    @contextlib.contextmanager
    def _in_window(
        self,
        surface: str,
        window: tuple[int | None, int | None] | None,
        index: int | None = None,
    ):
        """Restrict EVENT atoms minted on ``surface`` to the slice ``window`` or to the one
        event at ``index`` (both ``None``: unrestricted), and restore the enclosing
        restriction afterwards."""
        prev_window = self._active_window
        restricted = window is not None or index is not None
        self._active_window = (surface, window, index) if restricted else None
        try:
            yield
        finally:
            self._active_window = prev_window

    def _apply_window(self, spec):
        """Attach the active restriction to ``spec`` when it is an EVENT atom on the
        restricted surface: a fixed index as the atom's ``ordinal``, a window as its slice.
        Other atoms (other surfaces, static atoms, or none active) are returned unchanged."""
        active = self._active_window
        if active is None:
            return spec
        surface, window, index = active
        if spec.selector != surface or spec.domain != AtomDomain.EVENT:
            return spec
        # The position is part of the atom's content identity. Clear the unrestricted
        # hash so the interner recomputes it instead of collapsing distinct positions.
        if index is not None:
            return spec if spec.ordinal is not None else replace(spec, atom_id="", ordinal=index)
        if window is None or spec.is_slice:
            return spec
        return replace(
            spec,
            atom_id="",
            slice_start=window[0],
            slice_stop=window[1],
            is_slice=True,
        )

    @staticmethod
    def _find_flow(node):
        if isinstance(node, FlowOp):
            return node
        if isinstance(node, BinaryOp) and node.operator in ("and", "or"):
            return _Compiler._find_flow(node.left) or _Compiler._find_flow(node.right)
        if isinstance(node, UnaryOp) and node.operator == "not":
            return _Compiler._find_flow(node.operand)
        return None

    def _field_path(self, node) -> str:
        if isinstance(node, Identifier):
            return ""
        if isinstance(node, FieldAccess):
            if node.base is None:
                return ".".join(str(f) for f in node.fields[1:])
            base = self._field_path(node.base)
            tail = ".".join(str(f) for f in node.fields)
            return f"{base}.{tail}" if base else tail
        # A STATIC subscript into a field value: `field.items[0]` parses to
        # `_index(FieldAccess(...), Literal(0))`. The index is appended as a numeric path
        # segment, which the oracle's `matchers.field_access` grounds as a list index
        # (`items.0`) — exactly quarry's `_index(list, int)`. Only a LITERAL integer index
        # is static enough to intern into an atom's field key; a computed/non-literal index
        # is a data-dependent path (rejected) since the atom key must be fixed.
        if isinstance(node, FunctionCall) and node.name == "_index" and len(node.args) == 2:
            inner, idx = node.args
            if (
                isinstance(idx, Literal)
                and isinstance(idx.value, (int, str))
                and str(idx.value).lstrip("-").isdigit()
            ):
                base = self._field_path(inner)
                seg = str(int(idx.value))
                return f"{base}.{seg}" if base else seg
            raise UnsupportedConstruct("indexed field access needs a literal integer index")
        raise UnsupportedConstruct(f"unsupported field accessor {type(node).__name__}")

    def _content_atom(self, field_node, value_node, surface: str, predicate: str) -> T.Formula:
        fld = self._field_path(field_node)
        patterns, sem = self._resolve_patterns(value_node, force_regex=(predicate == "regex"))
        return self._mint_atom(surface, fld, predicate, patterns, sem=sem)

    def _resolve_patterns(self, node, force_regex=False):
        if isinstance(node, Identifier) and node.name.startswith("$"):
            return self._resolve_string_set(node.name, force_regex)
        if isinstance(node, Literal):
            sem = StringSemantics(is_regex=force_regex)
            return (str(node.value),), sem
        if isinstance(node, list):
            return tuple(str(v) for v in node), StringSemantics(is_regex=force_regex)
        raise UnsupportedConstruct(f"content value {type(node).__name__}")

    def _resolve_string_set(self, var_name: str, force_regex=False):
        val = self.strings.get(var_name)
        if val is None:
            raise UnsupportedConstruct(f"undefined string set {var_name}")
        sem = StringSemantics(is_regex=force_regex)
        if isinstance(val, tuple) and len(val) == 2 and hasattr(val[1], "__dict__"):
            inner, mods = val
            sem = StringSemantics(
                case_insensitive=bool(getattr(mods, "nocase", False)),
                fullword=bool(getattr(mods, "fullword", False)),
                base64=bool(getattr(mods, "base64", False)),
                is_regex=force_regex or bool(getattr(mods, "regex", False)),
            )
            val = inner
        if isinstance(val, str):
            patterns = (val,)
        elif isinstance(val, (list, tuple)):
            patterns = tuple(str(v) for v in val)
        else:
            patterns = (str(val),)
        return patterns, sem

    @staticmethod
    def _content_root_name(node: BinaryOp):
        """Root identifier name of a top-level content op's field side (or None). For
        ``in`` the field is on the right; otherwise on the left."""
        field_node = node.right if node.operator == "in" else node.left
        root = field_node
        while isinstance(root, FieldAccess) and root.base is not None:
            root = root.base
        if isinstance(root, FieldAccess) and root.base is None and root.fields:
            return str(root.fields[0])
        if isinstance(root, Identifier):
            return root.name
        return None

    def _static_system_prompt(self, node: BinaryOp) -> T.Formula:
        """`system_prompt contains/~ X` → a bare STATIC atom (selector=system_prompt).
        Holds at the boot step only; grounded once by the executor."""
        value_node = node.left if node.operator == "in" else node.right
        predicate = "regex" if node.operator in ("~", "matches") else "contains_any"
        patterns, sem = self._resolve_patterns(value_node, force_regex=(predicate == "regex"))
        return self._mint_atom(
            "system_prompt", "", predicate, patterns, sem=sem, domain=AtomDomain.STATIC
        )

    @staticmethod
    def _lit(node) -> str:
        if isinstance(node, Literal) and isinstance(node.value, str):
            return node.value
        raise UnsupportedConstruct("expected a string literal argument")

    @staticmethod
    def _lit_float(node) -> float:
        """A numeric literal argument (threshold) → float. A computed/non-literal value is
        rejected: it is part of the atom's content identity, so it must be static (same rule
        as the literal string/index handling)."""
        if (
            isinstance(node, Literal)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return float(node.value)
        raise UnsupportedConstruct("expected a numeric literal threshold argument")

    @staticmethod
    def _lit_index(node) -> int:
        """A non-negative integer literal argument (class/output index) → int. Non-literal or
        non-integer indices are rejected to keep atom identity static."""
        if (
            isinstance(node, Literal)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            if node.value < 0:
                raise UnsupportedConstruct("class_index must be a non-negative integer literal")
            return node.value
        raise UnsupportedConstruct("expected an integer literal class_index argument")

    @classmethod
    def _external_scoring_args(cls, fc: FunctionCall) -> tuple[float | None, int | None]:
        """Extract the literal ``(threshold, class_index)`` a scored external call folds into
        its atom. Arg layout is per-function (config + field occupy args 0–1):

          - ``model_scan(model, field, threshold, class_index)`` — arg 2 = threshold,
            arg 3 = class_index.
          - ``model_score(model, field, class_index)`` — arg 2 = class_index; no threshold
            (its threshold is the folded DSL comparator).

        Missing trailing args return ``None`` (function default applies). Non-literal args
        are rejected by the ``_lit_*`` helpers so atom identity stays static."""
        threshold: float | None = None
        class_index: int | None = None
        if fc.name == "model_scan":
            if len(fc.args) > 2:
                threshold = cls._lit_float(fc.args[2])
            if len(fc.args) > 3:
                class_index = cls._lit_index(fc.args[3])
        elif fc.name == "model_score":
            if len(fc.args) > 2:
                class_index = cls._lit_index(fc.args[2])
        return threshold, class_index

    @staticmethod
    def _scalar_alias(node):
        """If ``node`` is ``<scalar-expr> as v`` (an Assignment whose expression is NOT a
        string `<pat> in field` match), return ``(v, expr)``; else None. Such an alias
        names a scalar/collection value to reuse (``len(messages) as count``,
        ``get_tool_calls(x) as c``) — as opposed to a value capture (handled separately)."""
        if not isinstance(node, Assignment):
            return None
        var = str(getattr(node, "variable", "") or "")
        expr = getattr(node, "expression", None)
        if not var or expr is None:
            return None
        # A string value capture (`<pat> in field as v`) is NOT a scalar alias.
        if isinstance(expr, BinaryOp) and expr.operator == "in":
            return None
        return var, expr

    def _reject_capture_outside_positive_existentials(self, node, var: str) -> None:
        """A capture reads as ``∃v φ(v)``, evaluated by running the rule once per candidate
        value. That equals the rule only where the value is bound in monotone, existential
        position: a bind under ``not`` would be ``∃v ¬φ(v)``, and one inside ``all`` would
        move ``∃v`` across the universal. The sustained side of ``until`` also synthesizes
        a universal over prior events. Such binds are rejected instead of evaluated with
        the wrong meaning. A reference to a positively bound value may sit anywhere."""

        def binds_capture(n) -> bool:
            for child in walk_condition_nodes(n):
                if isinstance(child, Assignment) and str(getattr(child, "variable", "")) == var:
                    return True
            return False

        def visit(n, guarded: bool) -> None:
            if isinstance(n, UnaryOp) and n.operator == "not":
                visit(n.operand, True)
            elif isinstance(n, AllComprehension):
                visit(n.condition, True)
            elif isinstance(n, FlowOp):
                if n.operator == "until" and binds_capture(n.left):
                    raise UnsupportedConstruct(
                        f"value capture `{var}` on the sustained side of `until` is not "
                        "supported; move the capture outside the sustained operand"
                    )
                visit(n.left, guarded)
                visit(n.right, guarded)
            elif isinstance(n, BinaryOp) and n.operator in ("and", "or"):
                visit(n.left, guarded)
                visit(n.right, guarded)
            elif isinstance(n, AnyComprehension):
                visit(n.condition, guarded)
            elif guarded and binds_capture(n):
                raise UnsupportedConstruct(
                    f"value capture `{var}` under `not` or inside `all` is not supported"
                )

        visit(node, False)

    @staticmethod
    def _find_assignment(node):
        if isinstance(node, Assignment):
            return node
        for attr in ("left", "right", "operand", "condition"):
            child = getattr(node, attr, None)
            found = _Compiler._find_assignment(child) if child is not None else None
            if found is not None:
                return found
        return None

    def _find_reference(self, node, var: str):
        """First back-reference to ``var``: ``var in field`` or ``field contains var``."""
        if isinstance(node, BinaryOp):
            if (
                node.operator == "in"
                and isinstance(node.left, Identifier)
                and node.left.name == var
            ):
                return node
            if (
                node.operator == "contains"
                and isinstance(node.right, Identifier)
                and node.right.name == var
            ):
                return node
        for attr in ("left", "right", "operand", "condition"):
            child = getattr(node, attr, None)
            if child is not None:
                found = self._find_reference(child, var)
                if found is not None:
                    return found
        return None

    def _mint_capture(
        self,
        surface: str,
        field_path: str,
        role: str,
        predicate: str = "contains_any",
        patterns: tuple[str, ...] = (),
        sem: StringSemantics | None = None,
    ) -> T.Formula:
        """A value-parameterized capture atom; the executor substitutes the candidate value
        per instantiation. ``equals`` requires the field's whole text to be the value (a
        join's bind side), ``capture_match`` requires the value to be one of ``patterns``'
        matches in the field under their modifiers (an explicit bind), and ``contains_any``
        requires the field to contain it (a reference)."""
        spec = make_atom(
            domain=AtomDomain.EVENT if surface in _EVENT_SURFACES else AtomDomain.STATIC,
            selector=surface,
            value_expr_key=f"field:{field_path}" if field_path else "field:",
            predicate=predicate,
            patterns=tuple(patterns),
            string_semantics=sem or StringSemantics(),
            parameter_names=(self._capture_var,),
        )
        spec = self.atoms.intern(spec)
        # Collect EVERY capture atom id so the executor binds the candidate value
        # into all of them. A captured value may be REFERENCED at several sites (the token
        # appears in a tool response AND a tool-call argument); each mints its own ref atom,
        # and all must receive the binding — otherwise an orphaned ref atom stays permanently
        # false and part of the rule silently never fires.
        #
        # A second, DISTINCT *bind* site is still rejected: it would mean the value is defined
        # by matches in two different fields at once, which the single-value instantiation
        # model (one candidate set from one bind field) cannot represent — fail loudly rather
        # than mis-compile. Re-minting the SAME id is idempotent and always fine.
        if role == "capture_bind":
            if self._capture_bind_id and self._capture_bind_id != spec.atom_id:
                raise UnsupportedConstruct(
                    "multiple distinct value-capture bind sites are not supported "
                    "(a rule may bind `... as v` at one field only)"
                )
            self._capture_bind_id = spec.atom_id
            self._capture_bind_ids.add(spec.atom_id)
        else:
            if not self._capture_ref_id:
                self._capture_ref_id = spec.atom_id
            self._capture_ref_ids.add(spec.atom_id)
        return T.Atom(spec.atom_id)

    @staticmethod
    def _as_literal_list(node):
        """A Python list of literals if ``node`` is a bare list or a ``Literal`` wrapping
        a list, else ``None``."""
        if isinstance(node, list):
            return node
        if isinstance(node, Literal) and isinstance(node.value, list):
            return node.value
        return None

    def _mint_atom(
        self, surface, field_path, predicate, patterns, *, sem=None, domain=None
    ) -> T.Formula:
        if not patterns:
            return T.TRUE
        dom = domain or (AtomDomain.STATIC if surface in _STATIC_SURFACES else AtomDomain.EVENT)
        spec = make_atom(
            domain=dom,
            selector=surface,
            value_expr_key=f"field:{field_path}" if field_path else "field:",
            predicate=predicate,
            patterns=tuple(patterns),
            string_semantics=sem or StringSemantics(),
            cost=Cost.PURE,
        )
        spec = self._intern_atom(spec)
        return T.Atom(spec.atom_id)

    def _on_surface_atom(self, surface: str) -> T.Formula:
        """A zero-arg EVENT atom that is true exactly when the current event belongs to
        ``surface`` (the oracle's ``on_surface`` predicate: the surface gate decides it,
        the predicate returns True once the gate passes). Used as the antecedent of an
        ``all``-comprehension's universal so it ranges only over ``S``'s elements."""
        spec = make_atom(
            domain=AtomDomain.EVENT,
            selector=surface,
            value_expr_key="field:",
            predicate="on_surface",
            patterns=(),
            cost=Cost.PURE,
        )
        spec = self._intern_atom(spec)
        return T.Atom(spec.atom_id)

    def _mint_scalar(
        self, surface, field_path, predicate, op, number, patterns=(), semantics=None
    ) -> T.Formula:
        spec = make_atom(
            domain=AtomDomain.EVENT if surface in _EVENT_SURFACES else AtomDomain.STATIC,
            selector=surface,
            value_expr_key=f"field:{field_path}" if field_path else "field:",
            predicate=predicate,
            patterns=tuple(patterns),
            op=op,
            number=number,
            string_semantics=semantics or StringSemantics(),
            cost=Cost.PURE,
        )
        spec = self._intern_atom(spec)
        return T.Atom(spec.atom_id)

    def _external_atom(self, fc: FunctionCall, surface: str) -> T.Formula:
        name = fc.name
        cost = _EXTERNAL_FUNCS[name]
        # Two arg shapes: field-first probers (``topic_drift(field)`` — arg[0] is the
        # scanned field, no literal config) vs. config-first (``model_score(model, field)``
        # — arg[0] is model/prompt/signature/tag, field is arg[1]).
        patterns: tuple = ()
        input_extra: dict = {}
        if name in _FIELD_FIRST_EXTERNAL:
            detector_arg = ""
            field_arg = fc.args[0] if fc.args else None
        else:
            detector_arg = self._lit(fc.args[0]) if fc.args else ""
            field_arg = fc.args[1] if len(fc.args) > 1 else None
        resolved = self._resolve_field_target(field_arg)
        if resolved is None:
            raise UnsupportedConstruct(
                f"{name}() text argument must be a conversation field or surface"
            )
        fsurface, field_path, ordinal = resolved
        atom_surface = fsurface if fsurface is not None else surface
        # model_scan/model_score fold literal threshold + class_index (output index) onto the
        # atom; other externals leave both at their defaults. Omit a None so make_atom keeps
        # the field default (class_index=1) and atom identity stays stable.
        threshold, class_index = self._external_scoring_args(fc)
        extra: dict = dict(input_extra)
        if threshold is not None:
            extra["threshold"] = threshold
        if class_index is not None:
            extra["class_index"] = class_index
        spec = make_atom(
            domain=AtomDomain.EVENT if atom_surface in _EVENT_SURFACES else AtomDomain.STATIC,
            selector=atom_surface,
            value_expr_key=f"field:{field_path}" if field_path else "field:",
            predicate="external",
            patterns=patterns,
            cost=cost,
            detector=name,
            detector_arg=detector_arg,
            ordinal=ordinal,
            **extra,
        )
        spec = self._intern_atom(spec)
        self.notes.append(
            f"external {name}()[{cost.value}]"
            + (f" @ordinal {ordinal}" if ordinal is not None else "")
        )
        return T.Atom(spec.atom_id)

    def _resolve_field_target(self, node) -> tuple[str | None, str, int | None] | None:
        """Resolve a field argument to ``(surface_or_None, field_path, ordinal_or_None)``.

        Handles a plain ``FieldAccess`` and an indexed one whose root is
        ``_index(surface, i)`` (e.g. ``user_messages[0].content`` →
        ``("user_messages", "content", 0)``). Returns None if not a field."""
        # A bare surface Identifier used as the scanned field of a boolean external
        # (e.g. ``model_scan("...", system_prompt)`` / ``off_topic(system_prompt)``). Without
        # this the identifier is unresolved and the atom keeps the caller's default
        # ``messages`` surface, silently turning a static prompt scan into a per-message
        # content scan. Resolve it to that static surface with an empty field path so the
        # atom is grounded once against the prompt text at the boot step.
        if isinstance(node, Identifier) and node.name in _SURFACES:
            return (node.name, "", None)
        if not isinstance(node, FieldAccess):
            return None
        base = node.base
        tail = ".".join(str(f) for f in node.fields)
        if isinstance(base, FunctionCall) and base.name == "_index" and len(base.args) == 2:
            coll, idx_node = base.args
            surface = None
            if isinstance(coll, Identifier) and coll.name in _SURFACES:
                surface = coll.name
            elif isinstance(coll, FunctionCall) and coll.name in _ACCESSORS:
                accessor_surface, filter_field = _ACCESSORS[coll.name]
                if filter_field is not None and coll.args:
                    raise UnsupportedConstruct(
                        f"indexed {coll.name}() access would discard its exact name filter"
                    )
                surface = accessor_surface
            ordinal = int(idx_node.value) if isinstance(idx_node, Literal) else None
            return (surface, tail, ordinal)
        if base is None:
            root = str(node.fields[0]) if node.fields else ""
            surface = root if root in _SURFACES else None
            return (surface, ".".join(str(f) for f in node.fields[1:]), None)
        return (None, self._field_path(node), None)


_FLIP = {">": "<", "<": ">", ">=": "<=", "<=": ">=", "=": "=", "!=": "!="}


def _unknown_root_name(node, scope: Scope) -> str | None:
    """The first name ``node`` reads that is neither a surface nor a bound element."""
    for n in walk_condition_nodes(node):
        name = n.name if isinstance(n, Identifier) else None
        if isinstance(n, FieldAccess):
            name = _field_root_name(n)
        if name is not None and name not in _SURFACES and scope.resolve_element(name) is None:
            return name
    return None


def _field_root_name(field_node) -> str | None:
    """The identifier a field access starts from (``c`` in ``c.arguments.cmd``)."""
    root = field_node
    while isinstance(root, FieldAccess) and root.base is not None:
        root = root.base
    if isinstance(root, FieldAccess) and root.base is None and root.fields:
        return str(root.fields[0])
    if isinstance(root, Identifier):
        return root.name
    return None


def _and_all(fs: list[T.Formula]) -> T.Formula:
    if not fs:
        return T.TRUE
    acc = fs[0]
    for f in fs[1:]:
        acc = T.And(acc, f)
    return acc


def _or_all(fs: list[T.Formula]) -> T.Formula:
    if not fs:
        return T.FALSE
    acc = fs[0]
    for f in fs[1:]:
        acc = T.Or(acc, f)
    return acc


# Model operations a provider can refuse; the guards accept only expressions reading one.
_REFUSABLE_OPERATIONS = frozenset({"score", "complete"})
_REFUSAL_GUARDS = frozenset({"refused", "if_refused"})


def _convert(
    expression: ModelExpression, result_type: str | None, accepted: set[str]
) -> tuple[ModelExpression, str | None]:
    """Apply the language's implicit conversion (``TYPE_CONVERSIONS``): a message of the
    conversation, an index ``X[i]`` or a bound element, where ``Messages`` is expected is the
    one-message ``Messages`` of it (``as_messages``). A model reply is not a conversation
    message, so it is not converted."""

    if (
        "Messages" in accepted
        and result_type in {"Message", "AssistantMessage"}
        and expression.kind in {"index", "binding"}
    ):
        return ModelExpression("as_messages", None, (expression,)), "Messages"
    return expression, result_type


def _fallback_type(value_type: str, fallback_type: str) -> str:
    """Return the type of ``if_refused(value, fallback)``, or reject a mismatched fallback."""

    numeric = {"int", "float"}
    if value_type in numeric and fallback_type in numeric:
        return "float" if "float" in {value_type, fallback_type} else "int"
    if value_type.removesuffix("?") == "string" and fallback_type == "string":
        return value_type
    if value_type == fallback_type == "bool":
        return "bool"
    raise UnsupportedConstruct(
        f"if_refused() fallback of type {fallback_type} does not match a {value_type} value"
    )


_CallKey = tuple[str, tuple[tuple[str, str], ...]]
# Arguments that decide a refusable call's request; the optional tools argument may name the
# conversation's own tools, which the runtime treats as the same request as omitting it.
_REQUEST_ARGUMENTS = {"score": 2, "complete": 1}


def _call_identity(call: ModelCall) -> str:
    """Identify a model call by what decides its request, so reads that may share it match."""

    arguments = call.args[: _REQUEST_ARGUMENTS.get(call.operation, len(call.args))]
    return repr((call.alias, call.operation, arguments))


def reject_unguarded_refusal_reads(condition: ConditionNode) -> None:
    """Reject a rule that reads one model call both inside and outside a refusal guard.

    Each comparison is its own check, grounded cheapest first, so an unguarded read of a
    refused call could fail the rule before the guarded read decides it. A call is the same
    call within the same enclosing comprehensions; a read of an `expr as v` alias reads the
    calls in `expr`.
    """

    aliases: dict[str, set[_CallKey]] = {}
    guarded: set[_CallKey] = set()
    unguarded: set[_CallKey] = set()

    def visit(node: ConditionNode, inside: bool, scopes: tuple[tuple[str, str], ...]) -> None:
        if isinstance(node, FunctionCall) and node.name in _REFUSAL_GUARDS:
            for index, argument in enumerate(node.args):
                visit(argument, inside or index == 0, scopes)
            return
        if isinstance(node, AnyComprehension | AllComprehension | FilterComprehension):
            visit(node.collection, inside, scopes)
            visit(node.condition, inside, (*scopes, (node.variable, repr(node.collection))))
            return
        if isinstance(node, ModelCall):
            (guarded if inside else unguarded).add((_call_identity(node), scopes))
        elif isinstance(node, Identifier) and node.name in aliases:
            (guarded if inside else unguarded).update(aliases[node.name])
        elif isinstance(node, Assignment):
            aliases[node.variable] = {
                (_call_identity(call), scopes)
                for call in walk_condition_nodes(node.expression)
                if isinstance(call, ModelCall)
            }
        for child in iter_condition_children(node):
            visit(child, inside, scopes)

    # The first pass learns every alias, so a read before its definition still counts.
    visit(condition, False, ())
    guarded.clear()
    unguarded.clear()
    visit(condition, False, ())
    if guarded & unguarded:
        raise UnsupportedConstruct(
            "a model call read inside refused() or if_refused() is also read outside one; "
            "wrap every read of it in if_refused()"
        )


def compile_rule(
    rule,
    *,
    max_work: int | None = DEFAULT_MAX_WORK,
    decompose: bool = True,
    response_schemas: Mapping[str, str] | None = None,
) -> CompileResult:
    """Compile a parsed ``DSLRule`` into a :class:`CompiledRuleArtifact` through the core
    calculus (``quarry.engine.lower``). On any unsupported construct or construction limit,
    including the ``max_work`` budget of formula-node visits (``quarry.engine.budget``;
    ``None`` lifts it), the result's ``ok`` is False with located diagnostics and
    ``artifact=None``. ``decompose`` builds the rule's decision tree (see ``compile_core``).
    ``response_schemas`` maps each model alias to its completion response schema, which types
    the rule's ``.json`` reads; without one, a rule cannot read ``.json``."""
    from quarry.engine.lower import compile_core

    compatibility_issue = check_rule_compatibility(rule)
    if compatibility_issue is not None:
        diagnostics = Diagnostics()
        rule_name = str(getattr(rule, "name", ""))
        diagnostics.error(
            compatibility_issue.message,
            ErrorCategory.COMPATIBILITY,
            SourceRef(rule_name=rule_name),
        )
        return CompileResult(rule_name, None, diagnostics)
    return compile_core(
        rule, max_work=max_work, decompose=decompose, response_schemas=response_schemas
    )


class _NodeAliases:
    """The authored nodes that each node created by an AST rewrite stands for.

    An entry holds the created node itself, so no other object can take its id while the map
    is in use, and a lookup checks identity."""

    def __init__(self) -> None:
        self._entries: dict[int, tuple[object, tuple]] = {}

    def authored(self, node) -> tuple:
        """Return the authored nodes behind the node."""
        entry = self._entries.get(id(node))
        return entry[1] if entry is not None and entry[0] is node else (node,)

    def __contains__(self, node) -> bool:
        entry = self._entries.get(id(node))
        return entry is not None and entry[0] is node

    def derive(self, new, *sources):
        """Record that ``new`` stands for the authored nodes behind ``sources``."""
        if new not in self:
            self._entries[id(new)] = (new, tuple(a for s in sources for a in self.authored(s)))
        return new


def _reject_static_bind(surface: str | None) -> None:
    """Capture candidates are read from timeline events, so a value bound on a static surface
    (a tool definition, the system prompt) would never have a candidate and the rule would
    silently never match."""
    if surface in _STATIC_SURFACES:
        raise UnsupportedConstruct(
            f"a value capture cannot be bound from `{surface}`; bind it from a message or "
            f"tool call and test the `{surface}` field against it"
        )
