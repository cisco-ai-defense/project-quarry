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

"""Transform DSL rule tree into Rule object."""

import ast
import copy
import math
from dataclasses import dataclass
from typing import Annotated, cast

from lark import Transformer, v_args

from quarry.cache.constants import MAX_CONTAINS_ANY_SIZE, MAX_STRING_ARRAY_SIZE
from quarry.language.metadata import DSLParameter, dsl_compiler_form, dsl_construct
from quarry.language.runtime import reserved_variable_names
from quarry.types.dsl import (
    MODEL_OPERATION_TYPES,
    AllComprehension,
    AnyComprehension,
    Assignment,
    BinaryOp,
    ConditionNode,
    ContainsAnyOp,
    FieldAccess,
    FilterComprehension,
    FlowOp,
    FunctionCall,
    Identifier,
    Literal,
    MapComprehension,
    ModelCall,
    ParameterRef,
    PatternMatch,
    StringCount,
    StringModifiers,
    StringPosition,
    StringSetQuantifier,
    UnaryOp,
    condition_child_fields,
    walk_condition_nodes,
)
from quarry.types.parameters import ParameterSpec, ParameterType
from quarry.types.rules import DSLRule

# Maximum size of a resolved ``variables:`` expansion — enforced both on each binding and on
# the final condition after bindings are inserted. This limit stops a short chain of repeated
# references from expanding into millions of expression nodes before compilation.
MAX_RESOLVED_CONDITION_NODES = 50_000


class _VariableBindings(dict):
    """Marker for a parsed ``variables:`` block: ordered ``{name: ConditionNode}``.

    A dict subclass so it is easy to distinguish from the ``strings:``/``metadata:`` dicts
    in ``rule()`` while carrying the bindings. Not a public type — it exists only to survive
    the transform and get inlined into the condition."""


class _ParameterDeclarations(tuple):
    """Marker for a parsed ``parameters:`` block."""


@dataclass(frozen=True)
class _CollectBlock:
    """Marker that distinguishes ``collect:`` from the Boolean condition."""

    expression: FilterComprehension


# Comprehension node types whose ``.variable`` binds a loop element inside their body.
_COMPREHENSION_TYPES = (
    AnyComprehension,
    AllComprehension,
    FilterComprehension,
    MapComprehension,
)


class _NodeBudget:
    """Remaining-node budget for one variable expansion.

    Charged as each node of the resolved subtree is produced. A replacement subtree is
    charged — and may be rejected — BEFORE it is deep-copied, so an oversized expansion is
    never allocated. The limit is read at construction so tests can lower it.
    """

    def __init__(self, subject: str):
        self._subject = subject
        self._limit = MAX_RESOLVED_CONDITION_NODES
        self._remaining = self._limit

    def charge(self, count: int) -> None:
        """Charge the budget by the given count."""
        self._remaining -= count
        if self._remaining < 0:
            raise ValueError(f"variables: {self._subject} exceeds {self._limit} expression nodes")


def _substitute_variables(node, bindings: dict, budget: _NodeBudget):
    """Return ``node`` with every bare ``Identifier(name)`` for a bound variable replaced by
    a deep copy of that variable's (already-resolved) expression. Copies so a variable used
    twice yields two independent subtrees (the compiler mutates nodes in place).

    **Budget-aware:** every node of the result is charged against ``budget``; a replacement
    is charged *before* it is deep-copied, so a reference (or a chain of them) that would
    blow the node limit is rejected without ever allocating the oversized tree.

    **Scope-aware:** a comprehension (``any``/``all``/filter/map) introduces its
    ``.variable`` as a loop binding that SHADOWS a declared variable of the same name inside
    its body. So when a declared variable collides with a comprehension's loop var (e.g. a
    user names a variable ``msg`` and writes ``any msg in user_messages where …``), the
    loop-var references in the body must NOT be substituted. The ``collection`` keeps the
    full bindings (it is evaluated in the OUTER scope); the body fields
    (``condition``/``expression``) use bindings with the shadowed name removed. A
    ``matches { … }`` pattern block is a closed object-scope: its field identifiers are
    accessors, not variable references, so it is charged whole and left unsubstituted.

    Child-field discovery is delegated to the canonical condition traversal
    (``condition_child_fields``), so this is not a second hand-written list of AST children.
    """
    if isinstance(node, Identifier) and node.name in bindings:
        replacement = bindings[node.name]
        budget.charge(_count_nodes(replacement))
        return copy.deepcopy(replacement)

    if isinstance(node, PatternMatch):
        # Opaque object-scope: never substitute inside, but still count toward the budget.
        budget.charge(_count_nodes(node))
        return node

    budget.charge(1)

    if isinstance(node, _COMPREHENSION_TYPES):
        loop_var = getattr(node, "variable", None)
        # Body scope = bindings minus the shadowed loop var (only that one name is masked).
        body_bindings = (
            {k: v for k, v in bindings.items() if k != loop_var}
            if loop_var in bindings
            else bindings
        )
        for field_name in condition_child_fields(node):
            scope = bindings if field_name == "collection" else body_bindings
            _substitute_child_field(node, field_name, scope, budget)
        return node

    for field_name in condition_child_fields(node):
        _substitute_child_field(node, field_name, bindings, budget)
    return node


def _substitute_child_field(node, field_name: str, bindings: dict, budget: _NodeBudget):
    """Substitute one executable child field in place (a single node, list, or tuple)."""
    value = getattr(node, field_name)
    if value is None:
        return
    if isinstance(value, list):
        setattr(node, field_name, [_substitute_variables(c, bindings, budget) for c in value])
    elif isinstance(value, tuple):
        setattr(node, field_name, tuple(_substitute_variables(c, bindings, budget) for c in value))
    else:
        setattr(node, field_name, _substitute_variables(value, bindings, budget))


def _count_nodes(node) -> int:
    """Count the executable nodes under ``node`` (inclusive).

    Child discovery is delegated to the canonical condition traversal, so every executable
    child — including list-valued fields such as ``PatternMatch.patterns`` — is counted.
    """

    return sum(1 for _ in walk_condition_nodes(node))


def _reject_as_binding_name_collisions(variables: dict, condition, collect) -> None:
    """Reject a capture binding that would be replaced by a rule variable during inlining."""
    for root in (*variables.values(), condition, collect):
        if root is None:
            continue
        for node in walk_condition_nodes(root):
            if isinstance(node, Assignment) and node.variable in variables:
                name = node.variable
                raise ValueError(
                    f"variables: '{name}' conflicts with `as {name}`; "
                    "rename the declaration or the `as` binding"
                )


def _resolve_variable_bindings(raw: dict) -> dict:
    """Resolve inter-variable references so each binding is fully expanded (a later variable
    may reference an earlier one). Detects cycles and rejects references to reserved names.
    Returns ``{name: fully_inlined_expr}``.

    Inlining multiplies expression trees when one binding references another more than once.
    Reject an expanded binding above the node limit before a short rule consumes excessive
    memory during later compilation.
    """
    resolved: dict = {}
    resolving: set = set()

    def resolve(name):
        if name in resolved:
            return resolved[name]
        if name in resolving:
            raise ValueError(f"variables: cyclic reference involving '{name}'")
        resolving.add(name)
        expr = copy.deepcopy(raw[name])
        refs = {n for n in _referenced_names(expr) if n in raw}
        inlined = {r: resolve(r) for r in refs}
        # Budget the expansion as it is built (even with no references, this bounds the
        # binding's own size) so an oversized tree is rejected before it is fully allocated.
        expr = _substitute_variables(expr, inlined, _NodeBudget(f"expansion of '{name}'"))
        resolving.discard(name)
        resolved[name] = expr
        return expr

    for name in raw:
        if name in reserved_variable_names():
            raise ValueError(f"variables: '{name}' shadows a reserved DSL name")
        resolve(name)
    return resolved


def _referenced_names(node) -> set:
    """Bare identifier names referenced anywhere under ``node``.

    A ``matches { … }`` pattern block is treated as an opaque object-scope (its field
    identifiers are accessors, not variable references), matching how ``_substitute_variables``
    never substitutes inside one. A comprehension's loop variable shadows a binding of the same
    name inside its body, also as in substitution. Child discovery is otherwise the canonical
    traversal, so this shares one owner of AST structure with the counter and substitution.
    """
    names: set = set()

    def walk(n, shadowed: frozenset):
        if isinstance(n, Identifier):
            if n.name not in shadowed:
                names.add(n.name)
            return
        if isinstance(n, PatternMatch):
            return
        loop_var = getattr(n, "variable", None) if isinstance(n, _COMPREHENSION_TYPES) else None
        body_shadowed = shadowed | {loop_var} if loop_var is not None else shadowed
        for field_name in condition_child_fields(n):
            scope = shadowed if field_name == "collection" else body_shadowed
            value = getattr(n, field_name)
            children = value if isinstance(value, list | tuple) else (value,)
            for child in children:
                if child is not None:
                    walk(child, scope)

    walk(node, frozenset())
    return names


class _RawString(str):
    """Decoded string value that also retains its source token contents."""

    raw_value: str

    def __new__(cls, value: str, raw_value: str):
        result = super().__new__(cls, value)
        result.raw_value = raw_value
        return result

    def __reduce__(self):
        # The default reduction calls ``__new__`` with the value alone, so ``deepcopy`` (and
        # pickle) of any report holding an array literal's values would fail.
        return (_RawString, (str(self), self.raw_value))


def _protect_unknown_escapes(token: str) -> str:
    """Make regex escapes safe for ``ast.literal_eval`` without changing them."""

    single_character_escapes = frozenset("\\\"'abfnrtv01234567\n\r")
    hex_digits = frozenset("0123456789abcdefABCDEF")
    protected: list[str] = []
    index = 0
    while index < len(token):
        character = token[index]
        if character != "\\" or index + 1 >= len(token):
            protected.append(character)
            index += 1
            continue
        next_character = token[index + 1]
        remaining = token[index + 2 :]
        known = next_character in single_character_escapes
        if next_character in {"x", "u", "U"}:
            digit_count = {"x": 2, "u": 4, "U": 8}[next_character]
            known = len(remaining) >= digit_count and all(
                character in hex_digits for character in remaining[:digit_count]
            )
        elif next_character == "N":
            closing_brace = remaining.find("}")
            known = remaining.startswith("{") and closing_brace > 1
        protected.append("\\" if known else "\\\\")
        protected.append(next_character)
        index += 2
    return "".join(protected)


def _within_distance(within_value) -> int:
    """Validate a ``within: N`` bound; N counts steps, so a bound below 1 can never match."""
    distance = int(within_value)
    if distance < 1:
        raise ValueError(f"within: {distance} can never match; the distance must be at least 1")
    return distance


def _list_literal_site(node: ConditionNode, field_name: str) -> str:
    """Name the place in ``node`` where a rule wrote an unsupported list literal."""
    if isinstance(node, FunctionCall):
        if node.name == "_index":
            return "as an index"
        return f"as an argument to {node.name}()"
    if isinstance(node, ModelCall):
        return f"as an argument to model(...).{node.operation}()"
    if isinstance(node, FlowOp):
        return "as an operand of a flow operator"
    if isinstance(node, BinaryOp):
        if node.operator == "contains" and field_name == "right":
            return (
                "as the right operand of `contains`; use `contains_any [...]` to match any of "
                "several strings"
            )
        return f"as the {field_name} operand of `{node.operator}`"
    if isinstance(node, UnaryOp):
        return f"as the operand of `{node.operator}`"
    if isinstance(node, Assignment):
        return "before `as`"
    if isinstance(node, _COMPREHENSION_TYPES):
        return f"as the {field_name} of a comprehension"
    return f"in {type(node).__name__}.{field_name}"


def _reject_list_literals(root, root_site: str) -> None:
    """Fail when an array appears where the language requires one value."""
    if isinstance(root, list):
        raise ValueError(f"a list literal is not supported {root_site}")
    stack = [root]
    while stack:
        node = stack.pop()
        for field_name in condition_child_fields(node):
            value = getattr(node, field_name)
            if isinstance(value, list) and not all(
                isinstance(child, ConditionNode) for child in value
            ):
                site = _list_literal_site(node, field_name)
                raise ValueError(f"a list literal is not supported {site}")
            children = value if isinstance(value, list | tuple) else (value,)
            stack.extend(child for child in children if isinstance(child, ConditionNode))


@v_args(inline=True)
class DSLTransformer(Transformer):
    """Transform tree into a Rule object."""

    @staticmethod
    def _decode_string_token(string):
        """Decode a DSL string token without evaluating arbitrary code."""
        try:
            value = ast.literal_eval(_protect_unknown_escapes(str(string)))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid escaped string literal: {string}") from exc
        if not isinstance(value, str):
            raise ValueError("String literal did not decode to text")
        return value

    @dsl_construct(name="rule")
    def rule(self, name, *args):
        """Define one named rule with a required condition and optional metadata, strings,
        variables, parameters, and typed collection output."""
        metadata = {}
        strings = {}
        variables: dict = {}
        parameters: tuple[ParameterSpec, ...] = ()
        condition = None
        collect = None

        for arg in args:
            if isinstance(arg, _VariableBindings):
                variables = arg
            elif isinstance(arg, _ParameterDeclarations):
                parameters = tuple(arg)
            elif isinstance(arg, _CollectBlock):
                collect = arg.expression
            elif isinstance(arg, dict):
                # A dict whose keys are all `$` variables is the strings block; any other is metadata.
                if all(k.startswith("$") for k in arg):
                    strings = arg
                else:
                    metadata = arg
            elif isinstance(arg, ConditionNode):
                condition = arg

        # Inline named variables into the condition (pure sugar — the DFA compiler never
        # sees a variable reference). Undefined references left as-is are caught downstream
        # by the validator's semantic check.
        if variables:
            _reject_as_binding_name_collisions(variables, condition, collect)
            resolved = _resolve_variable_bindings(variables)
            # Budget each substituted expression: a resolved binding is individually within
            # the limit, but inserting bindings can still push the whole expression over it.
            if condition is not None:
                condition = _substitute_variables(
                    condition, resolved, _NodeBudget("the resolved condition")
                )
            if collect is not None:
                collect = _substitute_variables(
                    collect, resolved, _NodeBudget("the resolved collect expression")
                )

        if condition is None:
            raise ValueError("rule requires a condition")
        rule_obj = DSLRule(self._decode_string_token(name), metadata, condition, collect=collect)
        rule_obj.strings = strings
        rule_obj.parameters = parameters
        return rule_obj

    @dsl_construct(name="parameters")
    def parameters_block(self, *items):
        """Declare typed policy-bound or scan-bound inputs as the last block before ``condition:``."""

        names: set[str] = set()
        declarations: list[ParameterSpec] = []
        for spec in items:
            if spec.name in names:
                raise ValueError(f"parameters: duplicate definition of '{spec.name}'")
            names.add(spec.name)
            declarations.append(spec)
        return _ParameterDeclarations(declarations)

    @dsl_construct(name="parameter-declaration")
    def parameter_def(self, name, parameter_type, policy_bound=None):
        """Declare a per-scan value, or a policy-bound value when marked ``policy``."""

        binding = "policy" if policy_bound is not None else "scan"
        return ParameterSpec(str(name), cast(ParameterType, str(parameter_type)), binding)

    @dsl_construct(name="typed-collection")
    def collect_block(self, expression):
        """Return every item that survives one exhaustive collection filter.

        The rule's Boolean condition runs first. A false condition skips collection work;
        a true condition matches only when this block returns at least one item. The filter may
        call models and remains subject to the rule timeout and model-call budget.
        """
        _reject_list_literals(expression, "in collect:")
        return _CollectBlock(expression)

    @dsl_construct(name="variables")
    def variables_block(self, *items):
        """Declare reusable expressions that are expanded before validation and compilation.

        A binding may use another binding. Inside a comprehension, the loop variable hides a
        binding with the same name, so ``any y in user_messages where y.content ...`` reads
        the element even when a binding named ``y`` exists."""
        bindings = _VariableBindings()
        for name, expr in items:
            if name in bindings:
                raise ValueError(f"variables: duplicate definition of '{name}'")
            bindings[name] = expr
        return bindings

    def variable_def(self, name, expr):
        """A single ``name = expr`` binding."""
        _reject_list_literals(expr, f"as the value of variable '{name}'")
        return (str(name), expr)

    def condition(self, expr):
        """ConditionNode."""
        _reject_list_literals(expr, "as the rule condition")
        return expr

    def atom(self, base, *postfixes):
        """Apply postfix operations (attribute access/indexing)."""
        result = base
        for postfix in postfixes:
            result = postfix(result)
        return result

    def primary(self, item):
        """Base atom without postfix."""
        return item

    @dsl_construct(name="field-access")
    def attr_postfix(self, attr):
        """Read a named field with dot syntax, such as ``msg.content``."""
        attr = str(attr)

        def apply_attr(base):
            if isinstance(base, Identifier) and base.name == "params":
                return ParameterRef(attr)
            if isinstance(base, ParameterRef):
                raise ValueError("parameter references cannot use chained field access")
            if isinstance(base, FieldAccess):
                return FieldAccess(base.base, base.fields + [attr])
            return FieldAccess(base, [attr])

        return apply_attr

    @dsl_construct(name="wildcard-field-access")
    def wildcard_postfix(self, _wildcard):
        """Select wildcard child values with a final ``.*`` field step."""

        def apply_wildcard(base):
            if isinstance(base, FieldAccess):
                if base.fields and base.fields[-1] == "*":
                    raise ValueError("Wildcard (*) can only be used as the last accessor")
                return FieldAccess(base.base, base.fields + ["*"])
            return FieldAccess(base, ["*"])

        return apply_wildcard

    def index_postfix(self, index_expr):
        """Select one collection item with an index expression."""

        def apply_index(base):
            return FunctionCall("_index", [base, index_expr])

        return apply_index

    @dsl_construct(name="signed-index")
    def signed_index_postfix(self, idx):
        """Select one collection item by position, such as ``messages[-1]``.

        Negative positions count backward from the end of the collection.

        Internal:

        Produces the same ``_index(base, Literal(n))`` node as a literal ``index_postfix``,
        so a positive index compiles byte-identically; a negative index is resolved from the
        end of the surface at grounding time."""

        ordinal = int(str(idx))

        def apply_index(base):
            return FunctionCall("_index", [base, Literal(ordinal)])

        return apply_index

    def _build_slice(self, start: int | None, stop: int | None):
        """Windowed-surface postfix ``_slice(base, start, stop)`` with Python slice
        semantics; negative bounds are resolved from the end of the surface at grounding."""

        def apply_slice(base):
            return FunctionCall("_slice", [base, Literal(start), Literal(stop)])

        return apply_slice

    @dsl_construct(name="slice-all")
    def slice_all_postfix(self):
        """``[:]`` — the whole surface as a window (identity slice)."""
        return self._build_slice(None, None)

    @dsl_construct(name="slice-from")
    def slice_from_postfix(self, start):
        """``[start:]`` — from ``start`` to the end."""
        return self._build_slice(int(str(start)), None)

    @dsl_construct(name="slice-to")
    def slice_to_postfix(self, stop):
        """``[:stop]`` — from the beginning to ``stop``."""
        return self._build_slice(None, int(str(stop)))

    @dsl_construct(name="slice-range")
    def slice_range_postfix(self, start, stop):
        """``[start:stop]`` — a bounded window."""
        return self._build_slice(int(str(start)), int(str(stop)))

    # Receivers on which method-call syntax (``messages.called_tool("x")``) is supported.
    # The canonical message model exposes conversation predicates as methods on the message
    # collection; the compiler treats them as the equivalent conversation-wide FunctionCall, so
    # ``messages.called_tool("x")`` and bare ``called_tool("x")`` share one DFA lowering.
    _METHOD_RECEIVERS = frozenset({"messages", "conversation"})

    @staticmethod
    @dsl_compiler_form(
        name="model",
        return_type="model_reference",
        syntax='model("alias").operation(...)',
    )
    def _bind_model_reference(
        alias: Annotated[
            object,
            DSLParameter(accepted_types=("string",), literal_only=True),
        ],
    ) -> str:
        """Select a configured model by its literal alias for a following method call.

        Use this as the start of a model operation, such as
        `model("guard").score(messages, "policy")`. `model(...)` is not a condition by itself.
        """

        if not isinstance(alias, Literal) or not isinstance(alias.value, str) or not alias.value:
            raise ValueError("model() requires one non-empty literal alias")
        return alias.value

    @dsl_construct(name="method-call")
    def method_postfix(self, name, args=None):
        """Call a supported Quarry function on the value written before the dot.

        For example, ``messages.called_tool("read")`` checks the complete message collection.
        A ``message(...)`` model input can use ``.add_message(...)`` to append another message.
        Other values cannot use method-call syntax.

        Internal:

        Normalized to the equivalent conversation-wide ``FunctionCall`` so the compiler
        reuses its existing lowering with no engine change. Only the
        ``messages``/``conversation`` receivers are supported; anything else fails closed
        at load rather than compiling to something unintended."""
        method = str(name)
        call_args = list(args) if args else []

        def apply_method(base):
            if isinstance(base, FunctionCall) and base.name == "model":
                try:
                    alias = self._bind_model_reference(*base.args)
                except TypeError as exc:
                    raise ValueError("model() requires one non-empty literal alias") from exc
                if method not in MODEL_OPERATION_TYPES:
                    raise ValueError(f"model references do not support '.{method}()'")
                return ModelCall(
                    alias=alias,
                    operation=method,
                    args=call_args,
                    result_type=MODEL_OPERATION_TYPES[method],
                )
            if isinstance(base, Identifier) and base.name in self._METHOD_RECEIVERS:
                if method == "extract":
                    return FunctionCall(method, [base, *call_args])
                return FunctionCall(method, call_args)
            # Constructed model input: message(role, content).add_message(role, content)...
            # builds a synthetic conversation for llm_score. The receiver is the message()
            # builder call (or a further add_message chain); normalize to a nested FunctionCall
            # the compiler folds into a constructed-input atom (never a standalone atom).
            if method == "add_message" and self._is_message_builder(base):
                return FunctionCall("add_message", [base, *call_args])
            raise ValueError(
                f"method call '.{method}(...)' is only supported on "
                f"{sorted(self._METHOD_RECEIVERS)} receivers, or as an add_message chain on message()"
            )

        return apply_method

    @staticmethod
    def _is_message_builder(node) -> bool:
        """True for a ``message(...)`` builder call or a chained ``add_message(...)`` on one."""

        return isinstance(node, FunctionCall) and node.name in ("message", "add_message")

    def attribute(self, token):
        """Attribute token to string."""
        return str(token)

    def comparison(self, item):
        """Comparison."""
        return item

    def in_expr(self, item):
        """In expression pass-through."""
        return item

    def arith_expr(self, item):
        """Arithmetic expression pass-through."""
        return item

    def comprehension(self, item):
        """Comprehension."""
        return item

    def value(self, item):
        """Value."""
        return item

    @dsl_construct(name="boolean-or")
    def or_expr(self, left, *rights):
        """Match when at least one operand is true."""
        for op in rights:
            left = BinaryOp("or", left, op)
        return left

    @dsl_construct(name="boolean-and")
    def and_expr(self, left, *rights):
        """Match when every operand is true."""
        for op in rights:
            left = BinaryOp("and", left, op)
        return left

    def description(self, desc):
        """Description."""
        return ("description", self._decode_string_token(desc))

    def severity(self, level):
        """Severity."""
        return ("severity", str(level))

    @dsl_construct(name="boolean-not")
    def not_op(self, expr):
        """Invert one boolean expression."""
        return UnaryOp("not", expr)

    @dsl_construct(name="contains")
    def contains_op(self, left, right):
        """Test whether text or a collection contains a supplied value.

        Text comparison is case-sensitive. To match text regardless of letter case, declare the
        value in a ``strings:`` block with the ``nocase`` modifier and reference it here.
        """
        return BinaryOp("contains", left, right)

    @dsl_construct(name="equal-comparison")
    def eq_op(self, left, right):
        """Match when the two values are equal."""
        return BinaryOp("=", left, right)

    @dsl_construct(name="not-equal-comparison")
    def neq_op(self, left, right):
        """Match when the two values are not equal."""
        return BinaryOp("!=", left, right)

    @dsl_construct(name="greater-than-comparison")
    def gt_op(self, left, right):
        """Match when the left number is greater than the right number."""
        return BinaryOp(">", left, right)

    @dsl_construct(name="less-than-comparison")
    def lt_op(self, left, right):
        """Match when the left number is less than the right number."""
        return BinaryOp("<", left, right)

    @dsl_construct(name="greater-than-or-equal-comparison")
    def gte_op(self, left, right):
        """Match when the left number is greater than or equal to the right number."""
        return BinaryOp(">=", left, right)

    @dsl_construct(name="less-than-or-equal-comparison")
    def lte_op(self, left, right):
        """Match when the left number is less than or equal to the right number."""
        return BinaryOp("<=", left, right)

    @dsl_construct(name="addition")
    def add_op(self, left, right):
        """Add two numeric values."""
        return BinaryOp("+", left, right)

    @dsl_construct(name="subtraction")
    def sub_op(self, left, right):
        """Subtract the right numeric value from the left value."""
        return BinaryOp("-", left, right)

    @dsl_construct(name="multiplication")
    def mul_op(self, left, right):
        """Multiply two numeric values."""
        return BinaryOp("*", left, right)

    @dsl_construct(name="membership")
    def in_op(self, left, right):
        """Test whether a value or text pattern occurs in another value."""
        if isinstance(right, list):
            right = Literal(right)
        return BinaryOp("in", left, right)

    @dsl_construct(name="capture")
    def in_op_with_capture(self, left, right, var):
        """Capture matched text under a name for comparison elsewhere in the rule.

        The rule matches when some captured value satisfies every place that uses it, or when
        it holds with no value captured, every place that uses the value then being false. The
        capture itself cannot appear under ``not`` or inside an ``all`` comprehension; such a
        rule fails to load. A place that uses the value may appear there."""
        if str(var) == "params":
            raise ValueError("'params' is reserved for declared rule parameters")
        if isinstance(right, list):
            right = Literal(right)
        return Assignment(BinaryOp("in", left, right), str(var))

    @dsl_construct(name="pattern-match")
    def matches_op(self, expr, pattern):
        """Test one value against a block containing alternative field predicates."""
        return BinaryOp("matches", expr, pattern)

    @dsl_construct(name="function-call")
    def function_call(self, name, args=None):
        """Call a public Quarry function with positional arguments."""
        return FunctionCall(str(name), args or [])

    def arguments(self, *args):
        """Arguments node."""
        return list(args)

    def field_access(self, *fields):
        """Legacy compatibility helper (unused with postfix grammar)."""
        field_list = [str(f) for f in fields]
        return FieldAccess(None, field_list)

    def identifier(self, name):
        """Identifier node."""
        return Identifier(str(name))

    def string_literal(self, string):
        """String literal node."""
        token = str(string)
        return Literal(self._decode_string_token(token), raw_value=token[1:-1])

    def number_literal(self, num):
        """Number literal node."""
        num_str = str(num)
        value = int(num_str) if num_str.isdigit() else float(num_str)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Numeric literals must be finite")
        return Literal(value)

    def true_literal(self):
        """True literal node."""
        return Literal(True)

    def false_literal(self):
        """False literal node."""
        return Literal(False)

    def null_literal(self):
        """Null literal node."""
        return Literal(None)

    @dsl_construct(name="adjacent-alias")
    def assignment(self, expr, var):
        """Name a value for use in the clause immediately joined by the same ``and``."""
        if str(var) == "params":
            raise ValueError("'params' is reserved for declared rule parameters")
        return Assignment(expr, str(var))

    @dsl_construct(name="pattern-block")
    def pattern_block(self, *patterns):
        """Group alternative field checks between braces."""
        return PatternMatch(list(patterns))

    @dsl_construct(name="pattern-predicate")
    def pattern_expr(self, field, op, value):
        """Check one field inside a pattern block."""
        if not isinstance(field, ConditionNode):
            field = Identifier(str(field))
        return BinaryOp("=" if str(op) == "==" else str(op), field, value)

    @dsl_construct(name="any-comprehension")
    def any_comp(self, var, collection, condition):
        """Match when at least one item in a collection satisfies a condition.

        A part of the condition that does not read the item, such as ``called_tool("x")`` or
        a comprehension over another collection, is checked over the whole conversation, not
        only after the item. Over ``tools``, every test in the condition reads one tool
        definition. An empty collection never matches.

        One item can take part in several flows, and a flow operand can be another
        comprehension joined to the item (``any c in tool_calls where c.arguments.cmd
        contains m.content``). Three shapes are checked item by item, for at most 256 items of
        the collection per scan, where a scan with more items and no match is incomplete: a
        flow operand joined to the item, the item's own condition next to other alternatives
        on the left of ``until``, and an event before the item whose condition relates later
        events."""
        if str(var) == "params":
            raise ValueError("'params' is reserved for declared rule parameters")
        return AnyComprehension(str(var), collection, condition)

    @dsl_construct(name="all-comprehension")
    def all_comp(self, var, collection, condition):
        """Match when every item in a collection satisfies a condition.

        A part of the condition that does not read the item is checked over the whole
        conversation. An empty collection always matches."""
        if str(var) == "params":
            raise ValueError("'params' is reserved for declared rule parameters")
        return AllComprehension(str(var), collection, condition)

    @dsl_construct(name="array-literal")
    def array(self, *items):
        """Write an ordered literal list between brackets."""
        values = [self._literal_value(item) for item in items]
        return values

    def _literal_value(self, item):
        """Extract a literal value while preserving nested array structure."""
        if isinstance(item, Literal):
            if item.raw_value is not None and isinstance(item.value, str):
                return _RawString(item.value, item.raw_value)
            return item.value
        if isinstance(item, list):
            return [self._literal_value(child) for child in item]
        if item is None or isinstance(item, str | int | float | bool):
            return item
        raise ValueError(f"Array item is not a literal: {item!r}")

    def _array_size(self, value) -> int:
        """Count scalar and nested array elements for a bounded literal."""
        if not isinstance(value, list):
            return 1
        return sum(self._array_size(item) for item in value)

    @dsl_construct(name="filter-comprehension")
    def filter_comp(self, output_var, var, collection, condition):
        """Keep collection items that satisfy a condition.

        In ``[msg for msg in user_messages if msg.content contains "sudo"]``, the result
        contains only user messages whose content contains ``sudo``. Use the same item name
        before and after ``for``.

        Internal:

        Grammar: ``"[" IDENTIFIER "for" IDENTIFIER "in" expr "if" expr "]"`` — Lark passes
        BOTH identifiers. The first is the output element, the second is the loop variable;
        the filter yields matching loop items, so the loop var (``var``, the second token)
        is what binds inside the condition. Taking only three arguments would shift them:
        the output var would arrive as ``var`` and the collection as ``condition``.
        """
        if "params" in {str(output_var), str(var)}:
            raise ValueError("'params' is reserved for declared rule parameters")
        return FilterComprehension(str(var), collection, condition, output_variable=str(output_var))

    @dsl_construct(name="map-comprehension")
    def map_comp(self, expression, var, collection, condition=None):
        """Project each selected item for use as the first argument of ``join()``.

        The map is supported in ``message()`` content of a version 2 model input.
        Its value and optional filter cannot call a model.
        """
        if str(var) == "params":
            raise ValueError("'params' is reserved for declared rule parameters")
        return MapComprehension(expression, str(var), collection, condition)

    @dsl_construct(name="regex-match")
    def regex_op(self, text, pattern):
        """Match text with an RE2-compatible regular expression.

        Matching is case-sensitive. For case-insensitive matching, declare the pattern in a
        ``strings:`` block with ``regex nocase`` and match it with ``in``."""
        if isinstance(pattern, Literal) and pattern.raw_value is not None:
            pattern = Literal(pattern.raw_value)
        return BinaryOp("~", text, pattern)

    @dsl_construct(name="flow-after")
    def after_op(self, left, right):
        """Match when the left event occurs sometime after the right event."""
        return FlowOp("after", left, right, max_distance=None)

    @dsl_construct(name="flow-until")
    def until_op(self, left, right):
        """Match when the left condition remains true until the right event occurs.

        The left condition must hold at every event of its own kind before the right event:
        every tool call for ``called_tool(...)``, every user message for ``user_said(...)``, or
        every element of a comprehension's collection. Events of other kinds do not break it.
        If the right event is the first one, the left condition need not hold at all.

        Internal:

        This is the finite-trace LTLf ``U`` operator. The left operand is the sustained
        condition and the right operand is the release event."""
        return FlowOp("until", left, right, max_distance=None)

    @dsl_construct(name="flow-right-after")
    def right_after_op(self, left, right):
        """Match when the left event occurs immediately after the right event."""
        return FlowOp("after", left, right, max_distance=1)

    @dsl_construct(name="flow-after-within")
    def after_within_op(self, left, within_value, right):
        """Match when the left event occurs after the right event within a bounded distance.

        ``N`` counts timeline steps and must be at least 1. The window compiles to about
        ``N`` automaton states; a single window beyond about 14,000 steps exceeds the compiler's
        work budget and fails to load."""
        return FlowOp("after", left, right, max_distance=_within_distance(within_value))

    @dsl_construct(name="flow-before")
    def before_op(self, left, right):
        """Match when the left event occurs sometime before the right event."""
        return FlowOp("after", right, left, max_distance=None)

    @dsl_construct(name="flow-right-before")
    def right_before_op(self, left, right):
        """Match when the left event occurs immediately before the right event."""
        return FlowOp("after", right, left, max_distance=1)

    @dsl_construct(name="flow-before-within")
    def before_within_op(self, left, within_value, right):
        """Match when the left event occurs before the right event within a bounded distance.

        ``N`` counts timeline steps and must be at least 1. The window compiles to about
        ``N`` automaton states; a single window beyond about 14,000 steps exceeds the compiler's
        work budget and fails to load."""
        return FlowOp("after", right, left, max_distance=_within_distance(within_value))

    @dsl_construct(name="contains-any")
    def contains_any_op(self, left, right):
        """Match when text contains at least one value from a literal or named array.

        Comparison against a literal array is case-sensitive and has no case option. For
        case-insensitive matching, declare a ``nocase`` pattern in a ``strings:`` block --
        use a ``regex`` alternation such as ``"alpha|beta" nocase regex`` for several
        alternatives -- and match it with ``contains``.
        """
        if hasattr(right, "type") and right.type == "STRING_VAR":
            right = Identifier(str(right))

        if isinstance(right, Identifier) and str(right.name).startswith("$"):
            return BinaryOp("contains", left, right)

        if not isinstance(right, list):
            return Literal(False)

        array_values: list = right
        if not array_values:
            return Literal(False)

        array_size = self._array_size(array_values)
        if array_size > MAX_CONTAINS_ANY_SIZE:
            raise ValueError(
                f"contains_any list too large ({array_size} items). "
                f"Maximum allowed is {MAX_CONTAINS_ANY_SIZE}. "
                "Use a string variable instead."
            )

        return ContainsAnyOp(left, array_values)

    @dsl_construct(name="metadata")
    def metadata_block(self, *items):
        """Attach string metadata fields to a rule."""
        metadata = {}
        for key, value in items:
            metadata[key] = value
        return metadata

    def metadata_item(self, key, value):
        """Parse single metadata item."""
        return (str(key), self._decode_string_token(value))

    @dsl_construct(name="strings")
    def strings_block(self, *items):
        """Declare reusable literal or regular-expression patterns and their modifiers."""
        strings = {}
        for key, value in items:
            strings[key] = value
        return strings

    def string_def(self, var_name, value, modifiers=None):
        """Parse string definition with optional modifiers."""
        if isinstance(value, Literal):
            if modifiers is not None and modifiers.regex and value.raw_value is not None:
                final_value = value.raw_value
            else:
                final_value = value.value
        elif isinstance(value, list):
            final_value = (
                self._raw_regex_values(value)
                if modifiers is not None and modifiers.regex
                else value
            )
        else:
            token = str(value)
            if modifiers is not None and modifiers.regex:
                final_value = token[1:-1]
            else:
                final_value = self._decode_string_token(token)

        if isinstance(final_value, list) and self._array_size(final_value) > MAX_STRING_ARRAY_SIZE:
            raise ValueError(
                f"String array too large: {self._array_size(final_value)} items. "
                f"Maximum allowed is {MAX_STRING_ARRAY_SIZE}."
            )

        if modifiers is not None:
            return (str(var_name), (final_value, modifiers))
        return (str(var_name), final_value)

    def _raw_regex_values(self, value):
        """Restore raw token contents for regex strings nested in an array."""
        if isinstance(value, list):
            return [self._raw_regex_values(item) for item in value]
        if isinstance(value, _RawString):
            return value.raw_value
        return value

    @dsl_construct(name="string-variable-reference")
    def string_var_ref(self, var_name):
        """Refer to a pattern declared in the ``strings:`` block by its ``$name``."""
        return Identifier(str(var_name))

    @dsl_construct(name="string-modifiers")
    def string_modifiers(self, *modifiers):
        """Change a declared pattern with ``nocase``, ``fullword``, ``base64``, or ``regex``.

        ``nocase`` matches the pattern regardless of letter case. Text matching is otherwise
        case-sensitive, so ``nocase`` is the mechanism for case-insensitive `contains` and
        `contains_any` checks. Modifiers apply to a single pattern or an array of patterns,
        and to every use of the pattern, including ``#name`` counts and ``@name`` positions.
        """
        mods = StringModifiers()
        for mod in modifiers:
            mod_str = str(mod)
            if mod_str == "nocase":
                mods.nocase = True
            elif mod_str == "fullword":
                mods.fullword = True
            elif mod_str == "base64":
                mods.base64 = True
            elif mod_str == "regex":
                mods.regex = True
        return mods

    @dsl_construct(name="string-position")
    def string_position(self, identifier, wildcard=None):
        """Return the position of a declared string pattern in the inspected text.

        Write ``@name`` for a pattern declared as ``$name``. Add ``*`` to include every
        declared pattern whose name starts with the supplied prefix. The pattern's modifiers
        apply. Positions of patterns with different modifiers cannot be compared; such a rule
        fails to load."""
        var_name = str(identifier)
        if wildcard is not None:
            var_name += "*"
        return StringPosition(var_name)

    @dsl_construct(name="string-count")
    def string_count(self, identifier, wildcard=None):
        """Count occurrences of a declared string pattern in the inspected text.

        Write ``#name`` for a pattern declared as ``$name``. Add ``*`` to include every
        declared pattern whose name starts with the supplied prefix. The pattern's modifiers
        apply, so a ``nocase`` pattern counts matches in any letter case."""
        var_name = str(identifier)
        if wildcard is not None:
            var_name += "*"
        return StringCount(var_name)

    def quantifier(self, count):
        """Parse a numeric quantifier (``N of (...)``)."""
        return int(str(count))

    def quantifier_any(self):
        """Parse the ``any of (...)`` quantifier."""
        return "any"

    def quantifier_all(self):
        """Parse the ``all of (...)`` quantifier."""
        return "all"

    @dsl_construct(name="string-set")
    def string_set_expr(self, quantifier, string_list):
        """Match when any, all, or a requested number of declared patterns occurs in text.

        ``any of`` needs one listed pattern, ``all of`` needs every listed pattern, and
        ``N of`` needs at least ``N``. Each pattern keeps its own modifiers."""
        return StringSetQuantifier(quantifier, string_list)

    def string_list(self, *wildcards):
        """Parse list of string wildcards."""
        return list(wildcards)

    def string_wildcard(self, var_name, wildcard=None):
        """Parse string wildcard pattern like $exfil* or just $exfil."""
        var_str = str(var_name)
        if wildcard is not None:
            return var_str + "*"
        return var_str
