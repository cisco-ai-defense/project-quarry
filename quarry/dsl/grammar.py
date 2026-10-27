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

"""Grammar for the DSL and read-only helpers used by language introspection."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Literal

DSL_GRAMMAR = r"""
    ?start: rule
    rule: "rule" STRING metadata_block? strings_block? variables_block? parameters_block? "condition:" condition collect_block?
    collect_block: "collect:" filter_comp
    metadata_block: "metadata:" metadata_item+
    metadata_item: METADATA_KEY "=" STRING
    strings_block: "strings:" string_def+
    string_def: STRING_VAR "=" (STRING | array) string_modifiers?
    string_modifiers: MODIFIER+
    MODIFIER.2: "nocase" | "fullword" | "base64" | "regex"
    // Named boolean sub-expressions, referenced by name in `condition:`. Pure authoring
    // sugar: the transformer inlines each reference before compilation (the DFA engine
    // never sees a variable). Lets multiple thresholds over one model output read a single
    // inference (shared via atom interning).
    variables_block: "variables:" variable_def+
    variable_def: VAR_NAME "=" expr
    parameters_block: "parameters:" parameter_def+
    parameter_def: PARAM_NAME ":" PARAM_TYPE POLICY_BOUND?
    condition: expr
    ?expr: or_expr
    or_expr: and_expr ("or" and_expr)*
    and_expr: not_expr ("and" not_expr)*
    ?not_expr: "not" not_expr -> not_op | flow_expr

    ?flow_expr: comparison "before" "(" "within" ":" NUMBER ")" comparison -> before_within_op
            | comparison "right_before" comparison -> right_before_op
            | comparison "before" comparison -> before_op
            | comparison "after" "(" "within" ":" NUMBER ")" comparison -> after_within_op
            | comparison "right_after" comparison -> right_after_op
            | comparison "after" comparison -> after_op
            | comparison "until" comparison -> until_op
            | comparison

    function_call: IDENTIFIER "(" [arguments] ")"
    arguments: expr ("," expr)*
    postfix: "." IDENTIFIER "(" [arguments] ")" -> method_postfix
           | "." IDENTIFIER -> attr_postfix
           | "[" ":" "]" -> slice_all_postfix
           | "[" SIGNED_INT ":" "]" -> slice_from_postfix
           | "[" ":" SIGNED_INT "]" -> slice_to_postfix
           | "[" SIGNED_INT ":" SIGNED_INT "]" -> slice_range_postfix
           | "[" SIGNED_INT "]" -> signed_index_postfix
           | "[" expr "]" -> index_postfix
    wildcard_postfix: "." WILDCARD -> wildcard_postfix

    string_position: "@" IDENTIFIER WILDCARD?
    string_count: "#" IDENTIFIER WILDCARD?
    string_set_expr: quantifier "of" "(" string_list ")"
    quantifier: NUMBER
              | "any" -> quantifier_any
              | "all" -> quantifier_all
    string_list: string_wildcard ("," string_wildcard)*
    string_wildcard: STRING_VAR WILDCARD?

    primary: function_call | string_position | string_count | string_set_expr | STRING_VAR -> string_var_ref | IDENTIFIER -> identifier | "(" expr ")" | value | comprehension
    atom: primary postfix* wildcard_postfix?
    value: STRING -> string_literal | NUMBER -> number_literal | array | "true" -> true_literal | "false" -> false_literal | "null" -> null_literal
    array: "[" [value ("," value)*] "]"

    // Comparison operators
    comparison: in_expr "==" in_expr -> eq_op
               | in_expr "!=" in_expr -> neq_op
               | in_expr ">" in_expr -> gt_op
               | in_expr "<" in_expr -> lt_op
               | in_expr ">=" in_expr -> gte_op
               | in_expr "<=" in_expr -> lte_op
               | in_expr

    in_expr: arith_expr "contains" arith_expr -> contains_op
           | arith_expr "contains_any" (array | STRING_VAR) -> contains_any_op
           | arith_expr "~" arith_expr -> regex_op
           | arith_expr "matches" pattern_block -> matches_op
           | arith_expr "in" arith_expr "as" IDENTIFIER -> in_op_with_capture
           | arith_expr "in" arith_expr -> in_op
           | arith_expr "as" IDENTIFIER -> assignment
           | arith_expr

    ?arith_expr: arith_expr "+" product_expr -> add_op
               | arith_expr "-" product_expr -> sub_op
               | product_expr
    ?product_expr: product_expr "*" atom -> mul_op
                 | atom

    comprehension: any_comp | all_comp | filter_comp | map_comp
    any_comp: "any" IDENTIFIER "in" expr "where" expr
    all_comp: "all" IDENTIFIER "in" expr "where" expr
    filter_comp: "[" IDENTIFIER "for" IDENTIFIER "in" expr "if" expr "]"
    map_comp: "[" expr "for" IDENTIFIER "in" expr "]" | "[" expr "for" IDENTIFIER "in" expr "if" expr "]"
    pattern_block: "{" pattern_expr ("or" pattern_expr)* "}"
    pattern_expr: atom PATTERN_OPERATOR value | IDENTIFIER PATTERN_OPERATOR value

    RIGHT_AFTER: "right_after"
    WILDCARD: "*"
    // Signed integer used as a slice bound or a signed index. Higher priority than NUMBER,
    // and lookahead-anchored to a following ':' or ']' so it only fires inside subscripts.
    SIGNED_INT.2: /-?[0-9]+(?=\s*(?::|\]))/
    STRING_VAR: /\$[a-zA-Z_][a-zA-Z0-9_]*/
    PATTERN_OPERATOR: "contains" | "==" | "!=" | ">" | "<" | ">=" | "<="
    METADATA_KEY: /(?!condition\b|collect\b|strings\b|variables\b|parameters\b)[a-zA-Z_][a-zA-Z0-9_]*/
    // Variable-binding LHS name. Exclude block headers so each optional block terminates
    // cleanly. Match the colon to keep the same words available as binding names.
    VAR_NAME: /(?!(?:condition|collect|strings|variables|metadata|parameters)[ \t]*:)[a-zA-Z_][a-zA-Z0-9_]*/
    PARAM_NAME: /(?!(?:condition|params|policy)\b)[a-zA-Z_][a-zA-Z0-9_]*/
    PARAM_TYPE: "number" | "text"
    POLICY_BOUND: "policy"
    IDENTIFIER: /(?!right_after\b)[a-zA-Z_][a-zA-Z0-9_]*/
    %import common.ESCAPED_STRING -> STRING
    %import common.NUMBER
    %import common.WS
    %import common.NEWLINE
    LINE_COMMENT: /\/\/[^\r\n]*/
    %ignore WS
    %ignore NEWLINE
    %ignore LINE_COMMENT
"""

# These grammar words cannot safely be used as names in the ``variables:`` block. Conversation
# surfaces and accessor names are added from the generated language catalog by the transformer.
# Keeping this subset here makes the grammar semantics explicit without newly reserving every
# word that happens to occur in a contextual grammar position.
VARIABLE_RESERVED_KEYWORDS = frozenset(
    {
        "any",
        "all",
        "not",
        "and",
        "or",
        "in",
        "as",
        "where",
        "true",
        "false",
        "null",
        "collect",
        "params",
    }
)


@dataclass(frozen=True)
class OperatorPrecedence:
    """One operator family, ordered from the loosest to the tightest binding."""

    precedence: int
    family: str
    operators: tuple[str, ...]
    associativity: Literal["left", "right", "none"]
    description: str
    grammar_rule: str
    operand_rule: str


OPERATOR_PRECEDENCE = (
    OperatorPrecedence(
        1, "boolean or", ("or",), "left", "Evaluated after `and`.", "or_expr", "and_expr"
    ),
    OperatorPrecedence(
        2, "boolean and", ("and",), "left", "Evaluated before `or`.", "and_expr", "not_expr"
    ),
    OperatorPrecedence(
        3,
        "boolean not",
        ("not",),
        "right",
        "Applies to the expression immediately after it.",
        "not_expr",
        "flow_expr",
    ),
    OperatorPrecedence(
        4,
        "flow",
        (
            "before",
            "before(within:N)",
            "right_before",
            "after",
            "after(within:N)",
            "right_after",
            "until",
        ),
        "none",
        "Relates two event conditions. Each operand is one event: a single condition, an "
        "`any` comprehension, `and`/`or`/`not` over the element of an enclosing "
        "comprehension, or an `or` of such operands. `all` is allowed only on the sustained "
        "side of `until`. Chain flow operators only with parentheses.",
        "flow_expr",
        "comparison",
    ),
    OperatorPrecedence(
        5,
        "comparison",
        ("==", "!=", ">", "<", ">=", "<="),
        "none",
        "Compares two values. Chain comparisons only with parentheses.",
        "comparison",
        "in_expr",
    ),
    OperatorPrecedence(
        6,
        "text and membership",
        ("contains", "contains_any", "~", "matches", "in", "as"),
        "none",
        "Tests text or membership and binds captures.",
        "in_expr",
        "arith_expr",
    ),
    OperatorPrecedence(
        7,
        "addition",
        ("+", "-"),
        "left",
        "Adds or subtracts numeric values.",
        "arith_expr",
        "product_expr",
    ),
    OperatorPrecedence(
        8, "multiplication", ("*",), "left", "Multiplies numeric values.", "product_expr", "atom"
    ),
    OperatorPrecedence(
        9,
        "postfix access",
        (".field", ".method(...)", "[index]", "[slice]", ".*"),
        "left",
        "Reads fields, calls methods, indexes, slices, or expands object values.",
        "atom",
        "primary",
    ),
)

_QUOTED_LITERAL = re.compile(r'"(?:\\.|[^"\\])*"')
_RULE_START = re.compile(r"^\s*\??([a-z][a-z0-9_]*)\s*:\s*(.*)$")
_RULE_CONTINUATION = re.compile(r"^\s*\|\s*(.*)$")


def grammar_literals() -> frozenset[str]:
    """Return every literal spelling declared by the grammar."""

    source = "\n".join(
        line for line in DSL_GRAMMAR.splitlines() if not line.lstrip().startswith("//")
    )
    return frozenset(ast.literal_eval(match.group(0)) for match in _QUOTED_LITERAL.finditer(source))


def grammar_productions() -> dict[str, str]:
    """Return normalized lowercase grammar productions keyed by rule name."""

    productions: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in DSL_GRAMMAR.splitlines():
        if raw_line.lstrip().startswith("//"):
            continue
        line = raw_line.rstrip()
        if not line.strip():
            continue
        start = _RULE_START.match(line)
        if start:
            current = start.group(1)
            productions[current] = [start.group(2).strip()]
            continue
        continuation = _RULE_CONTINUATION.match(line)
        if continuation and current is not None:
            productions[current].append("| " + continuation.group(1).strip())
            continue
        current = None
    return {name: " ".join(parts).strip() for name, parts in productions.items()}


def operator_precedence() -> tuple[OperatorPrecedence, ...]:
    """Return the public operator order after checking it against the grammar."""

    productions = grammar_productions()
    for expected, record in enumerate(OPERATOR_PRECEDENCE, start=1):
        if record.precedence != expected:
            raise ValueError("operator precedence levels must be consecutive from 1")
        production = productions.get(record.grammar_rule)
        if production is None or record.operand_rule not in production:
            raise ValueError(
                f"operator precedence family {record.family!r} does not match "
                f"grammar rule {record.grammar_rule!r}"
            )
    return OPERATOR_PRECEDENCE


def _production_branches(production: str) -> tuple[str, ...]:
    """Split alternatives without breaking a grouped ``(left | right)`` expression."""

    branches: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(production):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "|" and depth == 0:
            branches.append(production[start:index].strip())
            start = index + 1
    branches.append(production[start:].strip())
    return tuple(branches)


def _without_handler_alias(production: str, handler_name: str) -> str:
    """Remove the exact trailing Lark alias that names ``handler_name``."""

    return re.sub(rf"\s*->\s*{re.escape(handler_name)}\s*$", "", production).strip()


def grammar_production_for_handler(handler_name: str) -> str:
    """Return the production branch implemented by one transformer handler."""

    productions = grammar_productions()
    if handler_name in productions:
        production = _without_handler_alias(productions[handler_name], handler_name)
        return f"{handler_name}: {production}"
    for rule_name, production in productions.items():
        for branch in _production_branches(production):
            syntax = _without_handler_alias(branch, handler_name)
            if syntax != branch:
                return f"{rule_name}: {syntax}"
    raise KeyError(f"no grammar production for transformer handler {handler_name!r}")


def transformer_handler_names() -> frozenset[str]:
    """Return every lowercase transformer handler named by the grammar."""

    names = set(grammar_productions()) - {"start"}
    names.update(re.findall(r"->\s*([a-z][a-z0-9_]*)", DSL_GRAMMAR))
    return frozenset(names)
