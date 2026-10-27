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

"""Public token records and typed exclusions beside Quarry's grammar."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

TokenCategory = Literal[
    "block_keyword",
    "flow_word",
    "operator",
    "modifier",
    "punctuation",
    "literal",
]
SyntaxExclusionReason = Literal["not_a_public_form", "rejected_spelling"]


@dataclass(frozen=True)
class TokenRecord:
    """One fixed word or symbol a rule author can write."""

    id: str
    spelling: str
    summary: str
    category: TokenCategory
    reserved_for_variables: bool = False


@dataclass(frozen=True)
class SyntaxExclusion:
    """Why one transformer handler does not get its own public construct record."""

    handler: str
    reason: SyntaxExclusionReason
    reference: str | None = None


_CATEGORY_SPELLINGS: dict[TokenCategory, tuple[str, ...]] = {
    "block_keyword": (
        "rule",
        "metadata:",
        "strings:",
        "variables:",
        "parameters:",
        "condition:",
        "collect:",
        "number",
        "text",
        "policy",
        "params",
    ),
    "flow_word": ("after", "before", "right_after", "right_before", "until", "within", "of"),
    "operator": (
        "any",
        "all",
        "and",
        "or",
        "not",
        "in",
        "as",
        "where",
        "for",
        "if",
        "contains",
        "contains_any",
        "matches",
        "=",
        "==",
        "!=",
        ">",
        "<",
        ">=",
        "<=",
        "~",
        "+",
        "-",
        "@",
        "#",
        "*",
    ),
    "modifier": ("nocase", "fullword", "base64", "regex"),
    "punctuation": ("(", ")", "[", "]", "{", "}", ".", ",", ":", "$", "//"),
    "literal": ("true", "false", "null"),
}

_RESERVED_FOR_VARIABLES = frozenset(
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
        "params",
    }
)
_SYMBOL_NAMES = {
    "!=": "not-equal",
    "#": "string-count-prefix",
    "$": "string-variable-prefix",
    "(": "left-parenthesis",
    ")": "right-parenthesis",
    "*": "wildcard",
    "+": "plus",
    ",": "comma",
    "-": "minus",
    ".": "dot",
    "//": "line-comment",
    ":": "colon",
    "<": "less-than",
    "<=": "less-than-or-equal",
    "=": "assign",
    "==": "equal-comparison",
    ">": "greater-than",
    ">=": "greater-than-or-equal",
    "@": "string-position-prefix",
    "[": "left-bracket",
    "]": "right-bracket",
    "{": "left-brace",
    "}": "right-brace",
    "~": "regex-match",
}


def _token_id(spelling: str) -> str:
    name = _SYMBOL_NAMES.get(spelling, spelling.removesuffix(":"))
    name = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"token:{name}"


TOKEN_RECORDS = tuple(
    TokenRecord(
        id=_token_id(spelling),
        spelling=spelling,
        summary=f"Quarry {category.replace('_', ' ')} `{spelling}`.",
        category=category,
        reserved_for_variables=spelling in _RESERVED_FOR_VARIABLES,
    )
    for category, spellings in _CATEGORY_SPELLINGS.items()
    for spelling in spellings
)


SYNTAX_EXCLUSIONS = (
    SyntaxExclusion("arguments", "not_a_public_form"),
    SyntaxExclusion("arith_expr", "not_a_public_form"),
    SyntaxExclusion("atom", "not_a_public_form"),
    SyntaxExclusion("comparison", "not_a_public_form"),
    SyntaxExclusion("comprehension", "not_a_public_form"),
    SyntaxExclusion("condition", "not_a_public_form"),
    SyntaxExclusion("expr", "not_a_public_form"),
    SyntaxExclusion("false_literal", "not_a_public_form"),
    SyntaxExclusion("flow_expr", "not_a_public_form"),
    SyntaxExclusion("identifier", "not_a_public_form"),
    SyntaxExclusion("in_expr", "not_a_public_form"),
    SyntaxExclusion("index_postfix", "not_a_public_form"),
    SyntaxExclusion("metadata_item", "not_a_public_form"),
    SyntaxExclusion("not_expr", "not_a_public_form"),
    SyntaxExclusion("null_literal", "not_a_public_form"),
    SyntaxExclusion("number_literal", "not_a_public_form"),
    SyntaxExclusion("postfix", "not_a_public_form"),
    SyntaxExclusion("primary", "not_a_public_form"),
    SyntaxExclusion("product_expr", "not_a_public_form"),
    SyntaxExclusion("quantifier", "not_a_public_form"),
    SyntaxExclusion("quantifier_all", "not_a_public_form"),
    SyntaxExclusion("quantifier_any", "not_a_public_form"),
    SyntaxExclusion("string_def", "not_a_public_form"),
    SyntaxExclusion("string_list", "not_a_public_form"),
    SyntaxExclusion("string_literal", "not_a_public_form"),
    SyntaxExclusion("string_wildcard", "not_a_public_form"),
    SyntaxExclusion("true_literal", "not_a_public_form"),
    SyntaxExclusion("value", "not_a_public_form"),
    SyntaxExclusion("variable_def", "not_a_public_form"),
)


__all__ = [
    "SYNTAX_EXCLUSIONS",
    "TOKEN_RECORDS",
    "SyntaxExclusion",
    "SyntaxExclusionReason",
    "TokenCategory",
    "TokenRecord",
]
