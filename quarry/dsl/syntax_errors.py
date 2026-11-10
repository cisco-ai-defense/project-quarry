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

"""Bounded, author-facing formatting for the first DSL syntax error."""

from collections.abc import Collection
from typing import Any

from lark import Lark
from lark.exceptions import UnexpectedCharacters, UnexpectedInput, UnexpectedToken
from lark.lexer import PatternStr

MAX_CONTEXT_CHARACTERS = 40
MAX_EXPECTED_CHOICES = 12
MAX_UNEXPECTED_CHARACTERS = 80

_TERMINAL_LABELS = {
    "IDENTIFIER": "a name",
    "METADATA_KEY": "a metadata key",
    "MODIFIER": "a string modifier",
    "NUMBER": "a number",
    "PARAM_NAME": "a parameter name",
    "PARAM_TYPE": "a parameter type",
    "PATTERN_OPERATOR": "a comparison operator",
    "SIGNED_INT": "an integer",
    "STRING": "a string",
    "STRING_VAR": "a $string reference",
    "VAR_NAME": "a variable name",
}


def format_syntax_error(error: UnexpectedInput, rule_text: str, parser: Lark) -> str:
    """Describe one failed parse without reparsing or exposing grammar terminal names."""

    position = _position(error, rule_text)
    line = rule_text.count("\n", 0, position) + 1
    column = position - rule_text.rfind("\n", 0, position)
    summary = _summary(error)
    lines = [f"{summary} at line {line}, column {column}.", _context(rule_text, position)]
    expected = _expected_choices(error, parser)
    if expected:
        shown = expected[:MAX_EXPECTED_CHOICES]
        choices = ", ".join(shown)
        if len(expected) > MAX_EXPECTED_CHOICES:
            choices += f", and {len(expected) - MAX_EXPECTED_CHOICES} more"
        lines.append(f"Expected one of: {choices}")
    return "\n".join(lines)


def _position(error: UnexpectedInput, rule_text: str) -> int:
    if isinstance(error, UnexpectedToken) and error.token.type == "$END":
        return len(rule_text.rstrip())
    position = getattr(error, "pos_in_stream", None)
    if not isinstance(position, int):
        position = len(rule_text)
    return min(max(position, 0), len(rule_text))


def _summary(error: UnexpectedInput) -> str:
    if isinstance(error, UnexpectedCharacters):
        return f"Unexpected character `{_display(error.char)}`"
    if isinstance(error, UnexpectedToken):
        if error.token.type == "$END":
            return "Unexpected end of the rule"
        return f"Unexpected token `{_display(error.token)}`"
    return "Unexpected syntax"


def _display(value: str) -> str:
    pieces: list[str] = []
    length = 0
    for character in value:
        escaped = repr(character)[1:-1].replace("`", r"\`")
        if length + len(escaped) > MAX_UNEXPECTED_CHARACTERS:
            return "".join(pieces)[: MAX_UNEXPECTED_CHARACTERS - 1] + "…"
        pieces.append(escaped)
        length += len(escaped)
    return "".join(pieces)


def _context(rule_text: str, position: int) -> str:
    left_limit = max(0, position - MAX_CONTEXT_CHARACTERS)
    newline = rule_text.rfind("\n", left_limit, position)
    start = newline + 1 if newline >= 0 else left_limit
    prefix = "…" if newline < 0 and start > 0 else ""

    right_limit = min(len(rule_text), position + MAX_CONTEXT_CHARACTERS)
    newline = rule_text.find("\n", position, right_limit)
    end = newline if newline >= 0 else right_limit
    suffix = "…" if newline < 0 and end < len(rule_text) else ""

    before = _display_context(rule_text[start:position])
    excerpt = _display_context(rule_text[start:end])
    caret = " " * (len(prefix) + len(before))
    return f"    {prefix}{excerpt}{suffix}\n    {caret}^"


def _display_context(value: str) -> str:
    return "".join(
        character if character.isprintable() else repr(character)[1:-1] for character in value
    )


def _expected_choices(error: UnexpectedInput, parser: Lark) -> tuple[str, ...]:
    labels = _terminal_labels(parser)
    choices = {
        labels[name] for name in _expected_names(error) if isinstance(name, str) and name in labels
    }
    return tuple(sorted(choices, key=str.casefold))


def _expected_names(error: UnexpectedInput) -> Collection[Any]:
    if isinstance(error, UnexpectedCharacters):
        interactive_parser = getattr(error, "interactive_parser", None)
        if interactive_parser is not None:
            return interactive_parser.accepts()
        # The lexer's ``allowed`` set can include terminals the parser would reject at this
        # position. Without parser state, omit choices rather than suggest invalid syntax.
        return ()
    if isinstance(error, UnexpectedToken):
        return error.accepts or error.expected or ()
    return ()


def _terminal_labels(parser: Lark) -> dict[str, str]:
    labels = dict(_TERMINAL_LABELS)
    labels["$END"] = "end of the rule"
    for terminal in parser.terminals:
        if terminal.name in labels or not isinstance(terminal.pattern, PatternStr):
            continue
        labels[terminal.name] = f"`{_display(terminal.pattern.value)}`"
    return labels


def validate_terminal_labels(parser: Lark) -> None:
    """Fail at parser construction when an author-visible regex terminal has no label."""

    labels = _terminal_labels(parser)
    ignored = set(parser.ignore_tokens)
    missing = sorted(
        terminal.name
        for terminal in parser.terminals
        if terminal.name not in labels and terminal.name not in ignored
    )
    if missing:
        raise ValueError(f"syntax error labels missing for terminals: {', '.join(missing)}")


__all__ = [
    "MAX_CONTEXT_CHARACTERS",
    "MAX_EXPECTED_CHOICES",
    "MAX_UNEXPECTED_CHARACTERS",
    "format_syntax_error",
    "validate_terminal_labels",
]
