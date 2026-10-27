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

"""Compiler diagnostics and source maps.

When the compiler lowers a rule AST into the typed IR, it must be able to
report precisely *what* went wrong and *where* — citing the rule, the offending
construct, and the AST node type — rather than raising an opaque error deep in
progression or DFA compilation. This module provides:

- :class:`SourceRef` — a back-reference from an IR node to the AST node it came from
  (node type + a human path + optional rule name), so errors and evidence can point
  back at the source.
- :class:`Diagnostics` — a collector the compiler passes through its passes; it can
  gather multiple errors/warnings before failing, so one bad rule reports all its
  problems at once instead of one-at-a-time.

Pure core module (no Quarry imports): the compiler supplies AST node-type *names* and
paths as plain strings, so diagnostics never depend on the AST classes themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ErrorCategory(StrEnum):
    """What kind of compile problem occurred."""

    COMPATIBILITY = "compatibility"
    SCOPE = "scope"
    TYPE = "type"
    ARITY = "arity"
    UNSUPPORTED = "unsupported"
    REFERENCE = "reference"
    INTERNAL = "internal"


@dataclass(frozen=True)
class SourceRef:
    """A back-reference from an IR node to its originating AST node.

    ``node_type`` is the AST class name (e.g. ``"FlowOp"``); ``path`` is a
    human-readable locator (e.g. ``"condition.left"`` or ``"any c in tool_calls"``);
    ``rule_name`` names the rule when known. All strings, so this module stays
    AST-agnostic.
    """

    node_type: str = ""
    path: str = ""
    rule_name: str = ""


@dataclass
class Diagnostic:
    """A single reported problem (error or warning)."""

    message: str
    category: ErrorCategory
    source: SourceRef
    is_error: bool = True


@dataclass
class Diagnostics:
    """Collector for compiler diagnostics across passes.

    Lets a pass accumulate several problems before the compiler decides to fail, so a
    rule with three issues reports all three rather than aborting on the first.
    """

    items: list[Diagnostic] = field(default_factory=list)

    def error(self, message: str, category: ErrorCategory, source: SourceRef | None = None) -> None:
        """Add an error diagnostic."""
        self.items.append(Diagnostic(message, category, source or SourceRef(), is_error=True))

    def warn(self, message: str, category: ErrorCategory, source: SourceRef | None = None) -> None:
        """Add a warning diagnostic."""
        self.items.append(Diagnostic(message, category, source or SourceRef(), is_error=False))

    @property
    def errors(self) -> list[Diagnostic]:
        """Return the list of error diagnostics."""
        return [d for d in self.items if d.is_error]

    @property
    def warnings(self) -> list[Diagnostic]:
        """Return the list of warning diagnostics."""
        return [d for d in self.items if not d.is_error]

    @property
    def has_errors(self) -> bool:
        """Return whether the diagnostics contain any errors."""
        return any(d.is_error for d in self.items)


class UnsupportedConstruct(Exception):
    """Internal signal that a construct can't be lowered; caught by the compiler and
    turned into a located diagnostic."""
