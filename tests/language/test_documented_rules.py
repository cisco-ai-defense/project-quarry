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

"""Every rule example in the user-facing documentation must parse and compile.

Hand-written docs drift from the language silently: an example keeps showing a form the
compiler no longer accepts, or a shape that was never supported. This test executes them.

A fenced block with no language tag, ``quarry``, or ``quarry-condition`` is an example. One
whose first non-comment line starts with ``rule`` must be a complete rule that parses and
compiles. Any other block is compiled as a rule condition; an untagged or ``quarry`` block
that does not parse as a condition is a grammar illustration and is skipped, while an
``quarry-condition`` block must parse. Tag a template with placeholders as ``text``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from quarry.dsl.parser import DSLParser
from quarry.engine.compiler import compile_rule
from quarry.types.exceptions import RuleParseError

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "examples" / "README.md",
    *sorted((REPO_ROOT / "docs").glob("*.md")),
    *sorted((REPO_ROOT / "docs").glob("*.MD")),
]
_FENCE = re.compile(r"^```([A-Za-z0-9_-]*)\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _examples() -> list[tuple[str, str, str]]:
    examples = []
    for document in DOCUMENTS:
        text = document.read_text()
        for match in _FENCE.finditer(text):
            language, body = match.group(1), match.group(2).strip()
            if language not in ("", "quarry", "quarry-condition") or not body:
                continue
            line = text.count("\n", 0, match.start()) + 1
            examples.append((f"{document.relative_to(REPO_ROOT)}:{line}", language, body))
    return examples


@pytest.mark.parametrize(
    ("location", "language", "body"), _examples(), ids=[loc for loc, _, _ in _examples()]
)
def test_documented_rule_example_compiles(location: str, language: str, body: str) -> None:
    lines = body.splitlines()
    while lines and (not lines[0].strip() or lines[0].lstrip().startswith("//")):
        lines.pop(0)
    if lines and lines[0].startswith("rule "):
        source = body
    else:
        source = 'rule "documented"\ncondition:\n    ' + body.replace("\n", "\n    ")
        try:
            DSLParser().parse(source)
        except RuleParseError:
            # An ``quarry-condition`` block states that it is a condition, so it must parse.
            if language == "quarry-condition":
                raise
            pytest.skip("not a rule condition (grammar illustration)")

    result = compile_rule(DSLParser().parse(source))

    assert result.ok, f"{location}: {[d.message for d in result.diagnostics.errors]}"
