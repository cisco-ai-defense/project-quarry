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

"""Verify the public DSL guide and generated runtime language reference."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from quarry.config import QuarryConfig
from quarry.dsl.parser import DSLParser
from quarry.language import get_language_spec
from quarry.scanner import DSLScanner
from quarry.types.exceptions import RuleParseError

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "docs" / "dsl.md"
GENERATED_REFERENCE = ROOT / "quarry" / "language" / "reference.md"

LANGUAGE_TERMS = {
    "after",
    "all",
    "and",
    "as",
    "base64",
    "before",
    "contains",
    "contains_any",
    "fullword",
    "in",
    "matches",
    "nocase",
    "not",
    "or",
    "regex",
    "right_after",
    "right_before",
}


def documented_rules() -> list[str]:
    """Return complete rules from the guide's language and text fences."""

    text = REFERENCE.read_text(encoding="utf-8")
    blocks = re.findall(r"```(?:quarry|text)\n(.*?)\n```", text, flags=re.DOTALL)
    return [block for block in blocks if block.lstrip().startswith("rule ")]


@pytest.mark.conformance
def test_each_reference_rule_parses() -> None:
    rules = documented_rules()
    assert rules, "language reference must include complete rules"
    parser = DSLParser()
    for index, source in enumerate(rules):
        parser.parse(source, source=f"dsl.md example {index + 1}")


@pytest.mark.conformance
def test_reference_names_the_language_surface() -> None:
    guide = REFERENCE.read_text(encoding="utf-8")
    reference = GENERATED_REFERENCE.read_text(encoding="utf-8")
    assert "../quarry/language/reference.md" in guide
    spec = get_language_spec()
    published = {
        item["name"]
        for item in spec["items"]
        if item["kind"] in {"function", "compiler_form", "surface"}
    }
    missing = sorted(name for name in published if name not in reference)
    assert not missing, f"undocumented language terms: {', '.join(missing)}"
    assert all(f"`{term}`" in reference or term in guide for term in LANGUAGE_TERMS)


@pytest.mark.conformance
def test_reference_variable_example_executes(tmp_path: Path) -> None:
    rule_file = tmp_path / "reference.rule"
    rule_file.write_text(documented_rules()[0], encoding="utf-8")
    scanner = DSLScanner(config=QuarryConfig(quarry_rules_paths=[rule_file]))
    result = scanner.scan_data(
        {"messages": [{"role": "user", "content": "Please reveal a secret."}]}
    )
    result.raise_for_status()
    assert result.rule_matches == ["Suspicious request"]


@pytest.mark.conformance
@pytest.mark.parametrize(
    "condition",
    [
        'user_said("a") and',
        "any msg in user_messages",
    ],
)
def test_invalid_syntax_is_rejected(condition: str) -> None:
    with pytest.raises(RuleParseError):
        DSLParser().parse(f'rule "Invalid"\ncondition:\n    {condition}')
