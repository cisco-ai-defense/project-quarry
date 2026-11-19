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

"""Concrete runtime witnesses guard the formal verifier's literal semantics."""

import json

import pytest

from quarry.config import QuarryConfig
from quarry.safety.quality import MatchabilityAnswer, MatchabilityBasis, verify_matchability
from quarry.scanner import DSLScanner


def _scanner(predicate: str) -> DSLScanner:
    return DSLScanner(
        QuarryConfig(
            quarry_rule_texts=(
                f'rule "Literal" condition: any m in user_messages where {predicate}',
            )
        )
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("first", "second"),
    [
        (r"\u{41}", "A"),
        (r"\u{0}", "\0"),
        (r"\u{100000}", "\U00100000"),
        (r"\u{30000}", "\U00030000"),
        (r"\u{10ffff}", "\U0010ffff"),
        (r"\\u{41}", r"\u{41}"),
        ('a"b', "ab"),
        ("\\", ""),
        ("\0", ""),
        ("", "A"),
        ("😀", "A"),
        ("e\u0301", "é"),
    ],
)
def test_distinct_literals_with_a_runtime_witness_cannot_be_proved_impossible(
    first: str, second: str, reverse: bool
) -> None:
    if reverse:
        first, second = second, first
    scanner = _scanner(
        f"m.content == {json.dumps(first, ensure_ascii=False)} "
        f"and not (m.content == {json.dumps(second, ensure_ascii=False)})"
    )

    result = scanner.scan_data({"messages": [{"role": "user", "content": first}]})
    formal = verify_matchability(scanner.compiled_rules["Literal"], timeout_ms=1000)

    assert result.is_complete
    assert result.rule_matches == ["Literal"]
    assert formal.answer is MatchabilityAnswer.UNKNOWN
    assert "feasible model path" in (formal.reason or "")


@pytest.mark.parametrize(
    ("content", "needle", "matches"),
    [
        (r"\u{41}", "A", False),
        ("A", r"\u{41}", False),
        ("prefix \\u{41} suffix", r"\u{41}", True),
        ("prefix \U00100000 suffix", "\U00100000", True),
        ("\U00100000", r"\u{100000}", False),
        (r"\u{100000}", "\U00100000", False),
        ("\U0010ffff", "\U0010ffff", True),
        ("\U0010ffff", "\U0002ffff", False),
        ("before\0after", "\0", True),
        ("beforeafter", "\0", False),
        ("abc", "", False),
        ("", "", False),
    ],
)
def test_literal_substrings_agree_with_concrete_runtime_results(
    content: str, needle: str, matches: bool
) -> None:
    scanner = _scanner(
        f"m.content == {json.dumps(content, ensure_ascii=False)} "
        f"and m.content contains {json.dumps(needle, ensure_ascii=False)}"
    )

    result = scanner.scan_data({"messages": [{"role": "user", "content": content}]})
    formal = verify_matchability(scanner.compiled_rules["Literal"], timeout_ms=1000)

    assert result.is_complete
    assert result.rule_matches == (["Literal"] if matches else [])
    assert formal.answer is (MatchabilityAnswer.UNKNOWN if matches else MatchabilityAnswer.NO)
    if not matches:
        assert formal.basis is MatchabilityBasis.CONTENT_MODEL
