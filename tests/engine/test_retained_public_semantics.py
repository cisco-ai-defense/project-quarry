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

"""Public string and capture behavior retained by the compiled engine."""

import pytest

from quarry.config import QuarryConfig
from quarry.scanner import DSLScanner


@pytest.mark.parametrize(
    ("source", "matched_text", "clean_text", "rule_name", "captured_value"),
    [
        pytest.param(
            r"""rule "Regex variable"
strings:
    $pattern = "(?i)\bwtf\b" regex
condition:
    any m in user_messages where m.content ~ $pattern""",
            "wtf happened",
            "what happened",
            "Regex variable",
            None,
            id="regex-modified-variable-is-a-regex-operand",
        ),
        pytest.param(
            """rule "Mixed string set"
strings:
    $plain = "alpha"
    $nocase = "BETA" nocase
    $regex = "GAMMA-[0-9]+" regex
    $choices = ["missing", "delta"]
condition:
    any m in user_messages where all of ($plain, $nocase, $regex, $choices) in m.content""",
            "alpha beta GAMMA-42 delta",
            "alpha beta GAMMA-42",
            "Mixed string set",
            None,
            id="mixed-literal-array-and-regex-string-set",
        ),
        pytest.param(
            '''rule "Array capture"
strings:
    $patterns = ["password", "absent", "pwd", "pass"]
condition:
    any m in user_messages where $patterns in m.content as matched and matched == "pwd"''',
            "pwd first, password later",
            "password later",
            "Array capture",
            "pwd",
            id="array-string-captures-a-later-entry-that-matches-first",
        ),
    ],
)
def test_compiled_string_rules_preserve_public_match_and_clean_outcomes(
    source: str,
    matched_text: str,
    clean_text: str,
    rule_name: str,
    captured_value: str | None,
) -> None:
    scanner = DSLScanner(QuarryConfig(quarry_rule_texts=(source,)))

    matched = scanner.scan_data(
        {"messages": [{"role": "user", "content": matched_text}]}, enable_explain=True
    )
    clean = scanner.scan_data(
        {"messages": [{"role": "user", "content": clean_text}]}, enable_explain=True
    )

    assert matched.completion_state == clean.completion_state == "complete"
    assert matched.rule_matches == [rule_name]
    assert clean.rule_matches == []
    assert [match.captured_value for match in matched.detailed_matches] == [captured_value]
