# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Public rules check the exact messages value supplied by their caller."""

import pytest

from quarry.config import QuarryConfig
from quarry.scanner import DSLScanner
from quarry.types.exceptions import RuleValidationError

_PI = 'rule "PI"\ncondition:\n    user_said("ignore")\n'
_CALLER = 'rule "R"\nimports:\n    pi = "PI"\ncondition:\n'
_CONVERSATION = {
    "messages": [
        {"role": "user", "content": "ignore"},
        {"role": "assistant", "content": "okay"},
    ]
}


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("pi(messages)", ["PI", "R"]),
        ("pi(messages.user_messages)", ["PI", "R"]),
        ("pi(messages[1:])", ["PI"]),
        ("pi(messages[0])", ["PI", "R"]),
    ],
)
def test_public_call_reads_selected_messages(condition: str, expected: list[str]) -> None:
    scanner = DSLScanner(QuarryConfig(quarry_rule_texts=(_PI + _CALLER + f"    {condition}\n",)))
    result = scanner.scan_data(_CONVERSATION)
    assert sorted(result.rule_matches) == expected
    assert result.evaluation_errors is None


@pytest.mark.parametrize(
    ("imports", "condition", "reason"),
    [
        ('    pi = "missing"', "pi(messages)", "invalid_import"),
        ('    pi = "PI"', "pi", "call_reference"),
        ('    pi = "PI"', "pi()", "call_arguments"),
    ],
)
def test_invalid_public_call_fails_at_load(imports: str, condition: str, reason: str) -> None:
    source = _PI + f'rule "R"\nimports:\n{imports}\ncondition:\n    {condition}\n'
    with pytest.raises(RuleValidationError) as failure:
        DSLScanner(QuarryConfig(quarry_rule_texts=(source,)))
    assert failure.value.details["reason"] == reason
