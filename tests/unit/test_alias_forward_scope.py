# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Value aliases bind subsequent clauses and retain their declaration's scope."""

import pytest

from quarry import DSLScanner, compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.types.exceptions import RuleValidationError
from tests.unit.test_nested_rule_runtime_contracts import _policy

CHAT = {
    "messages": [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "needle"},
        {"role": "assistant", "content": "reply"},
        {"role": "assistant", "content": "more"},
    ]
}
CASES = [
    ("len(messages) > 1 and ((user_messages as messages) and len(messages) == 1)", True),
    ("len(messages) == 1 and ((user_messages as messages) and len(messages) == 1)", False),
    ("count(messages) == 4 and (user_messages as messages) and count(messages) == 1", True),
    (
        'messages[0].content == "policy" and (user_messages as messages) and messages[0].content == "needle"',
        True,
    ),
    (
        "len(user_messages) == 1 and (assistant_messages as user_messages) and len(user_messages) == 2",
        True,
    ),
    (
        '$p in messages[0].content and (user_messages as messages) and messages[0].content == "needle"',
        True,
    ),
    (
        "(messages as ns) and len(ns) == 4 and (user_messages as messages) and len(messages) == 1 and len(ns) == 4",
        True,
    ),
    (
        "(messages as original) and (original[1:] as tail) and len(original) == 4 and len(tail) == 3 and (user_messages as messages) and len(messages) == 1 and len(original) == 4",
        True,
    ),
    (
        "len(messages) == 4 and len(user_messages) == 1 and (assistant_messages as user_messages) and len(user_messages) == 2 and (user_messages as messages) and len(messages) == 2",
        True,
    ),
    ("len(messages) == 4 and (messages[1:] as messages) and len(messages) == 3", True),
]


def _source(condition):
    return 'rule "Read"\nstrings:\n    $p = "policy"\ncondition: ' + condition


@pytest.mark.parametrize(("condition", "matched"), CASES)
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True])
def test_aliases_do_not_rewrite_preceding_reads_or_capture_earlier_aliases(
    condition, matched, restored, explain
):
    config = QuarryConfig(config_version=2)
    policy = compile_policy(_source(condition), runtime_config=config)
    if restored:
        wire = policy.to_bytes()
        policy = load_compiled_policy(wire, runtime_config=config)
        assert policy.to_bytes() == wire
    result = policy.scan(CHAT, explain=explain)
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == (["Read"] if matched else [])


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("matched", [False, True])
def test_callee_forward_aliases_are_independent_of_the_callers_aliases(
    depth, restored, explain, matched
):
    bound = 4 if matched else 1
    source = f'rule "Leaf" condition: len(messages) == {bound} and (user_messages as messages) and len(messages) == 1'
    previous = "Leaf"
    for number in range(depth):
        name = f"Caller{number}"
        source += f'\nrule "{name}" imports: check = "{previous}" condition: (messages as messages) and check(messages)'
        previous = name
    policy = _policy(source, QuarryConfig(config_version=2), restored)
    result = policy.scan(CHAT, explain=explain)
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ([previous] if matched else [])


@pytest.mark.parametrize(
    "condition",
    [
        "(messages as ns) and (user_messages as ns) and len(ns) > 0",
        "len(messages) > 0 and ((messages as ns) and (messages[1:] as ns) and len(ns) > 0)",
    ],
)
def test_duplicate_aliases_still_fail_before_execution(condition):
    with pytest.raises(RuleValidationError, match="names two values in one"):
        DSLScanner(QuarryConfig(config_version=2, quarry_rule_texts=(_source(condition),)))
