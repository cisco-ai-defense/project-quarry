# Copyright 2026 Cisco Systems, Inc. and its affilitates
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

"""Differential property tests: the scanner's verdict against a direct reading of the rule.

Hypothesis generates rules and conversations (``tests/engine/generators.py``) and shrinks any
failure to a minimal case. Every generated rule the scanner loads must give, on every generated
conversation, the verdict the first-order reading gives; a rule it rejects at load fails closed.
The last test needs no reference semantics: it compares two evaluations of one rule that must
agree by contract.
"""

from __future__ import annotations

from hypothesis import event, given
from hypothesis import strategies as st

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.safety.limits import SafetyLimits
from quarry.scanner import DSLScanner
from quarry.types.exceptions import RuleValidationError
from tests.engine.generators import Rule, conversations, flow_rules, rules, two_join_rules

_CONVERSATIONS = st.lists(conversations(), min_size=1, max_size=4)
_SEMANTIC_LIMITS = SafetyLimits(rule_timeout=None, inference_timeout=None)


def _loaded(rule: Rule) -> DSLScanner | None:
    try:
        scanner = DSLScanner(
            QuarryConfig(quarry_rule_texts=(rule.source(),), safety_limits=_SEMANTIC_LIMITS)
        )
    except RuleValidationError:
        event("rejected at load")
        return None  # rejected at load: fail-closed, never a wrong verdict
    event("loaded")
    return scanner


def _agrees(rule: Rule, generated: list[dict]) -> None:
    scanner = _loaded(rule)
    if scanner is None:
        return
    if "tools" in rule.text or "has_tool" in rule.text:
        event("reads tool definitions")
    for conversation in generated:
        result = scanner.scan_data(conversation)
        assert result.completion_state == "complete", result.completion_issues
        expected = rule.holds({}, conversation)
        event("match" if expected else "no match")
        assert (result.rule_matches == ["R"]) == expected, (rule.text, conversation)


@given(rule=rules, generated=_CONVERSATIONS)
def test_compiled_rules_agree_with_their_first_order_meaning(rule, generated):
    _agrees(rule, generated)


@given(rule=flow_rules(), generated=_CONVERSATIONS)
def test_compiled_flows_agree_with_their_first_order_meaning(rule, generated):
    _agrees(rule, generated)


@given(rule=two_join_rules(), generated=_CONVERSATIONS)
def test_rules_with_two_joins_agree_with_their_first_order_meaning(rule, generated):
    # Two joins are decided one value or one shared element at a time, or fail to load rather
    # than share one value.
    _agrees(rule, generated)


def _outcome(result) -> tuple:
    return (result.rule_matches, result.completion_state)


@given(
    rule=st.one_of(rules, flow_rules()), generated=st.lists(conversations(), min_size=1, max_size=3)
)
def test_explain_and_policy_serialization_keep_the_result(rule, generated):
    scanner = _loaded(rule)
    if scanner is None:
        return
    policy = compile_policy(
        rule.source(),
        runtime_config=QuarryConfig(config_version=2, safety_limits=_SEMANTIC_LIMITS),
    )
    loaded = load_compiled_policy(policy.to_bytes())
    for conversation in generated:
        plain = _outcome(scanner.scan_data(conversation))
        event("match" if plain[0] else "no match")
        assert _outcome(scanner.scan_data(conversation, enable_explain=True)) == plain, "explain"
        assert _outcome(loaded.scan(conversation)) == _outcome(policy.scan(conversation))
