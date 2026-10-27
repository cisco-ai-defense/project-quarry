# Copyright 2025 Cisco Systems, Inc. and its affiliates
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

"""Tests for rule validator."""

import pytest

from quarry.dsl.parser import DSLParser
from quarry.safety.validator import Complexity, RuleValidator
from quarry.types.dsl import (
    Assignment,
    ContainsAnyOp,
    FieldAccess,
    Identifier,
    Literal,
    MapComprehension,
    PatternMatch,
)
from quarry.types.rules import DSLRule


def test_validator_reports_invalid_pattern_braces_without_logger_failure() -> None:
    validator = RuleValidator()

    error = validator._validate_single_regex("({foo}")

    assert error is not None


def test_validator_flags_unbounded_flow_op():
    rule_text = """
rule "Unbounded Flow"
condition:
    called_tool("a") after called_tool("b")
"""

    parser = DSLParser()
    rule = parser.parse(rule_text)

    validator = RuleValidator()
    result = validator.validate(rule)

    assert len(result.warnings) > 0
    assert any("Unbounded 'after' operator" in w.message for w in result.warnings)


def test_validator_accepts_bounded_flow_op():
    rule_text = """
rule "Bounded Flow"
condition:
    called_tool("a") after(within: 3) called_tool("b")
"""

    parser = DSLParser()
    rule = parser.parse(rule_text)

    validator = RuleValidator()
    result = validator.validate(rule)

    unbounded_warnings = [w for w in result.warnings if "Unbounded" in w.message]
    assert len(unbounded_warnings) == 0


def test_validator_does_not_report_until_as_unbounded_after():
    rule = DSLParser().parse(
        'rule "Until"\ncondition:\n    called_tool("approve") until called_tool("publish")'
    )

    warnings = RuleValidator().validate(rule).warnings

    assert not any("Unbounded 'after' operator" in warning.message for warning in warnings)


def test_validator_estimates_complexity():
    simple_rule = """
rule "Simple"
condition:
    user_said("test")
"""

    complex_rule = """
rule "Complex"
condition:
    any msg in user_messages where
        msg.content ~ "(a+)+b" and
    any call in tool_calls where
        call.name == "test"
    after(within: 5)
    user_said("trigger")
"""

    parser = DSLParser()
    validator = RuleValidator()

    simple_result = validator.validate(parser.parse(simple_rule))
    complex_result = validator.validate(parser.parse(complex_rule))

    complexity_order = [Complexity.LOW, Complexity.MEDIUM, Complexity.HIGH, Complexity.EXTREME]
    simple_idx = complexity_order.index(simple_result.estimated_complexity)
    complex_idx = complexity_order.index(complex_result.estimated_complexity)

    assert simple_idx < complex_idx


@pytest.mark.parametrize(
    "condition",
    [
        FieldAccess(Identifier("$missing"), ["name"]),
        PatternMatch([Identifier("$missing")]),
        MapComprehension(Identifier("$missing"), "item", Literal([])),
        Assignment(Identifier("$missing"), "value"),
        ContainsAnyOp(Identifier("$missing"), ["value"]),
    ],
)
def test_validator_checks_variables_beneath_every_container(condition) -> None:
    rule = DSLRule("Nested variable", {}, condition)

    result = RuleValidator().validate(rule)

    assert [error.error_type for error in result.semantic_errors] == ["undefined_variable"]
