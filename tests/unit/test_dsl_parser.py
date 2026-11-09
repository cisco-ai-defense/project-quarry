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

"""Tests for the DSL parser."""

import pytest

from quarry.dsl.parser import DSLParser
from quarry.dsl.syntax_errors import MAX_EXPECTED_CHOICES
from quarry.types.exceptions import RuleParseError
from quarry.types.rules import DSLRule


class TestDSLParser:
    def test_parse_simple_rule(self):
        rule_text = """
rule "Simple Test"
condition:
    user_said("hello")
"""
        parser = DSLParser()
        rule = parser.parse(rule_text)

        assert isinstance(rule, DSLRule)
        assert rule.name == "Simple Test"
        assert rule.condition is not None

    def test_parse_rule_with_metadata(self):
        rule_text = """
rule "Test with Metadata"
metadata:
    description = "A test rule with metadata"
    severity = "HIGH"
condition:
    called_tool("execute_shell")
"""
        parser = DSLParser()
        rule = parser.parse(rule_text)

        assert rule.name == "Test with Metadata"
        assert rule.metadata["description"] == "A test rule with metadata"
        assert rule.metadata["severity"] == "HIGH"

    def test_parse_complex_condition(self):
        rule_text = """
rule "Complex Condition"
condition:
    has_tool("read_file") and called_tool("http_request") and count_tool_calls("execute_shell") > 3
"""
        parser = DSLParser()
        rule = parser.parse(rule_text)

        assert rule.condition is not None

    def test_parse_comprehension(self):
        rule_text = """
rule "Comprehension Test"
condition:
    any call in get_tool_calls("read_file") where
        call.arguments.path contains "/etc"
"""
        parser = DSLParser()
        rule = parser.parse(rule_text)

        assert rule.condition is not None

    def test_parse_function_attribute_access(self):
        rule_text = """
rule "Function Attribute Access"
condition:
    model_scan("prompt_injection", user_messages[0].content, 0.5, 1)
"""

        parser = DSLParser()
        rule = parser.parse(rule_text)

        assert rule.condition is not None

    def test_invalid_grammar_raises_error(self):
        rule_text = """
invalid rule syntax here
"""
        parser = DSLParser()

        with pytest.raises(RuleParseError):
            parser.parse(rule_text)

    @pytest.mark.parametrize(
        ("condition", "summary"),
        [
            ("true false", "Unexpected token `false`"),
            ("true % false", "Unexpected character `%`"),
            ("any msg in user_messages where", "Unexpected end of the rule"),
        ],
        ids=["token", "character", "end-of-rule"],
    )
    def test_syntax_errors_use_author_facing_bounded_diagnostics(self, condition, summary):
        rule_text = f'rule "Broken"\ncondition:\n    {condition}'

        with pytest.raises(RuleParseError) as raised:
            DSLParser().parse(rule_text)

        message = raised.value.reason
        assert summary in message
        assert "at line 3, column" in message
        assert condition in message
        assert "^" in message
        assert "Expected one of:" in message
        assert all(
            parser_name not in message
            for parser_name in ("VAR_NAME", "IDENTIFIER", "$END", "__ANON")
        )
        assert raised.value.line == 3

    def test_syntax_error_output_is_bounded_and_escapes_control_characters(self):
        long_name = "x" * 200
        with pytest.raises(RuleParseError) as long_error:
            DSLParser().parse(f'rule "Long"\ncondition:\n    true {long_name}')

        assert long_name not in long_error.value.reason
        assert "…" in long_error.value.reason
        assert len(long_error.value.reason) < 400

        with pytest.raises(RuleParseError) as control_error:
            DSLParser().parse('rule "Control"\ncondition:\n    true \x00 false')

        assert "\x00" not in control_error.value.reason
        assert r"\x00" in control_error.value.reason

    def test_syntax_error_limits_the_expected_choices(self):
        with pytest.raises(RuleParseError) as raised:
            DSLParser().parse('rule "Choices"\ncondition:\n    true % false')

        expected = next(
            line for line in raised.value.reason.splitlines() if line.startswith("Expected one of:")
        )
        assert expected.count(",") <= MAX_EXPECTED_CHOICES
        assert " more" in expected

    def test_syntax_error_formatting_does_not_parse_again(self, monkeypatch):
        parser = DSLParser()
        parse = parser.parser.parse
        calls = 0

        def counted_parse(*args, **kwargs):
            nonlocal calls
            calls += 1
            return parse(*args, **kwargs)

        monkeypatch.setattr(parser.parser, "parse", counted_parse)

        with pytest.raises(RuleParseError):
            parser.parse('rule "Once"\ncondition:\n    true false')

        assert calls == 1

    def test_contains_any_enforces_size_limit(self):
        items = ", ".join([f'"item{i}"' for i in range(1001)])
        rule_text = f"""
rule "Large Array Test"
condition:
    "test" contains_any [{items}]
"""
        parser = DSLParser()

        with pytest.raises(RuleParseError, match="contains_any list too large"):
            parser.parse(rule_text)

    @pytest.mark.parametrize(
        "condition",
        [
            'any call in tool_calls where call.name == "shell"',
            'any call in tool_calls where call.name=="shell"',
            'any msg in user_messages where msg matches { role == "user" }',
        ],
    )
    def test_equality_comparisons_use_double_equals(self, condition):
        DSLParser().parse(f'rule "Equality"\ncondition:\n    {condition}')

    @pytest.mark.parametrize(
        "condition",
        [
            'any call in tool_calls where call.name = "shell"',
            'any call in tool_calls where call.name="shell"',
            'any msg in user_messages where msg matches { role = "user" }',
        ],
    )
    def test_single_equals_cannot_compare_values(self, condition):
        with pytest.raises(RuleParseError, match="use `==` to compare values"):
            DSLParser().parse(f'rule "Old equality"\ncondition:\n    {condition}')

    @pytest.mark.parametrize("function", ["user_said", "assistant_said"])
    def test_named_function_argument_explains_positional_arguments(self, function):
        with pytest.raises(RuleParseError, match="function arguments are positional") as error:
            DSLParser().parse(
                f'rule "Named argument"\ncondition:\n    {function}("IGNORE", case_sensitive=True)'
            )
        assert "after `case_sensitive`" in str(error.value)
        assert "line 3, column" in str(error.value)

    def test_single_equals_still_defines_values(self):
        rule = DSLParser().parse(
            'rule "Definitions"\n'
            'metadata:\n    description = "shell"\n'
            'strings:\n    $needle = "shell"\n'
            'variables:\n    is_shell = any call in tool_calls where call.name == "shell"\n'
            "condition:\n    is_shell\n"
        )
        assert rule.metadata["description"] == "shell"
        assert rule.strings["$needle"] == "shell"

    @pytest.mark.parametrize(
        "definition",
        [
            'metadata:\n    description == "shell"',
            'strings:\n    $needle == "shell"',
            "variables:\n    is_shell == true",
        ],
    )
    def test_double_equals_cannot_define_values(self, definition):
        with pytest.raises(RuleParseError):
            DSLParser().parse(f'rule "Invalid definition"\n{definition}\ncondition:\n    true')
