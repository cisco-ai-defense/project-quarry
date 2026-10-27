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

"""Tests for malformed conversation input."""

from ..conftest import scanner_from_rules


class TestMalformedInput:
    def test_missing_messages_key(self, tmp_path):
        rule = 'rule "Test"\ncondition: len(messages) > 0'
        scanner = scanner_from_rules(tmp_path, rule)

        result = scanner.scan_data({"tools": []})
        assert result.error is not None
        assert "messages" in result.error.lower()

    def test_messages_wrong_type(self, tmp_path):
        rule = 'rule "Test"\ncondition: len(messages) > 0'
        scanner = scanner_from_rules(tmp_path, rule)

        result = scanner.scan_data({"messages": "not a list"})
        assert result.error is not None
        assert "must be a list" in result.error

    def test_tools_wrong_type(self, tmp_path):
        rule = 'rule "Test"\ncondition: len(messages) > 0'
        scanner = scanner_from_rules(tmp_path, rule)

        result = scanner.scan_data(
            {"messages": [{"role": "user", "content": "test"}], "tools": "not a list"}
        )
        assert result.error is not None
        assert "must be a list" in result.error

    def test_malformed_message_structure(self, tmp_path):
        rule = 'rule "Test"\ncondition: len(messages) > 0'
        scanner = scanner_from_rules(tmp_path, rule)

        for message in (
            {"content": "test"},
            {"role": "user"},
            {"role": "assistant", "tool_calls": 1},
        ):
            result = scanner.scan_data({"messages": [message]})
            assert result.completion_state == "invalid"
            assert result.validation_error

    def test_malformed_tool_call_arguments(self, tmp_path):
        rule = 'rule "Test"\ncondition: called_tool("test_tool")'
        scanner = scanner_from_rules(tmp_path, rule)

        conversation = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {
                                "name": "test_tool",
                                "arguments": "invalid json {{{",
                            },
                        }
                    ],
                }
            ]
        }

        result = scanner.scan_data(conversation)
        assert isinstance(result.rule_matches, list)

    def test_none_values_in_fields(self, tmp_path):
        rule = 'rule "Test"\ncondition: any msg in messages where msg.content contains "test"'
        scanner = scanner_from_rules(tmp_path, rule)

        conversation = {
            "messages": [
                {"role": "user", "content": None},
                {"role": "assistant", "content": ""},
            ]
        }

        result = scanner.scan_data(conversation)
        assert result.rule_matches == []

    def test_extremely_nested_tool_arguments(self, tmp_path):
        rule = (
            'rule "Test"\ncondition: any call in tool_calls where call.arguments.a.b.c.d.e != null'
        )
        scanner = scanner_from_rules(tmp_path, rule)

        conversation = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {
                                "name": "test",
                                "arguments": '{"a": {"b": null}}',
                            },
                        }
                    ],
                }
            ]
        }

        result = scanner.scan_data(conversation)
        assert isinstance(result.rule_matches, list)
