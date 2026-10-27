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

"""Tests for conversation parsing."""

import pytest

from quarry.core.conversation import Conversation
from quarry.types.exceptions import ConversationValidationError
from quarry.types.messages import Messages


class TestConversation:
    def test_openai_conversation_exposes_messages_tools_calls_and_searches(
        self, sample_openai_conversation
    ):
        conversation = Conversation(
            sample_openai_conversation["messages"], sample_openai_conversation["tools"]
        )
        assert isinstance(conversation.messages, Messages)
        assert [tool.name for tool in conversation.tools] == ["read_file", "http_request"]
        read_tool = conversation.tool_by_name("read_file")
        assert read_tool is not None
        assert read_tool.name == "read_file"
        assert read_tool.description == "Read contents of a file"
        assert "path" in read_tool.parameters["properties"]
        assert len(conversation.tool_responses) == 1
        response = conversation.tool_responses[0]
        assert response.id == "call_123"
        assert response.name == "read_file"
        assert "root:x:0:0:root" in response.result
        assert conversation.has_tool("read_file")
        assert conversation.has_tool("http_request")
        assert not conversation.has_tool("nonexistent_tool")
        assert conversation.has_tools("read_file", "http_request")
        assert not conversation.has_tools("read_file", "nonexistent_tool")
        assert conversation.called_tool("read_file")
        assert conversation.called_tool("http_request")
        assert not conversation.called_tool("execute_shell")
        assert conversation.called_tools("read_file", "http_request")
        assert conversation.count_tool_calls("read_file") == 1
        assert conversation.count_tool_calls("http_request") == 1
        assert conversation.count_tool_calls("nonexistent") == 0
        read_calls = conversation.get_tool_calls("read_file")
        assert len(read_calls) == 1
        assert read_calls[0].name == "read_file"
        read_responses = conversation.get_tool_responses("read_file")
        assert len(read_responses) == 1
        assert read_responses[0].name == "read_file"
        assert conversation.user_said("help me read a file")
        assert conversation.user_said("HELP", case_sensitive=False)
        assert not conversation.user_said("HELP", case_sensitive=True)
        assert conversation.assistant_said("I'll help you")
        assert not conversation.assistant_said("nonexistent phrase")
        assert conversation.call_sequence() == ["read_file", "http_request"]
        assert conversation.has_sequence("read_file", "http_request")
        assert conversation.has_sequence("read_file")
        assert not conversation.has_sequence("http_request", "read_file")

    def test_system_prompt_and_malformed_optional_fields_have_explicit_behavior(self):
        messages = [
            {"role": "system", "content": "You are a coding assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        assert Conversation(messages).system_prompt == "You are a coding assistant"

        malformed_messages = [
            {"role": "user"},
            {"content": "Hello"},
            {"role": "assistant", "content": None},
        ]
        with pytest.raises(ConversationValidationError):
            Conversation(malformed_messages, [])
        malformed_tools = [
            {"name": "test_tool"},
            {"type": "function"},
        ]
        with pytest.raises((ConversationValidationError, ValueError)):
            Conversation([], malformed_tools)
        with pytest.raises(ConversationValidationError, match="empty"):
            Conversation.from_openai({"messages": []})
        with pytest.raises(ConversationValidationError, match="unsupported top-level"):
            Conversation.from_openai({"messages": messages, "unknown": True})
