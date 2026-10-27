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

"""Public conversation and scanner behavior retained from the first OSS engine."""

import json
from typing import ClassVar

import pytest

from quarry import (
    AssistantMessage,
    AudioPart,
    Conversation,
    ConversationValidationError,
    ImagePart,
    Messages,
    SystemMessage,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from quarry.config import QuarryConfig
from quarry.scanner import DSLScanner


class ShellToolCall(ToolCall):
    """Typed view of a known tool call exposed through the public registry."""

    dsl_fields: ClassVar[frozenset[str]] = frozenset({"command"})

    @property
    def command(self) -> str:
        return self.arguments["command"]


def _conversation_data() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Follow policy."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect "},
                    {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
                    {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
                    {"type": "text", "text": "this"},
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "shell", "arguments": '{"command":"ls"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "file.txt"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "shell", "parameters": {"type": "object"}},
            }
        ],
    }


def test_conversation_keeps_one_typed_message_graph_and_round_trips() -> None:
    data = _conversation_data()
    conversation = Conversation.from_openai(
        data, tool_call_types={("function", "shell"): ShellToolCall}
    )

    assert isinstance(conversation.messages, Messages)
    assert [type(message) for message in conversation.messages] == [
        SystemMessage,
        UserMessage,
        AssistantMessage,
        ToolMessage,
    ]
    assert conversation.system_prompt == "Follow policy."
    user = conversation.messages[1]
    assert user.content == "inspect this"
    assert [type(part) for part in user.content_parts] == [
        TextPart,
        ImagePart,
        AudioPart,
        TextPart,
    ]
    assert conversation.user_said("INSPECT")
    assert not conversation.assistant_said("inspect")

    call = conversation.get_tool_calls("shell")[0]
    assert isinstance(call, ShellToolCall)
    assert call.command == "ls"
    assert call.response is conversation.tool_responses[0]
    assert call.response.call is call
    assert call.response.result == "file.txt"
    assert conversation.has_tool("shell")
    assert conversation.called_tool("shell")
    assert conversation.count_tool_calls("shell") == 1
    assert conversation.call_sequence() == ["shell"]
    assert conversation.has_sequence("shell")
    assert conversation.to_openai_dict() == data
    assert json.loads(conversation.to_json()) == data

    projection = conversation.raw_messages
    projection[1]["content"][0]["text"] = "changed"
    assert conversation.messages[1].content == "inspect this"
    with pytest.raises(AttributeError, match="Conversation is immutable"):
        conversation.messages = Messages()
    with pytest.raises(ConversationValidationError, match="root Messages"):
        Conversation(conversation.messages[1:])


def test_compiled_scanner_uses_the_same_public_tool_and_message_contract() -> None:
    scanner = DSLScanner(
        QuarryConfig(
            quarry_rule_texts=(
                'rule "Tool observed"\ncondition: called_tool("shell") and '
                'any m in user_messages where m.content contains "inspect"',
            )
        )
    )

    result = scanner.scan_data(_conversation_data(), enable_explain=True)

    assert result.is_complete, result.completion_issues
    assert result.rule_matches == ["Tool observed"]
    assert result.detailed_matches
    assert result.detailed_matches[0].to_dict()["rule_name"] == "Tool observed"


def test_untrusted_conversation_data_is_validated_at_entry() -> None:
    with pytest.raises(ConversationValidationError, match="Expected mapping"):
        Conversation.from_openai([])  # type: ignore[arg-type]
    with pytest.raises(ConversationValidationError):
        Conversation.from_openai({"messages": [{"role": "unknown", "content": "text"}]})
