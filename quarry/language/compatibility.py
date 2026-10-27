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

"""Rejected spellings documented outside the runtime language catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CompatibilityRecord:
    """One concrete spelling and the public pipeline stage that must reject it."""

    source_form: str
    rejected_by: Literal["parse", "validate", "link", "compile"]
    replacement: str | None = None


MAX_COMPATIBILITY_RECORDS = 40
"""The most records this table may hold before it has to be paginated.

The plugin serves the whole table in one response so an author sees every mistake to avoid
before writing a rule, which only stays affordable while the table is small. Generation
fails once this bound is passed, so growing the table forces a deliberate decision — raise
the bound, or start paginating — instead of quietly inflating every response."""


COMPATIBILITY_RECORDS = (
    CompatibilityRecord(
        'any call in tool_calls where call.name = "shell"',
        "parse",
        'any call in tool_calls where call.name == "shell"',
    ),
    CompatibilityRecord(
        'any msg in user_messages where msg matches { role = "user" }',
        "parse",
        'any msg in user_messages where msg matches { role == "user" }',
    ),
    CompatibilityRecord(
        'llm_score("m", "policy", conversation) > 0.5',
        "compile",
        'llm_score("m", "policy", conversation()) > 0.5',
    ),
    CompatibilityRecord("messages()", "link", "messages"),
    CompatibilityRecord("tools()", "link", "tools"),
    CompatibilityRecord("tool_calls()", "link", "tool_calls"),
    CompatibilityRecord("tool_responses()", "link", "tool_responses"),
    CompatibilityRecord("user_messages()", "link", "user_messages"),
    CompatibilityRecord("assistant_messages()", "link", "assistant_messages"),
    CompatibilityRecord("system_messages()", "link", "system_messages"),
    CompatibilityRecord("developer_messages()", "link", "developer_messages"),
    CompatibilityRecord("system_prompt()", "link", "system_prompt"),
    CompatibilityRecord("first(tool_calls)", "link", "tool_calls[0]"),
    CompatibilityRecord("last(tool_calls)", "link", "tool_calls[-1]"),
    CompatibilityRecord(
        'any msg in messages where regex_match(msg.content, "pattern")',
        "link",
        'any msg in messages where msg.content ~ "pattern"',
    ),
    CompatibilityRecord('regex_match_case("text", "pattern")', "link"),
    CompatibilityRecord('regex_extract("text", "pattern")', "link"),
    CompatibilityRecord(
        'user_said("IGNORE", case_sensitive=True)',
        "parse",
        'user_said("IGNORE", true)',
    ),
    CompatibilityRecord(
        'assistant_said("IGNORE", case_sensitive=True)',
        "parse",
        'assistant_said("IGNORE", true)',
    ),
    CompatibilityRecord(
        'model_score("m", system_prompt, 1)',
        "compile",
        'model_score("m", system_prompt, 1) > 0.5',
    ),
    CompatibilityRecord(
        "topic_drift(system_prompt)",
        "compile",
        "topic_drift(system_prompt) > 0.5",
    ),
    CompatibilityRecord(
        'llm_score("m", "policy", conversation())',
        "compile",
        'llm_score("m", "policy", conversation()) > 0.5',
    ),
    CompatibilityRecord(
        'llm_input_tokens("m", conversation())',
        "compile",
        'llm_input_tokens("m", conversation()) > 100',
    ),
    CompatibilityRecord(
        'model_score("m", get_tool_calls("read")[0].arguments, 1)',
        "compile",
    ),
    CompatibilityRecord(
        'model_score("m", get_tool_responses("read")[0].content, 1)',
        "compile",
    ),
    CompatibilityRecord("[msg.content for msg in user_messages]", "compile"),
    CompatibilityRecord(
        'msg.content matches { content contains "sudo" }',
        "compile",
        'msg matches { content contains "sudo" }',
    ),
)


__all__ = ["COMPATIBILITY_RECORDS", "MAX_COMPATIBILITY_RECORDS", "CompatibilityRecord"]
