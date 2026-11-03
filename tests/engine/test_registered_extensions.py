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

"""Load-bearing checks for registered JSON domains and their clocks."""

import pytest
from pydantic import BaseModel, ConfigDict

from quarry import DSLScanner, ExtensionSpec
from quarry.config import QuarryConfig
from quarry.types.exceptions import RuleValidationError


class Created(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    source_tool: str | None = None


class Checked(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allowed: bool


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    mode: str = "enforce"


EXTENSIONS = [
    ExtensionSpec("record", {"created": Created}),
    ExtensionSpec("audit", {"checked": Checked}, state=Policy),
]
CREATED_EVENT = {"id": "r", "type": "record.created", "payload": {"label": "x"}}
CHECKED_EVENT = {"id": "c", "type": "audit.checked", "payload": {"allowed": True}}
CONVERSATION = {
    "messages": [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "tool_a", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "done"},
    ]
}
TIMELINE = [
    {"message_index": 0},
    {"message_index": 1},
    {"tool_call_id": "call-1"},
    {"event_id": "r"},
    {"message_index": 2},
]
MIXED_FLOW = '(any item in record.created where item.id == "r") right_after called_tool("tool_a")'
EVENT_FLOW = (
    '(any check in audit.checked where check.id == "c") '
    'right_after (any item in record.created where item.id == "r")'
)
TWO_CLOCKS = (
    '(called_tool("tool_a") before(within: 1) '
    '(any reply in tool_responses where reply.content contains "done")) '
    f"and ({MIXED_FLOW})"
)
INTERLEAVED_TIMELINE = [*TIMELINE[:3], {"event_id": "c"}, *TIMELINE[3:]]


@pytest.mark.parametrize(
    ("condition", "conversation", "inputs", "state", "matches", "skipped"),
    [
        (
            "any item in record.created where item.payload.source_tool == null",
            None,
            {"events": [CREATED_EVENT]},
            "complete",
            True,
            None,
        ),
        (
            'any check in audit.checked where audit.state.mode == "enforce"',
            None,
            {"events": [CHECKED_EVENT], "extension_state": {"audit": {}}},
            "complete",
            True,
            None,
        ),
        (EVENT_FLOW, None, {"events": [CREATED_EVENT, CHECKED_EVENT]}, "complete", True, None),
        (
            MIXED_FLOW,
            CONVERSATION,
            {"events": [CREATED_EVENT], "timeline": TIMELINE},
            "complete",
            True,
            None,
        ),
        (
            TWO_CLOCKS,
            CONVERSATION,
            {"events": [CREATED_EVENT], "timeline": TIMELINE},
            "complete",
            True,
            None,
        ),
        (
            'called_tool("tool_a") before(within: 1) '
            '(any item in record.created where item.id == "r")',
            CONVERSATION,
            {"events": [CHECKED_EVENT, CREATED_EVENT], "timeline": INTERLEAVED_TIMELINE},
            "complete",
            False,
            None,
        ),
        (MIXED_FLOW, CONVERSATION, {"events": [CREATED_EVENT]}, "incomplete", False, None),
        (
            '(any item in record.created where item.id == "absent") '
            'right_after called_tool("tool_a")',
            CONVERSATION,
            {"events": [CREATED_EVENT]},
            "complete",
            False,
            None,
        ),
        (
            'not called_tool("tool_a")',
            None,
            {"events": [CREATED_EVENT]},
            "incomplete",
            False,
            ["R"],
        ),
        (
            'any check in audit.checked where audit.state.mode == "enforce"',
            None,
            {"events": []},
            "incomplete",
            False,
            None,
        ),
    ],
)
def test_extension_verdicts(condition, conversation, inputs, state, matches, skipped) -> None:
    scanner = DSLScanner(
        QuarryConfig(config_version=2, quarry_rule_texts=(f'rule "R"\ncondition: {condition}',)),
        extensions=EXTENSIONS,
    )
    result = scanner.scan_data(conversation, **inputs)
    assert result.completion_state == state
    assert result.rule_matches == (["R"] if matches else [])
    assert result.skipped_rules == skipped


@pytest.mark.parametrize(
    "bad_event",
    [
        {"id": "r", "type": "unknown.created", "payload": {"label": "x"}},
        {"id": "r", "type": "record.created", "payload": {"label": "x", "typo": True}},
    ],
)
def test_invalid_extension_event_fails_scan(bad_event) -> None:
    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            quarry_rule_texts=('rule "R"\ncondition: count(record.created) == 1',),
        ),
        extensions=EXTENSIONS,
    )
    result = scanner.scan_data(None, events=[bad_event])
    assert result.completion_state == "invalid"
    assert result.rule_matches == []


def test_unsupported_state_function_fails_at_rule_load() -> None:
    config = QuarryConfig(
        config_version=2,
        quarry_rule_texts=('rule "R"\ncondition: len(audit.state.mode) > 0',),
    )
    with pytest.raises(RuleValidationError, match="cannot read registered extension state"):
        DSLScanner(config, extensions=EXTENSIONS)
