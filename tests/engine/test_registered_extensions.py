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
from quarry import scanner as scanner_module
from quarry.config import QuarryConfig
from quarry.safety.limits import SafetyLimits
from quarry.safety.timer import QuarryTimeoutError, current_evaluation_deadline
from quarry.types.exceptions import ConfigurationError, RuleValidationError


class Created(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    label: str
    source_tool: str | None = None


class Checked(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allowed: bool


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    mode: str = "enforce"


class Tagged(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tags: list[str]


class BadDefault(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    count: int = "bad"


class NestedBadDefault(BaseModel):
    model_config = ConfigDict(extra="forbid")
    details: BadDefault


EXTENSIONS = [
    ExtensionSpec("record", {"created": Created}),
    ExtensionSpec("audit", {"checked": Checked}, state=Policy),
]
TAGGED_EXTENSIONS = [ExtensionSpec("record", {"created": Tagged})]
BAD_DEFAULT_EXTENSIONS = [ExtensionSpec("record", {"created": NestedBadDefault})]
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
PAIR_EVENTS = [CREATED_EVENT, CHECKED_EVENT]
EMPTY_CONVERSATION = {"messages": []}
RECORD_EVENTS = {"events": [CREATED_EVENT]}
OBSERVED_FEEDS = {"events": [CREATED_EVENT], "event_feeds": ["record", "audit"]}
OPTIONAL_RULE = "any item in record.created where item.payload.source_tool == null"
STATE_RULE = 'any check in audit.checked where audit.state.mode == "enforce"'
CORRECTNESS_LIMITS = SafetyLimits(rule_timeout=None)
STATE_ERROR = "cannot read registered extension state"
COLLECTION_ERROR = "Extension JSON collection/object operations"


@pytest.mark.parametrize(
    ("condition", "conversation", "inputs", "state", "matches", "skipped"),
    [
        (OPTIONAL_RULE, None, RECORD_EVENTS, "complete", True, None),
        (
            STATE_RULE,
            None,
            {"events": [CHECKED_EVENT], "extension_state": {"audit": {}}},
            "complete",
            True,
            None,
        ),
        (f"not ({EVENT_FLOW})", None, {"events": PAIR_EVENTS}, "incomplete", False, ["R"]),
        (EVENT_FLOW, EMPTY_CONVERSATION, {"events": PAIR_EVENTS}, "complete", True, None),
        ("count(audit.checked) == 0", CONVERSATION, RECORD_EVENTS, "incomplete", False, ["R"]),
        ("count(audit.checked) == 0", CONVERSATION, OBSERVED_FEEDS, "complete", True, None),
        (
            TWO_CLOCKS,
            CONVERSATION,
            {**OBSERVED_FEEDS, "timeline": TIMELINE},
            "complete",
            True,
            None,
        ),
        (MIXED_FLOW, CONVERSATION, OBSERVED_FEEDS, "incomplete", False, None),
        ('not called_tool("tool_a")', None, RECORD_EVENTS, "incomplete", False, ["R"]),
        (STATE_RULE, None, {"events": []}, "incomplete", False, None),
    ],
)
def test_extension_verdicts(condition, conversation, inputs, state, matches, skipped) -> None:
    scanner = DSLScanner(
        QuarryConfig(config_version=2, quarry_rule_texts=(f'rule "R"\ncondition: {condition}',)),
        extensions=EXTENSIONS,
        limits=CORRECTNESS_LIMITS,
    )
    result = scanner.scan_data(conversation, **inputs)
    assert result.completion_state == state
    assert result.rule_matches == (["R"] if matches else [])
    assert result.skipped_rules == skipped


@pytest.mark.parametrize(
    ("inputs", "limit_overrides", "diagnostic"),
    [
        (
            {"events": [{**CREATED_EVENT, "payload": {"label": "x", "typo": True}}]},
            {},
            "Invalid payload",
        ),
        ({**RECORD_EVENTS, "timeline": [{"event_id": "missing"}]}, {}, "unknown or repeated"),
        (RECORD_EVENTS, {"max_nesting_depth": 3}, "max_nesting_depth"),
    ],
)
def test_invalid_extension_input_fails_scan(inputs, limit_overrides, diagnostic) -> None:
    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            quarry_rule_texts=('rule "R"\ncondition: count(record.created) == 1',),
        ),
        extensions=EXTENSIONS,
        limits=SafetyLimits(rule_timeout=None, **limit_overrides),
    )
    result = scanner.scan_data(None, **inputs)
    assert result.completion_state == "invalid"
    assert result.rule_matches == []
    assert diagnostic in str(result.validation_error)


def test_bounded_flow_needs_every_event_feed() -> None:
    scanner = DSLScanner(
        QuarryConfig(config_version=2, quarry_rule_texts=(f'rule "R"\ncondition: {EVENT_FLOW}',)),
        extensions=[*EXTENSIONS, ExtensionSpec("background", {"tick": Checked})],
        limits=CORRECTNESS_LIMITS,
    )
    missing = scanner.scan_data(EMPTY_CONVERSATION, events=PAIR_EVENTS)
    observed = scanner.scan_data(
        EMPTY_CONVERSATION, events=PAIR_EVENTS, event_feeds=["record", "audit", "background"]
    )
    assert missing.completion_state == "incomplete"
    assert (observed.completion_state, observed.rule_matches) == ("complete", ["R"])


@pytest.mark.parametrize(
    ("condition", "extensions", "error_type", "diagnostic"),
    [
        (
            "audit.state.mode as s and len(s) > 0",
            EXTENSIONS,
            RuleValidationError,
            STATE_ERROR,
        ),
        *[
            (
                f"any item in record.created where {operation}",
                TAGGED_EXTENSIONS,
                ConfigurationError,
                COLLECTION_ERROR,
            )
            for operation in (
                "count(item.payload.tags) == 2",
                'item.payload.tags contains_any ["af"]',
            )
        ],
        (
            "count(record.created) == 1",
            BAD_DEFAULT_EXTENSIONS,
            ConfigurationError,
            "invalid default",
        ),
    ],
)
def test_invalid_extension_rule_or_schema_rejected_at_load(
    condition, extensions, error_type, diagnostic
) -> None:
    config = QuarryConfig(
        config_version=2, quarry_rule_texts=(f'rule "R"\ncondition: {condition}',)
    )
    with pytest.raises(error_type, match=diagnostic):
        DSLScanner(config, extensions=extensions, limits=CORRECTNESS_LIMITS)


def test_cross_source_preflight_obeys_rule_deadline(monkeypatch) -> None:
    def expire_during_preflight() -> None:
        # Simulate a spent budget at the preflight checkpoint, without wall-clock sleeps.
        if deadline := current_evaluation_deadline():
            raise QuarryTimeoutError(deadline.error_message)

    monkeypatch.setattr(scanner_module, "raise_if_evaluation_expired", expire_during_preflight)
    scanner = DSLScanner(
        QuarryConfig(config_version=2, quarry_rule_texts=(f'rule "R"\ncondition: {MIXED_FLOW}',)),
        extensions=EXTENSIONS,
        limits=SafetyLimits(rule_timeout=1),
    )
    result = scanner.scan_data(
        CONVERSATION, events=[CREATED_EVENT], event_feeds=["record", "audit"]
    )
    assert result.timed_out_rules == ["R"]
