# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Preserved calls consume resource limits regardless of their sender's role."""

from __future__ import annotations

import pytest

from quarry.config import QuarryConfig
from quarry.safety.limits import SafetyLimits
from quarry.scanner import DSLScanner
from quarry.types.context import ScanContext
from quarry.types.exceptions import ConversationValidationError

ROLES = ["assistant", "user", "system", "developer", "tool", "alien"]


def payload(role: str) -> dict:
    return {
        "messages": [
            {
                "role": role,
                "content": "visible",
                "tool_calls": [
                    {
                        "id": f"c{index}",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "é"},
                    }
                    for index in range(2)
                ],
            }
        ]
    }


def scanner(limits: SafetyLimits) -> DSLScanner:
    return DSLScanner(
        QuarryConfig(quarry_rule_texts=('rule "Always" condition: true',), safety_limits=limits)
    )


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize(
    "setting,over,boundary",
    [
        ("max_timeline_items", 2, 3),
        ("max_tool_calls", 1, 2),
        ("max_tool_argument_size", 1, 2),
    ],
)
def test_cross_role_calls_obey_each_limit_and_its_exact_boundary(role, setting, over, boundary):
    data = payload(role)
    tight = SafetyLimits(**{setting: over})
    with pytest.raises(ConversationValidationError, match=setting):
        ScanContext(data["messages"], limits=tight)
    rejected = scanner(tight).scan_data(data)
    assert rejected.completion_state == "invalid" and rejected.rule_matches == []
    assert setting in rejected.validation_error

    allowed = SafetyLimits(**{setting: boundary})
    context = ScanContext(data["messages"], limits=allowed)
    assert len(context.timeline) == 3
    assert len(context.conversation.tool_calls) == 2
    assert context.conversation.to_openai_dict() == data
    result = scanner(allowed).scan_data(data)
    assert result.is_complete and result.rule_matches == ["Always"]


@pytest.mark.parametrize("role", ROLES)
def test_cross_role_limit_errors_report_all_breaches_without_truncation(role):
    data = payload(role)
    limits = SafetyLimits(max_timeline_items=2, max_tool_calls=1, max_tool_argument_size=1)
    result = scanner(limits).scan_data(data)
    assert result.completion_state == "invalid" and result.rule_matches == []
    for setting in ["max_timeline_items", "max_tool_calls", "max_tool_argument_size"]:
        assert setting in result.validation_error
    assert "not truncated or partially skipped" in result.validation_error
    assert len(data["messages"][0]["tool_calls"]) == 2


@pytest.mark.parametrize("role", ROLES)
def test_cross_role_call_events_keep_their_owner_and_count_in_mixed_transcripts(role):
    data = payload(role)
    data["messages"].insert(0, {"role": "user", "content": "prefix"})
    with pytest.raises(ConversationValidationError, match="max_timeline_items"):
        ScanContext(data["messages"], limits=SafetyLimits(max_timeline_items=3))
    context = ScanContext(data["messages"], limits=SafetyLimits(max_timeline_items=4))
    assert len(context.timeline) == 4
    assert [event.message_index for _, event in context.timeline] == [0, 1, 1, 1]
    assert context.conversation.to_openai_dict() == data


@pytest.mark.parametrize("role", ROLES)
def test_explicitly_disabled_limits_still_preserve_cross_role_calls(role):
    data = payload(role)
    limits = SafetyLimits(
        enforce_limits=False, max_timeline_items=1, max_tool_calls=1, max_tool_argument_size=1
    )
    context = ScanContext(data["messages"], limits=limits)
    assert len(context.timeline) == 3 and context.conversation.to_openai_dict() == data
    result = scanner(limits).scan_data(data)
    assert result.is_complete and result.rule_matches == ["Always"]
