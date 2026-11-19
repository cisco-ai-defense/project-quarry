# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""A known positive call name remains readable through a selected message view."""

from __future__ import annotations

import pytest

from quarry import InvalidInputError
from quarry.types.messages import Messages


def transcript(fault_first: bool) -> list[dict]:
    calls = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "fault", "type": "other", "function": {"name": "lookup", "arguments": "{}"}}
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "good",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
    ]
    if not fault_first:
        calls.reverse()
    return [*calls, {"role": "user", "content": "outside selection"}]


@pytest.mark.parametrize("fault_first", [True, False])
@pytest.mark.parametrize("selection", ["role", "slice", "reordered"])
def test_positive_selected_call_name_precedes_unreadable_name_faults(fault_first, selection):
    source = Messages.from_openai(transcript(fault_first))
    if selection == "role":
        view = source.assistant_messages
    elif selection == "slice":
        view = source[:2]
    else:
        view = source[1::-1]
    assert source.called_tool("lookup")
    assert view.called_tool("lookup")
    assert view.called_tools("lookup")
    with pytest.raises(InvalidInputError):
        view.called_tool("absent")
    with pytest.raises(InvalidInputError):
        view.count_tool_calls("lookup")
    with pytest.raises(InvalidInputError):
        view.get_tool_calls("lookup")
    bad_index = 0 if fault_first else 1
    with pytest.raises(InvalidInputError):
        source[bad_index : bad_index + 1].called_tool("lookup")
    assert source[2:].called_tool("lookup") is False
    assert source[:0].called_tool("lookup") is False
