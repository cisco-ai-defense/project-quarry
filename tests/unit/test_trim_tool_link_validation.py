# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""A partial trim must not inherit split's flattened-evidence link exemption."""

import pytest

from tests.unit.test_operation_input_trim import trim_policy


@pytest.mark.parametrize("operation", ["score", "classify", "complete", "spans"])
@pytest.mark.parametrize("restored", [False, True])
def test_partial_trim_rejects_a_result_after_its_call_was_cut(tmp_path, operation, restored):
    data = {
        "messages": [
            {
                "role": "assistant",
                "content": "long assistant text",
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c", "content": "ok"},
        ]
    }
    policy, adapter, _ = trim_policy(
        tmp_path, operation, 'trim(messages, 6, from="end")', restored=restored
    )
    result = policy.scan(data)
    assert not result.is_complete
    assert "exact call/result links" in result.evaluation_errors["Read"]
    assert adapter.score_calls == adapter.classifications == adapter.completions == 0
    assert adapter.tokenizations > 0


@pytest.mark.parametrize("operation", ["score", "classify", "complete", "spans"])
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("selection", ["whole", "prefix-user", "suffix-user", "partial-result"])
def test_trim_keeps_valid_tool_links_and_exact_selected_payloads(
    tmp_path, operation, restored, selection
):
    assistant = {
        "role": "assistant",
        "content": "long assistant text",
        "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        ],
    }
    tool = {"role": "tool", "tool_call_id": "c", "content": "ok"}
    data = {"messages": [assistant, tool]}
    expected = data["messages"]
    expression = "trim(messages, 100)"
    if selection == "prefix-user":
        data = {"messages": [assistant, tool, {"role": "user", "content": "trailingtext"}]}
        expected = [assistant, tool, {"role": "user", "content": "tra"}]
        expression = 'trim(messages, 24, from="start")'
    elif selection == "suffix-user":
        data = {"messages": [{"role": "user", "content": "leadingtext"}, assistant, tool]}
        expected = [{"role": "user", "content": "lea"}, assistant, tool]
        expression = 'trim(messages, 24, from="end")'
    elif selection == "partial-result":
        data = {"messages": [assistant, {**tool, "content": "ok-long-result"}]}
        expected = [assistant, {**tool, "content": "ok-l"}]
        expression = "trim(messages, 23, context=messages[:1])"
    policy, adapter, _ = trim_policy(tmp_path, operation, expression, restored=restored)
    result = policy.scan(data)
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]
    payloads = (
        adapter.score_messages
        if operation == "score"
        else adapter.complete_inputs
        if operation == "complete"
        else adapter.classification_messages
    )
    assert payloads == [expected]
    assert adapter.score_calls + adapter.classifications + adapter.completions == 1


@pytest.mark.parametrize("direction", ["start", "end"])
@pytest.mark.parametrize("orphan", [False, True])
async def test_trim_of_context_only_keeps_input_untouched_and_requires_normal_links(
    direction, orphan
):
    from quarry.integrations.input_trim import trim_messages
    from quarry.types.messages import Messages

    raw = [
        {
            "role": "assistant",
            "content": "plan",
            "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c", "content": "ok"},
    ]
    source = Messages.from_openai(raw)
    messages = source[-1:] if orphan else source
    object.__setattr__(messages, "_chunk", True)
    before = messages.to_openai_messages()

    async def count(selected):
        return len(selected)

    result = await trim_messages(
        messages, 100, context=messages, direction=direction, cut_keep="start", count=count
    )
    assert result.messages.to_openai_messages() == before
    assert messages._chunk is True
    assert result.messages._chunk is False
    if orphan:
        with pytest.raises(ValueError, match="originating assistant tool call"):
            result.messages.assert_self_contained()
    else:
        result.messages.assert_self_contained()
