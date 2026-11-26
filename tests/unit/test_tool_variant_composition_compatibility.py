# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Reject conflicting tool units while retaining ordinary copied occurrences."""

import pytest

from quarry.integrations.content_transforms import apply_content_transform
from quarry.types.messages import Messages, _CombineMetrics
from quarry.types.transforms import ContentTransform, TransformOperation


def _conversation() -> Messages:
    return Messages.from_openai(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "synthetic-call",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "synthetic-call", "content": " SAMPLE "},
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("async_combine", [False, True])
async def test_conflicting_tool_results_stop_at_the_first_repeated_unit(async_combine):
    source = _conversation()
    first = await apply_content_transform(
        source.tool_responses, ContentTransform((TransformOperation("strip"),))
    )
    different = await apply_content_transform(
        source.tool_responses, ContentTransform((TransformOperation("casefold"),))
    )
    parts = tuple((first, different)[index % 2] for index in range(128))
    metrics = _CombineMetrics()

    with pytest.raises(ValueError, match="different content variants"):
        if async_combine:
            await Messages._combine_async(parts, _metrics=metrics)
        else:
            Messages.combine(parts, _metrics=metrics)

    assert metrics.selection_visits == 2
    assert source[1].content == " SAMPLE "
    assert source[0].tool_calls[0].response is source[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("async_combine", [False, True])
@pytest.mark.parametrize("copy_first", [False, True])
async def test_a_tool_result_and_its_ordinary_copy_keep_distinct_occurrences(
    async_combine, copy_first
):
    source = _conversation()
    ordinary = source.tool_responses.rewritten([{"role": "user", "content": " SAMPLE "}])
    tail = (ordinary, source.tool_responses) if copy_first else (source.tool_responses, ordinary)
    parts = (source.assistant_messages, *tail)

    if async_combine:
        combined = await Messages._combine_async(parts)
    else:
        combined = Messages.combine(parts)

    assert [message.role for message in combined] == [
        "assistant",
        *(["user", "tool"] if copy_first else ["tool", "user"]),
    ]
    assert [message.message_index for message in combined] == [0, 1, 1]
    tool_position = 2 if copy_first else 1
    assert combined[0].tool_calls[0].response is combined[tool_position]
    combined.assert_self_contained()
    assert source[0].tool_calls[0].response is source[1]
    assert source[1].role == "tool"
