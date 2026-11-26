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

"""Local whole-message transform contracts and text-preserving message variants."""

from __future__ import annotations

import asyncio

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.integrations.adapter_message_transforms import apply_adapter_transform
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.integrations.model_usage import model_call_budget
from quarry.types.exceptions import ConfigurationError, ModelError
from quarry.types.messages import AssistantMessage, Messages, UserMessage
from quarry.types.results import Span
from tests.unit.model_fakes import model_entry


class RewriteAdapter:
    local = True

    def __init__(self):
        self.calls = []

    def transform_messages(self, config, wire):
        self.calls.append(wire)
        return [
            dict(message, role="user") if not message.get("tool_calls") else message
            for message in wire
        ]


def service(tmp_path, adapter=None, definition=None):
    adapter = RewriteAdapter() if adapter is None else adapter
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"extractor": model_entry()},
            "transforms": {"rewrite": definition or {"adapter": "fake"}},
        },
        tmp_path,
        require_rules=False,
    )
    return config, ModelInferenceBackend(config, adapters={"fake": adapter}), adapter


def test_rewritten_identity_preserves_the_selection_object():
    source = Messages.from_openai([{"role": "assistant", "content": "same"}])
    assert source.rewritten(source.to_openai_messages()) is source


def test_role_rewrites_keep_indexes_text_ranges_and_role_views():
    source = Messages.from_openai(
        [{"role": "assistant", "content": "first"}, {"role": "user", "content": "second"}]
    )
    rewritten = source.rewritten(
        [{"role": "user", "content": "first"}, {"role": "assistant", "content": "second"}]
    )
    assert rewritten[0].message_index == 0 and rewritten[1].message_index == 1
    assert rewritten.user_messages.to_openai_messages() == [{"role": "user", "content": "first"}]
    assert rewritten.assistant_messages.to_openai_messages() == [
        {"role": "assistant", "content": "second"}
    ]
    assert [r.to_dict() for r in rewritten.source_ranges(0, 1, 4)] == [
        {"message_index": 0, "start": 1, "end": 4}
    ]
    combined = Messages.combine((rewritten[:1], source[1:]))
    assert [m.role for m in combined] == ["user", "user"]
    assert source[0].role == "assistant" and source[1].role == "user"


def test_a_user_message_can_be_read_as_a_tool_result_without_mutating_the_call_graph():
    source = Messages.from_openai(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "inspect", "arguments": "{}"},
                    }
                ],
            },
            {"role": "user", "content": "result"},
        ]
    )
    rewritten = source.rewritten(
        [source[0].to_openai_dict(), {"role": "tool", "tool_call_id": "c", "content": "result"}]
    )
    rewritten.assert_self_contained()
    assert rewritten[1].call is rewritten[0].tool_calls[0]
    assert rewritten[0].tool_calls[0].response is rewritten[1]
    assert rewritten.get_tool_responses("inspect") == [rewritten[1]]
    assert source[0].tool_calls[0].response is None
    assert source[1].role == "user"
    combined = Messages.combine((source[:1], rewritten[1:]))
    combined.assert_self_contained()
    assert combined[1].call is combined[0].tool_calls[0]
    span = combined._bind_span(Span(1, 0, 6))
    assert span.message_index == 1
    assert span.source_ranges[0].message_index == 1


def test_rewriting_a_tool_result_as_text_removes_hidden_response_links():
    source = Messages.from_openai(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "inspect", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c", "content": "result"},
        ]
    )
    rewritten = source.rewritten(
        [source[0].to_openai_dict(), {"role": "user", "content": "result"}]
    )
    assert rewritten[0].tool_calls[0].response is None
    assert rewritten.tool_responses.to_openai_messages() == []
    assert source[0].tool_calls[0].response is source[1]
    assert Messages.combine((rewritten,))[1].role == "user"


@pytest.mark.parametrize(
    "replacement",
    [
        [],
        [{"role": "user", "content": "changed"}],
        [{"role": "alien", "content": "same"}],
        [{"role": "user", "content": 4}],
        [
            {
                "role": "assistant",
                "content": "same",
                "tool_calls": [
                    {
                        "id": "new",
                        "type": "function",
                        "function": {"name": "inspect", "arguments": "{}"},
                    }
                ],
            }
        ],
    ],
)
def test_invalid_and_call_creating_rewrites_are_rejected(replacement):
    source = Messages.from_openai([{"role": "user", "content": "same"}])
    with pytest.raises((ValueError, ModelError)):
        source.rewritten(replacement)
    assert source[0].content == "same"


def test_call_holders_cannot_be_rewritten_even_when_text_is_unchanged():
    source = Messages.from_openai(
        [
            {
                "role": "assistant",
                "content": "same",
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "inspect", "arguments": "{}"},
                    }
                ],
            }
        ]
    )
    with pytest.raises(ValueError, match="no tool calls"):
        source.rewritten([{"role": "user", "content": "same"}])
    assert source.rewritten(source.to_openai_messages()) is source


def test_text_part_restructuring_keeps_original_character_provenance():
    source = Messages.from_openai(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "ab"}, {"type": "text", "text": "cd"}],
            }
        ]
    )
    rewritten = source.rewritten([{"role": "assistant", "content": "abcd"}])
    assert [r.to_dict() for r in rewritten.source_ranges(0, 1, 3)] == [
        {"message_index": 0, "part_index": 0, "start": 1, "end": 2},
        {"message_index": 0, "part_index": 1, "start": 0, "end": 1},
    ]


@pytest.mark.asyncio
async def test_whole_message_work_is_local_uncached_and_usable_by_prepared_rules(tmp_path):
    config, backend, adapter = service(tmp_path)
    assert backend.transform_parts("rewrite") == ()
    policy = compile_policy(
        'rule "R" condition: messages.transform("rewrite").user_messages[0].content contains "same"',
        runtime_config=config,
        model_inference_backend=backend,
    )
    loaded = load_compiled_policy(
        policy.to_bytes(), runtime_config=config, model_inference_backend=backend
    )
    data = {"messages": [{"role": "assistant", "content": "same"}]}
    with model_call_budget(1) as budget:
        first = await loaded.scan_async(data)
        second = await loaded.scan_async(data)
        assert first.rule_matches == second.rule_matches == ["R"]
        assert budget.calls_by_alias == {}
    assert len(adapter.calls) == 2


@pytest.mark.parametrize("flag", [None, 0, "true", False])
def test_whole_message_adapters_require_explicit_local_true(tmp_path, flag):
    adapter = RewriteAdapter()
    adapter.local = flag
    _, backend, _ = service(tmp_path, adapter)
    with pytest.raises(ConfigurationError):
        backend.require_transforms({"rewrite"})


@pytest.mark.parametrize(
    "definition", [{"model": "extractor"}, {"adapter": "fake", "parts": ["text"]}]
)
def test_whole_message_adapters_reject_model_aliases_and_part_selection(tmp_path, definition):
    _, backend, _ = service(tmp_path, definition=definition)
    with pytest.raises(ConfigurationError):
        backend.require_transforms({"rewrite"})


@pytest.mark.asyncio
async def test_adapter_payload_mutation_is_detached_and_bad_output_is_a_fault(tmp_path):
    class Mutating(RewriteAdapter):
        def transform_messages(self, config, wire):
            self.calls.append(wire)
            wire[0]["content"] = "changed"
            return wire

    _, backend, adapter = service(tmp_path, Mutating())
    source = Messages.from_openai([{"role": "user", "content": "same"}])
    for _ in range(2):
        with pytest.raises(ModelError) as fault:
            await backend.transform_messages("rewrite", source)
        assert fault.value.details["code"] == "invalid_output"
    assert len(adapter.calls) == 2
    assert source[0].content == "same"


@pytest.mark.asyncio
async def test_protected_messages_are_not_serialized_for_the_adapter(tmp_path):
    class Protected(UserMessage):
        def _readable_for_rewrite(self):
            return False

        def to_openai_dict(self):
            raise AssertionError("protected message was exposed")

    blocked = Protected(role="user", content="blocked")
    object.__setattr__(blocked, "_message_index", 1)
    allowed = AssistantMessage(role="assistant", content="same")
    object.__setattr__(allowed, "_message_index", 0)
    source = Messages._from_items((allowed, blocked), None, 100, source_root=None)
    _, backend, adapter = service(tmp_path)
    rewritten = await backend.transform_messages("rewrite", source)
    assert adapter.calls == [[{"role": "assistant", "content": "same"}]]
    assert rewritten[0].role == "user"
    assert rewritten[1] is blocked


def test_json_cycles_and_excessive_graphs_are_rejected_before_message_construction():
    source = Messages.from_openai([{"role": "user", "content": "same"}])
    cycle = []
    cycle.append(cycle)
    for content in (cycle, ["x"] * 100_001):
        with pytest.raises(ValueError, match="JSON"):
            source.rewritten([{"role": "user", "content": content}])


@pytest.mark.asyncio
async def test_async_message_adapter_obeys_the_active_evaluation_deadline(tmp_path):
    from quarry.safety.timer import QuarryTimeoutError, evaluation_deadline

    cancelled = asyncio.Event()

    class Waiting(RewriteAdapter):
        async def transform_messages(self, config, wire):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    _, backend, _ = service(tmp_path, Waiting())
    source = Messages.from_openai([{"role": "user", "content": "same"}])
    with pytest.raises(QuarryTimeoutError), evaluation_deadline(0.02):
        await backend.transform_messages("rewrite", source)
    await asyncio.wait_for(cancelled.wait(), 1)


@pytest.mark.asyncio
async def test_a_protected_tool_call_keeps_its_whole_message_out_of_the_adapter(tmp_path):
    from quarry.types.messages import ToolCall

    class ProtectedCall(ToolCall):
        def _readable_for_rewrite(self):
            return False

    call = ProtectedCall.model_validate(
        {"id": "c", "type": "function", "function": {"name": "inspect", "arguments": "{}"}},
        strict=True,
    )
    blocked = AssistantMessage(role="assistant", content="blocked", tool_calls=(call,))
    allowed = AssistantMessage(role="assistant", content="same")
    object.__setattr__(blocked, "_message_index", 0)
    object.__setattr__(allowed, "_message_index", 1)
    source = Messages._from_items((blocked, allowed), None, 100, source_root=None)
    _, backend, adapter = service(tmp_path)
    transformed = await backend.transform_messages("rewrite", source)
    assert adapter.calls == [[{"role": "assistant", "content": "same"}]]
    assert transformed[0] is blocked
    assert transformed[1].role == "user"


@pytest.mark.parametrize("model", [False, True])
def test_part_adapters_cannot_disagree_with_configured_locality(tmp_path, model):
    class Parts:
        local = model
        transform_parts = frozenset({"text"})

        def transform(self, config, part, *, timeout=None):
            raise AssertionError("misconfigured adapter was called")

    definition = {"model": "extractor"} if model else {"adapter": "fake"}
    _, backend, _ = service(tmp_path, Parts(), definition)
    with pytest.raises(ConfigurationError, match="locality"):
        backend.require_transforms({"rewrite"})


@pytest.mark.asyncio
async def test_part_adapters_accept_frozen_part_capabilities(tmp_path):
    class Parts:
        local = True
        transform_parts = frozenset({"text"})

        def transform(self, config, part, *, timeout=None):
            return part["text"]

    _, backend, _ = service(tmp_path, Parts())
    assert backend.transform_parts("rewrite") == ("text",)
    assert (
        await backend.transform_unit("rewrite", {"type": "text", "text": "same"})
    ).text == "same"


def test_combination_compares_part_only_origins_as_well_as_wire_data():
    from quarry.types.transforms import ContentSourcePart, ProvenanceSegment

    source = Messages.from_openai(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ],
            }
        ]
    )
    variants = []
    for part in (0, 1):
        message = source[0]._with_transformed_content(
            {0: "LABEL", 1: ""},
            (ProvenanceSegment(0, 5, (), "whole", (ContentSourcePart(0, part),)),),
            replace_media=True,
        )
        variants.append(source._with_overrides({0: message}))
    combined = Messages.combine(variants)
    assert [
        combined._bind_span(Span(index, 0, 5)).source_parts[0].part_index for index in range(2)
    ] == [0, 1]


@pytest.mark.asyncio
async def test_new_media_in_a_restructured_message_uses_message_context_not_an_invented_part(
    tmp_path,
):
    from quarry.types.adapter_transforms import TransformResult, TransformRun

    source = Messages.from_openai([{"role": "user", "content": "same"}])
    rewritten = source.rewritten(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "same"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ],
            }
        ]
    )

    class Parts:
        local = True
        transform_parts = ("image_url",)

        def transform(self, config, part, *, timeout=None):
            return TransformResult("LABEL", (TransformRun(0, 5),))

    _, backend, _ = service(tmp_path, Parts())
    transformed = await apply_adapter_transform(rewritten, backend, "rewrite")
    span = transformed._bind_span(Span(0, 4, 9))
    assert span.source_ranges == () and span.source_unmapped
    assert span.to_dict()["source_parts"] == [{"message_index": 0}]


def test_repeated_source_rewrites_preserve_each_occurrence():
    source = Messages.from_openai([{"role": "assistant", "content": "same"}])
    repeated = Messages.combine((source, source))
    rewritten = repeated.rewritten(
        [{"role": "user", "content": "same"}, {"role": "system", "content": "same"}]
    )
    assert [message.role for message in rewritten] == ["user", "system"]
    assert [message.message_index for message in rewritten] == [0, 0]
    assert source[0].role == "assistant"


def test_changing_a_tool_result_association_does_not_leave_a_hidden_old_link():
    calls = [
        {"id": call_id, "type": "function", "function": {"name": call_id, "arguments": "{}"}}
        for call_id in ("first", "second")
    ]
    source = Messages.from_openai(
        [
            {"role": "assistant", "tool_calls": calls},
            {"role": "tool", "tool_call_id": "first", "content": "result"},
        ]
    )
    rewritten = source.rewritten(
        [
            source[0].to_openai_dict(),
            {"role": "tool", "tool_call_id": "second", "content": "result"},
        ]
    )
    first, second = rewritten[0].tool_calls
    assert first.response is None
    assert second.response is rewritten[1]
    assert rewritten[1].call is second
    assert source[0].tool_calls[0].response is source[1]
    assert source[0].tool_calls[1].response is None
    combined = Messages.combine((rewritten[:1], rewritten[1:]))
    assert combined[0].tool_calls[0].response is None
    assert combined[0].tool_calls[1].response is combined[1]
