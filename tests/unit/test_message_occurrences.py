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

"""Source-backed occurrence identity through rewriting, selection and model inputs."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.integrations.adapter_message_transforms import apply_adapter_transform
from quarry.integrations.content_transforms import apply_content_transform
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.types.messages import Messages
from quarry.types.results import Span
from quarry.types.transforms import ContentTransform, TransformOperation
from tests.unit.model_fakes import ModelAdapter, NoWaitLimiter, model_entry
from tests.unit.test_message_rewrites import service


class WireCountingAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()
        self.counted = []

    def size(self, messages, tools):
        return len(json.dumps([messages, tools], ensure_ascii=False))

    async def tokenize(self, config, messages, tools, *, timeout=None):
        self.counted.append((deepcopy(messages), deepcopy(tools)))
        return {config.tokens.count_field: self.size(messages, tools)}


def setup_backend(tmp_path, kind):
    assert kind == "classification"
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"judge": model_entry(tokens=True, classify=0)}},
        tmp_path,
        require_rules=False,
    )
    adapter = WireCountingAdapter()
    return ModelInferenceBackend(
        config, adapters={"fake": adapter}, rate_limiter=NoWaitLimiter()
    ), adapter


def variants():
    source = Messages.from_openai([{"role": "assistant", "content": "same"}])
    repeated = Messages.combine((source, source, source))
    return source, repeated.rewritten(
        [{"role": role, "content": "same"} for role in ("user", "system", "assistant")]
    )


@pytest.mark.parametrize(
    "selection,roles",
    [
        (slice(None), ["user", "system", "assistant"]),
        (slice(None, None, -1), ["assistant", "system", "user"]),
        (slice(1, None), ["system", "assistant"]),
        (slice(None, None, 2), ["user", "assistant"]),
        (slice(1, 1), []),
    ],
)
def test_slices_preserve_occurrences(selection, roles):
    source, rewritten = variants()
    selected = rewritten[selection]
    assert [message.role for message in selected] == roles
    assert all(message.message_index == 0 for message in selected)
    assert selected.shares_root_with(source)
    assert rewritten[-1].role == "assistant"
    with pytest.raises(IndexError):
        rewritten[-4]


def test_role_filters_and_nested_position_selections_keep_exact_variants():
    _, rewritten = variants()
    assert [message.role for message in rewritten.user_messages] == ["user"]
    assert [message.role for message in rewritten.system_messages] == ["system"]
    assert [message.role for message in rewritten.assistant_messages] == ["assistant"]
    selected = rewritten._select_positions((2, 0, 2))[1:]
    assert [message.role for message in selected] == ["user", "assistant"]
    assert rewritten.rewritten(rewritten.to_openai_messages()) is rewritten
    with pytest.raises(TypeError):
        rewritten._position_variants[1] = rewritten[0]


def test_unchanged_and_changed_occurrences_of_one_message_are_independent():
    source = Messages.from_openai([{"role": "assistant", "content": "same"}])
    repeated = Messages.combine((source, source))
    rewritten = repeated.rewritten(
        [{"role": "assistant", "content": "same"}, {"role": "user", "content": "same"}]
    )
    assert rewritten[0] is source[0]
    assert rewritten[1].role == "user"
    assert [message.role for message in Messages.combine((rewritten, source))] == [
        "assistant",
        "user",
        "assistant",
    ]


def test_combination_of_slices_keeps_written_order_and_source_ranges():
    source, rewritten = variants()
    combined = Messages.combine((rewritten[1:], source, rewritten[:1]))
    assert [message.role for message in combined] == ["system", "assistant", "assistant", "user"]
    assert combined.span.to_dict() == {"message_index": 0, "start": 0, "end": 4}
    for index in range(len(combined)):
        span = combined._bind_span(Span(index, 1, 3))
        assert span.message_index == 0
        if combined[index]._provenance is None:
            assert span.source_ranges == ()
            assert span.to_dict() == {"message_index": 0, "start": 1, "end": 3}
        else:
            assert span.source_ranges[0].to_dict() == {"message_index": 0, "start": 1, "end": 3}
        assert combined.extract(span) == "am"
    assert [message.role for message in rewritten] == ["user", "system", "assistant"]


def test_mixed_root_combinations_and_appending_preserve_variants():
    source, rewritten = variants()
    synthetic = source[:0].add_message("user", "new")
    combined = Messages.combine((rewritten, synthetic))
    assert [message.role for message in combined] == ["user", "system", "assistant", "user"]
    assert combined[-1].content == "new"
    appended = rewritten.add_message("system", "tail")
    assert [message.role for message in appended] == ["user", "system", "assistant", "system"]
    assert appended[-1].content == "tail"
    assert appended[1].message_index == 0


@pytest.mark.asyncio
async def test_transform_chains_keep_different_part_layouts_and_character_origins():
    source = Messages.from_openai([{"role": "assistant", "content": " a b "}])
    repeated = Messages.combine((source, source))
    rewritten = repeated.rewritten(
        [
            {"role": "user", "content": " a b "},
            {
                "role": "system",
                "content": [{"type": "text", "text": " a "}, {"type": "text", "text": "b "}],
            },
        ]
    )
    transformed = await apply_content_transform(
        rewritten, ContentTransform((TransformOperation("strip"),))
    )
    assert [message.content for message in transformed] == ["a b", "ab"]
    assert [message.role for message in transformed] == ["user", "system"]
    assert transformed.extract(transformed._bind_span(Span(0, 1, 2))) == " "
    assert transformed.extract(transformed._bind_span(Span(1, 1, 2))) == "b"
    assert transformed._bind_span(Span(1, 1, 2)).source_ranges[0].start == 3
    assert transformed.span.to_dict() == {"message_index": 0, "start": 1, "end": 4}
    with pytest.raises(ValueError, match="ambiguous"):
        transformed.source_ranges(0, 1, 2)
    assert [message.content for message in transformed[::-1]] == ["ab", "a b"]


@pytest.mark.asyncio
async def test_part_adapter_processes_media_only_on_the_occurrence_that_contains_it(tmp_path):
    class Parts:
        local = True
        transform_parts = ("image_url",)

        def transform(self, config, part, *, timeout=None):
            return "LABEL"

    source = Messages.from_openai([{"role": "assistant", "content": "same"}])
    repeated = Messages.combine((source, source))
    rewritten = repeated.rewritten(
        [
            {"role": "user", "content": "same"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "same"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ],
            },
        ]
    )
    _, backend, _ = service(tmp_path, Parts())
    transformed = await apply_adapter_transform(rewritten, backend, "rewrite")
    assert [message.content for message in transformed] == ["same", "sameLABEL"]
    span = transformed._bind_span(Span(1, 4, 9))
    assert span.source_unmapped and span.source_parts[0].part_index is None
    assert transformed.extract(span) == "LABEL"
    assert source[0].content == "same"


@pytest.mark.asyncio
async def test_prepared_policy_rewrites_each_repeated_occurrence(tmp_path):
    class Rewrite:
        local = True

        def transform_messages(self, config, messages):
            return [
                dict(message, role=role)
                for message, role in zip(messages, ("user", "system"), strict=True)
            ]

    config, backend, _ = service(tmp_path, Rewrite())
    policy = compile_policy(
        'rule "R" condition: (messages + messages).transform("rewrite").system_messages[0].content == "same"',
        runtime_config=config,
        model_inference_backend=backend,
    )
    loaded = load_compiled_policy(
        policy.to_bytes(), runtime_config=config, model_inference_backend=backend
    )
    result = await loaded.scan_async({"messages": [{"role": "assistant", "content": "same"}]})
    assert result.is_complete and result.rule_matches == ["R"]


def test_tool_identity_duplicates_are_still_invalid_with_distinct_rewritten_results():
    source = Messages.from_openai(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ]
    )
    with pytest.raises(ValueError, match="duplicate tool results"):
        source.rewritten(
            [
                source[0].to_openai_dict(),
                {"role": "tool", "tool_call_id": "c", "content": "one"},
                {"role": "tool", "tool_call_id": "c", "content": "two"},
            ]
        )
    assert source[0].tool_calls[0].response is None


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_split_keeps_repeated_occurrences_in_their_written_order(tmp_path, reverse):

    source = Messages.from_openai([{"role": "assistant", "content": "word " * 40}])
    repeated = Messages.combine((source, source, source))
    rewritten = repeated.rewritten(
        [{"role": role, "content": source[0].content} for role in ("user", "system", "assistant")]
    )
    if reverse:
        rewritten = rewritten[::-1]
    backend, adapter = setup_backend(tmp_path, "classification")
    limit = max(adapter.size([message.to_openai_dict()], []) for message in rewritten)
    chunks = await backend.model_split_messages("judge", rewritten, limit)
    assert len(chunks) == 3
    assert [message.role for chunk in chunks for message in chunk] == [
        message.role for message in rewritten
    ]
    assert [message.content for chunk in chunks for message in chunk] == [source[0].content] * 3
    for chunk in chunks:
        assert adapter.size(chunk.to_openai_messages(), []) <= limit
        assert chunk[0].message_index == 0


@pytest.mark.asyncio
async def test_split_context_is_one_occurrence_rather_than_every_use_of_its_source(tmp_path):
    from quarry.integrations.model_expression import _chunk_spans

    source, rewritten = variants()
    backend, adapter = setup_backend(tmp_path, "classification")
    limit = max(
        adapter.size(rewritten._select_positions((0, index)).to_openai_messages(), [])
        for index in (1, 2)
    )
    chunks = await backend.model_split_messages("judge", rewritten, limit, context=rewritten[:1])
    assert [[message.role for message in chunk] for chunk in chunks] == [
        ["user", "system"],
        ["user", "assistant"],
    ]
    for chunk in chunks:
        context_span, own_span = _chunk_spans(chunk, rewritten[:1], source)
        assert context_span["message_index"] == own_span["message_index"] == 0
    with pytest.raises(ValueError, match="context"):
        await backend.model_split_messages("judge", rewritten, limit, context=rewritten[1:2])


@pytest.mark.asyncio
async def test_split_ranges_measure_and_preserve_each_transformed_variant(tmp_path):

    text = " a b " * 40
    source = Messages.from_openai([{"role": "assistant", "content": text}])
    repeated = Messages.combine((source, source))
    rewritten = repeated.rewritten(
        [
            {"role": "user", "content": text},
            {"role": "system", "content": [{"type": "text", "text": " a b "} for _ in range(40)]},
        ]
    )
    transformed = await apply_content_transform(
        rewritten, ContentTransform((TransformOperation("strip"),))
    )
    backend, adapter = setup_backend(tmp_path, "classification")
    chunks = await backend.model_split_messages("judge", transformed, 180)
    assert len(chunks) > 2
    for role in ("user", "system"):
        assert "".join(
            message.content for chunk in chunks for message in chunk if message.role == role
        ) == next(message.content for message in transformed if message.role == role)
    for chunk in chunks:
        assert adapter.size(chunk.to_openai_messages(), []) <= 180
        for position, message in enumerate(chunk):
            span = chunk._bind_span(Span(position, 0, len(message.content)))
            assert all(source.message_index == 0 for source in span.source_ranges)


@pytest.mark.asyncio
async def test_split_can_use_a_distinct_context_variant_and_a_context_outside_its_body(tmp_path):

    source, rewritten = variants()
    context = source.rewritten([{"role": "developer", "content": "same"}])
    backend, _ = setup_backend(tmp_path, "classification")
    chunks = await backend.model_split_messages("judge", rewritten, 200, context=context)
    assert [[message.role for message in chunk] for chunk in chunks] == [
        ["developer", "system", "assistant"]
    ]
    source = Messages.from_openai(
        [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "question"},
        ]
    )
    chunks = await backend.model_split_messages("judge", source[1:], 200, context=source[:1])
    assert chunks[0].to_openai_messages() == source.to_openai_messages()


def test_text_only_ranges_keep_part_origins_and_do_not_add_absent_assistant_fields():
    source = Messages.from_openai(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ],
            }
        ]
    )
    selected = source._select_ranges((0,), {0: (2, 8)})
    assert selected.to_openai_messages() == [{"role": "assistant", "content": "rstsec"}]
    nested = selected._select_ranges((0,), {0: (2, 5)})
    assert nested.to_openai_messages() == [{"role": "assistant", "content": "tse"}]
    span = nested._bind_span(Span(0, 0, 3))
    assert [(part.part_index, part.start, part.end) for part in span.source_ranges] == [
        (0, 4, 5),
        (1, 0, 2),
    ]


@pytest.mark.asyncio
async def test_versioned_part_cache_reuses_work_without_collapsing_occurrence_views(tmp_path):
    from quarry.integrations.model_usage import model_call_budget
    from quarry.types.adapter_transforms import TransformCacheIdentity

    class Parts:
        local = True
        transform_parts = ("text",)
        transform_cache = TransformCacheIdentity("owned.occurrences", "1")

        def __init__(self):
            self.calls = []

        def transform(self, config, part, *, timeout=None):
            self.calls.append(part)
            return part["text"].upper()

    source, rewritten = variants()
    adapter = Parts()
    _, backend, _ = service(tmp_path, adapter)
    with model_call_budget(1) as budget:
        transformed = await apply_adapter_transform(
            rewritten, backend, "rewrite", _message_value=True
        )
        assert budget.calls_by_alias == {}
    assert [m.role for m in transformed] == ["user", "system", "assistant"]
    assert [m.content for m in transformed] == ["SAME"] * 3
    assert len(adapter.calls) == 1
    for position in range(3):
        span = transformed._bind_span(Span(position, 0, 4))
        assert transformed.extract(span) == "SAME"
        assert span.source_ranges[0].to_dict() == {"message_index": 0, "start": 0, "end": 4}
    assert source[0].role == "assistant" and source[0].content == "same"


@pytest.mark.asyncio
async def test_shared_cache_hit_rebinds_each_occurrence_to_the_current_source(tmp_path):
    from quarry.types.adapter_transforms import TransformCacheIdentity

    class Parts:
        local = True
        transform_parts = ("text",)
        transform_cache = TransformCacheIdentity("owned.occurrence-origins", "1")

        def __init__(self):
            self.calls = 0

        def transform(self, config, part, *, timeout=None):
            self.calls += 1
            return part["text"].upper()

    for prefix in ([], [{"role": "system", "content": "prefix"}]):
        adapter = Parts()
        _, backend, _ = service(tmp_path, adapter)
        source = Messages.from_openai([*prefix, {"role": "assistant", "content": "same"}])
        repeated = Messages.combine((source[-1:], source[-1:]))
        rewritten = repeated.rewritten(
            [{"role": "user", "content": "same"}, {"role": "system", "content": "same"}]
        )
        transformed = await apply_adapter_transform(
            rewritten, backend, "rewrite", _message_value=True
        )
        assert adapter.calls == (0 if prefix else 1)
        for position in range(2):
            span = transformed._bind_span(Span(position, 0, 4))
            assert transformed.extract(span) == "SAME"
            assert span.source_ranges[0].message_index == len(prefix)


@pytest.mark.asyncio
async def test_unversioned_callbacks_process_distinct_occurrence_values_separately(tmp_path):
    class Counter:
        local = True
        transform_parts = ("text",)

        def __init__(self):
            self.calls = 0

        def transform(self, config, part, *, timeout=None):
            self.calls += 1
            return str(self.calls)

    _, rewritten = variants()
    adapter = Counter()
    _, backend, _ = service(tmp_path, adapter)
    transformed = await apply_adapter_transform(rewritten, backend, "rewrite", _message_value=True)
    assert [m.content for m in transformed] == ["1", "2", "3"]
    assert [m.role for m in transformed] == ["user", "system", "assistant"]
    assert adapter.calls == 3


@pytest.mark.asyncio
async def test_memo_sizing_charges_state_retained_only_by_a_later_occurrence():
    from quarry.cache.message_values import _retained_bytes

    source = Messages.from_openai([{"role": "assistant", "content": "same"}])
    repeated = Messages.combine((source, source))
    changed = repeated.rewritten(
        [
            source[0].to_openai_dict(),
            {"role": "assistant", "content": "same", "refusal": "x" * 20_000},
        ]
    )
    before = await _retained_bytes(repeated, 1_000_000)
    after = await _retained_bytes(changed, 1_000_000)
    assert after - before >= 20_000
    assert await _retained_bytes(changed, before + 10_000) > before + 10_000


@pytest.mark.asyncio
async def test_equal_final_text_keeps_deleted_provenance_on_every_occurrence():
    _, rewritten = variants()
    transformed = await apply_content_transform(
        rewritten,
        ContentTransform(
            (TransformOperation("replace", "same", ""), TransformOperation("prefix", "same"))
        ),
    )
    assert [m.content for m in transformed] == ["same"] * 3
    for position in range(3):
        with pytest.raises(ValueError, match="inserted text"):
            transformed._bind_span(Span(position, 0, 4))


def test_repeated_constructed_rewrites_reanchor_every_variant_on_a_new_root():
    literal = Messages().add_message("user", "same")
    repeated = Messages.combine((literal, literal))
    rewritten = repeated.rewritten(
        [{"role": "assistant", "content": "same"}, {"role": "developer", "content": "same"}]
    )
    combined = Messages.combine((rewritten, Messages().add_message("user", "other")))
    assert combined.to_openai_messages() == [
        {"role": "assistant", "content": "same"},
        {"role": "developer", "content": "same"},
        {"role": "user", "content": "other"},
    ]
    assert literal[0].role == "user"


@pytest.mark.asyncio
@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("text_join", [False, True])
async def test_mapped_content_constructors_keep_the_actual_occurrence_value(
    tmp_path, conditional, text_join
):
    class Layouts:
        local = True

        def transform_messages(self, config, messages):
            return [
                {"role": "user", "content": " a b "},
                {
                    "role": "system",
                    "content": [{"type": "text", "text": " a "}, {"type": "text", "text": "b "}],
                },
            ]

    judge = ModelAdapter()
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"judge": {**model_entry(classify=0), "adapter": "judge"}},
            "transforms": {
                "rewrite": {"adapter": "layouts"},
                "strip": {"operations": [{"op": "strip"}]},
            },
        },
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(config, adapters={"layouts": Layouts(), "judge": judge})
    owner = "(item if true else item)[0]" if conditional else "item"
    source = '(messages + messages).transform("rewrite").transform("strip")'
    if text_join:
        model_input = (
            'message("user", join([' + owner + ".content for item in " + source + '], "|"))'
        )
        expected = [{"role": "user", "content": "a b|ab"}]
    else:
        model_input = '[message("user", ' + owner + ".content) for item in " + source + "]"
        expected = [{"role": "user", "content": "a b"}, {"role": "user", "content": "ab"}]
    source_text = 'rule "R" condition: model("judge").classify(' + model_input + ") > .5"
    if text_join:
        from quarry.types.exceptions import RuleValidationError

        # String-valued joins of Messages are outside the current typed map contract.
        # Keep that compile-time guard; message-valued maps below retain occurrence values.
        with pytest.raises(RuleValidationError, match="structured items.*not Messages"):
            compile_policy(source_text, runtime_config=config, model_inference_backend=backend)
        assert judge.classification_messages == []
        return
    policy = compile_policy(source_text, runtime_config=config, model_inference_backend=backend)
    loaded = load_compiled_policy(
        policy.to_bytes(), runtime_config=config, model_inference_backend=backend
    )
    result = await loaded.scan_async({"messages": [{"role": "assistant", "content": " a b "}]})
    assert result.is_complete and result.rule_matches == ["R"]
    assert judge.classification_messages == [expected]
