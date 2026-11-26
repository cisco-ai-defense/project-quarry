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

"""Indexed range provenance and constructed-root rewrite regressions."""

from __future__ import annotations

import asyncio
from dataclasses import fields

import pytest

import quarry.integrations.message_ranges as ranges
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.safety.timer import QuarryTimeoutError
from quarry.types.exceptions import EvaluationError
from quarry.types.messages import Messages
from quarry.types.results import Span


class CharacterCounter:
    def __init__(self):
        self.classifications = 0
        self.scores = 0

    async def tokenize(self, config, messages, tools, *, timeout=None):
        def text(message):
            value = message.get("content")
            return (
                value
                if isinstance(value, str)
                else "".join(part.get("text", "") for part in value or [])
            )

        return {config.tokens.count_field: sum(len(text(message)) for message in messages)}

    async def classify_messages(self, config, messages, tools, *, timeout=None):
        self.classifications += 1
        raise AssertionError("splitting must not dispatch classification")

    async def score(self, config, messages, *, timeout=None):
        self.scores += 1
        raise AssertionError("failed splitting must not dispatch scoring")


def backend(tmp_path):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "counter": {
                    "adapter": "counter",
                    "model": "synthetic",
                    "classify": {"class_index": 0},
                    "tokens": {"adapter": "counter", "extraction": {"count": "count"}},
                }
            },
        },
        tmp_path,
        require_rules=False,
    )
    adapter = CharacterCounter()
    return ModelInferenceBackend(config, adapters={"counter": adapter}), adapter


def multipart(size):
    return Messages.from_openai(
        [{"role": "user", "content": [{"type": "text", "text": "x"} for _ in range(size)]}]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [32, 128, 512])
async def test_many_tiny_chunks_visit_only_their_own_source_parts(tmp_path, monkeypatch, size):
    service, adapter = backend(tmp_path)
    source = multipart(size)
    measured = {"projection_calls": 0, "slice_visits": 0}
    original_steps = type(source[0])._provenance_steps
    original_slice = ranges.slice_provenance

    def projection(message):
        measured["projection_calls"] += 1
        yield from original_steps(message)

    def sliced(segments, start, end):
        measured["slice_visits"] += len(segments)
        return original_slice(segments, start, end)

    monkeypatch.setattr(type(source[0]), "_provenance_steps", projection)
    monkeypatch.setattr(ranges, "slice_provenance", sliced)
    metrics = ranges.RangeMetrics()
    chunks = await service._split_messages(
        "counter", source, 1, 0, None, timeout=None, _range_metrics=metrics
    )
    assert len(chunks) == size
    assert "".join(message.content for chunk in chunks for message in chunk) == "x" * size
    assert measured["projection_calls"] == 1
    assert measured["slice_visits"] <= 4 * size
    assert metrics.prepared_segments == size
    assert metrics.work_units == sum(
        getattr(metrics, field.name)
        for field in fields(metrics)
        if field.name not in {"work_units", "checkpoints"}
    )
    assert adapter.classifications == 0
    for index, chunk in enumerate(chunks):
        span = chunk._bind_span(Span(0, 0, 1))
        assert [(part.part_index, part.start, part.end) for part in span.source_ranges] == [
            (index, 0, 1)
        ]


@pytest.mark.asyncio
async def test_repeated_occurrences_share_one_index_without_losing_their_ranges(tmp_path):
    service, _ = backend(tmp_path)
    source = multipart(32)
    repeated = Messages.combine((source, source))
    metrics = ranges.RangeMetrics()
    chunks = await service._split_messages(
        "counter", repeated, 1, 0, None, timeout=None, _range_metrics=metrics
    )
    assert len(chunks) == 64 and metrics.prepared_segments == 32
    origins = [chunk._bind_span(Span(0, 0, 1)).source_ranges[0].part_index for chunk in chunks]
    assert origins == list(range(32)) * 2


@pytest.mark.asyncio
async def test_range_work_is_bounded_before_excess_mapping_allocation(tmp_path, monkeypatch):
    service, adapter = backend(tmp_path)
    monkeypatch.setattr(ranges, "MAX_SPLIT_RANGE_WORK", 100)
    metrics = ranges.RangeMetrics()
    with pytest.raises(EvaluationError, match="work limit"):
        await service._split_messages(
            "counter", multipart(128), 1, 0, None, timeout=None, _range_metrics=metrics
        )
    assert metrics.work_units == 101
    assert metrics.prepared_segments == 0
    assert adapter.classifications == 0


@pytest.mark.asyncio
async def test_index_preparation_can_be_cancelled_between_raw_part_batches():
    budget = ranges.RangeBudget()
    task = asyncio.create_task(ranges.MessageRangeIndex.prepare(multipart(4096)[0], budget))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert 0 < budget.metrics.source_parts < 4096


@pytest.mark.asyncio
async def test_index_preparation_checks_the_deadline_during_projection(monkeypatch):
    budget = ranges.RangeBudget()

    def expire():
        if budget.metrics.prepared_segments >= 128:
            raise QuarryTimeoutError("synthetic range deadline")

    monkeypatch.setattr(ranges, "raise_if_evaluation_expired", expire)
    with pytest.raises(QuarryTimeoutError, match="synthetic range deadline"):
        await ranges.MessageRangeIndex.prepare(multipart(4096)[0], budget)
    assert 128 <= budget.metrics.prepared_segments < 512


@pytest.mark.asyncio
async def test_slice_queries_preserve_nested_original_part_offsets():
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
    index = await ranges.MessageRangeIndex.prepare(source[0], ranges.RangeBudget())
    first = index.segment(2, 8)
    nested = await ranges.MessageRangeIndex.prepare(first, ranges.RangeBudget())
    final = nested.segment(2, 5)
    assert final.to_openai_dict() == {"role": "assistant", "content": "tse"}
    assert [(part.part_index, part.start, part.end) for part in final._source_ranges(0, 3)] == [
        (0, 4, 5),
        (1, 0, 2),
    ]
    assert source[0].content == "firstsecond"


@pytest.mark.asyncio
async def test_query_work_and_output_are_charged_before_copying(monkeypatch):
    budget = ranges.RangeBudget()
    source = Messages.from_openai([{"role": "user", "content": "x" * 1000}])
    index = await ranges.MessageRangeIndex.prepare(source[0], budget)
    monkeypatch.setattr(ranges, "MAX_SPLIT_RANGE_WORK", budget.metrics.work_units + 10)
    with pytest.raises(EvaluationError, match="work limit"):
        index.segment(0, 1000)
    assert budget.metrics.selected_segments == 0


@pytest.mark.parametrize("role", ["system", "developer", "user", "assistant"])
def test_constructed_rewrite_anchors_survive_independent_root_grafts(role):
    source = Messages().add_message("user", "same")
    rewritten = source.rewritten([{"role": role, "content": "same"}])
    other = Messages().add_message("system", "next")
    combined = Messages.combine((rewritten, other))
    again = Messages.combine((other, combined[:1]))
    assert [message.role for message in combined] == [role, "system"]
    assert [message.role for message in again] == ["system", role]
    span = again._bind_span(Span(1, 0, 4))
    assert span.source_unmapped and span.source_ranges == ()
    assert again.extract(span) == "same"
    assert source[0].role == "user" and rewritten[0].role == role


def test_a_forged_constructed_rewrite_anchor_still_fails():
    source = Messages().add_message("user", "same")
    rewritten = source.rewritten([{"role": "assistant", "content": "same"}])
    other = Messages().add_message("system", "next")
    object.__setattr__(rewritten[0], "_rewrite_source", other[0])
    with pytest.raises(ValueError, match="does not match its source"):
        Messages.combine((rewritten, other))


@pytest.mark.asyncio
async def test_whole_and_part_only_origins_survive_an_indexed_range():
    from quarry.types.transforms import ContentSourcePart, ContentSourceRange, ProvenanceSegment

    source = Messages.from_openai([{"role": "user", "content": "abcd"}])
    original = ContentSourceRange(0, None, 0, 4, 0, 4)
    object.__setattr__(
        source[0],
        "_provenance",
        (
            ProvenanceSegment(0, 2, (original,), "whole"),
            ProvenanceSegment(2, 4, (), "whole", (ContentSourcePart(0, 7),)),
        ),
    )
    index = await ranges.MessageRangeIndex.prepare(source[0], ranges.RangeBudget())
    selected = source._with_position_overrides({0: index.segment(1, 3)})
    span = selected._bind_span(Span(0, 0, 2))
    assert span.source_ranges == (original,)
    assert span.source_unmapped and span.source_parts == (ContentSourcePart(0, 7),)
    assert selected.extract(span) == "bc"


@pytest.mark.asyncio
async def test_an_unordered_source_map_is_rejected_before_indexing():
    from quarry.types.transforms import ProvenanceSegment

    source = Messages.from_openai([{"role": "user", "content": "abcd"}])
    object.__setattr__(
        source[0],
        "_provenance",
        (
            ProvenanceSegment(2, 4, (), "whole"),
            ProvenanceSegment(0, 2, (), "whole"),
        ),
    )
    with pytest.raises(ValueError, match="ordered message map"):
        await ranges.MessageRangeIndex.prepare(source[0], ranges.RangeBudget())


@pytest.mark.asyncio
async def test_a_prepared_policy_reports_a_range_work_failure_as_incomplete(tmp_path, monkeypatch):
    from quarry import compile_policy, load_compiled_policy

    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "counter": {
                    "adapter": "counter",
                    "model": "synthetic",
                    "score": {
                        "message_builder": "constitution_pair",
                        "boundary_seed": 7,
                        "tokens": {"negative": "0", "positive": "1"},
                    },
                    "tokens": {"adapter": "counter", "extraction": {"count": "count"}},
                }
            },
        },
        tmp_path,
        require_rules=False,
    )
    adapter = CharacterCounter()
    service = ModelInferenceBackend(config, adapters={"counter": adapter})
    source = 'rule "R" condition: any chunk in model("counter").split(messages, 1) where model("counter").score(chunk, "policy") > 0.5'
    policy = compile_policy(source, runtime_config=config, model_inference_backend=service)
    loaded = load_compiled_policy(
        policy.to_bytes(), runtime_config=config, model_inference_backend=service
    )
    monkeypatch.setattr(ranges, "MAX_SPLIT_RANGE_WORK", 100)
    result = await loaded.scan_async({"messages": multipart(128).to_openai_messages()})
    assert not result.is_complete and result.rule_matches == []
    assert adapter.scores == adapter.classifications == 0


@pytest.mark.asyncio
async def test_different_occurrence_variants_get_their_own_range_index(tmp_path):
    service, _ = backend(tmp_path)
    source = multipart(32)
    repeated = Messages.combine((source, source))
    variants = repeated.rewritten(
        [dict(source[0].to_openai_dict(), role=role) for role in ["user", "system"]]
    )
    metrics = ranges.RangeMetrics()
    chunks = await service._split_messages(
        "counter", variants, 1, 0, None, timeout=None, _range_metrics=metrics
    )
    assert metrics.prepared_segments == 64
    assert [message.role for chunk in chunks for message in chunk] == ["user"] * 32 + [
        "system"
    ] * 32
    assert [
        chunk._bind_span(Span(0, 0, 1)).source_ranges[0].part_index for chunk in chunks
    ] == list(range(32)) * 2
    assert source[0].role == "user" and source[0].content == "x" * 32


@pytest.mark.asyncio
async def test_failed_preparation_does_not_poison_the_next_split(tmp_path, monkeypatch):
    service, _ = backend(tmp_path)
    source = multipart(128)
    original_limit = ranges.MAX_SPLIT_RANGE_WORK
    monkeypatch.setattr(ranges, "MAX_SPLIT_RANGE_WORK", 100)
    failed = ranges.RangeMetrics()
    with pytest.raises(EvaluationError, match="work limit"):
        await service._split_messages(
            "counter", source, 1, 0, None, timeout=None, _range_metrics=failed
        )
    monkeypatch.setattr(ranges, "MAX_SPLIT_RANGE_WORK", original_limit)
    fresh = ranges.RangeMetrics()
    chunks = await service._split_messages(
        "counter", source, 1, 0, None, timeout=None, _range_metrics=fresh
    )
    assert len(chunks) == 128 and fresh.prepared_segments == 128
    assert failed.work_units == 101
    assert [chunk[0].content for chunk in chunks] == ["x"] * 128


@pytest.mark.asyncio
@pytest.mark.parametrize("host_override", [False, True])
async def test_range_failure_cancels_and_joins_pending_token_probes(
    tmp_path, monkeypatch, host_override
):
    service, adapter = backend(tmp_path)
    active = set()

    async def count(_alias, wire, _tools, **kwargs):
        size = sum(len(message.get("content", "")) for message in wire)
        if size == 1:
            task = asyncio.current_task()
            active.add(task)
            try:
                await asyncio.Event().wait()
            finally:
                active.discard(task)
        return size if host_override else {_alias.tokens.count_field: size}

    if host_override:
        service._count_messages = count
    else:
        adapter.tokenize = count
    monkeypatch.setattr(ranges, "MAX_SPLIT_RANGE_WORK", 70)
    source = Messages.from_openai([{"role": "user", "content": "x" * 32}])
    try:
        with pytest.raises(EvaluationError, match="work limit"):
            await service.model_split_messages("counter", source, 1)
        assert not active, "range failure returned with owned token probes still running"
    finally:
        pending = list(active)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("host_override", [False, True])
async def test_repeated_split_cancellation_joins_token_probe_cleanup(tmp_path, host_override):
    service, adapter = backend(tmp_path)
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    active = set()

    async def count(_alias, wire, _tools, **kwargs):
        size = sum(len(message.get("content", "")) for message in wire)
        if size == 1:
            task = asyncio.current_task()
            active.add(task)
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                try:
                    await release.wait()
                finally:
                    active.discard(task)
        return size if host_override else {_alias.tokens.count_field: size}

    if host_override:
        service._count_messages = count
    else:
        adapter.tokenize = count
    source = Messages.from_openai([{"role": "user", "content": "x" * 32}])
    task = asyncio.create_task(service.model_split_messages("counter", source, 1))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.wait_for(cleaning.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done(), "repeated cancellation escaped before owned cleanup finished"
        assert active
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not active
    finally:
        release.set()
        pending = [task, *active]
        for child in pending:
            if not child.done():
                child.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
