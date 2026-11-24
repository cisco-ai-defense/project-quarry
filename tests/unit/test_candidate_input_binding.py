# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Candidate geometry stays inside exact native fields and original content parts."""

import gc
import weakref
from array import array

import pytest

from quarry.integrations import candidate_input
from quarry.integrations.candidate_input import bind_candidate_input
from quarry.types.candidates import (
    CandidateBatchLimits,
    CandidateLimitError,
    CandidateRange,
    CandidateReplyError,
)
from quarry.types.messages import Message, Messages
from quarry.types.origins import LINEAR, WHOLE, Derived, Place, joined
from quarry.types.results import InputText


async def bind(data, **limits):
    return await bind_candidate_input(Messages.from_openai(data), CandidateBatchLimits(**limits))


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "assistant", "system", "developer"])
async def test_native_unicode_ranges_bind_exactly_and_keep_input_out_of_repr(role):
    binding = await bind([{"role": role, "content": "a雪🙂b"}])
    assert [(p.kind, p.text) for p in binding.input.parts] == [(role, "a雪🙂b")]
    span = binding.bind(CandidateRange(0, 1, 3, 0.9))
    assert (span.message_index, span.start, span.end, span.precision) == (0, 1, 3, "exact")
    assert span.score.value == 0.9
    assert span.source_ranges[0].to_dict() == {"message_index": 0, "start": 1, "end": 3}
    assert "雪" not in repr(binding) + repr(binding.input)


@pytest.mark.asyncio
async def test_adjacent_content_parts_never_form_one_candidate_text():
    binding = await bind(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "ab"},
                    {"type": "text", "text": "cd"},
                ],
            }
        ]
    )
    assert [p.text for p in binding.input.parts] == ["ab", "cd"]
    span = binding.bind(CandidateRange(1, 0, 2, 0.0))
    assert (span.start, span.end) == (2, 4)
    assert span.source_ranges[0].to_dict() == {
        "message_index": 0,
        "part_index": 1,
        "start": 0,
        "end": 2,
    }
    with pytest.raises(CandidateReplyError, match="crosses"):
        binding.bind(CandidateRange(0, 1, 3, 0.0))
    with pytest.raises(CandidateReplyError, match="foreign"):
        binding.bind(CandidateRange(2, 0, 1, 0.0))


@pytest.mark.asyncio
async def test_selected_message_keeps_original_conversation_position():
    messages = Messages.from_openai(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
    )
    binding = await bind_candidate_input(messages._select_positions([1]), CandidateBatchLimits())
    span = binding.bind(CandidateRange(0, 1, 4, 1.0))
    assert (span.message_index, span.position, span.start, span.end) == (1, 1, 1, 4)
    assert span._message is messages[1]


@pytest.mark.asyncio
async def test_sliced_message_maps_back_to_original_offsets():
    messages = Messages.from_openai([{"role": "user", "content": "abcdef"}])
    selected = messages._select_ranges([0], {0: (2, 5)})
    binding = await bind_candidate_input(selected, CandidateBatchLimits())
    assert [p.text for p in binding.input.parts] == ["cde"]
    span = binding.bind(CandidateRange(0, 0, 2, 1.0))
    assert (span.start, span.end) == (2, 4)


@pytest.mark.asyncio
async def test_joined_native_sources_are_split_even_without_separators():
    originals = Messages.from_openai(
        [
            {"role": "user", "content": "ab"},
            {"role": "assistant", "content": "cd"},
        ]
    )
    origin = joined([(m.content, Place(i, "", 0, m)) for i, m in enumerate(originals)])
    messages = Messages.message_parts("user", ["ab", "cd"], origin)
    binding = await bind_candidate_input(messages, CandidateBatchLimits())
    assert [p.text for p in binding.input.parts] == ["ab", "cd"]
    assert binding.bind(CandidateRange(1, 0, 1, 1.0)).message_index == 1
    with pytest.raises(CandidateReplyError):
        binding.bind(CandidateRange(0, 1, 3, 1.0))


@pytest.mark.asyncio
async def test_raw_tool_arguments_remain_one_native_field_without_decoding(monkeypatch):
    import quarry.types.messages as message_module

    def no_decode(*args, **kwargs):
        raise AssertionError("candidate binding must not decode or join tool argument values")

    monkeypatch.setattr(message_module, "decode_tool_arguments", no_decode)
    raw = '{"left":"ab", "right":"cd"}'
    binding = await bind(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "synthetic-call",
                        "type": "function",
                        "function": {"name": "synthetic", "arguments": raw},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "synthetic-call", "content": "done"},
        ]
    )
    assert [p.text for p in binding.input.parts] == ["", raw, "done"]
    low = raw.index("ab")
    span = binding.bind(CandidateRange(1, low, low + 2, 1.0))
    assert (span.position, span.field, span.start, span.end) == (1, "raw_arguments", low, low + 2)
    assert binding.bind(CandidateRange(2, 0, 4, 1.0)).position == 2


@pytest.mark.asyncio
async def test_located_input_fields_do_not_join_authority():
    owner = object()
    source = InputText("synthetic_records", "synthetic-item", ("title", "body"))
    origin = joined(
        [
            ("ab", Place(0, "title", 0, owner, source)),
            ("cd", Place(0, "body", 0, owner, source)),
        ]
    )
    messages = Messages.message_parts("user", ["ab", "cd"], origin)
    binding = await bind_candidate_input(messages, CandidateBatchLimits())
    assert [p.text for p in binding.input.parts] == ["ab", "cd"]
    span = binding.bind(CandidateRange(1, 0, 2, 1.0))
    assert span.source == source and span.field == "body" and span._source is owner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,source_end,output_start", [(WHOLE, 2, 0), (LINEAR, 1, 0), (LINEAR, 2, 1)]
)
async def test_enclosing_interpolated_or_unmapped_text_is_refused(kind, source_end, output_start):
    originals = Messages.from_openai([{"role": "user", "content": "ab"}])
    place = Place(0, "", 0, originals[0])
    origin = Derived("ab", (place,), array("q", [output_start, 2, 0, 0, source_end, kind]))
    messages = Messages.message_parts("user", ["ab"], origin)
    with pytest.raises(CandidateReplyError, match="exact|source positions"):
        await bind_candidate_input(messages, CandidateBatchLimits())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,limits",
    [
        ("abcd", {"max_input_characters": 3}),
        ([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], {"max_parts": 1}),
    ],
)
async def test_input_limits_precede_flattening_and_native_origin_work(monkeypatch, content, limits):
    messages = Messages.from_openai([{"role": "user", "content": content}])

    def no_flatten(self):
        raise AssertionError("limit must be checked before flattening")

    monkeypatch.setattr(Message, "content", property(no_flatten))
    with pytest.raises(CandidateLimitError):
        await bind_candidate_input(messages, CandidateBatchLimits(**limits))


@pytest.mark.asyncio
async def test_low_mapping_work_refuses_before_origin_projection(monkeypatch):
    async def no_projection(*args, **kwargs):
        raise AssertionError("low work budget must refuse before origin projection")

    monkeypatch.setattr(candidate_input, "message_window_origin", no_projection)
    with pytest.raises(CandidateLimitError):
        await bind([{"role": "user", "content": "a" * 257}], max_work=1)


@pytest.mark.asyncio
async def test_empty_message_role_and_order_still_shape_identity():
    user = await bind([{"role": "user", "content": ""}])
    assistant = await bind([{"role": "assistant", "content": ""}])
    empty = await bind_candidate_input(Messages(), CandidateBatchLimits())
    assert len({user.input.digest, assistant.input.digest, empty.input.digest}) == 3


@pytest.mark.asyncio
async def test_closing_reader_releases_source_while_detached_input_survives():
    messages = Messages.from_openai([{"role": "user", "content": "synthetic"}])
    owner = weakref.ref(messages[0])
    binding = await bind_candidate_input(messages, CandidateBatchLimits())
    detached = binding.input
    del messages
    gc.collect()
    assert owner() is not None
    binding.close()
    binding.close()
    gc.collect()
    assert owner() is None
    assert detached.parts[0].text == "synthetic"
    with pytest.raises(CandidateReplyError, match="closed"):
        binding.bind(CandidateRange(0, 0, 1, 1.0))


@pytest.mark.asyncio
async def test_caller_held_evidence_owns_its_source_after_reader_close():
    messages = Messages.from_openai([{"role": "user", "content": "synthetic"}])
    owner = weakref.ref(messages[0])
    binding = await bind_candidate_input(messages, CandidateBatchLimits())
    span = binding.bind(CandidateRange(0, 0, 1, 1.0))
    binding.close()
    del messages
    gc.collect()
    assert owner() is not None and span._message is owner()
    del span
    gc.collect()
    assert owner() is None


@pytest.mark.asyncio
async def test_forged_range_is_revalidated_before_native_binding():
    binding = await bind([{"role": "user", "content": "ab"}])
    value = CandidateRange(0, 0, 1, 1.0)
    object.__setattr__(value, "start", True)
    with pytest.raises(CandidateReplyError):
        binding.bind(value)


@pytest.mark.asyncio
async def test_source_graph_depth_is_finite():
    messages = Messages.from_openai([{"role": "user", "content": "ab"}])
    origin = Place(0, "", 0, messages[0])
    for _ in range(102):
        origin = Derived("ab", (origin,), array("q", [0, 2, 0, 0, 2, LINEAR]))
    # Native construction already rejects graphs this deep. Simulate a hostile
    # post-construction mutation to exercise the candidate boundary independently.
    selected = Messages.message_parts("user", ["ab"], Place(0, "", 0, messages[0]))
    object.__setattr__(selected, "_origins", (origin,))
    with pytest.raises(CandidateLimitError, match="depth"):
        await bind_candidate_input(selected, CandidateBatchLimits())


@pytest.mark.asyncio
async def test_expired_native_deadline_propagates_unchanged(monkeypatch):
    from quarry.safety.timer import QuarryTimeoutError

    error = QuarryTimeoutError("synthetic deadline")

    def expired():
        raise error

    monkeypatch.setattr(candidate_input, "raise_if_owned_evaluation_expired", expired)
    with pytest.raises(QuarryTimeoutError) as caught:
        await bind([{"role": "user", "content": "ab"}])
    assert caught.value is error


@pytest.mark.asyncio
async def test_source_preparation_spends_selected_native_cpu_allowance(monkeypatch):
    from quarry.engine.local_work import local_work_scope
    from quarry.safety import local_match_work

    ticks = iter([0, 10])
    monkeypatch.setattr(local_match_work, "thread_time_ns", lambda: next(ticks))
    owner = local_match_work.LocalMatchWorkOwner(
        local_match_work.LocalMatchWorkGroup("synthetic", ("Read",), 1, "enforce")
    )
    try:
        with local_work_scope(owner), pytest.raises(local_match_work.LocalMatchWorkError):
            await bind([{"role": "user", "content": "synthetic"}])
        owner.close()
        report = owner.report()
        assert report["known_cpu_ns"] == 10 and report["segments"] == 1
        assert report["cut_phase"] == "prepare" and report["complete"]
    finally:
        owner.close()


@pytest.mark.asyncio
async def test_source_address_work_is_charged_before_input_hashing(monkeypatch):
    def no_hash(*args, **kwargs):
        raise AssertionError("source-address work must be refused before hashing")

    monkeypatch.setattr(candidate_input, "_detached_input", no_hash)
    with pytest.raises(CandidateLimitError):
        await bind([{"role": "user", "content": "a"}], max_work=10)


@pytest.mark.asyncio
async def test_absent_and_empty_located_item_ids_have_distinct_request_identity():
    bindings = []
    for item in (None, ""):
        source = InputText("synthetic_records", item, ("body",))
        messages = Messages.message_parts("user", ["ab"], Place(0, "body", 0, object(), source))
        bindings.append(await bind_candidate_input(messages, CandidateBatchLimits()))
    assert bindings[0].input.digest != bindings[1].input.digest
