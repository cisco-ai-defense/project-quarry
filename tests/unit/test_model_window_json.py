# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Native JSON window inputs keep exact text and conservative enclosing origins."""

import asyncio
import json
from dataclasses import replace

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.cache.async_cache import result_cache
from quarry.integrations import model_window_input
from quarry.types.messages import Messages
from quarry.types.results import Score
from tests.unit.model_fakes import model_entry
from tests.unit.test_model_token_head_contracts import backend
from tests.unit.test_model_window_runtime import WindowAdapter


async def render(messages, tools=None):
    function = getattr(model_window_input, "json_window_input", None)
    assert callable(function), "native JSON window rendering is required"
    return await function(messages, tools or [])


class JsonAdapter(WindowAdapter):
    async def decode_window_spans(
        self,
        config,
        raw,
        source,
        windows,
        threshold,
        *,
        encoding=None,
        token_scores=None,
        token_head=False,
    ):
        self.decodes.append((source.text, windows, encoding, token_scores, token_head))
        start = source.text.index("needle")
        return [replace(source.span(start, start + 6), score=Score(0.9))]


def runtime(tmp_path, adapter, unit):
    windows = {"unit": unit, "size": 1000, "renderer": "json"}
    return backend(
        tmp_path,
        {
            "classes": {**model_entry(classify=0), "windows": windows},
            "segments": {**model_entry(spans=0, category="SYNTHETIC"), "windows": windows},
        },
        adapter,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["characters", "tokens"])
@pytest.mark.parametrize("restored", [False, True])
async def test_json_window_policies_share_real_batches_and_exact_original_offsets(
    tmp_path, unit, restored
):
    adapter = JsonAdapter()
    service = runtime(tmp_path, adapter, unit)
    source = 'rule "R" condition: model("classes").classify(messages) > 0.5\ncollect: [s for s in model("segments").spans(messages, 0.5) if true]'
    policy = compile_policy(source, runtime_config=service.config, model_inference_backend=service)
    if restored:
        wire = policy.to_bytes()
        policy = load_compiled_policy(
            wire, runtime_config=service.config, model_inference_backend=service
        )
        assert policy.to_bytes() == wire
    data = {"messages": [{"role": "user", "content": 'say "needle" 雪'}]}
    result_cache.clear()
    result = await policy.scan_async(data)
    assert result.is_complete and result.rule_matches == ["R"], result.evaluation_errors
    expected = json.dumps(data["messages"], ensure_ascii=False, separators=(",", ":"))
    assert adapter.decodes[0][0] == expected
    assert len(adapter.requests) == 1 and adapter.requests[0][2] is True
    assert len(adapter.encodings) == (1 if unit == "tokens" else 0)
    span = result.outputs["R"].items[0].value
    assert (span.message_index, span.start, span.end, span.precision) == (0, 5, 11, "exact")


@pytest.mark.asyncio
@pytest.mark.parametrize("multipart", [False, True])
async def test_native_json_does_not_project_the_frozen_message_body(monkeypatch, multipart):
    import quarry.types.messages as message_module

    content = (
        [{"type": "text", "text": "prefix "}, {"type": "text", "text": "needle"}]
        if multipart
        else "prefix needle"
    )
    data = [{"role": "user", "content": content}]
    messages = Messages.from_openai(data)
    expected = json.dumps(messages.to_openai_messages(), ensure_ascii=False, separators=(",", ":"))

    def no_projection(*args, **kwargs):
        raise AssertionError("native frozen bodies must not be projected")

    monkeypatch.setattr(message_module, "_project_json", no_projection)
    rendered = await render(messages)
    assert rendered.source.text == expected
    begin = expected.index("needle")
    span = rendered.source.span(begin, begin + 6)
    assert (span.start, span.end, span.precision) == (7, 13, "exact")


@pytest.mark.asyncio
async def test_json_escape_ranges_are_enclosing_and_unescaped_text_stays_exact():
    messages = Messages.from_openai([{"role": "user", "content": "prefix\nneedle"}])
    rendered = await render(messages)
    start = rendered.source.text.index("\\n")
    escaped = rendered.source.span(start, start + 1)
    assert (escaped.start, escaped.end, escaped.precision) == (6, 7, "enclosing")
    start = rendered.source.text.index("needle")
    exact = rendered.source.span(start, start + 6)
    assert (exact.start, exact.end, exact.precision) == (7, 13, "exact")


@pytest.mark.asyncio
async def test_multipart_json_spans_retain_original_content_part_coordinates(tmp_path):
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "prefix "},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    {"type": "text", "text": "needle suffix"},
                ],
            }
        ]
    }
    adapter = JsonAdapter()
    service = runtime(tmp_path, adapter, "characters")
    messages = Messages.from_openai(data["messages"])
    result_cache.clear()
    spans = await service.model_span_messages("segments", messages, 0.5)
    span = spans.items[0].value
    assert (span.start, span.end, span.precision) == (7, 13, "exact")
    assert len(span.source_ranges) == 1
    part = span.source_ranges[0]
    assert (part.part_index, part.start, part.end) == (2, 0, 6)
    assert messages.extract(span) == "needle"


@pytest.mark.asyncio
async def test_message_wire_override_is_preserved_with_only_enclosing_coordinates(monkeypatch):
    messages = Messages.from_openai([{"role": "user", "content": "original body"}])
    monkeypatch.setattr(
        type(messages[0]), "to_openai_dict", lambda self: {"role": "user", "content": "host NEEDLE"}
    )
    rendered = await render(messages)
    assert json.loads(rendered.source.text)[0]["content"] == "host NEEDLE"
    start = rendered.source.text.index("NEEDLE")
    span = rendered.source.span(start, start + 6)
    assert (span.start, span.end, span.precision) == (0, 13, "enclosing")


@pytest.mark.asyncio
async def test_last_turn_json_interval_keeps_the_full_originating_cut():
    messages = Messages.from_openai(
        [{"role": "user", "content": "first"}, {"role": "assistant", "content": "needle"}]
    )
    rendered = await render(messages)
    lo, hi = rendered.last_turn
    assert json.loads(rendered.source.text[lo:hi]) == messages[1].to_openai_dict()
    start = rendered.source.text.index("needle")
    span = rendered.source.span(start, start + 6)
    assert (span.message_index, span.start, span.end) == (1, 0, 6)


@pytest.mark.asyncio
async def test_large_native_json_scalar_cooperates_with_cancellation(monkeypatch):
    from quarry.integrations import model_window_json

    messages = Messages.from_openai([{"role": "user", "content": "x" * 2_000_000}])
    content = messages[0].source_content
    entered = asyncio.Event()
    original = model_window_json._Writer.string

    async def observe(writer, value, *args):
        if value is content:
            entered.set()
        return await original(writer, value, *args)

    monkeypatch.setattr(model_window_json._Writer, "string", observe)
    task = asyncio.create_task(render(messages))
    await asyncio.wait_for(entered.wait(), 2)
    assert not task.done(), "large scalar rendering must yield after entering the scalar"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_json_window_boundary_rejects_external_media_before_encoder_dispatch(tmp_path):
    from quarry.types.exceptions import LLMError

    adapter = JsonAdapter()
    service = runtime(tmp_path, adapter, "tokens")
    messages = Messages.from_openai(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.invalid/synthetic.png"},
                    }
                ],
            }
        ]
    )
    result_cache.clear()
    with pytest.raises(LLMError) as error:
        await service.model_classify_messages("classes", messages)
    assert error.value.details["code"] == "invalid_output"
    assert adapter.encodings == adapter.requests == adapter.decodes == []


@pytest.mark.asyncio
async def test_opaque_dictionary_serialization_still_enforces_the_media_boundary(
    tmp_path, monkeypatch
):
    from quarry.types.exceptions import LLMError

    class HostDictionary(dict):
        pass

    messages = Messages.from_openai([{"role": "user", "content": "original"}])
    monkeypatch.setattr(
        type(messages[0]),
        "to_openai_dict",
        lambda self: HostDictionary(
            role="user",
            content=[
                {"type": "image_url", "image_url": {"url": "https://example.invalid/synthetic.png"}}
            ],
        ),
    )
    adapter = JsonAdapter()
    service = runtime(tmp_path, adapter, "tokens")
    result_cache.clear()
    with pytest.raises(LLMError) as error:
        await service.model_classify_messages("classes", messages)
    assert error.value.details["code"] == "invalid_output"
    assert adapter.encodings == adapter.requests == adapter.decodes == []


@pytest.mark.asyncio
async def test_host_dictionary_serialization_keeps_interpreter_behavior_and_call_count(monkeypatch):
    class HostDictionary(dict):
        calls = 0

        def items(self):
            type(self).calls += 1
            return [("role", "user"), ("content", f"host{type(self).calls} needle")]

    wire = HostDictionary(role="user", content="stored")
    expected = json.dumps([wire], ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    expected_calls = HostDictionary.calls
    HostDictionary.calls = 0
    messages = Messages.from_openai([{"role": "user", "content": "original"}])
    monkeypatch.setattr(type(messages[0]), "to_openai_dict", lambda self: wire)
    rendered = await render(messages)
    assert rendered.source.text == expected and HostDictionary.calls == expected_calls


@pytest.mark.asyncio
async def test_many_multipart_spans_prepare_original_provenance_once(tmp_path, monkeypatch):
    import re

    from quarry.types.messages import Message

    count, visits = 128, [0]
    original = Message._provenance_steps

    def observed(message):
        for segment in original(message):
            visits[0] += 1
            yield segment

    monkeypatch.setattr(Message, "_provenance_steps", observed)

    class EveryPart(JsonAdapter):
        async def decode_window_spans(self, config, raw, source, windows, threshold, **kwargs):
            return [
                replace(source.span(match.start(), match.end()), score=Score(0.9))
                for match in re.finditer("needle", source.text)
            ]

    messages = Messages.from_openai(
        [{"role": "user", "content": [{"type": "text", "text": "needle"} for _ in range(count)]}]
    )
    adapter = EveryPart()
    service = runtime(tmp_path, adapter, "characters")
    result_cache.clear()
    spans = await service.model_span_messages("segments", messages, 0.5)
    assert len(spans.items) == count
    assert visits[0] <= 2 * count, f"repeated whole-message provenance walks: {visits[0]}"
    assert [
        (item.value.source_ranges[0].part_index, item.value.start, item.value.end)
        for item in spans.items
    ] == [(index, 6 * index, 6 * index + 6) for index in range(count)]
