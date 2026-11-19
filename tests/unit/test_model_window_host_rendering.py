# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Host renderers receive request settings and cannot claim unrelated input origins."""

from array import array
from dataclasses import replace

import pytest

from quarry import compile_policy
from quarry.cache.async_cache import result_cache
from quarry.integrations.model_window_input import WindowInput, join_window_input
from quarry.types.exceptions import LLMError
from quarry.types.messages import Messages
from quarry.types.origins import LINEAR, Derived, Place, SourceText
from quarry.types.results import Score
from tests.unit.model_fakes import model_entry
from tests.unit.test_model_token_head_contracts import backend
from tests.unit.test_model_window_runtime import WindowAdapter


class HostAdapter(WindowAdapter):
    def __init__(self, mode="mapped"):
        super().__init__()
        self.mode, self.renders = mode, 0

    async def render_window_input(self, config, messages, tools):
        self.renders += 1
        assert config.spans is config.category is config.score is None
        assert config.classify is None
        if self.mode == "opaque":
            return "host:" + messages[0].content.upper()
        rendered = await join_window_input(messages, tools)
        return WindowInput(rendered.source, "synthetic context", rendered.last_turn)

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
        begin = source.text.lower().index("needle")
        return [replace(source.span(begin, begin + 6), score=Score(0.9))]


def runtime(tmp_path, adapter):
    windows = {"unit": "tokens", "size": 100, "renderer": "host"}
    return backend(
        tmp_path,
        {
            "classes": {**model_entry(classify=0), "windows": windows},
            "segments": {**model_entry(spans=0, category="SYNTHETIC"), "windows": windows},
        },
        adapter,
    )


@pytest.mark.asyncio
async def test_host_input_context_and_request_settings_share_across_class_and_span(tmp_path):
    adapter = HostAdapter()
    service = runtime(tmp_path, adapter)
    policy = compile_policy(
        'rule "R" condition: model("classes").classify(messages) > 0.5\ncollect: [s for s in model("segments").spans(messages, 0.5) if true]',
        runtime_config=service.config,
        model_inference_backend=service,
    )
    result_cache.clear()
    result = await policy.scan_async({"messages": [{"role": "user", "content": "a needle"}]})
    assert result.is_complete and result.rule_matches == ["R"], result.evaluation_errors
    assert (
        adapter.renders
        == len(adapter.encodings)
        == len(adapter.requests)
        == len(adapter.decodes)
        == 1
    )
    assert adapter.requests[0][1:] == ("synthetic context", True)
    span = result.outputs["R"].items[0].value
    assert (span.start, span.end, span.precision) == (2, 8, "exact")


@pytest.mark.asyncio
async def test_opaque_host_output_only_claims_the_selected_original_envelope(tmp_path):
    adapter = HostAdapter("opaque")
    service = runtime(tmp_path, adapter)
    messages = Messages.from_openai([{"role": "user", "content": "private:a needle:tail"}])
    selected = messages._select_ranges((0,), {0: (8, 16)})
    result_cache.clear()
    spans = await service.model_span_messages("segments", selected, 0.5)
    span = spans.items[0].value
    assert adapter.decodes[0][0] == "host:A NEEDLE"
    assert (span.start, span.end, span.precision) == (8, 16, "enclosing")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["foreign-owner", "outside-selection", "overlapping-runs", "cycle"]
)
async def test_host_origin_admission_rejects_invalid_graphs_before_encoder_work(tmp_path, fault):
    class Invalid(HostAdapter):
        async def render_window_input(self, config, messages, tools):
            owner = messages._root._items[0]
            if fault == "foreign-owner":
                owner = Messages.from_openai([{"role": "user", "content": "a needle"}])[0]
            place = Place(0, "", 0, owner)
            if fault in {"foreign-owner", "outside-selection"}:
                return WindowInput(SourceText("a needle", place, 0))
            runs = array("q", (0, 4, 0, 0, 4, LINEAR, 3, 8, 0, 3, 8, LINEAR))
            derived = Derived("a needle", (place,), runs)
            if fault == "cycle":
                object.__setattr__(derived, "sources", (derived,))
            return WindowInput(SourceText("a needle", derived, 0))

    adapter = Invalid()
    service = runtime(tmp_path, adapter)
    messages = Messages.from_openai([{"role": "user", "content": "a needle"}])
    if fault == "outside-selection":
        messages = messages._select_ranges((0,), {0: (2, 8)})
    result_cache.clear()
    with pytest.raises(LLMError) as error:
        await service.model_classify_messages("classes", messages)
    assert error.value.details["code"] == "invalid_output"
    assert adapter.encodings == adapter.requests == adapter.decodes == []


@pytest.mark.asyncio
async def test_host_labels_and_caps_keep_only_request_fields_and_share_physical_work(tmp_path):
    from quarry.cache.async_cache import scan_scope

    class Labels(HostAdapter):
        async def render_window_input(self, config, messages, tools):
            self.renders += 1
            assert config.spans is config.category is config.score is None
            assert config.classify.class_index == 0
            assert config.classify.max_input_tokens is None
            assert len(config.classify.labels) == 2
            return await join_window_input(messages, tools)

    adapter = Labels()
    windows = {"unit": "tokens", "size": 100, "renderer": "host"}
    models = {
        "classes": {**model_entry(classify=1), "windows": windows},
        "segments": {**model_entry(classify=0, spans=0, category="SYNTHETIC"), "windows": windows},
    }
    for name, cap in [("classes", 8), ("segments", 9)]:
        models[name]["classify"].update(labels=["clean", "flag"], max_input_tokens=cap)
    service = backend(tmp_path, models, adapter)
    messages = Messages.from_openai([{"role": "user", "content": "needle"}])
    result_cache.clear()
    with scan_scope():
        assert await service.model_classify_messages("classes", messages) == 0.2
        assert len((await service.model_span_messages("segments", messages, 0.5)).items) == 1
    assert adapter.renders == len(adapter.encodings) == len(adapter.requests) == 1
    assert adapter.requests[0][2] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("renderer", ["join", "json", "host"])
async def test_removed_text_is_not_claimed_in_decoded_window_source_ranges(tmp_path, renderer):
    from quarry.integrations.content_transforms import apply_content_transform
    from quarry.types.transforms import ContentTransform, TransformOperation

    adapter = HostAdapter()
    windows = {"unit": "characters", "size": 1000, "renderer": renderer}
    service = backend(
        tmp_path,
        {"segments": {**model_entry(spans=0, category="SYNTHETIC"), "windows": windows}},
        adapter,
    )
    original = Messages.from_openai([{"role": "user", "content": "neeSECRETDROPdle"}])
    selected = await apply_content_transform(
        original, ContentTransform((TransformOperation("replace", "SECRETDROP", ""),))
    )
    assert selected[0].content == "needle"
    result_cache.clear()
    result = await service.model_span_messages("segments", selected, 0.5)
    span = result.items[0].value
    assert (span.start, span.end) == (0, 16)
    assert [(item.start, item.end) for item in span.source_ranges] == [(0, 3), (13, 16)]


@pytest.mark.asyncio
async def test_host_cannot_claim_text_removed_from_the_selected_projection(tmp_path):
    from quarry.integrations.content_transforms import apply_content_transform
    from quarry.types.transforms import ContentTransform, TransformOperation

    class Removed(HostAdapter):
        async def render_window_input(self, config, messages, tools):
            owner = messages._root._items[0]
            return WindowInput(SourceText("SECRETDROP", Place(0, "", 3, owner), 0))

    adapter = Removed()
    service = runtime(tmp_path, adapter)
    original = Messages.from_openai([{"role": "user", "content": "neeSECRETDROPdle"}])
    selected = await apply_content_transform(
        original, ContentTransform((TransformOperation("replace", "SECRETDROP", ""),))
    )
    result_cache.clear()
    with pytest.raises(LLMError) as error:
        await service.model_classify_messages("classes", selected)
    assert error.value.details["code"] == "invalid_output"
    assert adapter.encodings == adapter.requests == []
