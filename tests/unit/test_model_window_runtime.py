# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Windowed policies use exact host batches, shared heads and full-origin decoders."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.cache.async_cache import AsyncFunctionCache, SlidingTTLCache, result_cache
from quarry.integrations import model_inference
from quarry.integrations.model_windows import TokenEncoding
from quarry.safety.limits import SafetyLimits
from quarry.types.exceptions import ConfigurationError, LLMError
from quarry.types.messages import Messages
from quarry.types.results import Score
from tests.unit.model_fakes import model_entry
from tests.unit.test_model_token_head_contracts import backend

DATA = {"messages": [{"role": "user", "content": "abcdef"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["offset", "precision", "range", "unmapped"])
async def test_decoder_cannot_change_the_geometry_of_a_bound_window_span(tmp_path, fault):
    from quarry.types.transforms import ContentSourceRange

    class Changed(WindowAdapter):
        async def decode_window_spans(self, config, raw, source, windows, threshold, **kwargs):
            span = replace(source.span(1, 2), score=Score(0.9))
            changes = {
                "offset": {"end": 100},
                "precision": {"precision": "enclosing"},
                "range": {"source_ranges": (ContentSourceRange(0, None, 5, 6, 5, 6),)},
                "unmapped": {"source_unmapped": True},
            }
            return [replace(span, **changes[fault])]

    runtime = service(tmp_path, "characters", Changed())
    result_cache.clear()
    with pytest.raises(LLMError) as error:
        await runtime.model_span_messages("segments", Messages.from_openai(DATA["messages"]), 0.5)
    assert error.value.details["code"] == "invalid_output"


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["characters", "tokens"])
async def test_compatible_readers_prepare_one_plan_per_scan(tmp_path, monkeypatch, unit):
    from quarry.cache.async_cache import scan_scope

    calls = []
    name = "character_windows" if unit == "characters" else "token_windows"
    original = getattr(model_inference, name)

    async def observed(*args, **kwargs):
        calls.append(name)
        return await original(*args, **kwargs)

    monkeypatch.setattr(model_inference, name, observed)
    adapter = WindowAdapter()
    runtime = service(tmp_path, unit, adapter)
    messages = Messages.from_openai(DATA["messages"])
    result_cache.clear()
    for count in (1, 2):
        with scan_scope():
            assert await runtime.model_classify_messages("classes", messages) == 0.8
            spans = await runtime.model_span_messages("segments", messages, 0.5)
            assert len(spans.items) == 3
            assert len(calls) == count


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["render", "plan", "measured"])
async def test_released_window_memo_cannot_retain_a_late_completed_value(operation):
    from quarry.integrations.model_window_input import WindowMemo, join_window_input
    from quarry.integrations.model_windows import CharacterWindow

    messages = Messages.from_openai(DATA["messages"])
    rendering = await join_window_input(messages, [])
    entered, release = asyncio.Event(), asyncio.Event()
    memo = WindowMemo(100_000)

    async def compute():
        entered.set()
        await release.wait()
        if operation == "render":
            return rendering
        if operation == "measured":
            return (0.8,), 0.8, 0
        return (CharacterWindow(rendering.source, 0, 6),), ("abcdef",)

    task = asyncio.create_task(getattr(memo, operation)(b"synthetic", compute))
    await entered.wait()
    memo.close()
    release.set()
    with pytest.raises(RuntimeError, match="released"):
        await task
    assert memo.budget.bytes == 0
    assert not memo.entries and not memo.plans
    assert not memo.measurements


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["characters", "tokens"])
async def test_window_probabilities_share_measurements_only_within_the_scan(
    tmp_path, monkeypatch, unit
):
    from quarry.cache.async_cache import scan_scope

    counts = []
    original = model_inference._validated_score

    def observed(value, model_id, description):
        if description == "Classifier probability":
            counts.append(model_id)
        return original(value, model_id, description)

    monkeypatch.setattr(model_inference, "_validated_score", observed)
    windows = {"unit": unit, "size": 3, "step": 2}
    models = {
        name: {**model_entry(classify=index), "windows": windows}
        for name, index in [("classes", 0), ("same", 0), ("other", 1)]
    }
    adapter = WindowAdapter()
    runtime = backend(tmp_path, models, adapter)
    messages = Messages.from_openai(DATA["messages"])
    result_cache.clear()
    for scan_number in (1, 2):
        with scan_scope():
            assert await runtime.model_classify_messages("classes", messages) == 0.8
            assert await runtime.model_classify_messages("same", messages) == 0.8
            assert await runtime.model_classify_messages("other", messages) == 0.2
            assert len(counts) == 6 * scan_number
    assert len(adapter.requests) == 1


@pytest.mark.asyncio
async def test_invalid_late_window_probability_does_not_become_a_shared_measurement(tmp_path):
    from quarry.cache.async_cache import scan_scope

    class Invalid(WindowAdapter):
        async def classify_window_batch(self, *args, **kwargs):
            raw = await super().classify_window_batch(*args, **kwargs)
            if len(self.requests) == 1:
                raw.chunks[-1].probabilities[0] = float("nan")
            return raw

    adapter = Invalid()
    runtime = service(tmp_path, "tokens", adapter)
    messages = Messages.from_openai(DATA["messages"])
    result_cache.clear()
    with scan_scope():
        with pytest.raises(LLMError) as error:
            await runtime.model_classify_messages("classes", messages)
        assert error.value.details["code"] == "invalid_output"
        assert await runtime.model_classify_messages("classes", messages) == 0.8
    assert len(adapter.requests) == 2 and len(adapter.encodings) == 1


@pytest.mark.asyncio
async def test_native_span_readers_share_selected_head_validation_per_scan(tmp_path, monkeypatch):
    from quarry.cache.async_cache import scan_scope

    original = getattr(model_inference, "selected_token_scores_async", None)
    assert callable(original), "native token-head validation must be cooperative and shared"
    calls = []

    async def observed(*args, **kwargs):
        calls.append(args[2])
        return await original(*args, **kwargs)

    monkeypatch.setattr(model_inference, "selected_token_scores_async", observed)
    adapter = WindowAdapter()
    runtime = service(tmp_path, "tokens", adapter)
    messages = Messages.from_openai(DATA["messages"])
    result_cache.clear()
    for count in (1, 2):
        with scan_scope():
            assert len((await runtime.model_span_messages("segments", messages, 0.5)).items) == 3
            assert len((await runtime.model_span_messages("segments", messages, 0.6)).items) == 3
            assert len(calls) == count
    assert len(adapter.requests) == 1 and len(adapter.decodes) == 4


@pytest.mark.asyncio
async def test_shared_window_heads_keep_different_selected_classes_separate(tmp_path, monkeypatch):
    from quarry.cache.async_cache import scan_scope

    calls = []
    original = model_inference.selected_token_scores_async

    async def observed(*args, **kwargs):
        calls.append(args[2])
        return await original(*args, **kwargs)

    monkeypatch.setattr(model_inference, "selected_token_scores_async", observed)
    windows = {"unit": "tokens", "size": 3, "step": 2}
    models = {
        "first": {**model_entry(spans=0, category="FIRST"), "windows": windows},
        "second": {**model_entry(spans=1, category="SECOND"), "windows": windows},
    }
    adapter = WindowAdapter()
    runtime = backend(tmp_path, models, adapter)
    messages = Messages.from_openai(DATA["messages"])
    result_cache.clear()
    with scan_scope():
        for alias in ["first", "second", "first", "second"]:
            assert len((await runtime.model_span_messages(alias, messages, 0.05)).items) == 3
    assert calls == [0, 1] and len(adapter.requests) == 1
    assert [entry[3][0][0] for entry in adapter.decodes] == [0.9, 0.1, 0.9, 0.1]


class WindowAdapter:
    supports_token_head = True
    window_units = frozenset({"characters", "tokens"})

    def __init__(self):
        self.encodings = []
        self.requests = []
        self.decodes = []
        self.bad_last_head = False

    async def encode_window_input(self, config, text, context, last_turn, *, timeout=None):
        self.encodings.append((text, context, last_turn))
        turn = None if last_turn is None else (last_turn[0] + 1, last_turn[1] + 1)
        return TokenEncoding(
            (111, *map(ord, text), 222), True, True, turn, context_tokens=len(context)
        )

    async def classify_window_batch(
        self, config, batches, context, *, timeout=None, token_head=False
    ):
        self.requests.append(
            (tuple(tuple(b) if isinstance(b, list) else b for b in batches), context, token_head)
        )
        head = [[[0.9, 0.1] for _ in batch] for batch in batches] if token_head else None
        if head and self.bad_last_head:
            head[-1][-1][0] = float("nan")
        return SimpleNamespace(
            chunks=[SimpleNamespace(probabilities=[0.8, 0.2]) for _ in batches], token_head=head
        )

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
        spans = []
        for window in windows:
            if encoding is None:
                span = window.span(0, 1)
            else:
                position = next(
                    p for p in window.positions if p not in (0, len(encoding.token_ids) - 1)
                )
                span = source.span(position - 1, position)
            spans.append(replace(span, score=Score(0.9)))
        return spans


def service(tmp_path, unit, adapter, *, cap=None, scope="all"):
    windows = {"unit": unit, "size": 3, "step": 2, "scope": scope}
    models = {
        "classes": {**model_entry(classify=0), "windows": windows},
        "segments": {**model_entry(spans=0, category="SYNTHETIC"), "windows": windows},
    }
    if cap is not None:
        models["classes"]["classify"]["max_input_tokens"] = cap
    return backend(tmp_path, models, adapter)


def expected_batches(unit):
    return (
        ("abc", "cde", "ef")
        if unit == "characters"
        else ((111, 97, 98, 99, 222), (111, 99, 100, 101, 222), (111, 101, 102, 222))
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["characters", "tokens"])
@pytest.mark.parametrize("span_first", [False, True])
async def test_native_window_class_and_span_readers_share_exact_batches(tmp_path, unit, span_first):
    adapter = WindowAdapter()
    runtime = service(tmp_path, unit, adapter)
    messages = Messages.from_openai(DATA["messages"])
    result_cache.clear()
    if span_first:
        spans = await runtime.model_span_messages("segments", messages, 0.5)
        score = await runtime.model_classify_messages("classes", messages)
    else:
        score = await runtime.model_classify_messages("classes", messages)
        spans = await runtime.model_span_messages("segments", messages, 0.5)
    assert score == 0.8
    assert adapter.requests == [(expected_batches(unit), "", True)]
    assert len(adapter.encodings) == (1 if unit == "tokens" else 0)
    assert len(adapter.decodes) == 1 and adapter.decodes[0][0] == "abcdef"
    assert [messages.extract(item.value) for item in spans.items] == ["a", "c", "e"]


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["characters", "tokens"])
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("collect", [False, True])
async def test_compiled_window_scans_request_only_selected_heads(tmp_path, unit, restored, collect):
    adapter = WindowAdapter()
    runtime = service(tmp_path, unit, adapter)
    source = 'rule "R" condition: model("classes").classify(messages) > 0.5'
    if collect:
        source += '\ncollect: [s for s in model("segments").spans(messages, 0.5) if true]'
    policy = compile_policy(source, runtime_config=runtime.config, model_inference_backend=runtime)
    if restored:
        raw = policy.to_bytes()
        policy = load_compiled_policy(
            raw, runtime_config=runtime.config, model_inference_backend=runtime
        )
        assert policy.to_bytes() == raw
    result_cache.clear()
    result = await policy.scan_async(DATA)
    assert result.is_complete and result.rule_matches == ["R"], result.evaluation_errors
    assert adapter.requests == [(expected_batches(unit), "", collect)]
    assert len(adapter.encodings) == (1 if unit == "tokens" else 0)
    assert len(adapter.decodes) == int(collect)


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["characters", "tokens"])
async def test_empty_window_input_does_no_encoder_provider_or_decoder_work(tmp_path, unit):
    adapter = WindowAdapter()
    runtime = service(tmp_path, unit, adapter)
    empty = Messages.from_openai([{"role": "user", "content": ""}])
    result_cache.clear()
    assert await runtime.model_classify_messages("classes", empty) == 0.0
    assert not (await runtime.model_span_messages("segments", empty, 0.5)).items
    assert adapter.encodings == adapter.requests == adapter.decodes == []


@pytest.mark.asyncio
async def test_token_caps_include_both_endpoint_tokens_before_classification(tmp_path):
    adapter = WindowAdapter()
    runtime = service(tmp_path, "tokens", adapter, cap=4)
    result_cache.clear()
    with pytest.raises(LLMError) as error:
        await runtime.model_classify_messages("classes", Messages.from_openai(DATA["messages"]))
    assert error.value.details["code"] == "conversation_too_large"
    assert len(adapter.encodings) == 1 and adapter.requests == adapter.decodes == []


@pytest.mark.asyncio
async def test_invalid_late_token_head_fails_before_the_host_decoder(tmp_path):
    adapter = WindowAdapter()
    adapter.bad_last_head = True
    runtime = service(tmp_path, "tokens", adapter)
    result_cache.clear()
    with pytest.raises(LLMError) as error:
        await runtime.model_span_messages("segments", Messages.from_openai(DATA["messages"]), 0.5)
    assert error.value.details["code"] == "invalid_output"
    assert len(adapter.requests) == 1 and adapter.decodes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["characters", "tokens"])
async def test_last_turn_batches_decode_against_the_full_two_message_origin(tmp_path, unit):
    adapter = WindowAdapter()
    runtime = service(tmp_path, unit, adapter, scope="last_turn")
    messages = Messages.from_openai(
        [{"role": "user", "content": "first"}, {"role": "assistant", "content": "abcdef"}]
    )
    result_cache.clear()
    spans = await runtime.model_span_messages("segments", messages, 0.5)
    assert adapter.requests == [(expected_batches(unit), "", True)]
    assert adapter.decodes[0][0] == "first\nabcdef"
    assert [
        (item.value.message_index, item.value.start, item.value.end) for item in spans.items
    ] == [(1, 0, 1), (1, 2, 3), (1, 4, 5)]
    assert [messages.extract(item.value) for item in spans.items] == ["a", "c", "e"]


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["characters", "tokens"])
async def test_multiple_empty_messages_do_not_create_input_from_join_separators(tmp_path, unit):
    adapter = WindowAdapter()
    runtime = service(tmp_path, unit, adapter)
    messages = Messages.from_openai(
        [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]
    )
    result_cache.clear()
    assert await runtime.model_classify_messages("classes", messages) == 0.0
    assert not (await runtime.model_span_messages("segments", messages, 0.5)).items
    assert adapter.encodings == adapter.requests == adapter.decodes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("operation", ["model_scan", "model_score"])
async def test_legacy_field_classifiers_use_the_same_configured_window_runtime(
    tmp_path, restored, operation
):
    adapter = WindowAdapter()
    runtime = service(tmp_path, "tokens", adapter)
    call = (
        'model_scan("classes", m.content, 0.5, 0)'
        if operation == "model_scan"
        else 'model_score("classes", m.content, 0) > 0.5'
    )
    source = f'rule "R" condition: any m in user_messages where {call}'
    policy = compile_policy(source, runtime_config=runtime.config, model_inference_backend=runtime)
    if restored:
        raw = policy.to_bytes()
        policy = load_compiled_policy(
            raw, runtime_config=runtime.config, model_inference_backend=runtime
        )
        assert policy.to_bytes() == raw
    result_cache.clear()
    result = await policy.scan_async(DATA)
    assert result.is_complete and result.rule_matches == ["R"], result.evaluation_errors
    assert adapter.requests == [(expected_batches("tokens"), "", False)]
    assert len(adapter.encodings) == 1


@pytest.mark.asyncio
async def test_bad_character_token_counts_are_rejected_and_retried(tmp_path):
    class Counter(WindowAdapter):
        count_calls = 0

        async def window_token_counts(self, config, batches, context, *, timeout=None):
            self.count_calls += 1
            return (
                [True] * len(batches)
                if self.count_calls == 1
                else [len(batch) for batch in batches]
            )

    adapter = Counter()
    runtime = service(tmp_path, "characters", adapter, cap=3)
    messages = Messages.from_openai(DATA["messages"])
    result_cache.clear()
    with pytest.raises(LLMError) as error:
        await runtime.model_classify_messages("classes", messages)
    assert error.value.details["code"] == "invalid_output" and adapter.requests == []
    assert await runtime.model_classify_messages("classes", messages) == 0.8
    assert adapter.count_calls == 2 and len(adapter.requests) == 1


@pytest.mark.asyncio
async def test_window_rendering_is_shared_within_the_scan_and_released_afterward(
    tmp_path, monkeypatch
):
    from quarry.integrations import model_window_input

    renders, memos = [], []
    original_render = model_inference.join_window_input
    original_init = model_window_input.WindowMemo.__init__

    async def render(*args):
        value = await original_render(*args)
        renders.append(value.source.text)
        return value

    def init(memo, *args):
        original_init(memo, *args)
        memos.append(memo)

    monkeypatch.setattr(model_inference, "join_window_input", render)
    monkeypatch.setattr(model_window_input.WindowMemo, "__init__", init)
    adapter = WindowAdapter()
    runtime = service(tmp_path, "tokens", adapter)
    policy = compile_policy(
        'rule "R" condition: model("classes").classify(messages) > 0.5\ncollect: [s for s in model("segments").spans(messages, 0.5) if true]',
        runtime_config=runtime.config,
        model_inference_backend=runtime,
    )
    result_cache.clear()
    result = await policy.scan_async(DATA)
    assert result.is_complete and result.rule_matches == ["R"], result.evaluation_errors
    assert renders == ["abcdef"] and memos
    assert all(
        not memo.entries and not memo.budget._objects and memo.budget.bytes == 0 for memo in memos
    )


@pytest.mark.parametrize("unit", ["characters", "tokens"])
def test_preparation_rejects_undeclared_window_units_before_work(tmp_path, unit):
    adapter = WindowAdapter()
    adapter.window_units = frozenset()
    runtime = service(tmp_path, unit, adapter)
    with pytest.raises(ConfigurationError, match="window"):
        compile_policy(
            'rule "R" condition: model("classes").classify(messages) > 0.5',
            runtime_config=runtime.config,
            model_inference_backend=runtime,
        )
    assert adapter.encodings == adapter.requests == adapter.decodes == []


@pytest.mark.asyncio
async def test_a_partial_window_response_is_rejected_and_not_reused(tmp_path):
    class Partial(WindowAdapter):
        async def classify_window_batch(self, *args, **kwargs):
            raw = await super().classify_window_batch(*args, **kwargs)
            if len(self.requests) == 1:
                raw.chunks.pop()
            return raw

    adapter = Partial()
    runtime = service(tmp_path, "tokens", adapter)
    messages = Messages.from_openai(DATA["messages"])
    result_cache.clear()
    with pytest.raises(LLMError) as error:
        await runtime.model_classify_messages("classes", messages)
    assert error.value.details["code"] == "invalid_output"
    assert await runtime.model_classify_messages("classes", messages) == 0.8
    assert len(adapter.requests) == 2 and len(adapter.encodings) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("enforce", [False, True])
async def test_window_counts_obey_the_ordinary_configured_safety_limit(tmp_path, enforce):
    adapter = WindowAdapter()
    original = service(tmp_path, "tokens", adapter)
    config = replace(
        original.config,
        safety_limits=SafetyLimits(max_comprehension_size=2, enforce_limits=enforce),
    )
    runtime = model_inference.ModelInferenceBackend(config, adapters={"fake": adapter})
    policy = compile_policy(
        'rule "R" condition: model("classes").classify(messages) > 0.5',
        runtime_config=config,
        model_inference_backend=runtime,
    )
    result_cache.clear()
    result = await policy.scan_async(DATA)
    assert result.is_complete is (not enforce)
    assert result.rule_matches == ([] if enforce else ["R"])
    assert len(adapter.requests) == (0 if enforce else 1)


@pytest.mark.asyncio
async def test_runtime_encodings_and_heads_have_fixed_five_minute_expiry(tmp_path, monkeypatch):
    now = [0.0]
    cache = AsyncFunctionCache(store=SlidingTTLCache(maxsize=16, ttl=1000, timer=lambda: now[0]))
    monkeypatch.setattr(model_inference, "result_cache", cache)
    adapter = WindowAdapter()
    runtime = service(tmp_path, "tokens", adapter)
    messages = Messages.from_openai(DATA["messages"])
    for instant in [0, 250, 299, 300]:
        now[0] = instant
        assert await runtime.model_classify_messages("classes", messages) == 0.8
    assert len(adapter.encodings) == len(adapter.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["resolve", "discard"])
async def test_deferred_window_evidence_keeps_then_releases_render_ownership(
    tmp_path, monkeypatch, finish
):
    from quarry.integrations import model_window_input

    memos = []
    original = model_window_input.WindowMemo.__init__

    def init(memo, *args):
        original(memo, *args)
        memos.append(memo)

    monkeypatch.setattr(model_window_input.WindowMemo, "__init__", init)
    adapter = WindowAdapter()
    runtime = service(tmp_path, "tokens", adapter)
    policy = compile_policy(
        'rule "R" condition: model("classes").classify(messages) > 0.5',
        runtime_config=runtime.config,
        model_inference_backend=runtime,
    )
    callbacks = []
    result_cache.clear()
    result = await policy._scanner._scan_data_with_parameters_async(
        DATA, {}, enable_explain=True, deferred_evidence=callbacks
    )
    assert result.is_complete and len(callbacks) == 1 and memos
    assert all(memo.entries and memo.budget.bytes > 0 for memo in memos)
    if finish == "resolve":
        result = await asyncio.create_task(callbacks[0](result))
        assert result.rule_matches == ["R"]
    else:
        callbacks[0].close()
    assert all(
        not memo.entries and not memo.budget._objects and memo.budget.bytes == 0 for memo in memos
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["characters", "tokens"])
async def test_an_empty_last_turn_does_no_work_even_with_earlier_context(tmp_path, unit):
    adapter = WindowAdapter()
    runtime = service(tmp_path, unit, adapter, scope="last_turn")
    messages = Messages.from_openai(
        [
            {"role": "user", "content": "earlier context"},
            {"role": "assistant", "content": ""},
        ]
    )
    result_cache.clear()
    assert await runtime.model_classify_messages("classes", messages) == 0.0
    assert not (await runtime.model_span_messages("segments", messages, 0.5)).items
    assert adapter.encodings == adapter.requests == adapter.decodes == []
