# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Read-local media faults, provider isolation and tool-result text arrays."""

from __future__ import annotations

import asyncio
import json
import tracemalloc
from types import SimpleNamespace

import pytest

from quarry import IncompleteScanError, InvalidInputError, compile_policy, load_compiled_policy
from quarry.cli import scan_command
from quarry.config import QuarryConfig
from quarry.integrations import model_inference
from quarry.integrations.adapter_message_transforms import apply_adapter_transform
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.integrations.model_usage import model_call_budget
from quarry.safety.timer import QuarryTimeoutError
from quarry.sdk import QuarryGuard
from quarry.types import media
from quarry.types.messages import Messages
from tests.unit.model_fakes import ModelAdapter, model_entry, passthrough_builder, v2_scanner
from tests.unit.test_adapter_transform_dsl import configured
from tests.unit.test_message_rewrites import service


def image(url="data:image/png;base64,AA=="):
    return {"type": "image_url", "image_url": {"url": url}}


def audio():
    return {"type": "input_audio", "input_audio": {"data": "opaque", "format": "wav"}}


def transcript(part, *, tool=False):
    content = [{"type": "text", "text": "visible"}, part]
    if not tool:
        return [{"role": "user", "content": content}]
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c", "content": content},
    ]


@pytest.mark.parametrize("body", ["", "AA", "AAA", "AAAA", "AA==", "AB==", "AAA="])
def test_supported_inline_images_do_not_require_pixel_decoding(body):
    assert media.media_fault("user", image("data:image/x-test;base64," + body)) is None


@pytest.mark.parametrize(
    "url",
    [
        "data:text/plain;base64,c2VjcmV0",
        "data:image/png,AA==",
        "data:image/png;BASE64,AA==",
        "data:image/png;foo=bar;base64,AA==",
        "data:IMAGE/png;base64,AA==",
        "data:image/;base64,AA==",
        "data:image/x/y;base64,AA==",
        *[
            "data:image/png;base64," + body
            for body in [
                "A",
                "AA=",
                "AAAA=",
                "A===",
                "=",
                "==",
                "AA==AAAA",
                "AA==\n",
                "A A=",
                "-_==",
            ]
        ],
    ],
)
def test_malformed_envelopes_fail_without_echoing_payload(url):
    with pytest.raises(InvalidInputError) as failure:
        media.require_media_part("user", image(url), owner="messages[2].content[1]")
    assert failure.value.details["code"] == "invalid_input"
    assert "messages[2].content[1]" in str(failure.value)
    assert url not in str(failure.value) and "c2VjcmV0" not in str(failure.value)


@pytest.mark.parametrize("part", [image(), audio()])
def test_tool_result_media_keeps_text_and_links_but_only_fails_its_reader(tmp_path, part):
    adapter = ModelAdapter()
    scanner = v2_scanner(
        tmp_path,
        {
            "Text": 'any m in tool_responses where m.content contains "visible"',
            "Model": 'model("reader").classify(messages) > 0.5',
        },
        {"reader": model_entry(classify=0)},
        adapter,
    )
    payload = {"messages": transcript(part, tool=True)}
    messages = Messages.from_openai(payload["messages"])
    assert messages[1].content == "visible" and messages[1].result == "visible"
    assert messages[1].call is messages[0].tool_calls[0]
    assert messages[0].tool_calls[0].response is messages[1]
    assert messages.to_openai_messages() == payload["messages"]
    result = scanner.scan_data(payload)
    assert result.rule_matches == ["Text"] and result.completion_state == "incomplete"
    assert result.validation_error is None and set(result.evaluation_errors) == {"Model"}
    assert result.failures[0].category.value == "invalid_input"
    assert adapter.classifications == 0


@pytest.mark.parametrize("prepared", [False, True])
def test_prepared_media_faults_do_not_poison_a_later_valid_scan(tmp_path, prepared):
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"reader": model_entry(classify=0)}},
        tmp_path,
        require_rules=False,
    )
    adapter = ModelAdapter()
    backend = ModelInferenceBackend(config, adapters={"fake": adapter})
    policy = compile_policy(
        'rule "Read" condition: model("reader").classify(messages) > 0.5',
        runtime_config=config,
        model_inference_backend=backend,
    )
    if prepared:
        policy = load_compiled_policy(
            policy.to_bytes(), runtime_config=config, model_inference_backend=backend
        )
    bad = policy.scan({"messages": transcript(image("data:text/plain;base64,c2VjcmV0"))})
    assert bad.rule_matches == [] and set(bad.evaluation_errors) == {"Read"}
    assert bad.failures[0].category.value == "invalid_input" and adapter.classifications == 0
    good = policy.scan({"messages": transcript(image("data:image/png;base64,AA"))})
    assert good.is_complete and good.rule_matches == ["Read"]
    assert adapter.classifications == 1


def backend_for(tmp_path):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "reader": model_entry(
                    classify=0,
                    score=True,
                    tokens=True,
                    spans=0,
                    category="ENTITY",
                    physical_model="text-model",
                )
            },
        },
        tmp_path,
        require_rules=False,
    )
    adapter = ModelAdapter()
    backend = ModelInferenceBackend(
        config, adapters={"fake": adapter}, message_builders={"test": passthrough_builder}
    )
    return backend, adapter


async def invoke(backend, operation, messages):
    if operation == "score":
        return await backend.model_score_messages("reader", "policy", messages)
    if operation == "legacy_score":
        return await backend.model_text_score("reader", "policy", messages)
    if operation == "classify":
        return await backend.model_classify_messages("reader", messages)
    if operation == "spans":
        return await backend.model_span_messages("reader", messages, 0.5)
    if operation == "complete":
        return await backend.model_complete("reader", messages)
    if operation == "legacy_complete":
        return await backend.model_text_completion("reader", messages)
    if operation == "tokens":
        return await backend.model_input_tokens_messages("reader", messages)
    if operation == "legacy_tokens":
        return await backend.model_input_tokens("reader", messages)
    if operation == "split":
        return await backend.model_split_messages("reader", messages, 1000)
    raise AssertionError(operation)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        "score",
        "legacy_score",
        "classify",
        "spans",
        "complete",
        "legacy_complete",
        "tokens",
        "legacy_tokens",
        "split",
    ],
)
async def test_direct_backend_checks_before_dispatch_and_budget(tmp_path, operation):
    backend, adapter = backend_for(tmp_path)
    messages = Messages.from_openai(transcript(image("data:text/plain;base64,c2VjcmV0")))
    with model_call_budget(1) as budget:
        with pytest.raises(InvalidInputError):
            await invoke(backend, operation, messages)
        assert budget.calls_by_alias == {}
    assert (
        adapter.classifications
        == adapter.score_calls
        == adapter.completions
        == adapter.tokenizations
        == 0
    )


@pytest.mark.asyncio
async def test_split_checks_context_before_any_token_probe(tmp_path):
    backend, adapter = backend_for(tmp_path)
    messages = Messages.from_openai(
        [
            {"role": "system", "content": [image("data:image/png;base64,A")]},
            {"role": "user", "content": "safe"},
        ]
    )
    with pytest.raises(InvalidInputError):
        await backend.model_split_messages("reader", messages[1:], 1000, context=messages[:1])
    assert adapter.tokenizations == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["score", "classify", "complete", "tokens"])
async def test_warm_valid_image_cache_cannot_legitimize_tool_role(tmp_path, operation):
    backend, adapter = backend_for(tmp_path)
    good = Messages.from_openai(transcript(image()))
    await invoke(backend, operation, good)
    before = (
        adapter.classifications,
        adapter.score_calls,
        adapter.completions,
        adapter.tokenizations,
    )
    bad = Messages.from_openai(transcript(image(), tool=True))
    with pytest.raises(InvalidInputError):
        await invoke(backend, operation, bad)
    assert (
        adapter.classifications,
        adapter.score_calls,
        adapter.completions,
        adapter.tokenizations,
    ) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [False, True])
async def test_selected_part_transform_rejects_before_host_or_budget(tmp_path, model):
    _, backend, adapter = configured(tmp_path, model=model)
    messages = Messages.from_openai(transcript(image("data:text/plain;base64,c2VjcmV0")))
    with model_call_budget(1) as budget:
        with pytest.raises(InvalidInputError):
            await apply_adapter_transform(messages, backend, "convert")
        assert budget.calls_by_alias == {}
    assert adapter.parts == []
    with pytest.raises(InvalidInputError):
        await backend.transform_unit("convert", image("data:text/plain;base64,c2VjcmV0"))
    assert adapter.parts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("part", [image(), audio()])
async def test_part_transform_rejects_tool_media_before_even_an_earlier_text_part(tmp_path, part):
    _, backend, adapter = configured(tmp_path)
    with pytest.raises(InvalidInputError):
        await apply_adapter_transform(
            Messages.from_openai(transcript(part, tool=True)), backend, "convert"
        )
    assert adapter.parts == []


@pytest.mark.asyncio
async def test_text_only_transform_preserves_unselected_unreadable_media(tmp_path):
    _, backend, adapter = configured(
        tmp_path, definitions={"convert": {"adapter": "fake", "parts": ["text"]}}
    )
    raw = transcript(image("data:text/plain;base64,c2VjcmV0"), tool=True)
    source = Messages.from_openai(raw)
    transformed = await apply_adapter_transform(source, backend, "convert")
    assert transformed[1].content == "VISIBLE" and source[1].content == "visible"
    assert transformed.to_openai_messages()[1]["content"][1] == raw[1]["content"][1]
    assert len(adapter.parts) == 1
    assert transformed[1].call is transformed[0].tool_calls[0]
    assert transformed.source_ranges(1, 0, 7)[0].message_index == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("direct", [False, True])
async def test_whole_message_rewrite_checks_media_before_host_code(tmp_path, direct):
    _, backend, adapter = service(tmp_path)
    source = Messages.from_openai(transcript(image("data:image/png;base64,A")))
    with pytest.raises(InvalidInputError):
        if direct:
            await backend.transform_messages("rewrite", source)
        else:
            await apply_adapter_transform(source, backend, "rewrite")
    assert adapter.calls == []


def test_custom_model_backend_cannot_bypass_rule_dispatch_guard(tmp_path):
    backend, adapter = backend_for(tmp_path)
    calls = []

    async def custom(*args, **kwargs):
        calls.append(args)
        return 0.99

    backend.model_classify_messages = custom
    policy = compile_policy(
        'rule "Read" condition: model("reader").classify(messages) > 0.5',
        runtime_config=backend.config,
        model_inference_backend=backend,
    )
    result = policy.scan({"messages": transcript(image("data:image/png;base64,A"))})
    assert (
        result.completion_state == "incomplete"
        and result.failures[0].category.value == "invalid_input"
    )
    assert calls == [] and adapter.classifications == 0


@pytest.mark.asyncio
async def test_custom_part_backend_cannot_bypass_role_guard(tmp_path):
    _, backend, adapter = configured(tmp_path)

    async def custom(*args, **kwargs):
        pytest.fail("unreadable media reached injected host code")

    backend.transform_unit = custom
    with pytest.raises(InvalidInputError):
        await apply_adapter_transform(
            Messages.from_openai(transcript(audio(), tool=True)), backend, "convert"
        )
    assert adapter.parts == []


def test_external_url_policy_remains_separate(tmp_path):
    adapter = ModelAdapter()
    scanner = v2_scanner(
        tmp_path,
        {"Read": 'model("reader").classify(messages) > 0.5'},
        {"reader": model_entry(classify=0)},
        adapter,
    )
    part = image("https://example.invalid/image.png")
    assert media.media_fault("user", part) is None
    result = scanner.scan_data({"messages": transcript(part)})
    assert not result.is_complete and adapter.classifications == 0


def test_audio_payload_is_opaque_outside_tool_results(tmp_path):
    adapter = ModelAdapter()
    scanner = v2_scanner(
        tmp_path,
        {"Read": 'model("reader").classify(messages) > 0.5'},
        {"reader": model_entry(classify=0)},
        adapter,
    )
    result = scanner.scan_data({"messages": transcript(audio())})
    assert result.is_complete and result.rule_matches == ["Read"] and adapter.classifications == 1


def test_large_inline_validation_avoids_a_payload_copy_and_checks_deadlines(monkeypatch):
    part = image("data:image/png;base64," + "A" * 2_000_000)
    tracemalloc.start()
    try:
        assert media.media_fault("user", part) is None
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 100_000
    calls = 0

    def deadline():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise QuarryTimeoutError("synthetic media deadline")

    monkeypatch.setattr(media, "raise_if_evaluation_expired", deadline)
    with pytest.raises(QuarryTimeoutError, match="synthetic media deadline"):
        media.media_fault("user", part)


@pytest.mark.asyncio
async def test_many_parts_yield_before_adapter_dispatch(monkeypatch):
    source = Messages.from_openai([{"role": "user", "content": [image()] * 500}])
    monkeypatch.setattr(media.Pacer, "due", lambda self: True)
    visited = []

    async def peer():
        visited.append("peer")

    task = asyncio.create_task(peer())
    await media.require_readable_media(source)
    assert visited == ["peer"]
    await task


def test_false_model_gate_does_not_read_media(tmp_path):
    adapter = ModelAdapter()
    scanner = v2_scanner(
        tmp_path,
        {"Skip": 'false and model("reader").classify(messages) > 0.5'},
        {"reader": model_entry(classify=0)},
        adapter,
    )
    result = scanner.scan_data({"messages": transcript(image("data:text/plain;base64,c2VjcmV0"))})
    assert result.is_complete and result.rule_matches == [] and adapter.classifications == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["score", "classify", "complete", "tokens"])
async def test_bad_media_is_rejected_before_the_cache_boundary(tmp_path, operation, monkeypatch):
    backend, _ = backend_for(tmp_path)

    async def cached(*args, **kwargs):
        pytest.fail("unreadable media reached the cache boundary")

    monkeypatch.setattr(backend, "_call", cached)
    with pytest.raises(InvalidInputError):
        await invoke(
            backend,
            operation,
            Messages.from_openai(transcript(image("data:text/plain;base64,c2VjcmV0"))),
        )


@pytest.mark.parametrize("content", [None, [], 4, ["raw string part"]])
def test_tool_result_structural_faults_still_reject_the_whole_input(content):
    payload = transcript(image(), tool=True)
    payload[1]["content"] = content
    result = compile_policy('rule "Always" condition: true').scan({"messages": payload})
    assert result.completion_state == "invalid"


def surface_config(tmp_path):
    paths = []
    for name, condition in {
        "Text": 'any m in messages where m.content contains "visible"',
        "Read": 'model("reader").classify(messages) > 0.5',
    }.items():
        path = tmp_path / (name + ".rule")
        path.write_text(f'rule "{name}" condition: {condition}')
        paths.append(str(path))
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {"config_version": 2, "rules": paths, "models": {"reader": model_entry(classify=0)}}
        )
    )
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_sdk_guard_fails_closed_with_retained_text_match(tmp_path, asynchronous, monkeypatch):
    adapter = ModelAdapter()
    monkeypatch.setattr(model_inference, "_default_adapters", lambda: {"fake": adapter})
    guard = QuarryGuard(surface_config(tmp_path))
    payload = {"messages": transcript(image("data:text/plain;base64,c2VjcmV0"))}
    with pytest.raises(IncompleteScanError) as failure:
        if asynchronous:
            await guard.scan_async(payload)
        else:
            await asyncio.to_thread(guard.scan, payload)
    result = failure.value.scan_result
    assert result.rule_matches == ["Text"] and result.completion_state == "incomplete"
    assert result.failures[0].category.value == "invalid_input" and adapter.classifications == 0


def test_cli_reports_partial_text_match_and_invalid_media_failure(tmp_path, monkeypatch, capsys):
    adapter = ModelAdapter()
    monkeypatch.setattr(model_inference, "_default_adapters", lambda: {"fake": adapter})
    target = tmp_path / "input.json"
    target.write_text(
        json.dumps({"messages": transcript(image("data:text/plain;base64,c2VjcmV0"))})
    )
    code = scan_command(
        SimpleNamespace(
            config=surface_config(tmp_path),
            target=target,
            no_limits=False,
            recursive=False,
            verbose=False,
            batch=False,
            validate_only=False,
            output_format="json",
            explain=False,
            show_timing=False,
            output=None,
        )
    )
    output = capsys.readouterr().out
    item = json.loads(output)[0]
    assert code == 2 and item["completion_state"] == "incomplete"
    assert item["matches"] == [{"rule": "Text"}]
    assert item["failures"] == [{"category": "invalid_input", "rule_name": "Read"}]
    assert "c2VjcmV0" not in output and adapter.classifications == 0
