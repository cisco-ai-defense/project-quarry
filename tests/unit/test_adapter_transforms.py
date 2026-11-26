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

"""Host transform configuration, result validation and execution policy."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

import quarry.integrations.model_inference as model_inference_module
from quarry.cache.async_cache import _result_bytes, result_cache, scan_scope
from quarry.config import LocalTransformConfig, ModelTransformConfig, QuarryConfig
from quarry.integrations.adapter_transforms import keep_transform, read_transform
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.integrations.model_usage import model_call_budget
from quarry.types.adapter_transforms import DecodeLimitError, TransformResult, TransformRun
from quarry.types.exceptions import MODEL_REFUSED, ConfigurationError, LLMError, ModelError
from tests.unit.model_fakes import model_entry


class TextAdapter:
    local = True
    transform_parts = ("text",)

    def __init__(self):
        self.calls = []
        self.result = None

    async def transform(self, config, part, *, timeout=None):
        self.calls.append((config, dict(part), timeout))
        await asyncio.sleep(0)
        return part["text"].upper() if self.result is None else self.result


def backend(tmp_path, adapter, *, model=False, **definitions):
    adapter.local = not (model or any("model" in item for item in definitions.values()))
    transforms = definitions or {"convert": {"model": "m"} if model else {"adapter": "fake"}}
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"m": model_entry(), "other": model_entry()} if model else {},
            "transforms": transforms,
        },
        tmp_path,
        require_rules=False,
    )
    return ModelInferenceBackend(config, adapters={"fake": adapter})


def text(value):
    return {"type": "text", "text": value}


def test_transform_settings_are_detached_nested_json_and_remain_frozen_after_replace():
    settings = {"steps": [{"enabled": True, "labels": ["first"]}], "count": 2}
    definition = LocalTransformConfig("fake", settings, parts=("text",))
    settings["steps"][0]["labels"].append("changed")
    assert definition.settings["steps"][0]["labels"] == ("first",)
    with pytest.raises(TypeError):
        definition.settings["steps"][0]["enabled"] = False
    copied = replace(definition)
    assert copied.settings == definition.settings
    assert copied is not definition


@pytest.mark.parametrize(
    "settings",
    [
        None,
        [],
        {1: "bad"},
        {"number": float("inf")},
        {"number": float("nan")},
        {"number": 2**64},
        {"value": object()},
        {"value": "\ud800"},
        {"value": "x" * 140_001},
        {"value": "é" * 100_000},
        {"value": list(range(10_001))},
    ],
)
def test_non_json_and_excessive_settings_are_rejected(settings):
    with pytest.raises((TypeError, ValueError)):
        LocalTransformConfig("fake", settings)


def test_recursive_and_deep_settings_fail_before_encoding():
    cycle = {}
    cycle["self"] = cycle
    with pytest.raises(ValueError, match="cycles"):
        LocalTransformConfig("fake", cycle)
    deep = {}
    for _ in range(18):
        deep = {"next": deep}
    with pytest.raises(ValueError, match="nesting"):
        LocalTransformConfig("fake", deep)


@pytest.mark.parametrize("parts", [[], "text", ["text", "text"], ["unknown"], [True], [None]])
def test_part_selection_is_strict(parts):
    with pytest.raises(ValueError):
        ModelTransformConfig("m", parts)


def test_config_parses_adapter_and_model_forms_and_checks_aliases(tmp_path):
    configured = backend(
        tmp_path,
        TextAdapter(),
        model=True,
        local={"adapter": "fake", "settings": {"mode": "synthetic"}, "parts": ["text"]},
        remote={"model": "m", "parts": ["text"]},
    ).config
    assert type(configured.transforms["local"]) is LocalTransformConfig
    assert type(configured.transforms["remote"]) is ModelTransformConfig
    with pytest.raises(ConfigurationError, match="unconfigured model"):
        QuarryConfig(transforms={"bad": ModelTransformConfig("missing")})
    for definition in (
        {"adapter": "fake", "model": "m"},
        {"model": "m", "settings": {}},
        {"adapter": "fake", "operations": [{"op": "strip"}]},
        {"adapter": "fake", "extra": True},
    ):
        with pytest.raises(ConfigurationError):
            backend(tmp_path, TextAdapter(), model=True, bad=definition)


def test_kept_transform_reconstructs_copies_and_preserves_explicit_mapping():
    unit = text("prefix BODY suffix")
    result = TransformResult(
        "prefix ITEM suffix",
        (
            TransformRun(0, 7, 0, 7, "linear"),
            TransformRun(7, 11, 7, 11, "whole"),
            TransformRun(11, 18, 11, 18, "linear"),
        ),
    )
    kept = keep_transform(result, unit, "convert")
    assert kept.inserted == "ITEM"
    assert kept.copied == (True, False, True)
    assert read_transform(kept, unit) == result
    identity = keep_transform(unit["text"], unit, "convert")
    assert identity.inserted == ""
    assert read_transform(identity, unit).text == unit["text"]
    # Same text with an explicit whole mapping must keep that mapping.
    explicit = TransformResult("prefix BODY suffix", (TransformRun(0, 18, 0, 18),))
    assert read_transform(keep_transform(explicit, unit, "convert"), unit) == explicit


@pytest.mark.parametrize(
    "reply",
    [
        None,
        {},
        {"text": "x", "runs": []},
        2,
        TransformResult(2, ()),
        TransformResult("x", []),
        TransformResult("x", ()),
        TransformResult("x", (TransformRun(0, 2),)),
        TransformResult("xy", (TransformRun(1, 2),)),
        TransformResult("xy", (TransformRun(0, 1), TransformRun(0, 2))),
        TransformResult("x", (TransformRun(False, 1),)),
        TransformResult("x", (TransformRun(0, True),)),
        TransformResult("x", (TransformRun(0, 1, None, 1),)),
        TransformResult("x", (TransformRun(0, 1, -1, 1),)),
        TransformResult("x", (TransformRun(0, 1, 0, 8),)),
        TransformResult("x", (TransformRun(0, 1, 0, 2, "linear"),)),
        TransformResult("x", (TransformRun(0, 1, kind="linear"),)),
        TransformResult("x", (TransformRun(0, 1, kind="invalid"),)),
        TransformResult("\ud800", (TransformRun(0, 1),)),
        "x" * 4_200,
    ],
)
def test_invalid_results_are_structured_faults_with_no_refusal(reply):
    with pytest.raises(ModelError) as error:
        keep_transform(reply, text("abc"), "convert")
    assert error.value.details == {"code": "invalid_output", "transform": "convert"}


def test_media_output_never_claims_offsets_into_encoded_payload():
    unit = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    result = read_transform(keep_transform("synthetic label", unit, "convert"), unit)
    assert result.runs == (TransformRun(0, 15),)
    with pytest.raises(ModelError, match="input text"):
        keep_transform(TransformResult("x", (TransformRun(0, 1, 0, 1),)), unit, "convert")


@pytest.mark.asyncio
async def test_local_transform_cache_and_model_budget_are_separate(tmp_path):
    adapter = TextAdapter()
    service = backend(tmp_path, adapter)
    with model_call_budget(1) as budget:
        assert (await service.transform_unit("convert", text("first"))).text == "FIRST"
        assert (await service.transform_unit("convert", text("second"))).text == "SECOND"
        assert (await service.transform_unit("convert", text("first"))).text == "FIRST"
        assert budget.calls_by_alias == {}
    assert len(adapter.calls) == 2
    assert type(adapter.calls[0][0]) is LocalTransformConfig


@pytest.mark.asyncio
async def test_model_transform_counts_actual_attempts_and_shares_identical_alias_requests(tmp_path):
    adapter = TextAdapter()
    service = backend(
        tmp_path,
        adapter,
        model=True,
        convert={"model": "m"},
        alias={"model": "other"},
    )
    with model_call_budget(1) as budget:
        await service.transform_unit("convert", text("first"))
        await service.transform_unit("convert", text("first"))
        await service.transform_unit("alias", text("first"))
        assert budget.calls_by_alias == {"m": 1}
        with pytest.raises(LLMError) as error:
            await service.transform_unit("convert", text("second"))
        assert error.value.details["code"] == "model_call_limit"
        await service.transform_unit("alias", text("second"))
        assert budget.calls_by_alias == {"m": 1, "other": 1}
    assert len(adapter.calls) == 2
    assert adapter.calls[0][0].model == "physical-model"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [False, True])
async def test_invalid_adapter_results_never_enter_cache(tmp_path, model):
    adapter = TextAdapter()
    adapter.result = TransformResult("bad", ())
    service = backend(tmp_path, adapter, model=model)
    for _ in range(2):
        with pytest.raises(ModelError) as error:
            await service.transform_unit("convert", text("input"))
        assert error.value.details["code"] == "invalid_output"
    adapter.result = "valid"
    assert (await service.transform_unit("convert", text("input"))).text == "valid"
    assert len(adapter.calls) == 3


@pytest.mark.asyncio
async def test_adapter_mutation_cannot_change_key_or_validation_source(tmp_path):
    class MutatingAdapter(TextAdapter):
        def transform(self, config, part, *, timeout=None):
            self.calls.append(part["text"])
            part["text"] = "changed"
            return TransformResult("ABC", (TransformRun(0, 3, 0, 3, "linear"),))

    adapter = MutatingAdapter()
    service = backend(tmp_path, adapter)
    unit = text("abc")
    assert (await service.transform_unit("convert", unit)).text == "ABC"
    assert (await service.transform_unit("convert", unit)).text == "ABC"
    assert unit == text("abc")
    assert adapter.calls == ["abc"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [False, True])
async def test_decoder_limits_are_faults_and_are_not_kept(tmp_path, model):
    class LimitedAdapter(TextAdapter):
        def transform(self, config, part, *, timeout=None):
            self.calls.append(part)
            raise DecodeLimitError()

    adapter = LimitedAdapter()
    service = backend(tmp_path, adapter, model=model)
    for _ in range(2):
        with pytest.raises(ModelError) as error:
            await service.transform_unit("convert", text("input"))
        assert error.value.details == {"code": "decode_limit", "transform": "convert"}
    assert len(adapter.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [False, True])
async def test_async_transform_timeout_cancels_final_waiter_and_does_not_keep_result(
    tmp_path, model, monkeypatch
):
    loop = asyncio.get_running_loop()
    waiter = asyncio.current_task()
    waiter_timeout = asyncio.timeout(None)
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class ControlledTimeouts:
        def timeout(self, seconds):
            if asyncio.current_task() is waiter:
                return waiter_timeout
            return asyncio.timeout(seconds)

        def __getattr__(self, name):
            return getattr(asyncio, name)

    monkeypatch.setattr(model_inference_module, "asyncio", ControlledTimeouts())

    def expire_waiter():
        entered.set()
        waiter_timeout.reschedule(loop.time())

    class WaitingAdapter(TextAdapter):
        async def transform(self, config, part, *, timeout=None):
            # Start the caller's deadline after the adapter has started.
            loop.call_soon_threadsafe(expire_waiter)
            try:
                await asyncio.Event().wait()
            finally:
                loop.call_soon_threadsafe(cancelled.set)

    service = backend(tmp_path, WaitingAdapter(), model=model)
    async with asyncio.timeout(5):
        with pytest.raises(ModelError) as error:
            await service.transform_unit("convert", text("input"), timeout=60)
        assert error.value.details["code"] == "timeout"
        assert entered.is_set()
        await cancelled.wait()


@pytest.mark.asyncio
async def test_capabilities_and_input_shapes_fail_before_any_adapter_call(tmp_path):
    adapter = TextAdapter()
    service = backend(tmp_path, adapter, convert={"adapter": "fake", "parts": ["input_audio"]})
    with pytest.raises(ConfigurationError, match="cannot read"):
        service.require_transforms({"convert"})
    service = backend(tmp_path, adapter)
    for unit in (
        {"type": "text", "text": "x", "extra": 1},
        text("x" * 1_000_001),
        {"type": "image_url", "image_url": {"url": "file:///unused"}},
    ):
        with pytest.raises(ModelError):
            await service.transform_unit("convert", unit)
    adapter.transform_parts = None
    with pytest.raises(ConfigurationError, match="declare"):
        service.require_transforms({"convert"})
    assert adapter.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, same_scan, expected_calls",
    [
        (True, False, 1),
        (True, True, 1),
        (False, False, 2),
        (False, True, 1),
    ],
)
async def test_pending_work_follows_model_sharing_and_local_scan_ownership(
    tmp_path, model, same_scan, expected_calls, monkeypatch
):
    attached = asyncio.Event()
    waiter_count = 0
    original_wait = result_cache._wait

    async def wait(pending):
        nonlocal waiter_count
        waiter_count += 1
        if waiter_count == 2:
            attached.set()
        return await original_wait(pending)

    monkeypatch.setattr(result_cache, "_wait", wait)
    first_started = asyncio.Event()
    all_started = asyncio.Event()
    release = asyncio.Event()

    class SharedAdapter(TextAdapter):
        async def transform(self, config, part, *, timeout=None):
            self.calls.append(part)
            first_started.set()
            if len(self.calls) == expected_calls:
                all_started.set()
            await release.wait()
            return "result"

    adapter = SharedAdapter()
    service = backend(tmp_path, adapter, model=model)
    with scan_scope() as first_scan:
        first = asyncio.create_task(service.transform_unit("convert", text("same")))
    await asyncio.wait_for(first_started.wait(), 1)
    with scan_scope(first_scan if same_scan else None):
        second = asyncio.create_task(service.transform_unit("convert", text("same")))
    # An explicit event confirms the expected number of physical calls before release.
    await asyncio.wait_for(all_started.wait(), 1)
    await asyncio.wait_for(attached.wait(), 1)
    release.set()
    results = await asyncio.gather(first, second)
    assert [result.text for result in results] == ["result", "result"]
    assert len(adapter.calls) == expected_calls
    with scan_scope():
        assert (await service.transform_unit("convert", text("same"))).text == "result"
    assert len(adapter.calls) == expected_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [False, True])
async def test_transform_provider_refusal_is_an_invalid_result_fault(tmp_path, model):
    class RefusingAdapter(TextAdapter):
        async def transform(self, config, part, *, timeout=None):
            raise LLMError("not a transform", {"code": MODEL_REFUSED})

    service = backend(tmp_path, RefusingAdapter(), model=model)
    with pytest.raises(ModelError) as error:
        await service.transform_unit("convert", text("input"))
    assert error.value.details == {"code": "invalid_output", "transform": "convert"}


@pytest.mark.asyncio
async def test_local_settings_and_adapter_instances_keep_distinct_cache_identities(tmp_path):
    class ConfiguredAdapter(TextAdapter):
        def transform(self, config, part, *, timeout=None):
            self.calls.append(part)
            return config.settings["prefix"] + part["text"]

    adapter = ConfiguredAdapter()
    service = backend(
        tmp_path,
        adapter,
        first={"adapter": "fake", "settings": {"prefix": "A"}},
        second={"adapter": "fake", "settings": {"prefix": "B"}},
    )
    assert (await service.transform_unit("first", text("input"))).text == "Ainput"
    assert (await service.transform_unit("second", text("input"))).text == "Binput"
    assert len(adapter.calls) == 2
    replacement = ConfiguredAdapter()
    service.adapters["fake"] = replacement
    assert (await service.transform_unit("first", text("input"))).text == "Ainput"
    assert len(replacement.calls) == 1


@pytest.mark.asyncio
async def test_expired_deadline_prevents_even_a_cached_transform_read(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from quarry.safety import timer
    from quarry.safety.timer import QuarryTimeoutError, scan_deadline

    adapter = TextAdapter()
    service = backend(tmp_path, adapter)
    await service.transform_unit("convert", text("input"))
    now = timer.time.monotonic()
    monkeypatch.setattr(timer, "time", SimpleNamespace(monotonic=lambda: now))
    with scan_deadline(10) as deadline:
        assert deadline is not None
        now = deadline.expires_at
        with pytest.raises(QuarryTimeoutError):
            await service.transform_unit("convert", text("input"))
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_model_retry_gets_fresh_part_and_spends_another_attempt(tmp_path, monkeypatch):
    from quarry.integrations import model_usage

    class RetryingAdapter(TextAdapter):
        def transform(self, config, part, *, timeout=None):
            self.calls.append(dict(part))
            if len(self.calls) == 1:
                part["text"] = "mutated by first attempt"
                raise LLMError("try again", {"code": "connection"})
            return part["text"].upper()

    async def no_delay(seconds):
        pass

    monkeypatch.setattr(model_usage, "_sleep", no_delay)
    adapter = RetryingAdapter()
    service = backend(tmp_path, adapter, model=True)
    with model_call_budget(2) as budget:
        assert (
            await service.transform_unit("convert", text("original"), timeout=1.0)
        ).text == "ORIGINAL"
        assert budget.calls_by_alias == {"m": 2}
    assert adapter.calls == [text("original"), text("original")]


def test_result_run_and_output_limits_are_checked_before_retention():
    reply = TransformResult("x" * 8_193, tuple(TransformRun(i, i + 1) for i in range(8_193)))
    with pytest.raises(ModelError, match="bounded tuple"):
        keep_transform(reply, text("a" * 2_000), "convert")
    with pytest.raises(ModelError, match="character limit"):
        keep_transform("x" * 2_000_001, {"type": "image_url"}, "convert")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "detail": "low"}},
        {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
    ],
)
async def test_media_parts_reach_registered_adapter_without_network_or_source_offsets(
    tmp_path, part
):
    class MediaAdapter(TextAdapter):
        transform_parts = ("image_url", "input_audio")

        def transform(self, config, unit, *, timeout=None):
            self.calls.append(unit)
            return "synthetic transcript"

    adapter = MediaAdapter()
    service = backend(tmp_path, adapter)
    result = await service.transform_unit("convert", part)
    assert result.text == "synthetic transcript"
    assert result.runs == (TransformRun(0, 20),)
    assert adapter.calls == [part]
    assert adapter.calls[0] is not part


def test_byte_accounting_counts_kept_new_text_without_retaining_copied_input():
    unit = text("x" * 100_000)
    identity = keep_transform(unit["text"], unit, "convert")
    assert _result_bytes(identity, 1_000_000) < 2_000
    changed = keep_transform("y" * 100_000, unit, "convert")
    assert _result_bytes(changed, 1_000_000) >= 100_000
