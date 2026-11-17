# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Versioned reuse preserves input identity, current coordinates and owned batch exits."""

import asyncio
import gc
import weakref
from dataclasses import replace
from unittest.mock import patch

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.cache.async_cache import AsyncFunctionCache, scan_scope
from quarry.config import LocalTransformConfig, QuarryConfig
from quarry.integrations import model_inference as inference
from quarry.integrations.adapter_message_transforms import apply_adapter_transform
from quarry.integrations.adapter_transforms import KeptTransform
from quarry.integrations.model_usage import model_call_budget
from quarry.integrations.shared_transforms import SharedTransformRecord, shared_record
from quarry.plugin.service import QuarryPlugin
from quarry.safety.limits import SafetyLimits
from quarry.types.adapter_transforms import TransformCacheIdentity, TransformResult, TransformRun
from quarry.types.exceptions import ConfigurationError, LLMError, ModelError
from quarry.types.messages import Messages
from quarry.types.results import Span


@pytest.fixture
def cache(monkeypatch):
    value = AsyncFunctionCache.expiring(max_entries=100_000, ttl_seconds=43_200)
    monkeypatch.setattr(inference, "result_cache", value)
    return value


class Adapter:
    local = True
    transform_parts = ("text", "image_url", "input_audio")
    transform_cache = TransformCacheIdentity("owned.normalizer", "1")

    def __init__(self):
        self.calls = []

    async def transform(self, config, part, *, timeout=None):
        self.calls.append(part)
        await asyncio.sleep(0)
        return part.get("text", "media").upper() + config.settings.get("suffix", "")


def backend(adapter=None, *, settings=None, parts=None, adapter_name="owned"):
    adapter = Adapter() if adapter is None else adapter
    cfg = QuarryConfig(
        config_version=2,
        transforms={"clean": LocalTransformConfig(adapter_name, settings or {}, parts)},
    )
    return inference.ModelInferenceBackend(cfg, adapters={adapter_name: adapter}), adapter


def text(value):
    return {"type": "text", "text": value}


@pytest.mark.parametrize("value", [None, False, 1, "", "x" * 129, "\ud800"])
@pytest.mark.parametrize("field", ["namespace", "version"])
def test_cache_identity_requires_bounded_explicit_unicode_strings(value, field):
    with pytest.raises(ValueError):
        TransformCacheIdentity(**{"namespace": "owned", "version": "1", field: value})


@pytest.mark.asyncio
async def test_equal_versioned_work_is_shared_across_instances_and_detached_configs(cache):
    a, left = backend(settings={"suffix": "!", "other": {"b": 2, "a": 1}})
    b, right = backend(settings={"other": {"a": 1, "b": 2}, "suffix": "!"})
    first = await a.transform_unit("clean", text("hello"))
    second = await b.transform_unit("clean", text("hello"))
    assert first == second and first.text == "HELLO!" and first is not second
    assert len(left.calls) == 1 and not right.calls
    assert isinstance(next(iter(cache.store.values())), SharedTransformRecord)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["namespace", "version", "settings", "parts", "input", "adapter"]
)
async def test_every_semantic_identity_dimension_is_part_of_reuse(cache, change):
    a, left = backend()
    await a.transform_unit("clean", text("hello"))
    right = Adapter()
    kwargs = {}
    if change in {"namespace", "version"}:
        right.transform_cache = replace(right.transform_cache, **{change: "changed"})
    if change == "settings":
        kwargs["settings"] = {"suffix": "!"}
    if change == "parts":
        kwargs["parts"] = ("text",)
    if change == "adapter":
        kwargs["adapter_name"] = "another"
    b, _ = backend(right, **kwargs)
    result = await b.transform_unit("clean", text("other" if change == "input" else "hello"))
    expected = "OTHER" if change == "input" else "HELLO!" if change == "settings" else "HELLO"
    assert result.text == expected
    assert len(left.calls) == len(right.calls) == 1


@pytest.mark.asyncio
async def test_unversioned_message_parts_keep_stateful_order_and_do_not_share_instances(cache):
    class Stateful(Adapter):
        transform_cache = None

        async def transform(self, config, part, *, timeout=None):
            self.calls.append(part)
            return str(len(self.calls))

    a, left = backend(Stateful())
    b, right = backend(Stateful())
    source = Messages.from_openai([{"role": "user", "content": [text("same"), text("same")]}])
    for target in [a, b]:
        result = await apply_adapter_transform(source, target, "clean", _message_value=True)
        assert result[0].content == "12"
    assert len(left.calls) == len(right.calls) == 2


@pytest.mark.asyncio
async def test_invalid_opt_in_is_rejected_before_adapter_dispatch(cache):
    adapter = Adapter()
    adapter.transform_cache = "1"
    target, _ = backend(adapter)
    with pytest.raises(ConfigurationError, match="TransformCacheIdentity"):
        await target.transform_unit("clean", text("hello"))
    assert not adapter.calls


@pytest.mark.asyncio
async def test_reused_relative_ranges_bind_to_current_message_and_part_coordinates(cache):
    a, left = backend()
    b, right = backend()
    first = Messages.from_openai([{"role": "user", "content": "same"}])
    second = Messages.from_openai(
        [{"role": "system", "content": "prefix"}, {"role": "user", "content": "same"}]
    )
    x = await apply_adapter_transform(first, a, "clean", _message_value=True)
    y = await apply_adapter_transform(second[1:], b, "clean", _message_value=True)
    assert x[0].content == y[0].content == "SAME"
    assert [r.to_dict() for r in x.source_ranges(0, 0, 4)] == [
        {"message_index": 0, "start": 0, "end": 4}
    ]
    assert [r.to_dict() for r in y.source_ranges(1, 0, 4)] == [
        {"message_index": 1, "start": 0, "end": 4}
    ]
    assert len(left.calls) == 1 and not right.calls


@pytest.mark.asyncio
async def test_equal_media_replies_get_fresh_current_part_origins(cache):
    a, left = backend()
    b, right = backend()
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    first = Messages.from_openai([{"role": "user", "content": [image]}])
    second = Messages.from_openai(
        [{"role": "user", "content": "prefix"}, {"role": "user", "content": [text("x"), image]}]
    )
    x = await apply_adapter_transform(first, a, "clean")
    y = await apply_adapter_transform(second[1:], b, "clean")
    assert x[0].content == "MEDIA" and y[0].content == "XMEDIA"
    span = y._bind_span(Span(0, 1, 6))
    assert [p.to_dict() for p in span.source_parts] == [{"message_index": 1, "part_index": 1}]
    assert span.source_ranges == () and span.source_unmapped
    assert len(left.calls) == 1 and len(right.calls) == 1  # Only the new text part runs.


@pytest.mark.asyncio
async def test_dsl_batches_are_bounded_ordered_and_deduplicate_equal_versioned_parts(cache):
    class Overlap(Adapter):
        active = peak = 0

        async def transform(self, config, part, *, timeout=None):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                return await super().transform(config, part, timeout=timeout)
            finally:
                self.active -= 1

    target, adapter = backend(Overlap())
    parts = [text(str(i % 20)) for i in range(100)]
    source = Messages.from_openai([{"role": "user", "content": parts}])
    result = await apply_adapter_transform(source, target, "clean", _message_value=True)
    assert result.to_openai_messages() == [{"role": "user", "content": parts}]
    assert len(adapter.calls) == 20 and 1 < adapter.peak <= 16 and adapter.active == 0
    assert source.to_openai_messages() == [{"role": "user", "content": parts}]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["clear", "evict", "refuse", "expire"])
async def test_process_retention_changes_only_cause_recomputation(cache, action):
    target, adapter = backend()
    expected = await target.transform_unit("clean", text("hello"))
    if action == "clear":
        cache.clear()
    elif action == "evict":
        cache.configure(max_entries=1, ttl_seconds=60)
        await target.transform_unit("clean", text("other"))
    elif action == "refuse":
        cache.configure(max_entries=1, ttl_seconds=60, max_bytes=1)
    else:
        cache.configure(max_entries=1, ttl_seconds=0)
    assert await target.transform_unit("clean", text("hello")) == expected
    assert len(adapter.calls) == (3 if action == "evict" else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption", ["object", "format", "key", "checksum", "unused", "unicode", "source", "copy"]
)
async def test_corrupt_shared_records_are_invalidated_and_recomputed(cache, corruption):
    target, adapter = backend()
    expected = await target.transform_unit("clean", text("hello"))
    key, record = next(iter(cache.store.items()))
    if corruption == "object":
        bad = object()
    elif corruption == "format":
        bad = replace(record, format_version=True)
    elif corruption == "key":
        bad = replace(record, key=b"other")
    elif corruption == "checksum":
        bad = replace(record, checksum=b"x" * 32)
    else:
        kept = {
            "unused": KeptTransform("unused", (), ()),
            "unicode": KeptTransform("\ud800", (TransformRun(0, 1),), (False,)),
            "source": KeptTransform("", (TransformRun(0, 9, 0, 9, "linear"),), (True,)),
            "copy": KeptTransform("", (TransformRun(0, 5, 0, 5, "whole"),), (True,)),
        }[corruption]
        bad = shared_record(key, kept)
    with cache.lock:
        cache.store[key] = bad
    assert await target.transform_unit("clean", text("hello")) == expected
    assert await target.transform_unit("clean", text("hello")) == expected
    assert len(adapter.calls) == 2


@pytest.mark.asyncio
async def test_corrupt_cached_offsets_do_not_invoke_user_index_hooks(cache):
    class Offset:
        touched = False

        def __index__(self):
            self.touched = True
            raise AssertionError("cached offset hook invoked")

    target, adapter = backend()
    await target.transform_unit("clean", text("hello"))
    key, record = next(iter(cache.store.items()))
    offset = Offset()
    run = TransformRun(0, 5, offset, 5, "linear")
    with cache.lock:
        cache.store[key] = replace(record, kept=KeptTransform("", (run,), (True,)))
    assert (await target.transform_unit("clean", text("hello"))).text == "HELLO"
    assert not offset.touched and len(adapter.calls) == 2


@pytest.mark.asyncio
async def test_partial_batch_failure_retains_only_valid_success_and_drains_sibling(cache):
    class Partial(Adapter):
        good = asyncio.Event()
        stopped = False
        fail = True

        async def transform(self, config, part, *, timeout=None):
            self.calls.append(part)
            if part["text"] == "good":
                self.good.set()
                return "GOOD"
            if self.fail and part["text"] == "bad":
                await self.good.wait()
                raise ValueError("owned batch failure")
            if self.fail:
                try:
                    await asyncio.Event().wait()
                finally:
                    self.stopped = True
            return part["text"].upper()

    target, adapter = backend(Partial())
    units = [text(v) for v in ["good", "bad", "slow"]]
    with pytest.raises(ModelError):
        await target.transform_units("clean", units)
    assert adapter.stopped
    adapter.fail = False
    assert [r.text for r in await target.transform_units("clean", units)] == ["GOOD", "BAD", "SLOW"]
    assert sum(p["text"] == "good" for p in adapter.calls) == 1
    assert sum(p["text"] == "bad" for p in adapter.calls) == 2


@pytest.mark.asyncio
async def test_cancelling_one_batch_does_not_cancel_another_scans_shared_work(cache):
    class Blocked(Adapter):
        started = asyncio.Event()
        release = asyncio.Event()

        async def transform(self, config, part, *, timeout=None):
            self.calls.append(part)
            self.started.set()
            await self.release.wait()
            return "DONE"

    a, left = backend(Blocked())
    b, right = backend(Blocked())

    async def run(target):
        with scan_scope():
            return await target.transform_units("clean", [text("same")])

    first = asyncio.create_task(run(a))
    await left.started.wait()
    second = asyncio.create_task(run(b))
    while next(iter(cache.store.values())).waiters != 2:
        await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not second.done()
    left.release.set()
    assert [r.text for r in await second] == ["DONE"]
    assert len(left.calls) == 1 and not right.calls


@pytest.mark.asyncio
async def test_shared_records_do_not_retain_the_adapter_instance_without_gc(cache):
    target, adapter = backend()
    ref = weakref.ref(adapter)
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        assert (await target.transform_unit("clean", text("hello"))).text == "HELLO"
        del target, adapter
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert ref() is None and len(cache.store) == 1
    finally:
        if enabled:
            gc.enable()


@pytest.mark.asyncio
async def test_batch_size_and_aggregate_input_are_checked_before_dispatch(cache):
    target, adapter = backend()
    for units in [[text("x")] * 17, [text("x" * 100_000)] * 11]:
        with pytest.raises(ModelError, match="batch exceeds"):
            await target.transform_units("clean", units)
    assert not adapter.calls


@pytest.mark.asyncio
async def test_dsl_splits_large_media_windows_instead_of_rejecting_valid_inputs(cache):
    target, adapter = backend()
    image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + "A" * 100_000},
    }
    source = Messages.from_openai([{"role": "user", "content": [image] * 16}])
    result = await apply_adapter_transform(source, target, "clean", _message_value=True)
    assert result[0].content == "MEDIA" * 16 and len(adapter.calls) == 1
    assert source.to_openai_messages() == [{"role": "user", "content": [image] * 16}]


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["detail", "audio_format"])
async def test_full_media_wire_identity_controls_reuse(cache, field):
    target, adapter = backend()
    if field == "detail":
        units = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "detail": v}}
            for v in ["low", "high"]
        ]
    else:
        units = [
            {"type": "input_audio", "input_audio": {"data": "AA==", "format": v}}
            for v in ["wav", "mp3"]
        ]
    for unit in units:
        assert (await target.transform_unit("clean", unit)).text == "MEDIA"
    assert len(adapter.calls) == 2


@pytest.mark.asyncio
async def test_batch_results_preserve_input_order_when_completion_order_differs(cache):
    class Reversed(Adapter):
        last = asyncio.Event()

        async def transform(self, config, part, *, timeout=None):
            self.calls.append(part)
            if part["text"] == "first":
                await self.last.wait()
            else:
                self.last.set()
            return part["text"].upper()

    target, _ = backend(Reversed())
    assert [
        r.text for r in await target.transform_units("clean", [text("first"), text("last")])
    ] == ["FIRST", "LAST"]


@pytest.mark.asyncio
async def test_invalid_adapter_output_is_never_shared_and_later_success_recovers(cache):
    class Invalid(Adapter):
        fail = True

        async def transform(self, config, part, *, timeout=None):
            self.calls.append(part)
            return object() if self.fail else "RECOVERED"

    target, adapter = backend(Invalid())
    with pytest.raises(ModelError, match="adapter must return"):
        await target.transform_unit("clean", text("hello"))
    assert not cache.store
    adapter.fail = False
    assert (await target.transform_unit("clean", text("hello"))).text == "RECOVERED"
    assert len(adapter.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["characters", "segments"])
async def test_batch_aggregate_output_and_map_budgets_apply_before_results_escape(cache, limit):
    class Large(Adapter):
        async def transform(self, config, part, *, timeout=None):
            if limit == "characters":
                return part["text"] * 3
            return TransformResult(
                part["text"], tuple(TransformRun(i, i + 1, i, i + 1, "linear") for i in range(1024))
            )

    target, _ = backend(Large())
    units = (
        [text(str(i) + "x" * 109_999) for i in range(9)]
        if limit == "characters"
        else [text(str(i) + "x" * 1023) for i in range(9)]
    )
    with pytest.raises(ModelError, match="batch exceeds its output limit"):
        await target.transform_units("clean", units)


@pytest.mark.asyncio
async def test_last_batch_waiter_cancellation_drains_provider_and_allows_retry(cache):
    class Blocked(Adapter):
        started = asyncio.Event()
        stopped = asyncio.Event()
        block = True

        async def transform(self, config, part, *, timeout=None):
            self.calls.append(part)
            self.started.set()
            if self.block:
                try:
                    await asyncio.Event().wait()
                finally:
                    self.stopped.set()
            return "RECOVERED"

    target, adapter = backend(Blocked())
    task = asyncio.create_task(target.transform_units("clean", [text("same")]))
    await adapter.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert adapter.stopped.is_set() and not cache.store
    adapter.block = False
    assert [r.text for r in await target.transform_units("clean", [text("same")])] == ["RECOVERED"]
    assert len(adapter.calls) == 2


@pytest.mark.asyncio
async def test_single_unit_host_override_is_preserved_inside_versioned_batches(cache):
    target, adapter = backend()

    async def override(name, unit):
        return unit["text"] + "!"

    target.transform_unit = override
    source = Messages.from_openai([{"role": "user", "content": [text("first"), text("next")]}])
    result = await apply_adapter_transform(source, target, "clean", _message_value=True)
    assert result[0].content == "first!next!" and not adapter.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("part_type", ["text", "image_url", "input_audio"])
async def test_batch_unicode_admission_still_precedes_hashing_and_dispatch(
    cache, monkeypatch, part_type
):
    target, adapter = backend()

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid transform input was hashed")

    monkeypatch.setattr(inference, "input_key", forbidden)
    unit = {
        "text": text("\ud800"),
        "image_url": {"type": "image_url", "image_url": {"url": "\ud800"}},
        "input_audio": {"type": "input_audio", "input_audio": {"data": "\ud800", "format": "wav"}},
    }[part_type]
    with pytest.raises(ModelError, match="valid Unicode"):
        await target.transform_units("clean", [unit])
    assert not adapter.calls and not cache.store


@pytest.mark.asyncio
async def test_persistent_corrupt_reader_is_bounded_and_does_not_remove_newer_record(
    cache, monkeypatch
):
    target, adapter = backend()
    assert (await target.transform_unit("clean", text("hello"))).text == "HELLO"
    key, valid = next(iter(cache.store.items()))
    original = cache.call
    reads = []

    async def broken(*args, **kwargs):
        reads.append(1)
        return object()

    monkeypatch.setattr(cache, "call", broken)
    with pytest.raises(ModelError, match="shared transform cache returned invalid data"):
        await target.transform_unit("clean", text("hello"))
    assert reads == [1, 1] and cache.store[key] is valid
    monkeypatch.setattr(cache, "call", original)
    assert (await target.transform_unit("clean", text("hello"))).text == "HELLO"
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("prepared", [False, True])
async def test_actual_policy_scans_share_versioned_work_and_preserve_prepared_bytes(
    cache, prepared, monkeypatch
):
    async def paid_work(*args, **kwargs):
        raise AssertionError("local transform entered model usage")

    monkeypatch.setattr(inference.ModelUsage, "run", paid_work)
    target, adapter = backend()
    config = replace(target.config, safety_limits=SafetyLimits(rule_timeout=None))
    policy = compile_policy(
        'rule "R" condition: messages.transform("clean")[0].content contains "HELLO"',
        runtime_config=config,
        model_inference_backend=target,
    )
    raw = policy.to_bytes()
    if prepared:

        def forbidden(*args, **kwargs):
            raise AssertionError("prepared transform policy recompiled")

        with (
            patch("quarry.dsl.parser.DSLParser.parse", forbidden),
            patch("quarry.scanner.compile_rule", forbidden),
            patch("quarry.engine.compiler.compile_rule", forbidden),
            patch("quarry.policy.compile_policy", forbidden),
        ):
            policy = load_compiled_policy(
                raw, runtime_config=config, model_inference_backend=target
            )
            assert policy.to_bytes() == raw
    for _ in range(2):
        result = await policy.scan_async({"messages": [{"role": "user", "content": "hello"}]})
        assert result.is_complete and result.rule_matches == ["R"]
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_actual_plugin_instances_share_pure_work_without_model_budget_or_usage(
    cache, monkeypatch
):
    async def paid_work(*args, **kwargs):
        raise AssertionError("local transform entered model usage")

    monkeypatch.setattr(inference.ModelUsage, "run", paid_work)
    adapters = []

    def registered():
        adapter = Adapter()
        adapters.append(adapter)
        return {"owned": adapter}

    monkeypatch.setattr(inference, "_default_adapters", registered)
    config = QuarryConfig(
        config_version=2,
        transforms={"clean": LocalTransformConfig("owned")},
        safety_limits=SafetyLimits(rule_timeout=None),
    )
    for _ in range(2):
        service = QuarryPlugin(config)
        result = await service.scan(
            rule_text='rule "R" condition: messages.transform("clean")[0].content contains "HELLO"',
            conversation={"messages": [{"role": "user", "content": "hello"}]},
            explain=True,
        )
        assert result["matched"] and result["completion_state"] == "complete"
    assert len(adapters) >= 2 and sum(len(adapter.calls) for adapter in adapters) == 1
    target, _ = backend()
    with model_call_budget(1) as budget:
        assert [r.text for r in await target.transform_units("clean", [text("hello")])] == ["HELLO"]
        assert budget.calls_by_alias == {}


@pytest.mark.asyncio
async def test_timed_out_batch_cleans_up_and_retries_without_failed_cache_entry(cache):
    class Timed(Adapter):
        stopped = False
        block = True

        async def transform(self, config, part, *, timeout=None):
            self.calls.append(part)
            if self.block:
                try:
                    await asyncio.Event().wait()
                finally:
                    self.stopped = True
            return "RECOVERED"

    target, adapter = backend(Timed())
    with pytest.raises(ModelError):
        await target.transform_units("clean", [text("same")], timeout=0.01)
    assert adapter.stopped and not cache.store
    adapter.block = False
    assert [r.text for r in await target.transform_units("clean", [text("same")])] == ["RECOVERED"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [ModelError, LLMError])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_versioned_batches_sanitize_typed_host_errors_before_retry(
    cache, error_type, asynchronous
):
    class Private(Adapter):
        fail = True

        def transform(self, config, part, *, timeout=None):
            self.calls.append(part)

            def value():
                if self.fail:
                    raise error_type(
                        "SYNTHETIC_PRIVATE_BODY", {"payload": "SYNTHETIC_PRIVATE_BODY"}
                    )
                return "RECOVERED"

            if asynchronous:

                async def later():
                    await asyncio.sleep(0)
                    return value()

                return later()
            return value()

    target, adapter = backend(Private())
    for _ in range(2):
        with pytest.raises((ModelError, LLMError)) as caught:
            await target.transform_units("clean", [text("same")])
        assert "SYNTHETIC_PRIVATE_BODY" not in str(caught.value)
        assert "SYNTHETIC_PRIVATE_BODY" not in repr(caught.value.details)
        assert not cache.store
    adapter.fail = False
    assert [r.text for r in await target.transform_units("clean", [text("same")])] == ["RECOVERED"]
    assert len(adapter.calls) == 3


@pytest.mark.asyncio
async def test_repeated_batch_cancellation_waits_for_every_override_cleanup(cache):
    target, adapter = backend()
    started = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    entered = []
    exited = []
    cleanups = []

    async def override(name, unit):
        entered.append(unit["text"])
        if len(entered) == 2:
            started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanups.append(unit["text"])
            if len(cleanups) == 2:
                cleaning.set()
            await release.wait()
            exited.append(unit["text"])

    target.transform_unit = override
    task = asyncio.create_task(target.transform_units("clean", [text("one"), text("two")]))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    await asyncio.wait_for(cleaning.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not exited
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(exited) == ["one", "two"]
    assert not adapter.calls and not cache.store
