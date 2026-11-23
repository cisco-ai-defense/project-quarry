# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Whole-request ownership, off-loop work, bounded artifact reuse and exact framing."""

import asyncio
import gc
import subprocess
import sys
import threading
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from quarry.integrations import local_tokenizer as local
from quarry.integrations import local_tokenizer_worker as worker
from quarry.types.exceptions import (
    LocalTokenizerInvalidInputError,
    LocalTokenizerLimitError,
    LocalTokenizerUnavailableError,
)
from quarry.types.local_tokenizer import LocalTokenizerLimits, TokenFramingMargins
from tests.unit.test_local_tokenizer_artifact import artifact, synthetic

MESSAGES = [{"role": "user", "content": "hello world"}, {"role": "assistant", "content": "é"}]
TEMPLATE = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"


def test_constructor_never_imports_optional_libraries_or_opens_an_artifact():
    script = """
import sys
class NoOptional:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'tokenizers', 'jinja2'}:
            raise AssertionError('eager optional import')
sys.meta_path.insert(0, NoOptional())
from quarry.integrations.local_tokenizer import LocalTokenizerAdapter
adapter = LocalTokenizerAdapter()
assert adapter.report()['pending'] == 0
assert adapter._owner._executor is None
adapter.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "tools",
    [
        [],
        [{"type": "function", "function": {"name": "synthetic", "parameters": {"type": "object"}}}],
    ],
)
async def test_whole_request_exact_rendering_and_one_framing_addition(tmp_path, tools):
    reference = synthetic()
    template = TEMPLATE + "{{ tools|tojson }}"
    cfg = artifact(
        tmp_path, reference, options={"template": template, "framing": TokenFramingMargins(7, 3)}
    )
    rendered = (
        ImmutableSandboxedEnvironment()
        .from_string(template)
        .render(messages=MESSAGES, tools=tools, add_generation_prompt=False)
    )
    expected = len(reference.encode(rendered, add_special_tokens=False).ids) + 7 + 3 * len(MESSAGES)
    adapter = local.LocalTokenizerAdapter()
    try:
        assert await adapter.count_request(cfg, MESSAGES, tools) == expected
        encoded = await adapter.encode_text(cfg, "hello world")
        assert encoded.token_ids == (1, 2)  # Framing is never fabricated as IDs.
        assert adapter.report()["pending"] == 0
    finally:
        adapter.close()


async def test_snapshot_render_load_and_encode_all_execute_off_the_caller_loop(
    tmp_path, monkeypatch
):
    cfg = artifact(tmp_path, options={"template": TEMPLATE})
    caller, threads = threading.get_ident(), []
    for name in ("snapshot_request", "render_local_request", "load_local_tokenizer"):
        original = getattr(local, name)

        def observed(*args, _original=original, **kwargs):
            threads.append(threading.get_ident())
            return _original(*args, **kwargs)

        monkeypatch.setattr(local, name, observed)
    adapter = local.LocalTokenizerAdapter()
    try:
        assert await adapter.count_request(cfg, MESSAGES, []) > 0
        assert len(threads) == 3 and len(set(threads)) == 1 and threads[0] != caller
    finally:
        adapter.close()


async def test_cancelled_render_keeps_its_slot_and_resources_until_native_termination(
    tmp_path, monkeypatch
):
    cfg = artifact(tmp_path, options={"template": TEMPLATE})
    entered, release = threading.Event(), threading.Event()
    original = local.render_local_request

    def held(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(local, "render_local_request", held)
    adapter = local.LocalTokenizerAdapter()
    task = asyncio.create_task(adapter.count_request(cfg, MESSAGES, []))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert adapter.report()["pending"] == adapter.report()["active"] == 1
        adapter.close()
        assert not adapter.report()["complete"]
        with pytest.raises(LocalTokenizerUnavailableError):
            await adapter.count_request(cfg, MESSAGES, [])
    finally:
        release.set()
        adapter.close()
        async with asyncio.timeout(5):
            while adapter.report()["pending"]:
                await asyncio.sleep(0.001)
    assert adapter.report()["complete"]


async def test_per_worker_cache_retains_only_two_artifacts_and_no_request_results(
    tmp_path, monkeypatch
):
    original = local.load_local_tokenizer
    references, loads = [], []

    class Artifact:
        def __init__(self, loaded):
            self.loaded = loaded

        def encode(self, *args):
            return self.loaded.encode(*args)

    def observed(cfg, job):
        loads.append(cfg.tokenizer_revision)
        proxy = Artifact(original(cfg, job))
        references.append(weakref.ref(proxy))
        return proxy

    monkeypatch.setattr(local, "load_local_tokenizer", observed)
    adapter = local.LocalTokenizerAdapter()
    adapter._owner.close()
    adapter._owner = worker._LocalWorkerOwner(workers=1, capacity=2)
    cfg = artifact(tmp_path, options={"template": TEMPLATE})
    try:
        for revision in ("one", "two", "three", "three", "two"):
            current = replace(cfg, tokenizer_revision=revision)
            assert await adapter.encode_text(current, "hello world")
        gc.collect()
        assert loads == ["one", "two", "three"]
        assert references[0]() is None and all(ref() is not None for ref in references[1:])
    finally:
        adapter.close()
    # Thread exit, rather than waiter cancellation or cache eviction under an
    # owner mutex, releases the last pinned native artifact references.
    async with asyncio.timeout(5):
        while any(ref() is not None for ref in references):
            await asyncio.sleep(0.001)
            gc.collect()


async def test_native_count_returned_after_its_deadline_is_never_delivered(tmp_path, monkeypatch):
    cfg = artifact(tmp_path, options={"template": TEMPLATE})
    now = [100.0]
    monkeypatch.setattr(worker, "monotonic", lambda: now[0])
    original = local.load_local_tokenizer

    class LateArtifact:
        def encode(self, text, config, job):
            value = original(config, job).encode(text, config, job)
            now[0] = 103.0
            return value

    monkeypatch.setattr(local, "load_local_tokenizer", lambda *args: LateArtifact())
    adapter = local.LocalTokenizerAdapter()
    try:
        with pytest.raises(LocalTokenizerLimitError):
            await adapter.count_request(cfg, MESSAGES, [])
        assert adapter.report()["pending"] == 0
    finally:
        adapter.close()


@pytest.mark.parametrize("timeout", [False, 0, -1, float("inf"), float("nan"), "1"])
async def test_bad_timeout_is_typed_and_does_not_start_a_worker(tmp_path, timeout):
    cfg = artifact(tmp_path)
    adapter = local.LocalTokenizerAdapter()
    try:
        with pytest.raises(LocalTokenizerInvalidInputError):
            await adapter.encode_text(cfg, "hello", timeout=timeout)
        assert adapter._owner._executor is None
    finally:
        adapter.close()


async def test_huge_integer_timeout_is_clipped_without_float_overflow(tmp_path):
    cfg = artifact(tmp_path)
    adapter = local.LocalTokenizerAdapter()
    try:
        assert (await adapter.encode_text(cfg, "hello", timeout=1 << 10000)).token_ids == (1,)
    finally:
        adapter.close()


async def test_no_template_and_unsupported_media_cannot_fall_back_to_partial_counts(
    tmp_path, monkeypatch
):
    cfg = artifact(tmp_path)

    def forbidden(*args):
        pytest.fail("invalid request loaded a tokenizer artifact")

    monkeypatch.setattr(local, "load_local_tokenizer", forbidden)
    adapter = local.LocalTokenizerAdapter()
    try:
        with pytest.raises(LocalTokenizerUnavailableError):
            await adapter.count_request(cfg, MESSAGES, [])
        with pytest.raises(LocalTokenizerInvalidInputError):
            await adapter.count_request(
                replace(cfg, template=TEMPLATE),
                [{"role": "user", "content": [{"type": "image"}]}],
                [],
            )
    finally:
        adapter.close()


def test_forged_config_and_huge_wall_allowance_have_typed_native_failures(tmp_path):
    from quarry.types.local_tokenizer import LocalTokenizerValidationError

    with pytest.raises(LocalTokenizerValidationError):
        LocalTokenizerLimits(max_wall_seconds=1 << 10000)
    cfg = artifact(tmp_path)
    object.__setattr__(cfg, "add_generation_prompt", object())
    with pytest.raises(LocalTokenizerInvalidInputError):
        local.LocalTokenizerAdapter._config(cfg, None)


async def test_model_revision_binding_is_checked_before_work(tmp_path):
    cfg = artifact(tmp_path, options={"template": TEMPLATE})
    adapter = local.LocalTokenizerAdapter()
    model = SimpleNamespace(
        tokens=SimpleNamespace(local=cfg, count_field="count"), revision="different"
    )
    try:
        with pytest.raises(LocalTokenizerInvalidInputError):
            await adapter.tokenize(model, MESSAGES, [])
        assert adapter._owner._executor is None
        model.revision = cfg.model_revision
        assert set(await adapter.tokenize(model, MESSAGES, [])) == {"count"}
    finally:
        adapter.close()
