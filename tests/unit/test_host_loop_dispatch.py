# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Awaited engine work leaves the host loop; host extensions keep their caller."""

import asyncio
import gc
import json
import os
import subprocess
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from contextvars import ContextVar
from types import SimpleNamespace

import pytest

from quarry import InputAdapter, compile_policy, load_compiled_policy
from quarry.cache.async_cache import result_cache
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.scanner import DSLScanner
from tests.unit.model_fakes import ModelAdapter, NoWaitLimiter, model_entry, passthrough_builder

DATA = {"messages": [{"role": "user", "content": "synthetic"}]}
_marker = ContextVar("synthetic_host_dispatch_marker", default="unset")


def observe_engine(monkeypatch):
    observed = []
    original = DSLScanner._scan_data_with_values_async

    async def observed_engine(self, *args, **kwargs):
        observed.append((threading.get_ident(), asyncio.get_running_loop(), _marker.get()))
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(DSLScanner, "_scan_data_with_values_async", observed_engine)
    return observed


def local_policy(restored=False):
    policy = compile_policy('rule "Read" condition: user_said("synthetic")')
    return load_compiled_policy(policy.to_bytes()) if restored else policy


@pytest.mark.parametrize("api", ["scanner", "policy", "restored", "batch", "restored-batch"])
async def test_awaited_scan_engine_leaves_the_callers_loop(monkeypatch, api):
    observed = observe_engine(monkeypatch)
    policy = local_policy(api.startswith("restored"))
    host_thread, host_loop = threading.get_ident(), asyncio.get_running_loop()
    token = _marker.set(api)
    try:
        if api == "scanner":
            results = [await policy._scanner.scan_data_async(DATA)]
        elif "batch" in api:
            results = await policy.scan_many_async([DATA, DATA])
        else:
            results = [await policy.scan_async(DATA)]
        assert _marker.get() == api
    finally:
        _marker.reset(token)
    assert all(result.is_complete and result.rule_matches == ["Read"] for result in results)
    assert observed and all(
        thread != host_thread and loop is not host_loop for thread, loop, _ in observed
    )
    assert {marker for _, _, marker in observed} == {api}
    assert len({thread for thread, _, _ in observed}) == 1


@pytest.mark.parametrize("api", ["scanner", "policy", "restored", "batch", "restored-batch"])
def test_synchronous_scan_keeps_its_calling_thread(monkeypatch, api):
    observed = observe_engine(monkeypatch)
    policy = local_policy(api.startswith("restored"))
    if api == "scanner":
        results = [policy._scanner.scan_data(DATA)]
    elif "batch" in api:
        results = policy.scan_many([DATA, DATA])
    else:
        results = [policy.scan(DATA)]
    assert all(result.is_complete and result.rule_matches == ["Read"] for result in results)
    assert observed and {thread for thread, _, _ in observed} == {threading.get_ident()}


@pytest.mark.parametrize("api", ["scanner", "batch"])
def test_async_entry_rejects_unknown_keywords_before_creating_a_coroutine(api):
    policy = local_policy()
    operation = policy._scanner.scan_data_async if api == "scanner" else policy.scan_many_async
    try:
        pending = operation(DATA, unsupported_synthetic_argument=True)
    except TypeError as error:
        assert "unsupported_synthetic_argument" in str(error)
    else:
        pending.close()
        pytest.fail("Invalid arguments were admitted as a pending coroutine")


async def test_host_can_release_a_synchronous_engine_preparation_barrier(monkeypatch):
    host_loop = asyncio.get_running_loop()
    released = threading.Event()
    completed = []
    original = DSLScanner._scan_data_with_values_async

    async def gated(self, *args, **kwargs):
        host_loop.call_soon_threadsafe(released.set)
        completed.append(released.wait(1))
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(DSLScanner, "_scan_data_with_values_async", gated)
    result = await local_policy().scan_async(DATA)
    assert result.is_complete
    assert completed == [True]


def score_policy(tmp_path, adapter, *, limiter=None):
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"reader": model_entry(score=True)}},
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(
        config,
        adapters={"fake": adapter},
        message_builders={"test": passthrough_builder},
        rate_limiter=NoWaitLimiter() if limiter is None else limiter,
    )
    return compile_policy(
        'rule "Read" condition: model("reader").score(messages, "synthetic") > 0.5',
        runtime_config=config,
        model_inference_backend=backend,
    )


async def test_adapter_factory_and_awaitable_keep_the_original_host(tmp_path, monkeypatch):
    observed = observe_engine(monkeypatch)
    host_thread, host_loop = threading.get_ident(), asyncio.get_running_loop()
    seen = []

    class Host(ModelAdapter):
        def score(self, config, messages, *, timeout=None):
            seen.append(
                ("factory", threading.get_ident(), asyncio.get_running_loop(), _marker.get())
            )

            async def finish():
                seen.append(
                    ("await", threading.get_ident(), asyncio.get_running_loop(), _marker.get())
                )
                _marker.set("host-child-only")
                return await ModelAdapter.score(self, config, messages, timeout=timeout)

            return finish()

    result_cache.clear()
    policy = score_policy(tmp_path, Host())
    token = _marker.set("caller")
    try:
        result = await policy.scan_async(DATA, logical_reads=True)
        assert _marker.get() == "caller"
    finally:
        _marker.reset(token)
    assert result.is_complete and result.rule_matches == ["Read"]
    assert [row[0] for row in seen] == ["factory", "await"]
    assert all(row[1:] == (host_thread, host_loop, "caller") for row in seen)
    assert observed and all(thread != host_thread for thread, _, _ in observed)


async def test_host_callback_can_await_a_nested_scan(tmp_path, monkeypatch):
    observed = observe_engine(monkeypatch)
    host_thread, host_loop = threading.get_ident(), asyncio.get_running_loop()
    nested = local_policy()
    entered = []

    class Host(ModelAdapter):
        async def score(self, config, messages, *, timeout=None):
            entered.append((threading.get_ident(), asyncio.get_running_loop()))
            result = await nested.scan_async(DATA)
            assert result.is_complete and result.rule_matches == ["Read"]
            return 0.99

    result_cache.clear()
    result = await asyncio.wait_for(score_policy(tmp_path, Host()).scan_async(DATA), timeout=5)
    assert result.is_complete and result.rule_matches == ["Read"]
    assert entered == [(host_thread, host_loop)]
    assert len(observed) == 2
    assert len({thread for thread, _, _ in observed}) == 1
    assert all(thread != host_thread for thread, _, _ in observed)


def test_independent_host_threads_share_one_engine_worker(monkeypatch):
    observed = observe_engine(monkeypatch)
    policy = local_policy()
    barrier = threading.Barrier(2)
    hosts = []

    def caller(label):
        hosts.append(threading.get_ident())
        _marker.set(label)
        barrier.wait(timeout=5)
        return asyncio.run(policy.scan_async(DATA))

    with ThreadPoolExecutor(max_workers=2) as pool:
        one, two = pool.submit(caller, "one"), pool.submit(caller, "two")
        results = [one.result(timeout=5), two.result(timeout=5)]
    assert all(result.is_complete and result.rule_matches == ["Read"] for result in results)
    assert {marker for _, _, marker in observed} == {"one", "two"}
    assert len({thread for thread, _, _ in observed}) == 1
    assert not set(hosts) & {thread for thread, _, _ in observed}


@pytest.mark.parametrize("repeat_cancel", [False, True])
async def test_cancellation_waits_for_engine_cleanup(monkeypatch, repeat_cancel):
    host_loop = asyncio.get_running_loop()
    entered, cleaning, released = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def suspended(_self, *args, **kwargs):
        host_loop.call_soon_threadsafe(entered.set)
        try:
            await asyncio.Future()
        finally:
            host_loop.call_soon_threadsafe(cleaning.set)
            await asyncio.sleep(0.01)
            host_loop.call_soon_threadsafe(released.set)

    monkeypatch.setattr(DSLScanner, "_scan_data_with_values_async", suspended)
    task = asyncio.create_task(local_policy().scan_async(DATA))
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    await asyncio.wait_for(cleaning.wait(), timeout=5)
    if repeat_cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released.is_set()


async def test_custom_backend_override_keeps_its_host_loop(tmp_path, monkeypatch):
    observed = observe_engine(monkeypatch)
    host_thread, host_loop = threading.get_ident(), asyncio.get_running_loop()
    seen = []

    class HostBackend(ModelInferenceBackend):
        async def model_classify_messages(self, model_id, messages, tools=None, *, timeout=None):
            seen.append((threading.get_ident(), asyncio.get_running_loop()))
            assert model_id == "reader" and messages[0].content == "synthetic"
            return 0.8

    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"reader": model_entry(classify=0)}},
        tmp_path,
        require_rules=False,
    )
    backend = HostBackend(config, adapters={"fake": ModelAdapter()})
    policy = compile_policy(
        'rule "Read" condition: model("reader").classify(messages) > 0.5',
        runtime_config=config,
        model_inference_backend=backend,
    )
    result = await policy.scan_async(DATA)
    assert result.is_complete and result.rule_matches == ["Read"]
    assert seen == [(host_thread, host_loop)]
    assert all(thread != host_thread for thread, _, _ in observed)


async def test_host_owned_rate_limiter_enters_and_exits_on_its_host(tmp_path):
    host = threading.get_ident(), asyncio.get_running_loop()
    seen = []

    class HostLimit:
        @asynccontextmanager
        async def limit(self, model):
            seen.append(("enter", threading.get_ident(), asyncio.get_running_loop()))
            assert seen[-1][1:] == host
            try:
                yield 0.0
            finally:
                seen.append(("exit", threading.get_ident(), asyncio.get_running_loop()))

    result_cache.clear()
    result = await score_policy(tmp_path, ModelAdapter(), limiter=HostLimit()).scan_async(DATA)
    assert result.is_complete and result.rule_matches == ["Read"], result.evaluation_errors
    assert [row[0] for row in seen] == ["enter", "exit"]
    assert all(row[1:] == host for row in seen)


@pytest.mark.parametrize("awaitable", [False, True])
async def test_legacy_span_decoder_still_requires_an_awaitable(tmp_path, awaitable):
    from quarry.types.results import Score, Span

    host = threading.get_ident(), asyncio.get_running_loop()
    seen = []

    class Host(ModelAdapter):
        def decode_spans(self, config, raw, messages, threshold):
            seen.append((threading.get_ident(), asyncio.get_running_loop()))
            value = [Span(0, 0, 1, Score(0.9))]
            if not awaitable:
                return value

            async def finish():
                return value

            return finish()

    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"reader": model_entry(spans=0, category="SYNTHETIC")}},
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(config, adapters={"fake": Host()}, rate_limiter=NoWaitLimiter())
    policy = compile_policy(
        'rule "Read" condition: true collect: [s for s in model("reader").spans(messages,0.5) if true]',
        runtime_config=config,
        model_inference_backend=backend,
    )
    result_cache.clear()
    result = await policy.scan_async(DATA)
    assert result.is_complete is awaitable
    assert result.rule_matches == (["Read"] if awaitable else [])
    assert seen == [host]


async def test_synchronous_custom_model_backend_is_not_silently_admitted(tmp_path):
    host = threading.get_ident(), asyncio.get_running_loop()
    seen = []

    class HostBackend(ModelInferenceBackend):
        def model_classify_messages(self, *args, **kwargs):
            seen.append((threading.get_ident(), asyncio.get_running_loop()))
            return 0.8

    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"reader": model_entry(classify=0)}},
        tmp_path,
        require_rules=False,
    )
    backend = HostBackend(config, adapters={"fake": ModelAdapter()})
    policy = compile_policy(
        'rule "Read" condition: model("reader").classify(messages)>0.5',
        runtime_config=config,
        model_inference_backend=backend,
    )
    result = await policy.scan_async(DATA)
    assert not result.is_complete and result.rule_matches == []
    assert seen == [host]


@pytest.mark.parametrize("callback", ["calibration", "declared-input", "prober"])
async def test_synchronous_extension_contract_does_not_await_an_unexpected_value(
    tmp_path, callback
):
    from quarry.async_runtime import run_engine
    from quarry.dsl.functions import off_topic
    from quarry.types.exceptions import ModelError

    host = threading.get_ident(), asyncio.get_running_loop()
    seen, awaited = [], []

    class UnexpectedAwaitable:
        def __await__(self):
            async def value():
                awaited.append(True)
                return True

            return value().__await__()

    def wrong(*args):
        seen.append((threading.get_ident(), asyncio.get_running_loop()))
        return UnexpectedAwaitable()

    if callback == "calibration":
        entry = model_entry(score=True)
        entry["score"]["calibration"] = "synthetic"
        config = QuarryConfig.from_mapping(
            {"config_version": 2, "models": {"reader": entry}}, tmp_path, require_rules=False
        )
        backend = ModelInferenceBackend(
            config,
            adapters={"fake": ModelAdapter()},
            message_builders={"test": passthrough_builder},
            calibrator_loader=lambda _name: wrong,
            rate_limiter=NoWaitLimiter(),
        )
        policy = compile_policy(
            'rule "Read" condition: model("reader").score(messages,"synthetic")>0.5',
            runtime_config=config,
            model_inference_backend=backend,
        )
        result_cache.clear()
        result = await policy.scan_async(DATA)
        assert not result.is_complete and result.rule_matches == []
    elif callback == "declared-input":

        class Native(InputAdapter):
            name, version, sections = "synthetic-native", "1", ("messages",)

            def convert(self, native, wanted):
                return wrong(native, wanted)

        scanner = DSLScanner(
            QuarryConfig(
                config_version=2,
                quarry_rule_texts=('rule "Read" condition: user_said("synthetic")',),
                adapters={"raw": {"adapter": "synthetic-native"}},
            ),
            input_adapters={"synthetic-native": Native()},
        )
        result = await scanner.scan_data_async({"raw": {"body": "synthetic"}})
        assert not result.is_complete and result.validation_error is not None
    else:
        context = SimpleNamespace(
            resource_manager=SimpleNamespace(topic_prober=SimpleNamespace(off_topic=wrong)),
            conversation=SimpleNamespace(user_messages=[SimpleNamespace(content="synthetic")]),
        )
        with pytest.raises(ModelError, match="expected bool"):
            await run_engine(lambda: off_topic(context, "synthetic"))
    assert seen == [host]
    assert not awaited


@pytest.mark.parametrize("trim", [False, True])
async def test_registered_score_builder_and_calibrator_keep_the_host(tmp_path, trim):
    host = threading.get_ident(), asyncio.get_running_loop()
    seen = []

    def builder(*args):
        seen.append(("builder", threading.get_ident(), asyncio.get_running_loop()))
        return passthrough_builder(*args)

    def calibrator(value):
        seen.append(("calibration", threading.get_ident(), asyncio.get_running_loop()))
        return value

    entry = model_entry(score=True, tokens=True)
    entry["score"].update(calibration="synthetic", max_input_tokens=20)
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"reader": entry}}, tmp_path, require_rules=False
    )
    backend = ModelInferenceBackend(
        config,
        adapters={"fake": ModelAdapter()},
        message_builders={"test": builder},
        calibrator_loader=lambda _name: calibrator,
        rate_limiter=NoWaitLimiter(),
    )
    selected = "trim(messages)" if trim else "messages"
    policy = compile_policy(
        f'rule "Read" condition: model("reader").score({selected}, "synthetic") > 0.5',
        runtime_config=config,
        model_inference_backend=backend,
    )
    result_cache.clear()
    result = await policy.scan_async(DATA)
    assert result.is_complete and result.rule_matches == ["Read"], result.evaluation_errors
    assert {row[0] for row in seen} == {"builder", "calibration"}
    assert all(row[1:] == host for row in seen)


async def test_declared_input_converter_keeps_its_synchronous_host_contract(monkeypatch):
    observed = observe_engine(monkeypatch)
    host = threading.get_ident(), asyncio.get_running_loop()
    seen = []

    class Native(InputAdapter):
        name = "synthetic-native"
        version = "1"
        sections = ("messages",)

        def convert(self, native, wanted):
            seen.append((threading.get_ident(), asyncio.get_running_loop()))
            assert wanted == frozenset({"messages"})
            return {"messages": [{"role": "user", "content": native["body"]}]}

    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            quarry_rule_texts=('rule "Read" condition: user_said("synthetic")',),
            adapters={"raw": {"adapter": "synthetic-native"}},
        ),
        input_adapters={"synthetic-native": Native()},
    )
    result = await scanner.scan_data_async({"raw": {"body": "synthetic"}})
    assert result.is_complete and result.rule_matches == ["Read"], result.evaluation_errors
    assert seen == [host]
    assert all(thread != host[0] for thread, _, _ in observed)


async def test_window_renderer_encoder_classifier_and_decoder_keep_the_host(tmp_path):
    from tests.unit.test_model_window_host_rendering import HostAdapter, runtime

    host = threading.get_ident(), asyncio.get_running_loop()
    adapter = HostAdapter()
    seen = []

    def observe(name):
        original = getattr(adapter, name)

        async def operation(*args, **kwargs):
            seen.append((name, threading.get_ident(), asyncio.get_running_loop()))
            return await original(*args, **kwargs)

        setattr(adapter, name, operation)

    for name in (
        "render_window_input",
        "encode_window_input",
        "classify_window_batch",
        "decode_window_spans",
    ):
        observe(name)
    backend = runtime(tmp_path, adapter)
    policy = compile_policy(
        'rule "Read" condition: model("classes").classify(messages)>0.5 '
        'collect: [s for s in model("segments").spans(messages,0.5) if true]',
        runtime_config=backend.config,
        model_inference_backend=backend,
    )
    result_cache.clear()
    result = await policy.scan_async({"messages": [{"role": "user", "content": "a needle"}]})
    assert result.is_complete and result.rule_matches == ["Read"], result.evaluation_errors
    assert {row[0] for row in seen} == {
        "render_window_input",
        "encode_window_input",
        "classify_window_batch",
        "decode_window_spans",
    }
    assert all(row[1:] == host for row in seen)


async def test_each_engine_pacer_has_an_independent_64_step_cadence():
    from quarry.async_runtime import run_engine
    from quarry.engine.pacing import Pacer

    async def cadence():
        one, two = Pacer(), Pacer(every=64)
        assert [one.due() for _ in range(63)] == [False] * 63
        assert [two.due() for _ in range(63)] == [False] * 63
        assert one.due() and two.due()
        assert [one.due() for _ in range(63)] == [False] * 63
        assert one.due()

    await run_engine(cadence)


async def test_queued_host_handoff_precedes_a_cooperative_local_resume():
    from quarry.async_runtime import run_engine
    from quarry.engine.dispatch import current_dispatch

    async def ordered():
        dispatch = current_dispatch()
        order = []
        wake = asyncio.get_running_loop().create_future()
        dispatch.locals.append(wake)
        dispatch.submit(lambda: order.append("host-handoff"))
        await wake
        order.append("local:0")
        assert order == ["host-handoff", "local:0"]

    await run_engine(ordered)


async def test_cancelled_queued_host_factory_is_not_invoked():
    from contextvars import copy_context

    from quarry.async_runtime import _Bridge

    calls = []
    bridge = _Bridge(asyncio.get_running_loop(), lambda: calls.append("called"), copy_context())
    bridge.cancel()
    bridge.start()
    with pytest.raises(asyncio.CancelledError):
        await bridge.wait()
    assert not calls


@pytest.mark.parametrize("cancel_before_start", [False, True])
async def test_closed_host_handoff_does_not_interrupt_other_dispatch(cancel_before_start):
    from contextvars import copy_context

    from quarry.async_runtime import _Bridge, run_engine
    from quarry.engine.dispatch import current_dispatch

    closed = asyncio.new_event_loop()
    closed.close()
    calls = []

    async def ordered():
        bridge = _Bridge(closed, lambda: calls.append("called"), copy_context())
        if cancel_before_start:
            bridge.cancel()
        dispatch = current_dispatch()
        order = []
        dispatch.submit(bridge.start)
        dispatch.submit(lambda: order.append("next-request"))
        await dispatch.pause()
        order.append("local")
        with pytest.raises(RuntimeError, match="Target event loop is closed"):
            await bridge.wait()
        assert bridge.factory is bridge.context is None
        assert order == ["next-request", "local"]

    await asyncio.wait_for(run_engine(ordered), timeout=5)
    assert not calls


async def test_rejected_remote_task_creation_does_not_strand_the_caller(monkeypatch):
    from contextvars import copy_context

    from quarry.async_runtime import _Bridge

    loop = asyncio.get_running_loop()
    calls = []
    error = RuntimeError("synthetic task creation rejected")

    def reject(*args, **kwargs):
        raise error

    bridge = _Bridge(loop, lambda: calls.append("called"), copy_context())
    with monkeypatch.context() as patch:
        patch.setattr(loop, "create_task", reject)
        bridge.start()
        with pytest.raises(RuntimeError) as caught:
            await bridge.wait()
    assert caught.value is error
    assert bridge.factory is bridge.context is bridge.task is None
    assert not calls


async def test_shutdown_finishes_before_a_new_worker_generation(monkeypatch):
    from quarry.async_runtime import shutdown_runtime

    observed = observe_engine(monkeypatch)
    policy = local_policy()
    first = await policy.scan_async(DATA)
    old_loop = observed[-1][1]
    await shutdown_runtime()
    assert old_loop.is_closed()
    second = await policy.scan_async(DATA)
    assert first.is_complete and second.is_complete
    assert observed[-1][1] is not old_loop


async def test_drained_generation_releases_its_owner_without_cyclic_gc():
    import quarry.async_runtime as runtime

    assert (await local_policy().scan_async(DATA)).is_complete
    old = runtime._generation
    reference, thread = weakref.ref(old), old.thread
    enabled = gc.isenabled()
    gc.disable()
    try:
        await runtime.shutdown_runtime()
        assert (await local_policy().scan_async(DATA)).is_complete
        thread.join(timeout=1)
        assert not thread.is_alive()
        del old
        assert reference() is None
    finally:
        if enabled:
            gc.enable()


async def test_shutdown_waits_for_host_cleanup_of_a_suspended_engine(monkeypatch):
    from quarry.async_runtime import call_host, shutdown_runtime

    host_loop = asyncio.get_running_loop()
    entered, released = asyncio.Event(), asyncio.Event()

    async def suspended(_self, *args, **kwargs):
        host_loop.call_soon_threadsafe(entered.set)
        try:
            await asyncio.Future()
        finally:
            await call_host(lambda: released.set())

    monkeypatch.setattr(DSLScanner, "_scan_data_with_values_async", suspended)
    task = asyncio.create_task(local_policy().scan_async(DATA))
    await asyncio.wait_for(entered.wait(), timeout=5)
    await asyncio.wait_for(shutdown_runtime(), timeout=5)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released.is_set()


async def test_failed_worker_start_does_not_poison_the_next_scan(monkeypatch):
    from quarry.async_runtime import shutdown_runtime

    await shutdown_runtime()
    policy = local_policy()

    def unavailable():
        raise RuntimeError("synthetic startup detail")

    with monkeypatch.context() as patch:
        patch.setattr(asyncio, "new_event_loop", unavailable)
        with pytest.raises(RuntimeError, match="Engine worker failed to initialize"):
            await asyncio.wait_for(policy.scan_async(DATA), timeout=5)
    result = await policy.scan_async(DATA)
    assert result.is_complete and result.rule_matches == ["Read"]


@pytest.mark.parametrize("with_waiter", [False, True])
async def test_rejected_worker_thread_releases_startup_waiters_and_allows_retry(
    monkeypatch, with_waiter
):
    import quarry.async_runtime as runtime

    await runtime.shutdown_runtime()
    policy = local_policy()
    entered, release = threading.Event(), threading.Event()
    original = threading.Thread.start

    def unavailable(thread):
        if thread.name != "quarry-engine":
            return original(thread)
        entered.set()
        assert release.wait(5)
        raise RuntimeError("synthetic thread start refused")

    async def scan_from_another_host():
        return await asyncio.to_thread(lambda: asyncio.run(policy.scan_async(DATA)))

    tasks = []
    try:
        with monkeypatch.context() as patch:
            patch.setattr(threading.Thread, "start", unavailable)
            tasks.append(asyncio.create_task(scan_from_another_host()))
            assert await asyncio.to_thread(entered.wait, 5)
            failed = runtime._generation
            assert failed is not None
            if with_waiter:
                tasks.append(asyncio.create_task(policy.scan_async(DATA)))
                await asyncio.sleep(0)
            release.set()
            outcomes = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=5
            )
        assert all(
            isinstance(error, RuntimeError) and str(error) == "Engine worker failed to initialize"
            for error in outcomes
        ), outcomes
        assert failed.ready.done() and failed.closed.done()
        assert failed.thread.ident is None
        assert runtime._generation is None
        result = await asyncio.wait_for(policy.scan_async(DATA), timeout=5)
        assert result.is_complete and result.rule_matches == ["Read"]
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("fault", ["thread", "event-loop"])
async def test_failed_worker_initialization_releases_its_scanner_without_cyclic_gc(
    monkeypatch, fault
):
    import quarry.async_runtime as runtime

    await runtime.shutdown_runtime()
    references = {}
    original = threading.Thread.start

    def unavailable_thread(thread):
        if thread.name != "quarry-engine":
            return original(thread)
        references["generation"] = weakref.ref(runtime._generation)
        references["thread"] = thread
        raise RuntimeError("synthetic thread start refused")

    def unavailable_loop():
        references["generation"] = weakref.ref(runtime._generation)
        references["thread"] = threading.current_thread()
        raise RuntimeError("synthetic event loop creation refused")

    async def failed_scan():
        policy = local_policy()
        references["scanner"] = weakref.ref(policy._scanner)
        with monkeypatch.context() as patch:
            if fault == "thread":
                patch.setattr(threading.Thread, "start", unavailable_thread)
            else:
                patch.setattr(asyncio, "new_event_loop", unavailable_loop)
            try:
                await policy.scan_async(DATA)
            except RuntimeError as error:
                assert str(error) == "Engine worker failed to initialize"
            else:
                pytest.fail("Expected the synthetic worker startup refusal")

    enabled = gc.isenabled()
    gc.disable()
    try:
        await failed_scan()
        thread = references["thread"]
        if thread.ident is not None:
            thread.join(timeout=5)
        assert not thread.is_alive()
        assert runtime._generation is None
        assert references["scanner"]() is None
        assert references["generation"]() is None
    finally:
        if enabled:
            gc.enable()


async def test_host_callback_cannot_deadlock_shutdown_of_its_own_engine():
    from quarry.async_runtime import call_host, run_engine, shutdown_runtime

    async def callback():
        with pytest.raises(RuntimeError, match="active engine operation"):
            await shutdown_runtime()
        return "finished"

    assert await asyncio.wait_for(run_engine(lambda: call_host(callback)), timeout=5) == "finished"


async def test_host_exception_keeps_identity_cause_and_notes():
    from quarry.async_runtime import call_host, run_engine

    error, cause = ValueError("synthetic error"), KeyError("synthetic cause")
    error.add_note("synthetic note")

    async def callback():
        raise error from cause

    with pytest.raises(ValueError) as caught:
        await run_engine(lambda: call_host(callback))
    assert caught.value is error and error.__cause__ is cause
    assert error.__notes__ == ["synthetic note"]


async def test_nonstandard_host_exit_is_bounded_and_leaves_the_worker_usable():
    from quarry.async_runtime import call_host, run_engine

    class Exit(BaseException):
        pass

    error = Exit("synthetic host detail")

    async def callback():
        raise error

    with pytest.raises(RuntimeError, match="unsupported BaseException") as caught:
        await run_engine(lambda: call_host(callback))
    assert caught.value.__cause__ is error
    assert "synthetic host detail" not in str(caught.value)
    assert (await local_policy().scan_async(DATA)).is_complete


async def test_production_local_scan_uses_worker_pacing(monkeypatch):
    from quarry.engine.pacing import Pacer

    original = Pacer.due
    calls = []

    def observed(self):
        due = original(self)
        if due and self._managed:
            calls.append(self._calls)
        return due

    monkeypatch.setattr(Pacer, "due", observed)
    data = {"messages": [{"role": "user", "content": "synthetic"} for _ in range(256)]}
    data["messages"][-1]["content"] = "needle"
    policy = compile_policy('rule "Read" condition: user_said("needle")')
    result = await policy.scan_async(data)
    assert result.is_complete and result.rule_matches == ["Read"]
    assert calls and all(call % 64 == 0 for call in calls)


@pytest.mark.parametrize("batch", [False, True])
async def test_inspection_deadline_includes_time_queued_for_the_worker(batch):
    from quarry.async_runtime import run_engine

    host_loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()

    async def occupied():
        host_loop.call_soon_threadsafe(entered.set)
        assert release.wait(2)

    blocker = asyncio.create_task(run_engine(occupied))
    await asyncio.wait_for(entered.wait(), timeout=5)
    policy = local_policy()
    pending = (
        policy.scan_many_async([DATA], deadline_seconds=0.01)
        if batch
        else policy.scan_async(DATA, deadline_seconds=0.01)
    )
    queued = asyncio.create_task(pending)
    try:
        await asyncio.sleep(0.05)
    finally:
        release.set()
    await blocker
    result = await asyncio.wait_for(queued, timeout=5)
    if batch:
        (result,) = result
    assert not result.is_complete
    assert result.timed_out_rules == ["Read"]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_child_replaces_a_worker_even_with_the_parent_manager_lock_held():
    script = r"""
import asyncio, json, os, signal, threading
import quarry.async_runtime as runtime
from quarry import compile_policy
policy = compile_policy('rule "Read" condition: user_said("synthetic")')
data = {"messages": [{"role": "user", "content": "synthetic"}]}
assert asyncio.run(policy.scan_async(data)).is_complete
parent = os.getpid()
locked, release = threading.Event(), threading.Event()
def hold():
    with runtime._manager_lock:
        locked.set()
        assert release.wait(10)
thread = threading.Thread(target=hold)
thread.start()
assert locked.wait(5)
read, write = os.pipe()
child = os.fork()
if child == 0:
    signal.alarm(5)
    os.close(read)
    result = asyncio.run(policy.scan_async(data))
    record = {"complete": result.is_complete, "child_worker_pid": runtime._generation.pid,
              "child_pid": os.getpid(), "parent_pid": parent}
    os.write(write, json.dumps(record).encode())
    os._exit(0)
os.close(write)
release.set()
thread.join(5)
raw = os.read(read, 65536)
os.close(read)
_, status = os.waitpid(child, 0)
assert status == 0, status
print(raw.decode())
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=15, check=True
    )
    record = json.loads(completed.stdout)
    assert record["complete"]
    assert record["child_worker_pid"] == record["child_pid"] != record["parent_pid"]
