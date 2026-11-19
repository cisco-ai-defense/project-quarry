# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Native window ownership when leased scans resume on independent worker loops."""

import asyncio
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import quarry
from quarry.cache.async_cache import result_cache, scan_scope
from quarry.cache.constants import RESULT_CACHE_MAX_BYTES
from quarry.cache.message_values import MAX_MESSAGE_VALUES
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.integrations.model_window_input import WindowMemo, join_window_input
from quarry.integrations.model_windows import CharacterWindow, TokenWindow
from quarry.types.messages import Messages


def _state(memo):
    return (memo.closed, memo.budget.bytes, len(memo.budget._objects), memo._entry_count())


@pytest.mark.parametrize("prewarm", [False, True])
@pytest.mark.parametrize("same_input", [False, True])
def test_resumed_native_callers_release_every_memo_after_the_last_lease(
    tmp_path, monkeypatch, prewarm, same_input
):
    class Host:
        window_units = {"characters"}

        async def classify_window_batch(self, config, batches, context, *, timeout=None):
            await asyncio.sleep(0)
            return SimpleNamespace(chunks=[SimpleNamespace(probabilities=[0.75]) for _ in batches])

    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "m": {
                    "adapter": "owned",
                    "model": "synthetic-scope",
                    "windows": {"unit": "characters", "size": 3},
                    "classify": {"class_index": 0},
                }
            },
        },
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(config, adapters={"owned": Host()})
    messages = Messages.from_openai([{"role": "user", "content": "SYNTHETIC-FIRST"}])
    other = (
        messages
        if same_input
        else Messages.from_openai([{"role": "user", "content": "SYNTHETIC-SECOND"}])
    )
    result_cache.clear()
    if prewarm:
        with scan_scope():
            assert asyncio.run(backend.model_classify_messages("m", messages)) == 0.75
        result_cache.clear()

    first_created = threading.Event()
    second_created = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    original = WindowMemo.__init__
    memos = []
    observe_lock = threading.Lock()

    def observe(memo, *args, **kwargs):
        original(memo, *args, **kwargs)
        with observe_lock:
            memos.append(memo)
            first = len(memos) == 1
        if first:
            first_created.set()
            assert release_first.wait(5)
        else:
            second_created.set()

    monkeypatch.setattr(WindowMemo, "__init__", observe)
    try:
        with scan_scope() as scope:

            def worker(value, second=False):
                with scan_scope(scope):
                    if second:
                        second_started.set()
                    return asyncio.run(backend.model_classify_messages("m", value))

            with ThreadPoolExecutor(max_workers=2) as pool:
                one = pool.submit(worker, messages)
                assert first_created.wait(5)
                two = pool.submit(worker, other, True)
                assert second_started.wait(5)
                # A competing factory may finish, or an atomic owner may hold it back.
                # Both shapes must release correctly; this wait never supplies approval.
                second_created.wait(0.5)
                release_first.set()
                assert one.result(timeout=5) == two.result(timeout=5) == 0.75
            live_before_release = sum(not memo.closed for memo in memos)
        assert scope._leases == 0 and not scope._release_callbacks
        assert all(_state(memo) == (True, 0, 0, 0) for memo in memos)
        assert live_before_release == 1
    finally:
        release_first.set()
        for memo in memos:
            memo.close()


_LOOP_PROBE = r"""
import asyncio
from concurrent.futures import Future
import json
import os
from pathlib import Path
import sys
import threading

import quarry
from quarry.integrations.model_window_input import WindowMemo, join_window_input
from quarry.integrations.model_windows import CharacterWindow
from quarry.types.messages import Messages
assert Path(quarry.__file__).resolve().parent == Path(os.environ["QUARRY_EXPECTED_PACKAGE"]).resolve()
operation, action = sys.argv[1:]
rendering = asyncio.run(join_window_input(Messages.from_openai([{"role":"user","content":"SYNTHETIC-LOOPS"}]), []))
value = rendering if operation == "render" else (((0.25, 0.75), 0.75, 1) if operation == "measured" else ((CharacterWindow(rendering.source, 0, len(rendering.source.text)),), (rendering.source.text,)))
memo = WindowMemo(65536)
first_entered, second_waiting = threading.Event(), threading.Event()
controls = {}
compute_calls = 0
async def first():
    controls["first_loop"] = asyncio.get_running_loop()
    controls["first_task"] = asyncio.current_task()
    controls["release"] = asyncio.Event()
    async def compute():
        global compute_calls
        compute_calls += 1
        first_entered.set()
        await controls["release"].wait()
        return value
    return await getattr(memo, operation)(b"same-native-key", compute)
async def second():
    controls["second_loop"] = asyncio.get_running_loop()
    controls["second_task"] = asyncio.current_task()
    controls["second_loop"].call_soon(second_waiting.set)
    async def compute():
        global compute_calls
        compute_calls += 1
        assert action == "cancel_owner", "Compatible successful work was recomputed"
        return value
    return await getattr(memo, operation)(b"same-native-key", compute)
def start(coroutine):
    result = Future()
    def worker():
        try:
            result.set_result(asyncio.run(coroutine(), debug=True))
        except BaseException as error:
            result.set_exception(error)
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread, result
def cancelled(result):
    try:
        result.result(timeout=5)
    except asyncio.CancelledError:
        return
    raise AssertionError("The cancelled native caller completed normally")
def released(result):
    try:
        result.result(timeout=5)
    except RuntimeError as error:
        assert "released" in str(error)
        return
    raise AssertionError("The closed native memo returned a value")
try:
    a, one = start(first)
    assert first_entered.wait(5)
    b, two = start(second)
    assert second_waiting.wait(5)
    if action == "close":
        memo.close()
        released(two)
        controls["first_loop"].call_soon_threadsafe(controls["release"].set)
        released(one)
        assert compute_calls == 1
    elif action == "cancel_owner":
        controls["first_loop"].call_soon_threadsafe(controls["first_task"].cancel)
        cancelled(one)
        assert two.result(timeout=5) is value and compute_calls == 2
    elif action == "cancel_waiter":
        controls["second_loop"].call_soon_threadsafe(controls["second_task"].cancel)
        cancelled(two)
        controls["first_loop"].call_soon_threadsafe(controls["release"].set)
        assert one.result(timeout=5) is value and compute_calls == 1
    else:
        controls["first_loop"].call_soon_threadsafe(controls["release"].set)
        assert one.result(timeout=5) is two.result(timeout=5) is value and compute_calls == 1
    a.join(5)
    b.join(5)
    assert not a.is_alive() and not b.is_alive()
    async def later():
        async def forbidden():
            raise AssertionError("A completed native entry was recomputed")
        return await getattr(memo, operation)(b"same-native-key", forbidden)
    if action != "close":
        assert asyncio.run(later(), debug=True) is value
    memo.close()
    assert memo.budget.bytes == 0 and not memo.budget._objects
    print(json.dumps({"complete": True, "operation": operation, "action": action, "compute_calls": compute_calls, "both_debug_loops_closed": True, "bytes": memo.budget.bytes}), flush=True)
except BaseException as error:
    print(json.dumps({"complete": False, "error_type": type(error).__name__, "error": str(error)}), flush=True)
    # This synthetic child contains a stranded waiter if a debug wake fails.
    os._exit(1)
"""


@pytest.mark.parametrize("operation", ["render", "plan", "measured"])
@pytest.mark.parametrize("action", ["complete", "cancel_owner", "cancel_waiter", "close"])
def test_native_memo_waiters_survive_separate_debug_event_loops(operation, action):
    env = dict(os.environ, QUARRY_EXPECTED_PACKAGE=str(Path(quarry.__file__).resolve().parent))
    result = subprocess.run(
        [sys.executable, "-c", _LOOP_PROBE, operation, action],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    row = json.loads(result.stdout)
    assert row["complete"] and row["both_debug_loops_closed"] and row["bytes"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["characters", "tokens"])
async def test_window_memo_entry_limit_is_aggregate_across_native_storage(unit):
    memo = WindowMemo(RESULT_CACHE_MAX_BYTES)
    rendering = await join_window_input(
        Messages.from_openai([{"role": "user", "content": "SYNTHETIC-PRESSURE"}]), []
    )
    plan = (
        ((CharacterWindow(rendering.source, 0, 3),), ("SYN",))
        if unit == "characters"
        else ((TokenWindow((1, 2), (0, 1)),), ((1, 2),))
    )
    try:
        for index in range(MAX_MESSAGE_VALUES):
            operation = ["render", "plan", "measured"][index % 3]
            value = (
                rendering
                if operation == "render"
                else plan
                if operation == "plan"
                else ((0.75,), 0.75, 0)
            )

            async def compute(value=value):
                return value

            await getattr(memo, operation)(str(index).encode(), compute)
        before = _state(memo)
        assert before[-1] == MAX_MESSAGE_VALUES

        async def forbidden():
            raise AssertionError("Over-limit entry must fail before computation")

        for operation in ["render", "plan", "measured"]:
            with pytest.raises(ValueError, match="entry limit"):
                await getattr(memo, operation)(("overflow-" + operation).encode(), forbidden)
        assert _state(memo) == before
    finally:
        memo.close()
    assert _state(memo) == (True, 0, 0, 0)
