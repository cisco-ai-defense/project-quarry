# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""A fork cannot retain its parent's released window ownership or affect that parent."""

import asyncio
import json
import os
import signal

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.cache.async_cache import result_cache
from quarry.integrations import model_window_input
from quarry.integrations.model_windows import CharacterWindow
from quarry.types.messages import Messages
from tests.unit.test_model_window_runtime import DATA, WindowAdapter, service


@pytest.mark.asyncio
@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
@pytest.mark.parametrize("unit", ["characters", "tokens"])
@pytest.mark.parametrize("restored", [False, True])
async def test_child_drops_inherited_window_storage_without_releasing_parent(
    tmp_path, monkeypatch, unit, restored
):
    memos = []
    original = model_window_input.WindowMemo.__init__

    def observed(memo, *args, **kwargs):
        original(memo, *args, **kwargs)
        memos.append(memo)

    monkeypatch.setattr(model_window_input.WindowMemo, "__init__", observed)
    runtime = service(tmp_path, unit, WindowAdapter())
    policy = compile_policy(
        'rule "R" condition: model("classes").classify(messages)>0.5',
        runtime_config=runtime.config,
        model_inference_backend=runtime,
    )
    if restored:
        wire = policy.to_bytes()
        policy = load_compiled_policy(
            wire, runtime_config=runtime.config, model_inference_backend=runtime
        )
        assert policy.to_bytes() == wire
    result_cache.clear()
    callbacks = []
    result = await policy._scanner._scan_data_with_parameters_async(
        DATA, {}, enable_explain=True, deferred_evidence=callbacks
    )
    assert result.is_complete and len(callbacks) == 1 and memos
    before = [(memo.budget.bytes, len(memo.budget._objects)) for memo in memos]
    assert all(size > 0 and objects > 0 for size, objects in before)
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        status = 0
        try:
            signal.alarm(3)
            callbacks[0].close()
            child = [
                {
                    "closed": memo.closed,
                    "bytes": memo.budget.bytes,
                    "objects": len(memo.budget._objects),
                    "renders": len(memo.entries),
                    "plans": len(memo.plans),
                    "measurements": len(memo.measurements),
                }
                for memo in memos
            ]
        except BaseException as error:
            status = 73
            child = {"error_type": type(error).__name__}
        os.write(write_fd, json.dumps(child).encode())
        os.close(write_fd)
        os._exit(status)
    os.close(write_fd)
    try:
        async with asyncio.timeout(5):
            while True:
                observed_pid, status = os.waitpid(pid, os.WNOHANG)
                if observed_pid:
                    break
                await asyncio.sleep(0.01)
        assert os.waitstatus_to_exitcode(status) == 0, {
            "child_exit": os.waitstatus_to_exitcode(status)
        }
        child = json.loads(os.read(read_fd, 4096))
        assert [(memo.budget.bytes, len(memo.budget._objects)) for memo in memos] == before
        assert all(not memo.closed for memo in memos)
        assert all(
            state
            == {
                "closed": True,
                "bytes": 0,
                "objects": 0,
                "renders": 0,
                "plans": 0,
                "measurements": 0,
            }
            for state in child
        ), child
    except BaseException:
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except ProcessLookupError:
            pass
        raise
    finally:
        os.close(read_fd)
        callbacks[0].close()
    assert all(
        memo.closed and memo.budget.bytes == 0 and not memo.budget._objects for memo in memos
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
@pytest.mark.parametrize("operation", ["render", "plan", "measured"])
async def test_child_cannot_resume_a_window_operation_its_parent_is_still_computing(operation):
    memo = model_window_input.WindowMemo(100_000)
    messages = Messages.from_openai(DATA["messages"])
    rendering = await model_window_input.join_window_input(messages, [])
    entered, release = asyncio.Event(), asyncio.Event()

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
    assert not task.done() and not memo.closed
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        status = 0
        try:
            signal.alarm(3)

            async def cannot_resume():
                with pytest.raises(RuntimeError, match="released"):
                    await getattr(memo, operation)(b"synthetic", compute)

            asyncio.run(cannot_resume())
            child = {
                "closed": memo.closed,
                "bytes": memo.budget.bytes,
                "objects": len(memo.budget._objects),
                "entries": memo._entry_count(),
            }
        except BaseException as error:
            status = 73
            child = {"error_type": type(error).__name__, "error": str(error)}
        os.write(write_fd, json.dumps(child).encode())
        os.close(write_fd)
        os._exit(status)
    os.close(write_fd)
    try:
        async with asyncio.timeout(5):
            while True:
                observed_pid, status = os.waitpid(pid, os.WNOHANG)
                if observed_pid:
                    break
                await asyncio.sleep(0.01)
        assert os.waitstatus_to_exitcode(status) == 0, {
            "operation": operation,
            "child_exit": os.waitstatus_to_exitcode(status),
        }
        child = json.loads(os.read(read_fd, 4096))
        assert child == {"closed": True, "bytes": 0, "objects": 0, "entries": 0}
        assert not task.done() and not memo.closed
        release.set()
        await task
        assert memo.budget.bytes > 0 and memo._entry_count() == 1
    except BaseException:
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except ProcessLookupError:
            pass
        raise
    finally:
        os.close(read_fd)
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        memo.close()
    assert memo.budget.bytes == 0 and not memo.budget._objects
