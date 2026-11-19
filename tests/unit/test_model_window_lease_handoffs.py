# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Closing native window ownership includes leases still returning from admission."""

import asyncio
import concurrent.futures
import json
import os
import signal
import sys
import threading
import time
from contextlib import nullcontext

import pytest

from quarry.cache.message_values import MessageValueBudget
from quarry.integrations.model_window_input import WindowMemo, join_window_input
from quarry.integrations.model_windows import CharacterWindow
from quarry.types.messages import Messages


@pytest.mark.parametrize("operation", ["render", "plan", "measured"])
@pytest.mark.parametrize("disposal", ["close", "fork", "fork_locked"])
def test_unpublished_native_leases_cannot_outlive_window_ownership(
    operation, disposal, monkeypatch
):
    if disposal != "close" and not hasattr(os, "fork"):
        pytest.skip("requires POSIX fork")
    memo = WindowMemo(1_000_000)
    rendering = asyncio.run(
        join_window_input(
            Messages.from_openai([{"role": "user", "content": "SYNTHETIC-PENDING-LEASE"}]), []
        )
    )
    admitted, release = threading.Event(), threading.Event()
    original = memo.budget._retain_graph
    leases = []

    def pause(lease):
        leases.append(lease)
        admitted.set()
        assert release.wait(10)
        return lease

    if disposal == "fork_locked":
        native = MessageValueBudget._retain_graph

        def during_native_installation(budget, graph):
            lease = native(budget, graph)
            return pause(lease) if budget is memo.budget else lease

        monkeypatch.setattr(MessageValueBudget, "_retain_graph", during_native_installation)
    else:
        memo.budget._retain_graph = lambda graph: pause(original(graph))

    async def compute():
        if operation == "render":
            return rendering
        if operation == "measured":
            return (0.8,), 0.8, 0
        return (CharacterWindow(rendering.source, 0, len(rendering.source.text)),), (
            rendering.source.text,
        )

    async def execute():
        return await getattr(memo, operation)(b"synthetic-lease", compute)

    def state():
        return {
            "closed": memo.closed,
            "bytes": memo.budget.bytes,
            "objects": len(memo.budget._objects),
            "entries": memo._entry_count(),
        }

    empty = {"closed": True, "bytes": 0, "objects": 0, "entries": 0}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(lambda: asyncio.run(execute()))
    try:
        assert admitted.wait(5)
        before = state()
        assert before["bytes"] > 0 and before["objects"] > 0 and not before["closed"]
        assert all(
            entry.lease is None
            for table in [memo.entries, memo.plans, memo.measurements]
            for entry in table.values()
        )
        if disposal == "close":
            memo.close()
            assert state() == empty
            with pytest.raises(RuntimeError, match="released"):
                leases[0].copy_to(MessageValueBudget(1_000_000))
        else:
            read_fd, write_fd = os.pipe()
            warning = (
                pytest.warns(DeprecationWarning, match="multi-threaded")
                if sys.version_info >= (3, 12)
                else nullcontext()
            )
            with warning:
                pid = os.fork()
                if pid == 0:
                    # The deprecation is emitted in the parent. Exit before its warning
                    # assertion can run against the child's empty capture context.
                    os.close(read_fd)
                    signal.alarm(3)
                    os.write(write_fd, json.dumps(state()).encode())
                    os.close(write_fd)
                    os._exit(0)
            os.close(write_fd)
            observed = False
            try:
                deadline = time.monotonic() + 5
                while True:
                    found, status = os.waitpid(pid, os.WNOHANG)
                    if found:
                        observed = True
                        break
                    assert time.monotonic() < deadline
                    time.sleep(0.005)
                assert os.waitstatus_to_exitcode(status) == 0, {
                    "child_exit": os.waitstatus_to_exitcode(status)
                }
                child = json.loads(os.read(read_fd, 4096))
                assert state() == before
                assert child == empty
            finally:
                if not observed:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                os.close(read_fd)
        release.set()
        if disposal == "close":
            with pytest.raises(RuntimeError, match="released"):
                future.result(timeout=5)
        else:
            future.result(timeout=5)
    finally:
        release.set()
        try:
            future.result(timeout=5)
        except RuntimeError:
            assert disposal == "close"
        executor.shutdown(wait=True, cancel_futures=True)
        memo.close()
    assert state() == empty
