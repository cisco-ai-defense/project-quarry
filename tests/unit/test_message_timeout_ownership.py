# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Native timer ownership does not depend on clock resolution after expiry."""

import asyncio
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from quarry.integrations import model_inference
from quarry.safety import timer
from quarry.types.exceptions import ModelError
from quarry.types.messages import Messages
from tests.unit.test_message_rewrites import RewriteAdapter, service


@pytest.mark.parametrize("owner", ["rule", "scan", "rule_inside_scan", "scan_inside_rule"])
async def test_expired_native_message_timer_preserves_its_evaluation_owner(
    tmp_path, monkeypatch, owner
):
    await _assert_native_message_timer_owner(tmp_path, monkeypatch, owner, 1024.0)


@pytest.mark.parametrize("owner", ["rule", "scan", "rule_inside_scan", "scan_inside_rule"])
async def test_rounded_deadline_preserves_native_message_timer_owner(tmp_path, monkeypatch, owner):
    await _assert_native_message_timer_owner(tmp_path, monkeypatch, owner, 1015.0000000000001)


async def _assert_native_message_timer_owner(tmp_path, monkeypatch, owner, clock_start):
    # Exercise exact subtraction and rounding across a binary exponent boundary.
    now = clock_start
    expected_remaining = (clock_start + 10) - clock_start
    monkeypatch.setattr(timer, "time", SimpleNamespace(monotonic=lambda: now))
    cancelled = asyncio.Event()

    class Waiting(RewriteAdapter):
        async def transform_messages(self, config, wire):
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

    _, backend, _ = service(tmp_path, Waiting())
    source = Messages.from_openai([{"role": "user", "content": "synthetic"}])
    observed = []

    class EarlyAsyncio:
        def timeout(self, seconds):
            observed.append(seconds)
            assert seconds == expected_remaining
            # Exercise a real expiring asyncio.Timeout while the cooperative
            # clock still reads before its deadline, as a coarse clock can do.
            return asyncio.timeout(0)

        def __getattr__(self, name):
            return getattr(asyncio, name)

    monkeypatch.setattr(model_inference, "asyncio", EarlyAsyncio())
    with ExitStack() as stack:
        if owner == "rule_inside_scan":
            stack.enter_context(timer.scan_deadline(20, "outer scan"))
        if owner == "scan_inside_rule":
            stack.enter_context(timer.evaluation_deadline(20, "outer rule"))
        manager = (
            timer.scan_deadline
            if owner in {"scan", "scan_inside_rule"}
            else timer.evaluation_deadline
        )
        stack.enter_context(manager(10, "selected evaluation owner"))
        with pytest.raises(timer.QuarryTimeoutError, match="^selected evaluation owner$"):
            await backend.transform_messages("rewrite", source)
        assert timer.remaining_evaluation_time() == expected_remaining
    assert observed == [expected_remaining] and cancelled.is_set()


async def test_adapter_timeout_does_not_borrow_an_unexpired_evaluation_owner(tmp_path):
    class SelfTimedOut(RewriteAdapter):
        async def transform_messages(self, config, wire):
            raise TimeoutError("synthetic provider timeout")

    _, backend, _ = service(tmp_path, SelfTimedOut())
    source = Messages.from_openai([{"role": "user", "content": "synthetic"}])
    with timer.evaluation_deadline(10), pytest.raises(ModelError) as caught:
        await backend.transform_messages("rewrite", source)
    assert caught.value.details["code"] == "timeout"
    assert not isinstance(caught.value, timer.QuarryTimeoutError)
