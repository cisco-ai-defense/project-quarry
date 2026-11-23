# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""A caught local-work denial cannot become a reusable negative or message value."""

import pytest

from quarry.cache.immediate_values import ImmediateValueMemo
from quarry.cache.message_values import MessageValueMemo
from quarry.engine import matchers
from quarry.engine.local_work import ensure_local_work_authoritative, local_work_scope
from quarry.safety import local_match_work as work
from quarry.types.messages import Messages


def setup_owner(monkeypatch, *, mode="enforce"):
    ticks = iter([0, 10])
    monkeypatch.setattr(work, "thread_time_ns", lambda: next(ticks))
    return work.LocalMatchWorkOwner(work.LocalMatchWorkGroup("group", ("Selected",), 10, mode))


def caught_denial(measured):
    assert measured.call("match", lambda: True)
    try:
        return matchers.match_contains("synthetic", "synthetic")
    except work.LocalMatchWorkError:
        return False


def test_caught_exact_boundary_denial_cannot_become_an_authoritative_false(monkeypatch):
    measured = setup_owner(monkeypatch)
    with local_work_scope(measured):
        assert caught_denial(measured) is False
        with pytest.raises(work.LocalMatchWorkError):
            ensure_local_work_authoritative()
    report = measured.report()
    assert report["known_cpu_ns"] == 10
    assert report["denied_segments"] == 1 and report["known_overshoot_cpu_ns"] == 0


async def test_immediate_cache_does_not_admit_a_caught_denial(monkeypatch):
    class Identity:
        pass

    measured = setup_owner(monkeypatch)
    memo = ImmediateValueMemo()
    expression, backend = Identity(), Identity()
    try:
        with local_work_scope(measured):
            fallback = caught_denial(measured)
            with pytest.raises(work.LocalMatchWorkError):
                await memo.keep(expression, backend, (fallback, (), ()), None)
        assert memo.get(expression, backend) is None
        assert memo.budget.entries == 0
    finally:
        memo.close()
        measured.close()


async def test_message_value_cache_rejects_a_valid_shaped_fallback_after_caught_denial(monkeypatch):
    measured = setup_owner(monkeypatch)
    memo = MessageValueMemo()
    key = ("synthetic",)
    calls = []
    fallback = Messages.from_openai([{"role": "user", "content": "synthetic fallback"}])

    async def failed_computation():
        calls.append("denied")
        assert caught_denial(measured) is False
        return fallback

    async def fresh_computation():
        calls.append("fresh")
        return Messages.from_openai([{"role": "user", "content": "synthetic complete"}])

    try:
        with local_work_scope(measured), pytest.raises(work.LocalMatchWorkError):
            await memo.evaluate(key, failed_computation)
        assert memo._entries[key].value is None and memo._entries[key].lease is None
        fresh = await memo.evaluate(key, fresh_computation)
        assert fresh[0].content == "synthetic complete"
        assert calls == ["denied", "fresh"]
    finally:
        memo.close()
        measured.close()


async def test_shadow_exhaustion_preserves_a_complete_native_value_and_cache(monkeypatch):
    class Identity:
        pass

    ticks = iter([0, 11])
    monkeypatch.setattr(work, "thread_time_ns", lambda: next(ticks))
    measured = work.LocalMatchWorkOwner(
        work.LocalMatchWorkGroup("group", ("Selected",), 10, "shadow")
    )
    memo = ImmediateValueMemo()
    expression, backend = Identity(), Identity()
    data = (False, (), ())
    try:
        with local_work_scope(measured):
            assert matchers.match_contains("synthetic", "missing") is False
            ensure_local_work_authoritative()
            await memo.keep(expression, backend, data, None)
        assert memo.get(expression, backend) == data
    finally:
        memo.close()
        measured.close()
