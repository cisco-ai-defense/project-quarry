# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Optional read replay follows value memoization without changing its decision budget."""

import asyncio
import gc
import weakref

import pytest

from quarry.cache.message_values import MessageValueMemo
from quarry.integrations.read_coverage import ReadInterval
from quarry.integrations.read_usage import (
    ReadUsageLimits,
    UnitMeasurement,
    current_read_usage,
    model_read_usage,
    read_usage_scope,
)
from quarry.types.messages import Messages


def measured_read(session, source=b"a" * 32):
    ticket = session.begin("synthetic-model", "score")
    if ticket is not None:
        ticket.finish(
            "succeeded",
            cut=b"c" * 32,
            characters=UnitMeasurement(
                "characters", "measured", (ReadInterval(source, "characters", 0, 12),)
            ),
            tokens=UnitMeasurement("text_tokens", "unavailable"),
        )


def value():
    return Messages.from_openai([{"role": "user", "content": "synthetic value"}])


@pytest.mark.asyncio
async def test_each_memo_reader_gets_logical_reads_and_parameter_reads_without_provider_rework():
    memo = MessageValueMemo()
    calls = 0
    result = value()
    with model_read_usage() as session:

        async def compute():
            nonlocal calls
            calls += 1
            measured_read(session)
            return result

        with read_usage_scope("first"):
            assert await memo.evaluate(("key",), compute, reads=[("limit", 3)]) is result
        for _ in range(2):
            parameters = []
            with read_usage_scope("second"):
                assert await memo.evaluate(("key",), compute, reads=parameters) is result
            assert parameters == [("limit", 3)]
        rows = session.snapshot()["rows"]
    assert calls == 1
    assert [row["rule"] for row in rows] == ["first", "second"]
    assert [row["observations"] for row in rows] == [1, 1]
    assert all(row["coverage"]["characters"]["value"] == 12 for row in rows)
    memo.close()
    assert not memo._entries and memo.budget.bytes == memo.budget.entries == 0


@pytest.mark.asyncio
async def test_accounting_does_not_change_the_value_graph_byte_limit_or_retention():
    result = value()
    first, second = MessageValueMemo(), MessageValueMemo()
    parameters = [("limit", 3)]

    async def plain():
        return result

    await first.evaluate(("key",), plain, reads=parameters)
    with model_read_usage() as session:

        async def with_read():
            measured_read(current_read_usage())
            return result

        second.budget.limit = first.budget.bytes
        await second.evaluate(("key",), with_read, reads=parameters)
        assert first.budget.bytes == second.budget.bytes
        assert first._entries[("key",)].logical_reads is None
        assert second._entries[("key",)].logical_reads is not None
        reference = weakref.ref(session)
    del session
    gc.collect()
    assert reference() is None
    first.close()
    second.close()
    assert first.budget.bytes == second.budget.bytes == 0


@pytest.mark.asyncio
async def test_metadata_limit_marks_incomplete_without_changing_or_recomputing_value():
    memo = MessageValueMemo()
    result = value()
    calls = 0
    with model_read_usage(ReadUsageLimits(max_events=1)) as session:

        async def compute():
            nonlocal calls
            calls += 1
            measured_read(session)
            measured_read(session, b"b" * 32)
            return result

        with read_usage_scope("first"):
            assert await memo.evaluate(("key",), compute) is result
        with read_usage_scope("second"):
            assert await memo.evaluate(("key",), compute) is result
        assert not session.snapshot()["observations_complete"]
    assert calls == 1
    memo.close()


@pytest.mark.asyncio
async def test_memo_created_without_accounting_cannot_invent_zero_reads_on_later_hit():
    memo = MessageValueMemo()
    result = value()
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        return result

    await memo.evaluate(("key",), compute)
    with model_read_usage() as session:
        assert await memo.evaluate(("key",), compute) is result
        report = session.snapshot()
    assert calls == 1
    assert not report["observations_complete"]
    assert report["rows"] == []
    memo.close()


@pytest.mark.asyncio
async def test_empty_but_observed_compute_is_a_complete_empty_footprint():
    memo = MessageValueMemo()
    result = value()

    async def compute():
        return result

    with model_read_usage() as session:
        await memo.evaluate(("key",), compute)
        with read_usage_scope("second"):
            assert await memo.evaluate(("key",), compute) is result
        report = session.snapshot()
    assert report["observations_complete"] and report["rows"] == []
    memo.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["failure", "cancellation", "byte_limit"])
async def test_unsuccessful_computation_never_publishes_a_replay_or_partial_value(kind):
    memo = MessageValueMemo()
    result = value()
    started = asyncio.Event()
    calls = 0
    with model_read_usage() as session:

        async def compute():
            nonlocal calls
            calls += 1
            measured_read(session)
            started.set()
            if kind == "failure":
                raise ValueError("synthetic computation failure")
            if kind == "cancellation":
                await asyncio.Event().wait()
            return result

        if kind == "byte_limit":
            memo.budget.limit = 1
        task = asyncio.create_task(memo.evaluate(("key",), compute))
        try:
            await asyncio.wait_for(started.wait(), 2)
            if kind == "cancellation":
                task.cancel()
            with pytest.raises(asyncio.CancelledError if kind == "cancellation" else ValueError):
                await task
            entry = memo._entries[("key",)]
            assert entry.value is entry.lease is entry.logical_reads is None
            assert memo.budget.bytes == 0
            memo.budget.limit = None

            async def retry():
                nonlocal calls
                calls += 1
                return result

            with read_usage_scope("retry"):
                assert await memo.evaluate(("key",), retry) is result
            with read_usage_scope("retry-hit"):
                assert await memo.evaluate(("key",), retry) is result
            assert [row["rule"] for row in session.snapshot()["rows"]] == [None]
            assert calls == 2
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            memo.close()


@pytest.mark.asyncio
async def test_concurrent_waiting_reader_gets_completed_replay_under_its_own_scope():
    memo = MessageValueMemo()
    result = value()
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0
    with model_read_usage() as session:

        async def compute():
            nonlocal calls
            calls += 1
            measured_read(session)
            started.set()
            await release.wait()
            return result

        async def read(rule):
            with read_usage_scope(rule):
                return await memo.evaluate(("key",), compute)

        first = asyncio.create_task(read("first"))
        second = None
        try:
            await asyncio.wait_for(started.wait(), 2)
            second = asyncio.create_task(read("second"))
            await asyncio.sleep(0)
            release.set()
            assert await first is result
            assert await second is result
            rows = session.snapshot()["rows"]
            assert [row["rule"] for row in rows] == ["first", "second"]
            assert [row["observations"] for row in rows] == [1, 1]
            assert calls == 1
        finally:
            for task in [first, second]:
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in [first, second] if task is not None), return_exceptions=True
            )
            memo.close()
