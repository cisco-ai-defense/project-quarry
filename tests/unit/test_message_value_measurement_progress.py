# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Repeated references still give a storage walk cancellation opportunities."""

import asyncio

import pytest

import quarry.cache.message_values as value_module


@pytest.mark.asyncio
async def test_repeated_storage_references_remain_cooperatively_cancellable(monkeypatch):
    from quarry.cache.message_values import MessageValueBudget

    monkeypatch.setattr(value_module.Pacer, "due", lambda self: True)
    budget = MessageValueBudget()
    task = asyncio.create_task(budget.retain([None] * 20_000))
    try:
        for _ in range(4):
            await asyncio.sleep(0)
        assert task.cancel(), "reference traversal completed without the required yield points"
        with pytest.raises(asyncio.CancelledError):
            await task
        assert budget.bytes == 0 and not budget._objects
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        elif not task.cancelled() and task.exception() is None:
            task.result().close()
