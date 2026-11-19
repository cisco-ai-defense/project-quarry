# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Engine-local cooperative scheduling with no application-runtime dependencies."""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress
from contextvars import Context, ContextVar


class _DispatchQueue:
    """Hand ready host requests off before resuming one cooperative local continuation."""

    def __init__(self, loop):
        self.loop = loop
        self.requests = deque()
        self.locals = deque()
        self.scheduled = False

    def _arm(self):
        if not self.scheduled:
            self.scheduled = True
            self.loop.call_soon(self._drain, context=Context())

    def submit(self, start):
        self.requests.append(start)
        self._arm()

    async def pause(self):
        wake = self.loop.create_future()
        self.locals.append(wake)
        self._arm()
        try:
            await wake
        finally:
            wake.cancel()

    def _drain(self):
        self.scheduled = False
        requests, self.requests = self.requests, deque()
        for start in requests:
            start()
        while self.locals:
            wake = self.locals.popleft()
            if not wake.done():
                wake.set_result(None)
                break
        if self.requests or self.locals:
            self._arm()


_dispatch: ContextVar[_DispatchQueue | None] = ContextVar("quarry_engine_dispatch", default=None)


def install_dispatch(dispatch):
    _dispatch.set(dispatch)


def current_dispatch():
    """Return the dispatcher owned by this actual engine event loop."""
    dispatch = _dispatch.get()
    if dispatch is not None:
        with suppress(RuntimeError):
            if asyncio.get_running_loop() is dispatch.loop:
                return dispatch
    return None


async def pause_engine():
    """Yield a cooperative continuation behind ready host handoffs."""
    dispatch = current_dispatch()
    if dispatch is None:
        await asyncio.sleep(0)
    else:
        await dispatch.pause()
