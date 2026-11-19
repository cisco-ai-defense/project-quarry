# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Run awaited engine work separately from host-owned asynchronous extensions."""

from __future__ import annotations

import asyncio
import atexit
import inspect
import os
import threading
import traceback
import weakref
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from contextlib import suppress
from contextvars import Context, ContextVar, copy_context
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeVar

from quarry.engine.dispatch import _DispatchQueue, install_dispatch

T = TypeVar("T")


@dataclass(slots=True)
class _Outcome:
    value: Any = None
    error: BaseException | None = None


async def _invoke(factory):
    value = None
    try:
        value = factory()
        return _Outcome(await value if inspect.isawaitable(value) else value)
    except BaseException as error:
        # Store nonstandard exits as data, so an extension cannot stop another loop.
        return _Outcome(error=error)
    finally:
        del factory, value


class _Bridge:
    """One remote task, with completion acknowledged only after its cleanup finishes."""

    def __init__(self, loop, factory, context):
        self.loop = loop
        self.factory = factory
        self.context = context
        self.done: Future[_Outcome] = Future()
        self.task: asyncio.Task | None = None
        self.cancelled = False

    def start(self):
        if self.done.done():
            self.factory = self.context = None
            return
        try:
            self.loop.call_soon_threadsafe(self._start, context=Context())
        except RuntimeError:
            self.factory = self.context = None
            if not self.done.done():
                self.done.set_result(_Outcome(error=RuntimeError("Target event loop is closed")))

    def _start(self):
        factory, self.factory = self.factory, None
        context, self.context = self.context, None
        if self.cancelled:
            self.done.set_result(_Outcome(error=asyncio.CancelledError()))
            return
        coroutine = _invoke(factory)
        try:
            self.task = self.loop.create_task(coroutine, context=context)
        except BaseException as error:
            coroutine.close()
            self.loop.call_soon(self._deliver, _Outcome(error=error), context=Context())
            return
        self.task.add_done_callback(self._finished, context=Context())

    def _finished(self, task):
        try:
            outcome = task.result()
        except BaseException as error:
            outcome = _Outcome(error=error)
        self.task = None
        # Deliver on the next turn, after the completed task callback drops its task/context.
        self.loop.call_soon(self._deliver, outcome, context=Context())
        del self, task, outcome

    def _deliver(self, outcome):
        if outcome.error is not None:
            # Preserve exception identity/cause/notes without pinning completed task frames.
            traceback.clear_frames(outcome.error.__traceback__)
        self.done.set_result(outcome)
        del self, outcome

    def cancel(self):
        def cancel_task():
            self.cancelled = True
            if self.task is not None:
                self.task.cancel()

        try:
            self.loop.call_soon_threadsafe(cancel_task, context=Context())
        except RuntimeError:
            if not self.done.done():
                self.done.set_result(_Outcome(error=RuntimeError("Target event loop is closed")))

    async def wait(self):
        future = asyncio.wrap_future(self.done)
        outcome = error = None
        try:
            try:
                outcome = await asyncio.shield(future)
            except asyncio.CancelledError:
                self.cancel()
                # Repeated outer cancellation must not interrupt remote finally blocks.
                while not future.done():
                    with suppress(asyncio.CancelledError):
                        await asyncio.shield(future)
                raise
            error = outcome.error
            if error is not None:
                if not isinstance(error, (Exception, asyncio.CancelledError)):
                    raise RuntimeError("Extension raised an unsupported BaseException") from error
                raise error
            return outcome.value
        finally:
            del self, future, outcome, error


@dataclass(frozen=True, slots=True)
class _Session:
    host_loop: asyncio.AbstractEventLoop
    worker_loop: asyncio.AbstractEventLoop
    dispatch: _DispatchQueue


_session: ContextVar[_Session | None] = ContextVar("quarry_host_runtime_session", default=None)
_inline: ContextVar[bool] = ContextVar("quarry_inline_runtime", default=False)
_manager_lock = threading.Lock()
_generation: _Generation | None = None


def _run_generation(reference):
    generation = reference()
    if generation is not None:
        generation._run()


class _Generation:
    def __init__(self):
        self.pid = os.getpid()
        self.ready: Future[asyncio.AbstractEventLoop | None] = Future()
        self.closed: Future[None] = Future()
        self.closing = False
        self.posters = 0
        self.stop_requested = False
        self.dispatch = None
        self.thread = threading.Thread(
            target=_run_generation,
            args=(weakref.ref(self),),
            name="quarry-engine",
            daemon=True,
        )

    def _run(self):
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.dispatch = _DispatchQueue(loop)
            self.ready.set_result(loop)
            loop.run_forever()
        except BaseException:
            if not self.ready.done():
                self.ready.set_result(None)
        finally:
            with _manager_lock:
                self.closing = True
            if loop is not None:
                loop.run_until_complete(self._cancel_tasks())
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
            self.closed.set_result(None)

    @staticmethod
    async def _cancel_tasks():
        tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self):
        def ready(future):
            loop = future.result()
            if loop is None:
                return

            async def finish():
                await _Generation._cancel_tasks()
                loop.stop()

            with suppress(RuntimeError):
                loop.call_soon_threadsafe(lambda: loop.create_task(finish()), context=Context())

        self.ready.add_done_callback(ready)

    def submit(self, loop, factory, context):
        bridge = _Bridge(loop, factory, context)
        with _manager_lock:
            if self is not _generation or self.closing:
                return None
            self.posters += 1
        try:
            bridge.start()
        finally:
            with _manager_lock:
                self.posters -= 1
                stop = self.closing and not self.posters and not self.stop_requested
                if stop:
                    self.stop_requested = True
            if stop:
                self.stop()
        return bridge


async def _worker():
    global _generation
    while True:
        with _manager_lock:
            generation = _generation
        if generation is None:
            candidate = _Generation()
            with _manager_lock:
                start = _generation is None
                if start:
                    _generation = candidate
                generation = _generation
            if start:
                try:
                    candidate.thread.start()
                except Exception:
                    # No worker can settle readiness after the OS rejects its thread.
                    # Wake every startup waiter and let the common failure path retire it.
                    with _manager_lock:
                        candidate.closing = True
                    candidate.ready.set_result(None)
                    candidate.closed.set_result(None)
        assert generation is not None
        try:
            loop = (
                generation.ready.result()
                if generation.ready.done()
                else await asyncio.shield(asyncio.wrap_future(generation.ready))
            )
            if loop is None:
                # Store only the startup status in the generation. A shared exception
                # would retain the caller's failed scan through its extended traceback.
                raise RuntimeError("Engine worker failed to initialize")
        except Exception:
            try:
                if not generation.closed.done():
                    await asyncio.shield(asyncio.wrap_future(generation.closed))
            finally:
                with _manager_lock:
                    if _generation is generation:
                        _generation = None
            raise
        if generation.closing:
            if not generation.closed.done():
                await asyncio.shield(asyncio.wrap_future(generation.closed))
            with _manager_lock:
                if _generation is generation:
                    _generation = None
            continue
        return generation, loop


async def run_engine(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Await one engine operation, keeping nested engine work on the same worker."""
    bridge = context = None
    try:
        host = asyncio.get_running_loop()
        current = _session.get()
        if _inline.get() or (current is not None and host is current.worker_loop):
            return await factory()
        while True:
            generation, loop = await _worker()

            context = copy_context()
            context.run(_session.set, _Session(host, loop, generation.dispatch))
            context.run(install_dispatch, generation.dispatch)
            bridge = generation.submit(loop, factory, context)
            if bridge is not None:
                return await bridge.wait()
    finally:
        del factory, bridge, context


async def call_host(factory: Callable[[], Any]) -> Any:
    """Invoke and await an extension on the loop/thread that submitted its scan."""
    bridge = None
    try:
        session = _session.get()
        if session is None or asyncio.get_running_loop() is session.host_loop:
            value = factory()
            return await value if inspect.isawaitable(value) else value
        bridge = _Bridge(session.host_loop, factory, copy_context())
        session.dispatch.submit(bridge.start)
        return await bridge.wait()
    finally:
        del factory, bridge


def _sync_value(factory):
    return (factory(),)


async def call_host_sync(factory: Callable[[], T]) -> T:
    """Keep a synchronous callback's return contract, including invalid awaitable values."""
    try:
        return (await call_host(partial(_sync_value, factory)))[0]
    finally:
        del factory


async def _await_factory(factory):
    try:
        return await factory()
    finally:
        del factory


async def call_host_async(factory: Callable[[], Any]) -> Any:
    """Keep an asynchronous callback's requirement to return an awaitable."""
    try:
        return await call_host(partial(_await_factory, factory))
    finally:
        del factory


def run_sync(coroutine: Coroutine[Any, Any, T]) -> T:
    """Keep synchronous APIs inline, with their existing asyncio.run contract."""
    token = _inline.set(True)
    try:
        return asyncio.run(coroutine)
    finally:
        _inline.reset(token)


async def shutdown_runtime() -> None:
    """Drain the current worker before another generation can start."""
    if _session.get() is not None:
        raise RuntimeError("Cannot shut down the worker from an active engine operation")
    with _manager_lock:
        generation = _generation
        if generation is None:
            return
        generation.closing = True
        stop = not generation.posters and not generation.stop_requested
        if stop:
            generation.stop_requested = True
    if stop:
        generation.stop()
    await asyncio.shield(asyncio.wrap_future(generation.closed))


def _after_fork():
    global _generation, _manager_lock
    _manager_lock = threading.Lock()
    _generation = None
    _session.set(None)
    _inline.set(False)
    install_dispatch(None)


def _at_exit():
    with _manager_lock:
        generation = _generation
        stop = generation is not None and not generation.stop_requested
        if generation is not None:
            generation.closing = True
            generation.stop_requested = True
    if generation is not None and not generation.closed.done():
        if stop:
            generation.stop()
        if generation.thread is not threading.current_thread():
            generation.thread.join(timeout=1)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
atexit.register(_at_exit)
