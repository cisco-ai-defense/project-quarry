# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Optional synchronous work observations, independent of scanner/runtime services.

Binding a scope does not start a measurement. Only synchronous matcher calls and
actual iterator steps use the supplied observer. An iterator keeps the owner that
created its matching work, so closing that owner can reject later consumption.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, ParamSpec, Protocol, TypeVar, cast

_P = ParamSpec("_P")
_T = TypeVar("_T")


class LocalWorkObserver(Protocol):
    def call(self, phase: str, function: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
        """Run and account for a synchronous local segment."""
        ...

    def iterate(self, iterator: Iterator[_T]) -> Iterator[_T]:
        """Account for actual lazy steps, rather than generator construction alone."""
        ...


_CURRENT: ContextVar[LocalWorkObserver | None] = ContextVar("quarry_local_match_work", default=None)


def current_local_work() -> LocalWorkObserver | None:
    return _CURRENT.get()


@contextmanager
def local_work_scope(observer: LocalWorkObserver | None) -> Generator[None, None, None]:
    """Select a caller-provided owner; explicit None isolates nested/default work."""
    token = _CURRENT.set(observer)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def measured_local_work(
    phase: str, *, iterator: bool = False
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Instrument a synchronous primitive without reading its arguments or result."""
    if phase not in {"prepare", "match", "iterate"}:
        raise ValueError("unknown local work phase")

    def decorate(function: Callable[_P, _T]) -> Callable[_P, _T]:
        @wraps(function)
        def measured(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            observer = _CURRENT.get()
            if observer is None:
                return function(*args, **kwargs)
            result = observer.call(phase, function, *args, **kwargs)
            if iterator:
                return cast(_T, observer.iterate(cast(Iterator[Any], result)))
            return result

        return measured

    return decorate
