# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Optional native text lineage hooks without accounting or inference dependencies."""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Protocol


class TextReadObserver(Protocol):
    """Bounded metadata hooks. Implementations never retain inputs or raise errors."""

    def copied(self, source: object, target: object) -> None: ...

    def authored(self, target: object, content: object) -> None: ...

    def copy_updated(self, source: object, target: object, update: object) -> None: ...

    def unavailable(self, target: object) -> None: ...

    def sliced(self, source: object, target: object, start: int, end: int, text: str) -> None: ...

    def derived(self, source: object, target: object, content: object) -> None: ...

    def located(self, target: object, content: object, origin: object) -> None: ...

    def constructed(
        self,
        frame: object,
        constructor: object,
        scope: tuple,
        target: object,
        text: str,
        source: object | None,
    ) -> None: ...


_text_read_observer: ContextVar[TextReadObserver | None] = ContextVar(
    "quarry_text_read_observer", default=None
)


def _reset_text_read_observer_after_fork() -> None:
    _text_read_observer.set(None)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_text_read_observer_after_fork)
