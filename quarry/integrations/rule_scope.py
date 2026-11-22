# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Live rule ownership across copied callback contexts, without retaining scans."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from os import getpid
from threading import Lock
from weakref import ReferenceType, ref


class _RuleOwner:
    __slots__ = ("_pid", "_name", "_lock", "_closed", "__weakref__")

    def __init__(self, name: str):
        self._pid = getpid()
        self._name: str | None = name
        self._lock = Lock()
        self._closed = False

    def name(self) -> str | None:
        # A fork can inherit a locked parent mutex. Reject that owner before locking.
        if self._pid != getpid():
            return None
        with self._lock:
            return None if self._closed else self._name

    def close(self) -> None:
        if self._pid != getpid():
            return
        with self._lock:
            self._closed = True
            self._name = None


_OWNER: ContextVar[ReferenceType[_RuleOwner] | None] = ContextVar(
    "quarry_active_rule_owner", default=None
)


def current_rule_name() -> str | None:
    """Return the active selected rule's name, or ``None`` outside its lifetime.

    An awaited adapter callback sees the rule that owns its call. A copied context
    cannot keep that ownership alive after the rule exits, including after a fork.
    This accessor performs no model call, logging, or input inspection.
    """
    reference = _OWNER.get()
    owner = reference() if reference is not None else None
    return None if owner is None else owner.name()


@contextmanager
def _rule_scope(name: str | None) -> Generator[None, None, None]:
    owner = _RuleOwner(name) if type(name) is str else None
    token = _OWNER.set(ref(owner) if owner is not None else None)
    try:
        yield
    finally:
        if owner is not None:
            owner.close()
        _OWNER.reset(token)
