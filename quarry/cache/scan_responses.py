# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Raw model replies owned by active scan work, separately from the process cache."""

from __future__ import annotations

import os
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quarry.config import ModelCacheConfig

_SCOPES: weakref.WeakValueDictionary[int, ScanResponses] = weakref.WeakValueDictionary()


@dataclass(eq=False)
class _Reply:
    namespace: Any
    key: bytes
    response: Any
    readers: int = 1
    accepted: bool = False
    rejected: bool = False


class ScanResponses:
    """An identity token with explicitly leased, process-local response ownership.

    A reply being read is provisional. Concurrent readers use the same provisional reply,
    each validating it independently. The first successful read keeps it until the last
    scan/deferred-work lease closes. A rejection invalidates that reply for later readers.
    Merely retaining this identity token in a completed task does not retain its replies.
    """

    def __init__(self, cache_policy: ModelCacheConfig | None = None) -> None:
        self.pid = os.getpid()
        self._lock = threading.Lock()
        self._leases = 0
        self._generation = 0
        self._replies: dict[tuple[int, bytes], _Reply] = {}
        # Revisions invalidate outstanding offers without retaining rejected raw replies.
        self._revisions: dict[tuple[int, bytes], int] = {}
        self._retention_owner = object()
        self._release_callbacks: dict[object, Callable[[object], None]] = {}
        self.cache_policy = cache_policy
        _SCOPES[id(self)] = self

    def retain(self) -> bool:
        if self.pid != os.getpid():
            return False
        with self._lock:
            if not self._leases:
                self._generation += 1
                self._retention_owner = object()
            self._leases += 1
        return True

    def release(self) -> None:
        if self.pid != os.getpid():
            return
        removed = revisions = callbacks = None
        owner = None
        with self._lock:
            self._leases -= 1
            assert self._leases >= 0
            if not self._leases:
                removed, self._replies = self._replies, {}
                revisions, self._revisions = self._revisions, {}
                callbacks, self._release_callbacks = self._release_callbacks, {}
                owner = self._retention_owner
        if callbacks is not None:
            for callback in callbacks.values():
                callback(owner)
            callbacks.clear()
        # Dropping a user reply can run its destructor. Never do so under our lock.
        if removed is not None:
            removed.clear()
        if revisions is not None:
            revisions.clear()

    def retention_owner(self, generation: int | None) -> object | None:
        """A process-local generation token, without retaining this scope in a cache."""
        if self.pid != os.getpid():
            return None
        with self._lock:
            return (
                self._retention_owner if self._leases and generation == self._generation else None
            )

    def on_release(
        self, generation: int | None, namespace: object, callback: Callable[[object], None]
    ) -> object | None:
        """Register one internal cleanup per cache and scan generation."""
        if self.pid != os.getpid():
            return None
        with self._lock:
            if not self._leases or generation != self._generation:
                return None
            self._release_callbacks.setdefault(namespace, callback)
            return self._retention_owner

    def _cleanup_callback(
        self,
        generation: int | None,
        namespace: object,
        callback: Callable[[object], None] | None = None,
    ) -> Callable[[object], None] | None:
        """Find or atomically claim one native resource's cleanup in this generation.

        Callers construct candidates and dispose losing candidates outside this lock.
        The registered callback is the sole authoritative owner until the last lease.
        """
        if self.pid != os.getpid():
            return None
        with self._lock:
            if not self._leases or generation != self._generation:
                return None
            if callback is None:
                return self._release_callbacks.get(namespace)
            return self._release_callbacks.setdefault(namespace, callback)

    def _reset_after_fork(self) -> None:
        self._lock = threading.Lock()
        self._leases = 0
        self._generation += 1
        self._retention_owner = object()
        self._replies = {}
        self._revisions = {}
        self._release_callbacks = {}
        self.pid = os.getpid()

    def generation(self) -> int | None:
        if self.pid != os.getpid():
            return None
        with self._lock:
            return self._generation if self._leases else None

    def acquire(
        self, namespace: Any, key: bytes, generation: int
    ) -> tuple[_Reply | None, int | None]:
        if self.pid != os.getpid():
            return None, None
        with self._lock:
            if not self._leases or self._generation != generation:
                return None, None
            identity = (id(namespace), key)
            entry = self._replies.get(identity)
            if entry is not None:
                entry.readers += 1
            return entry, self._revisions.get(identity, 0)

    def offer(
        self, namespace: Any, key: bytes, response: Any, generation: int, revision: int | None
    ) -> _Reply | None:
        if self.pid != os.getpid():
            return None
        with self._lock:
            if not self._leases or self._generation != generation:
                return None
            identity = (id(namespace), key)
            if revision is None or self._revisions.get(identity, 0) != revision:
                # A shared provider waiter may resume after another reader rejected its
                # physical reply. It can validate its result, but cannot pin it again.
                return None
            entry = self._replies.get(identity)
            if entry is None:
                # Keeping the namespace prevents its id being reused during this scan.
                entry = _Reply(namespace, key, response)
                self._replies[identity] = entry
            else:
                entry.readers += 1
            return entry

    def reject(self, entry: _Reply, generation: int) -> None:
        if self.pid != os.getpid():
            return
        with self._lock:
            if self._generation == generation:
                entry.rejected = True
                identity = (id(entry.namespace), entry.key)
                if self._replies.get(identity) is entry:
                    del self._replies[identity]
                    self._revisions[identity] = self._revisions.get(identity, 0) + 1

    def finish(self, entry: _Reply, generation: int, accepted: bool) -> None:
        if self.pid != os.getpid():
            return
        with self._lock:
            if not self._leases or self._generation != generation:
                return
            entry.readers -= 1
            assert entry.readers >= 0
            entry.accepted |= accepted and not entry.rejected
            if not entry.accepted and not entry.readers:
                identity = (id(entry.namespace), entry.key)
                if self._replies.get(identity) is entry:
                    del self._replies[identity]


def _reset_scopes_after_fork() -> None:
    scopes = tuple(_SCOPES.values())
    removed = [(scope._replies, scope._revisions, scope._release_callbacks) for scope in scopes]
    for scope in scopes:
        scope._lock = threading.Lock()
    for scope in scopes:
        ScanResponses._reset_after_fork(scope)
    # Every scope and cache is fresh before a raw reply's destructor can re-enter one.
    removed.clear()


class ScanLease:
    """One idempotently closed owner of a scan's accepted-response lifetime."""

    def __init__(self, owner: ScanResponses | None) -> None:
        self._owner = owner if owner is not None and owner.retain() else None
        self._pid = os.getpid()
        self._lock = threading.Lock()

    def close(self) -> None:
        if self._pid != os.getpid():
            self._owner = None
            return
        with self._lock:
            owner, self._owner = self._owner, None
        if owner is not None:
            owner.release()


class ResponseReader:
    """Acquire one raw reply, then report this reader's acceptance or rejection.

    The caller controls its wait timeout and its validation separately. ``close`` is
    mandatory on all paths, including cancellation; it clears even provisional replies
    when nobody accepted them. Validation results are never memoized here.
    """

    def __init__(self, cache: Any, key: bytes, scope: ScanResponses | None) -> None:
        self._cache = cache
        self._key = key
        self._scope = scope
        self._generation = scope.generation() if scope is not None else None
        self._entry: _Reply | None = None
        self._response: Any = None
        self._called = False
        self._accepted = False
        self._closed = False

    async def call(self, func, *args, **kwargs) -> Any:
        entry = response = scope = cache = None
        try:
            if self._called or self._closed:
                raise RuntimeError("Response reader has already been used or closed")
            scope, generation = self._scope, self._generation
            cache = self._cache
            assert cache is not None
            entry, revision = (
                scope.acquire(cache, self._key, generation)
                if scope is not None and generation is not None
                else (None, None)
            )
            if entry is None:
                response = await cache.call(self._key, func, *args, **kwargs)
                if scope is not None and generation is not None:
                    entry = scope.offer(cache, self._key, response, generation, revision)
            else:
                response = entry.response
            if entry is not None:
                response = entry.response
            self._entry = entry
            self._response = response
            self._called = True
            return response
        finally:
            # An escaping exception keeps this frame. Do not add input closures or
            # provisional/accepted replies to the references it retains.
            del func, args, kwargs, entry, response, scope, cache, self

    def accept(self) -> None:
        assert self._called and not self._closed
        self._accepted = True

    def reject(self) -> None:
        if not self._called or self._closed:
            return
        if self._scope is not None and self._entry is not None and self._generation is not None:
            self._scope.reject(self._entry, self._generation)
        # An older reader must not remove a newer physical reply admitted after eviction.
        cache = self._cache
        assert cache is not None
        cache.forget(self._key, expected=self._response)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        scope, entry, generation = self._scope, self._entry, self._generation
        self._scope = self._entry = self._generation = self._response = self._cache = None
        self._key = b""
        if scope is not None and entry is not None and generation is not None:
            scope.finish(entry, generation, self._accepted)
