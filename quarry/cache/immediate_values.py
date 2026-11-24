# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Optional completed immediate values, with weak owners and bounded frame storage."""

from __future__ import annotations

from weakref import ref

from quarry.cache.message_values import MAX_MESSAGE_VALUES, MessageValueBudget
from quarry.engine.local_work import ensure_local_work_authoritative
from quarry.integrations.read_usage import replay_read_usage
from quarry.safety.timer import raise_if_evaluation_expired

MAX_IMMEDIATE_BYTES = 1 << 20


class ImmediateValueMemo:
    """A frame's completed values; sibling frames share one optional storage allowance.

    Admission failure leaves ordinary evaluation intact. The cache owns no pending
    task or callback, and weak identity checks prevent recycled owner IDs becoming hits.
    """

    def __init__(self, budget: MessageValueBudget | None = None):
        self.budget = budget if budget is not None else MessageValueBudget(MAX_IMMEDIATE_BYTES)
        self.plans: dict = {}
        self.values: dict = {}
        self.closed = False

    def _require_open(self):
        if self.closed:
            raise RuntimeError("immediate value frame has been released")
        raise_if_evaluation_expired()
        ensure_local_work_authoritative()

    def _drop(self, table, key):
        entry = table.pop(key)
        entry[-1].close()
        self.budget.entries -= 1

    async def _retain(self, value):
        self._require_open()
        if self.budget.entries >= MAX_MESSAGE_VALUES:
            return None
        try:
            lease = await self.budget.retain(value)
        except ValueError:
            return None
        if self.closed:
            lease.close()
            self._require_open()
        try:
            raise_if_evaluation_expired()
            ensure_local_work_authoritative()
        except BaseException:
            lease.close()
            raise
        if self.budget.entries >= MAX_MESSAGE_VALUES:
            lease.close()
            return None
        return lease

    async def eligible(self, expression, classify):
        self._require_open()
        key = id(expression)
        entry = self.plans.get(key)
        if entry is not None:
            if entry[0]() is expression:
                return entry[1]
            self._drop(self.plans, key)
        result = await classify(expression)
        self._require_open()
        owner = ref(expression)
        data = key, owner, result
        lease = await self._retain(data)
        if lease is not None:
            if key in self.plans:
                lease.close()
            else:
                self.budget.entries += 1
                self.plans[key] = owner, result, lease
        return result

    def get(self, expression, backend):
        self._require_open()
        key = id(expression), id(backend)
        entry = self.values.get(key)
        if entry is None:
            return None
        if entry[0]() is not expression or entry[1]() is not backend:
            self._drop(self.values, key)
            return None
        replay_read_usage(entry[3])
        return entry[2]

    async def keep(self, expression, backend, data, logical_reads):
        self._require_open()
        key = id(expression), id(backend)
        if key in self.values:
            return
        try:
            owner, regex = ref(expression), ref(backend)
        except TypeError:
            return
        lease = await self._retain((key, owner, regex, data))
        if lease is None:
            return
        if key in self.values:
            lease.close()
            return
        self.budget.entries += 1
        # Logical-read replay has its own quotas and weak request ownership,
        # as for MessageValueMemo; it is not decision-value retained storage.
        self.values[key] = owner, regex, data, logical_reads, lease

    def close(self):
        self.closed = True
        for table in (self.values, self.plans):
            for key in tuple(table):
                self._drop(table, key)
