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

"""Bounded, host-neutral cache for compiled Quarry rule artifacts.

The agent plugin, Snowflake UDFs, and other long-lived hosts all receive rule text at
runtime. They can share bounded cache behavior without sharing a transport response shape
or a model-inference backend. Each host supplies the preparation function and a non-secret
runtime identity that distinguishes incompatible configurations.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from typing import Generic, TypeVar

from cachetools import LRUCache

Prepared = TypeVar("Prepared")
_CacheKey = tuple[str, str]


class PreparedRuleCache(Generic[Prepared]):
    """Keep prepared rules in a bounded worker-local LRU.

    ``runtime_identity`` must identify every trusted runtime setting that can change the
    prepared value, such as the Quarry/compiler version, configuration version, model
    profiles, text resources, and policy-bound parameters.  It must be a non-secret digest
    or version label: the cache deliberately stores it verbatim for equality checks.

    Preparation runs outside the lock, so concurrent callers can prepare the same key.
    Failed preparation is not cached, allowing a later call to retry.
    """

    def __init__(self, max_entries: int = 64) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self._values: LRUCache[_CacheKey, Prepared] = LRUCache(maxsize=max_entries)
        self._lock = threading.Lock()

    def get_or_prepare(
        self,
        rule_text: str,
        *,
        runtime_identity: str,
        prepare: Callable[[str], Prepared],
    ) -> Prepared:
        """Return a cached value or prepare and retain one for this key."""

        if not isinstance(rule_text, str) or not rule_text.strip():
            raise ValueError("rule_text must be a non-empty string")
        if not isinstance(runtime_identity, str) or not runtime_identity:
            raise ValueError("runtime_identity must be a non-empty string")
        if not callable(prepare):
            raise TypeError("prepare must be callable")

        key = (hashlib.sha256(rule_text.encode("utf-8")).hexdigest(), runtime_identity)
        with self._lock:
            if key in self._values:
                return self._values[key]

        prepared = prepare(rule_text)

        with self._lock:
            self._values[key] = prepared
        return prepared
