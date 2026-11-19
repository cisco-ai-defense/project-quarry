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

"""Cooperative pauses for local engine work and ready host requests.

Managed workers pause every 64 steps. Inline evaluation retains elapsed-time pacing.
Pauses change when other tasks and deadlines run while preserving the result.
"""

from __future__ import annotations

import time

from quarry.engine.dispatch import current_dispatch

INTERVAL = 0.001


class Pacer:
    """Use an independent worker cadence, or sample the inline clock every ``every`` calls."""

    def __init__(self, every: int = 1) -> None:
        self._every = every
        self._calls = 0
        self._last = time.monotonic()
        self._managed = current_dispatch() is not None

    def due(self) -> bool:
        self._calls += 1
        if self._managed:
            return self._calls % 64 == 0
        if self._calls % self._every:
            return False
        now = time.monotonic()
        if now - self._last < INTERVAL:
            return False
        self._last = now
        return True
