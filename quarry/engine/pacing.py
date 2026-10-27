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

"""How often long synchronous engine work hands control back to the event loop.

Grounding a local atom awaits nothing, so a run, a capture enumeration or a reference check
would otherwise hold the loop until it ends. Work pauses once ``INTERVAL`` has passed since
its last pause; pauses never change a result, only when other tasks and deadlines run.
"""

from __future__ import annotations

import time

INTERVAL = 0.001


class Pacer:
    """Says when a pause is due; reads the clock only every ``every`` calls, so a tight loop
    pays for it rarely."""

    def __init__(self, every: int = 1) -> None:
        self._every = every
        self._calls = 0
        self._last = time.monotonic()

    def due(self) -> bool:
        self._calls += 1
        if self._calls % self._every:
            return False
        now = time.monotonic()
        if now - self._last < INTERVAL:
            return False
        self._last = now
        return True
