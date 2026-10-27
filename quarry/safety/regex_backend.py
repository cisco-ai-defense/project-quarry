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

"""RE2 regex backend for the DFA engine.

The engine's matchers (`quarry.engine.matchers`) are a **pure core** module: their default
backend is stdlib ``re`` so the grounding path stays dependency-free and can be reasoned
about with no knowledge of Quarry. Production never uses that default. Quarry runs regex on
RE2 and only on RE2, and this module supplies RE2 to the engine through the matchers'
**injectable backend seam** (`matchers.RegexBackend`), which the scanner injects into the
oracle.

ReDoS is ruled out by the engine rather than by a timer. RE2 matches in time linear in the
input, so a pathological pattern cannot run away on this path and there is no blowup for an
inner timeout to interrupt. Bounding evaluation is left to the scanner's outer per-rule
`asyncio.timeout`, plus the task-local rule deadline that `regex_cache` checks whenever a
caller has installed one.

This module stays outside `quarry/engine/` because RE2 is reached through Quarry's shared
compile cache (`quarry.cache.regex_cache`), which the engine isolation invariant
(`tests/engine/test_async_runtime.py`) forbids engine modules from importing. Sharing the
cache is what keeps one RE2 options and rejection policy for all regex use.
"""

from __future__ import annotations

from quarry.cache.regex_cache import regex_cache
from quarry.engine.matchers import RegexBackend


class Re2RegexBackend(RegexBackend):
    """A :class:`~quarry.engine.matchers.RegexBackend` that grounds engine regex on RE2.

    Every operation delegates to the shared :mod:`quarry.cache.regex_cache`, so the engine
    path compiles through one LRU cache, one set of RE2 options, and
    one rejection policy: a pattern RE2 cannot compile raises ``re.error``. The cache also
    checks the task-local rule deadline, so where one is installed a rule that has already
    overrun aborts here instead of starting more matching work.
    """

    def search_match(self, pattern, text, flags=0):
        return regex_cache.search_match(pattern, text, flags)

    def findall(self, pattern, text, flags=0):
        return regex_cache.findall(pattern, text, flags)

    def iter_match_texts(self, pattern, text, flags=0):
        return regex_cache.iter_match_texts(pattern, text, flags)

    def count(self, pattern, text, flags=0):
        return regex_cache.count(pattern, text, flags)

    def count_up_to(self, pattern, text, limit, flags=0):
        return regex_cache.count_up_to(pattern, text, limit, flags)


# Stateless (the cache it delegates to is itself thread-safe), so one shared instance serves
# every rule and every scan.
RE2_REGEX = Re2RegexBackend()
