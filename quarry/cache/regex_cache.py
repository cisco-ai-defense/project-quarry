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

"""Reusable regex compilation and matching utilities, running on RE2."""

from __future__ import annotations

import re
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Any

import cachetools
import re2
from cachetools import LRUCache

# The module, not its names: the safety package imports this one while async_cache is still
# loading, so its result_cache is read when a match runs.
import quarry.cache.async_cache as async_cache
from quarry.cache.constants import REGEX_CACHE_SIZE
from quarry.cache.keys import input_key
from quarry.core.logging import get_app_logger
from quarry.safety.timer import raise_if_evaluation_expired

# RE2 builds its DFA lazily into a fixed memory budget, and when that budget runs out it does not
# fail -- it silently drops to a much slower path. The cliff belongs to one large merged union, not
# to many separate patterns: the 77 separate patterns of an unsafe-shell prefilter run ~6.5 ms at
# any budget, including RE2's own 8 MiB default, while the same patterns merged into a 5.5 KB union
# measured ~99 ms at 8, 16 and 32 MiB and then 0.214 ms at 48 MiB. 64 MiB keeps headroom above that
# edge for the merged shape. Do not lower it without re-measuring against a union that large.
RE2_MAX_MEM_BYTES = 64 * 1024 * 1024


logger = get_app_logger()


def _re2_options(pattern: str, flags: int) -> Any:
    """Build RE2 options for the two stdlib flags RE2 can represent, rejecting the rest."""

    unsupported = flags & ~(re.IGNORECASE | re.DOTALL)
    if unsupported:
        raise re.error(f"Regex flag is unsupported by RE2: {re.RegexFlag(unsupported)}", pattern)

    options = re2.Options()
    options.max_mem = RE2_MAX_MEM_BYTES
    # RE2 writes rejected patterns and "DFA out of memory" to stderr by default.
    options.log_errors = False
    # Case folding goes through Options, never a "(?i)" prefix: a prefix would also override a
    # pattern that scopes its own case flags.
    options.case_sensitive = not flags & re.IGNORECASE
    options.dot_nl = bool(flags & re.DOTALL)
    return options


def _cached(operation: str):
    """Keep a matching method's result in the process's result cache, under one digest of
    the operation and the method's arguments: matching depends on those alone, and the cache
    keeps no text."""

    return cachetools.cachedmethod(
        lambda _self: async_cache.result_cache.store,
        key=lambda _self, *args: input_key("regex", operation, *args),
        lock=lambda _self: async_cache.result_cache.lock,
    )


@dataclass(frozen=True)
class MatchSpan:
    """Where a pattern first matches a text: what callers read of a match. The result
    cache keeps only the span."""

    text: str
    begin: int
    finish: int

    def start(self) -> int:
        """Return the start index of the match."""
        return self.begin

    def end(self) -> int:
        """Return the end index of the match."""
        return self.finish

    def span(self) -> tuple[int, int]:
        """Return the start and end indices of the match."""
        return self.begin, self.finish

    def group(self, index: int = 0) -> str:
        """Return the substring of the match."""
        if index != 0:
            raise IndexError("a match span holds only its whole match")
        return self.text[self.begin : self.finish]


class RegexCache:
    """Compile and cache regex patterns on RE2, the only regex engine Quarry uses.

    RE2 matches in time linear in the input, so no pattern can be driven into backtracking and no
    inner match timeout is needed; rule deadlines still bound evaluation. RE2 is also the authority
    for accepted grammar, which is a strict subset of the stdlib's: a pattern using lookaround or
    backreferences raises ``re.error`` at compile time rather than being executed some other way.
    """

    def __init__(self) -> None:
        self._cache: LRUCache = LRUCache(maxsize=REGEX_CACHE_SIZE)
        self._lock = threading.RLock()

    def _compile_with_re2(self, pattern: str, flags: int) -> Any:
        options = _re2_options(pattern, flags)
        try:
            return re2.compile(pattern, options=options)
        except re2.error as exc:
            # RE2 names the construct it rejected -- "invalid perl operator: (?<=", "invalid escape
            # sequence: \1" -- so its own message is the most precise thing to hand back.
            detail = exc.args[0] if exc.args else exc
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", "replace")
            # Authored patterns may contain private data; keep them out of application logs.
            logger.warning("Unsupported regex pattern")
            raise re.error(f"Pattern is unsupported by RE2: {detail}", pattern) from exc
        except (ValueError, TypeError, OverflowError, RecursionError) as exc:
            # Keep the public exception contract stable whatever shape RE2 rejects a pattern in.
            logger.warning("Unsupported regex pattern")
            raise re.error(str(exc), pattern) from exc

    def compile(self, pattern: str, flags: int = 0) -> Any:
        """Compile a pattern on RE2, which decides what grammar is accepted.

        ``re`` accepts a superset of RE2's grammar, so pre-parsing with ``re`` could not reject
        anything RE2 accepts; RE2's own compile is the check.
        """

        if not isinstance(pattern, str):
            raise TypeError("Regex patterns must be strings")
        key = (pattern, flags)
        with self._lock:
            if key not in self._cache:
                self._cache[key] = self._compile_with_re2(key[0], flags)
            return self._cache[key]

    def _compiled_for_match(self, pattern: str, flags: int) -> Any:
        """Compile under the active rule deadline, checked on both sides of compilation."""

        raise_if_evaluation_expired()
        compiled = self.compile(pattern, flags)
        # Dynamic patterns are compiled after the active deadline is checked; re-check immediately
        # so compilation cannot become an unreported overrun before matching begins.
        raise_if_evaluation_expired()
        return compiled

    def search_match(self, pattern: str, text: str, flags: int = 0) -> MatchSpan | None:
        """Return where a pattern first matches, under any active rule deadline."""

        if pattern is None or text is None:
            return None
        if not isinstance(pattern, str):
            raise TypeError("Regex patterns must be strings")
        raise_if_evaluation_expired()
        text_str = str(text)
        span = self._first_span(pattern, text_str, flags)
        return None if span is None else MatchSpan(text_str, *span)

    def search(self, pattern: str, text: str, flags: int = 0) -> bool:
        """Return whether a pattern matches anywhere in the text."""

        return bool(self.search_match(pattern, text, flags))

    def count(self, pattern: str, text: str, flags: int = 0) -> int:
        """Count non-overlapping matches without materializing a result list."""

        if pattern is None or text is None:
            return 0
        if not isinstance(pattern, str):
            raise TypeError("Regex patterns must be strings")
        raise_if_evaluation_expired()
        return self._count(pattern, str(text), flags)

    def count_up_to(self, pattern: str, text: str, limit: int, flags: int = 0) -> int:
        """Count non-overlapping matches, stopping at ``limit``."""

        if pattern is None or text is None:
            return 0
        if not isinstance(pattern, str):
            raise TypeError("Regex patterns must be strings")
        raise_if_evaluation_expired()
        # islice rejects a stop above sys.maxsize, which a finite count bound can exceed; no
        # text has that many matches, so the cap never changes the count.
        return self._count_up_to(pattern, str(text), min(limit, sys.maxsize), flags)

    def findall(self, pattern: str, text: str, flags: int = 0) -> list:
        """Return every match's capture groups, as ``re.findall`` does."""

        if pattern is None or text is None:
            return []
        if not isinstance(pattern, str):
            raise TypeError("Regex patterns must be strings")
        raise_if_evaluation_expired()
        # A copy, so a caller that changes its list cannot change the cached one.
        return list(self._findall(pattern, str(text), flags))

    # The matching itself, kept in the process's result cache. A hit does no matching, so it
    # checks no deadline; a miss compiles under the active rule deadline.

    @_cached("search")
    def _first_span(self, pattern: str, text: str, flags: int) -> tuple[int, int] | None:
        match = self._compiled_for_match(pattern, flags).search(text)
        return None if match is None else match.span()

    @_cached("count")
    def _count(self, pattern: str, text: str, flags: int) -> int:
        count = 0
        for _ in self._compiled_for_match(pattern, flags).finditer(text):
            raise_if_evaluation_expired()
            count += 1
        return count

    @_cached("count-up-to")
    def _count_up_to(self, pattern: str, text: str, stop: int, flags: int) -> int:
        compiled = self._compiled_for_match(pattern, flags)
        return sum(1 for _ in islice(compiled.finditer(text), stop))

    @_cached("findall")
    def _findall(self, pattern: str, text: str, flags: int) -> tuple:
        return tuple(self._compiled_for_match(pattern, flags).findall(text))

    def iter_match_texts(self, pattern: str, text: str, flags: int = 0) -> Iterator[str]:
        """Yield every match's whole matched substring, in order.

        The pattern is compiled before returning, so an unsupported pattern raises here; the
        matches are produced lazily, so a caller can stop without materializing them all."""

        if pattern is None or text is None:
            return iter(())

        compiled = self._compiled_for_match(str(pattern), flags)
        return (m.group(0) for m in compiled.finditer(str(text)))


regex_cache = RegexCache()
