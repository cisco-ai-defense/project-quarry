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

"""Shared local primitive matchers.

Pure, synchronous predicate-extraction functions: given an explicit field value (or
values) and a pattern/modifier spec, each returns a boolean or scalar. These are the
leaf evaluators the DFA's atoms bind to — a matcher answers one boolean bit that
follows one guard transition. Matchers **never** see DFA state or rule logic;
composition, negation, flow, and counts are all handled by the guard graphs. They are
pure functions: no I/O, no mutable state, no non-determinism.

Scope is exactly the local predicates the DSL exposes, with the regex semantics of
`quarry/cache/regex_cache.py` (RE2 search, case-insensitive when `nocase`).

Deliberately **not** included, because the grammar does not expose them:
  - `entropy` (per-char Shannon entropy gate)
  - `ratio` (character-class fraction gate)
  - `fuzzy` (bounded-Levenshtein token match)
  - `decode` (base64/hex/url decode-then-match) — quarry has base64 *pattern* encoding
    via the `base64` modifier, but not field decoding
  - `normalize` (homoglyph/leet/zerowidth de-obfuscation)
  - `near` / `order` (intra-field positional proximity)
  - `capability` (tool-capability tag classification)
These slot in as new `predicate` kinds when the DSL grows; the `AtomSpec` field set
(`quarry/engine/atoms.py`) already reserves `Cost.DECODE`/`MODEL`/`LLM`/`YARA` tiers and
the `predicate`/`string_semantics` carriers for them.

This is a pure core module — no Quarry imports. `regex` uses the stdlib `re` directly;
production injects an RE2 backend through the seam below, which narrows the accepted
grammar rather than changing how these functions compose. Matchers stay pure.
"""

from __future__ import annotations

import base64 as _base64
import re
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Modifiers:
    """String-matching modifiers, mirroring quarry's ``StringModifiers`` (R2.1).

    Semantics:
      - ``base64`` re-encodes the *pattern* to its base64 form before matching.
      - ``regex`` treats the (post-base64) pattern as a regex, unescaped.
      - ``fullword`` escapes the pattern and wraps it in ``\\b…\\b`` (implies regex).
      - ``nocase`` folds case (regex: ``re.IGNORECASE``; literal: lowercase both sides).
    """

    nocase: bool = False
    fullword: bool = False
    base64: bool = False
    regex: bool = False


NO_MODS = Modifiers()


# Matchers stay a pure CORE module: by default they use stdlib `re` directly, so the
# DFA/grounding path is deterministic and dependency-free. Quarry itself runs regex on RE2
# and only on RE2, so the scanner injects `quarry.safety.regex_backend.RE2_REGEX` (backed by
# `regex_cache`) — the seam is what lets one matcher implementation serve both the pure
# default and production without pulling the cache into the pure core.
# (Divergence note: RE2's grammar is a strict subset of the stdlib's and its `$`/`\b` differ,
# so the pure default accepts and matches slightly more than production does. Rules are
# gated at load time by the RE2 cache, so nothing reaches production on the wider grammar.)


class RegexBackend:
    """Pure default regex backend (stdlib ``re``).

    Operations:
      - ``search_match`` — first match object.
      - ``findall`` — occurrence list.
      - ``count`` — occurrence count (the pure default just delegates to ``findall`` so
        offline runs stay dependency-free).

    Injecting a subclass (see ``quarry.safety.regex_backend``) is how production swaps in RE2
    without the engine importing the cache or safety layer (engine-isolation invariant).
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # A backend written for the old name would silently fall back to stdlib ``re`` for
        # capture enumeration, bypassing the backend's own safety.
        if hasattr(cls, "finditer_texts") and cls.iter_match_texts is RegexBackend.iter_match_texts:
            raise TypeError(
                f"{cls.__name__} overrides finditer_texts, which is now iter_match_texts "
                "(returning an iterator of match texts); rename the override"
            )

    def search_match(self, pattern: str, text: str, flags: int = 0) -> re.Match[str] | None:
        """Search for a match."""
        return re.search(pattern, text, flags)

    def findall(self, pattern: str, text: str, flags: int = 0) -> list:
        """Find all matches."""
        return re.compile(pattern, flags).findall(text)

    def iter_match_texts(self, pattern: str, text: str, flags: int = 0) -> Iterator[str]:
        """The matched substrings (``m.group(0)``) of every match, in order, lazily.

        Distinct from ``findall`` (which yields capture *groups*, not the whole match).
        Used by value-capture candidate enumeration; the RE2 backend overrides it so this
        path is linear too. The pattern compiles before returning, so an invalid
        pattern raises here rather than during iteration."""
        compiled = re.compile(pattern, flags)
        return (m.group(0) for m in compiled.finditer(text))

    def count(self, pattern: str, text: str, flags: int = 0) -> int:
        """Occurrence count. Default = ``len(findall)``; the RE2 backend overrides this so
        the count path streams matches instead of materializing them."""
        return len(self.findall(pattern, text, flags))

    def count_up_to(self, pattern: str, text: str, limit: int, flags: int = 0) -> int:
        """``min(count, limit)``. The default caps :meth:`count`, so an overriding backend's
        counting is kept; the RE2 backend stops at ``limit`` instead of counting the rest."""
        return min(self.count(pattern, text, flags), limit)


PURE_REGEX = RegexBackend()


def apply_modifiers(pattern: str, mods: Modifiers = NO_MODS) -> tuple[str, bool]:
    """Return ``(effective_pattern, use_regex)`` — the exact transform quarry applies.

    Order matches quarry: base64-encode the pattern first, then regex passthrough, then
    fullword-wrap. ``nocase`` is applied at match time, not here.
    """
    modified = pattern
    if mods.base64:
        modified = _base64.b64encode(pattern.encode()).decode()
    if mods.regex:
        return modified, True
    if mods.fullword:
        return rf"\b{re.escape(modified)}\b", True
    return modified, False


def _regex_flags(mods: Modifiers) -> int:
    return re.IGNORECASE if mods.nocase else 0


def match_contains(
    text: str, pattern: str, mods: Modifiers = NO_MODS, *, regex: RegexBackend = PURE_REGEX
) -> bool:
    """True if ``pattern`` occurs in ``text`` under ``mods``.

    Empty text or an empty pattern is a non-match, a fullword/regex/base64 pattern
    goes through the regex path, else a substring search (case-folded if ``nocase``).
    """
    return find_position(text, pattern, mods, regex=regex) is not None


def match_contains_any(
    text: str, patterns, mods: Modifiers = NO_MODS, *, regex: RegexBackend = PURE_REGEX
) -> bool:
    """True if ANY pattern in ``patterns`` matches (the ``contains_any`` list op)."""
    return any(match_contains(text, p, mods, regex=regex) for p in patterns)


def match_regex(
    text: str, pattern: str, nocase: bool = False, *, regex: RegexBackend = PURE_REGEX
) -> bool:
    """Boolean regex search — the ``~`` / ``matches`` operator semantics.

    Uses ``search`` (not fullmatch), matching quarry's ``regex_cache.search``. An invalid
    pattern raises ``re.error`` on the pure backend; on the RE2 path, rules are gated at
    load time, so an invalid pattern never reaches a scan."""
    if text is None or pattern is None:
        return False
    return (
        regex.search_match(str(pattern), str(text), _regex_flags(Modifiers(nocase=nocase)))
        is not None
    )


def find_position(
    text: str, pattern: str, mods: Modifiers = NO_MODS, *, regex: RegexBackend = PURE_REGEX
) -> int | None:
    """First match position, or ``None``."""
    if not text or not pattern:
        return None
    effective, use_regex = apply_modifiers(pattern, mods)
    if use_regex:
        m = regex.search_match(effective, text, _regex_flags(mods))
        return m.start() if m else None
    search_text = text.lower() if mods.nocase else text
    search_pat = effective.lower() if mods.nocase else effective
    pos = search_text.find(search_pat)
    return pos if pos != -1 else None


def count_occurrences(
    text: str,
    pattern: str,
    mods: Modifiers = NO_MODS,
    *,
    regex: RegexBackend = PURE_REGEX,
    limit: int | None = None,
) -> int:
    """Occurrence count, or ``limit`` once that many occur.

    The two paths differ: the regex path counts regex matches, the literal path uses
    ``str.count`` (non-overlapping).
    """
    if not text or not pattern:
        return 0
    effective, use_regex = apply_modifiers(pattern, mods)
    if use_regex:
        if limit is not None:
            # Stop at ``limit``: counting the rest cannot change a comparison it decides.
            return regex.count_up_to(effective, text, limit, _regex_flags(mods))
        # Route through the backend's count() (streamed on RE2, in the safety layer) rather
        # than materializing findall() on the occurrence-count path.
        return regex.count(effective, text, _regex_flags(mods))
    search_text = text.lower() if mods.nocase else text
    search_pat = effective.lower() if mods.nocase else effective
    found = search_text.count(search_pat)
    return found if limit is None else min(found, limit)


def extract_match(
    text: str, pattern: str, mods: Modifiers = NO_MODS, *, regex: RegexBackend = PURE_REGEX
) -> str | None:
    """The matched substring, or ``None`` (original casing preserved on the literal path)."""
    if not text or not pattern:
        return None
    effective, use_regex = apply_modifiers(pattern, mods)
    if use_regex:
        m = regex.search_match(effective, text, _regex_flags(mods))
        return m.group(0) if m else None
    search_text = text.lower() if mods.nocase else text
    search_pat = effective.lower() if mods.nocase else effective
    pos = search_text.find(search_pat)
    if pos == -1:
        return None
    return text[pos : pos + len(effective)]


_CMP = {
    "=": lambda a, b: a == b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
}


def compare(left, op: str, right) -> bool:
    """Relational/equality comparison. Raises ``ValueError`` on an unknown operator."""
    fn = _CMP.get(op)
    if fn is None:
        raise ValueError(f"unknown comparison operator {op!r}")
    return bool(fn(left, right))


def match_length(text: str, op: str, n: float) -> bool:
    """``len(text) <op> n`` size gate."""
    return compare(len(text or ""), op, n)


_MISSING = object()


def field_access(obj, path: str):
    """Traverse a dotted ``path`` into ``obj`` (dict/list/attr), returning the value or
    :data:`MISSING` if any segment is absent — preserving quarry's null-vs-missing
    distinction (a present ``None`` is returned as ``None``, an absent key is MISSING).

    Supports (mirroring quarry ``FieldAccess``):
      - dotted keys: ``arguments.command``
      - numeric segments as list/tuple indices: ``recipients.0.email``
      - ``*`` wildcard **as the last segment only**: returns all values of the current
        container — ``list(dict.values())`` for a dict, ``list(vars(obj).values())`` for
        an object (quarry raises if ``*`` is used mid-path; we mirror that as MISSING).
    """
    if not path:
        return obj
    cur = obj
    segments = path.split(".")
    for i, seg in enumerate(segments):
        if seg == "*":
            if i != len(segments) - 1:
                return _MISSING
            if isinstance(cur, dict):
                return list(cur.values())
            if isinstance(cur, (list, tuple)):
                return list(cur)
            if hasattr(cur, "__dict__"):
                return list(vars(cur).values())
            return _MISSING
        cur = _step(cur, seg)
        if cur is _MISSING:
            return _MISSING
    return cur


def _step(obj, seg: str):
    if isinstance(obj, dict):
        return obj.get(seg, _MISSING)
    if isinstance(obj, (list, tuple)):
        try:
            return obj[int(seg)]
        except (ValueError, IndexError):
            return _MISSING
    attr = getattr(obj, seg, _MISSING)
    return attr


MISSING = _MISSING


def truth(value) -> bool:
    """The truth of ``value`` used as a test. A :data:`MISSING` value is false."""
    return value is not _MISSING and bool(value)


def coerce_field_text(val) -> str | None:
    """Coerce an already-fetched field value to text, or ``None`` if missing or null.
    A list or tuple — from a ``*`` wildcard, an index, or a field whose own JSON value
    is an array — is joined so a downstream string matcher sees all elements."""
    if val is _MISSING or val is None:
        return None
    if isinstance(val, (list, tuple)):
        return "\n".join(str(v) for v in val)
    return str(val)


def field_text(obj, path: str) -> str | None:
    """Field value coerced to text, or ``None`` if missing. A list (from ``*`` or an
    index) is joined so a downstream string matcher sees all elements."""
    return coerce_field_text(field_access(obj, path))
