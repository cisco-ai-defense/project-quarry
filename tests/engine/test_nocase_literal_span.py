# Copyright 2026 Cisco Systems, Inc. and its affilitates
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

"""Offsets of ``nocase`` literal matches when lower-casing changes the text length.

``"İ".lower()`` is two code points, so offsets found in ``text.lower()`` have to be mapped back
to ``text``. That mapping should only cost what a caller needs: a boolean ``contains`` check
skips it, and a position or capture stops once the match end is mapped.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from quarry.engine import matchers
from quarry.engine.matchers import Modifiers, _literal_span, extract_match, find_position

NOCASE = Modifiers(nocase=True)
# A message near the scale of real inputs: an early match, then a long run of characters
# whose lowered form is longer than the original.
_LONG_SUFFIX = "İ" * 500_000


def _reference_span(text: str, literal: str) -> tuple[int, int] | None:
    """Brute force: the first (start, end) whose lowered slice equals the lowered literal."""
    needle = literal.lower()
    for start in range(len(text)):
        for end in range(start + 1, len(text) + 1):
            if text[start:end].lower() == needle:
                return start, end
    return None


@pytest.mark.parametrize(
    ("text", "literal"),
    [
        ("İstanbul secret", "SECRET"),
        ("İİ secret İ", "secret"),
        ("xİy", "i̇y"),
        ("abc", "B"),
        ("İ", "i̇"),
        ("İabc", "zzz"),
        ("token İ end", "TOKEN"),
    ],
)
def test_literal_span_matches_a_brute_force_reading(text: str, literal: str) -> None:
    assert _literal_span(text, literal, nocase=True) == _reference_span(text, literal)


def test_contains_does_not_map_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("a boolean contains check should not map offsets")

    monkeypatch.setattr(matchers, "_literal_span", fail)
    text = "Secret" + _LONG_SUFFIX
    assert matchers.match_contains(text, "SECRET", NOCASE)
    assert not matchers.match_contains(text, "missing", NOCASE)
    assert matchers.match_contains_any(text, ["missing", "secret"], NOCASE)


class _StopAfter(str):
    """A ``str`` whose iteration fails past ``limit`` characters."""

    limit: int

    def __iter__(self) -> Iterator[str]:
        for index, char in enumerate(str.__iter__(self)):
            if index > self.limit:
                raise AssertionError(f"read past character {self.limit}")
            yield char


def test_position_and_capture_stop_after_the_match() -> None:
    text = _StopAfter("İ Secret" + _LONG_SUFFIX)
    text.limit = len("İ Secret")
    assert find_position(text, "SECRET", NOCASE) == 2
    assert extract_match(text, "SECRET", NOCASE) == "Secret"
