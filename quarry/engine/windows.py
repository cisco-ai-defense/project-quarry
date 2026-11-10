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

"""The text ``text.extract(spans, left=, right=, head=)`` keeps: the head, and ``left`` and
``right`` characters around every span, merged. Each gap becomes a marker only when that
makes the text shorter in UTF-8 bytes, so no text is kept twice and the text never grows. A
text with no span is unchanged. Offsets count code points.

This module has no dependencies on scan state.
"""

from __future__ import annotations

from collections.abc import Sequence

# The marker that replaces a gap. It is not escaped: a text can already contain it. The
# marker and the merge rule decide what a compiled policy sends a judge, so changing either
# needs a compiler-version change.
WINDOW_MARKER = "…"

_BETWEEN = f"\n{WINDOW_MARKER}\n"
_BEFORE = f"{WINDOW_MARKER}\n"
_AFTER = f"\n{WINDOW_MARKER}"

Window = tuple[int, int]


def kept_windows(
    text: str, spans: Sequence[Window], left: int, right: int, head: int
) -> list[Window] | None:
    """The merged ranges of ``text`` to keep, or None when the text is kept unchanged."""

    if not spans:
        return None
    size = len(text)
    windows = sorted(
        [(max(0, start - left), min(size, end + right)) for start, end in spans]
        + ([(0, min(head, size))] if head > 0 else [])
    )
    merged = [windows[0]]
    for start, end in windows[1:]:
        last_start, last_end = merged[-1]
        if _shortens(text, last_end, start, _BETWEEN):
            merged.append((start, end))
        else:
            merged[-1] = (last_start, max(last_end, end))
    if not _shortens(text, 0, merged[0][0], _BEFORE):
        merged[0] = (0, merged[0][1])
    if not _shortens(text, merged[-1][1], size, _AFTER):
        merged[-1] = (merged[-1][0], size)
    if merged == [(0, size)]:
        return None
    return merged


def _shortens(text: str, start: int, end: int, marker: str) -> bool:
    """Whether replacing ``text[start:end]`` by ``marker`` makes the text shorter in bytes."""

    limit = len(marker.encode("utf-8"))
    # Every code point is at least one byte, so a gap with more code points is longer.
    return end - start > limit or len(text[start:end].encode("utf-8", "surrogatepass")) > limit


def render_windows(text: str, windows: Sequence[Window]) -> str:
    """``text`` with only ``windows`` kept and a marker in place of each gap."""

    body = _BETWEEN.join(text[start:end] for start, end in windows)
    before = _BEFORE if windows[0][0] > 0 else ""
    after = _AFTER if windows[-1][1] < len(text) else ""
    return before + body + after
