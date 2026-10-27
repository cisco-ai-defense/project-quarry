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

"""Topic / goal-drift prober.

Grounds "is this action still on the user's original topic?" by comparing an action's
text against the **first user message** (the intent baseline). Returns a *drift score*
in ``[0.0, 1.0]`` — 0.0 = perfectly on-topic, 1.0 = fully off-topic.

The default backend is a dependency-free lexical heuristic (Jaccard overlap of content
words → similarity → ``1 - similarity``). If a real embedder is wired onto the
``ResourceManager`` (a ``.embed(text) -> vector`` callable exposed as ``embedder``),
``TopicProber`` uses cosine distance instead. The lexical fallback is intentionally
crude but deterministic: it is meant to make the drift signatures *runnable and testable*,
not to be the production topic model.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "you",
        "your",
        "me",
        "my",
        "we",
        "our",
        "they",
        "them",
        "this",
        "these",
        "those",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "please",
        "help",
        "me",
        "can",
        "you",
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
    ]
)

_WORD_RE = re.compile(r"[a-z0-9]+")


class _Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        """Embed the text."""


class TopicProber:
    """Scores how far an action has drifted from the original user intent.

    Args:
        off_topic_threshold: drift score at/above which ``off_topic`` is True.
        embedder: optional ``.embed(text) -> vector`` backend; falls back to a lexical
            heuristic when absent.
    """

    def __init__(self, off_topic_threshold: float = 0.7, embedder: _Embedder | None = None) -> None:
        self.off_topic_threshold = off_topic_threshold
        self.embedder = embedder

    def drift_score(self, intent: str, action: str) -> float:
        """Drift of ``action`` from ``intent`` — 0.0 (on-topic) .. 1.0 (off-topic)."""
        if not intent or not action:
            return 0.0
        if self.embedder is not None:
            sim = self._cosine(self.embedder.embed(intent), self.embedder.embed(action))
        else:
            sim = self._lexical_similarity(intent, action)
        return max(0.0, min(1.0, 1.0 - sim))

    def off_topic(self, intent: str, action: str) -> bool:
        """True iff ``action`` has drifted off-topic past the threshold."""
        return self.drift_score(intent, action) >= self.off_topic_threshold

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}

    def _lexical_similarity(self, a: str, b: str) -> float:
        """Jaccard overlap of content words → ``[0.0, 1.0]``."""
        ta, tb = self._tokens(a), self._tokens(b)
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        union = len(ta | tb)
        return inter / union if union else 0.0

    @staticmethod
    def _cosine(u: list[float], v: list[float]) -> float:
        if not u or not v or len(u) != len(v):
            return 0.0
        # strict= is safe: the length guard above already rejected mismatched vectors.
        dot = sum(x * y for x, y in zip(u, v, strict=True))
        nu = math.sqrt(sum(x * x for x in u))
        nv = math.sqrt(sum(y * y for y in v))
        if nu == 0.0 or nv == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (nu * nv)))
