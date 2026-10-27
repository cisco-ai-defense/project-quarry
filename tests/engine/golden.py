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

"""Frozen interpreter oracle.

The verdicts of the legacy AST interpreter, which is no longer in the tree, are frozen into
a committed golden snapshot (:data:`GOLDEN_PATH`); the DFA engine's parity tests assert
against that snapshot as a regression snapshot rather than a live cross-check.

The corpus test calls :func:`frozen_verdict(rule_text, conversation)` and gets the recorded
``(verdict, error)`` for that pair, keyed by a content hash of ``(rule_text, conversation)``.
A pair the golden never recorded returns ``None``: rules and datasets added after the freeze
have no interpreter verdict. Because the interpreter is gone, the golden can only be
regenerated from a revision that still has it — see the project history for the capture procedure (it
is not runnable on this tree).

External classifiers (`model_scan`/`model_score`/`llm_score`/`yara_scan`) are encoded
**False** in the golden; no rule the corpus test checks uses one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
GOLDEN_PATH = _HERE / "snapshots" / "interpreter_verdicts.json"


def _key(rule_text: str, conversation: dict[str, Any]) -> str:
    """Content hash of a (rule, conversation) pair — stable and name-independent.

    Normalizes the rule text (strip trailing whitespace) and canonicalizes the
    conversation JSON (sorted keys) so incidental formatting never shifts the key.
    """
    h = hashlib.sha256()
    h.update(rule_text.strip().encode("utf-8"))
    h.update(b"\x00")
    h.update(json.dumps(conversation, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def _load_golden() -> dict[str, Any]:
    if GOLDEN_PATH.exists():
        return json.loads(GOLDEN_PATH.read_text())
    return {}


_GOLDEN: dict[str, Any] = _load_golden()


def frozen_verdict(
    rule_text: str, conversation: dict[str, Any]
) -> tuple[bool | None, str | None] | None:
    """Return ``(verdict, error_class_name)`` the interpreter recorded for this pair, or
    ``None`` when the pair was never recorded (a rule or dataset added after the freeze).

    ``verdict`` is the interpreter's boolean (or ``None`` if it raised); ``error`` is the
    exception class name (or ``None``). Externals are grounded False.
    """
    record = _GOLDEN.get(_key(rule_text, conversation))
    if record is None:
        return None
    return record["verdict"], record["error"]
