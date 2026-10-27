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

"""Data-only string operator nodes for rule patterns and sets.

Position, count, and set quantifiers carry structural fields the compiler uses to
create atoms. Runtime grounding uses the compiled engine's matchers.
"""

from __future__ import annotations

from dataclasses import dataclass

from quarry.types.dsl.base import ConditionNode


@dataclass
class StringPosition(ConditionNode):
    """String position operator (``@variable`` or ``@variable*``).

    Position of the first match of the string variable (name WITHOUT the ``$`` prefix).
    Used with ``in``: ``@password in msg.content``. ``text_to_search`` is set by the
    ``in`` operator during parsing/normalization.
    """

    variable: str
    text_to_search: str | None = None


@dataclass
class StringCount(ConditionNode):
    """String count operator (``#variable`` or ``#variable*``).

    Count of occurrences of the string variable (name WITHOUT the ``$`` prefix). Used with
    ``in``: ``#sql in msg.content``.
    """

    variable: str
    text_to_search: str | None = None


@dataclass
class StringSetQuantifier(ConditionNode):
    """String set quantifier (``any`` / ``all`` / ``N of ($patterns*)``).

    Used with ``in``: ``2 of ($patterns*) in msg.content``.
    """

    quantifier: str | int
    patterns: list[str]
    text_to_search: str | None = None
