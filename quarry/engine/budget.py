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

"""The construction budget of one rule compile, measured in visit-equivalent work.

Every call of the formula walkers that building a rule's automata repeats (``temporal``'s
``simplify``, ``progress``, ``restrict``, ``_merge_next_uncached`` and ``_norm_repr``) is one
visit, charged to the budget of the compile running in this context. A visit costs about 0.6 µs
in flow, window and count rules and up to 2 µs in a 400-way ``or``. Eager expansions such as
count formulas and variadic flows charge estimated visit-equivalent work *before* allocating
their formulas. Hoisting, whose cost is mostly outside the walkers, is bounded by its own cap
(``desugar._MAX_HOISTED_PARTS``). Places where work multiplies (a guard's leaves and branches,
a new automaton state, an ``N of`` expansion) check or charge the budget and fail with
:class:`DFAConstructionLimit` before exhausting the available work.

The budget is held in a ``ContextVar``, so compiles that run concurrently (``asyncio.to_thread``
copies the context) each count their own visits.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

# The visit-equivalent work ``compile_rule`` allows the direct attempt of a rule, and all its
# per-candidate retries together, by default: about 3 seconds of building each.
DEFAULT_MAX_WORK = 5_000_000


class DFAConstructionLimit(Exception):
    """Raised when building a rule's automata exceeds the work budget. Signals "split this
    rule", not an internal error."""


class WorkBudget:
    """The visit-equivalent work allowed to one compile attempt; ``allowed=None`` counts
    without a limit. Walkers charge one per visit; eager expansions charge their projected
    construction and deduplication work before allocation."""

    __slots__ = ("allowed", "used")

    def __init__(self, allowed: int | None):
        self.allowed = allowed
        self.used = 0

    def check(self) -> None:
        """Check if the work budget has been exceeded."""
        if self.allowed is not None and self.used > self.allowed:
            raise DFAConstructionLimit(
                f"compiling the rule exceeded its work budget of {self.allowed:,} formula-node "
                "visits: the rule tracks too many conditions at once (for example, long "
                "`within` windows in several flows). Split the rule or shorten its windows."
            )

    def charge(self, visits: int) -> None:
        """Charge ``visits`` up front, failing before the work they stand for is done."""
        self.used += visits
        self.check()


# The budget of the compile running in this context; ``None`` outside one.
current = ContextVar[WorkBudget | None]("quarry_work_budget", default=None)


@contextmanager
def spending(budget: WorkBudget) -> Generator[WorkBudget, None, None]:
    """Charge the visits made inside the block to ``budget``."""
    token = current.set(budget)
    try:
        yield budget
    finally:
        current.reset(token)


def check() -> None:
    """Fail if the compile running in this context is past its budget."""
    budget = current.get()
    if budget is not None:
        budget.check()


def charge(visits: int) -> None:
    """Charge ``visits`` to the compile running in this context, failing if that is more than
    it has left."""
    budget = current.get()
    if budget is not None:
        budget.charge(visits)
