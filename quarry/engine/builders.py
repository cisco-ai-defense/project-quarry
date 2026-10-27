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

"""Optimized formula builders for common shapes.

Constructors for the recurring temporal shapes the compiler lowers to: bounded windows
and at-least-N counts. Every builder produces a formula that ``compile_dfa`` accepts.
"""

from __future__ import annotations

from quarry.engine.budget import charge
from quarry.engine.temporal import (
    TRUE,
    And,
    BoundedEventually,
    Eventually,
    Formula,
    Next,
)


def at_least(inner: Formula, n: int) -> Formula:
    """Temporal count: ``inner`` holds at **at least ``n`` distinct steps** of the trace.

    ``AtLeast(P, 1) = F(P)``; ``AtLeast(P, n) = F(P ∧ X(AtLeast(P, n-1)))`` — find a P
    step, then require n−1 more P steps strictly after it. This is the LTLf encoding of
    ``count(<surface/predicate>) >= n``. ``n <= 0`` is trivially true. Compiles to a DFA
    with an n-bounded counter in its state space.

    Each recursion level MUST be re-wrapped in ``Eventually``: the satisfying steps have to be
    increasing but NOT adjacent. Accumulating ``And(inner, Next(acc))`` without the wrapper
    yields ``F(P ∧ XP ∧ XXP …)``, which demands ``n`` CONSECUTIVE steps — a strictly stronger
    (and wrong) formula that a single non-matching event in between defeats.
    """
    if n <= 0:
        return TRUE
    # The nested formula is allocated before the DFA walkers can charge their visits.
    # Its construction is linear, while the repeated downstream walks are at least
    # quadratic in the count bound. Reserve that work before creating any nodes.
    charge(n * n)
    acc = Eventually(inner)
    for _ in range(n - 1):
        acc = Eventually(And(inner, Next(acc)))
    return acc


def bounded_eventually(inner: Formula, n: int) -> Formula:
    """``F≤n φ`` — φ holds now or within the next ``n`` steps.

    ``F≤0 φ = φ``; ``F≤n φ = φ ∨ X(F≤(n-1) φ)``. The LTLf encoding of a ``within:N``
    window, as one :class:`BoundedEventually` node so its depth does not grow with ``n``.
    """
    if n < 0:
        raise ValueError(f"bounded_eventually window must be >= 0, got {n}")
    return BoundedEventually(inner, n)
