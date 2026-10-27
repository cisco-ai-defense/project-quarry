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

"""The automaton builder against finite-trace LTL, with past-time definitions.

Formulas mix LTLf operators with past definitions (``Y`` and bounded or unbounded ``O``) whose
tokens may appear under future operators and inside later definitions. The compiled automaton
must give the verdict that evaluating the formula directly on the trace gives.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from quarry.engine import temporal as T
from quarry.engine.budget import DFAConstructionLimit, WorkBudget, spending
from quarry.engine.symbolic_dfa import PAST_PREFIX, PastDef, compile_dfa, run_trace

_ATOMS = ["a", "b", "c"]


def _propositional(leaves: list[str]) -> st.SearchStrategy:
    base = st.sampled_from([T.Atom(name) for name in leaves])
    return st.recursive(
        base,
        lambda child: st.one_of(
            st.builds(T.Not, child), st.builds(T.And, child, child), st.builds(T.Or, child, child)
        ),
        max_leaves=4,
    )


@st.composite
def _past_definitions(draw) -> tuple[PastDef, ...]:
    definitions: list[PastDef] = []
    for k in range(draw(st.integers(min_value=0, max_value=3))):
        leaves = [*_ATOMS, *(d.token for d in definitions)]
        op = draw(st.sampled_from(["Y", "O"]))
        bound = (
            draw(st.one_of(st.none(), st.integers(min_value=0, max_value=3))) if op == "O" else None
        )
        definitions.append(PastDef(f"{PAST_PREFIX}{k}", op, draw(_propositional(leaves)), bound))
    return tuple(definitions)


def _formulas(leaves: list[str]) -> st.SearchStrategy:
    base = st.sampled_from([T.Atom(name) for name in leaves])

    def extend(child):
        bound = st.integers(min_value=0, max_value=4)
        return st.one_of(
            st.builds(T.Not, child),
            st.builds(T.And, child, child),
            st.builds(T.Or, child, child),
            st.builds(T.Implies, child, child),
            st.builds(T.Next, child),
            st.builds(T.Eventually, child),
            st.builds(T.Globally, child),
            st.builds(T.Until, child, child),
            st.builds(T.BoundedEventually, child, bound),
            # Shapes the state normalization rewrites: sibling Next, absorption, flows.
            st.builds(lambda x, y: T.Or(T.Next(x), T.Next(y)), child, child),
            st.builds(lambda x, y, z: T.Or(x, T.And(T.Or(x, y), z)), child, child, child),
            st.builds(
                lambda earlier, later, k: T.Eventually(
                    T.And(earlier, T.Next(T.BoundedEventually(later, k)))
                ),
                child,
                child,
                bound,
            ),
        )

    return st.recursive(base, extend, max_leaves=10)


@st.composite
def _cases(draw):
    definitions = draw(_past_definitions())
    formula = draw(_formulas([*_ATOMS, *(d.token for d in definitions)]))
    return formula, definitions


def _holds(formula, trace, i, definitions) -> bool:
    """Direct finite-trace semantics, past definitions read off the trace before ``i``."""
    n = len(trace)
    match formula:
        case T.Atom(name) if name in definitions:
            d = definitions[name]
            if d.op == "Y":
                return i > 0 and _holds(d.operand, trace, i - 1, definitions)
            start = 0 if d.bound is None else max(0, i - d.bound)
            return any(_holds(d.operand, trace, j, definitions) for j in range(start, i + 1))
        case T.Atom(name):
            return trace[i][name]
        case T.Not(child):
            return not _holds(child, trace, i, definitions)
        case T.And(left, right):
            return _holds(left, trace, i, definitions) and _holds(right, trace, i, definitions)
        case T.Or(left, right):
            return _holds(left, trace, i, definitions) or _holds(right, trace, i, definitions)
        case T.Implies(left, right):
            return not _holds(left, trace, i, definitions) or _holds(right, trace, i, definitions)
        case T.Next(child):
            return i + 1 < n and _holds(child, trace, i + 1, definitions)
        case T.Eventually(child):
            return any(_holds(child, trace, j, definitions) for j in range(i, n))
        case T.BoundedEventually(child, bound):
            return any(
                _holds(child, trace, j, definitions) for j in range(i, min(i + bound + 1, n))
            )
        case T.Globally(child):
            return all(_holds(child, trace, j, definitions) for j in range(i, n))
        case T.Until(left, right):
            return any(
                _holds(right, trace, j, definitions)
                and all(_holds(left, trace, k, definitions) for k in range(i, j))
                for j in range(i, n)
            )
    raise TypeError(formula)


_TRACES = st.lists(
    st.fixed_dictionaries({name: st.booleans() for name in _ATOMS}), min_size=1, max_size=7
)


@given(case=_cases(), traces=st.lists(_TRACES, min_size=1, max_size=8))
def test_automata_agree_with_finite_trace_semantics(case, traces):
    formula, definitions = case
    try:
        with spending(WorkBudget(1_000_000)):
            dfa = compile_dfa(formula, "property", past=definitions)
    except DFAConstructionLimit:
        return
    by_token = {d.token: d for d in definitions}
    for trace in traces:
        assert run_trace(dfa, trace) == _holds(formula, trace, 0, by_token), (
            formula,
            definitions,
            trace,
        )


@given(case=_cases())
def test_empty_trace_is_vacuously_satisfied(case):
    # The trace semantics above start at step 0, so they cannot read the empty word; the
    # automaton keeps the raw-LTLf convention that the empty word satisfies any formula.
    formula, definitions = case
    try:
        with spending(WorkBudget(1_000_000)):
            dfa = compile_dfa(formula, "property", past=definitions)
    except DFAConstructionLimit:
        return
    assert run_trace(dfa, []) is True, formula
