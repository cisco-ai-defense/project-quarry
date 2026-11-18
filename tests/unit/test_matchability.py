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

"""Synthetic tests for the bounded formal matchability slice."""

import sys

import pytest

from quarry.config import QuarryConfig
from quarry.safety import _matchability
from quarry.safety._matchability import (
    _CONSTRUCTION_LIMIT_REASON,
    _MAX_CONSTRUCTION_ATOMS,
    _MAX_CONSTRUCTION_PATTERNS,
    _MAX_LITERAL_CHARS,
    _MAX_TOTAL_LITERAL_CHARS,
)
from quarry.safety.quality import (
    MatchabilityAnswer,
    MatchabilityBasis,
    matchability_warning,
    verify_matchability,
)
from quarry.scanner import DSLScanner


def _compiled(condition: str):
    source = f'rule "Formal check"\ncondition:\n    {condition}\n'
    scanner = DSLScanner(QuarryConfig(quarry_rule_texts=(source,)))
    return scanner.loaded_rules["Formal check"], scanner.compiled_rules["Formal check"]


def _contains_any_condition(patterns: list[str]) -> str:
    literals = ", ".join(f'"{pattern}"' for pattern in patterns)
    return f"any m in user_messages where m.content contains_any [{literals}]"


def test_missing_solver_returns_unknown_without_rejecting_the_rule(monkeypatch) -> None:
    _, artifact = _compiled('any m in user_messages where m.content == "hello"')
    monkeypatch.setitem(sys.modules, "z3", None)
    monkeypatch.delitem(sys.modules, "quarry.safety._matchability", raising=False)

    result = verify_matchability(artifact)

    assert result.answer is MatchabilityAnswer.UNKNOWN
    assert result.basis is None
    assert result.reason == "formal verification requires the 'verification' optional dependency"


def test_dev_environment_installs_the_verification_solver() -> None:
    import z3

    assert callable(z3.Solver)


def test_conflicting_message_equalities_are_proved_unmatchable() -> None:
    _, artifact = _compiled(
        'any m in user_messages where m.content == "hello" and m.content == "goodbye"'
    )

    result = verify_matchability(artifact)

    assert result.answer is MatchabilityAnswer.NO
    assert result.basis is MatchabilityBasis.CONTENT_MODEL
    assert result.reason == (
        "no feasible matching trace exists for the supported string-message predicates"
    )


def test_atom_limit_precedes_guard_path_expansion(monkeypatch) -> None:
    _, artifact = _compiled('any m in user_messages where m.content == "hello"')
    artifact.dfa.atoms[:] = [
        f"oversized_atom_{index}" for index in range(_MAX_CONSTRUCTION_ATOMS + 1)
    ]

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("guard paths must not expand before the atom limit")

    monkeypatch.setattr(_matchability, "_guard_paths", fail_if_called)

    result = verify_matchability(artifact)

    assert result.answer is MatchabilityAnswer.UNKNOWN
    assert result.basis is None
    assert result.reason == _CONSTRUCTION_LIMIT_REASON


@pytest.mark.parametrize(
    "condition",
    [
        pytest.param(
            _contains_any_condition(
                [f"pattern-{index}" for index in range(_MAX_CONSTRUCTION_PATTERNS + 1)]
            ),
            id="pattern-count",
        ),
        pytest.param(
            f'any m in user_messages where m.content == "{"x" * (_MAX_LITERAL_CHARS + 1)}"',
            id="single-literal",
        ),
        pytest.param(
            _contains_any_condition(
                [
                    f"{index:03d}"
                    + "x"
                    * (
                        _MAX_TOTAL_LITERAL_CHARS // _MAX_CONSTRUCTION_PATTERNS
                        + 1
                        - len(f"{index:03d}")
                    )
                    for index in range(_MAX_CONSTRUCTION_PATTERNS)
                ]
            ),
            id="total-literal-size",
        ),
    ],
)
def test_constraint_construction_limits_return_unknown_before_z3_terms(
    condition, monkeypatch
) -> None:
    import z3

    _, artifact = _compiled(condition)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Z3 string construction must not run past preflight limits")

    monkeypatch.setattr(z3, "StringVal", fail_if_called)

    result = verify_matchability(artifact)

    assert result.answer is MatchabilityAnswer.UNKNOWN
    assert result.basis is None
    assert result.reason == _CONSTRUCTION_LIMIT_REASON


def test_feasible_path_remains_unknown_until_witness_replay_exists() -> None:
    _, artifact = _compiled(
        'any m in user_messages where m.content == "hello" and m.content contains "ell"'
    )

    result = verify_matchability(artifact)

    assert result.answer is MatchabilityAnswer.UNKNOWN
    assert result.basis is None
    assert result.reason is not None
    assert "feasible model path" in result.reason


def test_unsupported_predicate_returns_unknown() -> None:
    _, artifact = _compiled('any m in user_messages where m.content ~ "hello.*"')

    result = verify_matchability(artifact)

    assert result.answer is MatchabilityAnswer.UNKNOWN
    assert result.basis is None
    assert result.reason == "formal verification does not support predicate 'regex'"


def test_exhausted_solver_check_budget_returns_unknown() -> None:
    _, artifact = _compiled(
        'any m in user_messages where m.content == "hello" and m.content == "goodbye"'
    )

    result = verify_matchability(artifact, max_solver_checks=1)

    assert result.answer is MatchabilityAnswer.UNKNOWN
    assert result.basis is None
    assert result.reason == "formal verification solver-check budget exhausted"


def test_no_proof_becomes_advisory_but_static_duplicate_does_not() -> None:
    rule, artifact = _compiled(
        'any m in user_messages where m.content == "hello" and m.content == "goodbye"'
    )
    result = verify_matchability(artifact)

    warning = matchability_warning(rule, artifact, result)

    assert warning is not None
    assert warning.severity.value == "high"
    assert "bounded string-message model" in warning.message

    static_rule, static_artifact = _compiled("false")
    static_result = verify_matchability(static_artifact)
    assert static_result.answer is MatchabilityAnswer.NO
    assert static_result.basis is MatchabilityBasis.ABSTRACT_DFA
    assert matchability_warning(static_rule, static_artifact, static_result) is None


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"max_solver_checks": 0}, "formal verification limits must be positive integers"),
        ({"max_states": True}, "formal verification limits must be positive integers"),
        ({"timeout_ms": -1}, "formal verification limits must be positive integers"),
    ],
)
def test_invalid_limits_fail_safe(kwargs, reason) -> None:
    _, artifact = _compiled('any m in user_messages where m.content == "hello"')

    assert verify_matchability(artifact, **kwargs).reason == reason
