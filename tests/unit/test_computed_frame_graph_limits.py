# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Compilation bounds computed frames across intervening public rule calls."""

import pytest

from quarry.config import QuarryConfig
from quarry.types.exceptions import RuleValidationError
from tests.unit.test_nested_rule_runtime_contracts import _policy


def rules(depth):
    source = 'rule "Leaf" condition: user_said("needle")'
    previous = "Leaf"
    for index in range(depth):
        name = f"Frame{index}"
        source += (
            f'\nrule "{name}" imports: check="{previous}" condition: '
            f"any m in (messages[:{index + 1}] + messages[:0]) where check(m)"
        )
        previous = name
    return source, previous


@pytest.mark.parametrize("restored", [False, True])
def test_sixteen_computed_frames_round_trip_across_public_calls(restored):
    source, name = rules(16)
    compiled = _policy(source, QuarryConfig(config_version=2), restored)
    result = compiled.scan({"messages": [{"role": "user", "content": "needle"}]})
    assert result.is_complete and result.rule_matches == [name]


def test_seventeenth_computed_frame_is_rejected_during_compilation():
    source, _ = rules(17)
    with pytest.raises(RuleValidationError, match="computed message condition depth exceeds 16"):
        _policy(source, QuarryConfig(config_version=2), False)


@pytest.mark.parametrize("depth", [2, 3])
def test_aggregate_computed_count_includes_intervening_called_rules(monkeypatch, depth):
    import quarry.engine.contracts as contracts

    # Reduce only the aggregate limit; each independently compiled called rule
    # stays below the normal local family bound.
    monkeypatch.setattr(contracts, "MAX_COMPUTED_CONDITIONS", 2)
    source, name = rules(depth)
    if depth == 3:
        with pytest.raises(RuleValidationError, match="computed message condition count exceeds 2"):
            _policy(source, QuarryConfig(config_version=2), False)
    else:
        compiled = _policy(source, QuarryConfig(config_version=2), True)
        result = compiled.scan({"messages": [{"role": "user", "content": "needle"}]})
        assert result.is_complete and result.rule_matches == [name]
