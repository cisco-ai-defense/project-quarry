# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Pattern extraction applies its match limit separately to each pattern."""

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.engine.matchers import NO_MODS, matching_patterns
from quarry.safety.limits import SafetyLimits


@pytest.mark.parametrize(
    ("patterns", "expected"),
    [
        (["alpha", "beta"], [((0, 5), (0,)), ((6, 10), (1,))]),
        (["alpha", "alpha", "beta"], [((0, 5), (0, 1)), ((6, 10), (2,))]),
        (["absent", "alpha", "beta"], [((0, 5), (1,)), ((6, 10), (2,))]),
    ],
)
def test_pattern_streams_keep_their_own_index(patterns, expected):
    assert list(matching_patterns("alpha beta", [(p, NO_MODS) for p in patterns])) == expected


@pytest.mark.parametrize(
    ("alpha", "beta", "keeps_tail"),
    [(128, 129, False), (256, 256, False), (257, 1, True), (1, 257, True)],
)
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True])
def test_extraction_counts_each_pattern_before_retaining_the_tail(
    alpha, beta, keeps_tail, restored, explain
):
    source = """rule "Extract"
strings:
    $p = ["alpha", "beta"]
condition:
    messages[0].content.extract($p.spans(messages[0].content)) contains "UNSELECTED_TAIL"
"""
    config = QuarryConfig(config_version=2, safety_limits=SafetyLimits(rule_timeout=None))
    policy = compile_policy(source, runtime_config=config)
    if restored:
        wire = policy.to_bytes()
        policy = load_compiled_policy(wire, runtime_config=config)
        assert policy.to_bytes() == wire
    result = policy.scan(
        {
            "messages": [
                {"role": "user", "content": "alpha " * alpha + "beta " * beta + "UNSELECTED_TAIL"}
            ]
        },
        explain=explain,
    )
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == (["Extract"] if keeps_tail else [])
