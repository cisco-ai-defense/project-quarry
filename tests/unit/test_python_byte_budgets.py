# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Direct Python limits cannot silently turn an input byte cap into no bound."""

from dataclasses import replace
from enum import IntEnum

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.safety.limits import SafetyLimits


@pytest.mark.parametrize(
    "value",
    [True, False, -1, 1.25, float("nan"), float("inf"), -float("inf"), None, "64", b"64"],
    ids=[
        "true",
        "false",
        "negative",
        "fraction",
        "nan",
        "infinity",
        "negative-infinity",
        "none",
        "text",
        "bytes",
    ],
)
@pytest.mark.parametrize("replace_existing", [False, True])
def test_unsupported_python_byte_budgets_fail_at_construction(value, replace_existing):
    with pytest.raises(ValueError, match="max_total_size.*non-negative integer"):
        if replace_existing:
            replace(SafetyLimits(), max_total_size=value)
        else:
            SafetyLimits(max_total_size=value)


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("enforce", [False, True])
def test_zero_is_a_real_byte_budget_and_only_the_enforcement_switch_disables_it(enforce, restored):
    limits = SafetyLimits(max_total_size=0, enforce_limits=enforce, rule_timeout=None)
    config = QuarryConfig(config_version=2, safety_limits=limits)
    policy = compile_policy('rule "R" condition: true', runtime_config=config)
    if restored:
        wire = policy.to_bytes()
        policy = load_compiled_policy(wire, runtime_config=config)
        assert policy.to_bytes() == wire
    result = policy.scan({"messages": [{"role": "user", "content": "synthetic"}]})
    assert result.completion_state == ("invalid" if enforce else "complete")
    assert result.rule_matches == ([] if enforce else ["R"])
    if enforce:
        assert "size" in result.validation_error.lower()


def test_valid_integer_subclasses_remain_supported():
    class Budget(IntEnum):
        EMPTY = 0
        ORDINARY = 4096

    assert SafetyLimits(max_total_size=Budget.EMPTY).max_total_size == 0
    assert SafetyLimits(max_total_size=Budget.ORDINARY).max_total_size == 4096


def test_invalid_budget_diagnostics_do_not_format_the_supplied_object():
    class Opaque:
        def __repr__(self):
            raise AssertionError("unexpected host representation")

    with pytest.raises(ValueError, match="max_total_size.*non-negative integer"):
        SafetyLimits(max_total_size=Opaque())
