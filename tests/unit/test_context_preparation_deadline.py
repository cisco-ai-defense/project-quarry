# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Context construction observes the same inspection-timeout policy as other preparation."""

from dataclasses import replace

import pytest

import quarry.scanner as scanner_module
from quarry import DSLScanner, compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.safety.limits import SafetyLimits
from quarry.safety.timer import QuarryTimeoutError
from quarry.types.rules import ScanFailureCategory


@pytest.mark.parametrize("mode", ["scanner", "policy", "restored-policy"])
@pytest.mark.parametrize("fail_on_timeout", [False, True])
@pytest.mark.parametrize("timed_out", [False, True])
def test_context_preparation_preserves_timeout_policy(
    monkeypatch, mode, fail_on_timeout, timed_out
):
    config = QuarryConfig(
        config_version=2,
        safety_limits=SafetyLimits(rule_timeout=None, fail_on_timeout=fail_on_timeout),
    )
    source = 'rule "Read" condition: user_said("synthetic")'
    if mode == "scanner":
        prepared = DSLScanner(replace(config, quarry_rule_texts=(source,)))
        scan = prepared.scan_data
    else:
        prepared = compile_policy(source, runtime_config=config)
        if mode == "restored-policy":
            wire = prepared.to_bytes()
            prepared = load_compiled_policy(wire, runtime_config=config)
            assert prepared.to_bytes() == wire
        scan = prepared.scan

    def expired_context(*_args, **_kwargs):
        if timed_out:
            raise QuarryTimeoutError("Scan deadline exceeded")
        raise ValueError("synthetic context failure")

    monkeypatch.setattr(scanner_module, "ScanContext", expired_context)
    data = {"messages": [{"role": "user", "content": "synthetic"}]}
    if timed_out and fail_on_timeout:
        with pytest.raises(QuarryTimeoutError, match="Scan deadline exceeded"):
            scan(data)
    else:
        result = scan(data)
        assert not result.is_complete and result.rule_matches == []
        assert result.validation_error is None
        assert result.error_category == (
            ScanFailureCategory.TIMEOUT if timed_out else ScanFailureCategory.PREPARATION
        )
        if timed_out:
            assert result.error == "Input preparation timed out: Scan deadline exceeded"
        else:
            assert result.evaluation_errors == {
                "conversation_preparation": "ValueError: synthetic context failure"
            }
