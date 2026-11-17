# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Declared inputs preserve current configuration metadata and typed failure contracts."""

from dataclasses import replace

import pytest

import quarry.scanner as scanner_module
from quarry import DSLScanner, ScanFailureCategory, compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.input_registry import InputRegistry
from quarry.plugin.service import QuarryPlugin
from quarry.safety.limits import SafetyLimits
from quarry.safety.timer import QuarryTimeoutError
from quarry.types.exceptions import ModelError

INPUTS = {"memory.writes": {"facts": {"content": {"type": "string"}}}}
RULE = 'rule "Read" condition: any w in memory.writes where w.content == "x"'
DATA = {"memory": {"writes": [{"id": "w", "content": "x"}]}}


def configured(tmp_path):
    return QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "inputs": INPUTS,
            "models": {
                "guard": {
                    "adapter": "own",
                    "model": "own",
                    "description": "Inspect generic text",
                    "classify": {"class_index": 0},
                }
            },
        },
        tmp_path,
        require_rules=False,
    )


def test_reference_lists_model_metadata_and_inputs_together_without_inference(tmp_path):
    config = configured(tmp_path)
    reference = QuarryPlugin(config).reference(action="index")
    assert reference["model_aliases"]["guard"]["description"] == "Inspect generic text"
    assert reference["inputs"]["memory.writes"]["facts"]["content"] == {"type": "string"}
    assert "model_aliases" not in QuarryPlugin(config).reference(
        action="index", include_models=False
    )


def test_prepared_input_policy_keeps_descriptions_out_of_its_identity(tmp_path):
    config = configured(tmp_path)
    policy = compile_policy(RULE, runtime_config=config)
    changed = replace(config, model_descriptions={"guard": "Reword the host hint"})
    restored = load_compiled_policy(policy.to_bytes(), runtime_config=changed)
    assert restored.scan(DATA).rule_matches == ["Read"]


@pytest.mark.parametrize("declared", [False, True])
def test_conversation_preparation_keeps_typed_failure_classification(monkeypatch, declared):
    config = QuarryConfig(
        config_version=2,
        quarry_rule_texts=(RULE if declared else 'rule "Read" condition: true',),
        inputs=INPUTS if declared else {},
    )
    scanner = DSLScanner(config)

    def broken_context(*args, **kwargs):
        raise ModelError("owned preparation failure")

    monkeypatch.setattr(scanner_module, "ScanContext", broken_context)
    result = scanner.scan_data(
        DATA if declared else {"messages": [{"role": "user", "content": "x"}]}
    )
    assert result.error_category is ScanFailureCategory.PREPARATION
    assert result.evaluation_error_categories == {
        "conversation_preparation": ScanFailureCategory.MODEL
    }


@pytest.mark.parametrize("fail_on_timeout", [False, True])
def test_declared_input_preparation_honors_timeout_policy(monkeypatch, fail_on_timeout):
    scanner = DSLScanner(
        QuarryConfig(config_version=2, quarry_rule_texts=(RULE,), inputs=INPUTS),
        limits=SafetyLimits(fail_on_timeout=fail_on_timeout),
    )

    def expired(*args, **kwargs):
        raise QuarryTimeoutError("owned input preparation deadline")

    monkeypatch.setattr(InputRegistry, "prepare", expired)
    if fail_on_timeout:
        with pytest.raises(QuarryTimeoutError):
            scanner.scan_data(DATA)
    else:
        result = scanner.scan_data(DATA)
        assert result.completion_state == "incomplete"
        assert result.error_category is ScanFailureCategory.TIMEOUT
        assert all(failure.category is ScanFailureCategory.TIMEOUT for failure in result.failures)


def test_unexpected_input_preparation_failure_has_the_existing_preparation_category(monkeypatch):
    scanner = DSLScanner(QuarryConfig(config_version=2, quarry_rule_texts=(RULE,), inputs=INPUTS))

    def broken(*args, **kwargs):
        raise RuntimeError("owned preparation failure")

    monkeypatch.setattr(InputRegistry, "prepare", broken)
    result = scanner.scan_data(DATA)
    assert result.error_category is ScanFailureCategory.PREPARATION
    assert result.evaluation_error_categories == {"input_preparation": ScanFailureCategory.UNKNOWN}


@pytest.mark.parametrize("fail_on_timeout", [False, True])
def test_native_adapter_preparation_honors_timeout_policy(fail_on_timeout):
    from quarry import InputAdapter

    class Expired(InputAdapter):
        name = "expired"
        version = "1"
        sections = ("memory.writes",)

        def convert(self, native, wanted):
            raise QuarryTimeoutError("owned native preparation deadline")

    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            quarry_rule_texts=(RULE,),
            inputs=INPUTS,
            adapters={"raw": {"adapter": "expired"}},
        ),
        input_adapters={"expired": Expired()},
        limits=SafetyLimits(fail_on_timeout=fail_on_timeout),
    )
    if fail_on_timeout:
        with pytest.raises(QuarryTimeoutError):
            scanner.scan_data({"raw": {"items": []}})
    else:
        result = scanner.scan_data({"raw": {"items": []}})
        assert result.completion_state == "incomplete"
        assert result.error_category is ScanFailureCategory.TIMEOUT
