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


async def test_plugin_digest_ignores_unread_declared_input():
    config = QuarryConfig(
        config_version=2,
        inputs={**INPUTS, "other.records": {"facts": {"value": {"type": "string"}}}},
    )
    plugin = QuarryPlugin(config)
    baseline = await plugin.scan(rule_text=RULE, conversation=DATA, explore_rate=1.0)
    with_unread = await plugin.scan(
        rule_text=RULE,
        conversation={**DATA, "other": object()},
        explore_rate=1.0,
    )

    assert with_unread["completion_state"] == "complete"
    assert (
        with_unread["decision_path"]["conversation_sha256"]
        == baseline["decision_path"]["conversation_sha256"]
    )

    invalid = await plugin.scan(
        rule_text=RULE,
        conversation={"memory": {"writes": object()}},
        explore_rate=1.0,
    )
    assert invalid["completion_state"] == "incomplete"
    assert invalid["decision_path"]["conversation_sha256"] is None

    class Hostile(dict):
        def items(self):
            raise AssertionError("overridden items must not run")

        def __contains__(self, key):
            raise AssertionError("overridden membership must not run")

    bad_namespace = await plugin.scan(
        rule_text=RULE,
        conversation={"memory": Hostile(DATA["memory"])},
        explore_rate=1.0,
    )
    assert bad_namespace["completion_state"] == "incomplete"
    assert bad_namespace["decision_path"]["conversation_sha256"] is None

    bad_root = await plugin.scan(rule_text=RULE, conversation=Hostile(DATA), explore_rate=1.0)
    assert bad_root["completion_state"] == "invalid"
    assert bad_root["decision_path"]["conversation_sha256"] is None

    class HostileKey:
        def __hash__(self):
            return hash("writes")

        def __eq__(self, other):
            raise AssertionError("untrusted key comparison must not run")

    bad_keys = await plugin.scan(
        rule_text=RULE,
        conversation={"memory": {HostileKey(): []}},
        explore_rate=1.0,
    )
    assert bad_keys["completion_state"] == "incomplete"
    assert bad_keys["decision_path"]["conversation_sha256"] is None


def test_malformed_json_container_faults_only_its_declared_input():
    class HostileList(list):
        def __iter__(self):
            raise AssertionError("untrusted iterator must not run")

    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            quarry_rule_texts=(RULE, 'rule "Chat" condition: user_said("go")'),
            inputs=INPUTS,
        )
    )
    result = scanner.scan_data(
        {"messages": [{"role": "user", "content": "go"}], "memory": {"writes": HostileList()}}
    )
    assert result.rule_matches == ["Chat"]
    assert "Read" in result.evaluation_errors

    class HostileId(str):
        def __hash__(self):
            raise AssertionError("untrusted ID hash must not run")

    result = scanner.scan_data(
        {
            "messages": [{"role": "user", "content": "go"}],
            "memory": {"writes": [{"id": HostileId("w"), "content": "x"}]},
        }
    )
    assert result.rule_matches == ["Chat"]
    assert "Read" in result.evaluation_errors


def test_chat_tool_arguments_keep_string_subclass_support_with_declared_inputs():
    class Argument(str):
        __hash__ = None

        def __str__(self):
            raise AssertionError("overridden string method must not run")

    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            quarry_rule_texts=('rule "Chat" condition: called_tool("lookup")',),
            inputs=INPUTS,
        )
    )
    data = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": Argument("{}")},
                    }
                ],
            }
        ]
    }
    assert scanner.scan_data(data).rule_matches == ["Chat"]


def test_prepared_input_policy_keeps_descriptions_out_of_its_identity(tmp_path):
    config = configured(tmp_path)
    policy = compile_policy(RULE, runtime_config=config)
    changed = replace(config, model_descriptions={"guard": "Reword the host hint"})
    restored = load_compiled_policy(policy.to_bytes(), runtime_config=changed)
    assert restored.scan(DATA).rule_matches == ["Read"]


def test_chat_size_limit_only_blocks_rules_that_read_chat():
    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            quarry_rule_texts=(RULE, 'rule "Chat" condition: user_said("hello")'),
            inputs=INPUTS,
        ),
        limits=SafetyLimits(max_total_size=2_000),
    )
    data = {**DATA, "messages": [{"role": "user", "content": "hello" * 2_000}]}
    assert scanner.scan_data(data, rules_subset={"Read"}).rule_matches == ["Read"]
    result = scanner.scan_data(data)
    assert result.rule_matches == ["Read"]
    assert result.completion_state == "incomplete"
    assert "max_total_size" in result.evaluation_errors["Chat"]


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
