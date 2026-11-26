# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Declared-input admission and transport never execute untrusted Python hooks."""

import pytest

from quarry import DSLScanner, compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.plugin.service import QuarryPlugin
from quarry.safety.limits import SafetyLimits

INPUTS = {"memory.writes": {"facts": {"content": {"type": "string"}}}}
RULE = 'rule "Read" condition: any w in memory.writes where w.content == "safe"'
FLOW = 'rule "Read" condition: (any w in memory.writes where w.content == "safe") after user_said("go")'


class HookDict(dict):
    touched = False

    def hook(self, *args, **kwargs):
        self.touched = True
        raise AssertionError("untrusted container hook executed")

    __iter__ = __len__ = __getitem__ = __bool__ = items = keys = values = get = hook


class HookList(list):
    touched = False

    def hook(self, *args, **kwargs):
        self.touched = True
        raise AssertionError("untrusted list hook executed")

    __iter__ = __len__ = __getitem__ = __bool__ = hook


class HookLeaf:
    touched = False

    @property
    def __class__(self):
        self.touched = True
        raise AssertionError("untrusted class hook executed")

    def __sizeof__(self):
        self.touched = True
        raise AssertionError("untrusted size hook executed")


def payload(position):
    data = {"memory": {"writes": [{"id": "w", "content": "safe"}]}}
    source = RULE
    if position == "top":
        hostile = HookDict(data)
        data = hostile
    elif position == "namespace":
        hostile = HookDict({"writes": []})
        data["memory"] = hostile
    elif position == "section":
        hostile = HookList([])
        data["memory"]["writes"] = hostile
    elif position == "item":
        hostile = HookDict({"id": "w", "content": "safe"})
        data["memory"]["writes"] = [hostile]
    elif position == "leaf":
        hostile = HookLeaf()
        data["memory"]["writes"][0]["content"] = hostile
    else:
        hostile = HookList([{"message_index": 0}, {"input": "memory.writes", "id": "w"}])
        data["messages"] = [{"role": "user", "content": "go"}]
        data["timeline"] = hostile
        source = FLOW
    return source, data, hostile


@pytest.mark.parametrize("position", ["top", "namespace", "section", "item", "leaf", "timeline"])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("enforce", [False, True])
def test_declared_admission_never_calls_container_class_or_size_hooks(position, prepared, enforce):
    source, data, hostile = payload(position)
    config = QuarryConfig(
        config_version=2,
        inputs=INPUTS,
        quarry_rule_texts=(source,),
        safety_limits=SafetyLimits(rule_timeout=None, enforce_limits=enforce),
    )
    if prepared:
        policy = compile_policy(source, runtime_config=config)
        policy = load_compiled_policy(policy.to_bytes(), runtime_config=config)
        result = policy.scan(data, explain=True)
    else:
        result = DSLScanner(config).scan_data(data, enable_explain=True)
    assert result.completion_state == ("invalid" if position == "top" else "incomplete")
    assert not result.rule_matches and not hostile.touched
    assert all(f.category.value == "invalid_input" for f in result.failures)


@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["top", "namespace", "section", "item", "leaf", "timeline"])
@pytest.mark.parametrize("explain", [False, True, "support"])
async def test_plugin_partial_input_fault_never_hashes_hostile_values(position, explain):
    source, data, hostile = payload(position)
    plugin = QuarryPlugin(
        QuarryConfig(config_version=2, inputs=INPUTS, safety_limits=SafetyLimits(rule_timeout=None))
    )
    result = await plugin.scan(
        rule_text=source, conversation=data, explain=explain, visualize=True, explore_rate=1.0
    )
    assert result["completion_state"] == ("invalid" if position == "top" else "incomplete")
    assert not hostile.touched
    assert result["decision_path"]["conversation_sha256"] is None
