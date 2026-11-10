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

"""Verdicts for declared sections, their observed state, and the shared clock."""

import pytest

from quarry import DSLScanner, InputAdapter, compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.plugin.service import QuarryPlugin
from quarry.safety.limits import SafetyLimits

INPUTS = {"memory.writes": {"facts": {"content": {"type": "string"}}}}
ITEM = {"id": "w1", "content": "x"}
CHAT = {
    "messages": [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "done"},
    ]
}
TIMELINE = [
    {"message_index": 0},
    {"message_index": 1},
    {"tool_call_id": "c1"},
    {"input": "memory.writes", "id": "w1"},
    {"message_index": 2},
]
FLOW = '(any w in memory.writes where w.id == "w1") right_after called_tool("tool_a")'


@pytest.mark.parametrize(
    ("condition", "data", "state", "matches"),
    [
        (
            'any w in memory.writes where w.content == "x"',
            {"memory": {"writes": [ITEM]}},
            "complete",
            True,
        ),
        ('any w in memory.writes where w.content == "x"', {"memory": {}}, "incomplete", False),
        ("count(memory.writes) == 0", {"memory": {"writes": []}}, "complete", True),
        (FLOW, {**CHAT, "memory": {"writes": [ITEM]}, "timeline": TIMELINE}, "complete", True),
        (FLOW, {**CHAT, "memory": {"writes": [ITEM]}}, "incomplete", False),
    ],
)
def test_declared_input_verdicts(condition, data, state, matches):
    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            quarry_rule_texts=(f'rule "R"\ncondition: {condition}',),
            inputs=INPUTS,
        ),
        limits=SafetyLimits(rule_timeout=None),
    )
    result = scanner.scan_data(data)
    assert (result.completion_state, result.rule_matches) == (state, ["R"] if matches else [])


def test_faulty_chat_fails_its_reader_only():
    config = QuarryConfig(
        config_version=2,
        quarry_rule_texts=(
            'rule "Memory"\ncondition: any w in memory.writes where w.content == "x"',
            'rule "Chat"\ncondition: user_said("go")',
        ),
        inputs=INPUTS,
    )
    result = DSLScanner(config, limits=SafetyLimits(rule_timeout=None)).scan_data(
        {"memory": {"writes": [ITEM]}, "messages": "bad"}
    )
    assert result.rule_matches == ["Memory"]
    assert "Chat" in result.evaluation_errors


def test_compiled_policy_checks_input_declaration():
    source = 'rule "R"\ncondition: any w in memory.writes where w.content == "x"'
    config = QuarryConfig(config_version=2, inputs=INPUTS)
    policy = compile_policy(source, runtime_config=config)
    restored = load_compiled_policy(policy.to_bytes(), runtime_config=config)
    assert restored.scan({"memory": {"writes": [ITEM]}}).rule_matches == ["R"]
    changed = QuarryConfig(
        config_version=2, inputs={"memory.writes": {"facts": {"content": {"type": "integer"}}}}
    )
    with pytest.raises(ValueError, match="declarations differ"):
        load_compiled_policy(policy.to_bytes(), runtime_config=changed)


def test_adapter_receives_only_read_sections():
    class Sample(InputAdapter):
        name = "sample"
        version = "1"
        sections = ("memory.writes",)

        def convert(self, native, wanted):
            assert wanted == frozenset({"memory.writes"})
            return {"memory.writes": native["items"]}

    config = QuarryConfig(
        config_version=2,
        quarry_rule_texts=('rule "R"\ncondition: any w in memory.writes where w.content == "x"',),
        inputs=INPUTS,
        adapters={"raw": {"adapter": "sample"}},
    )
    scanner = DSLScanner(
        config, input_adapters={"sample": Sample()}, limits=SafetyLimits(rule_timeout=None)
    )
    assert scanner.scan_data({"raw": {"items": [ITEM]}}).rule_matches == ["R"]


def test_plugin_lists_declared_inputs():
    plugin = QuarryPlugin(QuarryConfig(config_version=2, inputs=INPUTS))
    assert plugin.reference(action="index")["inputs"]["memory.writes"]["facts"]["content"] == {
        "type": "string"
    }
