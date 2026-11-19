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

"""Declared inputs preserve eager chat validation and independent source clocks."""

from copy import deepcopy
from dataclasses import replace

import pytest

from quarry import DSLScanner, InputAdapter, compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.safety.limits import SafetyLimits

INPUTS = {
    "memory.writes": {"facts": {"content": {"type": "string"}}},
    "audit.records": {"facts": {"kind": {"type": "string"}}},
}
MEMORY = {"writes": [{"id": "w", "content": "safe"}]}
TOOLS = [
    {
        "type": "function",
        "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
    }
]


class ChatAdapter(InputAdapter):
    name = "owned-chat"
    version = "1"
    sections = ("messages",)

    def convert(self, native, wanted):
        assert wanted == frozenset({"messages"})
        return {"messages": native["messages"]}


def _chat(arguments: str) -> dict:
    return {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "bad-call",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": arguments},
                    },
                    {
                        "id": "good-call",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"q":"hi","limit":1208925819614629174706176}',
                        },
                    },
                ],
            }
        ],
    }


def _configuration(source: str, feature: str):
    config = QuarryConfig(
        config_version=2, quarry_rule_texts=(source,), safety_limits=SafetyLimits(rule_timeout=None)
    )
    if feature == "declared":
        return replace(config, inputs=INPUTS), {}
    return replace(config, adapters={"native": {"adapter": "owned-chat"}}), {
        "owned-chat": ChatAdapter()
    }


@pytest.mark.parametrize(
    "condition",
    [
        'called_tool("lookup") and called_tool("search")',
        'any c in get_tool_calls("lookup") where c.arguments.q contains "hi"',
        'any c in get_tool_calls("search") where c.arguments.q contains "hi"',
    ],
)
@pytest.mark.parametrize("feature", ["declared", "adapter"])
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("arguments", ["{not valid json", "[]"])
def test_chat_argument_validation_does_not_depend_on_unread_features(
    condition, feature, restored, explain, arguments
):
    source = f'rule "Chat" condition: {condition}'
    config, adapters = _configuration(source, feature)
    if restored:
        policy = compile_policy(source, runtime_config=config, input_adapters=adapters)
        policy = load_compiled_policy(
            policy.to_bytes(), runtime_config=config, input_adapters=adapters
        )
        result = policy.scan(_chat(arguments), explain=explain)
    else:
        result = DSLScanner(config, input_adapters=adapters).scan_data(
            _chat(arguments), enable_explain=explain
        )
    assert result.completion_state == "invalid"
    assert result.rule_matches == []
    assert result.validation_error


@pytest.mark.parametrize("adapted", [False, True])
@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("selected_chat", [False, True])
def test_bad_chat_arguments_do_not_fail_independent_inputs_tools_or_constants(
    adapted, explain, selected_chat
):
    config = QuarryConfig(
        config_version=2,
        inputs=INPUTS,
        safety_limits=SafetyLimits(rule_timeout=None),
        quarry_rule_texts=(
            'rule "Chat" condition: called_tool("lookup")',
            'rule "Memory" condition: any w in memory.writes where w.content == "safe"',
            'rule "Tools" condition: any t in tools where t.name == "lookup"',
            'rule "Constant" condition: true',
        ),
        adapters={"native": {"adapter": "owned-chat"}} if adapted else {},
    )
    data = {"memory": MEMORY, "tools": TOOLS}
    if adapted:
        data["native"] = _chat("{not valid json")
    else:
        data.update(_chat("{not valid json"))
    subset = {"Memory", "Tools", "Constant"} | ({"Chat"} if selected_chat else set())
    result = DSLScanner(
        config, input_adapters={"owned-chat": ChatAdapter()} if adapted else {}
    ).scan_data(data, enable_explain=explain, rules_subset=subset)
    assert set(result.rule_matches) == {"Memory", "Tools", "Constant"}
    assert result.completion_state == ("incomplete" if selected_chat else "complete")
    assert set(result.evaluation_errors or {}) == ({"Chat"} if selected_chat else set())


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("audit_fault", ["item", "namespace", "none"])
def test_unbounded_flow_ignores_a_faulty_section_read_by_another_rule(
    reverse, explain, audit_fault
):
    config = QuarryConfig(
        config_version=2,
        inputs=INPUTS,
        safety_limits=SafetyLimits(rule_timeout=None),
        quarry_rule_texts=(
            'rule "Flow" condition: (any w in memory.writes where w.id == "w") after user_said("go")',
            'rule "Bounded" condition: (any w in memory.writes where w.id == "w") after(within: 3) user_said("go")',
            'rule "Audit" condition: any a in audit.records where a.kind == "ok"',
            'rule "Constant" condition: true',
        ),
    )
    data = {
        "messages": [{"role": "user", "content": "go"}],
        "memory": MEMORY,
        "audit": {"records": [{"id": "a", "kind": 17 if audit_fault == "item" else "ok"}]},
    }
    if audit_fault == "namespace":
        data["audit"] = "invalid"
    entries = [
        {"message_index": 0},
        {"input": "audit.records", "id": "a"},
        {"input": "memory.writes", "id": "w"},
    ]
    data["timeline"] = list(reversed(entries)) if reverse else entries
    result = DSLScanner(config).scan_data(data, enable_explain=explain)
    assert ("Flow" in result.rule_matches) is (not reverse)
    assert "Constant" in result.rule_matches
    if audit_fault != "none":
        assert set(result.evaluation_errors) == {"Audit", "Bounded"}
        assert result.completion_state == "incomplete"
    else:
        assert not result.evaluation_errors
        assert ("Bounded" in result.rule_matches) is (not reverse)
        assert "Audit" in result.rule_matches


@pytest.mark.parametrize(
    "problem", ["unknown-section", "duplicate-item", "missing-item", "not-list"]
)
def test_clock_projection_does_not_hide_bad_timeline_structure(problem):
    config = QuarryConfig(
        config_version=2,
        inputs=INPUTS,
        safety_limits=SafetyLimits(rule_timeout=None),
        quarry_rule_texts=(
            'rule "Flow" condition: (any w in memory.writes where w.id == "w") after user_said("go")',
            'rule "Audit" condition: any a in audit.records where a.kind == "ok"',
            'rule "Constant" condition: true',
        ),
    )
    entries = [
        {"message_index": 0},
        {"input": "audit.records", "id": "a"},
        {"input": "memory.writes", "id": "w"},
    ]
    if problem == "unknown-section":
        entries.append({"input": "unknown.events", "id": "unknown"})
    elif problem == "duplicate-item":
        entries.append({"input": "memory.writes", "id": "w"})
    elif problem == "missing-item":
        entries.pop()
    data = {
        "messages": [{"role": "user", "content": "go"}],
        "memory": MEMORY,
        "audit": {"records": [{"id": "a", "kind": 17}]},
        "timeline": 42 if problem == "not-list" else entries,
    }
    result = DSLScanner(config).scan_data(data)
    assert result.rule_matches == ["Constant"]
    assert "Flow" in result.evaluation_errors and "Audit" in result.evaluation_errors


@pytest.mark.parametrize("bounded", [False, True])
def test_bad_chat_does_not_poison_an_independent_input_only_clock(bounded):
    op = "after(within: 3)" if bounded else "after"
    config = QuarryConfig(
        config_version=2,
        inputs=INPUTS,
        safety_limits=SafetyLimits(rule_timeout=None),
        quarry_rule_texts=(
            'rule "Chat" condition: called_tool("lookup")',
            f'rule "Flow" condition: (any w in memory.writes where w.id == "w2") {op} (any w in memory.writes where w.id == "w1")',
            'rule "Constant" condition: true',
        ),
    )
    data = _chat("{not valid json")
    data.update(
        {
            "memory": {"writes": [{"id": name, "content": "safe"} for name in ["w1", "w2"]]},
            "audit": {"records": []},
            "timeline": [
                {"message_index": 0},
                {"tool_call_id": "bad-call"},
                {"tool_call_id": "good-call"},
                {"input": "memory.writes", "id": "w1"},
                {"input": "memory.writes", "id": "w2"},
            ],
        }
    )
    result = DSLScanner(config).scan_data(data)
    assert set(result.rule_matches) == ({"Constant"} if bounded else {"Constant", "Flow"})
    assert set(result.evaluation_errors) == ({"Chat", "Flow"} if bounded else {"Chat"})


def test_valid_wide_argument_values_keep_exact_comparisons_with_declared_inputs():
    source = 'rule "Wide" condition: any c in get_tool_calls("search") where c.arguments.limit == 1208925819614629174706176'
    config, _ = _configuration(source, "declared")
    data = deepcopy(_chat('{"q":"hi"}'))
    result = DSLScanner(config).scan_data(data)
    assert result.is_complete and result.rule_matches == ["Wide"]


@pytest.mark.parametrize("intervening_item", [False, True])
def test_valid_clock_items_are_still_counted_by_bounded_flows(intervening_item):
    source = 'rule "Adjacent" condition: (any w in memory.writes where w.id == "w") after(within: 1) user_said("go")'
    config = QuarryConfig(
        config_version=2,
        inputs=INPUTS,
        quarry_rule_texts=(source,),
        safety_limits=SafetyLimits(rule_timeout=None),
    )
    entries = [{"message_index": 0}]
    if intervening_item:
        entries.append({"input": "audit.records", "id": "a"})
    entries.append({"input": "memory.writes", "id": "w"})
    data = {
        "messages": [{"role": "user", "content": "go"}],
        "memory": MEMORY,
        "audit": {"records": [{"id": "a", "kind": "ok"}] if intervening_item else []},
        "timeline": entries,
    }
    result = DSLScanner(config).scan_data(data)
    assert result.is_complete
    assert result.rule_matches == ([] if intervening_item else ["Adjacent"])


@pytest.mark.parametrize("feature", ["declared", "adapter"])
def test_argument_size_limits_precede_new_eager_decode(feature, monkeypatch):
    config, adapters = _configuration('rule "Chat" condition: called_tool("lookup")', feature)
    config = replace(
        config, safety_limits=SafetyLimits(rule_timeout=None, max_tool_argument_size=16)
    )

    def unexpected_decode(*args, **kwargs):
        pytest.fail("argument decoding happened before size-limit refusal")

    monkeypatch.setattr("quarry.scanner.validate_conversation_arguments", unexpected_decode)
    result = DSLScanner(config, input_adapters=adapters).scan_data(_chat("{" + "x" * 100))
    assert result.completion_state == "invalid" and result.rule_matches == []
    assert "max_tool_argument_size" in result.validation_error


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True])
def test_bad_chat_cannot_reach_a_model_or_explanation_call(restored, explain, monkeypatch):
    from quarry.integrations import model_inference

    class NoDispatch:
        def __init__(self):
            self.calls = 0

        async def score(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("malformed chat reached a model")

    adapter = NoDispatch()
    monkeypatch.setattr(model_inference, "_default_adapters", lambda: {"owned": adapter})
    source = 'rule "Model" condition: model("judge").score(messages, "Assess the sample") > 0.5'
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "inputs": INPUTS,
            "rule_timeout": None,
            "models": {
                "judge": {
                    "adapter": "owned",
                    "model": "owned-score",
                    "score": {
                        "message_builder": "constitution",
                        "boundary_seed": 9,
                        "tokens": {"negative": "0", "positive": "1"},
                    },
                }
            },
        },
        ".",
        require_rules=False,
    )
    config = replace(config, quarry_rule_texts=(source,))
    if restored:
        policy = compile_policy(source, runtime_config=config)
        policy = load_compiled_policy(policy.to_bytes(), runtime_config=config)
        result = policy.scan(_chat("{not valid json"), explain=explain)
    else:
        result = DSLScanner(config).scan_data(_chat("{not valid json"), enable_explain=explain)
    assert result.completion_state == "invalid" and result.rule_matches == []
    assert adapter.calls == 0
