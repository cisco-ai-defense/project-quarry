# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Lazy named arguments and per-rule invalid-input attribution."""

from __future__ import annotations

import pytest

from quarry import InvalidInputError, compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.safety.limits import SafetyLimits
from quarry.scanner import DSLScanner
from quarry.types import messages as message_types
from quarry.types.messages import Messages
from tests.unit.model_fakes import ModelAdapter, model_entry, v2_scanner


def payload(raw, *, name="lookup"):
    return {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "synthetic-call",
                        "type": "function",
                        "function": {"name": name, "arguments": raw},
                    }
                ],
            }
        ]
    }


def scanner(**conditions):
    return DSLScanner(
        QuarryConfig(
            quarry_rule_texts=tuple(
                f'rule "{name}" condition: {condition}' for name, condition in conditions.items()
            )
        )
    )


@pytest.mark.parametrize("raw", ["{unfinished", "", "[]", "null", "true", "12", '"text"'])
def test_freeform_and_nonobject_arguments_have_no_named_fields(raw):
    messages = Messages.from_openai(payload(raw)["messages"])
    call = messages.tool_calls[0]
    assert call.arguments == {} and call.raw_arguments == raw
    result = scanner(Absent="any c in tool_calls where c.arguments.missing == null").scan_data(
        payload(raw)
    )
    assert result.is_complete and result.rule_matches == ["Absent"]


@pytest.mark.parametrize(
    "raw", ['{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}', "[" * 101 + "0" + "]" * 101]
)
def test_a_decoded_numeric_or_depth_fault_only_fails_its_reading_rule(raw):
    rules = scanner(
        Names='called_tool("lookup")', Read="any c in tool_calls where c.arguments.x != null"
    )
    result = rules.scan_data(payload(raw))
    assert result.rule_matches == ["Names"] and result.completion_state == "incomplete"
    assert result.validation_error is None
    assert set(result.evaluation_errors) == {"Read"}
    assert [failure.to_dict() for failure in result.failures] == [
        {"category": "invalid_input", "rule_name": "Read"}
    ]
    assert "messages[0].tool_calls[0].function.arguments" in result.evaluation_errors["Read"]


def test_name_only_scans_do_not_decode_even_unreadable_numeric_arguments(monkeypatch):
    raw = '{"marker": NaN}'
    original = message_types.json.loads

    def decode(value, *args, **kwargs):
        if value == raw:
            raise AssertionError("unused arguments were decoded")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(message_types.json, "loads", decode)
    result = scanner(Name='called_tool("lookup")').scan_data(payload(raw))
    assert result.is_complete and result.rule_matches == ["Name"]


def test_invalid_decodes_are_memoized_without_retaining_an_exception(monkeypatch):
    raw = '{"marker": NaN}'
    call = Messages.from_openai(payload(raw)["messages"]).tool_calls[0]
    original = message_types.json.loads
    count = 0

    def decode(value, *args, **kwargs):
        nonlocal count
        if value == raw:
            count += 1
        return original(value, *args, **kwargs)

    monkeypatch.setattr(message_types.json, "loads", decode)
    errors = []
    for _ in range(2):
        with pytest.raises(InvalidInputError) as error:
            _ = call.arguments
        errors.append(error.value)
    assert count == 1 and errors[0] is not errors[1]
    assert isinstance(call._arguments_error, str)
    assert raw not in str(errors[0]) and "marker" not in str(errors[0])
    call._discard_decoded_arguments()
    with pytest.raises(InvalidInputError):
        _ = call.arguments
    assert count == 2


def test_a_false_guard_does_not_read_invalid_arguments():
    result = scanner(
        Skip='called_tool("never") and (any c in tool_calls where c.arguments.x != null)'
    ).scan_data(payload('{"x": NaN}'))
    assert result.is_complete and result.rule_matches == []


def test_size_limits_still_reject_the_input_before_argument_decoding(monkeypatch):
    raw = '{"marker":' + "0" * 50 + "}"
    original = message_types.json.loads

    def decode(value, *args, **kwargs):
        if value == raw:
            raise AssertionError("over-limit arguments were decoded")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(message_types.json, "loads", decode)
    rules = DSLScanner(
        QuarryConfig(quarry_rule_texts=('rule "Always" condition: true',)),
        limits=SafetyLimits(max_tool_argument_size=8),
    )
    result = rules.scan_data(payload(raw))
    assert (
        result.completion_state == "invalid" and "max_tool_argument_size" in result.validation_error
    )


@pytest.mark.parametrize("raw", ["{unfinished", '{"x": NaN}'])
def test_model_input_preserves_raw_argument_text_when_no_named_value_is_read(tmp_path, raw):
    adapter = ModelAdapter()
    rules = v2_scanner(
        tmp_path,
        {"Model": 'model("reader").classify(messages) > 0.5'},
        {"reader": model_entry(classify=0)},
        adapter,
    )
    result = rules.scan_data(payload(raw))
    assert result.is_complete and result.rule_matches == ["Model"]
    assert adapter.classification_messages[0][0]["tool_calls"][0]["function"]["arguments"] == raw


def test_prepared_argument_faults_keep_typed_attribution_across_scans():
    source = 'rule "Read" condition: any c in tool_calls where c.arguments.x == 1'
    loaded = load_compiled_policy(compile_policy(source).to_bytes())
    bad = loaded.scan(payload('{"x": NaN}'))
    good = loaded.scan(payload('{"x": 1}'))
    assert (
        bad.completion_state == "incomplete" and bad.failures[0].category.value == "invalid_input"
    )
    assert good.is_complete and good.rule_matches == ["Read"]


def test_a_cached_argument_fault_does_not_poison_another_rules_raw_model_input(tmp_path):
    adapter = ModelAdapter()
    rules = v2_scanner(
        tmp_path,
        {
            "Bad": "any c in tool_calls where c.arguments.x != null",
            "Model": 'model("reader").classify(messages) > 0.5',
        },
        {"reader": model_entry(classify=0)},
        adapter,
    )
    result = rules.scan_data(payload('{"x": NaN}'))
    assert result.rule_matches == ["Model"] and result.completion_state == "incomplete"
    assert set(result.evaluation_errors) == {"Bad"}
    assert len(adapter.classification_messages) == 1
    assert (
        adapter.classification_messages[0][0]["tool_calls"][0]["function"]["arguments"]
        == '{"x": NaN}'
    )
