# Copyright 2025 Cisco Systems, Inc. and its affilitates
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

"""How a scan answers input it cannot evaluate authoritatively.

Malformed or over-limit input is rejected as ``invalid`` before any rule runs -- never clipped to a
prefix that a rule could read as a clean verdict. A batch is only ``complete`` when every
conversation in it scanned authoritatively.
"""

import asyncio
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from quarry.config import QuarryConfig
from quarry.core.conversation import Conversation
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.safety.limits import SafetyLimits
from quarry.safety.regex_backend import RE2_REGEX
from quarry.safety.timer import QuarryTimeoutError
from quarry.safety.validator import validate_conversation, validate_conversation_arguments
from quarry.scanner import DSLScanner
from quarry.types import messages as canonical_messages
from quarry.types.exceptions import (
    ConversationValidationError,
    EvaluationError,
    IncompleteScanError,
)
from quarry.types.results import Score, Span

ALWAYS = 'rule "Always"\ncondition: true\n'


def _scanner(*rules: str, limits: SafetyLimits | None = None) -> DSLScanner:
    return DSLScanner(
        QuarryConfig(
            quarry_rule_texts=rules or (ALWAYS,),
            safety_limits=limits if limits is not None else SafetyLimits(rule_timeout=None),
        )
    )


def _user(*contents: str) -> dict:
    return {"messages": [{"role": "user", "content": content} for content in contents]}


def _call(name: str, call_id: str) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
        ],
    }


_FLOW_ACROSS_THREE_EVENTS = {
    "messages": [
        _call("start", "start"),
        {"role": "user", "content": "filler"},
        _call("end", "end"),
    ]
}
_NESTED_TOOL_PAYLOAD = {
    "messages": [{"role": "user", "content": "ok"}],
    "tools": [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"level": [{"payload": "x" * 10_000}]}},
        }
    ],
}


@pytest.mark.parametrize(
    ("conversation", "limits", "reason"),
    [
        pytest.param("not a dict", None, "Expected dict, got str", id="string"),
        pytest.param(["list"], None, "Expected dict, got list", id="list"),
        pytest.param({"tools": []}, None, "Missing required 'messages' key", id="no-messages"),
        pytest.param({"messages": "x"}, None, "'messages' must be a list", id="messages-not-list"),
        pytest.param(
            {"messages": [{"role": "user", "content": "t"}], "tools": "x"},
            None,
            "'tools' must be a list",
            id="tools-not-list",
        ),
        pytest.param(
            {"messages": [{"content": "test"}]},
            None,
            "Message at index 0 missing 'role' key",
            id="missing-role",
        ),
        pytest.param(
            {"messages": [{"role": "user"}]},
            None,
            "missing both 'content' and 'tool_calls'",
            id="missing-content",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": None}]},
            None,
            "content must be a string or OpenAI content-part list",
            id="null-user-content",
        ),
        pytest.param({"messages": []}, None, "'messages' list is empty", id="empty-messages"),
        pytest.param(
            _user(*(f"msg {i}" for i in range(500))),
            SafetyLimits(max_messages=100),
            "max_messages",
            id="over-message-count",
        ),
        pytest.param(
            _user("x" * 1_000_000),
            SafetyLimits(max_message_size=1000),
            "max_message_size",
            id="over-message-size",
        ),
        pytest.param(
            _NESTED_TOOL_PAYLOAD,
            SafetyLimits(max_total_size=5_000),
            "total_size",
            id="nested-tool-payload-counts-toward-total-size",
        ),
        pytest.param(
            _FLOW_ACROSS_THREE_EVENTS,
            SafetyLimits(max_timeline_items=2),
            "timeline_items",
            id="over-timeline-items",
        ),
    ],
)
def test_input_that_cannot_be_scanned_whole_is_invalid(conversation, limits, reason) -> None:
    result = _scanner(limits=limits).scan_data(conversation)

    assert result.rule_matches == []
    assert result.completion_state == "invalid"
    assert reason in result.validation_error
    assert result.error.startswith("Conversation validation error: ")
    assert result.completion_issues == [f"invalid conversation: {result.validation_error}"]


def test_total_size_counts_shared_python_objects_once_but_distinct_copies_separately() -> None:
    """The documented budget measures live Python objects, not serialized occurrences."""

    scanner = _scanner(limits=SafetyLimits(max_total_size=10_000))
    content = "shared" * 500
    shared = _user(*([content] * 8))
    copies = _user(*[(content + "!")[:-1] for _ in range(8)])

    assert [message["content"] for message in copies["messages"]] == [content] * 8
    shared_result = scanner.scan_data(shared)
    copies_result = scanner.scan_data(copies)

    assert (shared_result.completion_state, shared_result.rule_matches) == ("complete", ["Always"])
    assert (copies_result.completion_state, copies_result.rule_matches) == ("invalid", [])
    assert "max_total_size" in copies_result.validation_error


def test_malformed_tool_arguments_invalidate_the_whole_scan() -> None:
    scanner = _scanner(
        'rule "Names"\ncondition: called_tool("nonobject") and called_tool("nonfinite") and called_tool("good")',
        'rule "Nonobject"\ncondition: any c in get_tool_calls("nonobject") where c.arguments.value == 1',
        'rule "Nonfinite"\ncondition: any c in get_tool_calls("nonfinite") where c.arguments.value == 1',
        'rule "Good"\ncondition: any c in get_tool_calls("good") where c.arguments.value == 1',
    )
    calls = [
        {"id": "c1", "type": "function", "function": {"name": "nonobject", "arguments": "[]"}},
        {
            "id": "c2",
            "type": "function",
            "function": {"name": "nonfinite", "arguments": '{"value":NaN}'},
        },
        {"id": "c3", "type": "function", "function": {"name": "good", "arguments": '{"value":1}'}},
    ]

    result = scanner.scan_data({"messages": [{"role": "assistant", "tool_calls": calls}]})

    assert (result.rule_matches, result.completion_state) == ([], "invalid")
    assert "tool_calls[0].function.arguments" in result.validation_error


def test_a_flow_rejected_over_the_timeline_limit_matches_once_the_limit_allows_it() -> None:
    """Cutting the tail would have dropped ``end`` and answered the flow as absent."""

    rule = 'rule "Bounded Flow"\ncondition:\n    called_tool("end") after called_tool("start")\n'

    tight = _scanner(rule, limits=SafetyLimits(max_timeline_items=2))
    loose = _scanner(rule, limits=SafetyLimits(max_timeline_items=100))

    assert tight.scan_data(_FLOW_ACROSS_THREE_EVENTS).completion_state == "invalid"
    allowed = loose.scan_data(_FLOW_ACROSS_THREE_EVENTS)
    assert (allowed.rule_matches, allowed.completion_state) == (["Bounded Flow"], "complete")


@pytest.mark.parametrize(
    ("rule", "expected_operation"),
    [
        pytest.param(
            'rule "Evil Regex"\ncondition:\n    any m in user_messages where m.content ~ "(a+)+$"\n',
            "search",
            id="regex-match",
        ),
        pytest.param(
            'rule "Evil Regex"\nstrings:\n    $evil = "(a+)+$" regex\n'
            "condition:\n    any m in user_messages where #evil in m.content >= 1\n",
            "count",
            id="regex-count",
        ),
    ],
)
def test_a_catastrophic_backtracking_regex_scans_with_re2(
    rule, expected_operation, monkeypatch
) -> None:
    """The scanner must use its RE2 backend on regex search and count paths."""

    operations = []
    search = RE2_REGEX.search_match
    count = RE2_REGEX.count_up_to

    def record_search(*args, **kwargs):
        operations.append("search")
        return search(*args, **kwargs)

    def record_count(*args, **kwargs):
        operations.append("count")
        return count(*args, **kwargs)

    monkeypatch.setattr(RE2_REGEX, "search_match", record_search)
    monkeypatch.setattr(RE2_REGEX, "count_up_to", record_count)
    scanner = _scanner(rule)
    # Keep a finite input here; the backend assertion proves which regex engine ran.
    result = scanner.scan_data(_user("a" * 12 + "b"))

    assert expected_operation in operations
    assert result.rule_matches == []
    assert result.completion_state == "complete"


def test_an_empty_rule_subset_cannot_return_an_authoritative_clean_result() -> None:
    scanner = _scanner()

    clean = scanner.scan_data(_user("hello"), rules_subset=set())
    invalid = scanner.scan_data({"tools": []}, rules_subset=set())

    assert (clean.rule_matches, clean.completion_state) == ([], "incomplete")
    assert clean.error == "Rule selection error: explicit rule subset is empty"
    with pytest.raises(IncompleteScanError):
        clean.raise_for_status()
    assert invalid.completion_state == "invalid"


@pytest.mark.parametrize("subset", [{"Unknown"}, {"Always", "Unknown"}])
def test_unknown_rule_in_explicit_subset_fails_closed(subset: set[str]) -> None:
    result = _scanner().scan_data(_user("hello"), rules_subset=subset)

    assert result.rule_matches == []
    assert result.completion_state == "incomplete"
    assert result.error == "Rule selection error: unknown rule(s) selected: Unknown"
    with pytest.raises(IncompleteScanError):
        result.raise_for_status()


def test_a_known_rule_subset_still_evaluates_normally() -> None:
    result = _scanner().scan_data(_user("hello"), rules_subset={"Always"})

    assert (result.rule_matches, result.completion_state) == (["Always"], "complete")


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        pytest.param("{not json", "malformed JSON", id="malformed-json"),
        pytest.param("[1, 2]", "must decode to a JSON object", id="non-object-json"),
        pytest.param('{"value": NaN}', "non-finite number", id="non-finite-number"),
        pytest.param(
            '{"value": ' + "[" * 101 + "]" * 101 + "}",
            "max_nesting_depth",
            id="over-depth",
        ),
    ],
)
def test_invalid_tool_arguments_reject_even_a_name_only_rule(arguments: str, reason: str) -> None:
    conversation = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": arguments},
                    }
                ],
            }
        ]
    }
    result = _scanner('rule "Name"\ncondition: called_tool("lookup")\n').scan_data(conversation)

    assert (result.rule_matches, result.completion_state) == ([], "invalid")
    assert reason in result.validation_error


def test_argument_size_limit_runs_before_json_validation() -> None:
    conversation = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{not json"},
                    }
                ],
            }
        ]
    }
    result = _scanner(limits=SafetyLimits(max_tool_argument_size=4)).scan_data(conversation)

    assert result.completion_state == "invalid"
    assert "max_tool_argument_size" in result.validation_error


def test_valid_wide_integer_arguments_remain_accepted() -> None:
    # The rule-local decoder preserves exact integers within the active JSON
    # serializer's decimal-digit limit.
    arguments = '{"value": ' + "1" + "0" * 399 + "}"
    conversation = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": arguments},
                    }
                ],
            }
        ]
    }

    result = _scanner('rule "Name"\ncondition: called_tool("lookup")\n').scan_data(conversation)

    assert (result.rule_matches, result.completion_state) == (["Name"], "complete")
    # The direct validator retains its existing orjson parsing contract.
    with pytest.raises(ConversationValidationError, match="malformed JSON"):
        validate_conversation(conversation, parse_arguments=True)


@pytest.mark.skipif(sys.get_int_max_str_digits() == 0, reason="no active integer digit limit")
@pytest.mark.parametrize("extra_digit", [False, True], ids=["at-limit", "over-limit"])
@pytest.mark.parametrize("negative", [False, True], ids=["positive", "negative"])
def test_tool_integer_digit_limit_is_explicit_not_malformed(
    extra_digit: bool, negative: bool
) -> None:
    limit = sys.get_int_max_str_digits()
    digits = "1" * (limit + int(extra_digit))
    assert len(digits) < SafetyLimits().max_tool_argument_size
    integer_text = ("-" if negative else "") + digits
    conversation = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"value": ' + integer_text + "}",
                        },
                    }
                ],
            }
        ]
    }
    call = Conversation(conversation["messages"]).messages.tool_calls[0]
    scanner = _scanner('rule "Name"\ncondition: called_tool("lookup")\n')

    if extra_digit:
        with pytest.raises(EvaluationError, match="JSON integer digit limit") as error:
            _ = call.arguments
        assert f"active limit is {limit}" in str(error.value)
        result = scanner.scan_data(conversation)
        assert (result.rule_matches, result.completion_state) == ([], "invalid")
        assert f"active limit is {limit}" in result.validation_error
        assert "malformed JSON" not in result.validation_error
    else:
        assert call.arguments["value"] == int(integer_text)
        result = scanner.scan_data(conversation)
        assert (result.rule_matches, result.completion_state) == (["Name"], "complete")


def test_identical_tool_arguments_are_validated_once_per_scan(monkeypatch) -> None:
    arguments = '{"command": "read"}'
    conversation = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": arguments},
                    }
                    for index in range(3)
                ],
            }
        ]
    }
    decode = canonical_messages.json.loads
    calls: list[str] = []

    def counting_decode(value, *args, **kwargs):
        if value == arguments:
            calls.append(value)
        return decode(value, *args, **kwargs)

    monkeypatch.setattr(canonical_messages.json, "loads", counting_decode)
    scanner = _scanner(
        'rule "Name"\ncondition: any c in tool_calls where c.arguments.command == "read"\n'
    )

    first = scanner.scan_data(conversation)
    second = scanner.scan_data(conversation)

    assert (first.rule_matches, first.completion_state) == (["Name"], "complete")
    assert (second.rule_matches, second.completion_state) == (["Name"], "complete")
    # Each scan decodes once for preflight, then lazily for the rule read.
    assert calls == [arguments] * 4


def test_argument_preflight_does_not_retain_decoded_trees(monkeypatch) -> None:
    raw_values = ['{"value": [1, 2]}', '{"value": [3, 4]}', '{"value": [1, 2]}']
    messages = Conversation(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": arguments},
                    }
                    for index, arguments in enumerate(raw_values)
                ],
            }
        ]
    ).messages
    decode = canonical_messages.json.loads
    decoded: list[str] = []

    def counting_decode(value, *args, **kwargs):
        if value in raw_values:
            decoded.append(value)
        return decode(value, *args, **kwargs)

    monkeypatch.setattr(canonical_messages.json, "loads", counting_decode)
    validate_conversation_arguments(messages)

    assert decoded == raw_values[:2]
    assert all(call._arguments_cache is None for call in messages.tool_calls)
    assert messages.tool_calls[0].arguments == {"value": [1, 2]}
    assert decoded == raw_values[:2] + raw_values[:1]


@pytest.mark.parametrize("fail_on_timeout", [False, True])
def test_distinct_argument_preflight_stops_at_the_inspection_deadline(
    monkeypatch, fail_on_timeout: bool
) -> None:
    # A name-only rule still validates every argument. Distinct values defeat
    # the per-scan cache, so deadline checks must bound this eager pass.
    arguments = [f'{{"nonce": {index}}}' for index in range(100)]
    conversation = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": value},
                    }
                    for index, value in enumerate(arguments)
                ],
            }
        ]
    }
    decode = canonical_messages.json.loads
    decoded: list[str] = []

    def slow_decode(value, *args, **kwargs):
        if isinstance(value, str) and value.startswith('{"nonce":'):
            decoded.append(value)
            time.sleep(0.015)
        return decode(value, *args, **kwargs)

    monkeypatch.setattr(canonical_messages.json, "loads", slow_decode)
    scanner = _scanner(
        'rule "Name"\ncondition: called_tool("lookup")\n',
        limits=SafetyLimits(fail_on_timeout=fail_on_timeout),
    )

    if fail_on_timeout:
        with pytest.raises(QuarryTimeoutError, match="Scan deadline exceeded"):
            asyncio.run(scanner.scan_data_async(conversation, deadline_seconds=0.05))
    else:
        result = asyncio.run(scanner.scan_data_async(conversation, deadline_seconds=0.05))
    assert 0 < len(decoded) < len(arguments)
    if not fail_on_timeout:
        assert (result.rule_matches, result.completion_state) == ([], "incomplete")
        assert "Tool argument validation timed out: Scan deadline exceeded" in result.error
        with pytest.raises(IncompleteScanError):
            result.raise_for_status()


def test_invalid_argument_keeps_its_call_location_across_scans() -> None:
    scanner = _scanner('rule "Name"\ncondition: called_tool("lookup")\n')
    for bad_index in (1, 2):
        conversation = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{not json" if index == bad_index else "{}",
                            },
                        }
                        for index in range(3)
                    ],
                }
            ]
        }

        result = scanner.scan_data(conversation)

        assert result.completion_state == "invalid"
        assert result.validation_error is not None
        assert f"tool_calls[{bad_index}].function.arguments" in result.validation_error


def test_argument_cache_does_not_require_hashable_string_subclasses() -> None:
    class UnhashableString(str):
        __hash__ = None

    conversation = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": UnhashableString("{}"),
                        },
                    }
                ],
            }
        ]
    }

    result = _scanner('rule "Name"\ncondition: called_tool("lookup")\n').scan_data(conversation)

    assert (result.rule_matches, result.completion_state) == (["Name"], "complete")


_COUNTS = 'rule "Test"\ncondition: len(messages) > 0\n'


@pytest.mark.parametrize(
    ("batch", "limits", "matches", "state", "failed"),
    [
        pytest.param(
            [_user("hi"), _user("there")],
            None,
            ["conv_1:Test", "conv_2:Test"],
            "complete",
            [],
            id="every-conversation-clean",
        ),
        pytest.param(
            [_user("hi"), {"messages": []}],
            None,
            ["conv_1:Test"],
            "incomplete",
            ["conv_2"],
            id="an-invalid-conversation-blocks-complete",
        ),
        pytest.param(
            [_user("a", "b")],
            SafetyLimits(max_messages=1),
            [],
            "incomplete",
            ["conv_1"],
            id="an-over-limit-conversation-blocks-complete",
        ),
        pytest.param([], None, [], "invalid", [], id="empty-batch"),
    ],
)
def test_a_batch_is_complete_only_when_every_conversation_is(
    tmp_path, batch, limits, matches, state, failed
) -> None:
    result = _scanner(_COUNTS, limits=limits).scan_batch_data(batch, tmp_path / "batch.json")

    assert result.rule_matches == matches
    assert result.completion_state == state
    assert sorted(result.evaluation_errors or {}) == failed


def test_a_batch_scan_calls_a_two_argument_scan_data_override(tmp_path) -> None:
    # DSLScanner is public; a subclass that overrides scan_data(data, file_path) keeps
    # working in batch mode, which passes explain only when it is requested.
    class TwoArguments(DSLScanner):
        def scan_data(self, data, file_path=None):
            return super().scan_data(data, file_path)

    rule = 'rule "R" condition: any m in user_messages where m.content contains "a"'
    scanner = TwoArguments(QuarryConfig(quarry_rule_texts=(rule,)))

    result = scanner.scan_batch_data([_user("a")], tmp_path / "batch.json")

    assert result.rule_matches == ["conv_1:R"]


def test_an_incomplete_batch_fails_closed_on_raise_for_status(tmp_path) -> None:
    result = _scanner(_COUNTS).scan_batch_data(
        [_user("hi"), {"messages": []}], tmp_path / "batch.json"
    )

    with pytest.raises(IncompleteScanError):
        result.raise_for_status()


# ── collect: the producer's output is bounded and attributed per conversation ──────────────


class _SpanAdapter:
    """Fake model adapter: one classification and one span per character of the first message."""

    async def classify_messages(self, _config, _messages, _tools, *, timeout=None):
        return SimpleNamespace(chunks=[SimpleNamespace(probabilities=[0.9])])

    async def decode_spans(self, _config, _raw_result, messages, _threshold):
        content = messages[0].content
        return [Span(0, index, index + 1, Score(0.9)) for index in range(len(content))]


_SPANS = 'rule "Spans"\ncondition: true\ncollect:\n    [s for s in model("m").spans(messages, 0.5) if true]\n'


def _span_scanner(tmp_path: Path, **limits) -> DSLScanner:
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "m": {
                    "adapter": "fake",
                    "model": "span-model",
                    "category": "EMAIL",
                    "classify": {"class_index": 0},
                    "spans": {"class_index": 0},
                }
            },
            **limits,
        },
        tmp_path,
        require_rules=False,
    )
    config = replace(config, quarry_rule_texts=(_SPANS,))
    backend = ModelInferenceBackend(config, adapters={"fake": _SpanAdapter()})
    return DSLScanner(config, model_inference_backend=backend)


def test_a_collect_producer_over_the_comprehension_limit_fails_the_rule(tmp_path) -> None:
    result = _span_scanner(tmp_path, max_comprehension_size=1).scan_data(_user("ab"))

    assert result.rule_matches == []
    assert result.outputs == {}
    assert result.completion_state == "incomplete"
    assert result.evaluation_errors == {
        "Spans": "collect producer returned 2 items, exceeding max_comprehension_size=1"
    }


def test_batch_outputs_are_attributed_to_their_conversation(tmp_path) -> None:
    result = _span_scanner(tmp_path).scan_batch_data(
        [_user("ab"), _user("xyz")], tmp_path / "batch.json"
    )

    assert result.rule_matches == ["conv_1:Spans", "conv_2:Spans"]
    assert result.completion_state == "complete"
    assert {name: len(output.items) for name, output in result.outputs.items()} == {
        "conv_1:Spans": 2,
        "conv_2:Spans": 3,
    }


# ── a batch carries every conversation's completion signal, attributed to that conversation ──


class _SlowClassifier:
    """Fake model adapter whose classification outlasts any rule timeout used here."""

    async def classify_messages(self, _config, _messages, _tools, *, timeout=None):
        await asyncio.sleep(5)


def _slow_scanner(tmp_path: Path, *, fail_on_timeout: bool) -> DSLScanner:
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "rule_timeout": 0.05,
            "fail_on_timeout": fail_on_timeout,
            "models": {"m": {"adapter": "fake", "model": "slow", "classify": {"class_index": 0}}},
        },
        tmp_path,
        require_rules=False,
    )
    config = replace(
        config,
        quarry_rule_texts=('rule "Slow"\ncondition:\n    model("m").classify(messages) >= 0.5\n',),
    )
    backend = ModelInferenceBackend(config, adapters={"fake": _SlowClassifier()})
    return DSLScanner(config, model_inference_backend=backend)


_CAPPED_JOIN = (
    'rule "Capped"\ncondition:\n'
    "    (any m in user_messages where ((any c in tool_calls where c.arguments.cmd contains "
    'm.content) after called_tool("y")) and not (any d in tool_calls where d.name == "q" and '
    "d.arguments.cmd contains m.content)) or (all a in assistant_messages where a.content "
    'contains "z")\n'
)
_PAST_THE_CANDIDATE_CAP = {
    "messages": [
        *_user(*(f"m{i}" for i in range(257)))["messages"],
        {"role": "assistant", "content": "a"},
    ]
}


@pytest.mark.parametrize(
    ("make_scanner", "batch", "signals"),
    [
        pytest.param(
            lambda tmp_path: _slow_scanner(tmp_path, fail_on_timeout=False),
            [_user("hi")],
            (
                "Conversation 1: Rule Slow timeout",
                ["conv_1:Slow"],
                None,
                None,
                ["Conversation 1: Rule Slow timeout", "rules timed out: conv_1:Slow"],
            ),
            id="a-timed-out-rule",
        ),
        pytest.param(
            lambda tmp_path: _slow_scanner(tmp_path, fail_on_timeout=True),
            [_user("hi")],
            (
                "Conversation 1: Rule Slow timeout",
                None,
                {"conv_1": "Rule Slow timeout"},
                None,
                ["rule evaluation failed: conv_1"],
            ),
            id="a-timeout-raised-by-fail-on-timeout",
        ),
        pytest.param(
            lambda tmp_path: _span_scanner(tmp_path, max_comprehension_size=1),
            [_user("ab")],
            (
                None,
                None,
                {
                    "conv_1:Spans": (
                        "collect producer returned 2 items, exceeding max_comprehension_size=1"
                    )
                },
                None,
                ["rule evaluation failed: conv_1:Spans"],
            ),
            id="a-rule-evaluation-error",
        ),
        pytest.param(
            lambda _tmp_path: _scanner(_CAPPED_JOIN, limits=SafetyLimits(rule_timeout=60.0)),
            [_PAST_THE_CANDIDATE_CAP],
            (
                None,
                None,
                None,
                [
                    "conv_1: 'm' hit the candidate cap (256 user_messages items) without a "
                    "match; result may be incomplete"
                ],
                ["evaluation was limited"],
            ),
            id="a-candidate-cap-warning",
        ),
    ],
)
def test_a_batch_reports_each_conversations_incomplete_evaluation(
    tmp_path, make_scanner, batch, signals
) -> None:
    result = make_scanner(tmp_path).scan_batch_data(batch, tmp_path / "batch.json")

    assert (result.rule_matches, result.completion_state) == ([], "incomplete")
    assert (
        result.error,
        result.timed_out_rules,
        result.evaluation_errors,
        result.evaluation_warnings,
        result.completion_issues,
    ) == signals


# ── directories: every JSON file is scanned, and an unreadable one reports its own error ──


_SAID_HI = 'rule "Said hi"\ncondition:\n    user_said("hi")\n'
_UNREADABLE = "unexpected end of data: line 1 column 2 (char 1)"


@pytest.mark.parametrize(
    ("recursive", "verbose", "results", "progress"),
    [
        pytest.param(
            False,
            False,
            [("a.json", ["Said hi"], None), ("broken.json", [], _UNREADABLE)],
            "Scanning 2 files in {directory}\n[1/2] a.json\n[2/2] broken.json\n",
            id="top-level-files",
        ),
        pytest.param(
            True,
            False,
            [
                ("a.json", ["Said hi"], None),
                ("broken.json", [], _UNREADABLE),
                ("nested/b.json", [], None),
            ],
            "Scanning 3 files in {directory}\n[1/3] a.json\n[2/3] broken.json\n[3/3] b.json\n",
            id="recursive",
        ),
        pytest.param(
            False,
            True,
            [("a.json", ["Said hi"], None), ("broken.json", [], _UNREADABLE)],
            "",
            id="verbose-logs-instead-of-printing",
        ),
    ],
)
def test_a_directory_scan_answers_for_each_json_file(
    tmp_path, capsys, recursive, verbose, results, progress
) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.json").write_text('{"messages": [{"role": "user", "content": "hi"}]}')
    (tmp_path / "broken.json").write_text("{")
    (tmp_path / "nested" / "b.json").write_text(
        '{"messages": [{"role": "user", "content": "hello"}]}'
    )
    (tmp_path / "notes.txt").write_text("not a conversation")
    scanner = DSLScanner(
        QuarryConfig(quarry_rule_texts=(_SAID_HI,)), recursive=recursive, verbose=verbose
    )

    scanned = scanner.scan_directory(tmp_path)

    assert [
        (result.file_path.relative_to(tmp_path).as_posix(), result.rule_matches, result.error)
        for result in scanned
    ] == results
    assert capsys.readouterr().out == progress.format(directory=tmp_path)
