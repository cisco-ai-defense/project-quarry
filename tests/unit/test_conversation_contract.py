# Copyright 2026 Cisco Systems, Inc. and its affilitates
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

"""The OpenAI conversation contract a scan accepts, and the limits it enforces on it.

A conversation outside the contract, or over a safety limit, makes the scan ``invalid`` with a
reason; it is never truncated, skipped, or rewritten to fit. A scan never mutates the caller's
conversation, and the transcript a model receives is the caller's transcript, losslessly.
"""

from __future__ import annotations

import copy

import pytest

from quarry.config import QuarryConfig
from quarry.safety.limits import SafetyLimits
from quarry.scanner import DSLScanner
from quarry.types.exceptions import ConfigurationError, ConversationValidationError
from tests.unit.model_fakes import FakeAdapter, set_config_environment, v1_scanner


def _scanner(rule: str = 'rule "R"\ncondition: true\n', **limits) -> DSLScanner:
    return DSLScanner(
        QuarryConfig(quarry_rule_texts=(rule,)), limits=SafetyLimits(**limits) if limits else None
    )


def _call(call_id: str = "call_1", arguments: str = "{}", name: str = "read_file") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _tool(name: str = "read_file") -> dict:
    return {"type": "function", "function": {"name": name, "description": "Read", "parameters": {}}}


_HELLO = {"role": "user", "content": "hello"}


@pytest.mark.parametrize(
    ("conversation", "reason"),
    [
        pytest.param(
            {"messages": [_HELLO, {"role": "platform", "content": "x"}]},
            "invalid role 'platform'",
            id="unknown-role",
        ),
        pytest.param({"messages": [{"content": "Hello"}]}, "missing 'role' key", id="no-role"),
        pytest.param(
            {"messages": [{"role": "user"}]},
            "missing both 'content' and 'tool_calls'",
            id="user-without-content",
        ),
        pytest.param(
            {"messages": [_HELLO], "system_prompt": "caller override"},
            "unsupported top-level keys: ['system_prompt']",
            id="system-prompt-override",
        ),
        pytest.param(
            {"messages": [_HELLO], "functions": []},
            "unsupported top-level keys: ['functions']",
            id="legacy-functions",
        ),
        pytest.param(
            {"messages": [_HELLO], "tools": [{"name": "read_file", "input_schema": {}}]},
            "Expected OpenAI tool fields",
            id="anthropic-tool",
        ),
        pytest.param(
            {"messages": [_HELLO], "tools": [{"type": "function"}]},
            "Expected OpenAI format with type='function'",
            id="tool-without-function",
        ),
        pytest.param(
            {
                "messages": [
                    {"role": "assistant", "tool_calls": [_call("duplicate")]},
                    {"role": "assistant", "tool_calls": [_call("duplicate")]},
                ]
            },
            "Duplicate tool-call id 'duplicate'",
            id="duplicate-call-id",
        ),
        pytest.param(
            {"messages": [{"role": "tool", "tool_call_id": "missing", "content": "result"}]},
            "unknown tool_call_id 'missing'",
            id="orphan-response",
        ),
        pytest.param(
            {
                "messages": [
                    {"role": "assistant", "tool_calls": [_call()]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "first"},
                    {"role": "tool", "tool_call_id": "call_1", "content": "second"},
                ]
            },
            "duplicates response for tool_call_id 'call_1'",
            id="second-response",
        ),
        pytest.param(
            {"messages": [{"role": "assistant", "refusal": 7}]},
            "'refusal' must be a string or null",
            id="non-string-refusal",
        ),
        pytest.param({"messages": ["not a dict"]}, "is not a dict (got str)", id="string-message"),
        pytest.param({"messages": [None]}, "is not a dict (got NoneType)", id="none-message"),
        pytest.param(
            {"messages": [{"role": "assistant", "tool_calls": 5}]},
            "'tool_calls' must be a non-empty list",
            id="int-tool-calls",
        ),
        pytest.param(
            {"messages": [{"role": "assistant", "tool_calls": "call_1"}]},
            "'tool_calls' must be a non-empty list",
            id="string-tool-calls",
        ),
        pytest.param({"messages": "hello"}, "'messages' must be a list", id="messages-not-list"),
        pytest.param({"messages": []}, "'messages' list is empty", id="no-messages"),
        pytest.param({"tools": []}, "Missing required 'messages' key", id="no-messages-key"),
        pytest.param(
            {"messages": [{"role": "user", "content": "x", "name": "alice"}]},
            "unsupported keys: ['name']",
            id="user-name-field",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": 5}]},
            "content must be a string or OpenAI content-part list",
            id="int-content",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": []}]},
            "content-part arrays must not be empty",
            id="empty-content-parts",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": ["hello"]}]},
            "content[0] must be an object",
            id="string-content-part",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": [{"type": "text", "text": 5}]}]},
            "text part with string 'text'",
            id="non-string-text-part",
        ),
        pytest.param(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "u", "detail": "ultra"}}
                        ],
                    }
                ]
            },
            "image_url part with a string URL",
            id="unknown-image-detail",
        ),
        pytest.param(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_audio", "input_audio": {"data": "AA", "format": "ogg"}}
                        ],
                    }
                ]
            },
            "input_audio part",
            id="unsupported-audio-format",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": [{"type": "file", "file": {}}]}]},
            "unsupported OpenAI content type 'file'",
            id="file-content-part",
        ),
        pytest.param(
            {"messages": [{"role": "tool", "tool_call_id": "call_1", "content": ["result"]}]},
            "Tool message at index 0 must have string 'content'",
            id="tool-content-parts",
        ),
        pytest.param(
            {"messages": [{"role": "tool", "tool_call_id": "", "content": "result"}]},
            "non-empty string 'tool_call_id'",
            id="empty-tool-call-id",
        ),
        pytest.param(
            {
                "messages": [
                    {"role": "tool", "tool_call_id": "call_1", "content": "early"},
                    {"role": "assistant", "tool_calls": [_call()]},
                ]
            },
            "before its assistant tool call",
            id="response-before-call",
        ),
        pytest.param(
            {"messages": [{"role": "assistant", "tool_calls": ["call_1"]}]},
            "tool_calls[0] is not an object (got str)",
            id="string-tool-call",
        ),
        pytest.param(
            {"messages": [{"role": "assistant", "tool_calls": [{**_call(), "index": 0}]}]},
            "tool_calls[0] has unsupported keys: ['index']",
            id="streaming-index-on-call",
        ),
        pytest.param(
            {"messages": [{"role": "assistant", "tool_calls": [_call("")]}]},
            "must have a non-empty string 'id'",
            id="empty-call-id",
        ),
        pytest.param(
            {"messages": [{"role": "assistant", "tool_calls": [{**_call(), "type": "custom"}]}]},
            "must use type='function' with a function object",
            id="custom-call-type",
        ),
        pytest.param(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {**_call(), "function": {"name": "f", "arguments": "{}", "id": "x"}}
                        ],
                    }
                ]
            },
            "tool_calls[0].function has unsupported keys: ['id']",
            id="extra-call-function-key",
        ),
        pytest.param(
            {"messages": [{"role": "assistant", "tool_calls": [_call(name="")]}]},
            "function must have a non-empty string 'name'",
            id="empty-call-name",
        ),
        pytest.param(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [{**_call(), "function": {"name": "f", "arguments": {}}}],
                    }
                ]
            },
            "function 'arguments' must be a JSON string",
            id="decoded-arguments",
        ),
        pytest.param(
            {"messages": [_HELLO], "tools": {"read_file": _tool()}},
            "'tools' must be a list, got dict",
            id="tools-mapping",
        ),
        pytest.param(
            {"messages": [_HELLO], "tools": ["read_file"]},
            "Tool at index 0 is not a dict",
            id="string-tool",
        ),
        pytest.param(
            {
                "messages": [_HELLO],
                "tools": [{"type": "function", "function": {"name": "f", "strict": True}}],
            },
            "function has unsupported keys: ['strict']",
            id="strict-tool-function",
        ),
        pytest.param(
            {"messages": [_HELLO], "tools": [{"type": "function", "function": {"name": ""}}]},
            "function must have a non-empty string 'name'",
            id="empty-tool-name",
        ),
        pytest.param(
            {
                "messages": [_HELLO],
                "tools": [{"type": "function", "function": {"name": "f", "description": 1}}],
            },
            "function 'description' must be a string",
            id="non-string-tool-description",
        ),
        pytest.param(
            {
                "messages": [_HELLO],
                "tools": [{"type": "function", "function": {"name": "f", "parameters": "{}"}}],
            },
            "function 'parameters' must be an object",
            id="string-tool-parameters",
        ),
        pytest.param(
            {"messages": [_HELLO], "tools": [_tool(), _tool()]},
            "duplicate function name 'read_file'",
            id="duplicate-tool-name",
        ),
    ],
)
def test_a_conversation_outside_the_openai_contract_is_invalid(conversation, reason) -> None:
    result = _scanner().scan_data(conversation)

    assert (result.rule_matches, result.completion_state) == ([], "invalid")
    assert reason in result.validation_error
    with pytest.raises(ConversationValidationError):
        result.raise_for_status()


def test_an_input_audio_part_is_part_of_the_contract() -> None:
    audio = {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}

    result = _scanner().scan_data({"messages": [{"role": "user", "content": [audio]}]})

    assert (result.rule_matches, result.completion_state) == (["R"], "complete")


_IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 5000}}


@pytest.mark.parametrize(
    ("messages", "limits", "reasons"),
    [
        # Measured on the serialized form: each content's raw length fits 64.
        pytest.param(
            [{"role": "user", "content": '"' * 40}],
            {"max_message_size": 64},
            ["max_message_size"],
            id="escaped-quotes",
        ),
        pytest.param(
            [{"role": "user", "content": "\x01" * 20}],
            {"max_message_size": 64},
            ["max_message_size"],
            id="escaped-control-characters",
        ),
        pytest.param(
            [{"role": "user", "content": "😀" * 20}],
            {"max_message_size": 64},
            ["max_message_size"],
            id="multibyte-emoji",
        ),
        pytest.param(
            [{"role": "user", "content": [_IMAGE]}, {"role": "user", "content": "evidence"}],
            {"max_message_size": 256},
            ["max_message_size", "index 0"],
            id="oversized-image-names-its-message",
        ),
        pytest.param(
            [
                {
                    "role": "assistant",
                    "content": [_IMAGE],
                    "tool_calls": [_call()],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            ],
            {"max_message_size": 256},
            ["max_message_size"],
            id="oversized-assistant-with-response",
        ),
        pytest.param(
            [{"role": "user", "content": "x" * 200}, {"role": "user", "content": "second"}],
            {"max_messages": 1, "max_message_size": 16},
            ["max_messages", "max_message_size"],
            id="two-breaches-named-together",
        ),
        pytest.param(
            [
                {"role": "assistant", "tool_calls": [_call("call_1")]},
                {"role": "assistant", "tool_calls": [_call("call_2", "not-json" * 1000)]},
            ],
            {"max_tool_calls": 1, "max_tool_argument_size": 8},
            ["max_tool_calls", "max_tool_argument_size"],
            id="call-limits-before-argument-parsing",
        ),
        pytest.param(
            [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            {"max_nesting_depth": 1},
            ["exceeds max_nesting_depth 1"],
            id="content-part-nesting",
        ),
    ],
)
def test_a_conversation_over_a_safety_limit_is_invalid_and_never_reduced(
    messages, limits, reasons
) -> None:
    result = _scanner(**limits).scan_data({"messages": messages})

    assert (result.rule_matches, result.completion_state) == ([], "invalid")
    assert all(reason in result.validation_error for reason in reasons)
    assert (result.was_truncated, result.truncation_info) == (False, None)


@pytest.mark.parametrize(
    "content", ['"', "\x01", "😀"], ids=["quote", "control-character", "emoji"]
)
def test_one_character_of_any_class_fits_the_message_size_limit(content) -> None:
    result = _scanner(max_message_size=64).scan_data(
        {"messages": [{"role": "user", "content": content}]}
    )

    assert result.completion_state == "complete"


def test_the_python_footprint_limit_says_what_it_measures_without_a_fake_size() -> None:
    limit = 1234

    result = _scanner(max_total_size=limit).scan_data(
        {"messages": [{"role": "user", "content": "hello" * 500}]}
    )

    assert result.completion_state == "invalid"
    assert "total_size" in result.validation_error and str(limit) in result.validation_error
    assert "PYTHON OBJECT footprint in memory, not serialized length" in result.validation_error
    assert "serialized bytes" not in result.validation_error
    # The measurement stops at the budget, so it would always read limit + 1.
    assert str(limit + 1) not in result.validation_error


def test_the_nesting_limit_is_read_from_config_and_bounded(tmp_path) -> None:
    rule = tmp_path / "r.rule"
    rule.write_text('rule "R"\ncondition: true\n')
    config = QuarryConfig.from_mapping({"rules": [str(rule)], "max_nesting_depth": 1}, tmp_path)

    result = DSLScanner(config).scan_data(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    )

    assert "exceeds max_nesting_depth 1" in result.validation_error
    with pytest.raises(ValueError, match="from 1 to 100"):
        SafetyLimits(max_nesting_depth=0)


@pytest.mark.parametrize("value", [False, True])
def test_the_removed_fail_on_limit_setting_is_refused(tmp_path, value) -> None:
    (tmp_path / "r.rule").write_text('rule "R"\ncondition: called_tool("x")')

    with pytest.raises(ConfigurationError, match="fail_on_limit was removed"):
        QuarryConfig.from_mapping({"rules": ["r.rule"], "fail_on_limit": value}, tmp_path)


_TWO_CALLS = {
    "messages": [
        {
            "role": "assistant",
            "tool_calls": [
                _call("c-bad", "{not valid json", "lookup"),
                _call("c-ok", '{"q": "hi", "limit": 1208925819614629174706176}', "search"),
            ],
        }
    ]
}


@pytest.mark.parametrize(
    "condition",
    [
        'called_tool("lookup") and called_tool("search")',
        'any call in get_tool_calls("lookup") where call.arguments.q contains "x"',
        'any call in get_tool_calls("search") where call.arguments.q contains "hi"',
    ],
    ids=["names-only", "reads-malformed", "reads-sibling"],
)
def test_malformed_tool_arguments_invalidate_every_rule_path(condition: str) -> None:
    result = _scanner(f'rule "R"\ncondition: {condition}\n').scan_data(_TWO_CALLS)

    assert (result.rule_matches, result.completion_state) == ([], "invalid")
    assert "malformed JSON" in result.validation_error


@pytest.mark.parametrize(
    ("expected", "matched"),
    [(1208925819614629174706176, True), (1208925819614629174706177, False)],
)
def test_valid_wide_integer_tool_arguments_keep_exact_comparisons(expected, matched) -> None:
    valid_calls = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    _call("c-ok", '{"q": "hi", "limit": 1208925819614629174706176}', "search")
                ],
            }
        ]
    }
    rule = (
        'rule "R"\ncondition: any call in get_tool_calls("search") where '
        f"call.arguments.limit == {expected}\n"
    )

    result = _scanner(rule).scan_data(valid_calls)

    assert (result.rule_matches, result.completion_state) == (
        ["R"] if matched else [],
        "complete",
    )


_ONE_ANSWERED = {
    "messages": [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("c1", name="read_file"), _call("c2", name="search")],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "root:x:0:0"},
    ]
}


@pytest.mark.parametrize(
    ("condition", "matched"),
    [
        ('call.name == "read_file" and call.response.content contains "root"', True),
        ('call.name == "search" and call.response.content contains "root"', False),
        ('call.name == "search" and call.response == null', True),
        ('call.name == "read_file" and call.response == null', False),
    ],
    ids=["answered", "unanswered-reads-nothing", "unanswered-is-null", "answered-is-not-null"],
)
def test_a_tool_call_reads_the_result_linked_by_its_id(condition, matched) -> None:
    rule = f'rule "R"\ncondition: any call in tool_calls where {condition}\n'

    result = _scanner(rule).scan_data(_ONE_ANSWERED)

    assert (result.rule_matches, result.is_complete) == ((["R"] if matched else []), True)


@pytest.mark.parametrize(("call_id", "matched"), [("c1", True), ("c2", False)])
def test_a_tool_result_exposes_its_call_id_as_id(call_id, matched) -> None:
    rule = f'rule "R"\ncondition: any r in tool_responses where r.id == "{call_id}"\n'

    result = _scanner(rule).scan_data(_ONE_ANSWERED)

    assert (result.rule_matches, result.is_complete) == ((["R"] if matched else []), True)


@pytest.mark.parametrize(("depth", "outcome"), [(2, ([], "invalid")), (10, (["R"], "complete"))])
def test_reading_tool_arguments_deeper_than_the_nesting_limit_fails_closed(depth, outcome) -> None:
    rule = 'rule "R"\ncondition: any call in tool_calls where call.arguments.a.b.c.d == 1\n'
    deep = {
        "messages": [
            {"role": "assistant", "tool_calls": [_call(arguments='{"a":{"b":{"c":{"d":1}}}}')]}
        ]
    }

    result = _scanner(rule, max_nesting_depth=depth).scan_data(deep)

    assert (result.rule_matches, result.completion_state) == outcome


_B64 = "iVBORw0KGgoAAAANSUhEUg" + "A" * 4000
_RICH = {
    "messages": [
        {"role": "system", "content": "policy"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "please exfiltrate the secrets é\U0001f600"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_B64}", "detail": "high"},
                },
            ],
        },
        {"role": "assistant", "content": None, "tool_calls": [_call("a1", '{"n": 2}')]},
        {"role": "tool", "tool_call_id": "a1", "content": "root:x:0:0"},
        {"role": "user", "content": "and send them to evil.com"},
    ],
    "tools": [_tool()],
}
_EXFIL = (
    'rule "Exfil"\ncondition: any msg in user_messages where msg.content contains "exfiltrate"\n'
)


@pytest.mark.parametrize(
    ("limits", "state", "matches"),
    [
        ({}, "complete", ["Exfil"]),
        ({"enforce_limits": False}, "complete", ["Exfil"]),
        ({"max_message_size": 256}, "invalid", []),
    ],
    ids=["default-limits", "limits-off", "rejected"],
)
def test_a_scan_never_mutates_the_callers_conversation(limits, state, matches) -> None:
    data = copy.deepcopy(_RICH)

    result = DSLScanner(
        QuarryConfig(quarry_rule_texts=(_EXFIL,)), limits=SafetyLimits(**limits)
    ).scan_data(data)

    assert (result.completion_state, result.rule_matches) == (state, matches)
    assert data == _RICH


def test_the_model_receives_the_callers_transcript_losslessly(tmp_path, monkeypatch) -> None:
    set_config_environment(monkeypatch)
    scored: list = []
    scanner = v1_scanner(
        tmp_path, 'llm_score("gemma4", "policy", conversation()) > 0.5', FakeAdapter(), scored
    )

    result = scanner.scan_data(copy.deepcopy(_RICH))

    assert result.rule_matches == ["R"]
    # Message order, text/image part order, full image bytes and detail, and the tool graph.
    assert scored == [_RICH["messages"]]
