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

"""What a rule's model input sends to the provider, and when it sends nothing.

Every assertion is made at the adapter boundary: the recording builder sees the selected
conversation and the fake adapter sees the production wrapping of it. A model input that
cannot be built (a non-string field, a call whose arguments cannot be decoded) must fail the
scan closed before any provider call; an input with no content makes no call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quarry.cache.async_cache import result_cache
from quarry.cache.regex_cache import regex_cache
from quarry.config import QuarryConfig
from quarry.decision_report import path_report, report_tree, scanner_digests, telemetry_record
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.policy import compile_policy
from quarry.types.exceptions import MODEL_REFUSED, LLMError, RuleValidationError
from tests.unit.model_fakes import (
    FakeAdapter,
    ModelAdapter,
    model_entry,
    passthrough_builder,
    set_config_environment,
    v1_backend,
    v2_scanner,
    write_v1_config,
)


@pytest.fixture(scope="module")
def v1_config(tmp_path_factory) -> QuarryConfig:
    directory = tmp_path_factory.mktemp("model-input")
    placeholder = directory / "placeholder.rule"
    placeholder.write_text('rule "placeholder"\ncondition: true\n', encoding="utf-8")
    with pytest.MonkeyPatch.context() as monkeypatch:
        set_config_environment(monkeypatch)
        return write_v1_config(directory, [placeholder])


def _scan(config: QuarryConfig, condition: str, conversation: dict):
    """Scan one ``llm_score``-family rule; return the result, the adapter, and what was scored."""

    adapter = FakeAdapter()
    scored: list = []
    policy = compile_policy(
        f'rule "R"\ncondition: {condition}\n',
        runtime_config=config,
        model_inference_backend=v1_backend(config, adapter, scored),
    )
    return policy.scan(conversation), adapter, scored


def _wire_text(adapter: FakeAdapter) -> str:
    """All text the score adapter received in its last call."""

    parts = []
    for message in adapter.builder_messages[-1]:
        content = message["content"]
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(part.get("text", "") for part in content)
    return "\n".join(parts)


def _user(*contents: str) -> list[dict]:
    return [{"role": "user", "content": content} for content in contents]


def _calls(arguments: str, name: str = "wire") -> dict:
    return {
        "messages": [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            },
        ]
    }


_THREE = {
    "messages": [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]
}
_FIVE = {
    "messages": [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
    ]
}
_FIRST_AND_THIRD = _user("first", "third")
_SCORE = 'llm_score("gemma4", "policy", {}) > 0.85'
_PER_CALL = 'any call in get_tool_calls("wire") where ' + _SCORE


@pytest.mark.parametrize(
    ("model_input", "conversation", "scored"),
    [
        ("conversation()", _THREE, _THREE["messages"]),
        ("messages", _THREE, _THREE["messages"]),
        ("conversation().messages", _THREE, _THREE["messages"]),
        ("messages[-1:]", _THREE, _user("third")),
        ("conversation().messages[-1:]", _THREE, _user("third")),
        ("user_messages", _THREE, _FIRST_AND_THIRD),
        ("messages.user_messages", _THREE, _FIRST_AND_THIRD),
        ("conversation().user_messages", _THREE, _FIRST_AND_THIRD),
        ("messages.user_messages[-1:]", _THREE, _user("third")),
        ("user_messages.user_messages", _FIVE, _user("u1", "u2", "u3")),
        ("messages[:3].user_messages", _FIVE, _user("u1", "u2")),
        ("messages.user_messages[:3]", _FIVE, _user("u1", "u2", "u3")),
        ('message("user", "built")', _THREE, _user("built")),
        (
            'message("system", "policy text").add_message("user", "hello")',
            _THREE,
            [{"role": "system", "content": "policy text"}, *_user("hello")],
        ),
        ('message("user", messages[0].content)', _THREE, _user("first")),
        ('message("user", messages[-1].content)', _THREE, _user("third")),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_the_scorer_receives_exactly_the_selected_model_input(
    v1_config, model_input, conversation, scored
) -> None:
    result, _, received = _scan(v1_config, _SCORE.format(model_input), conversation)

    assert (result.rule_matches, result.is_complete) == (["R"], True)
    assert received == [scored]


@pytest.mark.parametrize(
    ("model_input", "arguments", "scored"),
    [
        ('message("user", call.arguments.command)', '{"command": "rm -rf /"}', _user("rm -rf /")),
        ('message("user", call.arguments.command)', '{"command": ""}', _user("")),
        (
            'message("user", call.arguments.command)'
            '.add_message("assistant", call.arguments.reason)',
            '{"command": "rm -rf /", "reason": "cleanup"}',
            [*_user("rm -rf /"), {"role": "assistant", "content": "cleanup"}],
        ),
    ],
    ids=["one-field", "empty-string-field", "two-fields-in-order"],
)
def test_constructed_content_is_filled_from_the_bound_call(
    v1_config, model_input, arguments, scored
) -> None:
    result, adapter, received = _scan(v1_config, _PER_CALL.format(model_input), _calls(arguments))

    assert (result.rule_matches, result.is_complete) == (["R"], True)
    assert received == [scored]
    assert "do it" not in _wire_text(adapter)


_WITH_TOOLS = {
    "messages": [
        {"role": "user", "content": "keep-user"},
        {"role": "assistant", "content": "drop-assistant"},
    ],
    "tools": [{"type": "function", "function": {"name": "wire_money", "parameters": {}}}],
}


@pytest.mark.parametrize(
    ("model_input", "tools_reach_the_model"),
    [
        ("conversation()", True),
        ("user_messages", False),
        ("conversation().messages.user_messages", False),
    ],
)
def test_only_a_conversation_input_carries_the_offered_tools(
    v1_config, model_input, tools_reach_the_model
) -> None:
    result, adapter, _ = _scan(v1_config, _SCORE.format(model_input), _WITH_TOOLS)

    assert result.rule_matches == ["R"]
    assert ("wire_money" in _wire_text(adapter)) is tools_reach_the_model


_NON_STRING = 'message("user", call.arguments.command)'


@pytest.mark.parametrize(
    ("condition", "conversation"),
    [
        pytest.param(_PER_CALL.format(_NON_STRING), _calls('{"command": 5}'), id="number-field"),
        pytest.param(_PER_CALL.format(_NON_STRING), _calls('{"command": true}'), id="bool-field"),
        pytest.param(
            _PER_CALL.format(_NON_STRING), _calls('{"command": ["a", "b"]}'), id="array-field"
        ),
        pytest.param(
            _PER_CALL.format(_NON_STRING), _calls('{"command": {"k": "v"}}'), id="object-field"
        ),
        pytest.param(
            _PER_CALL.format(_NON_STRING + '.add_message("assistant", call.arguments.reason)'),
            _calls('{"command": "ok", "reason": 42}'),
            id="second-field-not-a-string",
        ),
        pytest.param(
            _SCORE.format('message("user", messages[99].content)'),
            {"messages": _user("only one")},
            id="index-past-the-end",
        ),
        pytest.param(
            _SCORE.format('message("user", messages[-5].content)'),
            {"messages": _user("only one")},
            id="negative-index-before-the-start",
        ),
        pytest.param(
            _SCORE.format("messages[5:9]"), {"messages": _user("only one")}, id="empty-slice"
        ),
        pytest.param(
            _SCORE.format("assistant_messages"),
            {"messages": _user("only one")},
            id="empty-role-view",
        ),
        pytest.param(
            _SCORE.format("messages.user_messages.assistant_messages"),
            {"messages": _THREE["messages"]},
            id="cross-role-score",
        ),
        pytest.param(
            'llm_input_tokens("gemma4", messages.user_messages.assistant_messages) > 3',
            {"messages": _THREE["messages"]},
            id="cross-role-tokens",
        ),
        pytest.param(
            'llm_complete("gemma4", messages.user_messages.assistant_messages) == "x"',
            {"messages": _THREE["messages"]},
            id="cross-role-completion",
        ),
        pytest.param(
            'any c in llm_split_conversation("gemma4", messages.user_messages.assistant_messages,'
            ' 200) where llm_score("gemma4", "p", c) > 0.5',
            {"messages": _THREE["messages"]},
            id="cross-role-chunks",
        ),
    ],
)
def test_an_unbuildable_model_input_fails_closed_before_any_provider_call(
    v1_config, condition, conversation
) -> None:
    result, adapter, _ = _scan(v1_config, condition, conversation)

    assert (result.rule_matches, result.is_complete) == ([], False)
    assert not adapter.called()


_UNREACHABLE = 'llm_score("gemma4", "p", message("user", messages[99].content)) > 0.5'


@pytest.mark.parametrize(
    ("condition", "matches"),
    [
        (f"false and {_UNREACHABLE}", []),
        (f"{_UNREACHABLE} and false", []),
        (f"true or {_UNREACHABLE}", ["R"]),
        (f"{_UNREACHABLE} or true", ["R"]),
        (f"(false and false) and {_UNREACHABLE}", []),
    ],
)
def test_a_model_input_removed_by_constant_folding_is_never_built(
    v1_config, condition, matches
) -> None:
    result, adapter, _ = _scan(v1_config, condition, {"messages": _user("one")})

    assert (result.rule_matches, result.is_complete) == (matches, True)
    assert not adapter.called()


def test_mutating_the_input_after_a_scan_cannot_change_what_was_scored(v1_config) -> None:
    conversation = {"messages": _user("keep me")}

    _, adapter, scored = _scan(v1_config, _SCORE.format("conversation()"), conversation)
    conversation["messages"][0]["content"] = "changed after scan"

    assert scored == [_user("keep me")]
    assert "keep me" in _wire_text(adapter)
    assert "changed after scan" not in _wire_text(adapter)


# A version 1 function cannot read content that join() computes.
_VERSION_2_ONLY = "content computed with join() is supported only in a version 2 model expression"
_MAP_FORM = "a map comprehension is supported only as the first argument of join()"


@pytest.mark.parametrize(
    ("condition", "reason"),
    [
        pytest.param(
            _SCORE.format('"not a conversation"'),
            "accepts ['model_input'], got string",
            id="text-as-the-input",
        ),
        pytest.param(
            _SCORE.format("tool_calls"),
            "model input must be conversation()",
            id="tool-calls-as-the-input",
        ),
        pytest.param(_SCORE.format('message("tool", "x")'), "must be one of", id="tool-role"),
        pytest.param(
            _SCORE.format('message("user", 123)'),
            "accepts ['string'], got integer",
            id="number-as-content",
        ),
        pytest.param(
            'llm_complete("gemma4", message("user", messages[0].content)) == "x"',
            "does not support dynamic constructed content",
            id="llm-complete-rejects-field-content",
        ),
        pytest.param(
            _SCORE.format(
                'message("user", messages[0].content).add_message("assistant", messages[1].content)'
            ),
            "must reference the SAME message",
            id="field-content-from-two-messages",
        ),
        pytest.param(
            'llm_complete("gemma4", conversation())',
            "cannot be used as a bare condition",
            id="bare-llm-complete",
        ),
        pytest.param(
            'llm_complete("gemma4", conversation()) > 0.5',
            "unsupported top-level condition",
            id="llm-complete-compared-with-a-number",
        ),
        pytest.param(
            '"compl" ~ llm_complete("gemma4", conversation())',
            "requires llm_complete() on the LEFT",
            id="llm-complete-on-the-right",
        ),
        pytest.param(
            _SCORE.format("user_messages.messages"),
            "model input must be conversation()",
            id="messages-of-a-view",
        ),
        pytest.param(
            _SCORE.format("messages.user_messages.messages"),
            "model input must be conversation()",
            id="messages-of-a-view-of-messages",
        ),
        pytest.param(
            _SCORE.format("messages()"), "unknown function 'messages'", id="unknown-function"
        ),
        pytest.param(
            _SCORE.format('message("user", [c.name for c in tool_calls])'),
            _MAP_FORM,
            id="map-comprehension-as-content",
        ),
        pytest.param(
            _SCORE.format('message("user", [c.name for c in messages[-1].tool_calls])'),
            _MAP_FORM,
            id="map-comprehension-over-the-newest-message-as-content",
        ),
        pytest.param(
            _SCORE.format(
                'message("user", "x").add_message("user", [c.name for c in messages[-1].tool_calls])'
            ),
            _MAP_FORM,
            id="map-comprehension-as-added-content",
        ),
        pytest.param(
            _SCORE.format('message("user", len(tool_calls))'),
            "not FunctionCall",
            id="function-call-as-content",
        ),
        pytest.param(
            _SCORE.format('message("user", join([c.name for c in messages[-1].tool_calls], ","))'),
            _VERSION_2_ONLY,
            id="llm-score-rejects-joined-content",
        ),
        pytest.param(
            'llm_input_tokens("gemma4", message("user", join([c.name for c in '
            'messages[-1].tool_calls], ","))) > 3',
            _VERSION_2_ONLY,
            id="llm-input-tokens-rejects-joined-content",
        ),
        pytest.param(
            'llm_complete("gemma4", message("user", join([c.name for c in '
            'messages[-1].tool_calls], ","))) != "x"',
            _VERSION_2_ONLY,
            id="llm-complete-rejects-joined-content",
        ),
        pytest.param(
            'any c in llm_split_conversation("gemma4", message("user", join([t.name for t in '
            'messages[-1].tool_calls], ",")), 200) where llm_score("gemma4", "p", c) > 0.5',
            _VERSION_2_ONLY,
            id="llm-split-conversation-rejects-joined-content",
        ),
    ],
)
def test_an_invalid_model_input_is_rejected_at_load(v1_config, condition, reason) -> None:
    with pytest.raises(RuleValidationError, match="R") as error:
        _scan(v1_config, condition, {"messages": _user("x")})

    assert reason in str(error.value)


# ── model("judge").score / llm_score with message() content over tool calls ───────────────

_CALLS_TURN = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "exec_command", "arguments": '{"command": "rm -rf /tmp/x"}'},
        },
        {
            "id": "c2",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": '{"path": "/etc/hosts", "content": "x"}',
            },
        },
    ],
}
_ENDS_WITH_CALLS = {"messages": [{"role": "user", "content": "go"}, _CALLS_TURN]}
_ENDS_WITH_TOOL = {
    "messages": [
        *_ENDS_WITH_CALLS["messages"],
        {"role": "tool", "tool_call_id": "c1", "content": "done"},
    ]
}
_EXEC_TOOL = [
    {"type": "function", "function": {"name": "exec", "description": "d", "parameters": {}}}
]
_JUDGES = {
    "model-score": 'model("judge").score({input}, text_ref("p")) >= 0.5',
    "llm-score": 'llm_score("judge", text_ref("p"), {input}) >= 0.5',
}
_GUARD = '(any m in messages[-1:] where m.role == "assistant") and '
_ARGUMENTS_TEXT = [[{"role": "user", "content": '{"command": "rm -rf /tmp/x"}'}]]
_COMMAND_TEXT = [_user("rm -rf /tmp/x")]


# One judge input made from every qualifying tool call of the newest message: the commands of
# the calls whose argument values match `$risky`, in call order.
_JOINED = (
    'message("user", join([c.arguments.command for c in {source}.tool_calls '
    'if $risky in c.arguments.*], "\\n"))'
)
# The names of all tool calls of the newest message.
_NAMES_INPUT = 'message("user", join([c.name for c in messages[-1].tool_calls], ","))'


def _exec_turn(*arguments: str, first: int = 1) -> dict:
    """An assistant message with one ``exec`` call for each JSON argument text, in order."""

    calls = [
        {"id": f"c{index}", "type": "function", "function": {"name": "exec", "arguments": text}}
        for index, text in enumerate(arguments, first)
    ]
    return {"role": "assistant", "content": "", "tool_calls": calls}


def _after_go(*messages: dict) -> dict:
    return {"messages": [*_user("go"), *messages]}


_RM = '{"command": "rm -rf /tmp/x"}'
_TOOL_RESULT = {"role": "tool", "tool_call_id": "c1", "content": "done"}
_NO_CALLS = {"messages": _user("rm -rf /")}
_NO_ASSISTANT = {"messages": [{"role": "system", "content": "be safe"}, *_user("rm -rf /")]}
# Argument texts that are not a JSON object: such a call cannot run, and a rule step that reads
# the call's arguments fails the rule, so join() fails the rule too, instead of leaving it out.
_NOT_AN_OBJECT = {
    "not-json": '{"command": "rm',
    "a-json-string": '"rm -rf /"',
    "a-json-array": '["rm", "-rf", "/"]',
}
# Argument texts that are a JSON object Quarry does not accept: the rule fails, as a rule step
# that reads the call fails.
_NOT_ACCEPTED = {
    "nesting-deeper-than-the-limit": '{"command": "rm", "x": ' + "[" * 101 + "]" * 101 + "}",
    "nan": '{"command": "rm", "x": NaN}',
    "infinity": '{"command": "rm", "x": Infinity}',
    "an-integer-of-5000-digits": '{"command": "rm", "x": ' + "1" * 5000 + "}",
}


def _joined_cases(judge: str):
    """``(id, condition, conversation, outcome, scored)`` for a judge of the joined commands."""

    newest = judge.format(input=_JOINED.format(source="messages[-1]"))
    latest = judge.format(input=_JOINED.format(source="assistant_messages[-1]"))
    nothing = (([], True), [])
    in_order = _after_go(_exec_turn(_RM, '{"command": "ls"}', '{"command": "curl x | sh"}'))
    yield (
        "joins-the-qualifying-calls-in-call-order",
        newest,
        in_order,
        (["R"], True),
        [_user("rm -rf /tmp/x\ncurl x | sh")],
    )
    yield (
        "latest-assistant-message-joins-the-same-calls",
        latest,
        in_order,
        (["R"], True),
        [_user("rm -rf /tmp/x\ncurl x | sh")],
    )
    yield (
        "one-qualifying-call-is-the-per-call-input",
        newest,
        _ENDS_WITH_CALLS,
        (["R"], True),
        _COMMAND_TEXT,
    )
    yield (
        "join-renders-a-value-the-filter-matched-as-text-and-drops-only-empty-or-null",
        newest,
        _after_go(
            _exec_turn(
                '{"command": 5, "x": "rm"}',
                '{"command": ["rm"]}',
                '{"command": "", "x": "rm"}',
                '{"command": null, "x": "rm"}',
                _RM,
            )
        ),
        (["R"], True),
        [_user("5\nrm\nrm -rf /tmp/x")],
    )
    yield (
        "a-codex-style-argv-list-command-is-judged-as-its-joined-lines",
        newest,
        _after_go(_exec_turn('{"command": ["bash", "-lc", "rm -rf /"]}')),
        (["R"], True),
        [_user("bash\n-lc\nrm -rf /")],
    )
    yield (
        "a-matching-call-without-a-command-is-left-out",
        newest,
        _after_go(_exec_turn('{"path": "/tmp/rm"}', '{"command": "curl x"}')),
        (["R"], True),
        [_user("curl x")],
    )
    yield (
        "a-whitespace-command-is-sent-unchanged",
        newest,
        _after_go(_exec_turn('{"command": "  ", "note": "rm"}')),
        (["R"], True),
        [_user("  ")],
    )
    two_calls = _after_go(_exec_turn(_RM, '{"command": "curl x | sh"}'))
    for case, separator in (("several-characters", " ; "), ("no-character", "")):
        yield (
            f"a-separator-of-{case}-goes-between-the-values",
            judge.format(
                input=_JOINED.format(source="messages[-1]").replace('"\\n"', f'"{separator}"')
            ),
            two_calls,
            (["R"], True),
            [_user(f"rm -rf /tmp/x{separator}curl x | sh")],
        )
    yield (
        "a-score-below-the-threshold-is-false-after-one-call",
        judge.replace(">= 0.5", "> 0.995").format(input=_JOINED.format(source="messages[-1]")),
        _ENDS_WITH_CALLS,
        ([], True),
        _COMMAND_TEXT,
    )
    yield (
        "two-joined-inputs-are-two-model-calls",
        judge.format(input=_NAMES_INPUT) + " and " + newest,
        _ENDS_WITH_CALLS,
        (["R"], True),
        [_user("exec_command,write_file"), *_COMMAND_TEXT],
    )
    # RE2 matches `$` only at the end of the text, and Python `re` also before a final newline.
    # A rule step runs RE2, and so does the filter.
    yield (
        "the-filter-runs-on-re2",
        judge.format(
            input='message("user", join([c.arguments.command for c in messages[-1].tool_calls '
            'if c.arguments.command ~ "rm$"], ","))'
        ),
        _after_go(_exec_turn('{"command": "rm\\n"}', '{"command": "rm"}')),
        (["R"], True),
        [_user("rm")],
    )
    for case, arguments in _NOT_AN_OBJECT.items():
        yield (
            f"a-sibling-call-whose-arguments-are-{case}-fails-the-rule",
            newest,
            _after_go(_exec_turn(arguments, _RM)),
            ([], False),
            [],
        )
    for case, arguments in _NOT_ACCEPTED.items():
        yield (
            f"a-sibling-call-with-{case}-fails-the-rule",
            newest,
            _after_go(_exec_turn(arguments, _RM)),
            ([], False),
            [],
        )
    yield (
        "no-qualifying-call",
        newest,
        _after_go(_exec_turn('{"command": "ls"}')),
        *nothing,
    )
    done = {"role": "assistant", "content": "done"}
    yield "assistant-message-without-tool-calls", newest, _after_go(done), *nothing
    for role in ("tool", "user", "developer"):
        last = _TOOL_RESULT if role == "tool" else {"role": role, "content": "rm -rf /"}
        yield f"newest-message-is-{role}", newest, _after_go(_exec_turn(_RM), last), *nothing
    yield (
        "a-false-guard-skips-the-latest-assistant-message",
        'messages[-1].role == "assistant" and ' + latest,
        _after_go(_exec_turn(_RM), _TOOL_RESULT),
        *nothing,
    )
    yield (
        "an-older-turn-is-not-judged",
        newest,
        _after_go(_exec_turn(_RM), _TOOL_RESULT, _exec_turn('{"command": "ls"}', first=2)),
        *nothing,
    )
    for source, condition in (("messages", newest), ("assistant-messages", latest)):
        yield f"{source}-in-a-user-only-conversation", condition, _NO_CALLS, *nothing
        yield f"{source}-without-an-assistant-message", condition, _NO_ASSISTANT, *nothing
    # With a second indexed read the model input is one closed check that reads the messages
    # itself. It joins the same calls, and a message that does not exist gives no value.
    and_last_user = '.add_message("user", user_messages[-1].content)'
    yield (
        "another-indexed-read-joins-the-same-calls",
        judge.format(input=_JOINED.format(source="assistant_messages[-1]") + and_last_user),
        _after_go(_exec_turn(_RM, '{"command": "curl x | sh"}')),
        (["R"], True),
        [_user("rm -rf /tmp/x\ncurl x | sh", "go")],
    )
    yield (
        "an-empty-joined-turn-is-omitted-beside-a-populated-read",
        judge.format(input=_JOINED.format(source="assistant_messages[-1]") + and_last_user),
        _NO_CALLS,
        (["R"], True),
        [_user("rm -rf /")],
    )
    and_last_names = (
        '.add_message("user", join([c.name for c in assistant_messages[-1].tool_calls], ","))'
    )
    yield (
        "the-call-is-skipped-only-when-every-turn-is-empty",
        judge.format(input=_JOINED.format(source="assistant_messages[-1]") + and_last_names),
        _NO_CALLS,
        *nothing,
    )
    yield (
        "a-filter-that-reads-another-malformed-call-fails-the-rule",
        judge.format(
            input='message("user", join([c.name for c in messages[-1].tool_calls '
            'if $risky in messages[-1].tool_calls[0].arguments.*], ","))'
        ),
        _after_go(_exec_turn('{"command": "rm', _RM)),
        ([], False),
        [],
    )


def _judge_rows():
    for judge_id, judge in _JUDGES.items():

        def row(case_id, condition, conversation, outcome, scored, judge_id=judge_id):
            return pytest.param(
                condition, conversation, outcome, scored, id=f"{judge_id}-{case_id}"
            )

        per_call = "any c in tool_calls where " + judge
        yield row(
            "arguments-as-json-text",
            per_call.format(input='message("user", c.arguments)'),
            _ENDS_WITH_CALLS,
            (["R"], True),
            _ARGUMENTS_TEXT,
        )
        yield row(
            "one-field-of-each-call",
            per_call.format(input='message("user", c.arguments.command)'),
            _ENDS_WITH_CALLS,
            (["R"], True),
            _COMMAND_TEXT,
        )
        yield row(
            "missing-field-skips-that-call",
            per_call.format(input='message("user", c.arguments.path)'),
            _ENDS_WITH_CALLS,
            (["R"], True),
            [[{"role": "user", "content": "/etc/hosts"}]],
        )
        yield row(
            "field-absent-everywhere",
            per_call.format(input='message("user", c.arguments.url)'),
            _ENDS_WITH_CALLS,
            ([], True),
            [],
        )
        yield row(
            "false-guard-builds-nothing",
            _GUARD + judge.format(input="messages[-1:]"),
            _ENDS_WITH_TOOL,
            ([], True),
            [],
        )
        yield row(
            "true-guard-scores-the-last-turn",
            _GUARD + judge.format(input="messages[-1:]"),
            _ENDS_WITH_CALLS,
            (["R"], True),
            [[_CALLS_TURN]],
        )
        # A message that does not exist is empty input in a version 2 judge, so `not` holds; a
        # version 1 judge cannot build its input and fails the rule.
        for index in ("messages[5]", "user_messages[3]"):
            yield row(
                f"out-of-range-{index}-under-not",
                "not " + judge.format(input=f'message("user", {index}.content)'),
                {"messages": [*_user("go"), {"role": "assistant", "content": "ok"}]},
                ([], False) if judge_id == "llm-score" else (["R"], True),
                [],
            )
    # A version 1 judge rejects joined content when rules load (see
    # test_a_version_1_judge_rejects_joined_content), so only the model-score judge runs these.
    for case_id, condition, conversation, outcome, scored in _joined_cases(_JUDGES["model-score"]):
        yield pytest.param(condition, conversation, outcome, scored, id=f"model-score-{case_id}")


_RM_OR_CURL = '$risky = "rm|curl" regex'


def _judge_policy(
    tmp_path: Path,
    condition: str,
    adapter=None,
    variables: str | None = None,
    strings: str = _RM_OR_CURL,
    **limits,
):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"judge": model_entry(score=True, tokens=True)},
            "texts": {"p": "Judge the input."},
            **limits,
        },
        tmp_path,
        require_rules=False,
    )
    adapter = adapter or ModelAdapter()
    backend = ModelInferenceBackend(
        config, adapters={"fake": adapter}, message_builders={"test": passthrough_builder}
    )
    block = f"variables:\n    {variables}\n" if variables else ""
    policy = compile_policy(
        f'rule "R"\nstrings:\n    {strings}\n{block}condition:\n    {condition}\n',
        runtime_config=config,
        model_inference_backend=backend,
    )
    return policy, adapter


def _judge_scan(tmp_path: Path, condition: str, conversation: dict, strings: str = _RM_OR_CURL):
    policy, adapter = _judge_policy(tmp_path, condition, strings=strings)
    return policy.scan(conversation), adapter


@pytest.mark.parametrize(("condition", "conversation", "outcome", "scored"), _judge_rows())
def test_a_judge_scores_only_the_inputs_its_rule_reaches(
    tmp_path, condition, conversation, outcome, scored
) -> None:
    result, adapter = _judge_scan(tmp_path, condition, conversation)

    assert (result.rule_matches, result.is_complete) == outcome
    assert adapter.score_messages == scored


@pytest.mark.parametrize(
    "condition",
    [
        pytest.param(condition, id=f"llm-score-{case_id}")
        for case_id, condition, *_ in _joined_cases(_JUDGES["llm-score"])
    ],
)
def test_a_version_1_judge_rejects_joined_content(tmp_path, condition) -> None:
    with pytest.raises(RuleValidationError) as error:
        _judge_policy(tmp_path, condition)

    assert _VERSION_2_ONLY in str(error.value)


class _UnsafeVerdict(ModelAdapter):
    """Completes every input with ``{"unsafe": true}`` and records the input."""

    def __init__(self) -> None:
        super().__init__()
        self.completed: list = []

    async def complete_message(self, config, messages, _tools, *, timeout=None):
        self.completed.append(messages)
        return {"role": "assistant", "content": '{"unsafe": true}'}


_UNSAFE_SCHEMA = {
    "type": "object",
    "properties": {"unsafe": {"type": "boolean"}},
    "required": ["unsafe"],
    "additionalProperties": False,
}
_UNSAFE = 'model("verdict").complete({}).json.unsafe'
_JOINED_UNSAFE = _UNSAFE.format(
    'message("user", join([c.arguments.command for c in messages[-1].tool_calls '
    'if "rm" in c.arguments.*], "\\n"))'
)
_NO_QUALIFYING_CALL = _after_go(_exec_turn('{"command": "ls"}'))


def _value_without_input_rows():
    """``(body, conversation, outcome, completed)`` for a boolean value of a model result."""

    without_input = (
        ("without-a-qualifying-call", _NO_QUALIFYING_CALL),
        ("in-a-user-only-conversation", _NO_CALLS),
    )
    for case, conversation in without_input:
        for fallback in ("false", "true"):
            yield pytest.param(
                f"condition: if_refused({_JOINED_UNSAFE}, {fallback})",
                conversation,
                ([], True),
                [],
                id=f"{fallback}-fallback-{case}",
            )
        yield pytest.param(
            f"condition: not if_refused({_JOINED_UNSAFE}, false)",
            conversation,
            (["R"], True),
            [],
            id=f"not-{case}",
        )
    yield pytest.param(
        f"condition: (if_refused({_JOINED_UNSAFE}, false) "
        f"or if_refused({_JOINED_UNSAFE}, true)) == true",
        _NO_QUALIFYING_CALL,
        ([], True),
        [],
        id="or-of-two-values-without-input",
    )
    yield pytest.param(
        f"condition: (not if_refused({_JOINED_UNSAFE}, false)) == true",
        _NO_QUALIFYING_CALL,
        (["R"], True),
        [],
        id="not-inside-a-comparison-without-input",
    )
    yield pytest.param(
        'condition: true\ncollect:\n    [span for span in model("email").spans(messages, 0.70) '
        f"if if_refused({_JOINED_UNSAFE}, true)]",
        {"messages": _user("mail alice@example.com")},
        ([], True),
        [],
        id="collect-filter-without-input-leaves-the-span-out",
    )
    yield pytest.param(
        f"condition: if_refused({_JOINED_UNSAFE}, false)",
        _ENDS_WITH_CALLS,
        (["R"], True),
        _COMMAND_TEXT,
        id="a-qualifying-call-is-judged",
    )
    # A per-call input that reads a field the call does not have is also without input.
    yield pytest.param(
        "condition: any c in tool_calls where if_refused("
        + _UNSAFE.format('message("user", c.arguments.path)')
        + ", false)",
        _after_go(_exec_turn(_RM)),
        ([], True),
        [],
        id="per-call-input-from-a-field-the-call-does-not-have",
    )


@pytest.mark.parametrize(
    ("body", "conversation", "outcome", "completed"), _value_without_input_rows()
)
def test_a_boolean_model_value_without_input_is_false(
    tmp_path, body, conversation, outcome, completed
) -> None:
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "verdict": {**model_entry(), "complete": {"response_schema": _UNSAFE_SCHEMA}},
                "email": model_entry(category="EMAIL", spans=0),
            },
        },
        tmp_path,
        require_rules=False,
    )
    adapter = _UnsafeVerdict()
    backend = ModelInferenceBackend(config, adapters={"fake": adapter})
    policy = compile_policy(
        f'rule "R"\n{body}\n', runtime_config=config, model_inference_backend=backend
    )

    result = policy.scan(conversation)

    assert (result.rule_matches, result.is_complete) == outcome
    assert adapter.completed == completed


def test_judges_of_the_same_text_share_one_model_call(tmp_path) -> None:
    judge = _JUDGES["model-score"]
    adapter = ModelAdapter()
    scanner = v2_scanner(
        tmp_path,
        {
            "Literal": judge.format(input='message("user", "rm -rf /tmp/x")'),
            "PerCall": "any c in tool_calls where "
            + judge.format(input='message("user", c.arguments.command)'),
            "Joined": judge.format(
                input='message("user", join([c.arguments.command for c in messages[-1].tool_calls'
                ' if c.name == "exec_command"], "\\n"))'
            ),
        },
        {"judge": model_entry(score=True)},
        adapter,
        texts={"p": "Judge the input."},
    )

    result = scanner.scan_data(_ENDS_WITH_CALLS)

    assert (result.rule_matches, result.is_complete) == (["Literal", "PerCall", "Joined"], True)
    assert adapter.score_messages == _COMMAND_TEXT


@pytest.mark.parametrize(
    ("limits", "calls", "outcome", "errors", "scored"),
    [
        pytest.param(
            {"max_comprehension_size": 2},
            2,
            (["R"], True),
            None,
            [_user("rm -rf /tmp/x\nrm -rf /tmp/x")],
            id="at-the-limit",
        ),
        pytest.param(
            {"max_comprehension_size": 2},
            3,
            ([], False),
            {"R": "map comprehension source has 3 items, exceeding max_comprehension_size=2"},
            [],
            id="over-the-limit",
        ),
        pytest.param(
            {"max_comprehension_size": 2, "enforce_limits": False},
            3,
            (["R"], True),
            None,
            [_user("rm -rf /tmp/x\nrm -rf /tmp/x\nrm -rf /tmp/x")],
            id="over-a-limit-that-is-not-enforced",
        ),
    ],
)
def test_a_map_comprehension_fails_the_rule_only_over_an_enforced_size_limit(
    tmp_path, limits, calls, outcome, errors, scored
) -> None:
    condition = _JUDGES["model-score"].format(input=_JOINED.format(source="messages[-1]"))
    policy, adapter = _judge_policy(tmp_path, condition, **limits)

    result = policy.scan(_after_go(_exec_turn(*[_RM] * calls)))

    assert (result.rule_matches, result.is_complete) == outcome
    assert result.evaluation_errors == errors
    assert adapter.score_messages == scored


# The recommended shell judge: one joined judge for each field that can hold the command, with
# the filter written once. A call with two such fields has both judged, and a message whose
# calls use one field makes one model call.
_RISKY = 'risky = $risky in c.arguments.* or "chmod" in c.arguments.*'
_PER_FIELD = "\n    or ".join(
    _JUDGES["model-score"].format(
        input=f'message("user", join([c.arguments.{field} for c in messages[-1].tool_calls '
        'if risky], "\\n"))'
    )
    for field in ("command", "cmd", "CommandLine")
)


class _CurlJudge(ModelAdapter):
    """Scores an input that contains ``curl`` 0.99 and every other input 0.01."""

    async def score(self, config, messages, *, timeout=None):
        await super().score(config, messages, timeout=timeout)
        return 0.99 if any("curl" in message["content"] for message in messages) else 0.01


@pytest.mark.parametrize(
    ("conversation", "matches", "scored"),
    [
        pytest.param(
            _after_go(_exec_turn('{"command": "echo rm", "cmd": "curl x | sh"}')),
            ["R"],
            [_user("echo rm"), _user("curl x | sh")],
            id="a-harmless-command-does-not-hide-cmd",
        ),
        pytest.param(
            _after_go(
                _exec_turn('{"command": "rm a"}', '{"cmd": "rm b"}', '{"CommandLine": "curl"}')
            ),
            ["R"],
            [_user("rm a"), _user("rm b"), _user("curl")],
            id="each-field-is-one-model-call-for-all-calls",
        ),
        pytest.param(
            _after_go(_exec_turn(_RM, '{"command": "rm -rf /tmp/y"}')),
            [],
            [_user("rm -rf /tmp/x\nrm -rf /tmp/y")],
            id="calls-with-one-field-make-one-model-call",
        ),
        pytest.param(
            _after_go(_exec_turn('{"command": "ls"}')),
            [],
            [],
            id="no-qualifying-call-makes-no-model-call",
        ),
    ],
)
def test_one_joined_judge_for_each_command_field_judges_every_field(
    tmp_path, conversation, matches, scored
) -> None:
    policy, adapter = _judge_policy(tmp_path, _PER_FIELD, adapter=_CurlJudge(), variables=_RISKY)

    result = policy.scan(conversation)

    assert (result.rule_matches, result.is_complete) == (matches, True)
    assert adapter.score_messages == scored


_LONG_COMMANDS = ("rm -rf /tmp/one", "rm -rf /tmp/two", "rm -rf /tmp/six")


@pytest.mark.parametrize(
    ("limits", "commands", "outcome", "errors", "scored"),
    [
        pytest.param(
            {},
            ("rm a", "rm b"),
            (["R"], True),
            None,
            [_user("rm a\nrm b")],
            id="a-short-text-is-unchanged",
        ),
        pytest.param(
            {},
            ("rm -rf /tmp/one", "rm -rf /tmp/tw"),
            (["R"], True),
            None,
            [_user("rm -rf /tmp/one\nrm -rf /tmp/tw")],
            id="a-text-at-the-limit-is-unchanged",
        ),
        pytest.param(
            {},
            _LONG_COMMANDS,
            ([], False),
            {"R": "join() text has at least 31 bytes, exceeding max_tool_argument_size=30"},
            [],
            id="a-long-text-fails-the-rule-instead-of-truncating",
        ),
        pytest.param(
            {"enforce_limits": False},
            _LONG_COMMANDS,
            (["R"], True),
            None,
            [_user("\n".join(_LONG_COMMANDS))],
            id="a-limit-that-is-not-enforced-keeps-all-the-text",
        ),
    ],
)
def test_join_fails_the_rule_rather_than_truncating_a_text_over_the_argument_limit(
    tmp_path, limits, commands, outcome, errors, scored
) -> None:
    condition = _JUDGES["model-score"].format(input=_JOINED.format(source="messages[-1]"))
    policy, adapter = _judge_policy(tmp_path, condition, max_tool_argument_size=30, **limits)

    result = policy.scan(_after_go(_exec_turn(*(json.dumps({"command": c}) for c in commands))))

    assert (result.rule_matches, result.is_complete) == outcome
    assert result.evaluation_errors == errors
    assert adapter.score_messages == scored


def _same_as_a_step(test, arguments, matches, case_id, strings=_RM_OR_CURL):
    return pytest.param(strings, test, arguments, matches, id=case_id)


@pytest.mark.parametrize(
    ("strings", "test", "arguments", "matches"),
    [
        _same_as_a_step(
            'c.arguments.* contains "C:\\\\Windows"',
            '{"command": "del C:\\\\Windows\\\\Temp"}',
            ["R"],
            "contains-a-backslash",
        ),
        _same_as_a_step(
            'c.arguments contains "C:\\\\Windows"',
            '{"command": "del C:\\\\Windows\\\\Temp"}',
            [],
            "the-object-text-doubles-a-backslash",
        ),
        _same_as_a_step('c.arguments.* ~ "^rm"', '{"command": "rm -rf /"}', ["R"], "start-anchor"),
        _same_as_a_step(
            'c.arguments.* ~ "rm -rf /$"', '{"command": "rm -rf /"}', ["R"], "end-anchor"
        ),
        _same_as_a_step(
            'c.arguments.command ~ "rm$"', '{"command": "rm\\n"}', [], "a-regex-runs-on-re2"
        ),
        _same_as_a_step(
            'c.arguments.* contains "\'"', '{"command": "ls"}', [], "no-quote-in-the-values"
        ),
        _same_as_a_step(
            'c.arguments.* contains "it\'s"',
            '{"command": "echo \\"it\'s\\" && rm -rf ~"}',
            ["R"],
            "a-quote-in-a-value",
        ),
        _same_as_a_step('"rm" in c.arguments.*', '{"command": "rm -rf /"}', ["R"], "literal-in"),
        _same_as_a_step(
            "$risky in c.arguments",
            '{"command": "\\u0072m -rf /"}',
            ["R"],
            "a-string-set-in-the-object-reads-a-json-escape-decoded",
        ),
        _same_as_a_step(
            "$start in c.arguments.*",
            '{"command": "rm -rf /"}',
            ["R"],
            "a-string-set-anchored-at-the-start-of-a-value",
            strings='$start = "^rm" regex',
        ),
        _same_as_a_step(
            "$quote in c.arguments.*",
            '{"command": "rm"}',
            [],
            "a-string-set-reads-the-values-not-the-text-of-their-list",
            strings='$quote = "[\'"',
        ),
        _same_as_a_step(
            'c.arguments contains "rm -rf"',
            '{"command": "\\u0072m -rf /"}',
            ["R"],
            "contains-in-the-object-reads-a-json-escape-decoded",
        ),
        _same_as_a_step(
            'c.arguments.* ~ "a\\tb"', '{"command": "echo a\\tb"}', ["R"], "a-tab-in-a-value"
        ),
        _same_as_a_step(
            'c.arguments.* contains "a\\nb"',
            '{"x": "a", "y": "b"}',
            ["R"],
            "each-value-is-one-line",
        ),
        _same_as_a_step(
            'c.arguments.command ~ "^rm"',
            '{"command": ["rm", "-rf", "/"]}',
            ["R"],
            "a-list-value-is-one-item-on-each-line",
        ),
        _same_as_a_step(
            'c.arguments.command != "rm"', '{"x": 1}', ["R"], "not-equal-to-a-missing-field"
        ),
        _same_as_a_step(
            'c.arguments.command != "rm"', '{"command": null}', ["R"], "not-equal-to-a-null-field"
        ),
        _same_as_a_step(
            'c.arguments.command == "5"', '{"command": 5}', ["R"], "a-number-equals-its-text"
        ),
        _same_as_a_step(
            "c.arguments.command == 5", '{"command": "5"}', ["R"], "a-text-equals-a-number"
        ),
        _same_as_a_step(
            'c.arguments.command == "True"', '{"command": true}', ["R"], "a-boolean-equals-its-text"
        ),
        _same_as_a_step(
            "c.arguments.command == null", '{"x": 1}', ["R"], "a-missing-field-equals-null"
        ),
        _same_as_a_step(
            "c.arguments.command != null", '{"command": null}', [], "a-null-field-equals-null"
        ),
        _same_as_a_step(
            '"rm" == c.arguments.command', '{"command": "rm"}', ["R"], "the-text-on-the-left"
        ),
        _same_as_a_step(
            'c.arguments.* == "rm"', '{"command": "rm"}', [], "a-list-of-values-is-not-one-value"
        ),
    ],
)
def test_a_test_in_a_map_filter_reads_the_value_as_the_same_test_in_a_rule_step(
    tmp_path, strings, test, arguments, matches
) -> None:
    conversation = _after_go(_exec_turn(arguments))
    joined = _JUDGES["model-score"].format(
        input=f'message("user", join([c.id for c in messages[-1].tool_calls if {test}], ","))'
    )

    step, _ = _judge_scan(tmp_path, f"any c in tool_calls where {test}", conversation, strings)
    filtered, _ = _judge_scan(tmp_path, joined, conversation, strings)

    assert (step.rule_matches, step.is_complete) == (matches, True)
    assert (filtered.rule_matches, filtered.is_complete) == (matches, True)


class _RefusingJudge(ModelAdapter):
    """Records each score input and refuses it."""

    async def score(self, config, messages, *, timeout=None):
        await super().score(config, messages, timeout=timeout)
        raise LLMError("refused", {"code": MODEL_REFUSED, "reason": "refusal_field"})


_JOINED_SCORE = 'model("judge").score(' + _JOINED.format(source="messages[-1]") + ', text_ref("p"))'


@pytest.mark.parametrize(
    ("condition", "adapter"),
    [
        pytest.param(f"refused({_JOINED_SCORE})", _RefusingJudge, id="model-score-refused"),
        pytest.param(f"not refused({_JOINED_SCORE})", ModelAdapter, id="model-score-not-refused"),
        pytest.param(f"{_JOINED_SCORE} >= 0.5", ModelAdapter, id="model-score-comparison"),
    ],
)
def test_the_evidence_of_a_joined_judge_holds_positions_and_fields_not_text(
    tmp_path, condition, adapter
) -> None:
    # The qualifying call's command is the only thing that could leak; the sibling does not
    # pass the filter, and the evidence does not show either command's text.
    conversation = _after_go(_exec_turn(_RM, '{"command": "echo hi"}'))
    policy, _ = _judge_policy(tmp_path, condition, adapter=adapter())

    result = policy.scan(conversation, explain=True)
    evidence = repr(result.detailed_matches)

    assert result.rule_matches == ["R"]
    assert ("rm -rf" in evidence, "echo hi" in evidence, "rm|curl" in evidence) == (
        False,
        False,
        False,
    )
    matches = {item.rule_name: item for item in result.detailed_matches or []}
    pieces = matches["R"].model_results[0].input_span["pieces"]
    assert [piece["field"] for piece in pieces] == ["command"]
    assert isinstance(pieces[0]["position"], int)


def test_split_over_an_empty_join_makes_no_call_and_completes(tmp_path) -> None:
    check = (
        'any chunk in model("judge").split('
        + _JOINED.format(source="messages[-1]")
        + ', 200) where model("judge").score(chunk, text_ref("p")) > 0.5'
    )

    result, adapter = _judge_scan(tmp_path, check, _after_go(_exec_turn('{"command": "ls"}')))

    assert (result.rule_matches, result.is_complete) == ([], True)
    assert (adapter.score_messages, adapter.tokenizations) == ([], 0)


@pytest.mark.parametrize(
    "first_turn",
    [
        pytest.param('message("system", "Judge each line.")', id="read-only-in-the-filter"),
        pytest.param('message("user", params.tool)', id="read-in-the-filter-and-in-another-turn"),
    ],
)
def test_the_evidence_of_a_joined_judge_names_each_parameter_of_its_filter_one_time(
    tmp_path, first_turn
) -> None:
    condition = _JUDGES["model-score"].format(
        input=f'{first_turn}.add_message("user", join([c.arguments.command for c in '
        'messages[-1].tool_calls if c.name == params.tool], "\\n"))'
    )
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"judge": model_entry(score=True)},
            "texts": {"p": "Judge the input."},
        },
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(
        config, adapters={"fake": ModelAdapter()}, message_builders={"test": passthrough_builder}
    )
    policy = compile_policy(
        f'rule "R"\nparameters:\n    tool: text\ncondition:\n    {condition}\n',
        runtime_config=config,
        model_inference_backend=backend,
    )

    result = policy.scan(_after_go(_exec_turn(_RM)), explain=True, parameters={"tool": "exec"})
    (evidence,) = result.detailed_matches[0].model_results

    assert result.rule_matches == ["R"]
    assert [name for name, _ in evidence.arguments] == [
        "parameter:tool",
        "constitution",
        "tools_supplied",
    ]


def test_the_judges_of_one_rule_test_each_call_with_their_filter_one_time(
    tmp_path, monkeypatch
) -> None:
    # The RE2 searches are the work that the filter does for each call.
    searched: list[str] = []
    search = regex_cache.search_match

    def recording_search(pattern, text, flags=0):
        searched.append(text)
        return search(pattern, text, flags)

    monkeypatch.setattr(regex_cache, "search_match", recording_search)
    policy, adapter = _judge_policy(tmp_path, _PER_FIELD, variables=_RISKY)

    result = policy.scan(_after_go(_exec_turn('{"command": "ls"}', '{"cmd": "pwd"}')))

    assert (result.rule_matches, result.is_complete) == ([], True)
    assert searched == ["ls", "pwd"]
    assert adapter.score_messages == []


@pytest.mark.asyncio
async def test_a_long_map_comprehension_stops_at_the_rule_deadline(tmp_path) -> None:
    condition = _JUDGES["model-score"].format(
        input='message("user", join([c.arguments.command for c in messages[-1].tool_calls '
        'if c.name == "exec"], "\\n"))'
    )
    policy, adapter = _judge_policy(tmp_path, condition, rule_timeout=0.05)
    # Each item copies a large argument object, so all items take much longer than the deadline.
    heavy = '{"command": "rm", "pad": [' + ",".join(["0"] * 20_000) + "]}"
    conversation = _after_go(_exec_turn(*[heavy] * 500))

    (result,) = await policy.scan_many_async([conversation])

    assert (result.rule_matches, result.is_complete) == ([], False)
    assert result.timed_out_rules == ["R"]
    assert adapter.score_messages == []


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["messages[-1]", "assistant_messages[-1]"])
@pytest.mark.parametrize(
    "conversation", [_NO_CALLS, _NO_ASSISTANT], ids=["user-only", "without-an-assistant-message"]
)
async def test_an_explored_scan_without_an_assistant_message_calls_no_judge(
    tmp_path, source, conversation
) -> None:
    condition = _JUDGES["model-score"].format(input=_JOINED.format(source=source))
    policy, adapter = _judge_policy(tmp_path, condition)

    (result,) = await policy.scan_many_async([conversation], decision_paths=True, explore_rate=1.0)

    assert (result.rule_matches, result.is_complete) == ([], True)
    assert adapter.score_messages == []


@pytest.mark.parametrize(
    "check",
    [
        'any chunk in model("judge").split(messages[-1:], 200) '
        'where model("judge").score(chunk, text_ref("p")) > 0.5',
        '(any t in tools where t.name == "exec") and '
        'model("judge").score(messages[-1:], text_ref("p")) >= 0.5',
    ],
    ids=["chunked", "tools-conditioned"],
)
def test_a_false_guard_leaves_a_chunked_or_tools_conditioned_judge_uncalled(
    tmp_path, check
) -> None:
    guarded, adapter = _judge_scan(
        tmp_path, _GUARD + check, {**_ENDS_WITH_TOOL, "tools": _EXEC_TOOL}
    )

    assert (guarded.rule_matches, guarded.is_complete) == ([], True)
    assert (adapter.score_messages, adapter.tokenizations) == ([], 0)

    scored, adapter = _judge_scan(
        tmp_path, _GUARD + check, {**_ENDS_WITH_CALLS, "tools": _EXEC_TOOL}
    )

    assert (scored.rule_matches, scored.is_complete) == (["R"], True)
    assert adapter.score_messages == [[_CALLS_TURN]]


@pytest.mark.parametrize("judge", list(_JUDGES.values()), ids=list(_JUDGES))
def test_message_content_that_is_not_text_is_rejected_at_load(tmp_path, judge) -> None:
    condition = "any c in tool_calls where " + judge.format(input='message("user", c.arguments.*)')

    with pytest.raises(RuleValidationError, match=r"not object\[\]"):
        _judge_scan(tmp_path, condition, _ENDS_WITH_CALLS)


# ── model(...) expressions the compiler refuses ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def v2_policy_options(tmp_path_factory) -> dict:
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "judge": model_entry(score=True, tokens=True, classify=0),
                "email": model_entry(category="EMAIL", spans=0),
            },
            "texts": {"p": "Judge the input."},
        },
        tmp_path_factory.mktemp("model-expressions"),
        require_rules=False,
    )
    backend = ModelInferenceBackend(
        config, adapters={"fake": ModelAdapter()}, message_builders={"test": passthrough_builder}
    )
    return {"runtime_config": config, "model_inference_backend": backend}


_SPLIT = 'any chunk in model("judge").split(messages, 200) where '
_SCORED = 'condition: model("judge").score(message("user", {}), text_ref("p")) > 0.5'
_NAMES = "[c.name for c in messages[-1].tool_calls]"
_CHUNK_SCORE = 'model("judge").score(chunk, text_ref("p"))'
_SPAN_FILTER = (
    'condition: true\ncollect:\n    [span for span in model("email").spans(messages, 0.70) if {}]'
)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        pytest.param(
            'condition: model("judge").classify(foo) > 0.5',
            "unsupported model-expression reference 'foo'",
            id="unknown-reference",
        ),
        pytest.param(
            'condition: model("judge").classify(tools[:1]) > 0.5',
            "only Messages values can be sliced for a model",
            id="sliced-tools",
        ),
        pytest.param(
            'condition: model("judge").classify(message("user", text_ref("p")[0])) > 0.5',
            "only Messages or structured collection values can be indexed",
            id="indexed-text",
        ),
        pytest.param(
            'condition: model("judge").classify(add_message(text_ref("p"), "user", "x")) > 0.5',
            "add_message() receiver must be Messages",
            id="add-message-to-text",
        ),
        pytest.param(
            'condition: model("judge").classify(add_message(user_messages, "user", "x")) > 0.5',
            "add_message() receiver must be a root Messages collection",
            id="add-message-to-a-view",
        ),
        pytest.param(
            'condition: model("judge").classify(message("user", messages)) > 0.5',
            "message content must resolve to a string or text_ref, not Messages",
            id="messages-as-content",
        ),
        pytest.param(
            'condition: model("judge").classify(messages, 1, 2, 3) > 0.5',
            "model().classify() received the wrong number of arguments",
            id="classify-arity",
        ),
        pytest.param(
            'condition: model("judge").split(messages, 100) == 1',
            "model().split() is supported only as an any-comprehension collection",
            id="split-outside-a-comprehension",
        ),
        pytest.param(
            'condition: model("email").spans(messages, 0.5) == 1',
            "model().spans() is supported only as the direct collect source",
            id="spans-in-a-condition",
        ),
        pytest.param(
            'condition: model("judge").score(messages, text_ref("p")) + "x" > 1',
            "model arithmetic requires numeric operands",
            id="score-plus-text",
        ),
        pytest.param(
            'condition: model("judge").score(messages, text_ref("p")) == "x"',
            "model equality requires compatible operands",
            id="score-equals-text",
        ),
        pytest.param(
            'condition: model("judge").complete(messages).content > 1',
            "ordered model comparisons require numeric operands",
            id="text-ordered-against-a-number",
        ),
        pytest.param(
            'condition: model("judge").complete(messages).nope == "x"',
            "model result type AssistantMessage has no field 'nope'",
            id="unknown-completion-field",
        ),
        pytest.param(
            "condition: " + 'any c in model("email").spans(messages, 0.5) where true',
            "only model().split() can be a comprehension collection",
            id="spans-as-a-comprehension-collection",
        ),
        pytest.param(
            "condition: " + _SPLIT + "true",
            "model().split() must be consumed by a score predicate",
            id="split-without-a-score",
        ),
        pytest.param(
            "condition: " + _SPLIT + 'model("judge").score(chunk, text_ref("p"), chunk) > 0.5',
            "the bound chunk may only be used as model().score()'s messages argument",
            id="chunk-as-tools",
        ),
        pytest.param(
            "condition: "
            + 'any chunk in model("judge").split(messages, 200, tools) where '
            + _CHUNK_SCORE
            + " > 0.5",
            "model().split() takes no tools: tools are not messages; pass them to the chunk's score",
            id="split-takes-no-tools",
        ),
        pytest.param(
            "condition: " + _SPLIT + _CHUNK_SCORE + " + 1",
            "model split comprehension requires a boolean score predicate",
            id="numeric-split-predicate",
        ),
        pytest.param(
            "condition: " + _SPLIT + "not " + _CHUNK_SCORE,
            "not requires a boolean model expression",
            id="not-a-score",
        ),
        pytest.param(
            "condition: " + _SPLIT + _CHUNK_SCORE + " > 0.5 and (" + _CHUNK_SCORE + " + 1)",
            "and requires boolean model expressions",
            id="and-with-a-number",
        ),
        pytest.param(
            _SPAN_FILTER.format(
                'model("judge").classify(message("user", messages.extract(messages))) >= 0.7'
            ),
            "messages.extract() requires Messages and a span",
            id="extract-without-a-span",
        ),
        pytest.param(
            "parameters:\n    n: number\n"
            + _SPAN_FILTER.format(
                'model("judge").classify(message("user", messages.extract(span, params.n))) >= 0.7'
            ),
            "messages.extract() context bounds must be integers",
            id="extract-bound-from-a-number-parameter",
        ),
        pytest.param(
            _SCORED.format(_NAMES),
            "a map comprehension is supported only as the first argument of join()",
            id="map-comprehension-outside-join",
        ),
        pytest.param(
            _SCORED.format('join(messages[-1].tool_calls, ",")'),
            "join() values must be a map comprehension",
            id="join-over-a-collection",
        ),
        pytest.param(
            _SCORED.format(
                'join([c.name for c in messages[-1].tool_calls if model("judge").score('
                'message("user", c.name), text_ref("p")) > 0.5], ",")'
            ),
            "a map comprehension cannot call a model",
            id="model-call-in-the-map-filter",
        ),
        pytest.param(
            _SCORED.format(
                'join([model("judge").complete(message("user", c.name)).content '
                'for c in messages[-1].tool_calls], ",")'
            ),
            "a map comprehension cannot call a model",
            id="model-call-in-the-map-value",
        ),
        pytest.param(
            _SCORED.format('join([m.content for m in messages], ",")'),
            "a map comprehension needs a collection of structured items, such as a message's "
            "tool_calls",
            id="map-over-messages",
        ),
        pytest.param(
            _SCORED.format('join([c.name for c in messages[-1].tool_calls if c.name], ",")'),
            "a map comprehension filter must be a condition",
            id="map-filter-that-is-not-a-condition",
        ),
        pytest.param(
            'strings:\n    $risky = "rm"\n'
            + _SCORED.format(
                'join([c.name for c in messages[-1].tool_calls if $risky in c.function], ",")'
            ),
            "a string set is supported only as '$set in text'",
            id="string-set-in-a-structured-field",
        ),
        pytest.param(
            _SCORED.format("[c for c in messages[-1].tool_calls if true]"),
            "unsupported model expression FilterComprehension",
            id="filter-comprehension-as-content",
        ),
        pytest.param(
            _SCORED.format("len(messages)"),
            "function len() is not a model-expression input",
            id="function-as-content",
        ),
        pytest.param(
            _SCORED.format('join([c.argumentz.command for c in messages[-1].tool_calls], ",")'),
            "model result type ToolCall has no field 'argumentz'",
            id="unknown-field-of-a-map-item",
        ),
        pytest.param(
            _SCORED.format(f"join({_NAMES}, messages[0].content)"),
            "Function 'join' argument 'separator' must be written as a literal value",
            id="join-separator-not-literal",
        ),
        pytest.param(
            _SCORED.format(f"join({_NAMES})"),
            "Function 'join' requires at least 2 argument(s), but was called with 1",
            id="join-without-a-separator",
        ),
        pytest.param(
            "condition: " + _SPLIT + 'model("judge").score(chunk, join([chunk.name for chunk in '
            'messages[-1].tool_calls], ",")) > 0.5',
            "join() is supported only as message() or add_message() content in a model input",
            id="map-variable-named-like-the-chunk-in-a-split-score",
        ),
        pytest.param(
            f'condition: model("judge").complete(messages).content == join({_NAMES}, ",")',
            "join() is supported only as message() or add_message() content in a model input",
            id="join-outside-model-input-content",
        ),
    ],
)
def test_an_unsupported_model_expression_is_rejected_at_load(
    v2_policy_options, body, reason
) -> None:
    with pytest.raises(RuleValidationError) as error:
        compile_policy(f'rule "R"\n{body}\n', **v2_policy_options)

    assert reason in str(error.value)


# A judge of fixed positions, as each input reads the newest turns of _FIXED_CONVERSATION: the
# transcript span of a message list, or each message or call a constructed text read from, by
# position and field. Positions are timeline positions: 0 is the user message, 1 the assistant
# message, 2 its call.
_FIXED_INPUTS = {
    "one-message": ("messages[-1]", {"message_index": 1, "start": 0, "end": 0}),
    "slice": ("messages[-1:]", {"message_index": 1, "start": 0, "end": 0}),
    "field-of-a-message": (
        'message("user", user_messages[-1].content)',
        {"constructed": True, "pieces": [{"position": 0, "field": "content"}]},
    ),
    "two-indexes": (
        'message("user", user_messages[-1].content).add_message("assistant", '
        "assistant_messages[-1].tool_calls[0].arguments.command)",
        {
            "constructed": True,
            "pieces": [{"position": 0, "field": "content"}, {"position": 2, "field": "command"}],
        },
    ),
    "join": (
        'message("user", join([c.arguments.command for c in assistant_messages[-1].tool_calls'
        ' if c.name == "exec"], "\\n"))',
        {"constructed": True, "pieces": [{"position": 2, "field": "command"}]},
    ),
}
_FORMS = {
    "alone": ("{judge}", ["R"]),
    "beside-a-sibling": ('(any m in messages where m.role == "assistant") and {judge}', ["R"]),
    "under-not": ("not ({judge})", []),
}
_FIXED_CONVERSATION = _after_go(_exec_turn('{"command": "rm -rf /tmp/secret"}'))


async def _scan_with_path(tmp_path: Path, condition: str, conversation: dict):
    # Each scan reports what its own adapter was sent, as in a new process.
    result_cache.clear()
    policy, adapter = _judge_policy(tmp_path, condition)
    (result,) = await policy.scan_many_async(
        [conversation], decision_paths=True, explore_rate=1.0, explain=True
    )
    ((rule, artifact),) = policy._scanner.compiled_rules.items()
    tree = report_tree(artifact, scanner_digests(policy._scanner, policy._evidence_key))
    path = path_report(
        tree,
        (result.decision_paths or {}).get(rule),
        matched=bool(result.rule_matches) if result.is_complete else None,
        conversation=conversation,
        parameter_digest=result.parameter_digest,
        explain=True,
        explore_rate=1.0,
    )
    return result, adapter, path, telemetry_record(tree.report, path)


@pytest.mark.asyncio
async def test_a_judge_of_fixed_messages_is_scored_once_and_reported(tmp_path) -> None:
    judge = 'model("judge").score({input}, text_ref("p")) >= 0.5'
    for name, (judge_input, span) in _FIXED_INPUTS.items():
        for form, (template, matches) in _FORMS.items():
            condition = template.format(judge=judge.format(input=judge_input))
            result, adapter, path, telemetry = await _scan_with_path(
                tmp_path, condition, _FIXED_CONVERSATION
            )

            case = f"{name} {form}"
            assert (result.rule_matches, result.is_complete) == (matches, True), case
            assert len(adapter.score_messages) == 1, case
            scores = [step["score"] for step in path["path"] if "score" in step]
            assert scores == [
                {"value": 0.99, "operator": ">=", "threshold": 0.5, "evaluations": 1, "span": span}
            ], case
            steps = [step for step in telemetry["steps"] if "value" in step]
            assert [(s["value"], s["threshold"], s["span"]) for s in steps] == [
                (0.99, 0.5, span)
            ], case
            assert "secret" not in json.dumps([path, telemetry], default=str), case

    # A message that does not exist is empty input: no call, and the `not` form holds.
    absent = {"messages": _user("go")}
    for judge_input in (
        "assistant_messages[-1]",
        'message("user", assistant_messages[-1].content)',
        _FIXED_INPUTS["join"][0],
    ):
        for template, matches in (("{judge}", []), ("not ({judge})", ["R"])):
            condition = template.format(judge=judge.format(input=judge_input))
            result, adapter, *_ = await _scan_with_path(tmp_path, condition, absent)
            assert (result.rule_matches, result.is_complete, adapter.score_messages) == (
                matches,
                True,
                [],
            ), condition

    # A null content or a missing field is no content: that turn or that call is left out, and
    # an input left with no content makes no call. Arguments that are not a JSON object cannot
    # be decoded, so the rule fails.
    no_command = _after_go(_exec_turn('{"path": "/tmp"}'))
    for judge_input, conversation, outcome, scored in (
        (
            'message("user", assistant_messages[-1].content)',
            _after_go({**_exec_turn(_RM), "content": None}),
            ([], True),
            [],
        ),
        (_FIXED_INPUTS["join"][0], no_command, ([], True), []),
        (_FIXED_INPUTS["two-indexes"][0], no_command, (["R"], True), [_user("go")]),
        (_FIXED_INPUTS["join"][0], _after_go(_exec_turn('"rm -rf /"')), ([], False), []),
    ):
        condition = judge.format(input=judge_input)
        result, adapter, *_ = await _scan_with_path(tmp_path, condition, conversation)
        assert (result.rule_matches, result.is_complete) == outcome, condition
        assert adapter.score_messages == scored, condition

    # A per-call judge is scored at each call, so its score has no single place in the path.
    per_call = "any c in tool_calls where " + judge.format(
        input='message("user", c.arguments.command)'
    )
    result, adapter, path, _ = await _scan_with_path(tmp_path, per_call, _FIXED_CONVERSATION)
    assert (result.rule_matches, len(adapter.score_messages)) == (["R"], 1)
    assert [step for step in path["path"] if "score" in step] == []


_CHAT = {
    "messages": [
        {"role": "system", "content": "be safe"},
        *_user("mail alice@example.com"),
        {"role": "assistant", "content": "ok"},
    ]
}
_V2_SCORE = 'model("judge").score({}, text_ref("p")) >= 0.5'


@pytest.mark.parametrize(
    ("body", "conversation", "calls"),
    [
        pytest.param(
            "condition: " + _V2_SCORE.format("user_messages[-1]"),
            _CHAT,
            {"score_messages": [_user("mail alice@example.com")]},
            id="score",
        ),
        pytest.param(
            'condition: model("judge").classify(user_messages[-1]) > 0.5',
            _CHAT,
            {"classification_messages": [_user("mail alice@example.com")]},
            id="classify",
        ),
        pytest.param(
            'condition: model("judge").complete(user_messages[-1]).content == "blocked"',
            _CHAT,
            {"completions": 1},
            id="complete",
        ),
        pytest.param(
            'condition: model("judge").input_tokens(user_messages[-1]) == 22',
            _CHAT,
            {"tokenizations": 1},
            id="input-tokens",
        ),
        pytest.param(
            'condition: any chunk in model("judge").split(user_messages[-1], 200) where '
            + _V2_SCORE.format("chunk"),
            _CHAT,
            {"score_messages": [_user("mail alice@example.com")]},
            id="split-input",
        ),
        pytest.param(
            'condition: any chunk in model("judge").split(messages[1:], 200, system_messages[0]) '
            "where " + _V2_SCORE.format("chunk"),
            _CHAT,
            {"score_messages": [[*_CHAT["messages"]]]},
            id="split-context",
        ),
        pytest.param(
            'condition: any chunk in model_split("judge", user_messages[-1], 200) where '
            + _V2_SCORE.format("chunk"),
            _CHAT,
            {"score_messages": [_user("mail alice@example.com")]},
            id="model-split-input",
        ),
        pytest.param(
            'condition: true\ncollect:\n    [s for s in model("email").spans(user_messages[-1], '
            "0.70) if s.score.value > 0.5]",
            {"messages": _user("mail alice@example.com")},
            {"classification_messages": [_user("mail alice@example.com")]},
            id="spans",
        ),
        # A bound element always exists. The judge stays per element: the first element holds,
        # so it is the one message judged.
        pytest.param(
            "condition: any m in user_messages where " + _V2_SCORE.format("m"),
            {"messages": _user("go", "mail alice@example.com")},
            {"score_messages": [_user("go")]},
            id="element-of-a-comprehension",
        ),
    ],
)
def test_a_message_is_accepted_where_a_model_input_is_expected(
    tmp_path, body, conversation, calls
) -> None:
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "judge": model_entry(classify=0, score=True, tokens=True),
                "email": model_entry(category="EMAIL", spans=0),
            },
            "texts": {"p": "Judge the input."},
        },
        tmp_path,
        require_rules=False,
    )
    adapter = ModelAdapter()
    backend = ModelInferenceBackend(
        config, adapters={"fake": adapter}, message_builders={"test": passthrough_builder}
    )
    policy = compile_policy(
        f'rule "R"\n{body}\n', runtime_config=config, model_inference_backend=backend
    )

    result = policy.scan(conversation)

    assert (result.rule_matches, result.is_complete) == (["R"], True)
    assert {name: getattr(adapter, name) for name in calls} == calls


def test_a_model_reply_is_not_a_model_input(tmp_path) -> None:
    with pytest.raises(RuleValidationError, match="must be model_input or a message of the"):
        _judge_policy(tmp_path, _V2_SCORE.format('model("judge").complete(messages)'))
