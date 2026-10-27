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

"""Explain output: the evidence a match cites, and the runtime trace a policy records.

With explain on, a matched rule reports the timeline positions its verdict rests on: the
minimal set of turns that reproduces the match when the rule reads them, or every turn of the
domain when an absence or a universal is what matched. A recorded trace keeps every attempt
the scan made (one per captured value, one check per tool definition) for visualization.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from quarry import compile_policy
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.scanner import DSLScanner


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _call(name: str, call_id: str | None = None) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id or f"id-{name}",
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


def _response(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _tool(name: str, description: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": description}}


def _rule(condition: str, *strings: str) -> str:
    block = "strings:\n" + "".join(f"    {s}\n" for s in strings) if strings else ""
    return f'rule "R"\n{block}condition:\n    {condition}\n'


def _explained(rule: str, messages: list[dict], tools: list[dict] = ()):
    """The scan result and the matched rule's evidence, with explain on."""
    scanner = DSLScanner(QuarryConfig(quarry_rule_texts=(rule,)))
    result = scanner.scan_data({"messages": messages, "tools": list(tools)}, enable_explain=True)
    assert result.is_complete, result.completion_issues
    assert result.rule_matches == ["R"]
    (evidence,) = result.detailed_matches
    return result, evidence


_DIGITS = '$digits = "[0-9]+" regex'
_LATE_MATCH = [*(_user(f"filler {i}") for i in range(49)), _user("the target is here")]


_EVIDENCE = [
    # ── a match cites the turns that reproduce it ──────────────────────
    pytest.param(
        _rule('any m in user_messages where m.content contains "target"'),
        _LATE_MATCH,
        (),
        (49,),
        id="late-match-cites-one-turn",
    ),
    pytest.param(
        _rule('any m in user_messages where not (m.content contains "A")'),
        [_user("A"), _user("x"), _user("A")],
        (),
        (1,),
        id="negated-existential-step",
    ),
    pytest.param(
        _rule(
            '(any m in user_messages where m.content contains "A" and m.content contains "B") and (any m in user_messages where not (m.content contains "A"))'
        ),
        [_user("x"), _user("y"), _user("AB")],
        (),
        (0, 2),
        id="mixed-polarity-existentials",
    ),
    pytest.param(
        _rule('not (all m in user_messages where m.content contains "A")'),
        [_user("A"), _user("x"), _user("A")],
        (),
        (1,),
        id="negated-universal-cites-its-counterexample",
    ),
    pytest.param(
        _rule(
            '(any t in tools where not (t.description contains "A")) and (any m in user_messages where m.content contains "B")'
        ),
        [_user("filler"), _user("B")],
        [_tool("X", "X")],
        (1,),
        id="negative-tool-test-beside-an-event",
    ),
    pytest.param(
        _rule(
            '(any t in tools where not (t.description contains "A") and any m in user_messages where m.content contains "TARGET") and not (all m in user_messages where m.content contains "filler")'
        ),
        [_user("filler"), _user("filler"), _user("TARGET")],
        [_tool("X", "X")],
        (2,),
        id="tools-body-with-a-comprehension",
    ),
    pytest.param(
        _rule(
            '(any m in user_messages where not (m.content contains "A") and m.content contains "C") after(within: 2) (any m in user_messages where m.content contains "B")'
        ),
        [_user("B"), _user("A"), _user("C")],
        (),
        (0, 2),
        id="bounded-flow-with-a-negated-step",
    ),
    pytest.param(
        _rule(
            '(any m in user_messages where not (m.content contains "A") and m.content contains "C") after (any m in user_messages where m.content contains "B")'
        ),
        [_user("B"), _user("A"), _user("C")],
        (),
        (0, 2),
        id="unbounded-flow-with-a-negated-step",
    ),
    pytest.param(
        _rule('any m in user_messages where m.content contains "a" and called_tool("t")'),
        [_user("hi"), _user("a"), _user("b"), _call("t")],
        (),
        (1, 4),
        id="closed-part-cites-its-own-event",
    ),
    pytest.param(
        _rule(
            'any m in user_messages where m.content contains "a" and (any n in assistant_messages where n.content contains "b" and called_tool("t"))'
        ),
        [_user("a"), _assistant("b"), _assistant("c"), _call("t")],
        (),
        (0, 1, 4),
        id="closed-part-inside-a-closed-part",
    ),
    pytest.param(
        _rule(
            '(called_tool("b0") after(within: 10) called_tool("a0")) and (called_tool("b1") after(within: 10) called_tool("a1"))'
        ),
        [_call("a0"), _user("x"), _call("b0"), _user("y"), _call("a1"), _call("b1")],
        (),
        (1, 4, 7, 9),
        id="independent-flows-cite-each-flow",
    ),
    pytest.param(
        _rule(
            'any m in user_messages where m.content contains "a" and not (all n in assistant_messages where n.content contains "b")'
        ),
        [_user("hi"), _user("a"), _assistant("b"), _assistant("c"), _assistant("b")],
        (),
        (1, 3),
        id="false-universal-part-cites-its-counterexample",
    ),
    pytest.param(
        _rule(
            'any m in user_messages where m.content contains "x" and (m.content contains "a" or called_tool("t"))'
        ),
        [_user("x"), _user("xa"), _call("u")],
        (),
        (1,),
        id="false-part-beside-a-deciding-element-test",
    ),
    pytest.param(
        _rule('all t in tools where t.name == "x" and called_tool("y")'),
        [_user("hi"), _call("y")],
        [_tool("x", "X")],
        (2,),
        id="all-tools-body-cites-its-true-part",
    ),
    pytest.param(
        _rule('all t in tools where t.name == "x" or called_tool("y")'),
        [_user("hi"), _call("y")],
        [_tool("x", "X"), _tool("w", "W")],
        (2,),
        id="all-tools-body-cites-the-part-one-tool-needs",
    ),
    # ── an absence or a universal cites its whole domain ───────────────
    pytest.param(
        _rule('all m in user_messages where m.content contains "A" or m.content contains "B"'),
        [_user("A"), _user("B")],
        (),
        (0, 1),
        id="universal-domain",
    ),
    pytest.param(
        _rule('all m in user_messages where m.content contains "ok"'),
        [_user("ok"), _assistant("irrelevant"), _user("ok")],
        (),
        (0, 2),
        id="universal-cites-only-its-surface",
    ),
    pytest.param(
        _rule('not (any m in user_messages where m.content contains "bad")'),
        [_user("clean"), _user("clean")],
        (),
        (0, 1),
        id="absence-domain",
    ),
    pytest.param(
        _rule('all m in user_messages where m.content contains "ok"'),
        [_assistant("x")],
        (),
        (),
        id="universal-over-an-empty-surface-cites-nothing",
    ),
    pytest.param(_rule("true"), [_user("irrelevant")], (), (), id="constant-cites-nothing"),
    pytest.param(
        _rule(
            '(not (any m in user_messages where m.content contains "bad")) or (any m in user_messages where m.content contains "good")'
        ),
        [_user("clean"), _user("bad"), _user("good")],
        (),
        (0, 1, 2),
        id="absence-alternative-stays-conservative",
    ),
    pytest.param(
        _rule(
            'not ((any m in user_messages where m.content contains "X") and (any m in user_messages where m.content contains "Y"))'
        ),
        [_user("please help"), _assistant("ok"), _user("please X")],
        (),
        (0, 2),
        id="compound-absence",
    ),
    pytest.param(
        _rule(
            '(any t in tools where t.description contains "A") and not (any m in user_messages where m.content contains "bad")'
        ),
        [_user("clean"), _user("clean")],
        [_tool("A", "A")],
        (0, 1),
        id="tool-test-beside-an-absence",
    ),
    pytest.param(
        _rule(
            'not ((any m in user_messages where m.content contains "B") after(within: 1) (any m in user_messages where m.content contains "A"))'
        ),
        [_user("A"), _user("x"), _user("B")],
        (),
        (0, 1, 2),
        id="negated-bounded-flow",
    ),
    pytest.param(
        _rule(
            'not ((any m in user_messages where m.content contains "C") after (any m in user_messages where m.content contains "B"))'
        ),
        [_user("B"), _user("x"), _user("x")],
        (),
        (0, 1, 2),
        id="negated-unbounded-flow",
    ),
    pytest.param(
        _rule(
            '(any m in user_messages where m.content contains "A") until (any m in user_messages where m.content contains "B")'
        ),
        [_user("A"), _user("A"), _user("B")],
        (),
        (0, 1, 2),
        id="until-sustained-prefix",
    ),
    pytest.param(
        _rule(
            'not ((any m in user_messages where m.content contains "A") until (any m in user_messages where m.content contains "B"))'
        ),
        [_user("A"), _user("x"), _user("B")],
        (),
        (0, 1),
        id="negated-until-break",
    ),
    pytest.param(
        _rule('any m in user_messages where m.content contains "a" and not called_tool("t")'),
        [_user("hi"), _user("a"), _call("u")],
        (),
        (0, 1, 3),
        id="negated-closed-part-cites-its-domain",
    ),
    # ── a captured value cites where it was bound and referenced ───────
    pytest.param(
        _rule(
            "any m in user_messages where ($digits in m.content as value) and any r in tool_responses where r.content contains value",
            _DIGITS,
        ),
        [_user("42"), _user("7"), _call("lookup", "c1"), _response("c1", "42 and 7")],
        (),
        (1, 4),
        id="capture-cites-the-winning-value",
    ),
    pytest.param(
        _rule(
            'any a in user_messages where ("TOKEN" in a.content as tok) and any b in user_messages where not (b.content contains tok) and b.content contains "TARGET"'
        ),
        [_user("TOKEN"), _user("filler"), _user("TARGET")],
        (),
        (0, 1, 2),
        id="capture-with-a-negated-reference",
    ),
    pytest.param(
        _rule(
            'any a in user_messages where ("TOKEN" in a.content as tok) and not (any r in tool_responses where r.content contains tok)'
        ),
        [_user("TOKEN"), _call("x", "c1"), _response("c1", "nothing here")],
        (),
        (0, 3),
        id="capture-with-an-absent-reference",
    ),
]


@pytest.mark.parametrize(("rule", "messages", "tools", "positions"), _EVIDENCE)
def test_explain_cites_the_positions_the_match_rests_on(rule, messages, tools, positions):
    _, evidence = _explained(rule, messages, tools)
    assert tuple(step.position for step in evidence.steps) == positions


@pytest.mark.parametrize(
    ("rule", "messages", "captured"),
    [
        pytest.param(
            _rule(
                "any m in user_messages where ($digits in m.content as value) and any r in tool_responses where r.content contains value",
                _DIGITS,
            ),
            [_user("42"), _user("7"), _call("lookup", "c1"), _response("c1", "42 and 7")],
            "7",
            id="minimized-capture-reports-the-cited-value",
        ),
        pytest.param(
            _rule(
                "any m in user_messages where ($digits in m.content as value) and any r in tool_responses where r.content contains value",
                _DIGITS,
            ),
            [_user("id 42 and 7"), _call("lookup", "c1"), _response("c1", "selected 7")],
            "7",
            id="regex-capture-reports-the-matched-text",
        ),
        pytest.param(
            _rule(
                'any a in user_messages where ($digits in a.content as value) and any b in user_messages where not (b.content contains value) and b.content contains "TARGET"',
                _DIGITS,
            ),
            [_user("42"), _user("7"), _user("TARGET")],
            "42",
            id="first-value-when-the-whole-domain-is-cited",
        ),
    ],
)
def test_explain_reports_the_captured_value(rule, messages, captured):
    _, evidence = _explained(rule, messages)
    assert evidence.captured_value == captured


def test_explain_evidence_names_each_cited_turn_and_serializes():
    rule = _rule(
        'any m in user_messages where ("TOKEN" in m.content as tok) '
        "and any r in tool_responses where r.content contains tok"
    )
    messages = [_user("my TOKEN here"), _call("x", "c1"), _response("c1", "echoing TOKEN")]
    result, evidence = _explained(rule, messages)

    assert evidence.captured_value == "TOKEN"
    assert [step.source_path for step in evidence.steps] == [
        "user_messages[0]",
        "tool_responses[0]",
    ]
    assert [step.excerpt for step in evidence.steps] == ["my TOKEN here", "echoing TOKEN"]
    assert result.rule_traces is None
    serialized = json.loads(json.dumps(evidence.to_dict()))
    assert serialized["omitted_witness_steps"] == 0
    assert "trace" not in serialized


def test_explain_redacts_secrets_in_excerpts_and_captured_values():
    secret = "AKIAIOSFODNN7EXAMPLE"
    rule = _rule(
        "any m in user_messages where ($key in m.content as k) and "
        "any r in tool_responses where r.content contains k",
        '$key = "AKIA[0-9A-Z]{16}" regex',
    )
    messages = [_user(f"key {secret} here"), _call("x", "c1"), _response("c1", f"got {secret}")]
    _, evidence = _explained(rule, messages)

    assert secret not in json.dumps(evidence.to_dict())
    assert "«redacted»" in evidence.steps[0].excerpt


def test_a_scan_without_explain_reports_no_evidence():
    scanner = DSLScanner(
        QuarryConfig(
            quarry_rule_texts=(_rule('any m in user_messages where m.content contains "hi"'),)
        )
    )
    result = scanner.scan_data({"messages": [_user("hi")]})
    assert result.rule_matches == ["R"]
    assert result.detailed_matches is None


class _SlowOnLongText:
    """A configured classifier that answers ``E`` positively and stalls on long texts."""

    async def classify_messages(self, _config, messages, _tools, *, timeout=None):
        text = messages[0]["content"]
        if len(text) > 1000:
            await asyncio.sleep(5)
        probability = 0.9 if text == "E" else 0.1
        return SimpleNamespace(
            chunks=[SimpleNamespace(probabilities=[1 - probability, probability])]
        )


async def test_a_stalled_evidence_replay_does_not_cost_the_decided_match():
    # Dropping "zz" in a replay puts the long message right after "a", so the replay asks the
    # classifier about an event the verdict never needed, and that call stalls. The deadline
    # cuts the replay short; the match and a complete scan stand.
    rule = _rule(
        '(any m in user_messages where m.content contains "c") and '
        '(any m in user_messages where m.content contains "b") and '
        '((any m in user_messages where model_scan("pi", m.content, 0.5, 1)) '
        'after(within: 1) (any m in user_messages where m.content contains "a"))'
    )
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "pi": {
                    "adapter": "fake",
                    "model": "test-classifier",
                    "classify": {"class_index": 1},
                }
            },
        },
        Path(__file__).resolve().parents[2],
        require_rules=False,
    )
    config = replace(config, quarry_rule_texts=(rule,))
    scanner = DSLScanner(
        config,
        model_inference_backend=ModelInferenceBackend(config, adapters={"fake": _SlowOnLongText()}),
    )
    messages = [_user(c) for c in ["c a", "zz", "b " + "x" * 3000, "a", "E"]]

    result = await asyncio.wait_for(
        scanner.scan_data_async(
            {"messages": messages},
            enable_explain=True,
            deadline_seconds=0.5,
        ),
        timeout=4,
    )

    assert result.rule_matches == ["R"]
    assert result.is_complete, result.completion_issues
    assert result.detailed_matches


async def _trace(rule: str, messages: list[dict], tools: list[dict] = ()):
    """The recorded runtime trace of one explained scan, as a policy returns it."""
    (result,) = await compile_policy(rule).scan_many_async(
        [{"messages": messages, "tools": list(tools)}], explain=True, record_traces=True
    )
    assert result.is_complete, result.completion_issues
    return result, result.rule_traces["R"]


_LOOKUP = [_call("lookup", "c1")]


async def test_a_trace_keeps_one_check_per_tool_definition_and_never_merges_them():
    # Each tool satisfies one conjunct only; the checks must not combine into a match.
    result, trace = await _trace(
        _rule(
            'any tool in tools where tool.description contains "A" and tool.description contains "B"'
        ),
        [_user("hello")],
        [_tool("a", "A"), _tool("b", "B")],
    )
    assert result.rule_matches == []
    (attempt,) = trace.attempts
    assert [check.tool_binding_index for check in attempt.tool_attempts] == [0, 1]
    assert [check.matched for check in attempt.tool_attempts] == [False, False]


async def test_a_trace_selects_every_failed_tool_check_behind_a_negation():
    result, trace = await _trace(
        _rule(
            'not (any t in tools where t.description contains "A" or t.description contains "B")'
        ),
        [_user("hello")],
        [_tool("x", "X"), _tool("y", "Y")],
    )
    assert result.rule_matches == ["R"]
    checks = trace.selected_attempt.tool_attempts
    assert [(check.selected, check.matched) for check in checks] == [(True, False), (True, False)]


async def test_a_trace_keeps_one_attempt_per_captured_value_and_selects_the_match():
    result, trace = await _trace(
        _rule(
            "any m in user_messages where ($digits in m.content as value) "
            "and any r in tool_responses where r.content contains value",
            _DIGITS,
        ),
        [_user("ids 42 and 7"), *_LOOKUP, _response("c1", "selected 7")],
    )
    assert result.rule_matches == ["R"]
    assert [(a.attempt_id, a.matched) for a in trace.attempts] == [
        ("capture:0", False),
        ("capture:1", True),
    ]
    assert trace.selected_attempt_id == "capture:1"


async def test_a_trace_counts_the_unmatched_attempts_beyond_its_limit():
    candidates = " ".join(str(value) for value in range(100, 120))
    _, trace = await _trace(
        _rule(
            "any m in user_messages where ($digits in m.content as value) "
            "and any r in tool_responses where r.content contains value",
            _DIGITS,
        ),
        [_user(f"ids {candidates} 7"), *_LOOKUP, _response("c1", "selected 7")],
    )
    attempt_ids = [attempt.attempt_id for attempt in trace.attempts]
    assert attempt_ids == [f"capture:{index}" for index in range(16)] + ["capture:20"]
    assert trace.omitted_attempts == 4
    assert trace.to_dict()["omitted_attempts"] == 4
    assert trace.selected_attempt_id == "capture:20"
