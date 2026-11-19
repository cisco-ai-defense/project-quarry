# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Computed frames retain declared clocks, field isolation and rule-owned storage."""

from pathlib import Path

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from tests.unit.model_fakes import FakeAdapter, v1_backend
from tests.unit.test_nested_rule_runtime_contracts import CHAT, _policy
from tests.unit.test_nested_rule_runtime_contracts import model_config as model_config


def configured():
    return QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "inputs": {
                "memory.notes": {"facts": {"kind": {"type": "string"}}},
                "audit.unread": {"facts": {"kind": {"type": "string"}}},
            },
            "transforms": {"strip": {"operations": [{"op": "strip"}]}},
            "rule_timeout": None,
        },
        Path("/tmp"),
        require_rules=False,
    )


def policy(condition, restored):
    cfg = configured()
    result = compile_policy('rule "Frame" condition: ' + condition, runtime_config=cfg)
    assert result._scanner.compiled_rules["Frame"].computed_conditions
    if restored:
        wire = result.to_bytes()
        result = load_compiled_policy(wire, runtime_config=cfg)
        assert result.to_bytes() == wire
    return result


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
@pytest.mark.parametrize(
    "selection,can_order",
    [
        ('messages.transform("strip")', True),
        ("(messages[:1] + messages[1:])", True),
        ("(messages[1:] + messages[:1] + messages[:1])", False),
    ],
)
@pytest.mark.parametrize("clock", ["absent", "invalid", "chat-first", "note-first"])
def test_computed_frame_orders_only_against_the_supplied_shared_clock(
    restored, explain, selection, can_order, clock
):
    condition = (
        f'any m in {selection} where (m.content contains "go") '
        'before (any n in memory.notes where n.kind == "note")'
    )
    data = {
        "messages": [
            {"role": "user", "content": " go "},
            {"role": "assistant", "content": "stop"},
        ],
        "memory": {"notes": [{"id": "n", "kind": "note"}]},
        # A frame must not validate or materialize an unrelated declared section.
        "audit": {"unread": "invalid-unused-section"},
    }
    chat = [{"message_index": 0}, {"message_index": 1}]
    note = {"input": "memory.notes", "id": "n"}
    if clock == "chat-first":
        data["timeline"] = [*chat, note]
    elif clock == "note-first":
        data["timeline"] = [note, *chat]
    elif clock == "invalid":
        data["timeline"] = [*chat, {"input": "memory.notes", "id": "missing"}]
    result = policy(condition, restored).scan(data, explain=explain)
    incomplete = clock in {"absent", "invalid"} or not can_order
    assert result.completion_state == ("incomplete" if incomplete else "complete"), (
        result.evaluation_errors
    )
    assert result.rule_matches == (["Frame"] if clock == "chat-first" and can_order else [])
    if incomplete:
        assert [failure.to_dict() for failure in result.failures] == [
            {"category": "invalid_input", "rule_name": "Frame"}
        ]
        if clock in {"chat-first", "note-first"}:
            assert "cannot order a derived messages check" in result.evaluation_errors["Frame"]


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
def test_constructed_frame_cannot_borrow_a_conversation_clock(restored, explain):
    condition = (
        'any m in (message("user", "go") + messages[:0]) '
        'where (m.content contains "go") '
        'before (any n in memory.notes where n.kind == "note")'
    )
    data = {
        "messages": [{"role": "user", "content": "unrelated"}],
        "memory": {"notes": [{"id": "n", "kind": "note"}]},
        "timeline": [{"message_index": 0}, {"input": "memory.notes", "id": "n"}],
    }
    result = policy(condition, restored).scan(data, explain=explain)
    assert not result.is_complete and result.rule_matches == []
    assert [failure.to_dict() for failure in result.failures] == [
        {"category": "invalid_input", "rule_name": "Frame"}
    ]


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
def test_computed_span_body_does_not_read_unused_nonfinite_arguments(restored, explain):
    source = (
        'rule "Frame" strings: $needle="needle" condition: '
        'any m in messages.transform("strip") '
        "where len($needle.spans(m.content)) == 1"
    )
    cfg = configured()
    result = compile_policy(source, runtime_config=cfg)
    assert result._scanner.compiled_rules["Frame"].computed_conditions
    if restored:
        wire = result.to_bytes()
        result = load_compiled_policy(wire, runtime_config=cfg)
        assert result.to_bytes() == wire
    scanned = result.scan(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": " needle ",
                    "tool_calls": [
                        {
                            "id": "c",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"n": NaN}'},
                        }
                    ],
                }
            ]
        },
        explain=explain,
    )
    assert scanned.is_complete, scanned.evaluation_errors
    assert scanned.rule_matches == ["Frame"]


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
@pytest.mark.parametrize("negated", [False, True])
@pytest.mark.parametrize("index", [1, -2])
def test_computed_child_missing_model_input_cannot_become_a_negated_match(
    model_config, restored, explain, negated, index
):
    selection = "(messages[1:2] + messages[:0])"
    score = f'llm_score("gemma4", "p", message("user", {selection}[{index}].content)) > 0.5'
    body = f"not ({score})" if negated else score
    source = f'rule "Frame" condition: any m in {selection} where {body}'
    adapter = FakeAdapter()
    compiled = _policy(source, model_config, restored, v1_backend(model_config, adapter))
    assert compiled._scanner.compiled_rules["Frame"].computed_conditions
    result = compiled.scan(CHAT, explain=explain)
    assert result.completion_state == "incomplete"
    assert result.rule_matches == []
    assert "out of range" in result.evaluation_errors["Frame"]
    assert not adapter.called()
