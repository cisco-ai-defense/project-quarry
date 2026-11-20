# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Nested frames preserve cross-section until and lazy message-field guards."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest

from quarry import DSLScanner, InputAdapter, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.integrations.resources import requirements_from_artifact
from quarry.policy import CompiledPolicy
from tests.engine.test_cross_section_until_clock import AUDIT, CONDITION, DATA, MEMORY, config
from tests.unit.test_nested_rule_guard_integration import config as transform_config
from tests.unit.test_nested_rule_runtime_contracts import _policy, _wrappers


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "clock", ["absent", "invalid", "audit-first", "release-first", "empty-audit", "empty-memory"]
)
def test_nested_until_preserves_explicit_order_and_empty_surface_settlement(
    restored, reverse, clock
):
    source, name = _wrappers('rule "Leaf" condition: ' + CONDITION, depth=2)
    data = deepcopy(DATA)
    # The caller explicitly passes messages; the argument's section must be present.
    data["messages"] = [{"role": "user", "content": "unrelated"}]
    chat = {"message_index": 0}
    if clock == "invalid":
        data["timeline"] = [chat, AUDIT, {"input": "memory.writes", "id": "missing"}]
    elif clock in {"audit-first", "release-first"}:
        data["timeline"] = (
            [chat, AUDIT, MEMORY] if clock == "audit-first" else [chat, MEMORY, AUDIT]
        )
    elif clock == "empty-audit":
        data["audit"]["records"] = []
    elif clock == "empty-memory":
        data["memory"]["writes"] = []
    result = _policy(source, config(reverse), restored).scan(data, explain="support")
    assert result.completion_state == (
        "incomplete" if clock in {"absent", "invalid"} else "complete"
    ), result.evaluation_errors
    assert result.rule_matches == ([name] if clock in {"release-first", "empty-audit"} else [])
    if clock in {"absent", "invalid"}:
        assert [f.to_dict() for f in result.failures] == [
            {"category": "invalid_input", "rule_name": name}
        ]


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("selected", ["messages[0]", 'messages.transform("strip")[0]'])
@pytest.mark.parametrize("explain", [False, True, "support"])
def test_called_content_span_does_not_read_unused_nonfinite_arguments(
    tmp_path, restored, selected, explain
):
    source, name = _wrappers(
        'rule "Leaf" strings: $p="needle" condition: len($p.spans(messages[0].content))==1',
        depth=2,
        selected=selected,
    )
    data = {
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
    }
    result = _policy(source, transform_config(tmp_path), restored).scan(data, explain=explain)
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == [name]


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("note_first", [False, True])
def test_native_selected_frames_keep_clock_order_without_reading_unused_sections(
    restored, note_first
):
    class Native(InputAdapter):
        name = "nested-native"
        version = "1"
        sections = ("messages", "memory.notes", "audit.records")

        def convert(self, native, wanted):
            assert wanted == frozenset({"messages", "memory.notes"})
            return {"messages": native["chat"], "memory.notes": native["notes"]}

    source, name = _wrappers(
        'rule "Leaf" condition: (any n in memory.notes where n.id=="n") after user_said("go")',
        depth=2,
    )
    cfg = QuarryConfig(
        config_version=2,
        inputs={
            "memory.notes": {"content": ["body"]},
            "audit.records": {"facts": {"kind": {"type": "string"}}},
        },
        adapters={"raw": {"adapter": "nested-native"}},
    )
    adapters = {"nested-native": Native()}
    scanner = DSLScanner(replace(cfg, quarry_rule_texts=(source,)), input_adapters=adapters)
    artifact = scanner.compiled_rules[name]
    entry = DSLScanner._from_compiled_policy(
        cfg,
        rule=scanner.loaded_rules[name],
        artifact=artifact,
        resource_requirements=requirements_from_artifact(artifact),
        input_adapters=adapters,
    )
    policy = CompiledPolicy._from_scanner(entry, source)
    if restored:
        raw = policy.to_bytes()

        def forbidden(*args, **kwargs):
            raise AssertionError("loading a nested artifact entered the parser/compiler")

        with (
            patch("quarry.dsl.parser.DSLParser.parse", forbidden),
            patch("quarry.dsl.parser.DSLParser.parse_many", forbidden),
            patch("quarry.scanner.compile_rule", forbidden),
        ):
            policy = load_compiled_policy(raw, runtime_config=cfg, input_adapters=adapters)
            assert policy.to_bytes() == raw
    chat, note = {"message_index": 0}, {"input": "memory.notes", "id": "n"}
    result = policy.scan(
        {
            "raw": {
                "chat": [{"role": "user", "content": "go"}],
                "notes": [{"id": "n", "body": "go"}],
                "unread": "not-an-item-list",
            },
            "timeline": [note, chat] if note_first else [chat, note],
        },
        explain=True,
    )
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ([] if note_first else [name])
