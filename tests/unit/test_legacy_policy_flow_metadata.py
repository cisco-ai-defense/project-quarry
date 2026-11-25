# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Legacy chat artifacts keep their bytes without weakening declared-input checks."""

import json
from unittest.mock import patch

import orjson
import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.engine.contracts import COMPILER_VERSION


def legacy_wire(policy):
    value = json.loads(policy.to_bytes())
    del value["rule"]["artifact"]["flow_requirements"]
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


@pytest.mark.parametrize(
    "condition",
    [
        'any m in messages where m.content == "start"',
        '(any m in messages where m.content == "start") before '
        '(any m in messages where m.content == "finish")',
    ],
)
def test_legacy_chat_policy_preserves_bytes_identity_and_verdicts(condition):
    current = compile_policy(f'rule "Read" condition: {condition}')
    wire = legacy_wire(current)
    with patch("quarry.dsl.parser.DSLParser.parse", side_effect=AssertionError("parser called")):
        loaded = load_compiled_policy(wire)
    assert loaded.to_bytes() == wire
    assert loaded.policy_digest == current.policy_digest
    for data in [
        {"messages": [{"role": "user", "content": "start"}]},
        {
            "messages": [
                {"role": "user", "content": "start"},
                {"role": "assistant", "content": "finish"},
            ]
        },
        {"messages": [{"role": "user", "content": "other"}]},
    ]:
        before, after = current.scan(data), loaded.scan(data)
        assert before.is_complete and after.is_complete
        assert before.rule_matches == after.rule_matches
    assert "flow_requirements" in json.loads(current.to_bytes())["rule"]["artifact"]


def test_legacy_unsupported_compiler_reports_version_before_schema():
    value = json.loads(legacy_wire(compile_policy('rule "Read" condition: true')))
    value["rule"]["artifact"]["compiler_version"] = "compiler-11"
    with pytest.raises(ValueError, match=f"compiler-11.*unsupported.*{COMPILER_VERSION}"):
        load_compiled_policy(orjson.dumps(value))


@pytest.mark.parametrize("remove_input_fingerprints", [False, True])
def test_declared_input_cannot_omit_flow_requirements(tmp_path, remove_input_fingerprints):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "inputs": {"memory.writes": {"facts": {"content": {"type": "string"}}}},
        },
        tmp_path,
        require_rules=False,
    )
    policy = compile_policy(
        'rule "Read" condition: any w in memory.writes where w.content == "x"',
        runtime_config=config,
    )
    value = json.loads(legacy_wire(policy))
    if remove_input_fingerprints:
        del value["inputs"]
    with pytest.raises(ValueError, match="flow_requirements is required for declared-input"):
        load_compiled_policy(orjson.dumps(value), runtime_config=config)


def test_legacy_shape_still_rejects_unknown_artifact_fields():
    value = json.loads(legacy_wire(compile_policy('rule "Read" condition: true')))
    value["rule"]["artifact"]["unknown"] = None
    with pytest.raises(ValueError, match="unknown unknown"):
        load_compiled_policy(orjson.dumps(value))


def test_legacy_shape_cannot_claim_an_input_clock_without_flow_metadata():
    value = json.loads(legacy_wire(compile_policy('rule "Read" condition: true')))
    value["rule"]["artifact"]["notes"].append("orders input items")
    with pytest.raises(ValueError, match="flow_requirements is required for declared-input"):
        load_compiled_policy(orjson.dumps(value))
