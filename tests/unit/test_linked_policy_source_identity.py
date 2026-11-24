# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""A prepared entry's source identity includes its linked helper contents."""

import json
from dataclasses import replace

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.safety.limits import SafetyLimits


@pytest.mark.parametrize("configured_helper", [False, True], ids=["inline", "file"])
def test_saved_policy_source_hash_tracks_linked_helper_contents(tmp_path, configured_helper):
    helper_path = tmp_path / "synthetic-helper.rule"
    entry = 'rule "Entry" imports: helper = "Helper" condition: helper(messages)'
    config = QuarryConfig(
        config_version=2,
        quarry_rules_paths=(helper_path,) if configured_helper else (),
        safety_limits=SafetyLimits(rule_timeout=None),
    )
    hashes = []
    for literal, expected in [("needle", ["Entry"]), ("other text", [])]:
        helper = 'rule "Helper" condition: user_said(' + json.dumps(literal) + ")"
        if configured_helper:
            helper_path.write_text(helper, encoding="utf-8")
            source = entry
        else:
            source = helper + "\n" + entry
        policy = compile_policy(source, entry_rule="Entry", runtime_config=config)
        wire = policy.to_bytes()
        hashes.append(json.loads(wire)["source_sha256"])
        if configured_helper:
            helper_path.unlink()
        restored = load_compiled_policy(wire, runtime_config=replace(config, quarry_rules_paths=()))
        assert restored.to_bytes() == wire
        for prepared in [policy, restored]:
            result = prepared.scan({"messages": [{"role": "user", "content": "needle"}]})
            assert result.is_complete, result.evaluation_errors
            assert result.rule_matches == expected
    assert hashes[0] != hashes[1]
