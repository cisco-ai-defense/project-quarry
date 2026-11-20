# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Declared sections retain the newer lazy chat and opaque-transform contracts."""

from dataclasses import replace

import pytest

from quarry import compile_policy, load_compiled_policy
from tests.unit.test_adapter_transform_dsl import configured

INPUTS = {"memory.writes": {"facts": {"content": {"type": "string"}}}}


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
@pytest.mark.parametrize(
    "opaque",
    [
        {"role": "system", "content": "opaque", "refusal": "no"},
        {"role": "user", "content": "opaque", "refusal": "no"},
        {"role": "alien", "content": [{"type": "image_url", "image_url": {"url": "data:bad"}}]},
    ],
)
def test_input_and_transform_reads_leave_opaque_holders_and_tools_unread(
    tmp_path, prepared, explain, opaque
):
    config, backend, provider = configured(tmp_path)
    config = replace(config, inputs=INPUTS)
    source = (
        'rule "Read" condition: messages.transform("convert")[1].content == "BLOCK" '
        'and (any w in memory.writes where w.content == "safe")'
    )
    policy = compile_policy(source, runtime_config=config, model_inference_backend=backend)
    if prepared:
        wire = policy.to_bytes()
        policy = load_compiled_policy(wire, runtime_config=config, model_inference_backend=backend)
        assert policy.to_bytes() == wire
    data = {
        "messages": [opaque, {"role": "user", "content": "block"}],
        "tools": [{"type": "unknown", "function": {"name": "lookup"}}],
        "memory": {"writes": [{"id": "w", "content": "safe"}]},
    }
    result = policy.scan(data, explain=explain)
    assert result.is_complete and result.rule_matches == ["Read"]
    assert provider.parts == [{"type": "text", "text": "block"}]
    assert data["messages"][0] == opaque
