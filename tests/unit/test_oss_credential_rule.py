# Copyright 2026 Cisco Systems, Inc. and its affiliates
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

"""Exercise credential rule selection at the model boundary with a fake judge."""

import json
from pathlib import Path

import pytest

from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.policy import compile_policy
from tests.unit.model_fakes import model_entry, passthrough_builder

ROOT = Path(__file__).resolve().parents[2]
RULE_DIR = (
    ROOT
    / "rules/rule-packs/core/coding-agents-runtime-behavior-rules/credential_access_sensitive_data_tool_call"
)


class RecordingJudge:
    def __init__(self, score):
        self.calls = []
        self.value = score

    async def score(self, _config, messages, *, timeout=None):
        self.calls.append(messages)
        return self.value


def make_policy(tmp_path, score):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"credential_access_judge": model_entry(score=True)},
            "texts": {
                "credential_access_judge_v1": (
                    RULE_DIR / "resources/credential_access_judge_v1.md"
                ).read_text()
            },
        },
        tmp_path,
        require_rules=False,
    )
    adapter = RecordingJudge(score)
    backend = ModelInferenceBackend(
        config, adapters={"fake": adapter}, message_builders={"test": passthrough_builder}
    )
    source = (RULE_DIR / "credential_access_sensitive_data_tool_call.rule").read_text()
    return compile_policy(source, runtime_config=config, model_inference_backend=backend), adapter


def conversation(*commands):
    return {
        "messages": [
            {"role": "user", "content": "Inspect this workflow."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": command}),
                        },
                    }
                    for i, command in enumerate(commands)
                ],
            },
        ]
    }


def test_nonmatching_command_skips_judge(tmp_path):
    policy, adapter = make_policy(tmp_path, 0.9)
    result = policy.scan(conversation("git status"), {"threshold": 0.5})
    assert result.is_complete
    assert not result.rule_matches
    assert not adapter.calls


@pytest.mark.parametrize(("score", "matched"), [(0.9, True), (0.2, False)])
def test_only_selected_command_is_scored_against_threshold(tmp_path, score, matched):
    policy, adapter = make_policy(tmp_path, score)
    result = policy.scan(conversation("git status", "gh auth token"), {"threshold": 0.5})
    assert result.is_complete
    assert bool(result.rule_matches) is matched
    assert len(adapter.calls) == 1
    assert "gh auth token" in adapter.calls[0][1]["content"]
    assert "git status" not in adapter.calls[0][1]["content"]


def test_multiple_selected_calls_are_scored_separately(tmp_path):
    policy, adapter = make_policy(tmp_path, 0.2)

    async def score_by_command(_config, messages, *, timeout=None):
        adapter.calls.append(messages)
        return 0.9 if "gh auth token" in messages[1]["content"] else 0.2

    adapter.score = score_by_command
    result = policy.scan(
        conversation("vault kv get secret/example", "gh auth token"), {"threshold": 0.5}
    )
    assert result.is_complete
    assert result.rule_matches
    assert len(adapter.calls) == 2
    assert "vault kv get" in adapter.calls[0][1]["content"]
    assert "gh auth token" in adapter.calls[1][1]["content"]


def test_sensitive_file_read_reaches_judge_without_command_field(tmp_path):
    policy, adapter = make_policy(tmp_path, 0.9)
    item = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "/home/example/.aws/credentials"}),
                        },
                    }
                ],
            }
        ],
    }
    result = policy.scan(item, {"threshold": 0.5})
    assert result.is_complete and result.rule_matches
    assert len(adapter.calls) == 1
    assert ".aws/credentials" in adapter.calls[0][1]["content"]
