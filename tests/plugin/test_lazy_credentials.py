# Copyright 2025 Cisco Systems, Inc. and its affilitates
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


"""Synthetic ownership controls for lazy plugin model credential validation."""

import json
from pathlib import Path

import pytest

from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.plugin.service import QuarryPlugin
from quarry.types.exceptions import ConfigurationError

LOCAL_RULE = 'rule "LocalOnly" condition: true'
MODEL_RULE = 'rule "Selected" condition: model("used").score(messages, "policy") > 0.5'
DATA = {"messages": [{"role": "user", "content": "synthetic local input"}]}


def unused_config(tmp_path: Path, count: int) -> QuarryConfig:
    return QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                f"unused_{index}": {"model": "openai/synthetic-unused"} for index in range(count)
            },
        },
        tmp_path,
        require_rules=False,
    )


@pytest.mark.parametrize("model_count", [0, 1, 2])
@pytest.mark.parametrize(
    "entry", ["startup", "index", "models", "compile", "tree", "scan", "batch"]
)
async def test_local_tools_never_construct_unused_model_backends(
    monkeypatch, tmp_path, model_count, entry
):
    def forbidden(*args, **kwargs):
        raise AssertionError("unused model backend constructed")

    monkeypatch.setattr(ModelInferenceBackend, "__init__", forbidden)
    plugin = QuarryPlugin(unused_config(tmp_path, model_count))
    if entry in {"index", "models"}:
        assert plugin.reference(action="index", limit=1, include_models=entry == "models")
    elif entry == "compile":
        assert plugin.compile(rule_text=LOCAL_RULE)["valid"] is True
    elif entry == "tree":
        assert plugin.decision_tree(rule_text=LOCAL_RULE)["valid"] is True
    elif entry in {"scan", "batch"}:
        arguments = {"conversation": DATA} if entry == "scan" else {"conversations": [DATA, DATA]}
        result = await plugin.scan(rule_text=LOCAL_RULE, **arguments)
        assert result["completion_state"] == "complete"


class SyntheticScoreAdapter:
    def __init__(self):
        self.calls = 0

    async def score(self, config, messages, *, timeout=None):
        self.calls += 1
        return 0.75


def selected_config(tmp_path: Path) -> QuarryConfig:
    return QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "unused": {"model": "synthetic-unavailable", "adapter": "unregistered"},
                "used": {
                    "model": "synthetic-used",
                    "adapter": "synthetic",
                    "score": {
                        "message_builder": "constitution",
                        "boundary_seed": 7,
                        "tokens": {"negative": "0", "positive": "1"},
                    },
                },
            },
        },
        tmp_path,
        require_rules=False,
    )


@pytest.mark.parametrize("available", [False, True])
def test_only_used_alias_checks_credentials_when_its_rule_is_prepared(
    monkeypatch, tmp_path, available
):
    checks = []

    def credentials(self, aliases):
        checks.append(sorted(aliases))
        assert set(aliases) == {"used"}, "unused alias reached credential checks"
        if not available:
            raise ConfigurationError("Synthetic selected model credentials unavailable")

    monkeypatch.setattr(ModelInferenceBackend, "require_credentials", credentials)
    adapter = SyntheticScoreAdapter()
    plugin = QuarryPlugin(selected_config(tmp_path), model_adapters={"synthetic": adapter})
    assert checks == []
    report = plugin.compile(rule_text=MODEL_RULE)
    assert checks == [["used"]]
    assert report["valid"] is available
    assert adapter.calls == 0
    if not available:
        assert report["stage"] == "configure"
        assert (
            report["diagnostics"][0]["message"]
            == "Synthetic selected model credentials unavailable"
        )


async def test_used_alias_still_dispatches_once_and_unused_alias_stays_unloaded(
    monkeypatch, tmp_path
):
    checks = []

    def credentials(self, aliases):
        checks.append(sorted(aliases))
        assert set(aliases) == {"used"}

    monkeypatch.setattr(ModelInferenceBackend, "require_credentials", credentials)
    adapter = SyntheticScoreAdapter()
    plugin = QuarryPlugin(selected_config(tmp_path), model_adapters={"synthetic": adapter})
    assert checks == []
    result = await plugin.scan(rule_text=MODEL_RULE, conversation=DATA)
    assert result["completion_state"] == "complete"
    assert checks == [["used"]]
    assert adapter.calls == 1


@pytest.mark.parametrize(
    "provider,variable", [("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")]
)
@pytest.mark.parametrize("tool", ["quarry_compile", "quarry_scan"])
async def test_real_mcp_fails_closed_for_a_used_alias_without_credentials(
    monkeypatch, tmp_path, provider, variable, tool
):
    from quarry.integrations import model_inference
    from quarry.plugin import server

    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    async def forbidden(*args, **kwargs):
        raise AssertionError("provider call reached despite missing credentials")

    monkeypatch.setattr(model_inference.LiteLLMAdapter, "score", forbidden)
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "used": {
                    "model": provider + "/synthetic-unavailable",
                    "score": {
                        "message_builder": "constitution",
                        "boundary_seed": 7,
                        "tokens": {"negative": "0", "positive": "1"},
                    },
                }
            },
        },
        tmp_path,
        require_rules=False,
    )
    monkeypatch.setattr(server, "_service", QuarryPlugin(config))
    arguments = {"rule_text": MODEL_RULE}
    if tool == "quarry_scan":
        arguments["conversation"] = DATA
    packet = await server.mcp.call_tool(tool, arguments)
    payload = packet.structured_content
    assert json.loads(packet.content[0].text) == payload
    diagnostics = payload["diagnostics"]
    assert len(diagnostics) == 1 and variable in diagnostics[0]["message"]
    assert len(diagnostics[0]["message"]) < 300
    if tool == "quarry_scan":
        assert payload["completion_state"] == "invalid"
    else:
        assert payload["valid"] is False and payload["stage"] == "configure"
