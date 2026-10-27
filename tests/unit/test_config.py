# Copyright 2025 Cisco Systems, Inc. and its affiliates
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

"""Tests for configuration loading and validation."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from quarry.config import QuarryConfig
from quarry.safety.limits import SafetyLimits
from quarry.types.exceptions import ConfigurationError


class TestQuarryConfig:
    def test_config_default_values(self):
        config = QuarryConfig()

        assert config.quarry_rules_paths == ()

        assert config.models == {}
        assert config.safety_limits.rule_timeout == 1.0
        assert config.safety_limits.max_nesting_depth == 100
        with pytest.raises(ValueError, match="max_nesting_depth"):
            SafetyLimits(max_nesting_depth=True)

    def test_version_two_uses_one_rule_wide_timeout(self, tmp_path: Path):
        config = QuarryConfig(config_version=2)

        assert config.safety_limits.rule_timeout == 60.0
        assert config.safety_limits.inference_timeout is None

        parsed = QuarryConfig.from_mapping(
            {"config_version": 2, "rules": ["test.rule"], "rule_timeout": 12.5},
            tmp_path,
        )
        assert parsed.safety_limits.rule_timeout == 12.5
        assert parsed.safety_limits.inference_timeout is None

        with pytest.raises(ConfigurationError, match="inference_timeout.*config_version 1"):
            QuarryConfig.from_mapping(
                {
                    "config_version": 2,
                    "rules": ["test.rule"],
                    "inference_timeout": 2.0,
                },
                tmp_path,
            )

        legacy = QuarryConfig.from_mapping(
            {"rules": ["test.rule"], "inference_timeout": 2.0},
            tmp_path,
        )
        assert legacy.safety_limits.inference_timeout == 2.0

    def test_version_two_model_alias_defaults_and_classify_head(self, tmp_path: Path):
        config = QuarryConfig.from_mapping(
            {
                "config_version": 2,
                "rules": ["test.rule"],
                "models": {
                    "guard": {
                        "model": "provider/guard",
                        "classify": {"class_index": 2},
                    }
                },
            },
            tmp_path,
        )

        assert config.models["guard"].adapter == "litellm"
        assert config.models["guard"].classify is not None
        assert config.models["guard"].classify.class_index == 2

        with pytest.raises(ValueError, match="non-negative integer"):
            QuarryConfig.from_mapping(
                {
                    "config_version": 2,
                    "rules": ["test.rule"],
                    "models": {
                        "guard": {
                            "model": "provider/guard",
                            "classify": {"class_index": True},
                        }
                    },
                },
                tmp_path,
            )

    def test_configuration_detaches_and_freezes_nested_containers(self):
        paths = [Path("rules")]
        models = {}
        prompts = {"policy": "original"}
        rate_limits = {}
        config = QuarryConfig(
            quarry_rules_paths=paths,
            models=models,
            judge_prompts=prompts,
            llm_rate_limits=rate_limits,
        )

        paths.append(Path("other"))
        models["late"] = object()
        prompts["policy"] = "changed"
        rate_limits["late"] = object()

        assert config.quarry_rules_paths == (Path("rules"),)
        assert dict(config.models) == {}
        assert dict(config.judge_prompts) == {"policy": "original"}
        assert dict(config.llm_rate_limits) == {}
        with pytest.raises(FrozenInstanceError):
            config.config_version = 2
        with pytest.raises(TypeError):
            config.models["late"] = object()
        with pytest.raises(FrozenInstanceError):
            config.get_safety_limits().max_model_calls = 1

    def test_inference_config_rejects_unknown_nested_field(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """\
rules:
  - test.rule
models:
  gemma4:
    adapter: litellm
    model: bedrock/gemma4
    score:
      boundary_seed: 1
      tokens:
        negative: "0"
        positive: "1"
      extraction:
        probabilites: probabilities
""",
            encoding="utf-8",
        )

        with pytest.raises(
            ConfigurationError,
            match="Inference model 'gemma4' score extraction has unknown field: 'probabilites'",
        ):
            QuarryConfig.from_yaml(config_path)

        config_path.write_text(
            """\
rules: [test.rule]
models:
  legacy:
    model: local/model
    tokenizer: local/tokenizer
    threshold: 0.5
""",
            encoding="utf-8",
        )
        with pytest.raises(ConfigurationError, match="unknown fields: 'threshold', 'tokenizer'"):
            QuarryConfig.from_yaml(config_path)

        config_path.write_text(
            """\
rules: [test.rule]
default_judge_model: openai/model
""",
            encoding="utf-8",
        )
        with pytest.raises(
            ConfigurationError,
            match="Configuration has unknown field: 'default_judge_model'",
        ):
            QuarryConfig.from_yaml(config_path)

        config_path.write_text(
            "rules: [test.rule]\nmax_nesting_depth: true\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="max_nesting_depth"):
            QuarryConfig.from_yaml(config_path)

        config_path.write_text(
            """\
rules: [test.rule]
llm_rate_limits:
  provider:
    max_concurent: 2
""",
            encoding="utf-8",
        )
        with pytest.raises(
            ConfigurationError,
            match="LLM rate limit configuration for 'provider'.*max_concurent",
        ):
            QuarryConfig.from_yaml(config_path)

        config_path.write_text(
            "rules: [test.rule]\nllm_rate_limits: provider\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="llm_rate_limits.*mapping"):
            QuarryConfig.from_yaml(config_path)


def test_from_mapping_takes_an_assembled_document(tmp_path):
    (tmp_path / "policy.md").write_text("policy text")
    config = QuarryConfig.from_mapping(
        {"rules": ["/injected/signatures"], "texts": {"p": {"file": "policy.md"}}}, tmp_path
    )
    assert config.quarry_rules_paths == (Path("/injected/signatures"),)
    assert config.judge_prompts["p"] == "policy text"


def test_from_mapping_resolves_relative_files_against_the_given_dir(tmp_path):
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    (policy_dir / "policy.md").write_text("policy text")
    config = QuarryConfig.from_mapping(
        {"rules": ["/r"], "texts": {"p": {"file": "policy.md"}}}, policy_dir
    )
    assert config.judge_prompts["p"] == "policy text"


def test_an_invalid_utf8_text_file_is_named_in_the_error(tmp_path: Path) -> None:
    policy = tmp_path / "policy.md"
    policy.write_bytes(b"policy \xff text")

    with pytest.raises(ValueError) as exc_info:
        QuarryConfig.from_mapping(
            {"texts": {"policy": {"file": "policy.md"}}}, tmp_path, require_rules=False
        )

    assert str(exc_info.value) == f"Text resource 'policy' file unreadable: {policy}"


def test_from_yaml_still_resolves_against_its_own_directory(tmp_path):
    (tmp_path / "policy.md").write_text("policy text")
    rule = tmp_path / "r.rule"
    rule.write_text("")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump({"rules": [str(rule)], "texts": {"p": {"file": "policy.md"}}})
    )
    assert QuarryConfig.from_yaml(config_file).judge_prompts["p"] == "policy text"


@pytest.mark.parametrize("document", [["a"], "text", 42, None, ("a",)])
def test_from_mapping_rejects_a_non_mapping_with_valueerror(document, tmp_path):
    with pytest.raises(ValueError, match="must be a mapping"):
        QuarryConfig.from_mapping(document, tmp_path)
