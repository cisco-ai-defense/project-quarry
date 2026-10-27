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

"""Loading, validating and freezing an Quarry configuration document."""

import dataclasses
from pathlib import Path
from types import MappingProxyType

import pytest
import yaml

from quarry.config import QuarryConfig
from quarry.types.exceptions import ConfigurationError


@pytest.mark.parametrize(
    ("config_version", "rule_timeout", "inference_timeout"),
    [pytest.param(1, 1.0, 30.0, id="version-1"), pytest.param(2, 60.0, None, id="version-2")],
)
def test_timeouts_default_by_config_version(config_version, rule_timeout, inference_timeout):
    limits = QuarryConfig(config_version=config_version).get_safety_limits()

    assert (limits.rule_timeout, limits.inference_timeout) == (rule_timeout, inference_timeout)
    assert limits.max_model_calls == 128


@pytest.mark.parametrize("config_version", [True, 1.0, 3])
def test_only_integer_config_versions_one_and_two_exist(config_version) -> None:
    with pytest.raises(ValueError, match="config_version must be 1 or 2"):
        QuarryConfig(config_version=config_version)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        pytest.param(
            {
                "models": {
                    "gemma4": {
                        "adapter": "litellm",
                        "model": "bedrock/gemma4",
                        "score": {
                            "boundary_seed": 1,
                            "tokens": {"negative": "0", "positive": "1"},
                            "extraction": {"probabilites": "probabilities"},
                        },
                    }
                }
            },
            "Inference model 'gemma4' score extraction has unknown field: 'probabilites'",
            id="nested-score-field",
        ),
        pytest.param(
            {
                "models": {
                    "legacy": {
                        "model": "local/model",
                        "tokenizer": "local/tokenizer",
                        "threshold": 0.5,
                    }
                }
            },
            "unknown fields: 'threshold', 'tokenizer'",
            id="model-fields",
        ),
        pytest.param(
            {"default_judge_model": "openai/model"},
            "Configuration has unknown field: 'default_judge_model'",
            id="top-level-field",
        ),
    ],
)
def test_an_unknown_field_is_rejected_instead_of_ignored(tmp_path: Path, document, message):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"rules": ["test.rule"], **document}), encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message):
        QuarryConfig.from_yaml(config_path)


def test_from_mapping_resolves_relative_files_against_the_given_directory(tmp_path):
    """The document need not sit beside the files it names."""

    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    (policy_dir / "policy.md").write_text("policy text")

    config = QuarryConfig.from_mapping(
        {"rules": ["/injected/signatures"], "texts": {"p": {"file": "policy.md"}}}, policy_dir
    )

    assert config.quarry_rules_paths == (Path("/injected/signatures"),)
    assert config.judge_prompts["p"] == "policy text"


def test_from_yaml_resolves_relative_files_against_its_own_directory(tmp_path):
    (tmp_path / "policy.md").write_text("policy text")
    rule = tmp_path / "r.rule"
    rule.write_text("")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump({"rules": [str(rule)], "texts": {"p": {"file": "policy.md"}}})
    )

    assert QuarryConfig.from_yaml(config_file).judge_prompts["p"] == "policy text"


def test_resources_can_be_loaded_before_inline_rules_are_supplied(tmp_path):
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "texts": {"policy": "Do not leak secrets."}},
        tmp_path,
        require_rules=False,
    )

    assert config.quarry_rules_paths == ()
    assert config.judge_prompts == {"policy": "Do not leak secrets."}


def test_configuration_is_immutable_after_construction(tmp_path):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"judge": {"model": "openai/test"}},
            "llm_rate_limits": {"default": {"max_concurrent": 1}},
            "texts": {"policy": "keep secrets"},
        },
        tmp_path,
        require_rules=False,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.models = {}
    with pytest.raises(TypeError):
        config.models["other"] = config.models["judge"]
    with pytest.raises(TypeError):
        config.judge_prompts["policy"] = "changed"
    with pytest.raises(TypeError):
        config.llm_rate_limits["default"] = config.llm_rate_limits["default"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.llm_rate_limits["default"].max_concurrent = 99
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.safety_limits.rule_timeout = 0.5


def test_changing_the_callers_mappings_after_construction_does_not_reach_the_config(tmp_path):
    models = {"judge": {"model": "openai/test"}}
    rate_limits = {"default": {"max_concurrent": 1}}
    texts = {"policy": "keep secrets"}
    document = {
        "config_version": 2,
        "models": models,
        "llm_rate_limits": rate_limits,
        "texts": texts,
    }
    config = QuarryConfig.from_mapping(document, tmp_path, require_rules=False)
    prompts = {"policy": "keep secrets"}
    direct = QuarryConfig(judge_prompts=MappingProxyType(prompts))

    models.clear()
    rate_limits.clear()
    texts.clear()
    document.clear()
    prompts["policy"] = "allow everything"

    assert list(config.models) == ["judge"]
    assert config.llm_rate_limits["default"].max_concurrent == 1
    assert config.judge_prompts == {"policy": "keep secrets"}
    assert direct.judge_prompts == {"policy": "keep secrets"}


def test_an_inference_model_defaults_to_litellm_but_never_to_an_upstream_model(tmp_path):
    def models(model: dict):
        return QuarryConfig.from_mapping(
            {"models": {"guard": model}}, tmp_path, require_rules=False
        ).models

    default = models({"model": "openai/example-model"})["guard"]
    custom = models({"adapter": "custom_classifier", "model": "local/guard"})["guard"]

    assert (default.adapter, default.model) == ("litellm", "openai/example-model")
    assert custom.adapter == "custom_classifier"
    with pytest.raises(ValueError, match="requires a non-empty 'model' string"):
        models({})
    with pytest.raises(ValueError, match="requires a non-empty 'adapter' string"):
        models({"adapter": None, "model": "openai/example-model"})


def test_an_inference_model_carries_its_category_and_span_class_index(tmp_path):
    config = QuarryConfig.from_mapping(
        {
            "models": {
                "privacy_email": {
                    "adapter": "fake",
                    "model": "privacy-model",
                    "category": "PII_ENTITY_TYPE_EMAIL",
                    "classify": {"class_index": 0},
                    "spans": {"class_index": 4},
                }
            }
        },
        tmp_path,
        require_rules=False,
    )

    model = config.models["privacy_email"]
    assert model.category == "PII_ENTITY_TYPE_EMAIL"
    assert model.spans.class_index == 4


@pytest.mark.parametrize(
    ("model", "message"),
    [
        pytest.param(
            {"model": "privacy-model", "spans": {"class_index": 4}},
            "requires a non-empty string 'category' when spans is configured",
            id="spans-without-category",
        ),
        pytest.param(
            {"model": "privacy-model", "category": "", "spans": {"class_index": 4}},
            "requires a non-empty string 'category' when spans is configured",
            id="spans-with-empty-category",
        ),
        pytest.param(
            {"model": "privacy-model", "category": "EMAIL", "spans": {"class_index": -1}},
            "non-negative integer 'class_index'",
            id="negative-class-index",
        ),
        pytest.param(
            {"model": "privacy-model", "category": "EMAIL", "spans": {"class_index": True}},
            "non-negative integer 'class_index'",
            id="boolean-class-index",
        ),
        pytest.param(
            {
                "model": "privacy-model",
                "category": "EMAIL",
                "spans": {"category": "EMAIL", "class_index": 1},
            },
            "unknown field: 'category'",
            id="category-inside-spans",
        ),
        pytest.param(
            {
                "model": "privacy-model",
                "category": "EMAIL",
                "spans": {"class_index": 1, "extra": 2},
            },
            "unknown field: 'extra'",
            id="unknown-spans-field",
        ),
    ],
)
def test_an_invalid_span_configuration_is_rejected(tmp_path, model, message):
    with pytest.raises((ValueError, ConfigurationError), match=message):
        QuarryConfig.from_mapping({"models": {"privacy": model}}, tmp_path, require_rules=False)


def test_inference_timeout_is_only_available_in_version_one(tmp_path):
    version_one = QuarryConfig.from_mapping({"rules": ["rule"], "inference_timeout": 2.0}, tmp_path)
    assert version_one.get_safety_limits().inference_timeout == 2.0

    with pytest.raises(ConfigurationError, match="only supported by config_version 1"):
        QuarryConfig.from_mapping(
            {"config_version": 2, "rules": ["rule"], "inference_timeout": 2.0}, tmp_path
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.9, "10"])
def test_the_comprehension_limit_must_be_a_positive_integer(tmp_path, value) -> None:
    with pytest.raises(ValueError, match="max_comprehension_size must be a positive integer"):
        QuarryConfig.from_mapping({"max_comprehension_size": value}, tmp_path, require_rules=False)


@pytest.mark.parametrize("document", [["a"], "text", None], ids=["list", "string", "none"])
def test_a_document_that_is_not_a_mapping_is_rejected(document, tmp_path):
    with pytest.raises(ValueError, match="must be a mapping"):
        QuarryConfig.from_mapping(document, tmp_path)


@pytest.mark.parametrize("key", ["enforce_limits", "fail_on_timeout", "allow_stub_probers"])
def test_boolean_options_reject_quoted_strings(tmp_path, key):
    with pytest.raises(ValueError, match=f"{key} must be true or false"):
        QuarryConfig.from_mapping({key: "false"}, tmp_path, require_rules=False)


def _judge(**fields) -> dict:
    """A configuration document whose one model, ``judge``, carries ``fields``."""

    return {"models": {"judge": {"model": "openai/judge", **fields}}}


@pytest.mark.parametrize(
    ("document", "message"),
    [
        pytest.param(
            {"config_version": True}, "config_version must be 1 or 2", id="boolean-version"
        ),
        pytest.param({"config_version": 3}, "config_version must be 1 or 2", id="unknown-version"),
        pytest.param(
            {"rules": []},
            "Config must specify at least one Quarry rule path in 'rules' field",
            id="no-rules",
        ),
        pytest.param(
            {"models": {"judge": "openai/judge"}},
            "Model 'judge' config must be a mapping",
            id="model-not-a-mapping",
        ),
        pytest.param(
            {"llm_rate_limits": {"openai": 5}},
            "LLM rate limit configuration for 'openai' must be a mapping",
            id="rate-limit-not-a-mapping",
        ),
        pytest.param(
            _judge(endpoint=""),
            "Inference model 'judge' 'endpoint' must be a non-empty string or null",
            id="empty-endpoint",
        ),
        pytest.param(
            _judge(model="${QUARRY_TEST_UNSET_MODEL}"),
            "Inference model 'judge' references unset environment variable "
            "'QUARRY_TEST_UNSET_MODEL'",
            id="unset-environment-reference",
        ),
        pytest.param(
            _judge(category=5),
            "Inference model 'judge' category must be a non-empty string",
            id="non-string-category",
        ),
        pytest.param(
            _judge(classify=0),
            "Inference model 'judge' classify must be a mapping",
            id="classify-not-a-mapping",
        ),
        pytest.param(
            _judge(classify={"class_index": -1}),
            "Inference model 'judge' classify requires a non-negative integer 'class_index'",
            id="negative-classify-index",
        ),
        pytest.param(
            _judge(category="EMAIL", spans=1),
            "Inference model 'judge' spans must be a mapping",
            id="spans-not-a-mapping",
        ),
        pytest.param(
            _judge(tokens="count"),
            "Inference model 'judge' tokens must be a mapping",
            id="tokens-not-a-mapping",
        ),
        pytest.param(
            _judge(tokens={"adapter": "litellm"}),
            "Inference model 'judge' tokens requires an 'extraction' mapping",
            id="tokens-without-extraction",
        ),
        pytest.param(
            _judge(score=1),
            "Inference model 'judge' score must be a mapping",
            id="score-not-a-mapping",
        ),
        pytest.param(
            _judge(score={}),
            "Inference model 'judge' score requires an integer 'boundary_seed'",
            id="score-without-boundary-seed",
        ),
        pytest.param(
            _judge(score={"boundary_seed": 1, "tokens": "01"}),
            "Inference model 'judge' score tokens must be a mapping",
            id="score-tokens-not-a-mapping",
        ),
        pytest.param(
            _judge(score={"boundary_seed": 1, "tokens": {"negative": " ", "positive": "1"}}),
            "Inference model 'judge' score positive and negative tokens must be non-empty",
            id="blank-score-token",
        ),
        pytest.param(
            _judge(score={"boundary_seed": 1, "extraction": "probabilities"}),
            "Inference model 'judge' score extraction must be a mapping",
            id="score-extraction-not-a-mapping",
        ),
        pytest.param(
            {"texts": {"policy": "a"}, "judge_prompts": {"policy": "b"}},
            "'judge_prompts' redefines names already configured in another text block: "
            "policy. Each name may be defined once",
            id="text-defined-in-two-blocks",
        ),
        pytest.param(
            {"texts": ["policy"]},
            "'texts' must be a mapping of name to the text or {file: path}",
            id="text-block-not-a-mapping",
        ),
        pytest.param(
            {"texts": {"policy": {"file": "policy.md", "path": "policy.md"}}},
            "Text resource 'policy' has unsupported keys: path. "
            "Use the text directly, or a mapping with only 'file'",
            id="text-file-entry-with-extra-key",
        ),
        pytest.param(
            {"texts": {"policy": {}}},
            "Text resource 'policy' must be the text, or a mapping with 'file'",
            id="text-mapping-without-file",
        ),
        pytest.param(
            {"texts": {"policy": {"file": "/nonexistent/quarry/policy.md"}}},
            "Text resource 'policy' file not found: /nonexistent/quarry/policy.md",
            id="absolute-text-file-missing",
        ),
        pytest.param(
            {"texts": {"policy": 5}},
            "Text resource 'policy' must be the text, or a mapping with 'file'",
            id="text-is-a-number",
        ),
        pytest.param(
            {"texts": {"leak policy": "keep secrets"}},
            "Invalid text resource name 'leak policy': names may contain only letters, digits, "
            "'_', '.' and '-', so that references to them stay unambiguous",
            id="text-name-with-a-space",
        ),
        pytest.param(
            {"judge_prompts": {"policy": "  \n"}},
            "Judge prompt 'policy' is empty",
            id="blank-judge-prompt",
        ),
    ],
)
def test_an_invalid_configuration_is_rejected_with_its_reason(
    tmp_path, monkeypatch, document, message
) -> None:
    monkeypatch.delenv("QUARRY_TEST_UNSET_MODEL", raising=False)

    with pytest.raises(ValueError) as exc_info:
        QuarryConfig.from_mapping({"rules": ["policy.rule"], **document}, tmp_path)

    assert str(exc_info.value) == message


def test_an_unreadable_text_file_is_named_in_the_error(tmp_path) -> None:
    policy = tmp_path / "policy.md"
    policy.write_text("keep secrets", encoding="utf-8")
    policy.chmod(0)

    with pytest.raises(ValueError) as exc_info:
        QuarryConfig.from_mapping(
            {"texts": {"policy": {"file": "policy.md"}}}, tmp_path, require_rules=False
        )

    assert str(exc_info.value) == f"Text resource 'policy' file unreadable: {policy}"


@pytest.mark.parametrize(
    ("limit", "parsed"),
    [
        pytest.param(None, (None, 60.0, None), id="null-means-unlimited"),
        pytest.param(
            {"requests_per_interval": 10, "interval_seconds": 5, "max_concurrent": 2},
            (10, 5.0, 2),
            id="requests-per-interval",
        ),
        pytest.param({"requests_per_minute": 30}, (30, 60.0, None), id="requests-per-minute"),
        pytest.param({"requests_per_second": 3}, (3, 1.0, None), id="requests-per-second"),
        pytest.param(
            {"requests_per_second": 3, "interval_seconds": 2},
            (3, 2.0, None),
            id="interval-overrides-the-unit",
        ),
        pytest.param(
            {"requests_per_interval": None, "interval_seconds": None, "max_concurrent": None},
            (None, 60.0, None),
            id="null-fields-keep-defaults",
        ),
    ],
)
def test_a_rate_limit_is_parsed_into_requests_interval_and_concurrency(
    tmp_path, limit, parsed
) -> None:
    config = QuarryConfig.from_mapping(
        {"llm_rate_limits": {"openai": limit}}, tmp_path, require_rules=False
    )

    rate = config.llm_rate_limits["openai"]
    assert (rate.requests_per_interval, rate.interval_seconds, rate.max_concurrent) == parsed


def test_input_size_limits_are_read_from_the_document(tmp_path) -> None:
    limits = QuarryConfig.from_mapping(
        {
            "max_messages": 7,
            "max_message_size": 100,
            "max_tool_argument_size": 200,
            "max_tool_calls": 3,
            "max_nesting_depth": 4,
            "max_total_size": 500,
            "max_timeline_items": 9,
        },
        tmp_path,
        require_rules=False,
    ).get_safety_limits()

    assert (
        limits.max_messages,
        limits.max_message_size,
        limits.max_tool_argument_size,
        limits.max_tool_calls,
        limits.max_nesting_depth,
        limits.max_total_size,
        limits.max_timeline_items,
    ) == (7, 100, 200, 3, 4, 500, 9)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param("", "Empty or invalid YAML config: {path}", id="empty-file"),
        pytest.param("- policy.rule\n", "Config must be a mapping: {path}", id="yaml-list"),
    ],
)
def test_a_yaml_file_without_a_mapping_is_rejected(tmp_path, text, message) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        QuarryConfig.from_yaml(path)

    assert str(exc_info.value) == message.format(path=path)


_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}},
    "required": ["label"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    ("config_version", "schema", "message"),
    [
        (
            2,
            {**_VERDICT_SCHEMA, "properties": {"label": {"anyOf": [{"type": "string"}]}}},
            "'anyOf' at /properties/label is not supported",
        ),
        (
            2,
            {**_VERDICT_SCHEMA, "properties": {"label": {"type": "string", "pattern": "x"}}},
            "'pattern' at /properties/label is not supported",
        ),
        (
            2,
            {key: value for key, value in _VERDICT_SCHEMA.items() if key != "additionalProperties"},
            "needs additionalProperties: false",
        ),
        (
            2,
            {**_VERDICT_SCHEMA, "properties": {"label": {"type": ["number", "null"]}}},
            "the only list type",
        ),
        (2, {**_VERDICT_SCHEMA, "required": ["label", 1]}, "must list every property"),
        (
            2,
            {
                **_VERDICT_SCHEMA,
                "properties": {"label": {"type": "number", "minimum": 2, "maximum": 1}},
            },
            "minimum at /properties/label exceeds maximum",
        ),
        (
            2,
            {
                **_VERDICT_SCHEMA,
                "properties": {
                    "label": {"type": "array", "items": {"type": "string"}, "minItems": None}
                },
            },
            "minItems at /properties/label must be a non-negative integer",
        ),
        (
            2,
            {
                **_VERDICT_SCHEMA,
                "properties": {
                    "label": {"type": "string", "enum": [f"v{index}" for index in range(1001)]}
                },
            },
            "has 1001 enum values; the limit is 1000",
        ),
        (
            2,
            {
                **_VERDICT_SCHEMA,
                "properties": {"right_after": {"type": "string"}},
                "required": ["right_after"],
            },
            "reserved by the rule grammar",
        ),
        (1, _VERDICT_SCHEMA, "only supported by config_version 2"),
    ],
    ids=[
        "union",
        "regex",
        "open-object",
        "nullable-number",
        "malformed-required",
        "unsatisfiable",
        "null-bound",
        "too-many-enum-values",
        "reserved-name",
        "version-1",
    ],
)
def test_a_response_schema_outside_the_supported_subset_is_rejected(
    tmp_path: Path, config_version, schema, message
) -> None:
    document = {
        "config_version": config_version,
        "rules": ["r.rule"],
        "models": {"verdict": {"model": "x", "complete": {"response_schema": schema}}},
    }

    with pytest.raises(ValueError, match=message):
        QuarryConfig.from_mapping(document, tmp_path)


def test_a_response_schema_is_kept_as_compact_json_in_property_order(tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {"zeta": {"type": "string"}, "alpha": {"type": ["string", "null"]}},
        "required": ["zeta", "alpha"],
        "additionalProperties": False,
    }
    document = {
        "config_version": 2,
        "rules": ["r.rule"],
        "models": {"verdict": {"model": "x", "complete": {"response_schema": schema}}},
    }

    complete = QuarryConfig.from_mapping(document, tmp_path).models["verdict"].complete

    assert complete is not None
    assert complete.response_schema == (
        '{"type":"object","properties":{"zeta":{"type":"string"},'
        '"alpha":{"type":["string","null"]}},"required":["zeta","alpha"],'
        '"additionalProperties":false}'
    )
