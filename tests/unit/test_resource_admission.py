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

"""A scanner fails at construction when a rule needs a model the configuration cannot serve.

Every model reference is found wherever it sits in the rule (condition, comprehension body,
variable, collect), and each operation the rule performs must be configured and supported by the
model's adapter. A model that is configured but unused never blocks loading.
"""

from dataclasses import replace
from pathlib import Path

import orjson
import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.scanner import DSLScanner
from quarry.types.exceptions import ConfigurationError


class _ClassifierAdapter:
    """Implements classification only, so completion and span decoding are unsupported."""

    async def classify(self, *_args, **_kwargs):
        return 0.5

    async def classify_messages(self, *_args, **_kwargs):
        return 0.5


_PLAIN = {"adapter": "litellm", "model": "provider/model"}
_UNREGISTERED = {"adapter": "unregistered", "model": "provider/unregistered"}
_CLASSIFIER = {"adapter": "fake", "model": "provider/classifier"}
_EMAIL_WITHOUT_SPANS = {"adapter": "fake", "model": "m", "classify": {"class_index": 0}}
_EMAIL_WITH_SPANS = {
    "adapter": "fake",
    "model": "m",
    "category": "EMAIL",
    "spans": {"class_index": 4},
}
_SPANS_COLLECT = 'rule "R"\ncondition: true\ncollect:\n    [s for s in model("email").spans(messages, 0.7) if true]\n'


def _scanner(tmp_path: Path, version: int, models: dict, rule: str, adapter) -> DSLScanner:
    config = QuarryConfig.from_mapping(
        {"config_version": version, "models": models}, tmp_path, require_rules=False
    )
    config = replace(config, quarry_rule_texts=(rule,))
    if adapter is None:
        return DSLScanner(config)
    backend = ModelInferenceBackend(config, adapters={"fake": adapter})
    return DSLScanner(config, model_inference_backend=backend)


@pytest.mark.parametrize(
    ("version", "models", "rule", "adapter", "error", "message"),
    [
        pytest.param(
            1,
            {},
            'rule "R"\ncondition: model_scan("toxicity", user_messages[0].content, 0.5, 1)\n',
            None,
            RuntimeError,
            "required resources unavailable",
            id="model-in-the-condition",
        ),
        pytest.param(
            1,
            {},
            'rule "R"\ncondition: any m in user_messages where model_scan("toxicity", m.content, 0.5, 1)\n',
            None,
            RuntimeError,
            "inference models not configured: toxicity",
            id="model-inside-a-comprehension",
        ),
        pytest.param(
            1,
            {},
            'rule "R"\nvariables:\n    t = model_scan("toxicity", user_messages[0].content, 0.5, 1)\n'
            "condition: t\n",
            None,
            RuntimeError,
            "inference models not configured: toxicity",
            id="model-inside-a-variable",
        ),
        pytest.param(
            2,
            {},
            _SPANS_COLLECT,
            None,
            RuntimeError,
            "inference models not configured: email. Available models: none",
            id="model-inside-collect",
        ),
        pytest.param(
            1,
            {"model": _PLAIN},
            'rule "R"\ncondition: llm_score("model", "policy", messages) > 0.5\n',
            None,
            RuntimeError,
            "score configuration for inference models: model",
            id="score-without-score-configuration",
        ),
        pytest.param(
            1,
            {"model": _PLAIN},
            'rule "R"\ncondition: llm_input_tokens("model", messages) > 5\n',
            None,
            RuntimeError,
            "input-token configuration for inference models: model",
            id="token-count-without-token-configuration",
        ),
        pytest.param(
            1,
            {"invalid": _UNREGISTERED},
            'rule "R"\ncondition: llm_complete("invalid", messages) == "x"\n',
            None,
            ConfigurationError,
            "Inference model 'invalid' uses unregistered adapter 'unregistered'",
            id="referenced-model-with-an-unregistered-adapter",
        ),
        pytest.param(
            1,
            {"classifier": _CLASSIFIER},
            'rule "R"\ncondition: llm_complete("classifier", messages) == "x"\n',
            _ClassifierAdapter(),
            ConfigurationError,
            "Inference model 'classifier' adapter 'fake' cannot complete",
            id="completion-on-a-classification-only-adapter",
        ),
        pytest.param(
            2,
            {"email": _EMAIL_WITHOUT_SPANS},
            _SPANS_COLLECT,
            _ClassifierAdapter(),
            RuntimeError,
            "spans configuration for inference models: email",
            id="spans-without-spans-configuration",
        ),
        pytest.param(
            2,
            {"email": _EMAIL_WITH_SPANS},
            _SPANS_COLLECT,
            _ClassifierAdapter(),
            ConfigurationError,
            "Inference model 'email' adapter 'fake' cannot decode spans",
            id="spans-on-an-adapter-that-cannot-decode-them",
        ),
    ],
)
def test_a_rule_needing_an_unavailable_model_operation_fails_to_load(
    tmp_path, version, models, rule, adapter, error, message
) -> None:
    with pytest.raises(error, match=message):
        _scanner(tmp_path, version, models, rule, adapter)


@pytest.mark.parametrize(
    ("models", "rule", "adapter"),
    [
        pytest.param(
            {"required": _PLAIN, "unused": _UNREGISTERED},
            'rule "R"\ncondition: llm_complete("required", messages) == "x"\n',
            None,
            id="an-unused-invalid-model-does-not-block-a-required-one",
        ),
        pytest.param(
            {"classifier": _CLASSIFIER},
            'rule "R"\ncondition: any m in user_messages where model_scan("classifier", m.content, 0.5, 1)\n',
            _ClassifierAdapter(),
            id="classification-on-a-classification-only-adapter",
        ),
    ],
)
def test_a_rule_whose_model_operations_are_served_loads(tmp_path, models, rule, adapter) -> None:
    scanner = _scanner(tmp_path, 1, models, rule, adapter)

    assert list(scanner.loaded_rules) == ["R"]


@pytest.mark.parametrize("injected", [False, True], ids=["built-in", "injected"])
@pytest.mark.parametrize(
    ("model", "environment", "message"),
    [
        pytest.param(
            {"model": "openai/gpt-4o-mini"},
            {},
            "Inference model 'judge' needs OPENAI_API_KEY in the environment",
            id="openai-without-a-key",
        ),
        pytest.param(
            {"model": "openai/gpt-4o-mini"},
            {"OPENAI_API_KEY": "set"},
            None,
            id="openai-with-a-key",
        ),
        pytest.param(
            {"model": "bedrock/model", "aws_profile": "absent"},
            {},
            "Inference model 'judge' uses AWS profile 'absent', which is not configured",
            id="bedrock-with-a-missing-profile",
        ),
        pytest.param(
            {"model": "bedrock/model", "aws_profile": "present"},
            {},
            None,
            id="bedrock-with-a-configured-profile",
        ),
        pytest.param({"model": "bedrock/model"}, {}, None, id="bedrock-without-a-profile"),
        pytest.param({"model": "hosted_vllm/model"}, {}, None, id="a-provider-without-a-key"),
        pytest.param(
            {"model": "azure/deployment", "endpoint": "https://example.openai.azure.com"},
            {},
            None,
            id="a-provider-with-other-credential-sources",
        ),
    ],
)
def test_a_litellm_model_without_its_credentials_fails_to_load(
    tmp_path, monkeypatch, injected, model, environment, message
) -> None:
    pytest.importorskip("litellm")
    if model["model"].startswith("bedrock/"):
        pytest.importorskip("botocore")
    for name in (
        "OPENAI_API_KEY",
        "AZURE_API_BASE",
        "AZURE_API_KEY",
        "AZURE_API_VERSION",
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    aws_config = tmp_path / "aws-config"
    aws_config.write_text("[profile present]\nregion = us-east-1\n", encoding="utf-8")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(aws_config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "aws-credentials"))
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    config = QuarryConfig.from_mapping(
        {"config_version": 1, "models": {"judge": {"adapter": "litellm", **model}}},
        tmp_path,
        require_rules=False,
    )
    config = replace(
        config,
        quarry_rule_texts=('rule "R"\ncondition: llm_complete("judge", messages) == "x"\n',),
    )
    backend = (
        ModelInferenceBackend(config, models_to_load=["judge"], completion_models_to_load=["judge"])
        if injected
        else None
    )

    if message is None:
        assert list(DSLScanner(config, model_inference_backend=backend).loaded_rules) == ["R"]
    else:
        with pytest.raises(ConfigurationError) as exc_info:
            DSLScanner(config, model_inference_backend=backend)
        assert str(exc_info.value) == message


# ── an injected backend must serve the configuration the rules were admitted against ──────


_CLASSIFIES = 'rule "R"\ncondition: model("m").classify(messages) >= 0.5\n'
_CLASSIFY_MODEL = {"adapter": "fake", "model": "provider/m", "classify": {"class_index": 0}}


@pytest.mark.parametrize(
    ("scanner_models", "backend_version", "backend_models", "reason"),
    [
        pytest.param(
            {"m": _CLASSIFY_MODEL},
            1,
            {"m": _CLASSIFY_MODEL},
            "Version 2 rules require a backend built from config_version: 2",
            id="backend-from-a-version-1-config",
        ),
        pytest.param(
            {"m": _CLASSIFY_MODEL},
            2,
            {"m": {**_CLASSIFY_MODEL, "model": "provider/other"}},
            "Injected backend model configuration differs for: m",
            id="backend-serving-another-physical-model",
        ),
        pytest.param(
            {"m": _CLASSIFIER},
            2,
            {"m": _CLASSIFIER},
            "Rules require classify configuration for inference models: m",
            id="classification-without-classify-configuration",
        ),
    ],
)
def test_a_version_2_rule_fails_to_load_against_a_backend_that_cannot_serve_it(
    tmp_path, scanner_models, backend_version, backend_models, reason
) -> None:
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": scanner_models}, tmp_path, require_rules=False
    )
    backend_config = QuarryConfig.from_mapping(
        {"config_version": backend_version, "models": backend_models},
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(backend_config, adapters={"fake": _ClassifierAdapter()})

    with pytest.raises(RuntimeError) as exc_info:
        DSLScanner(
            replace(config, quarry_rule_texts=(_CLASSIFIES,)), model_inference_backend=backend
        )

    assert str(exc_info.value) == (
        f"Cannot proceed - required resources unavailable:\n  - {reason}"
    )


# ── a compiled policy fingerprints exactly the resources its rule reads ───────────────────


class _EveryOperationAdapter(_ClassifierAdapter):
    async def score(self, *_args, **_kwargs):
        return 0.5

    async def complete(self, *_args, **_kwargs):
        return "x"

    async def tokenize(self, *_args, **_kwargs):
        return {"count": 1}


@pytest.fixture(scope="module")
def every_operation_runtime(tmp_path_factory):
    config = QuarryConfig.from_mapping(
        {
            "texts": {"policy": "be safe", "unused": "never read"},
            "models": {
                "m": {
                    "adapter": "fake",
                    "model": "provider/m",
                    "score": {"message_builder": "plain", "boundary_seed": 7},
                    "tokens": {"adapter": "fake", "extraction": {"count": "count"}},
                },
                "unused": {"adapter": "fake", "model": "provider/unused"},
            },
        },
        tmp_path_factory.mktemp("models"),
        require_rules=False,
    )
    backend = ModelInferenceBackend(
        config,
        adapters={"fake": _EveryOperationAdapter()},
        message_builders={"plain": lambda _policy, messages, *_args: messages},
    )
    return config, backend


@pytest.mark.parametrize(
    ("condition", "models", "texts"),
    [
        pytest.param(
            'any m in user_messages where model_scan("m", m.content, 0.5, 1)',
            ["m"],
            [],
            id="classification",
        ),
        pytest.param('llm_complete("m", messages) == "x"', ["m"], [], id="completion"),
        pytest.param('llm_input_tokens("m", messages) > 5', ["m"], [], id="token-count"),
        pytest.param(
            'llm_score("m", text_ref("policy"), messages) > 0.5',
            ["m"],
            ["policy"],
            id="score-against-a-text-resource",
        ),
        pytest.param(
            'any chunk in llm_split_conversation("m", messages, 200) where '
            'llm_score("m", text_ref("policy"), chunk) > 0.5',
            ["m"],
            ["policy"],
            id="score-over-split-chunks",
        ),
        pytest.param(
            'any chunk in llm_split_conversation("m", messages, 200) where '
            'llm_input_tokens("m", chunk) > 5',
            ["m"],
            [],
            id="token-count-over-split-chunks",
        ),
    ],
)
def test_a_compiled_policy_fingerprints_only_the_resources_its_rule_reads(
    every_operation_runtime, condition, models, texts
) -> None:
    config, backend = every_operation_runtime
    policy = compile_policy(
        f'rule "R"\ncondition: {condition}\n',
        runtime_config=config,
        model_inference_backend=backend,
    )
    if texts:
        # A text resource can be fingerprinted in memory, but the public wire format
        # rejects it because a digest of low-entropy policy text would disclose it.
        with pytest.raises(ValueError, match="text_ref resources.*cannot serialize"):
            policy.to_bytes()
        return
    encoded = policy.to_bytes()
    document = orjson.loads(encoded)
    loaded = load_compiled_policy(encoded, runtime_config=config, model_inference_backend=backend)

    assert (sorted(document["models"]), sorted(document["text_resources"])) == (models, texts)
    assert loaded.to_bytes() == encoded
