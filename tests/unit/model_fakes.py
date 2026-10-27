# Copyright 2026 Cisco Systems, Inc. and its affilitates
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

"""Fakes at the model-adapter boundary, shared by the model tests.

Each fake implements the adapter operations ``ModelInferenceBackend`` calls with their real
signatures, and records what it was sent. Tests assert on those records: they are the wire
payload a provider would receive.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend, build_constitution_messages
from quarry.scanner import DSLScanner
from quarry.types.results import Score, Span

POLICY_TEXT = "A command is unsafe when it destroys data outside the working tree."


def classification(*probabilities):
    """A classification response with one chunk carrying ``probabilities``."""

    return SimpleNamespace(chunks=[SimpleNamespace(probabilities=list(probabilities))])


class NoWaitLimiter:
    """Permit calls immediately while recording maximum physical concurrency."""

    def __init__(self) -> None:
        self.active = 0
        self.maximum = 0

    @asynccontextmanager
    async def limit(self, _model):
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            yield 0.0
        finally:
            self.active -= 1


class ModelAdapter:
    """Deterministic implementation of every version 2 adapter operation.

    ``classify_messages`` answers 0.1 on head 0 for text made only of ``a`` and 0.8 otherwise;
    head 1 is always 0.4. ``score`` is always 0.99. ``complete_message`` returns a tool call
    unless the physical model is ``text-model``. ``tokenize`` counts content characters plus one
    per offered tool.
    """

    def __init__(self) -> None:
        self.classification_calls = []
        self.classification_messages = []
        self.decoded_spans = []
        self.score_messages = []
        self.score_active = 0
        self.maximum_score_concurrency = 0
        self.completions = 0
        self.tokenizations = 0

    @property
    def classifications(self) -> int:
        return len(self.classification_calls)

    @property
    def score_calls(self) -> int:
        return len(self.score_messages)

    async def score(self, _config, messages, *, timeout=None):
        self.score_messages.append(messages)
        self.score_active += 1
        self.maximum_score_concurrency = max(self.maximum_score_concurrency, self.score_active)
        await asyncio.sleep(0)
        self.score_active -= 1
        return 0.99

    async def classify_messages(self, config, messages, _tools, *, timeout=None):
        self.classification_calls.append((config.model, timeout))
        self.classification_messages.append(messages)
        await asyncio.sleep(0)
        content = " ".join(
            str(message.get("content") or "") for message in messages if isinstance(message, dict)
        )
        head_zero = 0.1 if content and set(content) <= {"a"} else 0.8
        return classification(head_zero, 0.4)

    async def decode_spans(self, config, raw_result, messages, threshold):
        self.decoded_spans.append((config, raw_result, messages, threshold))
        content = messages[0].content
        start = content.index("alice@example.com")
        return [Span(0, start, start + len("alice@example.com"), Score(0.91))]

    async def complete_message(self, config, _messages, _tools, *, timeout=None):
        self.completions += 1
        if config.model == "text-model":
            return {"role": "assistant", "content": "safe"}
        return {
            "role": "assistant",
            "content": "blocked",
            "refusal": None,
            "tool_calls": [
                {
                    "id": "generated_call",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{}"},
                }
            ],
        }

    async def tokenize(self, config, messages, tools, *, timeout=None):
        self.tokenizations += 1
        count = sum(
            len(message.get("content") or "")
            for message in messages
            if isinstance(message.get("content"), str)
        )
        return {config.tokens.count_field: count + len(tools)}


def model_entry(
    *,
    classify: int | None = None,
    score: bool = False,
    tokens: bool = False,
    category: str | None = None,
    spans: int | None = None,
    physical_model: str = "physical-model",
) -> dict:
    """One version 2 ``models:`` entry served by the ``fake`` adapter."""

    config = {"adapter": "fake", "model": physical_model}
    if category is not None:
        config["category"] = category
    if classify is not None:
        config["classify"] = {"class_index": classify}
    if score:
        config["score"] = {"message_builder": "test", "boundary_seed": 7}
    if tokens:
        config["tokens"] = {"adapter": "fake", "endpoint": None, "extraction": {"count": "count"}}
    if spans is not None:
        config["spans"] = {"class_index": spans}
    return config


def passthrough_builder(_constitution, messages, _tools, _boundary, _score_config):
    """A version 2 score builder that sends the selected messages unchanged."""

    return messages


def v2_scanner(
    tmp_path: Path, rules: dict[str, str], models: dict, adapter, **config
) -> DSLScanner:
    """A version 2 scanner over ``rules`` (name -> condition) served by ``adapter``."""

    paths = []
    for name, condition in rules.items():
        path = tmp_path / f"{name.lower()}.rule"
        path.write_text(f'rule "{name}"\ncondition: {condition}\n', encoding="utf-8")
        paths.append(str(path))
    quarry_config = QuarryConfig.from_mapping(
        {"config_version": 2, "rules": paths, "models": models, **config}, tmp_path
    )
    backend = ModelInferenceBackend(
        quarry_config,
        adapters={"fake": adapter},
        message_builders={"test": passthrough_builder},
        rate_limiter=NoWaitLimiter(),
    )
    return DSLScanner(quarry_config, model_inference_backend=backend)


# ── version 1 configuration (llm_score / llm_input_tokens / model_scan) ─────────────────────


class FakeAdapter:
    """Version 1 adapter: scores 0.8, completes ``completed``, counts content characters."""

    def __init__(self, scores: list[float] | None = None) -> None:
        self.builder_messages = []
        self.classification_calls = []
        self.token_configs = []
        self.tools = []
        self.timeouts = []
        self._scores = list(scores or [])

    async def complete(self, config, messages, tools, *, timeout=None):
        self.tools.append(tools)
        self.timeouts.append(timeout)
        return "completed"

    async def score(self, config, messages, *, timeout=None):
        index = len(self.builder_messages)
        self.builder_messages.append(messages)
        self.timeouts.append(timeout)
        await asyncio.sleep(0)
        if self._scores:
            return self._scores[index] if index < len(self._scores) else 0.0
        return 0.8

    async def classify(self, config, text, *, timeout=None):
        self.classification_calls.append((config.model, text))
        self.timeouts.append(timeout)
        return classification(0.1, 0.9, 0.2, 0.8)

    async def tokenize(self, config, messages, tools, *, timeout=None):
        self.token_configs.append(config.tokens)
        self.tools.append(tools)
        self.timeouts.append(timeout)
        count = sum(
            len(message.get("content", ""))
            for message in messages
            if isinstance(message.get("content"), str)
        )
        return {config.tokens.count_field: count}

    def called(self) -> bool:
        """Whether any provider operation reached this adapter."""

        return bool(
            self.builder_messages or self.token_configs or self.classification_calls or self.tools
        )


V1_MODEL = [
    "models:",
    "  gemma4:",
    "    adapter: fake",
    "    model: test/gemma4",
    "    endpoint: ${MODEL_ENDPOINT}",
    "    aws_profile: ${AWS_PROFILE}",
    "    region: ${AWS_REGION}",
    "    tokens:",
    "      adapter: fake",
    "      endpoint: ${TOKEN_ENDPOINT}",
    "      extraction:",
    "        count: count",
    "    score:",
    "      message_builder: test_builder",
    "      boundary_seed: 7",
    "      tokens:",
    '        negative: "0"',
    '        positive: "1"',
    "      extraction:",
    "        probabilities: probabilities",
    "      calibration: calibration.joblib",
    "texts:",
    f'  unsafe_shell: "{POLICY_TEXT}"',
    "rule_timeout: 12.5",
]


def set_config_environment(monkeypatch) -> None:
    """Provide the environment variables the version 1 config interpolates."""

    monkeypatch.setenv("AWS_PROFILE", "test-profile")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("MODEL_ENDPOINT", "https://model.example")
    monkeypatch.setenv("TOKEN_ENDPOINT", "/tokenize")


def write_v1_config(tmp_path: Path, rules: list[Path]) -> QuarryConfig:
    """A version 1 config with the ``gemma4`` scorer/tokenizer and the ``unsafe_shell`` text."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(["rules:", *(f"  - {rule}" for rule in rules), *V1_MODEL]), encoding="utf-8"
    )
    return QuarryConfig.from_yaml(config_path)


def recording_builder(scored: list):
    """The production version 1 score builder, recording the conversation it is given."""

    def builder(constitution, conversation, boundary, score_config):
        scored.append(conversation.to_openai_messages())
        return build_constitution_messages(constitution, conversation, boundary, score_config)

    return builder


def v1_backend(config: QuarryConfig, adapter, scored: list | None = None) -> ModelInferenceBackend:
    """A backend whose calibrator adds 0.1 to every adapter score."""

    return ModelInferenceBackend(
        config,
        classification_models_to_load=["gemma4"],
        adapters={"fake": adapter},
        message_builders={"test_builder": recording_builder([] if scored is None else scored)},
        rate_limiter=NoWaitLimiter(),
        calibrator_loader=lambda _path: lambda score: score + 0.1,
    )


def v1_scanner(tmp_path: Path, condition: str, adapter, scored: list | None = None) -> DSLScanner:
    """A version 1 scanner for the single rule ``R`` with ``condition``."""

    rule = tmp_path / "r.rule"
    rule.write_text(f'rule "R"\ncondition: {condition}\n', encoding="utf-8")
    config = write_v1_config(tmp_path, [rule])
    return DSLScanner(config, model_inference_backend=v1_backend(config, adapter, scored))
