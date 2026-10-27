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

"""Configuration management."""

import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import cast

import yaml

from quarry.cache.constants import RESULT_CACHE_MAX_ENTRIES, RESULT_CACHE_TTL_SECONDS
from quarry.safety.limits import SafetyLimits
from quarry.types.exceptions import ConfigurationError
from quarry.types.rate_limit import RateLimitConfig
from quarry.types.response_schema import response_schema_text

# Text resource names are referenced from rules as text_ref("<name>"), so each name has to be
# unambiguous: no colons, no whitespace.
JUDGE_PROMPT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

DEFAULT_INFERENCE_ADAPTER = "litellm"

# A System One choice question accepts at most 255 options.
MAX_CLASSIFY_LABELS = 255

# Configuration blocks holding named text resources, each mapping a name to the text or {file: path}.
# 'texts' is the general name; 'judge_prompts' is what the same block was called first. Both load
# into one namespace, so a rule's reference resolves the same either way.
TEXT_RESOURCE_BLOCKS: tuple[tuple[str, str, str], ...] = (
    ("texts", "text resource", "the text"),
    ("judge_prompts", "judge prompt", "the prompt text"),
)

_CONFIG_FIELDS = frozenset(
    {
        "rules",
        "config_version",
        "models",
        "texts",
        "judge_prompts",
        "llm_rate_limits",
        "rule_timeout",
        "inference_timeout",
        "explore_timeout",
        "allow_stub_probers",
        "max_messages",
        "max_message_size",
        "max_tool_argument_size",
        "max_tool_calls",
        "max_nesting_depth",
        "max_total_size",
        "max_timeline_items",
        "max_comprehension_size",
        "max_model_calls",
        "model_cache",
        "enforce_limits",
        "fail_on_timeout",
        "fail_on_limit",
    }
)


@dataclass(frozen=True)
class InputTokensConfig:
    """Tokenizer adapter, endpoint/model, and count extraction for one remote model."""

    adapter: str
    endpoint: str | None
    count_field: str
    model: str | None = None


@dataclass(frozen=True)
class ScoreConfig:
    """Prompt construction plus optional adapter-specific score settings."""

    message_builder: str
    boundary_seed: int
    negative_token: str | None = None
    positive_token: str | None = None
    probabilities_field: str | None = None
    calibration: str | None = None
    # With a budget, the model reasons before it answers, ending on ``ANSWER: <token>``,
    # instead of answering with the token alone.
    reasoning_max_tokens: int | None = None


@dataclass(frozen=True)
class ClassifyConfig:
    """Select the one classifier output head exposed by a model alias."""

    class_index: int
    # A decision model has no fixed heads: it chooses among the labels it is asked about, in
    # this order, each with an optional description. A fixed-head classifier ignores both.
    instructions: str | None = None
    labels: tuple[tuple[str, str | None], ...] | None = None


@dataclass(frozen=True)
class SpanConfig:
    """Select the adapter's span output class for a model alias."""

    class_index: int


@dataclass(frozen=True)
class CompleteConfig:
    """Completion settings for a model alias."""

    # The JSON Schema a completion's content must match, as compact JSON text: a rule reads
    # the matching object by field (``.json.label``), and a reply that does not match is a
    # refused call. See ``quarry.types.response_schema`` for the supported subset.
    response_schema: str


@dataclass(frozen=True)
class InferenceModelConfig:
    """Provider settings and optional operation settings for one model."""

    adapter: str
    model: str
    endpoint: str | None
    aws_profile: str | None
    region: str | None
    tokens: InputTokensConfig | None
    score: ScoreConfig | None
    category: str | None = None
    classify: ClassifyConfig | None = None
    spans: SpanConfig | None = None
    complete: CompleteConfig | None = None


def _default_llm_rate_limits() -> dict[str, RateLimitConfig]:
    """Default LLM rate limit settings."""

    return {
        "default": RateLimitConfig(
            requests_per_interval=60,
            interval_seconds=60.0,
            max_concurrent=5,
        )
    }


def _flag(document: Mapping, key: str, default: bool) -> bool:
    """A YAML boolean option. A quoted ``"false"`` is a non-empty string, so coercing it with
    ``bool()`` would turn the option on; anything but a real boolean is rejected."""
    value = document.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false, got {value!r}")
    return value


def _frozen_tuple(value: Iterable) -> tuple:
    """Detach a caller-provided sequence into an immutable tuple, keeping one already frozen."""

    return value if isinstance(value, tuple) else tuple(value)


class _FrozenMapping(Mapping):
    """Read-only mapping over a private copy that only the configuration creates.

    A ``MappingProxyType`` cannot mark who built it: a caller's proxy over its own mutable
    dict looks the same as one built here. This type can, so an existing frozen value is
    shared by identity while anything else, a caller's proxy included, is copied."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping):
        # A read-only view over a private copy: nothing reachable from outside can change it.
        self._data = MappingProxyType(dict(data))

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(self._data)!r})"


def _frozen_mapping(value: Mapping) -> Mapping:
    """Detach a caller-provided mapping into a read-only copy, keeping one already frozen.

    Copies the entries once, so a later change to the caller's original mapping cannot reach
    into the configuration, and exposes no mutating methods."""

    return value if isinstance(value, _FrozenMapping) else _FrozenMapping(value)


@dataclass(frozen=True)
class ModelCacheConfig:
    """Bounds for the process-wide result cache."""

    ttl_seconds: float = RESULT_CACHE_TTL_SECONDS
    max_entries: int = RESULT_CACHE_MAX_ENTRIES

    def __post_init__(self) -> None:
        if (
            isinstance(self.ttl_seconds, bool)
            or not isinstance(self.ttl_seconds, int | float)
            or not isfinite(self.ttl_seconds)
            or self.ttl_seconds <= 0
        ):
            raise ConfigurationError("model_cache.ttl_seconds must be a positive number")
        if (
            isinstance(self.max_entries, bool)
            or not isinstance(self.max_entries, int)
            or self.max_entries < 1
        ):
            raise ConfigurationError("model_cache.max_entries must be a positive integer")


def _parse_model_cache(document: dict) -> ModelCacheConfig:
    data = document.get("model_cache")
    if data is None:
        return ModelCacheConfig()
    if not isinstance(data, dict):
        raise ConfigurationError("model_cache must be a mapping")
    _reject_unknown_fields(data, {"ttl_seconds", "max_entries"}, "model_cache")
    defaults = ModelCacheConfig()
    return ModelCacheConfig(
        ttl_seconds=data.get("ttl_seconds", defaults.ttl_seconds),
        max_entries=data.get("max_entries", defaults.max_entries),
    )


@dataclass(frozen=True)
class QuarryConfig:
    """Configuration for rules and integrations.

    Deeply immutable after construction: rule paths and texts are tuples, and models,
    prompts, and rate-limit definitions are read-only mappings of frozen values. One
    configuration graph is built once and safely shared by reference across parallel work.
    Per-scan variants come from ``dataclasses.replace`` changing only the rule source,
    which shares the immutable models/prompts/limits by identity. Mutable runtime services
    (the shared ``LLMRateLimiter``, scan-local scanner state) live outside this graph.
    """

    quarry_rules_paths: tuple[Path, ...] = ()
    config_version: int = 1

    llm_rate_limits: Mapping[str, RateLimitConfig] = field(default_factory=_default_llm_rate_limits)

    # Named text resources, from the 'texts' and 'judge_prompts' blocks, referenced from rules as
    # text_ref("<name>"). Keeps a long policy in one place instead of duplicating it into every rule
    # that scores against it. The attribute keeps the name used by the original configuration block.
    judge_prompts: Mapping[str, str] = field(default_factory=dict)

    safety_limits: SafetyLimits | None = None
    models: Mapping[str, InferenceModelConfig] = field(default_factory=dict)
    model_cache: ModelCacheConfig = field(default_factory=ModelCacheConfig)

    quarry_rule_texts: tuple[str, ...] = ()

    # Semantic-drift probers (topic_drift/off_topic/capability_tags/unrequested_action) fail
    # closed at load unless a real backend is registered; True opts into the dependency-free
    # keyword/lexical STUBS (local dev/testing only, never a production model).
    allow_stub_probers: bool = False

    def __post_init__(self) -> None:
        """Detach caller-provided containers into immutable form and pick timeout defaults."""

        if (
            isinstance(self.config_version, bool)
            or not isinstance(self.config_version, int)
            or self.config_version not in {1, 2}
        ):
            raise ValueError("config_version must be 1 or 2")

        # Freeze the nested containers once. Already-frozen values (as passed by
        # ``dataclasses.replace``) are kept by identity so scan variants share them.
        object.__setattr__(self, "quarry_rules_paths", _frozen_tuple(self.quarry_rules_paths))
        object.__setattr__(self, "quarry_rule_texts", _frozen_tuple(self.quarry_rule_texts))
        object.__setattr__(self, "models", _frozen_mapping(self.models))
        object.__setattr__(self, "judge_prompts", _frozen_mapping(self.judge_prompts))
        object.__setattr__(self, "llm_rate_limits", _frozen_mapping(self.llm_rate_limits))

        if self.safety_limits is None:
            object.__setattr__(
                self,
                "safety_limits",
                SafetyLimits()
                if self.config_version == 1
                else SafetyLimits(rule_timeout=60.0, inference_timeout=None),
            )

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "QuarryConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            QuarryConfig instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML is malformed or missing required fields
        """
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty or invalid YAML config: {yaml_path}")
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a mapping: {yaml_path}")
        return cls.from_mapping(data, yaml_path.parent.resolve())

    @classmethod
    def from_mapping(
        cls,
        document: dict,
        config_dir: Path,
        *,
        require_rules: bool = True,
    ) -> "QuarryConfig":
        """Build a config from an already-parsed mapping.

        Separate from from_yaml so a caller that assembles the document itself -- injecting a value,
        merging two sources -- does not have to write a temp file. config_dir resolves relative
        `file:` paths, so that temp file would also have had to sit beside the files it names.

        Args:
            document: The parsed configuration mapping
            config_dir: Directory that relative `file:` paths resolve against
            require_rules: Require at least one rule path. Runtime adapters that supply
                rule text separately may set this to ``False`` while reusing the same
                validated resource configuration.

        Returns:
            QuarryConfig instance

        Raises:
            ValueError: If the mapping is malformed or missing required fields
        """
        # A public entry point, so the type is checked here rather than left to surface as an
        # incidental TypeError or a misleading "unknown fields" list from iterating a string.
        if not isinstance(document, dict):
            raise ValueError(f"Config must be a mapping, got {type(document).__name__}")

        _reject_unknown_fields(document, set(_CONFIG_FIELDS), "Configuration")

        config_version = document.get("config_version", 1)
        if isinstance(config_version, bool) or not isinstance(config_version, int):
            raise ValueError("config_version must be 1 or 2")
        if config_version not in {1, 2}:
            raise ValueError("config_version must be 1 or 2")
        if config_version == 2 and "inference_timeout" in document:
            raise ConfigurationError(
                "Configuration field 'inference_timeout' is only supported by config_version 1; "
                "version 2 uses the rule-wide 'rule_timeout'"
            )

        # Collect and validate every value first, then build one immutable configuration. Nothing
        # is mutated on the object after construction; ``__post_init__`` detaches these containers.
        rules_paths = tuple(Path(p).expanduser() for p in document.get("rules", []))
        if require_rules and not rules_paths:
            raise ValueError("Config must specify at least one Quarry rule path in 'rules' field")

        models: dict[str, InferenceModelConfig] = {}
        if "models" in document and document["models"]:
            for model_name, model_data in document["models"].items():
                if not isinstance(model_data, dict):
                    raise ValueError(f"Model '{model_name}' config must be a mapping")
                models[model_name] = _parse_inference_model(model_name, model_data)
                if config_version == 1 and models[model_name].complete is not None:
                    raise ValueError(
                        f"Inference model '{model_name}' complete is only supported by "
                        "config_version 2"
                    )

        judge_prompts = _parse_text_resources(document, config_dir)

        allow_stub_probers = _flag(document, "allow_stub_probers", False)

        llm_rate_limits: dict[str, RateLimitConfig] = _default_llm_rate_limits()
        if "llm_rate_limits" in document and not isinstance(document["llm_rate_limits"], dict):
            raise ValueError("Configuration field 'llm_rate_limits' must be a mapping")
        if "llm_rate_limits" in document:
            for provider, limit_data in document["llm_rate_limits"].items():
                if limit_data is None:
                    llm_rate_limits[provider] = RateLimitConfig()
                    continue

                if not isinstance(limit_data, dict):
                    raise ValueError(
                        f"LLM rate limit configuration for '{provider}' must be a mapping"
                    )

                llm_rate_limits[provider] = _parse_rate_limit_config(limit_data, provider)

        limits_kwargs = {}
        if "rule_timeout" in document and document["rule_timeout"] is not None:
            limits_kwargs["rule_timeout"] = float(document["rule_timeout"])
        if "inference_timeout" in document and document["inference_timeout"] is not None:
            limits_kwargs["inference_timeout"] = float(document["inference_timeout"])
        if "explore_timeout" in document:
            limits_kwargs["explore_timeout"] = float(document["explore_timeout"])
        if "max_messages" in document:
            limits_kwargs["max_messages"] = int(document["max_messages"])
        if "max_message_size" in document:
            limits_kwargs["max_message_size"] = int(document["max_message_size"])
        if "max_tool_argument_size" in document:
            limits_kwargs["max_tool_argument_size"] = int(document["max_tool_argument_size"])
        if "max_tool_calls" in document:
            limits_kwargs["max_tool_calls"] = int(document["max_tool_calls"])
        if "max_nesting_depth" in document:
            limits_kwargs["max_nesting_depth"] = document["max_nesting_depth"]
        if "max_total_size" in document:
            limits_kwargs["max_total_size"] = int(document["max_total_size"])
        if "max_timeline_items" in document:
            limits_kwargs["max_timeline_items"] = int(document["max_timeline_items"])
        if "max_comprehension_size" in document:
            limits_kwargs["max_comprehension_size"] = document["max_comprehension_size"]
        if "max_model_calls" in document:
            limits_kwargs["max_model_calls"] = document["max_model_calls"]
        if "enforce_limits" in document:
            limits_kwargs["enforce_limits"] = _flag(document, "enforce_limits", True)
        if "fail_on_timeout" in document:
            limits_kwargs["fail_on_timeout"] = _flag(document, "fail_on_timeout", False)
        # Still an ACCEPTED key (so this raises instead of the generic "unknown field"), but it no
        # longer selects anything: breaching an input limit always fails the scan. Rejected rather
        # than ignored — silently accepting it would let a config ask for the old lossy behavior
        # and appear to get it.
        if "fail_on_limit" in document:
            raise ConfigurationError(
                "fail_on_limit was removed: breaching a safety limit always fails the scan. "
                "Conversation and Message are the canonical record of the original conversation, "
                "so an over-limit input is reported, never truncated or partially skipped. "
                "Remove the key, and raise the specific limit if the input is legitimately large."
            )

        # Build the safety limits from the version's defaults plus any overrides. With no
        # overrides, leave it None so the constructor picks the version default in one place.
        safety_limits: SafetyLimits | None = None
        if limits_kwargs:
            base_limits = (
                SafetyLimits()
                if config_version == 1
                else SafetyLimits(rule_timeout=60.0, inference_timeout=None)
            )
            safety_limits = replace(base_limits, **limits_kwargs)

        return cls(
            config_version=config_version,
            quarry_rules_paths=rules_paths,
            models=models,
            judge_prompts=judge_prompts,
            llm_rate_limits=llm_rate_limits,
            safety_limits=safety_limits,
            model_cache=_parse_model_cache(document),
            allow_stub_probers=allow_stub_probers,
        )

    def get_safety_limits(self) -> SafetyLimits:
        """Get safety limits instance."""
        assert self.safety_limits is not None
        return self.safety_limits


_ENV_REFERENCE_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _required_string(data: dict, field_name: str, owner: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} requires a non-empty '{field_name}' string")
    return value


def _optional_string(data: dict, field_name: str, owner: str) -> str | None:
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} '{field_name}' must be a non-empty string or null")
    return value


def _reject_unknown_fields(data: dict, allowed: set[str], owner: str) -> None:
    """Reject misspelled or unsupported fields in a configuration mapping."""

    unknown = sorted(str(field) for field in data if field not in allowed)
    if unknown:
        label = "field" if len(unknown) == 1 else "fields"
        raise ConfigurationError(
            f"{owner} has unknown {label}: {', '.join(repr(field) for field in unknown)}"
        )


def _resolve_environment_reference(value: str | None, owner: str) -> str | None:
    """Resolve an exact ${NAME} reference while leaving literal strings unchanged."""

    if value is None:
        return None
    match = _ENV_REFERENCE_PATTERN.fullmatch(value)
    if match is None:
        return value
    variable = match.group(1)
    resolved = os.getenv(variable)
    if not resolved:
        raise ValueError(f"{owner} references unset environment variable '{variable}'")
    return resolved


def _parse_inference_model(
    model_name: str,
    data: dict,
) -> InferenceModelConfig:
    """Parse one configured inference model."""

    owner = f"Inference model '{model_name}'"
    _reject_unknown_fields(
        data,
        {
            "adapter",
            "model",
            "endpoint",
            "aws_profile",
            "region",
            "category",
            "tokens",
            "score",
            "classify",
            "spans",
            "complete",
        },
        owner,
    )
    tokens = _parse_input_tokens(model_name, data["tokens"]) if "tokens" in data else None
    score = _parse_score_config(model_name, data["score"]) if "score" in data else None
    classify = _parse_classify_config(model_name, data["classify"]) if "classify" in data else None
    spans = _parse_span_config(model_name, data["spans"]) if "spans" in data else None
    complete = _parse_complete_config(model_name, data["complete"]) if "complete" in data else None
    category = data.get("category")
    if spans is not None and (not isinstance(category, str) or not category.strip()):
        raise ValueError(
            f"Inference model '{model_name}' requires a non-empty string 'category' "
            "when spans is configured"
        )
    if category is not None and (not isinstance(category, str) or not category.strip()):
        raise ValueError(f"Inference model '{model_name}' category must be a non-empty string")
    adapter = (
        _required_string(data, "adapter", owner) if "adapter" in data else DEFAULT_INFERENCE_ADAPTER
    )
    return InferenceModelConfig(
        adapter=adapter,
        model=cast(
            str,
            _resolve_environment_reference(_required_string(data, "model", owner), owner),
        ),
        endpoint=_resolve_environment_reference(_optional_string(data, "endpoint", owner), owner),
        aws_profile=_resolve_environment_reference(
            _optional_string(data, "aws_profile", owner), owner
        ),
        region=_resolve_environment_reference(_optional_string(data, "region", owner), owner),
        tokens=tokens,
        score=score,
        category=category,
        classify=classify,
        spans=spans,
        complete=complete,
    )


def _parse_complete_config(model_name: str, value: object) -> CompleteConfig:
    """Parse one alias's completion settings: the JSON Schema its replies must match."""

    owner = f"Inference model '{model_name}' complete"
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a mapping")
    _reject_unknown_fields(value, {"response_schema"}, owner)
    if "response_schema" not in value:
        raise ValueError(f"{owner} requires a 'response_schema'")
    try:
        return CompleteConfig(response_schema=response_schema_text(value["response_schema"]))
    except ValueError as exc:
        raise ValueError(f"{owner} response_schema {exc}") from exc


def _parse_classify_config(model_name: str, value: object) -> ClassifyConfig:
    """Parse the classifier head exposed by one logical model alias."""

    owner = f"Inference model '{model_name}' classify"
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a mapping")
    fields: dict[object, object] = dict(value)
    _reject_unknown_fields(fields, {"class_index", "instructions", "labels"}, owner)
    class_index = fields.get("class_index")
    if isinstance(class_index, bool) or not isinstance(class_index, int) or class_index < 0:
        raise ValueError(f"{owner} requires a non-negative integer 'class_index'")
    instructions = _optional_string(fields, "instructions", owner)
    labels = _parse_classify_labels(fields["labels"], owner) if "labels" in fields else None
    if labels is not None and class_index >= len(labels):
        raise ValueError(f"{owner} class_index {class_index} is outside its {len(labels)} labels")
    return ClassifyConfig(class_index=class_index, instructions=instructions, labels=labels)


def _parse_classify_labels(value: object, owner: str) -> tuple[tuple[str, str | None], ...]:
    """Parse ordered decision labels: a list of names, or a mapping of name to description."""

    if isinstance(value, list):
        entries: list[tuple[object, object]] = [(name, None) for name in value]
    elif isinstance(value, dict):
        entries = list(value.items())
    else:
        raise ValueError(f"{owner} labels must be a list of names or a mapping of descriptions")
    if not 2 <= len(entries) <= MAX_CLASSIFY_LABELS:
        raise ValueError(f"{owner} labels must name 2 to {MAX_CLASSIFY_LABELS} choices")
    labels: list[tuple[str, str | None]] = []
    for name, description in entries:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{owner} label names must be non-empty strings")
        if description is not None and not isinstance(description, str):
            raise ValueError(f"{owner} label '{name}' description must be a string")
        labels.append((name, description))
    if len({name for name, _ in labels}) != len(labels):
        raise ValueError(f"{owner} label names must be unique")
    return tuple(labels)


def _parse_span_config(model_name: str, value: object) -> SpanConfig:
    """Parse the span output class exposed by one logical alias."""

    owner = f"Inference model '{model_name}' spans"
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a mapping")
    fields: dict[object, object] = dict(value)
    _reject_unknown_fields(fields, {"class_index"}, owner)
    class_index = fields.get("class_index")
    if isinstance(class_index, bool) or not isinstance(class_index, int) or class_index < 0:
        raise ValueError(f"{owner} requires a non-negative integer 'class_index'")
    return SpanConfig(class_index=class_index)


def _parse_input_tokens(model_name: str, value: object) -> InputTokensConfig:
    """Parse the tokenizer endpoint/model and response count field for one model."""

    owner = f"Inference model '{model_name}' tokens"
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a mapping")
    fields: dict[object, object] = dict(value)
    _reject_unknown_fields(fields, {"adapter", "endpoint", "extraction", "model"}, owner)
    extraction = fields.get("extraction")
    if not isinstance(extraction, dict):
        raise ValueError(f"{owner} requires an 'extraction' mapping")
    _reject_unknown_fields(extraction, {"count"}, f"{owner} extraction")
    return InputTokensConfig(
        adapter=_required_string(fields, "adapter", owner),
        endpoint=_resolve_environment_reference(_optional_string(fields, "endpoint", owner), owner),
        count_field=_required_string(extraction, "count", f"{owner} extraction"),
        model=_resolve_environment_reference(_optional_string(fields, "model", owner), owner),
    )


def _parse_score_config(model_name: str, value: object) -> ScoreConfig:
    """Parse one model score prompt and its optional adapter settings."""

    owner = f"Inference model '{model_name}' score"
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a mapping")
    fields: dict[object, object] = dict(value)
    _reject_unknown_fields(
        fields,
        {"message_builder", "boundary_seed", "tokens", "extraction", "calibration", "reasoning"},
        owner,
    )

    boundary_seed = fields.get("boundary_seed")
    if isinstance(boundary_seed, bool) or not isinstance(boundary_seed, int):
        raise ValueError(f"{owner} requires an integer 'boundary_seed'")

    negative_token = None
    positive_token = None
    if "tokens" in fields:
        score_tokens = fields["tokens"]
        if not isinstance(score_tokens, dict):
            raise ValueError(f"{owner} tokens must be a mapping")
        _reject_unknown_fields(score_tokens, {"negative", "positive"}, f"{owner} tokens")
        negative_token = _required_string(score_tokens, "negative", f"{owner} tokens").strip()
        positive_token = _required_string(score_tokens, "positive", f"{owner} tokens").strip()
        if not negative_token or not positive_token:
            raise ValueError(f"{owner} positive and negative tokens must be non-empty")
        if negative_token == positive_token:
            raise ValueError(f"{owner} positive and negative tokens must differ")

    probabilities_field = None
    if "extraction" in fields:
        extraction = fields["extraction"]
        if not isinstance(extraction, dict):
            raise ValueError(f"{owner} extraction must be a mapping")
        _reject_unknown_fields(extraction, {"probabilities"}, f"{owner} extraction")
        probabilities_field = _optional_string(extraction, "probabilities", f"{owner} extraction")

    reasoning_max_tokens = None
    if "reasoning" in fields:
        reasoning = fields["reasoning"]
        if not isinstance(reasoning, dict):
            raise ValueError(f"{owner} reasoning must be a mapping")
        _reject_unknown_fields(reasoning, {"max_tokens"}, f"{owner} reasoning")
        reasoning_max_tokens = reasoning.get("max_tokens")
        if (
            isinstance(reasoning_max_tokens, bool)
            or not isinstance(reasoning_max_tokens, int)
            or reasoning_max_tokens < 16
        ):
            raise ValueError(f"{owner} reasoning max_tokens must be an integer of at least 16")

    return ScoreConfig(
        message_builder=_optional_string(fields, "message_builder", owner) or "constitution",
        boundary_seed=boundary_seed,
        negative_token=negative_token,
        positive_token=positive_token,
        probabilities_field=probabilities_field,
        calibration=_resolve_environment_reference(
            _optional_string(fields, "calibration", owner), owner
        ),
        reasoning_max_tokens=reasoning_max_tokens,
    )


def _parse_text_resources(data: dict, config_dir: Path) -> dict[str, str]:
    """Load every named text resource block into one namespace.

    A rule references a resource by name, not by the block it came from, so a name defined in both
    blocks is a configuration error rather than one of the two silently winning.

    Args:
        data: The whole parsed config document
        config_dir: Directory of the config file, used to resolve relative ``file`` paths

    Returns:
        Mapping of resource name to text

    Raises:
        ValueError: If a block is malformed, or the same name is defined in more than one block
    """
    resources: dict[str, str] = {}
    for block, label, noun in TEXT_RESOURCE_BLOCKS:
        if data.get(block) is None:
            continue
        parsed = _parse_judge_prompts(data[block], config_dir, block, label, noun)
        duplicates = sorted(set(parsed) & set(resources))
        if duplicates:
            raise ValueError(
                f"'{block}' redefines names already configured in another text block: "
                f"{', '.join(duplicates)}. Each name may be defined once"
            )
        resources.update(parsed)
    return resources


def _parse_judge_prompts(
    data: object,
    config_dir: Path,
    block: str = "judge_prompts",
    label: str = "judge prompt",
    noun: str = "the prompt text",
) -> dict[str, str]:
    """Parse one text resource block into a name -> text mapping.

    Each entry is either the text inline, or a mapping with a ``file`` key pointing at a file
    holding the text. Relative paths resolve against the directory containing the config file, so a
    config and its policies can be moved together.

        texts:
          unsafe_shell_v2: |
            Decide whether the command would cause ...
          pii_disclosure:
            file: policies/pii_disclosure.md

    Args:
        data: The raw value of the block from YAML
        config_dir: Directory of the config file, used to resolve relative ``file`` paths
        block: Name of the block being parsed, for error messages
        label: What one entry is called, for error messages
        noun: What the inline form holds, for error messages

    Returns:
        Mapping of resource name to text

    Raises:
        ValueError: If the block is malformed, a name is unusable, a file is missing, or an entry
            resolves to empty text
    """
    if not isinstance(data, dict):
        raise ValueError(f"'{block}' must be a mapping of name to {noun} or {{file: path}}")

    # yaml.safe_load yields untyped containers, so re-bind each one with an explicit element type
    # before indexing or joining it. Without this the narrowed dict keys type as `Never`.
    entries: dict[object, object] = dict(data)

    prompts: dict[str, str] = {}
    for name, entry in entries.items():
        if not isinstance(name, str) or not JUDGE_PROMPT_NAME_PATTERN.match(name):
            raise ValueError(
                f"Invalid {label} name {name!r}: names may contain only letters, digits, "
                f"'_', '.' and '-', so that references to them stay unambiguous"
            )

        if isinstance(entry, str):
            text = entry
        elif isinstance(entry, dict):
            fields: dict[object, object] = dict(entry)
            unknown_keys = sorted(str(key) for key in fields if key != "file")
            if unknown_keys:
                raise ValueError(
                    f"{label.capitalize()} '{name}' has unsupported keys: "
                    f"{', '.join(unknown_keys)}. "
                    f"Use {noun} directly, or a mapping with only 'file'"
                )
            if "file" not in fields:
                raise ValueError(
                    f"{label.capitalize()} '{name}' must be {noun}, or a mapping with 'file'"
                )

            prompt_path = Path(str(fields["file"])).expanduser()
            if not prompt_path.is_absolute():
                prompt_path = config_dir / prompt_path
            if not prompt_path.is_file():
                raise ValueError(f"{label.capitalize()} '{name}' file not found: {prompt_path}")

            try:
                text = prompt_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ValueError(
                    f"{label.capitalize()} '{name}' file unreadable: {prompt_path}"
                ) from exc
        else:
            raise ValueError(
                f"{label.capitalize()} '{name}' must be {noun}, or a mapping with 'file'"
            )

        # An empty policy would silently judge against nothing, so treat it as a config error.
        if not text.strip():
            raise ValueError(f"{label.capitalize()} '{name}' is empty")

        prompts[name] = text

    return prompts


def _parse_rate_limit_config(data: dict, provider: str) -> RateLimitConfig:
    """Parse a rate limit configuration entry from YAML."""

    _reject_unknown_fields(
        data,
        {
            "requests_per_interval",
            "interval_seconds",
            "requests_per_minute",
            "requests_per_second",
            "max_concurrent",
        },
        f"LLM rate limit configuration for '{provider}'",
    )

    requests_per_interval = None
    interval_seconds = 60.0
    max_concurrent = None

    if "requests_per_interval" in data:
        raw_value = data["requests_per_interval"]
        requests_per_interval = int(raw_value) if raw_value is not None else None
        if "interval_seconds" in data:
            raw_interval = data["interval_seconds"]
            interval_seconds = float(raw_interval) if raw_interval is not None else interval_seconds
    elif "requests_per_minute" in data:
        raw_value = data["requests_per_minute"]
        requests_per_interval = int(raw_value) if raw_value is not None else None
        interval_seconds = 60.0
    elif "requests_per_second" in data:
        raw_value = data["requests_per_second"]
        requests_per_interval = int(raw_value) if raw_value is not None else None
        interval_seconds = 1.0

    if "interval_seconds" in data:
        raw_interval_override = data["interval_seconds"]
        interval_seconds = (
            float(raw_interval_override) if raw_interval_override is not None else interval_seconds
        )

    if "max_concurrent" in data:
        raw_concurrency = data["max_concurrent"]
        max_concurrent = int(raw_concurrency) if raw_concurrency is not None else None

    return RateLimitConfig(
        requests_per_interval=requests_per_interval,
        interval_seconds=interval_seconds,
        max_concurrent=max_concurrent,
    )
