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
from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import cast

import yaml

from quarry.cache.constants import (
    RESULT_CACHE_MAX_BYTES,
    RESULT_CACHE_MAX_ENTRIES,
    RESULT_CACHE_TTL_SECONDS,
)
from quarry.inputs import (
    AdapterConfig,
    InputDeclaration,
    parse_adapters,
    parse_declaration,
    parse_inputs,
)
from quarry.safety.limits import SafetyLimits
from quarry.safety.local_match_work import LocalMatchWorkConfig
from quarry.types.adapter_transforms import (
    LocalTransformConfig,
    ModelTransformConfig,
    TransformConfig,
)
from quarry.types.candidates import (
    CandidateBatchIdentity,
    CandidateBatchLimits,
)
from quarry.types.candidates import (
    _digest as _candidate_digest,
)
from quarry.types.candidates import (
    _name as _candidate_name,
)
from quarry.types.exceptions import ConfigurationError
from quarry.types.external_spans import ExternalSpanLimits
from quarry.types.local_tokenizer import (
    LocalTokenizerConfig,
    LocalTokenizerLimits,
    TokenFramingMargins,
)
from quarry.types.rate_limit import RateLimitConfig
from quarry.types.response_schema import response_schema_text
from quarry.types.score_verdicts import (
    ScoreArguments,
    ScoreLabels,
    ScoreVerdictConfig,
    score_arguments,
    score_labels,
    score_path,
    validate_score_sides,
)
from quarry.types.tokenizer_width import TokenizerWidthClaim
from quarry.types.transforms import (
    MAX_TRANSFORM_DEFINITIONS,
    MAX_TRANSFORM_NAME_CHARACTERS,
    MAX_TRANSFORM_OPERATIONS,
    ContentTransform,
    TransformOperation,
)

# Text resource names are referenced from rules as text_ref("<name>"), so each name has to be
# unambiguous: no colons, no whitespace.
JUDGE_PROMPT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

DEFAULT_INFERENCE_ADAPTER = "litellm"

# A System One choice question accepts at most 255 options.
MAX_CLASSIFY_LABELS = 255

# Model descriptions are returned by the plugin reference tool, so bound each configured value.
MAX_MODEL_DESCRIPTION_LENGTH = 500

# Model templates are untrusted configuration input. These limits bound both the inheritance walk
# and the recursive merge before provider settings are parsed into immutable runtime objects.
MAX_MODEL_CONFIG_DEFINITIONS = 256
MAX_MODEL_CONFIG_INHERITANCE_DEPTH = 32
MAX_MODEL_CONFIG_NESTING_DEPTH = 16
MAX_MODEL_CONFIG_VALUES = 20_000
MAX_RESOLVED_MODEL_CONFIG_VALUES = 20_000
MAX_RESOLVED_MODEL_CONFIG_BYTES = 1_000_000

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
        "model_templates",
        "inputs",
        "adapters",
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
        "transforms",
        "enforce_limits",
        "fail_on_timeout",
        "fail_on_limit",
        "local_match_work",
    }
)

_INFERENCE_MODEL_FIELDS = frozenset(
    {
        "description",
        "adapter",
        "model",
        "revision",
        "windows",
        "endpoint",
        "aws_profile",
        "region",
        "category",
        "tokens",
        "score",
        "classify",
        "spans",
        "complete",
    }
)

_ANY_MAPPING = object()
_LIST_OR_MAPPING = object()
_SCORE_LABELS = object()
_SCORE_PATH = object()
_MODEL_TEMPLATE_SCHEMA: dict[str, object] = {
    "adapter": str,
    "model": str,
    "revision": str,
    "windows": {"unit": str, "size": int, "step": int, "scope": str, "renderer": str},
    "endpoint": str,
    "aws_profile": str,
    "region": str,
    "category": str,
    "tokens": {
        "adapter": str,
        "endpoint": str,
        "extraction": {"count": str},
        "model": str,
        "local": _ANY_MAPPING,
        "source_width": _ANY_MAPPING,
    },
    "score": {
        "max_input_tokens": int,
        "message_builder": str,
        "boundary_seed": int,
        "tokens": {"negative": _SCORE_LABELS, "positive": _SCORE_LABELS},
        "arguments": _ANY_MAPPING,
        "extraction": {
            "probabilities": str,
            "label": _SCORE_PATH,
            "confidence": _SCORE_PATH,
            "max_chars": int,
        },
        "calibration": str,
        "reasoning": {"max_tokens": int},
        "explanation": {"max_tokens": int},
    },
    "classify": {
        "max_input_tokens": int,
        "class_index": int,
        "instructions": str,
        "labels": _LIST_OR_MAPPING,
    },
    "spans": {
        "class_index": int,
        "max_input_tokens": int,
        "backend": str,
        "renderer": str,
        "limits": {
            "max_input_characters": int,
            "max_spans": int,
            "max_coverage_ranges": int,
            "max_work": int,
        },
    },
    "complete": {"response_schema": _ANY_MAPPING, "max_input_tokens": int},
}


@dataclass(frozen=True)
class InputTokensConfig:
    """Tokenizer adapter, endpoint/model, and count extraction for one remote model."""

    adapter: str
    endpoint: str | None
    count_field: str
    model: str | None = None
    local: LocalTokenizerConfig | None = field(default=None, metadata={"cache_omit_none": True})
    source_width: TokenizerWidthClaim | None = field(
        default=None, metadata={"cache_omit_none": True}
    )

    def __post_init__(self) -> None:
        if self.local is not None:
            if (
                type(self.local) is not LocalTokenizerConfig
                or self.adapter != "local_tokenizer"
                or self.endpoint is not None
                or self.model is not None
                or type(self.count_field) is not str
                or not self.count_field
            ):
                raise ValueError("local tokenization requires its native adapter and configuration")
            object.__setattr__(self, "local", replace(self.local))
        if self.source_width is not None:
            if type(self.source_width) is not TokenizerWidthClaim or self.local is None:
                raise ValueError("source width requires a native local tokenizer and width claim")
            claim = replace(self.source_width)
            if (
                claim.artifact_sha256 != self.local.artifact_sha256
                or claim.model_revision != self.local.model_revision
                or claim.tokenizer_revision != self.local.tokenizer_revision
                or claim.count_settings_sha256 != self.local.count_settings_sha256
            ):
                raise ValueError("source width claim must match the local tokenizer settings")
            object.__setattr__(self, "source_width", claim)


_MIN_EXPLANATION_TOKENS = 16
_MAX_EXPLANATION_TOKENS = 4096


def _model_revision(value: object) -> str | None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError("model revision must be a non-empty string or null")
    return value


def _input_token_limit(value: object) -> int | None:
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError("max_input_tokens must be a positive integer")
    return value


def _explanation_token_budget(value: object, reasoning: int | None, owner: str) -> int:
    """Validate explanation settings at both the mapping and direct configuration boundaries."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not _MIN_EXPLANATION_TOKENS <= value <= _MAX_EXPLANATION_TOKENS
    ):
        raise ValueError(
            f"{owner} explanation max_tokens must be an integer from "
            f"{_MIN_EXPLANATION_TOKENS} to {_MAX_EXPLANATION_TOKENS}"
        )
    if reasoning is not None:
        raise ValueError(f"{owner} explanation and reasoning are mutually exclusive")
    return value


@dataclass(frozen=True)
class ScoreConfig:
    """Prompt construction plus optional adapter-specific score settings."""

    message_builder: str
    boundary_seed: int
    negative_token: ScoreLabels | None = None
    positive_token: ScoreLabels | None = None
    probabilities_field: str | None = None
    calibration: str | None = None
    # With a budget, the model reasons before it answers, ending on ``ANSWER: <token>``,
    # instead of answering with the token alone.
    reasoning_max_tokens: int | None = None
    # A short user-facing explanation, separately opted into from private reasoning.
    explanation_max_tokens: int | None = None
    max_input_tokens: int | None = None
    arguments: ScoreArguments | None = field(default=None, metadata={"cache_omit_none": True})
    verdict: ScoreVerdictConfig | None = field(default=None, metadata={"cache_omit_none": True})

    def __post_init__(self) -> None:
        _input_token_limit(self.max_input_tokens)
        if self.reasoning_max_tokens is not None and (
            type(self.reasoning_max_tokens) is not int or self.reasoning_max_tokens < 16
        ):
            raise ValueError("score reasoning max_tokens must be an integer of at least 16")
        if self.reasoning_max_tokens is not None and self.reasoning_max_tokens > 4096:
            raise ValueError("score reasoning max_tokens must not exceed 4096")
        object.__setattr__(self, "negative_token", score_labels(self.negative_token))
        object.__setattr__(self, "positive_token", score_labels(self.positive_token))
        validate_score_sides(self.negative_token, self.positive_token)
        object.__setattr__(self, "arguments", score_arguments(self.arguments))
        if self.verdict is not None:
            if type(self.verdict) is not ScoreVerdictConfig:
                raise ValueError("score verdict must be a validated ScoreVerdictConfig")
            if self.negative_token is None or self.positive_token is None:
                raise ValueError("score JSON extraction requires positive and negative labels")
            if self.reasoning_max_tokens is not None or self.explanation_max_tokens is not None:
                raise ValueError(
                    "score JSON extraction cannot be combined with reasoning or explanation"
                )
            if self.probabilities_field is not None:
                raise ValueError("score JSON extraction cannot also extract token probabilities")
        budget = self.explanation_max_tokens or self.reasoning_max_tokens
        if budget is not None and dict(self.arguments or ()).get("max_tokens", budget) > budget:
            raise ValueError(
                "score arguments max_tokens must not exceed its reasoning or explanation budget"
            )
        if self.explanation_max_tokens is not None:
            _explanation_token_budget(
                self.explanation_max_tokens, self.reasoning_max_tokens, "score"
            )


@dataclass(frozen=True)
class ClassifyConfig:
    """Select the one classifier output head exposed by a model alias."""

    class_index: int
    # A decision model has no fixed heads: it chooses among the labels it is asked about, in
    # this order, each with an optional description. A fixed-head classifier ignores both.
    instructions: str | None = None
    labels: tuple[tuple[str, str | None], ...] | None = None
    max_input_tokens: int | None = None

    def __post_init__(self) -> None:
        _input_token_limit(self.max_input_tokens)


@dataclass(frozen=True)
class SpanConfig:
    """Select the adapter's span output class for a model alias."""

    class_index: int
    max_input_tokens: int | None = None

    def __post_init__(self) -> None:
        _input_token_limit(self.max_input_tokens)


@dataclass(frozen=True)
class ExternalSpanConfig:
    """Explicit generic external-span dispatch, separate from classifier span heads."""

    backend: str = "external"
    renderer: str = "join"
    limits: ExternalSpanLimits = field(default_factory=ExternalSpanLimits)
    max_input_tokens: int | None = None

    def __post_init__(self) -> None:
        if type(self.backend) is not str or self.backend != "external":
            raise ValueError("external spans backend must be 'external'")
        if type(self.renderer) is not str or self.renderer not in ("join", "json"):
            raise ValueError("external spans renderer must be 'join' or 'json'")
        if type(self.limits) is not ExternalSpanLimits:
            raise ValueError("external spans limits must be ExternalSpanLimits")
        object.__setattr__(self, "limits", replace(self.limits))
        _input_token_limit(self.max_input_tokens)


@dataclass(frozen=True)
class CandidateSpanConfig:
    """One reader of an explicitly registered, revision-bound candidate batch."""

    batch: str
    channel: str
    settings_digest: str
    backend: str = "candidates"
    limits: CandidateBatchLimits = field(default_factory=CandidateBatchLimits)
    max_input_tokens: int | None = None

    def __post_init__(self) -> None:
        if type(self.backend) is not str or self.backend != "candidates":
            raise ValueError("candidate spans backend must be 'candidates'")
        _candidate_name(self.batch)
        _candidate_name(self.channel)
        _candidate_digest(self.settings_digest)
        if type(self.limits) is not CandidateBatchLimits:
            raise ValueError("candidate spans limits must be CandidateBatchLimits")
        object.__setattr__(self, "limits", replace(self.limits))
        _input_token_limit(self.max_input_tokens)


@dataclass(frozen=True)
class CompleteConfig:
    """Completion settings for a model alias."""

    # The JSON Schema a completion's content must match, as compact JSON text: a rule reads
    # the matching object by field (``.json.label``), and a reply that does not match is a
    # refused call. See ``quarry.types.response_schema`` for the supported subset.
    response_schema: str
    max_input_tokens: int | None = None

    def __post_init__(self) -> None:
        _input_token_limit(self.max_input_tokens)


@dataclass(frozen=True)
class ModelWindowsConfig:
    """Generic overlapping model cuts, with no product-specific role policy."""

    unit: str
    size: int
    step: int | None = None
    scope: str = "all"
    renderer: str = "join"

    def __post_init__(self) -> None:
        if self.unit not in ("characters", "tokens"):
            raise ValueError("windows unit must be 'characters' or 'tokens'")
        if type(self.size) is not int or self.size <= 0:
            raise ValueError("windows size must be a positive integer")
        if self.step is None:
            object.__setattr__(self, "step", self.size)
        if type(self.step) is not int or not 0 < self.step <= self.size:
            raise ValueError("windows step must be a positive integer no larger than size")
        if self.scope not in ("all", "last_turn"):
            raise ValueError("windows scope must be 'all' or 'last_turn'")
        if self.renderer not in ("join", "json", "host"):
            raise ValueError("windows renderer must be 'join', 'json', or 'host'")


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
    spans: SpanConfig | ExternalSpanConfig | CandidateSpanConfig | None = None
    complete: CompleteConfig | None = None
    revision: str | None = field(default=None, metadata={"cache_omit_none": True})
    windows: ModelWindowsConfig | None = field(default=None, metadata={"cache_omit_none": True})

    def __post_init__(self) -> None:
        _model_revision(self.revision)
        if (
            type(self.tokens) is InputTokensConfig
            and self.tokens.local is not None
            and self.tokens.local.model_revision != self.revision
        ):
            raise ValueError(
                "local tokenizer model revision must match the configured model revision"
            )
        if self.windows is not None and not isinstance(self.windows, ModelWindowsConfig):
            raise ValueError("windows must be a ModelWindowsConfig or None")
        if isinstance(self.spans, ExternalSpanConfig) and self.windows is not None:
            raise ValueError("external spans cannot be combined with model windows")
        if isinstance(self.spans, CandidateSpanConfig):
            if self.windows is not None:
                raise ValueError("candidate spans cannot be combined with model windows")
            if self.revision is None:
                raise ValueError("candidate spans require an explicit provider revision")
            CandidateBatchIdentity(self.model, self.revision, self.spans.batch)

    def identity_dict(self) -> dict[str, object]:
        """Canonical model configuration, retaining pre-explanation policy identity.

        Disabled optional explanation generation has exactly the old configuration shape.
        An enabled budget remains part of policy and served-model fingerprints.
        """

        document = asdict(self)
        if self.revision is None:
            document.pop("revision")
        if self.windows is None:
            document.pop("windows")
        if self.tokens is not None:
            if self.tokens.source_width is None:
                document["tokens"].pop("source_width", None)
            if self.tokens.local is None:
                document["tokens"].pop("local", None)
            else:
                document["tokens"]["local"] = self.tokens.local.identity_dict()
        for operation in ("score", "classify", "spans", "complete"):
            section = document[operation]
            if section is not None and section.get("max_input_tokens") is None:
                section.pop("max_input_tokens", None)
        if self.score is not None and self.score.explanation_max_tokens is None:
            document["score"].pop("explanation_max_tokens")
        if self.score is not None:
            for name in ("arguments", "verdict"):
                if getattr(self.score, name) is None:
                    document["score"].pop(name)
        return document


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
    """Shared result-table bounds and per-scan reuse policy."""

    ttl_seconds: float = RESULT_CACHE_TTL_SECONDS
    max_entries: int = RESULT_CACHE_MAX_ENTRIES
    max_bytes: int = RESULT_CACHE_MAX_BYTES
    retention: str = "process"
    reusable_categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.retention) is not str or self.retention not in {"process", "scan"}:
            raise ConfigurationError("model_cache.retention must be 'process' or 'scan'")
        categories = {"model_responses", "token_counts", "local_transforms", "probes"}
        if (
            type(self.reusable_categories) not in (list, tuple)
            or len(self.reusable_categories) > len(categories)
            or any(
                type(name) is not str or name not in categories for name in self.reusable_categories
            )
            or len(set(self.reusable_categories)) != len(self.reusable_categories)
        ):
            raise ConfigurationError(
                "model_cache.reusable_categories must name unique model_responses, "
                "token_counts, local_transforms or probes categories"
            )
        object.__setattr__(self, "reusable_categories", tuple(self.reusable_categories))
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
        if (
            isinstance(self.max_bytes, bool)
            or not isinstance(self.max_bytes, int)
            or self.max_bytes < 1
        ):
            raise ConfigurationError("model_cache.max_bytes must be a positive integer")

    def retention_for(self, category: str) -> str:
        if type(category) is not str or category not in {
            "model_responses",
            "token_counts",
            "local_transforms",
            "probes",
        }:
            raise ConfigurationError("unknown model-cache result category")
        return "process" if category in self.reusable_categories else self.retention


def _parse_model_cache(document: dict) -> ModelCacheConfig:
    data = document.get("model_cache")
    if data is None:
        return ModelCacheConfig()
    if not isinstance(data, dict):
        raise ConfigurationError("model_cache must be a mapping")
    _reject_unknown_fields(
        data,
        {"ttl_seconds", "max_entries", "max_bytes", "retention", "reusable_categories"},
        "model_cache",
    )
    defaults = ModelCacheConfig()
    return ModelCacheConfig(
        ttl_seconds=data.get("ttl_seconds", defaults.ttl_seconds),
        max_entries=data.get("max_entries", defaults.max_entries),
        max_bytes=data.get("max_bytes", defaults.max_bytes),
        retention=data.get("retention", defaults.retention),
        reusable_categories=data.get("reusable_categories", defaults.reusable_categories),
    )


def _parse_content_transforms(value: object) -> dict[str, TransformConfig]:
    """Parse operations, registered local adapters, or configured model transforms."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError("Configuration field 'transforms' must be a mapping")
    if len(value) > MAX_TRANSFORM_DEFINITIONS:
        raise ConfigurationError(
            f"Configuration supports at most {MAX_TRANSFORM_DEFINITIONS} content transforms"
        )
    transforms: dict[str, TransformConfig] = {}
    for name, definition in value.items():
        if (
            not isinstance(name, str)
            or len(name) > MAX_TRANSFORM_NAME_CHARACTERS
            or not JUDGE_PROMPT_NAME_PATTERN.fullmatch(name)
        ):
            raise ConfigurationError(
                f"Invalid transform name {name!r}: use only letters, digits, '_', '.' and '-'"
            )
        owner = f"Transform {name!r}"
        if not isinstance(definition, dict):
            raise ConfigurationError(f"{owner} must be a mapping")
        if "adapter" in definition or "model" in definition:
            try:
                if "adapter" in definition:
                    _reject_unknown_fields(definition, {"adapter", "settings", "parts"}, owner)
                    transforms[name] = LocalTransformConfig(
                        adapter=definition["adapter"],
                        settings=definition.get("settings", {}),
                        parts=definition.get("parts"),
                    )
                else:
                    _reject_unknown_fields(definition, {"model", "parts"}, owner)
                    transforms[name] = ModelTransformConfig(
                        model=definition["model"], parts=definition.get("parts")
                    )
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"{owner}: {exc}") from exc
            continue
        _reject_unknown_fields(definition, {"operations"}, owner)
        operations = definition.get("operations")
        if not isinstance(operations, list) or not operations:
            raise ConfigurationError(f"{owner} operations must be a non-empty list")
        if len(operations) > MAX_TRANSFORM_OPERATIONS:
            raise ConfigurationError(
                f"{owner} supports at most {MAX_TRANSFORM_OPERATIONS} operations"
            )
        parsed = []
        for index, operation in enumerate(operations):
            operation_owner = f"{owner} operation {index}"
            if not isinstance(operation, dict):
                raise ConfigurationError(f"{operation_owner} must be a mapping")
            kind = operation.get("op")
            try:
                if isinstance(kind, str) and kind in {"strip", "casefold"}:
                    _reject_unknown_fields(operation, {"op"}, operation_owner)
                    parsed.append(TransformOperation(kind))
                elif isinstance(kind, str) and kind in {"prefix", "suffix"}:
                    _reject_unknown_fields(operation, {"op", "text"}, operation_owner)
                    text = operation.get("text")
                    if not isinstance(text, str):
                        raise ConfigurationError(f"{operation_owner} text must be a string")
                    parsed.append(TransformOperation(kind, text))
                elif kind == "replace":
                    _reject_unknown_fields(operation, {"op", "old", "new"}, operation_owner)
                    old, new = operation.get("old"), operation.get("new")
                    if not isinstance(old, str) or not isinstance(new, str):
                        raise ConfigurationError(f"{operation_owner} old and new must be strings")
                    parsed.append(TransformOperation("replace", old, new))
                else:
                    raise ConfigurationError(
                        f"{operation_owner} op must be strip, casefold, replace, prefix, or suffix"
                    )
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"{operation_owner} is invalid: {exc}") from exc
        try:
            transforms[name] = ContentTransform(tuple(parsed))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"{owner} is invalid: {exc}") from exc
    return transforms


@dataclass(frozen=True)
class QuarryConfig:
    """Configuration for rules and integrations.

    Deeply immutable after construction: rule paths and texts are tuples, and models,
    descriptions, prompts, and rate-limit definitions are read-only mappings of frozen values.
    One configuration graph is built once and safely shared by reference across parallel work.
    Per-scan variants come from ``dataclasses.replace`` changing only the rule source, which
    shares the immutable models/descriptions/prompts/limits by identity. Mutable runtime
    services (the shared ``LLMRateLimiter``, scan-local scanner state) live outside this graph.
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
    # Human-facing purpose of each alias. Kept outside ``models`` so rewording does not change
    # provider requests, cache keys, or a compiled policy's model digest.
    model_descriptions: Mapping[str, str] = field(default_factory=dict)
    inputs: Mapping[str, InputDeclaration] = field(default_factory=dict)
    adapters: Mapping[str, AdapterConfig] = field(default_factory=dict)
    model_cache: ModelCacheConfig = field(default_factory=ModelCacheConfig)

    quarry_rule_texts: tuple[str, ...] = ()

    # Semantic-drift probers (topic_drift/off_topic/capability_tags/unrequested_action) fail
    # closed at load unless a real backend is registered; True opts into the dependency-free
    # keyword/lexical STUBS (local dev/testing only, never a production model).
    allow_stub_probers: bool = False
    # Named reviewed transform definitions; adapters are supplied by the trusted host.
    transforms: Mapping[str, TransformConfig] = field(default_factory=dict)
    local_match_work: LocalMatchWorkConfig | None = None

    def __post_init__(self) -> None:
        """Detach caller-provided containers into immutable form and pick timeout defaults."""

        if self.local_match_work is not None:
            if type(self.local_match_work) is not LocalMatchWorkConfig:
                raise ConfigurationError("local_match_work must be a native configuration value")
            object.__setattr__(
                self, "local_match_work", LocalMatchWorkConfig(self.local_match_work.groups)
            )

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
        object.__setattr__(self, "model_descriptions", _frozen_mapping(self.model_descriptions))
        # Validate here, not only while parsing a mapping: hosts may construct or replace the
        # config directly, and descriptions are included in plugin reference responses.
        for model_name, description in self.model_descriptions.items():
            if not isinstance(description, str) or not description:
                raise ConfigurationError(
                    f"Inference model '{model_name}' description must be a non-empty string"
                )
            if len(description) > MAX_MODEL_DESCRIPTION_LENGTH:
                raise ConfigurationError(
                    f"Inference model '{model_name}' description must be at most "
                    f"{MAX_MODEL_DESCRIPTION_LENGTH} characters"
                )
        for section, declaration in self.inputs.items():
            if isinstance(declaration, InputDeclaration) and declaration.section != section:
                raise ConfigurationError(
                    f"inputs {section!r} holds the declaration of {declaration.section!r}"
                )
        object.__setattr__(
            self,
            "inputs",
            _frozen_mapping(
                {
                    section: parse_declaration(
                        section,
                        {
                            key: value
                            for key, value in declaration.describe().items()
                            if key != "section"
                        },
                    )
                    if isinstance(declaration, InputDeclaration)
                    else parse_declaration(section, declaration)
                    for section, declaration in self.inputs.items()
                }
            ),
        )
        for key, entry in self.adapters.items():
            if isinstance(entry, AdapterConfig) and entry.key != key:
                raise ConfigurationError(f"adapters {key!r} holds the entry of {entry.key!r}")
        object.__setattr__(
            self,
            "adapters",
            _frozen_mapping(
                parse_adapters(
                    {
                        key: {"adapter": entry.adapter}
                        if isinstance(entry, AdapterConfig)
                        else entry
                        for key, entry in self.adapters.items()
                    }
                )
            ),
        )
        object.__setattr__(self, "judge_prompts", _frozen_mapping(self.judge_prompts))
        object.__setattr__(self, "llm_rate_limits", _frozen_mapping(self.llm_rate_limits))
        if len(self.transforms) > MAX_TRANSFORM_DEFINITIONS:
            raise ConfigurationError(
                f"Configuration supports at most {MAX_TRANSFORM_DEFINITIONS} content transforms"
            )
        checked_transforms = {}
        for name, transform in self.transforms.items():
            if (
                not isinstance(name, str)
                or len(name) > MAX_TRANSFORM_NAME_CHARACTERS
                or not JUDGE_PROMPT_NAME_PATTERN.fullmatch(name)
            ):
                raise ConfigurationError(
                    f"Invalid transform name {name!r}: use only letters, digits, '_', '.' and '-'"
                )
            if type(transform) not in {
                ContentTransform,
                LocalTransformConfig,
                ModelTransformConfig,
            }:
                raise ConfigurationError(
                    f"Transform {name!r} must be a ContentTransform immutable value "
                    "or a LocalTransformConfig or ModelTransformConfig value"
                )
            if type(transform) is ModelTransformConfig and transform.model not in self.models:
                raise ConfigurationError(
                    f"Transform {name!r} references unconfigured model {transform.model!r}"
                )
            checked_transforms[name] = transform
        object.__setattr__(self, "transforms", _frozen_mapping(checked_transforms))

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
        model_descriptions: dict[str, str] = {}
        resolved_models, environment_resolved = _resolve_model_definitions(document)
        for model_name, model_data in resolved_models.items():
            models[model_name] = _parse_inference_model(
                model_name,
                model_data,
                resolve_environment=not environment_resolved,
                config_dir=config_dir,
            )
            description = model_data.get("description")
            if description is not None:
                model_descriptions[model_name] = description
            if config_version == 1 and models[model_name].complete is not None:
                raise ValueError(
                    f"Inference model '{model_name}' complete is only supported by config_version 2"
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
            model_descriptions=model_descriptions,
            inputs=parse_inputs(document.get("inputs")),
            adapters=parse_adapters(document.get("adapters")),
            judge_prompts=judge_prompts,
            llm_rate_limits=llm_rate_limits,
            safety_limits=safety_limits,
            model_cache=_parse_model_cache(document),
            allow_stub_probers=allow_stub_probers,
            transforms=_parse_content_transforms(document.get("transforms")),
            local_match_work=(
                LocalMatchWorkConfig.from_mapping(document["local_match_work"])
                if "local_match_work" in document
                else None
            ),
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


def _resolve_environment_reference(
    value: str | None,
    owner: str,
    *,
    cache: dict[str, str | None] | None = None,
) -> str | None:
    """Resolve an exact ${NAME} reference while leaving literal strings unchanged."""

    if value is None:
        return None
    match = _ENV_REFERENCE_PATTERN.fullmatch(value)
    if match is None:
        return value
    variable = match.group(1)
    if cache is not None and variable in cache:
        resolved = cache[variable]
    else:
        resolved = os.getenv(variable)
        if cache is not None:
            cache[variable] = resolved
    if not resolved:
        raise ValueError(f"{owner} references unset environment variable '{variable}'")
    return resolved


def _configured_environment_value(
    value: str | None,
    owner: str,
    *,
    resolve_environment: bool,
) -> str | None:
    """Resolve at the direct-config boundary, or retain a value resolved during inheritance."""

    return _resolve_environment_reference(value, owner) if resolve_environment else value


def _named_model_block(document: dict, field_name: str) -> dict[str, dict]:
    """Validate one top-level block of named model definitions."""

    if field_name not in document:
        return {}
    block = document[field_name]
    if not isinstance(block, dict):
        raise ConfigurationError(f"Configuration field '{field_name}' must be a mapping")

    definitions: dict[str, dict] = {}
    singular = "Model template" if field_name == "model_templates" else "Model"
    for name, value in block.items():
        if not isinstance(name, str) or not name:
            raise ConfigurationError(f"{singular} names must be non-empty strings")
        if not isinstance(value, dict):
            error = f"{singular} {name!r} config must be a mapping"
            if field_name == "models":
                raise ValueError(error)
            raise ConfigurationError(error)
        definitions[name] = value
    return definitions


def _validate_model_config_bounds(definitions: Mapping[str, dict]) -> None:
    """Bound nested containers before inheritance recursively merges them."""

    value_count = 0
    active: set[int] = set()

    def visit(value: object, depth: int, owner: str) -> None:
        nonlocal value_count
        if not isinstance(value, dict | list | tuple):
            return
        if depth > MAX_MODEL_CONFIG_NESTING_DEPTH:
            raise ConfigurationError(
                f"{owner} exceeds the model configuration nesting limit of "
                f"{MAX_MODEL_CONFIG_NESTING_DEPTH}"
            )
        identity = id(value)
        if identity in active:
            raise ConfigurationError(f"{owner} contains a recursive container")
        active.add(identity)
        try:
            children = value.values() if isinstance(value, dict) else value
            for child in children:
                value_count += 1
                if value_count > MAX_MODEL_CONFIG_VALUES:
                    raise ConfigurationError(
                        "Model configuration exceeds the aggregate limit of "
                        f"{MAX_MODEL_CONFIG_VALUES} values"
                    )
                visit(child, depth + 1, owner)
        finally:
            active.remove(identity)

    for name, definition in definitions.items():
        visit(definition, 1, f"Model or template {name!r}")


def _validate_template_value(value: object, schema: object, owner: str) -> None:
    """Validate a supplied template value while allowing null to remove inheritance."""

    if value is None:
        return
    if schema is _SCORE_LABELS or schema is _SCORE_PATH:
        try:
            (score_labels if schema is _SCORE_LABELS else score_path)(value)
        except ValueError as exc:
            raise ConfigurationError(f"{owner}: {exc}") from exc
        return
    if schema is _ANY_MAPPING:
        if not isinstance(value, dict):
            raise ConfigurationError(f"{owner} must be a mapping or null")
        return
    if schema is _LIST_OR_MAPPING:
        if not isinstance(value, list | dict):
            raise ConfigurationError(f"{owner} must be a list or mapping, or null")
        entries = ((label, None) for label in value) if isinstance(value, list) else value.items()
        if any(
            not isinstance(label, str)
            or not label.strip()
            or (description is not None and not isinstance(description, str))
            for label, description in entries
        ):
            raise ConfigurationError(
                f"{owner} must contain non-empty names with string or null descriptions"
            )
        return
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise ConfigurationError(f"{owner} must be a mapping or null")
        _reject_unknown_fields(value, set(schema), owner)
        for field_name, field_value in value.items():
            _validate_template_value(
                field_value,
                schema[field_name],
                f"{owner} {field_name}",
            )
        return
    if schema is str and (not isinstance(value, str) or not value):
        raise ConfigurationError(f"{owner} must be a non-empty string or null")
    if schema is int and (isinstance(value, bool) or not isinstance(value, int)):
        raise ConfigurationError(f"{owner} must be an integer or null")


def _validate_model_template_fragment_semantics(owner: str, data: dict) -> None:
    """Validate supplied fragment values without requiring omitted model settings."""

    category = data.get("category")
    if isinstance(category, str) and not category.strip():
        raise ConfigurationError(f"{owner} category must be a non-empty string or null")

    spans = data.get("spans")
    if isinstance(spans, dict):
        class_index = spans.get("class_index")
        if class_index is not None and class_index < 0:
            raise ConfigurationError(
                f"{owner} spans class_index must be a non-negative integer or null"
            )
        backend = spans.get("backend")
        renderer = spans.get("renderer")
        if backend is not None and backend != "external":
            raise ConfigurationError(f"{owner} spans backend must be 'external' or null")
        if renderer is not None and renderer not in ("join", "json"):
            raise ConfigurationError(f"{owner} spans renderer must be 'join', 'json', or null")
        if isinstance(spans.get("limits"), dict):
            try:
                ExternalSpanLimits(**{k: v for k, v in spans["limits"].items() if v is not None})
            except ValueError as exc:
                raise ConfigurationError(f"{owner} spans has invalid external limits") from exc

    classify = data.get("classify")
    if isinstance(classify, dict):
        class_index = classify.get("class_index")
        if class_index is not None and class_index < 0:
            raise ConfigurationError(
                f"{owner} classify class_index must be a non-negative integer or null"
            )
        labels = classify.get("labels")
        if isinstance(labels, list):
            try:
                parsed_labels = _parse_classify_labels(labels, f"{owner} classify")
            except ValueError as exc:
                raise ConfigurationError(str(exc)) from exc
            if class_index is not None and class_index >= len(parsed_labels):
                raise ConfigurationError(
                    f"{owner} classify class_index {class_index} is outside its "
                    f"{len(parsed_labels)} labels"
                )
        elif isinstance(labels, dict) and len(labels) > MAX_CLASSIFY_LABELS:
            raise ConfigurationError(
                f"{owner} classify labels may name at most {MAX_CLASSIFY_LABELS} choices"
            )

    score = data.get("score")
    if isinstance(score, dict):
        arguments = score.get("arguments")
        if isinstance(arguments, dict):
            try:
                score_arguments(
                    {key: value for key, value in arguments.items() if value is not None}
                )
            except ValueError as exc:
                raise ConfigurationError(f"{owner} score: {exc}") from exc
        score_tokens = score.get("tokens")
        if isinstance(score_tokens, dict):
            negative = score_tokens.get("negative")
            positive = score_tokens.get("positive")
            try:
                validate_score_sides(score_labels(negative), score_labels(positive))
            except ValueError as exc:
                raise ConfigurationError(f"{owner} score: {exc}") from exc

        reasoning = score.get("reasoning")
        reasoning_max_tokens = reasoning.get("max_tokens") if isinstance(reasoning, dict) else None
        if reasoning_max_tokens is not None and (
            type(reasoning_max_tokens) is not int or reasoning_max_tokens < _MIN_EXPLANATION_TOKENS
        ):
            raise ConfigurationError(
                f"{owner} score reasoning max_tokens must be an integer of at least "
                f"{_MIN_EXPLANATION_TOKENS} or null"
            )
        if reasoning_max_tokens is not None and reasoning_max_tokens > _MAX_EXPLANATION_TOKENS:
            raise ConfigurationError(
                f"{owner} score reasoning max_tokens must not exceed {_MAX_EXPLANATION_TOKENS}"
            )

        explanation = score.get("explanation")
        explanation_max_tokens = (
            explanation.get("max_tokens") if isinstance(explanation, dict) else None
        )
        if explanation_max_tokens is not None:
            try:
                _explanation_token_budget(
                    explanation_max_tokens,
                    reasoning_max_tokens,
                    f"{owner} score",
                )
            except ValueError as exc:
                raise ConfigurationError(str(exc)) from exc


def _validate_model_template(name: str, data: dict) -> None:
    """Validate a partial model template without requiring runtime-only fields."""

    owner = f"Model template {name!r}"
    _reject_unknown_fields(data, set(_MODEL_TEMPLATE_SCHEMA) | {"extends"}, owner)
    if "extends" in data:
        extends = data["extends"]
        if not isinstance(extends, str) or not extends.strip():
            raise ConfigurationError(f"{owner} extends must be a non-empty string")
    for field_name, value in data.items():
        if field_name != "extends":
            _validate_template_value(
                value, _MODEL_TEMPLATE_SCHEMA[field_name], f"{owner} {field_name}"
            )
    _validate_model_template_fragment_semantics(owner, data)


def _expand_model_environment_references(
    name: str,
    data: dict,
    *,
    template: bool,
    environment_cache: dict[str, str | None],
) -> dict:
    """Resolve each environment-backed field in one definition exactly once."""

    owner = f"Model template {name!r}" if template else f"Inference model {name!r}"

    def expand_fields(fields: dict, names: tuple[str, ...], field_owner: str) -> dict:
        expanded = dict(fields)
        for field_name in names:
            value = fields.get(field_name)
            if not isinstance(value, str):
                continue
            try:
                expanded[field_name] = _resolve_environment_reference(
                    value,
                    field_owner,
                    cache=environment_cache,
                )
            except ValueError as exc:
                raise ConfigurationError(str(exc)) from exc
        return expanded

    expanded = expand_fields(
        data,
        ("model", "endpoint", "aws_profile", "region"),
        owner,
    )
    tokens = data.get("tokens")
    if isinstance(tokens, dict):
        expanded["tokens"] = expand_fields(
            tokens,
            ("endpoint", "model"),
            f"{owner} tokens",
        )
    score = data.get("score")
    if isinstance(score, dict):
        expanded["score"] = expand_fields(
            score,
            ("calibration",),
            f"{owner} score",
        )
    return expanded


def _merge_model_definition(parent: dict, child: dict, depth: int = 1) -> dict:
    """Recursively merge one validated definition, with null removing an inherited field."""

    if depth > MAX_MODEL_CONFIG_NESTING_DEPTH:
        raise ConfigurationError(
            "Resolved model configuration exceeds the nesting limit of "
            f"{MAX_MODEL_CONFIG_NESTING_DEPTH}"
        )
    merged = dict(parent)
    for key, value in child.items():
        if depth == 1 and key == "extends":
            continue
        if value is None:
            if key in merged:
                merged.pop(key)
            else:
                # Preserve validation behavior for a null field that did not override anything.
                # Null is inheritance removal syntax, not a way to bypass a required field.
                merged[key] = None
            continue
        inherited = merged.get(key)
        if isinstance(inherited, dict) and isinstance(value, dict):
            merged[key] = _merge_model_definition(inherited, value, depth + 1)
        else:
            merged[key] = value
    return merged


def _validate_resolved_model_semantics(name: str, data: dict, *, template: bool) -> None:
    """Fully validate each configured section after inherited fragments have merged."""

    owner = f"Model template {name!r}" if template else f"Inference model {name!r}"
    try:
        _model_revision(data.get("revision"))
        if data.get("windows") is not None:
            _parse_model_windows(data["windows"], owner=f"{owner} windows")
        if data.get("tokens") is not None:
            _parse_input_tokens(
                name,
                data["tokens"],
                owner=f"{owner} tokens",
                resolve_environment=False,
            )
        if data.get("score") is not None:
            _parse_score_config(
                name,
                data["score"],
                owner=f"{owner} score",
                resolve_environment=False,
            )
        if data.get("classify") is not None:
            _parse_classify_config(name, data["classify"], owner=f"{owner} classify")
        spans = (
            _parse_span_config(name, data["spans"], owner=f"{owner} spans")
            if data.get("spans") is not None
            else None
        )
        if isinstance(spans, ExternalSpanConfig) and data.get("windows") is not None:
            raise ValueError("external spans cannot be combined with model windows")
        if isinstance(spans, CandidateSpanConfig) and data.get("windows") is not None:
            raise ValueError("candidate spans cannot be combined with model windows")
        if data.get("complete") is not None:
            _parse_complete_config(name, data["complete"], owner=f"{owner} complete")
    except ConfigurationError:
        raise
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc

    category = data.get("category")
    if spans is not None and (not isinstance(category, str) or not category.strip()):
        raise ConfigurationError(
            f"{owner} requires a non-empty string 'category' when spans is configured"
        )
    if category is not None and (not isinstance(category, str) or not category.strip()):
        raise ConfigurationError(f"{owner} category must be a non-empty string")


def _config_scalar_bytes(value: object) -> int:
    """A deterministic work-size estimate for one scalar in a resolved definition."""

    if isinstance(value, str):
        return len(value.encode("utf-8", errors="surrogatepass"))
    if value is None:
        return 4
    if isinstance(value, bool):
        return 4 if value else 5
    if isinstance(value, int):
        # Estimate decimal digits without constructing an attacker-sized decimal string.
        digits = max(1, (abs(value).bit_length() * 30_103) // 100_000 + 1)
        return digits + (1 if value < 0 else 0)
    if isinstance(value, float):
        return 24
    return 16


def _charge_resolved_model_output(
    value: object,
    remaining_values: int,
    remaining_bytes: int,
) -> tuple[int, int]:
    """Charge logical values and scalar/key bytes before retaining a resolved output."""

    value_count = 0
    byte_count = 0
    stack = [value]
    while stack:
        current = stack.pop()
        if not isinstance(current, dict | list | tuple):
            continue
        if isinstance(current, dict):
            byte_count += sum(_config_scalar_bytes(key) for key in current)
            children = current.values()
        else:
            children = current
        for child in children:
            value_count += 1
            if value_count > remaining_values:
                raise ConfigurationError(
                    "Resolved model configurations exceed the aggregate limit of "
                    f"{MAX_RESOLVED_MODEL_CONFIG_VALUES} values"
                )
            if isinstance(child, dict | list | tuple):
                stack.append(child)
            else:
                byte_count += _config_scalar_bytes(child)
            if byte_count > remaining_bytes:
                raise ConfigurationError(
                    "Resolved model configurations exceed the aggregate limit of "
                    f"{MAX_RESOLVED_MODEL_CONFIG_BYTES} bytes"
                )
    return value_count, byte_count


def _resolve_model_definitions(document: dict) -> tuple[dict[str, dict], bool]:
    """Resolve model/template inheritance without mutating the caller's document."""

    raw_models = document.get("models")
    uses_inheritance = "model_templates" in document or (
        isinstance(raw_models, dict)
        and any(isinstance(value, dict) and "extends" in value for value in raw_models.values())
    )
    if not uses_inheritance:
        # Preserve the original path, diagnostics, and accepted limits for configurations that do
        # not opt into inheritance.
        if not raw_models:
            return {}, False
        if not isinstance(raw_models, dict):
            raise ConfigurationError("Configuration field 'models' must be a mapping")
        for name, value in raw_models.items():
            if not isinstance(value, dict):
                raise ValueError(f"Model '{name}' config must be a mapping")
        return dict(raw_models), False

    templates = _named_model_block(document, "model_templates")
    models = _named_model_block(document, "models")
    if len(templates) + len(models) > MAX_MODEL_CONFIG_DEFINITIONS:
        raise ConfigurationError(
            "Configuration defines more than "
            f"{MAX_MODEL_CONFIG_DEFINITIONS} models and model templates"
        )

    duplicates = sorted(set(templates) & set(models))
    if duplicates:
        raise ConfigurationError(
            "Model and model template names must be unique; defined in both blocks: "
            + ", ".join(repr(name) for name in duplicates)
        )

    definitions = {**templates, **models}
    _validate_model_config_bounds(definitions)
    for name, data in templates.items():
        _validate_model_template(name, data)
    for name, data in models.items():
        _reject_unknown_fields(
            data,
            set(_INFERENCE_MODEL_FIELDS) | {"extends"},
            f"Inference model {name!r}",
        )
        if "extends" in data and (
            not isinstance(data["extends"], str) or not data["extends"].strip()
        ):
            raise ConfigurationError(f"Inference model {name!r} extends must be a non-empty string")

    resolved: dict[str, dict] = {}
    resolved_depths: dict[str, int] = {}
    resolved_value_count = 0
    resolved_byte_count = 0
    environment_cache: dict[str, str | None] = {}
    resolving: list[str] = []

    def resolve(name: str) -> dict:
        nonlocal resolved_byte_count, resolved_value_count
        cached = resolved.get(name)
        if cached is not None:
            return cached
        if name in resolving:
            cycle = resolving[resolving.index(name) :] + [name]
            raise ConfigurationError(
                f"Model or template {name!r} is part of an extends cycle: "
                + " -> ".join(repr(item) for item in cycle)
            )
        resolving.append(name)
        try:
            data = definitions[name]
            parent_name = data.get("extends")
            parent: dict = {}
            ancestry_depth = 1
            if parent_name is not None:
                if parent_name not in definitions:
                    raise ConfigurationError(
                        f"Model or template {name!r} extends unknown model or template "
                        f"{parent_name!r}"
                    )
                resolved_parent = resolve(parent_name)
                ancestry_depth = resolved_depths[parent_name] + 1
                if ancestry_depth > MAX_MODEL_CONFIG_INHERITANCE_DEPTH:
                    raise ConfigurationError(
                        f"Model or template {name!r} exceeds the inheritance limit of "
                        f"{MAX_MODEL_CONFIG_INHERITANCE_DEPTH}"
                    )
                # Alias metadata describes one name and is intentionally never inherited.
                parent = {
                    key: value for key, value in resolved_parent.items() if key != "description"
                }
            expanded_data = _expand_model_environment_references(
                name,
                data,
                template=name in templates,
                environment_cache=environment_cache,
            )
            result = _merge_model_definition(parent, expanded_data)
            value_charge, byte_charge = _charge_resolved_model_output(
                result,
                MAX_RESOLVED_MODEL_CONFIG_VALUES - resolved_value_count,
                MAX_RESOLVED_MODEL_CONFIG_BYTES - resolved_byte_count,
            )
            _validate_resolved_model_semantics(name, result, template=name in templates)
            resolved_value_count += value_charge
            resolved_byte_count += byte_charge
            resolved[name] = result
            resolved_depths[name] = ancestry_depth
            return result
        finally:
            resolving.pop()

    # Resolve every definition so an invalid unused template cannot hide a cycle or bad parent.
    for name in sorted(definitions):
        resolve(name)
    return {name: resolved[name] for name in models}, True


def _parse_inference_model(
    model_name: str,
    data: dict,
    *,
    resolve_environment: bool = True,
    config_dir: Path | None = None,
) -> InferenceModelConfig:
    """Parse one configured inference model."""

    owner = f"Inference model '{model_name}'"
    _reject_unknown_fields(data, set(_INFERENCE_MODEL_FIELDS), owner)
    tokens = (
        _parse_input_tokens(
            model_name,
            data["tokens"],
            resolve_environment=resolve_environment,
            config_dir=config_dir,
        )
        if "tokens" in data
        else None
    )
    score = (
        _parse_score_config(
            model_name,
            data["score"],
            resolve_environment=resolve_environment,
        )
        if "score" in data
        else None
    )
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
    try:
        revision = _model_revision(
            _configured_environment_value(
                _model_revision(data.get("revision")),
                owner,
                resolve_environment=resolve_environment,
            )
        )
    except ValueError as exc:
        raise ConfigurationError(f"{owner} revision must be a non-empty string or null") from exc
    adapter = (
        _required_string(data, "adapter", owner) if "adapter" in data else DEFAULT_INFERENCE_ADAPTER
    )
    return InferenceModelConfig(
        adapter=adapter,
        model=cast(
            str,
            _configured_environment_value(
                _required_string(data, "model", owner),
                owner,
                resolve_environment=resolve_environment,
            ),
        ),
        endpoint=_configured_environment_value(
            _optional_string(data, "endpoint", owner),
            owner,
            resolve_environment=resolve_environment,
        ),
        aws_profile=_configured_environment_value(
            _optional_string(data, "aws_profile", owner),
            owner,
            resolve_environment=resolve_environment,
        ),
        region=_configured_environment_value(
            _optional_string(data, "region", owner),
            owner,
            resolve_environment=resolve_environment,
        ),
        tokens=tokens,
        score=score,
        category=category,
        classify=classify,
        spans=spans,
        complete=complete,
        revision=revision,
        windows=_parse_model_windows(data["windows"], owner=f"{owner} windows")
        if data.get("windows") is not None
        else None,
    )


def _parse_model_windows(value: object, *, owner: str) -> ModelWindowsConfig:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{owner} must be a mapping")
    _reject_unknown_fields(value, {"unit", "size", "step", "scope", "renderer"}, owner)
    if "unit" not in value or "size" not in value:
        raise ConfigurationError(f"{owner} requires unit and size")
    try:
        return ModelWindowsConfig(
            unit=value["unit"],
            size=value["size"],
            step=value.get("step"),
            scope=value.get("scope", "all"),
            renderer=value.get("renderer", "join"),
        )
    except ValueError as exc:
        raise ConfigurationError(f"{owner}: {exc}") from exc


def _parse_complete_config(
    model_name: str,
    value: object,
    *,
    owner: str | None = None,
) -> CompleteConfig:
    """Parse one alias's completion settings: the JSON Schema its replies must match."""

    owner = owner or f"Inference model '{model_name}' complete"
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a mapping")
    _reject_unknown_fields(value, {"response_schema", "max_input_tokens"}, owner)
    if "response_schema" not in value:
        raise ValueError(f"{owner} requires a 'response_schema'")
    limit = _input_token_limit(value.get("max_input_tokens"))
    try:
        return CompleteConfig(
            response_schema=response_schema_text(value["response_schema"]),
            max_input_tokens=limit,
        )
    except ValueError as exc:
        raise ValueError(f"{owner} response_schema {exc}") from exc


def _parse_classify_config(
    model_name: str,
    value: object,
    *,
    owner: str | None = None,
) -> ClassifyConfig:
    """Parse the classifier head exposed by one logical model alias."""

    owner = owner or f"Inference model '{model_name}' classify"
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a mapping")
    fields: dict[object, object] = dict(value)
    _reject_unknown_fields(
        fields, {"class_index", "instructions", "labels", "max_input_tokens"}, owner
    )
    class_index = fields.get("class_index")
    if isinstance(class_index, bool) or not isinstance(class_index, int) or class_index < 0:
        raise ValueError(f"{owner} requires a non-negative integer 'class_index'")
    instructions = _optional_string(fields, "instructions", owner)
    labels = _parse_classify_labels(fields["labels"], owner) if "labels" in fields else None
    if labels is not None and class_index >= len(labels):
        raise ValueError(f"{owner} class_index {class_index} is outside its {len(labels)} labels")
    return ClassifyConfig(
        class_index=class_index,
        instructions=instructions,
        labels=labels,
        max_input_tokens=_input_token_limit(fields.get("max_input_tokens")),
    )


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


def _parse_span_config(
    model_name: str,
    value: object,
    *,
    owner: str | None = None,
) -> SpanConfig | ExternalSpanConfig | CandidateSpanConfig:
    """Parse the span output class exposed by one logical alias."""

    owner = owner or f"Inference model '{model_name}' spans"
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a mapping")
    fields: dict[object, object] = dict(value)
    if fields.get("backend") == "candidates":
        _reject_unknown_fields(
            fields,
            {"backend", "batch", "channel", "settings_digest", "limits", "max_input_tokens"},
            owner,
        )
        limits = fields.get("limits", {})
        if type(limits) is not dict:
            raise ValueError(f"{owner} limits must be a mapping")
        _reject_unknown_fields(
            limits,
            {
                "max_input_characters",
                "max_parts",
                "max_ranges_per_channel",
                "max_coverage_per_channel",
                "max_work",
            },
            f"{owner} limits",
        )
        return CandidateSpanConfig(
            batch=_required_string(fields, "batch", owner),
            channel=_required_string(fields, "channel", owner),
            settings_digest=_required_string(fields, "settings_digest", owner),
            limits=CandidateBatchLimits(**cast(dict[str, int], limits)),
            max_input_tokens=_input_token_limit(fields.get("max_input_tokens")),
        )
    if "backend" in fields:
        _reject_unknown_fields(fields, {"backend", "renderer", "limits", "max_input_tokens"}, owner)
        limits = fields.get("limits", {})
        if type(limits) is not dict:
            raise ValueError(f"{owner} limits must be a mapping")
        _reject_unknown_fields(
            limits,
            {"max_input_characters", "max_spans", "max_coverage_ranges", "max_work"},
            f"{owner} limits",
        )
        return ExternalSpanConfig(
            backend=_required_string(fields, "backend", owner),
            renderer=_required_string(fields, "renderer", owner)
            if "renderer" in fields
            else "join",
            limits=ExternalSpanLimits(**cast(dict[str, int], limits)),
            max_input_tokens=_input_token_limit(fields.get("max_input_tokens")),
        )
    _reject_unknown_fields(fields, {"class_index", "max_input_tokens"}, owner)
    class_index = fields.get("class_index")
    if isinstance(class_index, bool) or not isinstance(class_index, int) or class_index < 0:
        raise ValueError(f"{owner} requires a non-negative integer 'class_index'")
    return SpanConfig(
        class_index=class_index, max_input_tokens=_input_token_limit(fields.get("max_input_tokens"))
    )


def _parse_input_tokens(
    model_name: str,
    value: object,
    *,
    owner: str | None = None,
    resolve_environment: bool = True,
    config_dir: Path | None = None,
) -> InputTokensConfig:
    """Parse the tokenizer endpoint/model and response count field for one model."""

    owner = owner or f"Inference model '{model_name}' tokens"
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a mapping")
    fields: dict[object, object] = dict(value)
    _reject_unknown_fields(
        fields, {"adapter", "endpoint", "extraction", "model", "local", "source_width"}, owner
    )
    extraction = fields.get("extraction")
    if not isinstance(extraction, dict):
        raise ValueError(f"{owner} requires an 'extraction' mapping")
    _reject_unknown_fields(extraction, {"count"}, f"{owner} extraction")
    return InputTokensConfig(
        adapter=_required_string(fields, "adapter", owner),
        endpoint=_configured_environment_value(
            _optional_string(fields, "endpoint", owner),
            owner,
            resolve_environment=resolve_environment,
        ),
        count_field=_required_string(extraction, "count", f"{owner} extraction"),
        model=_configured_environment_value(
            _optional_string(fields, "model", owner),
            owner,
            resolve_environment=resolve_environment,
        ),
        local=_parse_local_tokenizer(fields["local"], config_dir=config_dir)
        if "local" in fields
        else None,
        source_width=_parse_source_width(fields["source_width"])
        if "source_width" in fields
        else None,
    )


def _parse_source_width(value: object) -> TokenizerWidthClaim:
    if not isinstance(value, dict):
        raise ValueError("source width claim must be a native mapping")
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError("source width claim must be a native mapping")
    _reject_unknown_fields(value, set(TokenizerWidthClaim.__dataclass_fields__), "source width")
    try:
        return TokenizerWidthClaim(**value)
    except TypeError:
        raise ValueError(
            "source width requires all revision, artifact and settings fields"
        ) from None


def _parse_local_tokenizer(value: object, *, config_dir: Path | None) -> LocalTokenizerConfig:
    """Strict native local configuration. Parsing never opens the pinned artifact."""
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError("local tokenizer configuration must be a native mapping")
    _reject_unknown_fields(value, set(LocalTokenizerConfig.__dataclass_fields__), "local tokenizer")
    fields = dict(value)
    for name, kind in (("limits", LocalTokenizerLimits), ("framing", TokenFramingMargins)):
        if name in fields:
            raw = fields[name]
            if type(raw) is not dict or any(type(key) is not str for key in raw):
                raise ValueError("local tokenizer limits/framing must be native mappings")
            _reject_unknown_fields(raw, set(kind.__dataclass_fields__), "local tokenizer " + name)
            fields[name] = kind(**raw)
    if "template_variables" in fields:
        variables = fields["template_variables"]
        if type(variables) is not dict or len(variables) > 32:
            raise ValueError("local tokenizer template variables must be a bounded native mapping")
        fields["template_variables"] = tuple(variables.items())
    try:
        config = LocalTokenizerConfig(**fields)
    except TypeError:
        raise ValueError(
            "local tokenizer requires artifact path/digest and model/tokenizer revisions"
        ) from None
    if config_dir is not None and not Path(config.artifact_path).is_absolute():
        config = replace(config, artifact_path=str(config_dir / config.artifact_path))
    return config


def _parse_score_config(
    model_name: str,
    value: object,
    *,
    owner: str | None = None,
    resolve_environment: bool = True,
) -> ScoreConfig:
    """Parse one model score prompt and its optional adapter settings."""

    owner = owner or f"Inference model '{model_name}' score"
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a mapping")
    fields: dict[object, object] = dict(value)
    _reject_unknown_fields(
        fields,
        {
            "message_builder",
            "boundary_seed",
            "tokens",
            "extraction",
            "calibration",
            "reasoning",
            "explanation",
            "max_input_tokens",
            "arguments",
        },
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
        if any(
            type(token) is str and len(token) <= 256 and not token.strip()
            for token in (score_tokens.get("negative"), score_tokens.get("positive"))
        ):
            raise ValueError(f"{owner} positive and negative tokens must be non-empty")
        negative_token = score_labels(score_tokens.get("negative"))
        positive_token = score_labels(score_tokens.get("positive"))
        if negative_token is None or positive_token is None:
            raise ValueError(f"{owner} positive and negative tokens must be non-empty")
        validate_score_sides(negative_token, positive_token)

    probabilities_field = None
    verdict = None
    if "extraction" in fields:
        extraction = fields["extraction"]
        if not isinstance(extraction, dict):
            raise ValueError(f"{owner} extraction must be a mapping")
        _reject_unknown_fields(
            extraction, {"probabilities", "label", "confidence", "max_chars"}, f"{owner} extraction"
        )
        probabilities_field = _optional_string(extraction, "probabilities", f"{owner} extraction")
        if "label" in extraction:
            verdict = ScoreVerdictConfig(
                label_path=score_path(extraction["label"]),
                confidence_path=(
                    score_path(extraction["confidence"]) if "confidence" in extraction else None
                ),
                max_chars=extraction.get("max_chars", 16_384),
            )
        elif "confidence" in extraction or "max_chars" in extraction:
            raise ValueError(f"{owner} JSON confidence and limits require a label path")

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
        if reasoning_max_tokens > 4096:
            raise ValueError(f"{owner} reasoning max_tokens must not exceed 4096")

    explanation_max_tokens = None
    if "explanation" in fields:
        explanation = fields["explanation"]
        if not isinstance(explanation, dict):
            raise ValueError(f"{owner} explanation must be a mapping")
        _reject_unknown_fields(explanation, {"max_tokens"}, f"{owner} explanation")
        explanation_max_tokens = _explanation_token_budget(
            explanation.get("max_tokens"), reasoning_max_tokens, owner
        )

    return ScoreConfig(
        message_builder=_optional_string(fields, "message_builder", owner) or "constitution",
        boundary_seed=boundary_seed,
        negative_token=negative_token,
        positive_token=positive_token,
        probabilities_field=probabilities_field,
        calibration=_configured_environment_value(
            _optional_string(fields, "calibration", owner),
            owner,
            resolve_environment=resolve_environment,
        ),
        reasoning_max_tokens=reasoning_max_tokens,
        explanation_max_tokens=explanation_max_tokens,
        max_input_tokens=_input_token_limit(fields.get("max_input_tokens")),
        arguments=score_arguments(fields.get("arguments")),
        verdict=verdict,
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
