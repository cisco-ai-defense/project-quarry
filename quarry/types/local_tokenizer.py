# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Bounded native configuration and code-point geometry for local tokenization.

These values neither import a tokenizer nor open an artifact. Runtime adapters
must verify the pinned bytes and execute under their own bounded worker owner.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from math import isfinite

_MAX_CHARACTERS = 1_048_576
_MAX_TOKENS = 262_144
_MAX_TOKEN_ID = (1 << 32) - 1


class LocalTokenizerValidationError(ValueError):
    """A native tokenizer configuration or reply violates its finite contract."""


def _integer(value: object, low: int, high: int, name: str) -> None:
    if type(value) is not int or not low <= value <= high:
        raise LocalTokenizerValidationError(f"invalid local tokenizer {name}")


def _label(value: object, name: str) -> None:
    if type(value) is not str or not 1 <= len(value) <= 256 or not value.isprintable():
        raise LocalTokenizerValidationError(f"invalid local tokenizer {name}")


@dataclass(frozen=True, slots=True)
class LocalTokenizerLimits:
    """Absolute caps also bound a host's direct construction of native values."""

    max_artifact_bytes: int = 16_777_216
    max_input_characters: int = 262_144
    max_input_bytes: int = 1_048_576
    max_tokens: int = _MAX_TOKENS
    max_messages: int = 256
    max_tools: int = 64
    max_template_characters: int = 65_536
    max_template_nodes: int = 2048
    max_template_steps: int = 32_768
    max_output_characters: int = 262_144
    max_wall_seconds: float = 2.0

    def __post_init__(self) -> None:
        for name, high in (
            ("max_artifact_bytes", 16_777_216),
            ("max_input_characters", _MAX_CHARACTERS),
            ("max_input_bytes", 4_194_304),
            ("max_tokens", _MAX_TOKENS),
            ("max_messages", 256),
            ("max_tools", 64),
            ("max_template_characters", 65_536),
            ("max_template_nodes", 2048),
            ("max_template_steps", 32_768),
            ("max_output_characters", _MAX_CHARACTERS),
        ):
            _integer(
                getattr(self, name), 0 if name in {"max_messages", "max_tools"} else 1, high, name
            )
        if (
            type(self.max_wall_seconds) not in (int, float)
            or not 0 < self.max_wall_seconds <= 30
            or not isfinite(self.max_wall_seconds)
        ):
            raise LocalTokenizerValidationError("invalid local tokenizer wall allowance")
        object.__setattr__(self, "max_wall_seconds", float(self.max_wall_seconds))


@dataclass(frozen=True, slots=True)
class TokenFramingMargins:
    """Declared whole-request additions; never implicit padding or truncation."""

    per_request: int = 0
    per_message: int = 0

    def __post_init__(self) -> None:
        _integer(self.per_request, 0, _MAX_TOKENS, "request framing")
        _integer(self.per_message, 0, _MAX_TOKENS, "message framing")

    def total(self, message_count: int) -> int:
        _integer(message_count, 0, 256, "message count")
        return self.per_request + self.per_message * message_count


@dataclass(frozen=True, slots=True)
class LocalTokenizerConfig:
    """A pinned local artifact and explicit template, with no filesystem discovery."""

    artifact_path: str
    artifact_sha256: str
    model_revision: str
    tokenizer_revision: str
    template: str | None = field(default=None, repr=False)
    template_variables: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    add_generation_prompt: bool = False
    trim_blocks: bool = False
    lstrip_blocks: bool = False
    framing: TokenFramingMargins = field(default_factory=TokenFramingMargins)
    limits: LocalTokenizerLimits = field(default_factory=LocalTokenizerLimits)

    def __post_init__(self) -> None:
        if (
            type(self.artifact_path) is not str
            or not 1 <= len(self.artifact_path) <= 4096
            or "\x00" in self.artifact_path
            or "://" in self.artifact_path
        ):
            raise LocalTokenizerValidationError("local tokenizer requires a local artifact path")
        if (
            type(self.artifact_sha256) is not str
            or len(self.artifact_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.artifact_sha256)
        ):
            raise LocalTokenizerValidationError("invalid local tokenizer artifact digest")
        _label(self.model_revision, "model revision")
        _label(self.tokenizer_revision, "tokenizer revision")
        for name in ("add_generation_prompt", "trim_blocks", "lstrip_blocks"):
            if type(getattr(self, name)) is not bool:
                raise LocalTokenizerValidationError("invalid local tokenizer template option")
        if type(self.template_variables) not in (tuple, list) or len(self.template_variables) > 32:
            raise LocalTokenizerValidationError("invalid local tokenizer template variables")
        variables = {}
        for pair in self.template_variables:
            if type(pair) not in (tuple, list) or len(pair) != 2:
                raise LocalTokenizerValidationError("invalid local tokenizer template variable")
            key, value = pair
            if (
                type(key) is not str
                or not 1 <= len(key) <= 64
                or not key.isascii()
                or not key.isidentifier()
                or key.startswith("_")
                or key
                in {"messages", "tools", "add_generation_prompt", "loop", "true", "false", "none"}
                or key in variables
                or type(value) is not str
                or len(value) > 4096
            ):
                raise LocalTokenizerValidationError("invalid local tokenizer template variable")
            try:
                value.encode("utf-8")
            except UnicodeError:
                raise LocalTokenizerValidationError(
                    "invalid local tokenizer variable Unicode"
                ) from None
            variables[key] = value
        object.__setattr__(self, "template_variables", tuple(sorted(variables.items())))
        if (
            type(self.framing) is not TokenFramingMargins
            or type(self.limits) is not LocalTokenizerLimits
        ):
            raise LocalTokenizerValidationError(
                "local tokenizer requires native limits and framing"
            )
        object.__setattr__(self, "framing", replace(self.framing))
        object.__setattr__(self, "limits", replace(self.limits))
        if self.template is not None:
            if (
                type(self.template) is not str
                or not 1 <= len(self.template) <= self.limits.max_template_characters
            ):
                raise LocalTokenizerValidationError("invalid local tokenizer template size")
            try:
                self.template.encode("utf-8")
            except UnicodeError:
                raise LocalTokenizerValidationError(
                    "invalid local tokenizer template Unicode"
                ) from None

    def identity_dict(self) -> dict[str, object]:
        """Count semantics only: exclude the artifact path and template plaintext."""
        return {
            "version": 1,
            "artifact_sha256": self.artifact_sha256,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "template_sha256": sha256(self.template.encode("utf-8")).hexdigest()
            if self.template is not None
            else None,
            "template_variables_sha256": sha256(
                json.dumps(
                    dict(self.template_variables),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "add_generation_prompt": self.add_generation_prompt,
            "trim_blocks": self.trim_blocks,
            "lstrip_blocks": self.lstrip_blocks,
            "framing": asdict(self.framing),
            "limits": asdict(self.limits),
            "padding": False,
            "truncation": False,
            "post_processor": False,
            "add_special_tokens": False,
            "offset_unit": "unicode_code_points",
        }


@dataclass(frozen=True, slots=True)
class LocalTokenizerEncoding:
    """Detached token IDs and original-text offsets; gaps/overlaps are explicit.

    Special tokens have no source offset. Multiple byte-level tokens may map to
    one code point; offsets can overlap, without claiming whole-input coverage.
    """

    token_ids: tuple[int, ...]
    character_offsets: tuple[tuple[int, int] | None, ...]
    special_tokens_mask: tuple[bool, ...]
    source_characters: int

    def __post_init__(self) -> None:
        _integer(self.source_characters, 0, _MAX_CHARACTERS, "source character count")
        for values in (self.token_ids, self.character_offsets, self.special_tokens_mask):
            if type(values) not in (tuple, list) or len(values) > _MAX_TOKENS:
                raise LocalTokenizerValidationError("invalid local tokenizer encoding size")
        if not len(self.token_ids) == len(self.character_offsets) == len(self.special_tokens_mask):
            raise LocalTokenizerValidationError("local tokenizer encoding vectors differ in size")
        offsets = []
        previous_start = previous_end = 0
        for token, offset, special in zip(
            self.token_ids, self.character_offsets, self.special_tokens_mask, strict=True
        ):
            _integer(token, 0, _MAX_TOKEN_ID, "token ID")
            if type(special) is not bool:
                raise LocalTokenizerValidationError("invalid local tokenizer special-token mask")
            if special:
                if offset is not None:
                    raise LocalTokenizerValidationError(
                        "special token has a fabricated source offset"
                    )
                offsets.append(None)
                continue
            if offset is None or type(offset) not in (tuple, list) or len(offset) != 2:
                raise LocalTokenizerValidationError("invalid local tokenizer code-point offset")
            start, end = offset
            _integer(start, 0, self.source_characters, "offset start")
            _integer(end, start, self.source_characters, "offset end")
            if start < previous_start or end < previous_end:
                raise LocalTokenizerValidationError("local tokenizer offsets are out of order")
            previous_start, previous_end = start, end
            offsets.append((start, end))
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        object.__setattr__(self, "character_offsets", tuple(offsets))
        object.__setattr__(self, "special_tokens_mask", tuple(self.special_tokens_mask))
