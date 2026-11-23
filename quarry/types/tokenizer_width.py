# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Native width claims and explicit exact/overflow count values.

A claim needs runtime attestation against the actual tokenizer and settings.
Construction performs no I/O, imports no optional backend and establishes no
verified bound. Consumers must validate native provenance before retaining an
overflow result; an overflow lower bound is never an exact count.
"""

import json
from dataclasses import dataclass
from hashlib import sha256

_MAX = (1 << 63) - 1
_METHOD = "bytelevel-bpe-complete-alphabet-v1"


class TokenizerWidthValidationError(ValueError):
    """A native width claim or capped-count shape is invalid."""


def _integer(value: object, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise TokenizerWidthValidationError("invalid tokenizer width integer")


def _digest(value: object) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise TokenizerWidthValidationError("invalid tokenizer width identity digest")


def _revision(value: object) -> None:
    if type(value) is not str or not 1 <= len(value) <= 256 or not value.isprintable():
        raise TokenizerWidthValidationError("invalid tokenizer width revision")


@dataclass(frozen=True, slots=True)
class TokenizerWidthClaim:
    """Caller-supplied upper source width, requiring native runtime verification."""

    max_source_characters_per_token: int
    artifact_sha256: str
    model_revision: str
    tokenizer_revision: str
    count_settings_sha256: str
    method: str = _METHOD

    def __post_init__(self) -> None:
        _integer(self.max_source_characters_per_token, 1, 1_048_576)
        _digest(self.artifact_sha256)
        _digest(self.count_settings_sha256)
        _revision(self.model_revision)
        _revision(self.tokenizer_revision)
        if type(self.method) is not str or self.method != _METHOD:
            raise TokenizerWidthValidationError("unsupported tokenizer width attestation method")

    def identity_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "method": self.method,
            "max_source_characters_per_token": self.max_source_characters_per_token,
            "artifact_sha256": self.artifact_sha256,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "count_settings_sha256": self.count_settings_sha256,
        }

    @property
    def fingerprint(self) -> str:
        return sha256(
            json.dumps(self.identity_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class CappedTokenCount:
    """An exact canonical count or a proved strict lower bound above one cap.

    Exact canonical counts can themselves exceed the cap. Overflow values carry
    no exact count. Their native attestation fingerprint names configuration,
    never input text; runtime ownership/binding must still be verified by the
    adapter/backend before admission, independently of this shape validation.
    """

    cap: int
    lower_bound: int
    exact_count: int | None = None
    proof_fingerprint: str | None = None

    def __post_init__(self) -> None:
        _integer(self.cap, 0, _MAX)
        _integer(self.lower_bound, 0, _MAX)
        if self.exact_count is not None:
            _integer(self.exact_count, 0, _MAX)
            if self.lower_bound != self.exact_count or self.proof_fingerprint is not None:
                raise TokenizerWidthValidationError("invalid exact tokenizer count shape")
        else:
            if self.lower_bound <= self.cap:
                raise TokenizerWidthValidationError(
                    "tokenizer overflow must strictly exceed its cap"
                )
            _digest(self.proof_fingerprint)

    @property
    def is_exact(self) -> bool:
        return self.exact_count is not None

    @property
    def exceeds_cap(self) -> bool:
        return self.lower_bound > self.cap
