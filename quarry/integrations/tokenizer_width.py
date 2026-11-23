# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Attest a conservative source-width bound for a modeled native BPE pipeline.

The complete model byte alphabet covers every UTF8 byte. Isolated ByteLevel
splits retain input bytes; optional prefix space only adds bytes. Injective
vocabulary IDs and concatenating merges bound each model token's symbol length.
Literal added tokens have their own bounded source width, without stripping or
normalization. Together these establish ceil(original code points / width) as a
lower bound on token count. Unsupported or mismatched pipelines return no proof.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any

from quarry.integrations.local_tokenizer_artifact import _LoadedTokenizer
from quarry.integrations.local_tokenizer_worker import _Job
from quarry.types.local_tokenizer import LocalTokenizerConfig
from quarry.types.tokenizer_width import TokenizerWidthClaim

_SEAL = object()


def _fingerprint(claim: TokenizerWidthClaim, derived: int, settings: str) -> str:
    return sha256(
        json.dumps(
            {
                "version": 1,
                "claim": claim.identity_dict(),
                "attested_max_width": derived,
                "settings": settings,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedTokenizerWidth:
    """Detached mathematical attestation, created by the actual native verifier.

    This retains bounded configuration metadata and no tokenizer, artifact graph,
    request, source text, mutex or worker. Capped replies still need their own
    actual alias/adapter/revision/cap binding before cache admission.
    """

    claim: TokenizerWidthClaim
    attested_max_source_characters: int
    count_settings_sha256: str
    fingerprint: str
    _seal: object = field(repr=False, compare=False)

    def __init__(self):
        raise TypeError("tokenizer width requires native runtime attestation")

    def is_compatible(self, config: LocalTokenizerConfig, claim: TokenizerWidthClaim) -> bool:
        if (
            type(self) is not VerifiedTokenizerWidth
            or type(config) is not LocalTokenizerConfig
            or type(claim) is not TokenizerWidthClaim
        ):
            return False
        try:
            if self._seal is not _SEAL or type(self.claim) is not TokenizerWidthClaim:
                return False
            own, requested, checked = replace(self.claim), replace(claim), replace(config)
            width = self.attested_max_source_characters
            if (
                type(width) is not int
                or not 1 <= width <= 1_048_576
                or width > own.max_source_characters_per_token
            ):
                return False
            settings = checked.count_settings_sha256
            return (
                own == requested
                and own.artifact_sha256 == checked.artifact_sha256
                and own.model_revision == checked.model_revision
                and own.tokenizer_revision == checked.tokenizer_revision
                and own.count_settings_sha256 == settings
                and self.count_settings_sha256 == settings
                and self.fingerprint == _fingerprint(own, width, settings)
            )
        except (AttributeError, TypeError, ValueError):
            return False


def _verified(claim: TokenizerWidthClaim, width: int, settings: str) -> VerifiedTokenizerWidth:
    result = object.__new__(VerifiedTokenizerWidth)
    object.__setattr__(result, "claim", replace(claim))
    object.__setattr__(result, "attested_max_source_characters", width)
    object.__setattr__(result, "count_settings_sha256", settings)
    object.__setattr__(result, "fingerprint", _fingerprint(claim, width, settings))
    object.__setattr__(result, "_seal", _SEAL)
    return result


def _one_byte_level(node: Any) -> bool:
    for _ in range(9):
        if type(node) is not dict:
            return False
        if node.get("type") == "ByteLevel":
            return set(node) <= {"type", "add_prefix_space", "trim_offsets", "use_regex"} and all(
                type(node.get(key)) is bool
                for key in ("add_prefix_space", "trim_offsets", "use_regex")
            )
        if node.get("type") != "Sequence" or set(node) != {"type", "pretokenizers"}:
            return False
        children = node["pretokenizers"]
        if type(children) is not list or len(children) != 1:
            return False
        node = children[0]
    return False


def attest_loaded_width(
    config: LocalTokenizerConfig,
    claim: TokenizerWidthClaim,
    loaded: _LoadedTokenizer,
    job: _Job,
) -> VerifiedTokenizerWidth | None:
    """Model the actual canonical native state. A claim alone never returns proof."""
    job.checkpoint()
    if (
        type(config) is not LocalTokenizerConfig
        or type(claim) is not TokenizerWidthClaim
        or type(loaded) is not _LoadedTokenizer
    ):
        return None
    try:
        config, claim = replace(config), replace(claim)
    except (TypeError, ValueError):
        return None
    settings = config.count_settings_sha256
    if (
        type(loaded.count_settings_sha256) is not str
        or loaded.count_settings_sha256 != settings
        or claim.count_settings_sha256 != settings
        or claim.artifact_sha256 != config.artifact_sha256
        or claim.model_revision != config.model_revision
        or claim.tokenizer_revision != config.tokenizer_revision
    ):
        return None
    import tokenizers

    native = loaded.tokenizer
    if tokenizers.__version__ != "0.22.2" or type(native) is not tokenizers.Tokenizer:
        return None
    if (
        type(native.model) is not tokenizers.models.BPE
        or native.normalizer is not None
        or native.truncation is not None
        or native.padding is not None
        or native.post_processor is not None
        or native.decoder is not None
    ):
        return None
    # get_vocab reads the model's original map. A reverse-ID serialization can
    # silently lose entries when IDs collide, so serialization alone is unsafe.
    vocab = native.get_vocab(with_added_tokens=False)
    job.checkpoint()
    if type(vocab) is not dict or not 256 <= len(vocab) <= 262_144:
        return None
    ids: dict[int, str] = {}
    width = 1
    for index, (word, token) in enumerate(vocab.items()):
        if index % 256 == 0:
            job.checkpoint()
        if (
            type(word) is not str
            or not 1 <= len(word) <= 4096
            or type(token) is not int
            or not 0 <= token < 1 << 32
            or token in ids
        ):
            return None
        ids[token] = word
        width = max(width, len(word))
    alphabet = tokenizers.pre_tokenizers.ByteLevel.alphabet()
    if (
        len(alphabet) != 256
        or len(set(alphabet)) != 256
        or any(char not in vocab for char in alphabet)
    ):
        return None
    serialized = native.to_str()
    job.checkpoint()
    if (
        type(serialized) is not str
        or len(serialized) > config.limits.max_artifact_bytes * 2 + 65_536
    ):
        return None
    document = json.loads(serialized)
    if type(document) is not dict or not _one_byte_level(document.get("pre_tokenizer")):
        return None
    model = document.get("model")
    if type(model) is not dict or model.get("type") != "BPE" or model.get("vocab") != vocab:
        return None
    if (
        model.get("continuing_subword_prefix") not in (None, "")
        or model.get("end_of_word_suffix") not in (None, "")
        or model.get("dropout") not in (None, 0, 0.0)
    ):
        return None
    merges = model.get("merges")
    if type(merges) is not list or len(merges) > 262_144:
        return None
    for index, row in enumerate(merges):
        if index % 256 == 0:
            job.checkpoint()
        if type(row) is str:
            row = row.split(" ")
        if type(row) is not list or len(row) != 2:
            return None
        left, right = row
        if type(left) is not str or type(right) is not str:
            return None
        if (
            left not in vocab
            or right not in vocab
            or len(left) + len(right) > width
            or left + right not in vocab
        ):
            return None
    added = document.get("added_tokens")
    if type(added) is not list or len(added) > 4096:
        return None
    for row in added:
        job.checkpoint()
        if (
            type(row) is not dict
            or row.get("lstrip") is not False
            or row.get("rstrip") is not False
        ):
            return None
        content, token = row.get("content"), row.get("id")
        if (
            type(content) is not str
            or not 1 <= len(content) <= 4096
            or type(token) is not int
            or not 0 <= token < 1 << 32
        ):
            return None
        if token in ids and ids[token] != content:
            return None
        ids[token] = content
        width = max(width, len(content))
    job.checkpoint()
    if claim.max_source_characters_per_token < width:
        return None
    return _verified(claim, width, settings)
