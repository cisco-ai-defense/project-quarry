# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Pinned local tokenizer bytes and conservative native allocation admission.

Only the encoding components are instantiated. Artifact decoders, padding,
truncation and post-processors never execute: explicit request templates contain
their own markers and framing additions are separate scalar counts. Unknown
components fail closed. This is an allocation/work admission bound, not a claim
that a Python thread can preempt native work or enforce an OS memory quota.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from math import isfinite
from typing import Any, NoReturn

from quarry.integrations.local_tokenizer_worker import _Job
from quarry.types.exceptions import (
    LocalTokenizerInvalidInputError,
    LocalTokenizerInvalidOutputError,
    LocalTokenizerLimitError,
    LocalTokenizerUnavailableError,
)
from quarry.types.local_tokenizer import LocalTokenizerConfig, LocalTokenizerEncoding

_NATIVE_TOKENS = 262_144
_NATIVE_BYTES = 4_194_304
_NATIVE_SEARCH = 4_194_304
_JSON_NODES = 1_048_576


def _unavailable() -> NoReturn:
    raise LocalTokenizerUnavailableError("Local tokenizer artifact is unavailable or unsupported")


def _limit() -> NoReturn:
    raise LocalTokenizerLimitError("Local tokenizer artifact or native work limit was reached")


def _bound(value: int, cap: int = _NATIVE_BYTES) -> int:
    if value > cap:
        _limit()
    return value


def _read(config: LocalTokenizerConfig, job: _Job) -> bytes:
    """A bounded regular-file read; all native parsing uses these verified bytes."""
    job.checkpoint()
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(config.artifact_path, flags)
    except (OSError, ValueError):
        _unavailable()
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            _unavailable()
        if not 0 < before.st_size <= config.limits.max_artifact_bytes:
            _limit()
        data = bytearray()
        while True:
            job.checkpoint()
            part = os.read(fd, min(65_536, config.limits.max_artifact_bytes + 1 - len(data)))
            if not part:
                break
            data.extend(part)
            if len(data) > config.limits.max_artifact_bytes:
                _limit()
        after = os.fstat(fd)
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            _unavailable()
    except OSError:
        _unavailable()
    finally:
        os.close(fd)
    job.checkpoint()
    if hashlib.sha256(data).hexdigest() != config.artifact_sha256:
        _unavailable()
    return bytes(data)


def _json_preflight(text: str, job: _Job) -> None:
    # Bound object count and nesting BEFORE json.loads allocates the native graph.
    # This is deliberately only a lexical bound; json.loads validates the grammar.
    depth = nodes = 0
    quoted = escaped = atom = False
    for ordinal, char in enumerate(text):
        if ordinal % 4096 == 0:
            job.checkpoint()
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted, atom = True, False
            nodes += 1
        elif char in "[{":
            depth += 1
            nodes += 1
            atom = False
        elif char in "]}":
            depth -= 1
            atom = False
        elif char in ",: \t\r\n":
            atom = False
        elif not atom:
            nodes += 1
            atom = True
        if depth > 32 or nodes > _JSON_NODES:
            _limit()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            _unavailable()
        result[key] = value
    return result


def _constant(_: str) -> NoReturn:
    _unavailable()


def _graph(value: Any, job: _Job, depth: int = 0) -> None:
    job.checkpoint()
    if depth > 32:
        _limit()
    if type(value) is dict:
        for key, item in value.items():
            _graph(key, job, depth + 1)
            _graph(item, job, depth + 1)
    elif type(value) is list:
        for item in value:
            _graph(item, job, depth + 1)
    elif type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeError:
            _unavailable()
    elif (
        type(value) is float
        and not isfinite(value)
        or type(value) is int
        and not -(1 << 63) <= value < 1 << 64
    ):
        _unavailable()


def _shape(value: Any, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) - fields:
        _unavailable()
    return value


def _string(value: Any, *, maximum: int = 4096, empty: bool = False) -> str:
    if type(value) is not str or not (0 if empty else 1) <= len(value) <= maximum:
        _unavailable()
    return value


def _boolean(value: Any) -> None:
    if type(value) is not bool:
        _unavailable()


def _stages(value: Any, *, normalizer: bool) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []

    def visit(node: Any, depth: int) -> None:
        if node is None:
            return
        if depth > 8 or type(node) is not dict or type(node.get("type")) is not str:
            _unavailable()
        kind = node["type"]
        if kind == "Sequence":
            key = "normalizers" if normalizer else "pretokenizers"
            _shape(node, {"type", key})
            children = node.get(key)
            if type(children) is not list or len(children) > 64:
                _unavailable()
            for child in children:
                visit(child, depth + 1)
            return
        if len(result) >= 64:
            _limit()
        if normalizer:
            fields = {
                "Lowercase": set(),
                "NFC": set(),
                "NFD": set(),
                "NFKC": set(),
                "NFKD": set(),
                "StripAccents": set(),
                "Strip": {"strip_left", "strip_right"},
                "Prepend": {"prepend"},
                "Replace": {"pattern", "content"},
                "BertNormalizer": {
                    "clean_text",
                    "handle_chinese_chars",
                    "strip_accents",
                    "lowercase",
                },
            }.get(kind)
        else:
            fields = {
                "Whitespace": set(),
                "WhitespaceSplit": set(),
                "BertPreTokenizer": set(),
                "UnicodeScripts": set(),
                "Digits": {"individual_digits"},
                "Punctuation": {"behavior"},
                "CharDelimiterSplit": {"delimiter"},
                "ByteLevel": {"add_prefix_space", "trim_offsets", "use_regex"},
                "Metaspace": {"replacement", "prepend_scheme", "split", "add_prefix_space"},
                "Split": {"pattern", "behavior", "invert"},
            }.get(kind)
        if fields is None:
            _unavailable()
        _shape(node, fields | {"type"})
        for key in fields & {
            "strip_left",
            "strip_right",
            "clean_text",
            "handle_chinese_chars",
            "lowercase",
            "individual_digits",
            "add_prefix_space",
            "trim_offsets",
            "use_regex",
            "split",
            "invert",
        }:
            if key in node:
                _boolean(node[key])
        if "strip_accents" in node and node["strip_accents"] is not None:
            _boolean(node["strip_accents"])
        for key in ("prepend", "content", "replacement", "delimiter", "behavior", "prepend_scheme"):
            if key in node:
                _string(node[key], empty=key in {"prepend", "content"})
        if kind in {"Replace", "Split"}:
            pattern = _shape(node.get("pattern"), {"String"})
            _string(pattern.get("String"), empty=True)
        if kind in {"Metaspace", "CharDelimiterSplit"}:
            key = "replacement" if kind == "Metaspace" else "delimiter"
            if len(_string(node.get(key))) != 1:
                _unavailable()
        result.append(node)

    visit(value, 0)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _Bounds:
    normalizers: tuple[dict[str, Any], ...]
    pretokenizers: tuple[dict[str, Any], ...]
    model_multiplier: int
    search_width: int

    def normalize(self, size: int, ascii_only: bool, pieces: int, job: _Job) -> tuple[int, bool]:
        for node in self.normalizers:
            job.checkpoint()
            kind = node["type"]
            if kind in {"NFC", "NFD", "NFKC", "NFKD"}:
                # Unicode compatibility decomposition has at most18 scalars per
                # input scalar. 72bytes/scalar also covers recomposition forms.
                size *= 1 if ascii_only else 72
            elif kind == "Lowercase":
                size *= 1 if ascii_only else 4
            elif kind == "BertNormalizer" and not ascii_only:
                size *= 3 if node.get("handle_chinese_chars", True) else 1
                size *= 4 if node.get("lowercase", True) else 1
                if node.get("strip_accents", node.get("lowercase", True)) is not False:
                    size *= 72
            elif kind == "Prepend":
                content = node.get("prepend", "")
                size += len(content.encode()) * pieces
                ascii_only &= content.isascii()
            elif kind == "Replace":
                content = node.get("content", "")
                # An empty pattern can match at both ends of every scalar.
                size += (size + pieces) * len(content.encode())
                ascii_only &= content.isascii()
            _bound(size)
        return size, ascii_only

    def admit(self, text: str, job: _Job) -> None:
        size = len(text.encode("utf-8"))
        # Added-token extraction can divide the input into at most2n+1 pieces;
        # each nonempty added token consumes at least one original scalar.
        pieces = 2 * size + 1
        size, _ = self.normalize(size, text.isascii(), pieces, job)
        for node in self.pretokenizers:
            job.checkpoint()
            if node["type"] == "ByteLevel":
                size = 2 * (size + pieces)
            elif node["type"] == "Metaspace":
                size = 4 * (size + pieces)
            _bound(size)
            pieces = max(pieces, 2 * size + 1)
        _bound((size + pieces) * self.model_multiplier, _NATIVE_TOKENS)
        _bound(size * min(max(size, 1), self.search_width), _NATIVE_SEARCH)


def _model(value: Any) -> tuple[int, int]:
    if type(value) is not dict:
        _unavailable()
    kind = value.get("type")
    if type(kind) is not str:
        _unavailable()
    common = {"type", "vocab", "unk_token"}
    fields = {
        "WordLevel": common,
        "WordPiece": common | {"continuing_subword_prefix", "max_input_chars_per_word"},
        "BPE": common
        | {
            "dropout",
            "merges",
            "continuing_subword_prefix",
            "end_of_word_suffix",
            "fuse_unk",
            "byte_fallback",
            "ignore_merges",
        },
        "Unigram": {"type", "vocab", "unk_id", "byte_fallback"},
    }.get(kind)
    if fields is None:
        _unavailable()
    _shape(value, fields)
    if value.get("dropout") not in (None, 0, 0.0) or type(value.get("dropout")) is bool:
        _unavailable()
    vocab = value.get("vocab")
    if kind == "Unigram":
        if type(vocab) is not list or not 1 <= len(vocab) <= _NATIVE_TOKENS:
            _unavailable()
        words = []
        for row in vocab:
            if type(row) is not list or len(row) != 2 or type(row[1]) not in (int, float):
                _unavailable()
            words.append(_string(row[0]))
    else:
        if type(vocab) is not dict or not 1 <= len(vocab) <= _NATIVE_TOKENS:
            _unavailable()
        words = [_string(key) for key in vocab]
        if any(type(v) is not int or not 0 <= v < 1 << 32 for v in vocab.values()):
            _unavailable()
    multiplier = 1
    for key in ("continuing_subword_prefix", "end_of_word_suffix"):
        if value.get(key) is not None:
            multiplier += len(_string(value[key], maximum=64, empty=True).encode())
    if kind == "BPE":
        merges = value.get("merges")
        if type(merges) is not list or len(merges) > _NATIVE_TOKENS:
            _unavailable()
        prefix = value.get("continuing_subword_prefix") or ""
        for row in merges:
            if type(row) is str:
                row = row.split(" ")
            if type(row) is not list or len(row) != 2 or any(type(s) is not str for s in row):
                _unavailable()
            left, right = row
            if left not in vocab or right not in vocab or not right.startswith(prefix):
                _unavailable()
            if left + right[len(prefix) :] not in vocab:
                _unavailable()
    width = 1
    if kind == "Unigram":
        width = max(len(word.encode()) for word in words)
    elif kind == "WordPiece":
        width = value.get("max_input_chars_per_word", 100)
        if type(width) is not int or not 1 <= width <= 1024:
            _unavailable()
    return multiplier, width


@dataclass(frozen=True, slots=True)
class _LoadedTokenizer:
    tokenizer: Any
    bounds: _Bounds

    def encode(
        self, text: object, config: LocalTokenizerConfig, job: _Job
    ) -> LocalTokenizerEncoding:
        text = validate_local_text(text, config, job)
        self.bounds.admit(text, job)
        job.checkpoint()
        try:
            native = self.tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            _unavailable()
        job.checkpoint()
        ids = native.ids
        if len(ids) > config.limits.max_tokens:
            _limit()
        offsets, masks = native.offsets, native.special_tokens_mask
        if (
            len(ids) != len(offsets)
            or len(ids) != len(masks)
            or any(type(value) is not int or value not in (0, 1) for value in masks)
        ):
            raise LocalTokenizerInvalidOutputError(
                "Local tokenizer returned invalid encoding vectors"
            )
        return LocalTokenizerEncoding(
            tuple(ids),
            tuple(None if mask else offset for offset, mask in zip(offsets, masks, strict=True)),
            tuple(bool(mask) for mask in masks),
            len(text),
        )


def validate_local_text(text: object, config: LocalTokenizerConfig, job: _Job) -> str:
    """Reject raw invalid input before optional imports or artifact I/O are needed."""
    job.checkpoint()
    if type(text) is not str:
        raise LocalTokenizerInvalidInputError("Local tokenizer text must be a native string")
    if len(text) > config.limits.max_input_characters:
        _limit()
    try:
        size = len(text.encode("utf-8"))
    except UnicodeError:
        raise LocalTokenizerInvalidInputError("Local tokenizer text has invalid Unicode") from None
    if size > config.limits.max_input_bytes:
        _limit()
    return text


def load_local_tokenizer(config: LocalTokenizerConfig, job: _Job) -> _LoadedTokenizer:
    """No network, discovery, Python artifact code, or implicit model downloads."""
    raw = _read(config, job)
    try:
        text = raw.decode("utf-8")
        _json_preflight(text, job)
        document = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeError, ValueError, RecursionError):
        _unavailable()
    _graph(document, job)
    document = _shape(
        document,
        {
            "version",
            "truncation",
            "padding",
            "added_tokens",
            "normalizer",
            "pre_tokenizer",
            "post_processor",
            "decoder",
            "model",
        },
    )
    multiplier, width = _model(document.get("model"))
    bounds = _Bounds(
        _stages(document.get("normalizer"), normalizer=True),
        _stages(document.get("pre_tokenizer"), normalizer=False),
        multiplier,
        width,
    )
    added = document.get("added_tokens", [])
    if type(added) is not list or len(added) > 4096:
        _unavailable()
    added_bytes = 0
    added_ascii = True
    for row in added:
        _shape(row, {"id", "content", "single_word", "lstrip", "rstrip", "normalized", "special"})
        content = _string(row.get("content"))
        for name in ("single_word", "lstrip", "rstrip", "normalized", "special"):
            _boolean(row.get(name))
        if type(row.get("id")) is not int or not 0 <= row["id"] < 1 << 32:
            _unavailable()
        added_bytes += len(content.encode())
        added_ascii &= content.isascii()
    _bound(added_bytes, _NATIVE_TOKENS)
    # Added tokens are normalized while the native matcher is constructed.
    bounds.normalize(added_bytes, added_ascii, len(added), job)
    for key in ("truncation", "padding", "post_processor", "decoder"):
        document[key] = None
    job.checkpoint()
    try:
        import tokenizers

        if tokenizers.__version__ != "0.22.2":
            _unavailable()
        native = tokenizers.Tokenizer.from_str(json.dumps(document, ensure_ascii=False))
        if document["model"]["type"] in {"BPE", "Unigram"}:
            # Version-pinned native hook: no request text/derived word cache may
            # outlive a job. The tokenizer/immutable vocabulary can be retained.
            native.model._resize_cache(0)
        native.no_padding()
        native.no_truncation()
    except Exception:
        _unavailable()
    job.checkpoint()
    return _LoadedTokenizer(native, bounds)
