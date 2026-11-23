# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Fresh synthetic artifacts, exact IDs/Unicode geometry, and pre-native limits."""

import hashlib
import json
import os
import sys
from dataclasses import replace
from time import monotonic
from types import SimpleNamespace

import pytest
from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers, processors

from quarry.integrations.local_tokenizer_artifact import load_local_tokenizer
from quarry.integrations.local_tokenizer_worker import _Abandoned, _Expired, _Job
from quarry.types.exceptions import (
    LocalTokenizerInvalidInputError,
    LocalTokenizerLimitError,
    LocalTokenizerUnavailableError,
)
from quarry.types.local_tokenizer import LocalTokenizerLimits
from tests.unit.test_local_tokenizer_types import config


def job():
    return _Job(monotonic() + 5)


def synthetic(kind="WordLevel"):
    vocab = {"[UNK]": 0, "hello": 1, "world": 2, "é": 3, "👩": 4, "💻": 5, "e": 6, "\u0301": 7}
    if kind == "WordLevel":
        model = models.WordLevel(vocab, unk_token="[UNK]")
    elif kind == "WordPiece":
        model = models.WordPiece(vocab, unk_token="[UNK]")
    elif kind == "BPE":
        model = models.BPE(vocab, [], unk_token="[UNK]")
    else:
        model = models.Unigram([(key, -1.0) for key in vocab], unk_id=0)
    result = Tokenizer(model)
    result.pre_tokenizer = pre_tokenizers.Whitespace()
    return result


def artifact(tmp_path, tokenizer=None, *, document=None, raw=None, options=None):
    if raw is None:
        raw = (
            json.dumps(document).encode()
            if document is not None
            else (tokenizer or synthetic()).to_str().encode()
        )
    path = tmp_path / "synthetic-tokenizer.json"
    path.write_bytes(raw)
    return config(
        artifact_path=str(path),
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
        **(options or {}),
    )


@pytest.mark.parametrize("kind", ["WordLevel", "WordPiece", "BPE", "Unigram"])
@pytest.mark.parametrize("text", ["hello world", "é e\u0301 👩\u200d💻", "", "unknown text"])
def test_fresh_native_artifacts_have_exact_ids_and_code_point_offsets(tmp_path, kind, text):
    reference = synthetic(kind)
    cfg = artifact(tmp_path, reference)
    loaded = load_local_tokenizer(cfg, job())
    actual = loaded.encode(text, cfg, job())
    expected = reference.encode(text, add_special_tokens=False)
    assert actual.token_ids == tuple(expected.ids)
    assert actual.character_offsets == tuple(expected.offsets)
    assert actual.special_tokens_mask == tuple(False for _ in expected.ids)
    assert actual.source_characters == len(text)
    assert all(end <= len(text) for start, end in actual.character_offsets)


def test_padding_truncation_and_repeated_postprocessor_cannot_change_explicit_text_count(tmp_path):
    reference = synthetic()
    reference.post_processor = processors.TemplateProcessing(
        single="[UNK] $A $A [UNK]", special_tokens=[("[UNK]", 0)]
    )
    reference.enable_truncation(max_length=1)
    reference.enable_padding(length=32)
    document = json.loads(reference.to_str())
    # These components are never instantiated, even if the artifact contains an
    # unsupported decoder or an adversarial repeated sequence post-processor.
    document["decoder"] = {"type": "UnsupportedUntrustedDecoder", "filename": "no-read"}
    cfg = artifact(tmp_path, document=document)
    loaded = load_local_tokenizer(cfg, job())
    assert loaded.tokenizer.padding is None and loaded.tokenizer.truncation is None
    assert loaded.tokenizer.post_processor is None and loaded.tokenizer.decoder is None
    assert loaded.encode("hello world hello", cfg, job()).token_ids == (1, 2, 1)


@pytest.mark.parametrize(
    "normalizer",
    [
        normalizers.Lowercase(),
        normalizers.NFC(),
        normalizers.NFD(),
        normalizers.NFKC(),
        normalizers.NFKD(),
        normalizers.BertNormalizer(),
        normalizers.Strip(),
        normalizers.StripAccents(),
        normalizers.Prepend("hello "),
        normalizers.Replace("unknown", "hello"),
        normalizers.Sequence([normalizers.Lowercase(), normalizers.Replace("hello", "world")]),
    ],
)
def test_supported_normalization_preserves_native_original_text_geometry(tmp_path, normalizer):
    reference = synthetic()
    reference.normalizer = normalizer
    cfg = artifact(tmp_path, reference)
    text = "  HELLO é e\u0301  "
    actual = load_local_tokenizer(cfg, job()).encode(text, cfg, job())
    expected = reference.encode(text, add_special_tokens=False)
    assert actual.token_ids == tuple(expected.ids)
    assert actual.character_offsets == tuple(expected.offsets)


@pytest.mark.parametrize(
    "pretokenizer",
    [
        pre_tokenizers.ByteLevel(),
        pre_tokenizers.Metaspace(),
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.BertPreTokenizer(),
        pre_tokenizers.UnicodeScripts(),
        pre_tokenizers.Digits(),
        pre_tokenizers.Punctuation(),
        pre_tokenizers.CharDelimiterSplit("|"),
        pre_tokenizers.Split("|", behavior="isolated"),
        pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.ByteLevel()]),
    ],
)
def test_supported_pretokenizers_preserve_native_ids_and_offsets(tmp_path, pretokenizer):
    reference = synthetic()
    reference.pre_tokenizer = pretokenizer
    cfg = artifact(tmp_path, reference)
    text = "hello|世界 é 123"
    actual = load_local_tokenizer(cfg, job()).encode(text, cfg, job())
    expected = reference.encode(text, add_special_tokens=False)
    assert actual.token_ids == tuple(expected.ids)
    assert actual.character_offsets == tuple(expected.offsets)


def test_explicit_added_markers_are_encoded_once_without_implicit_endpoints(tmp_path):
    reference = synthetic()
    reference.add_special_tokens([AddedToken("[BEGIN]", normalized=False, special=True)])
    cfg = artifact(tmp_path, reference)
    result = load_local_tokenizer(cfg, job()).encode("[BEGIN]hello", cfg, job())
    assert result.token_ids == tuple(reference.encode("[BEGIN]hello", add_special_tokens=False).ids)
    assert len(result.token_ids) == 2
    # An explicitly supplied marker is real source text, not an inserted endpoint.
    assert result.character_offsets[0] == (0, 7)


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"{}",
        b"\xff",
        b'{"model":1,"model":2}',
        b'{"value":NaN}',
        b'{"value":1e999}',
        b'{"value":"\\ud800"}',
    ],
)
def test_corrupt_artifacts_are_typed_unavailable(tmp_path, raw):
    with pytest.raises(LocalTokenizerUnavailableError):
        load_local_tokenizer(artifact(tmp_path, raw=raw), job())


def test_digest_mismatch_and_unreadable_path_cannot_fall_back_or_download(tmp_path, monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        pytest.fail("artifact loading attempted network I/O")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    cfg = artifact(tmp_path)
    with pytest.raises(LocalTokenizerUnavailableError):
        load_local_tokenizer(replace(cfg, artifact_sha256="0" * 64), job())
    with pytest.raises(LocalTokenizerUnavailableError):
        load_local_tokenizer(replace(cfg, artifact_path=str(tmp_path / "missing")), job())
    assert load_local_tokenizer(cfg, job()).encode("hello", cfg, job()).token_ids == (1,)


def test_nonregular_and_oversized_artifacts_are_rejected_before_native_import(
    tmp_path, monkeypatch
):
    cfg = artifact(tmp_path)
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    with pytest.raises(LocalTokenizerUnavailableError):
        load_local_tokenizer(replace(cfg, artifact_path=str(tmp_path)), job())
    with pytest.raises(LocalTokenizerLimitError):
        load_local_tokenizer(
            replace(cfg, limits=LocalTokenizerLimits(max_artifact_bytes=16)), job()
        )
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)
        with pytest.raises(LocalTokenizerUnavailableError):
            load_local_tokenizer(replace(cfg, artifact_path=str(fifo)), job())


@pytest.mark.parametrize(
    "component", ["regex", "precompiled", "dropout", "unknown_model", "bad_merge"]
)
def test_unbounded_unknown_or_nondeterministic_components_never_reach_native_parser(
    tmp_path, monkeypatch, component
):
    document = json.loads(synthetic("BPE").to_str())
    if component == "regex":
        document["normalizer"] = {"type": "Replace", "pattern": {"Regex": "(a+)+$"}, "content": "x"}
    elif component == "precompiled":
        document["normalizer"] = {"type": "Precompiled", "precompiled_charsmap": [0] * 8}
    elif component == "dropout":
        document["model"]["dropout"] = 0.1
    elif component == "unknown_model":
        document["model"]["type"] = "Unknown"
    else:
        document["model"]["continuing_subword_prefix"] = "💻"
        document["model"]["merges"] = [["hello", "é"]]
    cfg = artifact(tmp_path, document=document)

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported artifact reached native parser")

    monkeypatch.setitem(
        sys.modules,
        "tokenizers",
        SimpleNamespace(__version__="0.22.2", Tokenizer=SimpleNamespace(from_str=forbidden)),
    )
    with pytest.raises(LocalTokenizerUnavailableError):
        load_local_tokenizer(cfg, job())


def test_added_token_normalization_expansion_is_rejected_before_native_construction(
    tmp_path, monkeypatch
):
    document = json.loads(synthetic().to_str())
    document["normalizer"] = {
        "type": "Sequence",
        "normalizers": [{"type": "Replace", "pattern": {"String": "x"}, "content": "x" * 4096}] * 4,
    }
    document["added_tokens"] = [
        {
            "id": 99,
            "content": "x",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": True,
            "special": False,
        }
    ]
    cfg = artifact(tmp_path, document=document)
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    with pytest.raises(LocalTokenizerLimitError):
        load_local_tokenizer(cfg, job())


@pytest.mark.parametrize("expansion", ["replace", "prepend", "bytelevel", "unigram_search"])
def test_expansion_and_search_are_rejected_before_native_encoding(tmp_path, expansion):
    reference = synthetic("Unigram" if expansion == "unigram_search" else "WordLevel")
    if expansion == "replace":
        reference.normalizer = normalizers.Sequence([normalizers.Replace("x", "x" * 256)] * 4)
    elif expansion == "prepend":
        reference.normalizer = normalizers.Sequence([normalizers.Prepend("x" * 4096)] * 4)
    elif expansion == "bytelevel":
        reference.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.ByteLevel()] * 8)
    else:
        reference = Tokenizer(models.Unigram([("[UNK]", 0.0), ("x" * 4096, -1.0)], unk_id=0))
    cfg = artifact(tmp_path, reference)
    loaded = load_local_tokenizer(cfg, job())

    def forbidden(*args, **kwargs):
        pytest.fail("unbounded text reached native encoding")

    loaded = replace(loaded, tokenizer=SimpleNamespace(encode=forbidden))
    with pytest.raises(LocalTokenizerLimitError):
        loaded.encode("x" * 4096, cfg, job())


def test_input_output_caps_and_missing_optional_package_are_typed(tmp_path, monkeypatch):
    cfg = artifact(tmp_path)
    loaded = load_local_tokenizer(cfg, job())
    with pytest.raises(LocalTokenizerInvalidInputError):
        loaded.encode(object(), cfg, job())
    with pytest.raises(LocalTokenizerInvalidInputError):
        loaded.encode("\ud800", cfg, job())
    with pytest.raises(LocalTokenizerLimitError):
        loaded.encode("hello world", replace(cfg, limits=LocalTokenizerLimits(max_tokens=1)), job())
    with pytest.raises(LocalTokenizerLimitError):
        loaded.encode(
            "hello", replace(cfg, limits=LocalTokenizerLimits(max_input_characters=4)), job()
        )
    with pytest.raises(LocalTokenizerLimitError):
        loaded.encode("ééé", replace(cfg, limits=LocalTokenizerLimits(max_input_bytes=4)), job())
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    with pytest.raises(LocalTokenizerUnavailableError):
        load_local_tokenizer(cfg, job())


def test_json_depth_limit_and_native_job_checkpoints_apply_before_loading(tmp_path):
    cfg = artifact(tmp_path, raw=b"[" * 1000 + b"0" + b"]" * 1000)
    with pytest.raises(LocalTokenizerLimitError):
        load_local_tokenizer(cfg, job())
    with pytest.raises(_Expired):
        load_local_tokenizer(cfg, _Job(monotonic() - 1))
    abandoned = job()
    abandoned.cancelled.set()
    with pytest.raises(_Abandoned):
        load_local_tokenizer(cfg, abandoned)
