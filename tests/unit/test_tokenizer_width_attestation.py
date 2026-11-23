# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Actual native proofs reject lossy pipelines, ID aliasing and mismatched state."""

import json
from dataclasses import replace
from time import monotonic

import pytest
from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers, processors

from quarry.integrations.local_tokenizer_artifact import load_local_tokenizer
from quarry.integrations.local_tokenizer_worker import _Abandoned, _Expired, _Job
from quarry.integrations.tokenizer_width import VerifiedTokenizerWidth, attest_loaded_width
from quarry.types.tokenizer_width import TokenizerWidthClaim
from tests.unit.test_local_tokenizer_artifact import artifact


def job():
    return _Job(monotonic() + 5)


def bpe(*, missing=False, collision=False):
    vocab = {char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    merges = []
    if missing:
        absent_id = vocab.pop("x")
        # Keep IDs dense: cardinality/density cannot prove alphabet coverage.
        vocab["unreachable"] = absent_id
    elif collision:
        vocab["aa"] = vocab["a"]
        merges = [("a", "a")]
    else:
        vocab["aa"], vocab["aaaa"] = 256, 257
        merges = [("a", "a"), ("aa", "aa")]
    tokenizer = Tokenizer(models.BPE(vocab, merges))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, trim_offsets=False, use_regex=False
    )
    return tokenizer


def claim(config, width=4, **changes):
    return TokenizerWidthClaim(
        **{
            "max_source_characters_per_token": width,
            "artifact_sha256": config.artifact_sha256,
            "model_revision": config.model_revision,
            "tokenizer_revision": config.tokenizer_revision,
            "count_settings_sha256": config.count_settings_sha256,
            **changes,
        }
    )


@pytest.mark.parametrize(
    "prefix,regex", [(False, False), (False, True), (True, False), (True, True)]
)
def test_complete_native_bpe_attestation_bounds_fresh_ascii_unicode_and_whitespace(
    tmp_path, prefix, regex
):
    reference = bpe()
    reference.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=prefix, trim_offsets=True, use_regex=regex
    )
    cfg = artifact(tmp_path, reference)
    loaded = load_local_tokenizer(cfg, job())
    requested = claim(cfg)
    proof = attest_loaded_width(cfg, requested, loaded, job())
    assert type(proof) is VerifiedTokenizerWidth
    assert proof.attested_max_source_characters == 4
    assert proof.is_compatible(cfg, requested)
    for text in (
        "a" * 1024,
        "é e\u0301 👩\u200d💻" * 10,
        " " * 100,
        "\x00\n\r\t",
        "",
        "aaaaé" * 40,
    ):
        canonical = loaded.encode(text, cfg, job())
        lower_bound = (
            len(text) + requested.max_source_characters_per_token - 1
        ) // requested.max_source_characters_per_token
        assert lower_bound <= len(canonical.token_ids)


def test_added_literals_extend_the_attested_source_width(tmp_path):
    reference = bpe()
    reference.add_special_tokens([AddedToken("[LONG_TOKEN]", normalized=False, special=True)])
    cfg = artifact(tmp_path, reference)
    loaded = load_local_tokenizer(cfg, job())
    assert attest_loaded_width(cfg, claim(cfg, width=4), loaded, job()) is None
    requested = claim(cfg, width=len("[LONG_TOKEN]"))
    proof = attest_loaded_width(cfg, requested, loaded, job())
    assert proof.is_compatible(cfg, requested)
    assert proof.attested_max_source_characters == len("[LONG_TOKEN]")
    assert len(loaded.encode("[LONG_TOKEN]", cfg, job()).token_ids) == 1


@pytest.mark.parametrize("side", ["lstrip", "rstrip"])
def test_added_whitespace_consumption_cannot_authorize_a_width_proof(tmp_path, side):
    reference = bpe()
    reference.add_tokens([AddedToken("X", **{side: True})])
    cfg = artifact(tmp_path, reference)
    loaded = load_local_tokenizer(cfg, job())
    text = " " * 100 + "X" if side == "lstrip" else "X" + " " * 100
    assert len(loaded.encode(text, cfg, job()).token_ids) == 1
    assert attest_loaded_width(cfg, claim(cfg), loaded, job()) is None


def test_lost_bytes_and_fused_unknowns_require_model_alphabet_coverage(tmp_path):
    reference = bpe(missing=True)
    cfg = artifact(tmp_path, reference)
    loaded = load_local_tokenizer(cfg, job())
    assert len(loaded.tokenizer.get_vocab(with_added_tokens=False)) == 256
    assert loaded.encode("x" * 128, cfg, job()).token_ids == ()
    assert attest_loaded_width(cfg, claim(cfg, width=16), loaded, job()) is None


def test_sparse_native_ids_reject_before_unbounded_reverse_vocab_serialization(
    tmp_path, monkeypatch
):
    reference = bpe()
    document = json.loads(reference.to_str())
    document["model"]["vocab"]["a"] = 1_000_000_000
    cfg = artifact(tmp_path, document=document)
    loaded = load_local_tokenizer(cfg, job())
    assert loaded.encode("a", cfg, job()).token_ids == (1_000_000_000,)

    def forbidden(*args, **kwargs):
        raise AssertionError("native serializer reached sparse vocabulary IDs")

    monkeypatch.setattr(Tokenizer, "to_str", forbidden)
    assert attest_loaded_width(cfg, claim(cfg), loaded, job()) is None


def test_colliding_ids_can_compress_an_unbounded_run_beyond_declared_word_lengths(tmp_path):
    reference = bpe(collision=True)
    # Preserve the original colliding model map in the input JSON. Serializing
    # through the native reverse-ID map first can erase one of the colliding keys.
    document = json.loads(reference.to_str())
    document["model"]["vocab"] = reference.get_vocab(with_added_tokens=False)
    document["model"]["merges"] = [["a", "a"]]
    cfg = artifact(tmp_path, document=document)
    loaded = load_local_tokenizer(cfg, job())
    assert len(loaded.encode("a" * 256, cfg, job()).token_ids) == 1
    assert attest_loaded_width(cfg, claim(cfg, width=2), loaded, job()) is None


@pytest.mark.parametrize(
    "change",
    ["normalization", "destructive_split", "truncation", "padding", "postprocessor", "decoder"],
)
def test_unmodeled_or_lossy_native_state_cannot_return_attestation(tmp_path, change):
    reference = bpe()
    if change == "normalization":
        reference.normalizer = normalizers.Replace("x", "")
    elif change == "destructive_split":
        reference.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    cfg = artifact(tmp_path, reference)
    loaded = load_local_tokenizer(cfg, job())
    # Change actual native state after canonical construction to test the actual
    # state guard, rather than configuration booleans alone.
    if change == "truncation":
        loaded.tokenizer.enable_truncation(max_length=1)
    elif change == "padding":
        loaded.tokenizer.enable_padding(length=100)
    elif change == "postprocessor":
        loaded.tokenizer.post_processor = processors.TemplateProcessing(single="$A $A")
    elif change == "decoder":
        from tokenizers import decoders

        loaded.tokenizer.decoder = decoders.ByteLevel()
    assert attest_loaded_width(cfg, claim(cfg), loaded, job()) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("artifact_sha256", "0" * 64),
        ("model_revision", "different"),
        ("tokenizer_revision", "different"),
        ("count_settings_sha256", "0" * 64),
        ("max_source_characters_per_token", 1),
    ],
)
def test_wrong_bound_or_binding_is_unproven(tmp_path, field, value):
    cfg = artifact(tmp_path, bpe())
    loaded = load_local_tokenizer(cfg, job())
    assert attest_loaded_width(cfg, replace(claim(cfg), **{field: value}), loaded, job()) is None


def test_loaded_artifact_and_proof_remain_bound_to_actual_full_count_settings(tmp_path):
    cfg = artifact(tmp_path, bpe())
    loaded = load_local_tokenizer(cfg, job())
    first = claim(cfg)
    proof = attest_loaded_width(cfg, first, loaded, job())
    other = replace(cfg, template="different whole request")
    assert attest_loaded_width(other, claim(other), loaded, job()) is None
    assert not proof.is_compatible(other, claim(other))
    forged = replace(first)
    object.__setattr__(forged, "max_source_characters_per_token", True)
    assert not proof.is_compatible(cfg, forged)
    assert proof.is_compatible(cfg, first)
    with pytest.raises(TypeError):
        VerifiedTokenizerWidth()
    assert not hasattr(proof, "tokenizer") and not hasattr(proof, "source_text")


def test_attestation_checks_actual_worker_cancellation_and_deadline(tmp_path):
    cfg = artifact(tmp_path, bpe())
    loaded = load_local_tokenizer(cfg, job())
    with pytest.raises(_Expired):
        attest_loaded_width(cfg, claim(cfg), loaded, _Job(monotonic() - 1))
    cancelled = job()
    cancelled.cancelled.set()
    with pytest.raises(_Abandoned):
        attest_loaded_width(cfg, claim(cfg), loaded, cancelled)
