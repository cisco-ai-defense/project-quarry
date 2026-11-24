# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Claims require attestation; capped values cannot imply invented exact counts."""

from dataclasses import FrozenInstanceError, replace

import pytest

from quarry.types.tokenizer_width import (
    CappedTokenCount,
    TokenizerWidthClaim,
    TokenizerWidthValidationError,
)


def claim(**changes):
    return TokenizerWidthClaim(
        **{
            "max_source_characters_per_token": 8,
            "artifact_sha256": "a" * 64,
            "model_revision": "synthetic-model-r1",
            "tokenizer_revision": "synthetic-tokenizer-r1",
            "count_settings_sha256": "b" * 64,
            **changes,
        }
    )


@pytest.mark.parametrize(
    "width", [0, -1, True, False, 1.0, float("inf"), float("nan"), "8", None, 1 << 10000, 1_048_577]
)
def test_width_is_a_finite_positive_bounded_native_integer(width):
    with pytest.raises(TokenizerWidthValidationError):
        claim(max_source_characters_per_token=width)


@pytest.mark.parametrize("field", ["artifact_sha256", "count_settings_sha256"])
@pytest.mark.parametrize("digest", [None, False, "", "a" * 63, "A" * 64, "g" * 64])
def test_digests_require_exact_native_lowercase_sha256(field, digest):
    with pytest.raises(TokenizerWidthValidationError):
        claim(**{field: digest})


@pytest.mark.parametrize("field", ["model_revision", "tokenizer_revision"])
@pytest.mark.parametrize("revision", [None, True, "", "x" * 257, "bad\x00label"])
def test_revisions_are_bounded_native_printable_strings(field, revision):
    with pytest.raises(TokenizerWidthValidationError):
        claim(**{field: revision})


def test_caller_boolean_cannot_establish_attestation_and_unknown_method_is_rejected():
    value = claim()
    assert not hasattr(value, "verified")
    fields = {name: getattr(value, name) for name in value.__dataclass_fields__}
    with pytest.raises(TypeError):
        TokenizerWidthClaim(**fields, verified=True)
    with pytest.raises(TokenizerWidthValidationError):
        claim(method="untrusted.callback")


def test_claim_construction_opens_no_file_and_imports_no_optional_backend(monkeypatch):
    import builtins
    from pathlib import Path

    original_import = builtins.__import__

    def forbidden(*args, **kwargs):
        pytest.fail("claim construction opened a file or loaded an optional backend")

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"tokenizers", "jinja2"}:
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert len(claim().fingerprint) == 64


def test_claim_identity_names_each_revision_setting_and_width_without_runtime_io():
    original = claim()
    for changed in (
        replace(original, max_source_characters_per_token=9),
        replace(original, artifact_sha256="c" * 64),
        replace(original, model_revision="different-model-r2"),
        replace(original, tokenizer_revision="different-tokenizer-r2"),
        replace(original, count_settings_sha256="d" * 64),
    ):
        assert changed.fingerprint != original.fingerprint
    with pytest.raises(FrozenInstanceError):
        original.max_source_characters_per_token = 1


@pytest.mark.parametrize(
    "count,cap,exceeded", [(0, 0, False), (9, 10, False), (10, 10, False), (11, 10, True)]
)
def test_canonical_exact_counts_keep_their_value_on_both_sides_of_the_cap(count, cap, exceeded):
    value = CappedTokenCount(cap, count, exact_count=count)
    assert value.is_exact and value.exact_count == count
    assert value.exceeds_cap is exceeded and value.proof_fingerprint is None


def test_overflow_never_fabricates_an_exact_count():
    value = CappedTokenCount(10, 11, proof_fingerprint="c" * 64)
    assert value.exceeds_cap and not value.is_exact and value.exact_count is None
    with pytest.raises(FrozenInstanceError):
        value.exact_count = 11


@pytest.mark.parametrize(
    "fields",
    [
        {"cap": True, "lower_bound": 3, "exact_count": 3},
        {"cap": -1, "lower_bound": 3, "exact_count": 3},
        {"cap": 1 << 10000, "lower_bound": 3, "exact_count": 3},
        {"cap": 10, "lower_bound": 9, "exact_count": True},
        {"cap": 10, "lower_bound": 9, "exact_count": 8},
        {"cap": 10, "lower_bound": 11},
        {"cap": 10, "lower_bound": 10, "proof_fingerprint": "a" * 64},
        {"cap": 10, "lower_bound": 11, "proof_fingerprint": "unverified"},
        {"cap": 10, "lower_bound": 11, "exact_count": 11, "proof_fingerprint": "a" * 64},
        {"cap": 10, "lower_bound": True, "exact_count": 1},
    ],
)
def test_invalid_unknown_or_mislabelled_count_shapes_cannot_cross_the_boundary(fields):
    with pytest.raises(TokenizerWidthValidationError):
        CappedTokenCount(**fields)
