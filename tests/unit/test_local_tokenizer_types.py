# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Pinned native tokenizer values validate without opening files or optional imports."""

import builtins
import json
from dataclasses import replace

import pytest

from quarry.types.local_tokenizer import (
    LocalTokenizerConfig,
    LocalTokenizerEncoding,
    LocalTokenizerLimits,
    LocalTokenizerValidationError,
    TokenFramingMargins,
)


def config(**kwargs):
    values = dict(
        artifact_path="/synthetic/not-opened/tokenizer.json",
        artifact_sha256="a" * 64,
        model_revision="synthetic-model-r1",
        tokenizer_revision="synthetic-tokenizer-r1",
    )
    return LocalTokenizerConfig(**{**values, **kwargs})


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, float("nan"), float("inf"), 16_777_217])
def test_artifact_bounds_reject_non_native_or_unbounded_values(value):
    with pytest.raises(LocalTokenizerValidationError):
        LocalTokenizerLimits(max_artifact_bytes=value)


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), 30.1, "2"])
def test_worker_wall_allowance_is_finite_and_strict(value):
    with pytest.raises(LocalTokenizerValidationError):
        LocalTokenizerLimits(max_wall_seconds=value)


@pytest.mark.parametrize(
    "path", ["", "https://example.invalid/tokenizer.json", "bad\x00path", object()]
)
def test_native_artifact_path_is_explicit_and_local(path):
    with pytest.raises(LocalTokenizerValidationError):
        config(artifact_path=path)


@pytest.mark.parametrize("digest", ["", "a" * 63, "a" * 65, "G" * 64, True, object()])
def test_artifact_identity_requires_an_exact_native_digest(digest):
    with pytest.raises(LocalTokenizerValidationError):
        config(artifact_sha256=digest)


def test_native_configuration_defers_files_and_optional_imports(monkeypatch):
    imported = builtins.__import__

    def restricted(name, *args, **kwargs):
        if name.split(".")[0] in {"jinja2", "tokenizers"}:
            pytest.fail("configuration eagerly imported an optional backend")
        return imported(name, *args, **kwargs)

    def opened(*args, **kwargs):
        pytest.fail("configuration opened an artifact")

    monkeypatch.setattr(builtins, "__import__", restricted)
    monkeypatch.setattr(builtins, "open", opened)
    value = config(template="synthetic template")
    assert value.identity_dict()["padding"] is False
    assert value.identity_dict()["truncation"] is False


def test_identity_hides_paths_and_template_plaintext_but_binds_semantics():
    first = config(template="synthetic private template")
    moved = replace(first, artifact_path="/synthetic/another/tokenizer.json")
    assert first.identity_dict() == moved.identity_dict()
    encoded = json.dumps(first.identity_dict())
    assert "private template" not in encoded and "/synthetic/" not in encoded
    for changed in [
        replace(first, template="synthetic other template"),
        replace(first, artifact_sha256="b" * 64),
        replace(first, model_revision="synthetic-model-r2"),
        replace(first, tokenizer_revision="synthetic-tokenizer-r2"),
        replace(first, framing=TokenFramingMargins(1, 2)),
    ]:
        assert changed.identity_dict() != first.identity_dict()


def test_forged_limits_are_revalidated_and_native_values_are_detached():
    limits = LocalTokenizerLimits()
    framing = TokenFramingMargins(1, 2)
    checked = config(limits=limits, framing=framing)
    assert checked.limits == limits and checked.limits is not limits
    assert checked.framing == framing and checked.framing is not framing
    object.__setattr__(limits, "max_tokens", True)
    with pytest.raises(LocalTokenizerValidationError):
        config(limits=limits)


def test_framing_is_charged_once_per_request_and_once_per_message():
    margins = TokenFramingMargins(per_request=7, per_message=3)
    assert margins.total(0) == 7
    assert margins.total(2) == 13
    for value in (True, -1, 257, 2.0):
        with pytest.raises(LocalTokenizerValidationError):
            margins.total(value)


def test_unicode_geometry_keeps_original_codepoints_overlaps_and_special_tokens():
    text = "Aé👩\u200d💻x"
    offsets = [None, [0, 1], [1, 2], [2, 5], [2, 5], [5, 6]]
    ids = [99, 1, 2, 3, 4, 5]
    mask = [True, False, False, False, False, False]
    value = LocalTokenizerEncoding(ids, offsets, mask, len(text))
    offsets[1][0] = 6
    ids.clear()
    mask.clear()
    assert value.token_ids == (99, 1, 2, 3, 4, 5)
    assert value.character_offsets[1] == (0, 1)
    assert text[slice(*value.character_offsets[3])] == "👩\u200d💻"
    assert value.character_offsets[0] is None
    assert LocalTokenizerEncoding((), (), (), 0).token_ids == ()


@pytest.mark.parametrize(
    "ids,offsets,mask,characters",
    [
        ([True], [(0, 1)], [False], 1),
        ([-1], [(0, 1)], [False], 1),
        ([1 << 32], [(0, 1)], [False], 1),
        ([1], [None], [False], 1),
        ([1], [(0, 0)], [True], 1),
        ([1], [(0, 2)], [False], 1),
        ([1], [(True, 1)], [False], 1),
        ([1], [(0, 1)], [1], 1),
        ([1], [], [False], 1),
        ([1, 2], [(2, 3), (1, 2)], [False, False], 3),
        ([1], [(0, 1)], [False], True),
        ([1] * 262_145, [None], [True], 0),
    ],
)
def test_unbound_non_native_or_fabricated_geometry_is_rejected(ids, offsets, mask, characters):
    with pytest.raises(LocalTokenizerValidationError):
        LocalTokenizerEncoding(ids, offsets, mask, characters)
