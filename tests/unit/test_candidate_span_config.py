# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Explicit candidate capability bindings, before any provider is registered or invoked."""

from dataclasses import replace

import pytest

from quarry.cache.keys import input_key
from quarry.config import CandidateSpanConfig, QuarryConfig
from quarry.types.candidates import CandidateBatchLimits
from quarry.types.exceptions import ConfigurationError


def mapping(**spans):
    return {
        "config_version": 2,
        "models": {
            "reader": {
                "adapter": "registered",
                "model": "synthetic-candidates",
                "revision": "1",
                "category": "SYNTHETIC",
                "spans": {
                    "backend": "candidates",
                    "batch": "heads",
                    "channel": "left",
                    "settings_digest": "0" * 64,
                    **spans,
                },
            }
        },
    }


def configuration(tmp_path, **spans):
    return QuarryConfig.from_mapping(mapping(**spans), tmp_path, require_rules=False)


def test_candidate_reader_is_an_explicit_detached_capability_binding(tmp_path):
    doc = mapping(limits={"max_work": 1234})
    cfg = QuarryConfig.from_mapping(doc, tmp_path, require_rules=False)
    doc["models"]["reader"]["spans"]["limits"]["max_work"] = 1
    section = cfg.models["reader"].spans
    assert type(section) is CandidateSpanConfig
    assert section.batch == "heads" and section.channel == "left"
    assert section.settings_digest == "0" * 64
    assert section.limits.max_work == 1234
    assert section.max_input_tokens is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch", ""),
        ("batch", "../unregistered"),
        ("channel", True),
        ("channel", "x\n"),
        ("settings_digest", "0" * 63),
        ("settings_digest", "G" * 64),
        ("settings_digest", None),
        ("limits", None),
        ("limits", {"unknown": 1}),
        ("limits", {"max_work": 0}),
        ("limits", {"max_work": True}),
        ("limits", {"max_parts": 4097}),
        ("limits", {"max_ranges_per_channel": 4097}),
        ("limits", {"max_coverage_per_channel": -1}),
        ("limits", {"max_input_characters": 4_000_001}),
        ("renderer", "join"),
        ("class_index", 0),
        ("callback", "not.imported"),
    ],
)
def test_invalid_candidate_reader_configuration_is_refused_before_runtime(tmp_path, field, value):
    with pytest.raises((ConfigurationError, ValueError)):
        configuration(tmp_path, **{field: value})


@pytest.mark.parametrize("field", ["batch", "channel", "settings_digest"])
def test_candidate_configuration_cannot_omit_a_binding_field(tmp_path, field):
    doc = mapping()
    del doc["models"]["reader"]["spans"][field]
    with pytest.raises((ConfigurationError, ValueError)):
        QuarryConfig.from_mapping(doc, tmp_path, require_rules=False)


@pytest.mark.parametrize("change", ["revision", "namespace", "windows"])
def test_candidate_models_require_explicit_provider_identity_without_window_joining(
    tmp_path, change
):
    doc = mapping()
    model = doc["models"]["reader"]
    if change == "revision":
        del model["revision"]
    elif change == "namespace":
        model["model"] = "not a capability namespace"
    else:
        model["windows"] = {"unit": "characters", "size": 4}
    with pytest.raises((ConfigurationError, ValueError)):
        QuarryConfig.from_mapping(doc, tmp_path, require_rules=False)


@pytest.mark.parametrize(
    "changes",
    [
        {"channel": "right"},
        {"batch": "other"},
        {"settings_digest": "1" * 64},
        {"limits": CandidateBatchLimits(max_work=123)},
    ],
)
def test_every_reader_and_limit_change_remains_part_of_policy_identity(tmp_path, changes):
    model = configuration(tmp_path).models["reader"]
    assert input_key(replace(model, spans=replace(model.spans, **changes))) != input_key(model)


def test_unchanged_model_identity_has_no_candidate_fields_or_absent_options(tmp_path):
    cfg = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"plain": {"adapter": "synthetic", "model": "synthetic"}}},
        tmp_path,
        require_rules=False,
    )
    assert cfg.models["plain"].identity_dict() == {
        "adapter": "synthetic",
        "model": "synthetic",
        "endpoint": None,
        "aws_profile": None,
        "region": None,
        "tokens": None,
        "score": None,
        "category": None,
        "classify": None,
        "spans": None,
        "complete": None,
    }


def test_candidate_aliases_share_a_template_and_override_only_their_channels(tmp_path):
    doc = mapping(limits={"max_work": 1234})
    template = doc["models"]["reader"]
    del template["spans"]["channel"]
    doc["model_templates"] = {
        "candidates": template,
        "bounded": {
            "extends": "candidates",
            "spans": {"limits": {"max_parts": 12, "max_ranges_per_channel": 4}},
        },
    }
    doc["models"] = {
        name: {"extends": "bounded", "spans": {"channel": name}} for name in ("left", "right")
    }

    config = QuarryConfig.from_mapping(doc, tmp_path, require_rules=False)

    for name in ("left", "right"):
        assert config.models[name].spans == CandidateSpanConfig(
            batch="heads",
            channel=name,
            settings_digest="0" * 64,
            limits=CandidateBatchLimits(max_work=1234, max_parts=12, max_ranges_per_channel=4),
        )


@pytest.mark.parametrize(
    "spans,reason",
    [
        ({"batch": "../unregistered"}, "bounded ASCII identifiers"),
        ({"settings_digest": "bad"}, "invalid candidate request digest"),
        ({"limits": {"max_parts": 4097}}, "invalid candidate limits"),
        ({"backend": "candidates", "limits": {"max_spans": 1}}, "max_spans"),
    ],
)
def test_unused_candidate_template_fragments_keep_candidate_validation(tmp_path, spans, reason):
    doc = mapping()
    doc["model_templates"] = {"unused": {"spans": spans}}

    with pytest.raises(ConfigurationError, match=reason):
        QuarryConfig.from_mapping(doc, tmp_path, require_rules=False)
