# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Explicit external-span configuration cannot alter an existing classifier alias."""

from dataclasses import replace

import pytest

from quarry.config import ExternalSpanConfig, ModelWindowsConfig, QuarryConfig, SpanConfig
from quarry.types.exceptions import ConfigurationError
from quarry.types.external_spans import ExternalSpanLimits


def model(tmp_path, spans, **extra):
    return QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "reader": {
                    "adapter": "fake",
                    "model": "synthetic-external",
                    "category": "SYNTHETIC",
                    "spans": spans,
                    **extra,
                }
            },
        },
        tmp_path,
        require_rules=False,
    ).models["reader"]


def test_external_mode_is_explicit_and_part_of_request_identity(tmp_path):
    configured = model(tmp_path, {"backend": "external"})
    assert configured.spans == ExternalSpanConfig()
    assert configured.identity_dict()["spans"] == {
        "backend": "external",
        "renderer": "join",
        "limits": {
            "max_input_characters": 1_000_000,
            "max_spans": 4096,
            "max_coverage_ranges": 4096,
            "max_work": 262_144,
        },
    }
    for changed in (
        ExternalSpanConfig(renderer="json"),
        ExternalSpanConfig(limits=ExternalSpanLimits(max_work=100)),
    ):
        assert replace(configured, spans=changed).identity_dict() != configured.identity_dict()


def test_legacy_span_identity_has_no_new_discriminator_or_limits(tmp_path):
    configured = model(tmp_path, {"class_index": 2})
    assert configured.spans == SpanConfig(2)
    assert configured.identity_dict()["spans"] == {"class_index": 2}
    assert "windows" not in configured.identity_dict()
    assert "revision" not in configured.identity_dict()


@pytest.mark.parametrize(
    "settings",
    [
        {"backend": "native"},
        {"backend": None},
        {"backend": True},
        {"backend": "external", "class_index": 0},
        {"backend": "external", "endpoint": "https://synthetic.example"},
        {"renderer": "json"},
        {"backend": "external", "renderer": "host"},
        {"backend": "external", "renderer": None},
        {"backend": "external", "limits": None},
        {"backend": "external", "limits": []},
        {"backend": "external", "limits": {"unknown": 1}},
        {"backend": "external", "limits": {"max_spans": True}},
        {"backend": "external", "limits": {"max_spans": 4097}},
        {"backend": "external", "limits": {"max_input_characters": -1}},
        {"backend": "external", "limits": {"max_work": 0}},
        {"backend": "external", "max_input_tokens": True},
    ],
)
def test_ambiguous_unsupported_or_unbounded_external_settings_fail(tmp_path, settings):
    with pytest.raises((ValueError, ConfigurationError)):
        model(tmp_path, settings)


def test_external_limits_are_owned_and_zero_capacity_is_explicit(tmp_path):
    limits = ExternalSpanLimits(max_spans=0, max_coverage_ranges=0, max_input_characters=0)
    configured = ExternalSpanConfig(limits=limits)
    assert configured.limits == limits and configured.limits is not limits
    parsed = model(
        tmp_path,
        {"backend": "external", "limits": {"max_spans": 0, "max_input_characters": 0}},
    )
    assert parsed.spans.limits.max_spans == parsed.spans.limits.max_input_characters == 0


def test_external_spans_cannot_silently_ignore_window_settings(tmp_path):
    with pytest.raises(ValueError, match="cannot be combined"):
        model(tmp_path, {"backend": "external"}, windows={"unit": "characters", "size": 10})
    configured = model(tmp_path, {"backend": "external"})
    with pytest.raises(ValueError, match="cannot be combined"):
        replace(configured, windows=ModelWindowsConfig("tokens", 8))


def test_external_template_inheritance_matches_flat_identity(tmp_path):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "model_templates": {
                "base": {
                    "adapter": "fake",
                    "model": "synthetic-external",
                    "category": "SYNTHETIC",
                    "spans": {"backend": "external", "limits": {"max_spans": 8}},
                }
            },
            "models": {"reader": {"extends": "base", "spans": {"renderer": "json"}}},
        },
        tmp_path,
        require_rules=False,
    )
    flat = model(tmp_path, {"backend": "external", "renderer": "json", "limits": {"max_spans": 8}})
    assert config.models["reader"].identity_dict() == flat.identity_dict()


@pytest.mark.parametrize(
    "fragment",
    [
        {"backend": "private-service"},
        {"renderer": "host"},
        {"limits": {"max_work": 0}},
        {"limits": {"max_spans": True}},
    ],
)
def test_unused_external_template_fragments_are_validated(tmp_path, fragment):
    with pytest.raises(ConfigurationError):
        QuarryConfig.from_mapping(
            {"model_templates": {"unused": {"spans": fragment}}},
            tmp_path,
            require_rules=False,
        )
