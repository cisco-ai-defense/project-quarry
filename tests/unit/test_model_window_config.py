# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Generic window declarations are immutable, closed and part of model identity."""

from dataclasses import FrozenInstanceError

import pytest

from quarry.config import QuarryConfig
from quarry.types.exceptions import ConfigurationError
from tests.unit.model_fakes import model_entry
from tests.unit.test_model_token_head_contracts import LegacyAdapter, backend


@pytest.mark.parametrize("renderer", ["join", "json", "host"])
def test_renderer_is_an_explicit_immutable_model_setting(tmp_path, renderer):
    service = backend(
        tmp_path,
        {
            "m": {
                **model_entry(classify=0),
                "windows": {"unit": "tokens", "size": 8, "renderer": renderer},
            }
        },
        LegacyAdapter(),
    )
    value = service.config.models["m"].windows
    assert value.renderer == renderer
    with pytest.raises(FrozenInstanceError):
        value.renderer = "json"


@pytest.mark.parametrize("renderer", ["yaml", "", True, 1])
def test_renderer_admission_is_closed_and_does_not_coerce_values(tmp_path, renderer):
    with pytest.raises(ConfigurationError):
        backend(
            tmp_path,
            {
                "m": {
                    **model_entry(classify=0),
                    "windows": {"unit": "tokens", "size": 8, "renderer": renderer},
                }
            },
            LegacyAdapter(),
        )


def test_renderer_templates_merge_and_null_removes_inherited_choice(tmp_path):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "model_templates": {
                "base": {
                    **model_entry(classify=0),
                    "windows": {"unit": "tokens", "size": 8, "renderer": "json"},
                }
            },
            "models": {
                "json": {"extends": "base"},
                "join": {"extends": "base", "windows": {"renderer": None}},
            },
        },
        tmp_path,
        require_rules=False,
    )
    assert config.models["json"].windows.renderer == "json"
    assert config.models["join"].windows.renderer == "join"
    assert config.models["json"].identity_dict() != config.models["join"].identity_dict()


@pytest.mark.parametrize("unit", ["characters", "tokens"])
@pytest.mark.parametrize("scope", ["all", "last_turn"])
def test_window_configuration_is_detached_and_immutable(tmp_path, unit, scope):
    declaration = {"unit": unit, "size": 8, "step": 4, "scope": scope}
    service = backend(
        tmp_path, {"m": {**model_entry(classify=0), "windows": declaration}}, LegacyAdapter()
    )
    windows = service.config.models["m"].windows
    assert (windows.unit, windows.size, windows.step, windows.scope) == (unit, 8, 4, scope)
    declaration["size"] = 99
    assert windows.size == 8
    with pytest.raises(FrozenInstanceError):
        windows.size = 99


@pytest.mark.parametrize("unit", ["characters", "tokens"])
def test_window_defaults_have_one_canonical_identity(tmp_path, unit):
    models = {
        "implicit": {**model_entry(classify=0), "windows": {"unit": unit, "size": 8}},
        "explicit": {
            **model_entry(classify=0),
            "windows": {"unit": unit, "size": 8, "step": 8, "scope": "all"},
        },
    }
    service = backend(tmp_path, models, LegacyAdapter())
    left, right = (service.config.models[name] for name in ["implicit", "explicit"])
    assert left.windows == right.windows
    assert left.identity_dict() == right.identity_dict()
    assert service._settings_digest("classify_messages", left) == service._settings_digest(
        "classify_messages", right
    )


@pytest.mark.parametrize(
    "change", [{"unit": "tokens"}, {"size": 9}, {"step": 3}, {"scope": "last_turn"}]
)
def test_each_window_setting_separates_provider_and_policy_identity(tmp_path, change):
    windows = {"unit": "characters", "size": 8, "step": 4, "scope": "all"}
    service = backend(
        tmp_path,
        {
            "left": {**model_entry(classify=0), "windows": windows},
            "right": {**model_entry(classify=0), "windows": {**windows, **change}},
        },
        LegacyAdapter(),
    )
    left, right = (service.config.models[name] for name in ["left", "right"])
    assert left.identity_dict() != right.identity_dict()
    assert service._settings_digest("classify_messages", left) != service._settings_digest(
        "classify_messages", right
    )


@pytest.mark.parametrize(
    "declaration",
    [
        {},
        [],
        "tokens",
        True,
        {"unit": "bytes", "size": 8},
        {"unit": "tokens", "size": True},
        {"unit": "tokens", "size": 0},
        {"unit": "tokens", "size": -1},
        {"unit": "tokens", "size": 2.5},
        {"unit": "tokens", "size": 8, "step": 0},
        {"unit": "tokens", "size": 8, "step": True},
        {"unit": "tokens", "size": 8, "step": 9},
        {"unit": "tokens", "size": 8, "scope": "assistant"},
        {"unit": "tokens", "size": 8, "extra": "synthetic"},
    ],
)
def test_invalid_window_declarations_fail_configuration(tmp_path, declaration):
    with pytest.raises(ConfigurationError):
        backend(
            tmp_path, {"m": {**model_entry(classify=0), "windows": declaration}}, LegacyAdapter()
        )


def test_window_templates_merge_and_null_removes_the_inherited_section(tmp_path):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "model_templates": {
                "base": {
                    **model_entry(classify=0),
                    "windows": {"unit": "tokens", "size": 8, "step": 4},
                }
            },
            "models": {
                "inherited": {"extends": "base", "windows": {"scope": "last_turn"}},
                "plain": {"extends": "base", "windows": None},
            },
        },
        tmp_path,
        require_rules=False,
    )
    selected = config.models["inherited"].windows
    assert (selected.unit, selected.size, selected.step, selected.scope) == (
        "tokens",
        8,
        4,
        "last_turn",
    )
    assert config.models["plain"].windows is None
    assert "windows" not in config.models["plain"].identity_dict()


def test_an_unused_invalid_window_template_is_not_hidden(tmp_path):
    with pytest.raises(ConfigurationError):
        QuarryConfig.from_mapping(
            {
                "config_version": 2,
                "model_templates": {"unused": {"windows": {"unit": "bytes", "size": 8}}},
                "models": {"m": model_entry(classify=0)},
            },
            tmp_path,
            require_rules=False,
        )


def test_omitted_and_null_windows_preserve_the_public_parent_identity(tmp_path):
    service = backend(
        tmp_path,
        {
            "plain": model_entry(classify=0),
            "null": {**model_entry(classify=0), "windows": None},
        },
        LegacyAdapter(),
    )
    plain, null = (service.config.models[name] for name in ["plain", "null"])
    assert plain.windows is None and null.windows is None
    assert plain.identity_dict() == null.identity_dict()
    assert "windows" not in plain.identity_dict()
    assert (
        service._settings_digest("classify_messages", plain).hex()
        == "55dd7d977edbcf1541c7723d4e8b37fbb82ae3280feff9d96403f6599862d7cf"
    )
