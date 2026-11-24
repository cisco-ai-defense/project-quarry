"""Display metadata stays local and separate from model requests and policy identity."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest
import yaml

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.safety.limits import SafetyLimits
from quarry.types.exceptions import ConfigurationError

MODEL = {
    "adapter": "synthetic",
    "model": "synthetic-score",
    "score": {
        "message_builder": "constitution",
        "boundary_seed": 7,
        "tokens": {"negative": "0", "positive": "1"},
    },
}


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("entrypoint", ["mapping", "yaml"])
def test_display_metadata_is_local_through_inheritance(tmp_path, version, entrypoint):
    document = {
        "config_version": version,
        "model_templates": {"base": MODEL},
        "models": {
            "parent": {"extends": "base", "display_name": "Parent label", "description": "Parent"},
            "implicit": {"extends": "parent"},
            "explicit": {"extends": "parent", "display_name": "Own label"},
            "cleared": {"extends": "parent", "display_name": None},
        },
    }
    original = deepcopy(document)
    if entrypoint == "yaml":
        rule = tmp_path / "read.rule"
        rule.write_text('rule "Read" condition: true')
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({**document, "rules": [str(rule)]}))
        config = QuarryConfig.from_yaml(path)
    else:
        config = QuarryConfig.from_mapping(document, tmp_path, require_rules=False)
    assert document == original
    assert config.model_display_names == {"parent": "Parent label", "explicit": "Own label"}
    assert config.model_descriptions == {"parent": "Parent"}
    assert all(value == config.models["parent"] for value in config.models.values())


@pytest.mark.parametrize("field", ["description", "display_name"])
def test_alias_metadata_is_not_admitted_in_templates(tmp_path, field):
    with pytest.raises(ConfigurationError, match=field):
        QuarryConfig.from_mapping(
            {"model_templates": {"unused": {**MODEL, field: "Not an alias"}}},
            tmp_path,
            require_rules=False,
        )


@pytest.mark.parametrize("label", ["A" * 100, "界" * 100, "Policy 🔎", "${QUARRY_LABEL_SENTINEL}"])
def test_display_metadata_remains_literal_and_preserves_rich_descriptions(
    tmp_path, monkeypatch, label
):
    monkeypatch.setenv("QUARRY_LABEL_SENTINEL", "synthetic environment value")
    config = QuarryConfig.from_mapping(
        {
            "models": {
                "judge": {**MODEL, "display_name": label, "description": "Line one\nLine two"}
            }
        },
        tmp_path,
        require_rules=False,
    )
    assert config.model_display_names == {"judge": label}
    assert config.model_descriptions == {"judge": "Line one\nLine two"}


@pytest.mark.parametrize(
    "control", ["\x00", "\x1b", "\x85", "\u2028", "\u2029", "\u202e", "\u200b", "\ud800"]
)
def test_control_and_format_characters_are_not_display_labels(tmp_path, control):
    with pytest.raises(ConfigurationError, match="printable"):
        QuarryConfig.from_mapping(
            {"models": {"judge": {**MODEL, "display_name": "Label" + control + "text"}}},
            tmp_path,
            require_rules=False,
        )


def test_oversized_label_is_refused_before_content_checks(tmp_path):
    with pytest.raises(ConfigurationError, match="at most 100"):
        QuarryConfig.from_mapping(
            {"models": {"judge": {**MODEL, "display_name": " " + "x" * 100_000}}},
            tmp_path,
            require_rules=False,
        )


def test_display_metadata_does_not_change_prepared_policy_or_cached_requests(tmp_path):
    class Adapter:
        def __init__(self):
            self.requests = []

        async def score(self, model, messages, *, timeout=None):
            self.requests.append((model, messages))
            return 0.8

    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"judge": {**MODEL, "display_name": "Original label"}}},
        tmp_path,
        require_rules=False,
    )
    config = replace(config, safety_limits=SafetyLimits(rule_timeout=None))
    renamed = replace(config, model_display_names={"judge": "New label"})
    adapter = Adapter()
    backend = ModelInferenceBackend(config, adapters={"synthetic": adapter})
    source = 'rule "Read" condition: model("judge").score(messages, "criterion") > 0.5'
    first = compile_policy(source, runtime_config=config, model_inference_backend=backend)
    second = compile_policy(source, runtime_config=renamed, model_inference_backend=backend)
    assert first.policy_digest == second.policy_digest
    assert first.to_bytes() == second.to_bytes()
    data = {"messages": [{"role": "user", "content": "sample"}]}
    assert first.scan(data).rule_matches == ["Read"]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("prepared policy was recompiled")

    with (
        patch("quarry.dsl.parser.DSLParser.parse", forbidden),
        patch("quarry.scanner.compile_rule", forbidden),
        patch("quarry.engine.compiler.compile_rule", forbidden),
        patch("quarry.policy.compile_policy", forbidden),
    ):
        loaded = load_compiled_policy(
            first.to_bytes(), runtime_config=renamed, model_inference_backend=backend
        )
        assert loaded.to_bytes() == first.to_bytes()
        assert loaded.scan(data).rule_matches == ["Read"]
    assert len(adapter.requests) == 1
    assert config.models is renamed.models
