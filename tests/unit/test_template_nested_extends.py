# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Inheritance directives do not consume nested classifier or schema names."""

import json
from copy import deepcopy

import pytest
import yaml

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from tests.unit.model_fakes import classification


def _config(tmp_path, document, entrypoint):
    if entrypoint == "mapping":
        return QuarryConfig.from_mapping(document, tmp_path, require_rules=False)
    path = tmp_path / "config.yaml"
    rule = tmp_path / "owned.rule"
    rule.write_text('rule "Owned" condition: true')
    path.write_text(yaml.safe_dump({**document, "rules": [str(rule)]}))
    return QuarryConfig.from_yaml(path)


def _document(parent_kind, base, patch):
    document = {"config_version": 2, parent_kind: {"base": base}}
    document.setdefault("models", {})["judge"] = {"extends": "base", **patch}
    return document


@pytest.mark.parametrize("operation", ["add", "replace", "remove"])
@pytest.mark.parametrize("parent_kind", ["models", "model_templates"])
@pytest.mark.parametrize("entrypoint", ["mapping", "yaml"])
def test_extends_is_an_ordinary_classifier_label_when_merging(
    tmp_path, operation, parent_kind, entrypoint
):
    labels = {"allow": "Allow", "deny": "Deny"}
    if operation != "add":
        labels["extends"] = "Old description"
    patch = None if operation == "remove" else "Escalate"
    expected = {**labels}
    if operation == "remove":
        expected.pop("extends")
    else:
        expected["extends"] = patch
    document = _document(
        parent_kind,
        {"model": "owned/model", "classify": {"class_index": 0, "labels": labels}},
        {"classify": {"labels": {"extends": patch}}},
    )
    original = deepcopy(document)
    config = _config(tmp_path, document, entrypoint)
    assert config.models["judge"].classify.labels == tuple(expected.items())
    assert document == original


@pytest.mark.parametrize("operation", ["add", "replace", "remove"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("parent_kind", ["models", "model_templates"])
@pytest.mark.parametrize("entrypoint", ["mapping", "yaml"])
def test_extends_is_an_ordinary_response_property_at_every_merge_depth(
    tmp_path, operation, nested, parent_kind, entrypoint
):
    properties = {"keep": {"type": "string"}}
    if operation != "add":
        properties["extends"] = {"type": "string", "description": "before"}
    base = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    patch_value = (
        {"type": "string", "description": "after"}
        if operation == "add"
        else {"description": "after"}
    )
    if operation == "remove":
        patch_value = None
    patch = {
        "properties": {"extends": patch_value},
        "required": ["keep"] if operation == "remove" else ["keep", "extends"],
    }
    if nested:
        base = {
            "type": "object",
            "properties": {"details": base},
            "required": ["details"],
            "additionalProperties": False,
        }
        patch = {"properties": {"details": patch}}
    document = _document(
        parent_kind,
        {"model": "owned/model", "complete": {"response_schema": base}},
        {"complete": {"response_schema": patch}},
    )
    original = deepcopy(document)
    config = _config(tmp_path, document, entrypoint)
    resolved = json.loads(config.models["judge"].complete.response_schema)
    if nested:
        resolved = resolved["properties"]["details"]
    if operation == "remove":
        assert "extends" not in resolved["properties"]
    else:
        assert resolved["properties"]["extends"] == {"type": "string", "description": "after"}
    assert resolved["properties"]["keep"] == {"type": "string"}
    assert document == original


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("entrypoint", ["mapping", "yaml"])
def test_nested_classifier_label_reaches_the_adapter_and_policy_fingerprint(
    tmp_path, restored, explain, entrypoint
):
    labels = {"allow": "Allow", "deny": "Deny", "extends": "Escalate"}
    base = {
        "adapter": "owned",
        "model": "owned/model",
        "classify": {"class_index": 0, "labels": {"allow": "Allow", "deny": "Deny"}},
    }
    inherited = _config(
        tmp_path,
        _document("model_templates", base, {"classify": {"labels": {"extends": "Escalate"}}}),
        entrypoint,
    )
    flat = _config(
        tmp_path,
        {
            "config_version": 2,
            "models": {"judge": {**base, "classify": {"class_index": 0, "labels": labels}}},
        },
        entrypoint,
    )

    class Adapter:
        def __init__(self):
            self.labels = []

        async def classify_messages(self, config, messages, tools, *, timeout=None):
            self.labels.append(config.classify.labels)
            return classification(0.8, 0.1, 0.1)

    digests = []
    for config in (inherited, flat):
        adapter = Adapter()
        backend = ModelInferenceBackend(config, adapters={"owned": adapter})
        policy = compile_policy(
            'rule "Read" condition: model("judge").classify(messages) > 0.5',
            runtime_config=config,
            model_inference_backend=backend,
        )
        if restored:
            wire = policy.to_bytes()
            policy = load_compiled_policy(
                wire, runtime_config=config, model_inference_backend=backend
            )
            assert policy.to_bytes() == wire
        result = policy.scan(
            {"messages": [{"role": "user", "content": "owned text"}]}, explain=explain
        )
        assert result.is_complete and result.rule_matches == ["Read"]
        assert adapter.labels == [tuple(labels.items())]
        digests.append((policy.policy_digest, policy._model_hashes))
    assert digests[0] == digests[1]
