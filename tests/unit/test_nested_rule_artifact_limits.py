# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Called-rule graphs have the same explicit limits at linking and wire boundaries."""

import pytest

from quarry import policy_serialization as wire
from quarry.config import QuarryConfig
from quarry.dsl import composition
from quarry.types.exceptions import RuleValidationError
from tests.unit.test_nested_rule_runtime_contracts import _policy, _wrappers
from tests.unit.test_rule_composition_budget import module


@pytest.mark.parametrize("boundary", ["load", "dump"])
@pytest.mark.parametrize("limit", ["depth", "count"])
def test_prepared_calls_enforce_graph_limits(boundary, limit, monkeypatch):
    source, name = _wrappers('rule "Leaf" condition: len(messages)>0', depth=3)
    policy = _policy(source, QuarryConfig(config_version=2), False)
    artifact = policy._scanner.compiled_rules[name]
    raw = wire._artifact_to_wire(artifact)
    constant = "MAX_NESTED_RULE_DEPTH" if limit == "depth" else "MAX_NESTED_RULE_CHECKS"
    monkeypatch.setattr(wire, constant, 1 if limit == "depth" else 3, raising=False)
    with pytest.raises(
        ValueError, match="called-rule.*" + ("depth" if limit == "depth" else "count")
    ):
        wire._artifact_from_wire(raw) if boundary == "load" else wire._artifact_to_wire(artifact)


@pytest.mark.parametrize("limit", ["depth", "count"])
def test_linker_rejects_excessive_check_graphs_before_copying(limit, monkeypatch):
    source, _ = _wrappers('rule "Leaf" condition: true', depth=3)
    rules = module(source)
    constant = "MAX_NESTED_RULE_DEPTH" if limit == "depth" else "MAX_NESTED_RULE_CHECKS"
    monkeypatch.setattr(composition, constant, 1 if limit == "depth" else 3, raising=False)
    copies = 0
    original = composition.copy.deepcopy

    def counted(*args, **kwargs):
        nonlocal copies
        copies += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(composition.copy, "deepcopy", counted)
    with pytest.raises(RuleValidationError, match="called-rule"):
        composition.link_public_rule_calls(rules, {})
    assert copies == 0


@pytest.mark.parametrize("boundary", ["load", "dump"])
def test_called_artifact_cycles_are_rejected(boundary):
    source, name = _wrappers('rule "Leaf" condition: true', depth=1)
    artifact = _policy(source, QuarryConfig(config_version=2), False)._scanner.compiled_rules[name]
    raw = wire._artifact_to_wire(artifact)
    if boundary == "load":
        raw["split_checks"][0][1] = raw
    else:
        artifact.split_checks[next(iter(artifact.split_checks))] = artifact
    with pytest.raises(ValueError, match="called-rule.*cycle"):
        wire._artifact_from_wire(raw) if boundary == "load" else wire._artifact_to_wire(artifact)


@pytest.mark.parametrize("boundary", ["load", "dump"])
def test_shared_siblings_count_as_separate_check_occurrences(boundary, monkeypatch):
    source = (
        'rule "Leaf" condition: true\nrule "Caller" imports: leaf="Leaf" '
        "condition: leaf(messages) and leaf(messages[1:])"
    )
    artifact = _policy(source, QuarryConfig(config_version=2), False)._scanner.compiled_rules[
        "Caller"
    ]
    first, second = artifact.split_checks
    artifact.split_checks[second] = artifact.split_checks[first]
    raw = wire._artifact_to_wire(artifact)
    raw["split_checks"][1][1] = raw["split_checks"][0][1]
    monkeypatch.setattr(wire, "MAX_NESTED_RULE_CHECKS", 2)
    with pytest.raises(ValueError, match="called-rule.*count"):
        wire._artifact_from_wire(raw) if boundary == "load" else wire._artifact_to_wire(artifact)


def test_graph_limits_accept_the_exact_depth_and_count_boundary(monkeypatch):
    source, name = _wrappers('rule "Leaf" condition: true', depth=3)
    artifact = _policy(source, QuarryConfig(config_version=2), False)._scanner.compiled_rules[name]
    monkeypatch.setattr(wire, "MAX_NESTED_RULE_DEPTH", 3)
    monkeypatch.setattr(wire, "MAX_NESTED_RULE_CHECKS", 4)
    raw = wire._artifact_to_wire(artifact)
    assert wire._artifact_to_wire(wire._artifact_from_wire(raw)) == raw
