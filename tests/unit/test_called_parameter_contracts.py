# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Called checks track only their parameter declarations and preserve root usage."""

import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from quarry import DSLScanner, load_compiled_policy
from quarry.integrations.resources import requirements_from_artifact
from quarry.policy import CompiledPolicy
from quarry.types.exceptions import RuleValidationError
from tests.unit.test_policy_parameters import _runtime

SOURCES = {
    "child-only": 'rule "Leaf" parameters: threshold: number condition: model("judge").score(messages,"c") >= params.threshold\nrule "Caller" imports: leaf="Leaf" condition: leaf(messages)',
    "caller-only": 'rule "Leaf" condition: len(messages)>0\nrule "Caller" imports: leaf="Leaf" parameters: threshold: number condition: leaf(messages) and model("judge").score(messages,"c") >= params.threshold',
    "siblings": 'rule "First" parameters: first: number policy condition: model("judge").score(messages,"first") >= params.first\nrule "Second" parameters: threshold: number condition: model("judge").score(messages,"second") >= params.threshold\nrule "Caller" imports: first="First" second="Second" condition: first(messages) and second(messages)',
    "deep": 'rule "Leaf" parameters: threshold: number condition: model("judge").score(messages,"c") >= params.threshold\nrule "Middle" imports: leaf="Leaf" condition: leaf(messages)\nrule "Caller" imports: middle="Middle" condition: middle(messages)',
}


def policy(source, config, backend, policy_parameters=None):
    scanner = DSLScanner(
        replace(config, quarry_rule_texts=(source,)),
        model_inference_backend=backend,
        _allow_parameters=True,
    )
    name = next(reversed(scanner.compiled_rules))
    artifact = scanner.compiled_rules[name]
    entry = DSLScanner._from_compiled_policy(
        config,
        rule=scanner.loaded_rules[name],
        artifact=artifact,
        resource_requirements=requirements_from_artifact(artifact),
        model_inference_backend=backend,
    )
    return CompiledPolicy._from_scanner(entry, source, policy_parameters)


@pytest.mark.parametrize("case", list(SOURCES))
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
def test_parameters_used_only_by_a_child_caller_sibling_or_descendant(
    tmp_path, case, restored, explain
):
    config, backend, _, _ = _runtime(tmp_path)
    prepared = policy(
        SOURCES[case], config, backend, {"first": 0.5} if case == "siblings" else None
    )
    if restored:
        raw = prepared.to_bytes()

        def forbidden(*args, **kwargs):
            raise AssertionError("parameterized artifact load entered parser/compiler")

        with (
            patch("quarry.dsl.parser.DSLParser.parse", forbidden),
            patch("quarry.dsl.parser.DSLParser.parse_many", forbidden),
            patch("quarry.scanner.compile_rule", forbidden),
        ):
            prepared = load_compiled_policy(
                raw, runtime_config=config, model_inference_backend=backend
            )
        assert prepared.to_bytes() == raw
    for threshold, matched in [(0.7, True), (0.95, False)]:
        result = prepared.scan(
            {"messages": [{"role": "user", "content": "high"}]},
            {"threshold": threshold},
            explain=explain,
        )
        assert result.is_complete, result.evaluation_errors
        assert result.rule_matches == (["Caller"] if matched else [])
    root = prepared._scanner.compiled_rules["Caller"]
    if case == "siblings":
        assert sorted(
            sorted(p.name for p in child.parameter_schema) for child in root.split_checks.values()
        ) == [["first"], ["threshold"]]
    elif case == "caller-only":
        assert all(not child.parameter_schema for child in root.split_checks.values())


@pytest.mark.parametrize("explain", [False, True, "support"])
def test_a_parameter_used_only_in_the_call_argument_is_not_declared_in_the_child(tmp_path, explain):
    config, backend, _, _ = _runtime(tmp_path)
    source = 'rule "Leaf" condition: len(messages)>0\nrule "Caller" imports: leaf="Leaf" parameters: instruction: text policy condition: leaf(message("user",params.instruction))'
    prepared = policy(source, config, backend, {"instruction": "synthetic policy text"})
    result = prepared.scan({"messages": [{"role": "user", "content": "high"}]}, explain=explain)
    assert result.is_complete and result.rule_matches == ["Caller"]
    assert "synthetic policy text" not in json.dumps(
        [e.to_dict() for e in result.explanations or []]
    )
    assert all(
        not child.parameter_schema
        for child in prepared._scanner.compiled_rules["Caller"].split_checks.values()
    )


@pytest.mark.parametrize(
    "source,reason",
    [
        (
            'rule "Leaf" parameters: unused: number condition: true\nrule "Caller" imports: leaf="Leaf" condition: leaf(messages)',
            "declared parameters are not used",
        ),
        (
            'rule "Leaf" condition: model("judge").score(messages,"c") >= params.missing\nrule "Caller" imports: leaf="Leaf" condition: leaf(messages)',
            "undeclared parameter",
        ),
        (
            'rule "First" parameters: threshold: number policy condition: model("judge").score(messages,"c") >= params.threshold\nrule "Second" parameters: threshold: number condition: model("judge").score(messages,"c") >= params.threshold\nrule "Caller" imports: first="First" second="Second" condition: first(messages) and second(messages)',
            "differs between",
        ),
    ],
)
def test_unused_undeclared_and_conflicting_parameters_remain_rejected(tmp_path, source, reason):
    config, backend, _, _ = _runtime(tmp_path)
    with pytest.raises(RuleValidationError, match=reason):
        policy(source, config, backend)
