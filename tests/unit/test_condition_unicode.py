# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Rendered conditions preserve Unicode scalar values and executable verdicts."""

import json

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.dsl.parser import DSLParser
from quarry.engine.compiler import compile_rule
from quarry.plugin.visualization import build_rule_visualization
from quarry.types.dsl import Literal
from quarry.types.dsl.render import ConditionRenderError, render_condition

SCALARS = [chr(0x10000), "🔥", "雪🙂", "👩‍💻", chr(0x10FFFF)]


def parse(text):
    return DSLParser().parse('rule "Owned" condition: ' + text)


@pytest.mark.parametrize("value", [*SCALARS, chr(0xD83D), chr(0xDD25), chr(0xD83D) + chr(0xDD25)])
def test_programmatic_literals_preserve_scalars_and_separate_surrogate_units(value):
    rendered = render_condition(Literal(value))
    assert parse(rendered).condition == Literal(value)
    assert rendered.isascii()


@pytest.mark.parametrize("value", SCALARS)
@pytest.mark.parametrize("nested", [False, True])
def test_authored_unicode_round_trips_through_rendered_conditions(value, nested):
    condition = "messages[0].content == " + json.dumps(value, ensure_ascii=False)
    if nested:
        condition = "not (false or (" + condition + ")) if true else false"
    rule = parse(condition)
    assert parse(render_condition(rule.condition)).condition == rule.condition


@pytest.mark.parametrize("value", SCALARS)
@pytest.mark.parametrize("loaded", [False, True])
def test_rendered_conditions_keep_their_scan_verdict_after_prepared_reload(value, loaded):
    original = "messages[0].content == " + json.dumps(value, ensure_ascii=False)
    rendered = render_condition(parse(original).condition)
    policies = [compile_policy('rule "Owned" condition: ' + text) for text in [original, rendered]]
    if loaded:
        policies = [load_compiled_policy(policy.to_bytes()) for policy in policies]
    for content, expected in [(value, ["Owned"]), (value + "other", [])]:
        for policy in policies:
            result = policy.scan({"messages": [{"role": "user", "content": content}]})
            assert result.is_complete and result.rule_matches == expected


@pytest.mark.parametrize("value", SCALARS)
def test_decision_and_plugin_labels_survive_json_transport_without_changing_the_condition(value):
    rule = parse("messages[0].content == " + json.dumps(value, ensure_ascii=False))
    result = compile_rule(rule)
    assert result.ok and result.artifact is not None
    for node in result.artifact.decision_tree.nodes:
        assert parse(json.loads(json.dumps(node.label))).condition == rule.condition
    graph = build_rule_visualization(rule, result.artifact)
    root = next(node for node in graph.nodes if node.id == graph.root)
    assert parse(json.loads(json.dumps(root.label))).condition == rule.condition


def test_literal_backslashes_and_regex_spelling_remain_distinct_from_unicode_scalars():
    for value in [r"\U0001f525", r"\ud83d\udd25", "🔥"]:
        assert parse(render_condition(Literal(value))).condition == Literal(value)
    regex = 'messages[0].content ~ "🔥"'
    rendered = render_condition(parse(regex).condition)
    assert '"🔥"' in rendered and parse(rendered).condition == parse(regex).condition


def test_unicode_escape_output_limit_counts_the_complete_dsl_spelling():
    assert render_condition(Literal("🔥"), max_output=12) == '"\\U0001f525"'
    with pytest.raises(ConditionRenderError, match="output characters"):
        render_condition(Literal("🔥"), max_output=11)
