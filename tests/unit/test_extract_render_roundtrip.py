# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Rendering preserves extraction operations, source graphs and saved-policy outcomes."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.dsl.parser import DSLParser
from quarry.plugin.visualization import build_rule_visualization
from quarry.types.dsl import FunctionCall, Identifier
from quarry.types.dsl.render import ConditionRenderError, render_condition
from quarry.types.exceptions import RuleValidationError

RECEIVERS = [
    "messages",
    "conversation",
    "messages[:1]",
    "messages[0:1]",
    "messages[-1:]",
    "user_messages",
    "conversation().messages",
    "(messages + messages)",
    'messages.transform("strip")',
    'messages[:1].transform("strip")',
    '[m for m in messages if m.role=="user"]',
    "[m for m in messages]",
    "(messages if true else user_messages)",
    'message("user", "x")',
    "selected",
    'text_ref("a")',
    "messages[0].content",
    "memory.writes",
]
BOUNDS = [
    "",
    ",left=1",
    ",right=2",
    ",left=1,right=2",
    ",right=2,left=1",
    ",1",
    ",1,2",
    ",null",
    ",1,null",
]
RUNTIME_RECEIVERS = [
    receiver
    for receiver in RECEIVERS
    if receiver
    not in {
        "conversation",
        "conversation().messages",
        'message("user", "x")',
        'text_ref("a")',
        "messages[0].content",
        "memory.writes",
    }
]
PREFIX = 'rule "Read" strings: $p="x" condition: '


def _condition(receiver, bounds="", comparison='contains "x"'):
    expression = f"any s in $p.spans({receiver}) where extract({receiver},s{bounds}) {comparison}"
    if receiver == "selected":
        expression = "(messages[:1] as selected) and (" + expression + ")"
    return expression


def _config():
    return QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "rule_timeout": None,
            "transforms": {"strip": {"operations": [{"op": "strip"}]}},
        },
        Path("."),
        require_rules=False,
    )


@contextmanager
def _without_source_compiler():
    def forbidden(*_args, **_kwargs):
        raise AssertionError("saved extraction policy entered parser/compiler")

    with ExitStack() as stack:
        for target in [
            "quarry.dsl.parser.DSLParser.parse",
            "quarry.dsl.parser.DSLParser.parse_many",
            "quarry.scanner.compile_rule",
            "quarry.engine.compiler.compile_rule",
            "quarry.policy.compile_policy",
        ]:
            stack.enter_context(patch(target, forbidden))
        yield


def _restore(policy, config):
    wire = policy.to_bytes()
    with _without_source_compiler():
        restored = load_compiled_policy(wire, runtime_config=config)
        assert restored.to_bytes() == wire
    return restored


def _scan(policy, content, explain):
    wire = policy.to_bytes()
    with _without_source_compiler():
        result = policy.scan({"messages": [{"role": "user", "content": content}]}, explain=explain)
        assert policy.to_bytes() == wire
    return result


@pytest.mark.parametrize("receiver", RECEIVERS)
@pytest.mark.parametrize("bounds", BOUNDS)
def test_extract_function_and_source_graph_keep_the_parsed_operation(receiver, bounds):
    parser = DSLParser()
    original = parser.parse(PREFIX + _condition(receiver, bounds))
    rendered = render_condition(original.condition)
    assert parser.parse(PREFIX + rendered).condition == original.condition
    graph = build_rule_visualization(original)
    assert graph.nodes[0].label == rendered
    assert parser.parse(PREFIX + graph.nodes[0].label).condition == original.condition


@pytest.mark.parametrize(
    "receiver",
    [
        "messages[0].content",
        "messages[-1].content",
        "messages[0].tool_calls[0].raw_arguments",
        "messages[0].tool_calls[0].arguments.command",
        "messages[0].tool_responses[0].result",
        "audit.records[0].body",
    ],
)
@pytest.mark.parametrize(
    "bounds",
    [
        "",
        ",left=1",
        ",right=2",
        ",head=3",
        ",left=1,right=2",
        ",head=3,right=2,left=1",
        ",null",
        ",1,null",
        ",1,2,null",
    ],
)
def test_text_extraction_methods_keep_their_distinct_operation(receiver, bounds):
    parser = DSLParser()
    condition = f'{receiver}.extract($p.spans({receiver}){bounds}) contains "x"'
    original = parser.parse(PREFIX + condition)
    rendered = render_condition(original.condition)
    assert parser.parse(PREFIX + rendered).condition == original.condition


@pytest.mark.parametrize("receiver", RUNTIME_RECEIVERS)
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
def test_supported_extract_receivers_keep_outcomes_through_render_and_reload(
    receiver, restored, explain
):
    config = _config()
    original = DSLParser().parse(PREFIX + _condition(receiver, ",right=2,left=1"))
    rendered = render_condition(original.condition)
    policies = [
        compile_policy(PREFIX + condition, runtime_config=config)
        for condition in [_condition(receiver, ",right=2,left=1"), rendered]
    ]
    if restored:
        policies = [_restore(policy, config) for policy in policies]
    for content, expected in [(" x ", ["Read"]), ("quiet", [])]:
        for policy in policies:
            result = _scan(policy, content, explain)
            assert result.is_complete, result.failures
            assert result.rule_matches == expected
            assert not result.failures


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
def test_reported_sliced_receiver_retains_the_exact_equality_verdict(restored, explain):
    config = _config()
    original = DSLParser().parse(PREFIX + _condition("messages[:1]", comparison='== "x"'))
    rendered = render_condition(original.condition)
    for condition in [_condition("messages[:1]", comparison='== "x"'), rendered]:
        policy = compile_policy(PREFIX + condition, runtime_config=config)
        if restored:
            policy = _restore(policy, config)
        for content, expected in [(" x ", ["Read"]), ("quiet", [])]:
            result = _scan(policy, content, explain)
            assert result.is_complete and result.rule_matches == expected
            assert not result.failures


@pytest.mark.parametrize("bounds", [",null", ",1,null", ",1,2,null"])
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
def test_text_extraction_explicit_default_bounds_survive_render_and_reload(
    bounds, restored, explain
):
    config = _config()
    condition = f'messages[0].content.extract($p.spans(messages[0].content){bounds}) contains "x"'
    original = DSLParser().parse(PREFIX + condition)
    rendered = render_condition(original.condition)
    for text in [condition, rendered]:
        policy = compile_policy(PREFIX + text, runtime_config=config)
        if restored:
            policy = _restore(policy, config)
        for content, expected in [(" x ", ["Read"]), ("quiet", [])]:
            result = _scan(policy, content, explain)
            assert result.is_complete and result.rule_matches == expected
            assert not result.failures


@pytest.mark.parametrize(
    "receiver,bounds,diagnostic",
    [
        ("conversation", "", "unsupported model-expression reference 'conversation'"),
        ("conversation().messages", "", "function conversation() is not a model-expression input"),
        ("messages[0].content", "", "messages.extract() requires Messages and a span"),
        ("memory.writes", "", "unsupported model-expression reference 'memory'"),
        ("messages", ",left=true", "got boolean"),
        ("messages", ',right="x"', "got string"),
        ("messages", ",null,1", "got null"),
        ("messages", ",right=2", "got null"),
        ("messages", ",right=false", "got boolean"),
    ],
)
def test_rendering_does_not_admit_unsupported_receivers_or_bounds(receiver, bounds, diagnostic):
    config = _config()
    original = DSLParser().parse(PREFIX + _condition(receiver, bounds))
    rendered = render_condition(original.condition)
    for condition in [_condition(receiver, bounds), rendered]:
        with pytest.raises(RuleValidationError) as failure:
            compile_policy(PREFIX + condition, runtime_config=config)
        assert diagnostic in str(failure.value)


@pytest.mark.parametrize("limits", [{"max_nodes": 1}, {"max_depth": 1}, {"max_output": 12}])
def test_non_root_function_extraction_retains_renderer_limits(limits):
    condition = DSLParser().parse(PREFIX + _condition("messages[:1]", ",right=2")).condition
    with pytest.raises(ConditionRenderError):
        render_condition(condition, **limits)


def test_extract_function_round_trips_for_a_sliced_receiver():
    parser = DSLParser()
    source = PREFIX + 'any s in $p.spans(messages) where extract(messages[:1], s) == "x"'
    original = parser.parse(source).condition
    rendered = render_condition(original)
    assert "extract(messages[:1], s)" in rendered
    assert parser.parse(PREFIX + rendered).condition == original


def test_renderer_checks_list_budget_before_scanning_host_arguments():
    condition = FunctionCall("extract", [Identifier("messages") for _ in range(100_000)])
    with pytest.raises(ConditionRenderError, match="exceeds 2 condition nodes or edges"):
        render_condition(condition, max_nodes=2)
