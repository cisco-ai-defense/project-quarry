# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Owned C19 round-trip composition diagnostics and bounded lexical analysis."""

import pytest

import quarry.engine.compiler as compiler_module
from quarry.dsl.parser import DSLParser
from quarry.engine.compiler import compile_rule
from quarry.plugin.visualization import build_rule_visualization
from quarry.types.dsl import (
    AnyComprehension,
    BinaryOp,
    ConditionalValue,
    ConditionNode,
    FieldAccess,
    FilterComprehension,
    Identifier,
    Literal,
    ModelCall,
    UnaryOp,
    walk_condition_nodes,
)
from quarry.types.dsl.render import render_condition


def _compile_expression(expression: str) -> list[str]:
    rule = DSLParser().parse(f'rule "R"\ncondition: {expression}\n')
    result = compile_rule(rule)
    return [error.message for error in result.diagnostics.errors]


def test_filter_comprehension_render_parse_round_trip_and_source_graph() -> None:
    source = 'rule "R"\ncondition: model("judge").classify([selected for item in messages if item.role == "user"]) > 0.5\n'
    rule = DSLParser().parse(source)
    filtered = next(
        node
        for node in walk_condition_nodes(rule.condition)
        if isinstance(node, FilterComprehension)
    )
    rendered_filter = render_condition(filtered)
    rendered_condition = render_condition(rule.condition)
    reparsed = DSLParser().parse(f'rule "R"\ncondition: {rendered_condition}\n')
    reparsed_filter = next(
        node
        for node in walk_condition_nodes(reparsed.condition)
        if isinstance(node, FilterComprehension)
    )
    graph = build_rule_visualization(rule)
    filter_node = next(node for node in graph.nodes if node.node_type == "FilterComprehension")
    assert rendered_filter == '[selected for item in messages if (item.role == "user")]'
    assert reparsed_filter == filtered
    assert filter_node.label == rendered_filter


def test_mixed_constructor_rendering_round_trips_as_authored_syntax() -> None:
    rule = DSLParser().parse(
        'rule "R"\ncondition: model("judge").classify(messages[:1] + message("assistant", "review")) > 0.5\n'
    )
    rendered = render_condition(rule.condition)
    reparsed = DSLParser().parse(f'rule "R"\ncondition: {rendered}\n')
    assert reparsed.condition == rule.condition
    assert 'messages[:1] + message("assistant", "review")' in rendered


@pytest.mark.parametrize(
    "condition",
    [
        "messages",
        "conversation().messages",
        "true or messages[:1]",
        "(messages[:1] + user_messages[-1:]) as selected",
        "true before messages[:1]",
        "messages[:1] if true else user_messages[-1:]",
        'model("judge").classify([item for item in messages if messages[:1]]) > 0.5',
        'model("judge").classify(messages if messages else user_messages) > 0.5',
        "any item in messages where messages[:1]",
    ],
)
def test_value_groups_in_verdict_positions_have_one_generic_diagnostic(condition) -> None:
    errors = _compile_expression(condition)
    assert len(errors) == 1
    assert "cannot be used as a Boolean verdict" in errors[0]
    assert "operation that returns a condition" in errors[0]
    assert len(errors[0]) < 240


@pytest.mark.parametrize("quantifier", ["any", "all"])
def test_unparenthesized_where_scope_has_bounded_parenthesis_guidance(quantifier) -> None:
    marker = "opaque-marker-" + "x" * 1000
    condition = f'{quantifier} item in messages where item.role == "{marker}" and messages[:1] + user_messages[-1:]'
    errors = _compile_expression(condition)
    assert len(errors) == 1
    assert f"`{quantifier} ... where ...`" in errors[0]
    assert f"({quantifier} item in collection where condition)" in errors[0]
    assert marker not in errors[0]
    assert len(errors[0]) < 260


@pytest.mark.parametrize(
    "condition",
    [
        'messages as selected and model("judge").classify(selected) > 0.5',
        '(messages as selected and model("judge").classify(selected) > 0.5) and true',
        'messages as selected and ((model("judge").classify(selected) > 0.5) as verdict and verdict)',
        '(model("judge").classify(messages) > 0.5) as verdict and (true or verdict)',
        '(messages if true else user_messages) as selected and model("judge").classify(selected) > 0.5',
        'messages as selected and model("judge").classify(selected if true else user_messages) > 0.5',
        '[item for item in messages if item.role == "user"] as selected and model("judge").classify(selected) > 0.5',
        'messages as selected and model("judge").classify([item for item in selected if item.role == "user"]) > 0.5',
        'messages as selected and any item in selected where item.role == "user"',
        'model("judge").classify(user_messages) as messages and messages > 0.5',
        'called_tool("lookup") as messages and (messages or false)',
        'any messages in user_messages where messages.role == "user"',
        "messages as item and any item in tools where true or item",
        'any item in messages where item.content as selected and selected contains "synthetic"',
    ],
)
def test_scoped_aliases_preserve_compile_core_behavior(condition: str) -> None:
    from quarry.engine.lower import compile_core

    rule = DSLParser().parse(f'rule "R"\ncondition: {condition}\n')
    core = compile_core(rule)
    wrapped = compile_rule(rule)
    assert core.ok
    assert wrapped.ok
    assert [error.message for error in wrapped.diagnostics.errors] == [
        error.message for error in core.diagnostics.errors
    ]


@pytest.mark.parametrize(
    "condition",
    [
        "messages as selected and (true or selected)",
        "messages as selected and (selected and true)",
        "messages as first and (first as second and (true or second))",
        "user_messages as messages and (true or messages)",
        "messages as selected and (true if selected else false)",
        'messages as selected and (called_tool("lookup") after selected)',
        'messages as selected and model("judge").classify([item for item in selected if true or selected]) > 0.5',
        "any item in messages where true or item",
        "any item in messages where item as selected and (true or selected)",
    ],
)
def test_message_aliases_are_rejected_in_every_verdict_context(condition: str) -> None:
    errors = _compile_expression(condition)
    assert len(errors) == 1
    assert (
        "cannot be used as a Boolean verdict" in errors[0]
        or "trailing value group is inside" in errors[0]
    )


def test_nested_scalar_alias_shadows_an_outer_message_alias() -> None:
    condition = 'messages as selected and ((model("judge").classify(selected) > 0.5) as selected and (true or selected))'
    rule = DSLParser().parse(f'rule "R"\ncondition: {condition}\n')
    assert compiler_module._composition_verdict_issue(rule.condition) is None
    assert compile_rule(rule).ok


@pytest.mark.parametrize(
    "condition",
    [
        'called_tool("a") after messages',
        "messages if true else false",
        "[item for item in messages if messages]",
        "any item in messages where true and messages",
        "(messages as selected)",
    ],
)
def test_actual_composed_values_still_get_the_generic_verdict_diagnostic(condition: str) -> None:
    errors = _compile_expression(condition)
    assert len(errors) == 1
    assert (
        "cannot be used as a Boolean verdict" in errors[0]
        or "trailing value group is inside" in errors[0]
    )


def test_where_trailing_classification_respects_an_adjacent_shadowing_alias() -> None:
    errors = _compile_expression(
        'any item in user_messages where model("judge").classify(user_messages) as messages and messages'
    )
    assert errors
    assert all("trailing value group is inside" not in error for error in errors)
    assert all("cannot be used as a Boolean verdict" not in error for error in errors)


def _conditional_chain(length: int) -> ConditionalValue:
    node = ConditionalValue(Literal(False), Literal(True), Literal(False))
    for _ in range(length - 1):
        node = ConditionalValue(node, Literal(True), Literal(False))
    return node


def test_composition_analysis_visits_nested_conditionals_linearly() -> None:
    length = 200
    stats: dict[str, int] = {}
    issue = compiler_module._composition_verdict_issue(_conditional_chain(length), _stats=stats)
    assert issue is None
    assert stats["classification_nodes"] == length * 3 + 1
    assert stats["classification_edges"] == length * 3
    assert stats["analysis_nodes"] == length * 3 + 1
    assert stats["work"] <= length * 10


def test_composition_analysis_visits_wide_shared_graph_linearly() -> None:
    shared = ConditionalValue(Literal(False), Literal(True), Literal(False))
    arguments: list[ConditionNode] = [shared] * 5000
    condition = BinaryOp(">", ModelCall("judge", "classify", arguments, "float"), Literal(0.5))
    stats: dict[str, int] = {}
    issue = compiler_module._composition_verdict_issue(condition, _stats=stats)
    assert issue is None
    assert stats["classification_nodes"] == 7
    assert stats["classification_edges"] == 5005
    assert stats["analysis_nodes"] == 7
    assert stats["work"] <= len(arguments) + 32


def test_equivalent_lexical_scopes_share_the_reviewer_dag_analysis() -> None:
    shared: ConditionNode = Literal(True)
    for _ in range(975):
        shared = BinaryOp("and", shared, Literal(True))
    shared = UnaryOp("not", shared)
    branches = [AnyComprehension("item", Identifier("messages"), shared) for _ in range(100)]
    condition = branches[0]
    for branch in branches[1:]:
        condition = BinaryOp("and", condition, branch)
    stats: dict[str, int] = {}
    issue = compiler_module._composition_verdict_issue(condition, _stats=stats)
    assert issue is None
    assert stats["classification_nodes"] == 2251
    assert stats["classification_edges"] == 2349
    assert stats["analysis_nodes"] == 2251
    assert stats["distinct_environments"] == 1
    assert stats["work"] < 10000


@pytest.mark.parametrize("reference_local_alias", [False, True])
def test_unrelated_alias_names_do_not_duplicate_shared_subtree_analysis(
    reference_local_alias: bool,
) -> None:
    shared: ConditionNode = Literal(True)
    for _ in range(975):
        shared = BinaryOp("and", shared, Literal(True))
    shared = UnaryOp("not", shared)
    branches = []
    for index in range(100):
        name = f"local_{index}"
        body = shared
        if reference_local_alias:
            body = BinaryOp("and", shared, BinaryOp("=", Identifier(name), Identifier(name)))
        branches.append(AnyComprehension(name, Identifier("messages"), body))
    condition = branches[0]
    for branch in branches[1:]:
        condition = BinaryOp("and", condition, branch)
    stats: dict[str, int] = {}
    issue = compiler_module._composition_verdict_issue(condition, _stats=stats)
    assert issue is None
    assert stats["analysis_nodes"] <= stats["classification_nodes"] * 2
    assert stats["work"] < 20000


def test_locally_shadowed_names_do_not_depend_on_the_calling_scope() -> None:
    shared: ConditionNode = Literal(True)
    for _ in range(975):
        shared = BinaryOp("and", shared, Literal(True))
    for index in range(40):
        name = f"local_{index}"
        check = BinaryOp("=", FieldAccess(Identifier(name), ["name"]), Literal("lookup"))
        shared = AnyComprehension(name, Identifier("tools"), BinaryOp("and", check, shared))
    branches = [
        AnyComprehension(f"local_{index % 40}", Identifier("messages"), shared)
        for index in range(100)
    ]
    condition = branches[0]
    for branch in branches[1:]:
        condition = BinaryOp("and", condition, branch)
    stats: dict[str, int] = {}
    issue = compiler_module._composition_verdict_issue(condition, _stats=stats)
    assert issue is None
    assert stats["analysis_nodes"] <= stats["classification_nodes"] * 2
    assert stats["work"] < 20000


@pytest.mark.parametrize("message_scope_first", [False, True])
def test_shared_nodes_keep_genuinely_distinct_shadow_classifications(
    message_scope_first: bool,
) -> None:
    shared = BinaryOp("=", Identifier("item"), Identifier("item"))
    message_scope = AnyComprehension("item", Identifier("messages"), shared)
    scalar_scope = AnyComprehension("item", Identifier("tools"), shared)
    left, right = (
        (message_scope, scalar_scope) if message_scope_first else (scalar_scope, message_scope)
    )
    stats: dict[str, int] = {}
    issue = compiler_module._composition_verdict_issue(BinaryOp("and", left, right), _stats=stats)
    assert issue is None
    assert stats["distinct_environments"] == 3
    assert stats["work"] < 64


def test_composition_analysis_stops_at_depth_without_python_recursion() -> None:
    node: ConditionalValue = _conditional_chain(1)
    for _ in range(2000):
        node = ConditionalValue(node, Literal(True), Literal(False))
    stats: dict[str, int] = {}
    issue = compiler_module._composition_verdict_issue(node, _stats=stats)
    assert issue is not None
    assert "too complex to validate safely" in issue
    assert "cannot be used as a Boolean verdict" not in issue
    assert stats["work"] < compiler_module._COMPOSITION_DIAGNOSTIC_WORK
    rule = DSLParser().parse('rule "R"\ncondition: true\n')
    rule.condition = node
    result = compile_rule(rule)
    assert not result.ok
    assert [error.message for error in result.diagnostics.errors] == [issue]


@pytest.mark.parametrize("operator", ["and", "or"])
@pytest.mark.parametrize("invalid_position", [0, 150, 300])
def test_long_flat_boolean_chains_find_values_at_every_position(
    operator: str, invalid_position: int
) -> None:
    terms = ["true"] * 301
    terms[invalid_position] = "messages"
    errors = _compile_expression(f" {operator} ".join(terms))
    assert len(errors) == 1
    assert "cannot be used as a Boolean verdict" in errors[0]


@pytest.mark.parametrize("operator", ["and", "or"])
def test_long_flat_boolean_chains_preserve_valid_core_behavior(operator: str) -> None:
    rule = DSLParser().parse(f"""rule "R"\ncondition: {f" {operator} ".join(["true"] * 301)}\n""")
    from quarry.engine.lower import compile_core

    core = compile_core(rule)
    wrapped = compile_rule(rule)
    assert core.ok
    assert wrapped.ok
    assert wrapped.diagnostics.errors == core.diagnostics.errors


def test_flat_chain_analysis_is_linear_for_left_and_right_spines() -> None:
    length = 1000
    left: ConditionNode = Literal(True)
    right: ConditionNode = Literal(True)
    for _ in range(length - 1):
        left = BinaryOp("and", left, Literal(True))
        right = BinaryOp("or", Literal(True), right)
    for condition in (left, right):
        stats: dict[str, int] = {}
        assert compiler_module._composition_verdict_issue(condition, _stats=stats) is None
        assert stats["classification_nodes"] == length * 2 - 1
        assert stats["classification_edges"] == length * 2 - 2
        assert stats["analysis_nodes"] == length * 2 - 1
        assert stats["work"] <= length * 6


def test_mixed_deep_chain_fails_closed_with_a_distinct_bounded_diagnostic() -> None:
    condition: ConditionNode = Literal(True)
    for index in range(400):
        condition = BinaryOp("and" if index % 2 else "or", condition, Literal(True))
    stats: dict[str, int] = {}
    issue = compiler_module._composition_verdict_issue(condition, _stats=stats)
    assert issue is not None
    assert "too complex to validate safely" in issue
    assert "cannot be used as a Boolean verdict" not in issue
    assert stats["work"] < compiler_module._COMPOSITION_DIAGNOSTIC_WORK


def test_composition_work_exhaustion_fails_closed(monkeypatch) -> None:
    condition: ConditionNode = Literal(True)
    for _ in range(20):
        condition = BinaryOp("and", condition, Literal(True))
    monkeypatch.setattr(compiler_module, "_COMPOSITION_DIAGNOSTIC_WORK", 10)
    rule = DSLParser().parse('rule "R"\ncondition: true\n')
    rule.condition = condition
    result = compile_rule(rule)
    assert not result.ok
    assert [error.message for error in result.diagnostics.errors] == [
        compiler_module._COMPOSITION_LIMIT_MESSAGE
    ]
