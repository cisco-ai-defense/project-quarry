# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Public span syntax and lexical scopes survive grammar-faithful rendering."""

from unittest.mock import patch

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.dsl.parser import DSLParser
from quarry.types.dsl import FunctionCall, Literal
from quarry.types.dsl.render import ConditionRenderError, render_condition


@pytest.mark.parametrize(
    "condition",
    [
        "len($p.spans(messages)) > 0",
        "len(union($p.spans(messages), $q.spans(messages))) > 0",
        "len(outermost($p.spans(messages))) > 0",
        'messages[0].content.extract($p.spans(messages[0].content)) contains "ignore"',
        'messages[0].content.extract($p.spans(messages[0].content), left=1) contains "ignore"',
        'messages[0].content.extract($p.spans(messages[0].content), right=2) contains "ignore"',
        'messages[0].content.extract($p.spans(messages[0].content), left=1, right=2, head=3) contains "ignore"',
        '(messages + messages.transform("strip"))[0].content contains "ignore"',
        '(message("user", "ignore").add_message("assistant", "review"))[0].content contains "ignore"',
        "(messages as selected) and ((user_messages as selected) and len(selected)==1) and len(selected)==4",
        'messages as selected and ((model("judge").classify(selected)>0.5) as selected and (true or selected))',
        "((messages as selected) and len(selected)==4) and ((user_messages as messages) and len(selected)==4)",
    ],
)
def test_public_spans_extract_keywords_and_nested_aliases_round_trip(condition):
    parser = DSLParser()
    prefix = 'rule "Read" strings: $p="ignore" $q="ignore previous" condition: '
    original = parser.parse(prefix + condition)
    rendered = render_condition(original.condition)
    restored = parser.parse(prefix + rendered)
    assert restored.condition == original.condition


@pytest.mark.parametrize("name", ["true", "false", "null", "not", "any", "all", "right_after"])
def test_reserved_function_names_do_not_render_as_another_node_or_invalid_syntax(name):
    with pytest.raises(ConditionRenderError, match="function name is reserved"):
        render_condition(FunctionCall(name, [Literal(True)]))


def test_internal_transform_diagnostic_keeps_the_default_receiver():
    rendered = render_condition(FunctionCall("_transform", []))
    assert rendered == "messages.transform()"
    assert "_transform" not in rendered


@pytest.mark.parametrize("name", ["if", "else", "where", "for", "and", "or", "after", "model"])
def test_keyword_like_function_names_that_the_grammar_accepts_still_round_trip(name):
    condition = FunctionCall(name, [Literal(True)])
    rendered = render_condition(condition)
    assert DSLParser().parse('rule "Read" condition: ' + rendered).condition == condition


@pytest.mark.parametrize(
    "condition,matching,nonmatching",
    [
        ('messages[0].content == "🔥"', "🔥", "other"),
        (r'messages[0].content ~ "\d+\s雪"', "42 雪", "other"),
    ],
)
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
def test_rendered_unicode_and_regex_keep_verdicts_through_guarded_reload(
    condition, matching, nonmatching, restored, explain
):
    original = DSLParser().parse('rule "Read" condition: ' + condition)
    rendered = render_condition(original.condition)
    policies = [compile_policy('rule "Read" condition: ' + text) for text in [condition, rendered]]
    if restored:

        def forbidden(*_args, **_kwargs):
            raise AssertionError("saved policy entered parser/compiler")

        loaded = []
        for policy in policies:
            raw = policy.to_bytes()
            with (
                patch("quarry.dsl.parser.DSLParser.parse", forbidden),
                patch("quarry.dsl.parser.DSLParser.parse_many", forbidden),
                patch("quarry.scanner.compile_rule", forbidden),
            ):
                policy = load_compiled_policy(raw)
                assert policy.to_bytes() == raw
            loaded.append(policy)
        policies = loaded
    for content, expected in [(matching, ["Read"]), (nonmatching, [])]:
        for policy in policies:
            result = policy.scan(
                {"messages": [{"role": "user", "content": content}]}, explain=explain
            )
            assert result.is_complete, result.evaluation_errors
            assert result.rule_matches == expected
