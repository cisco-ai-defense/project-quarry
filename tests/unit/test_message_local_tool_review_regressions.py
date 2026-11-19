# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Local binder scope and aggregate evidence survive prepared-policy round trips."""

import pytest

from quarry import compile_policy, load_compiled_policy
from tests.unit.test_message_local_tool_conditions import DATA
from tests.unit.test_message_local_tool_paid_conditions import SCORE, local_policy


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize(
    "body,matched",
    [
        ('any m in assistant_messages[-1].tool_calls where m.name == "lookup"', True),
        ('any m in assistant_messages[-1].tool_calls where m.name == "absent"', False),
        ('all m in assistant_messages[-1].tool_calls where m.name != "absent"', True),
        ('all m in assistant_messages[-1].tool_calls where m.name == "lookup"', False),
        ('count([m for m in assistant_messages[-1].tool_calls if m.name == "lookup"]) == 1', True),
        ('count([m for m in assistant_messages[-1].tool_calls if m.name == "absent"]) == 1', False),
        ('any m in m.tool_calls where m.name == "lookup"', True),
        ('count([m for m in m.tool_calls if m.name == "lookup"]) == 1', True),
        (
            'any c in assistant_messages[-1].tool_calls where m.content == "plan" and c.name == "lookup"',
            True,
        ),
        (
            'any m in [m for m in assistant_messages[-1].tool_calls if m.name == "lookup"] where m.response.result == "found"',
            True,
        ),
    ],
)
def test_closed_shadowed_bodies_and_enclosing_sources_reload(body, matched, restored):
    policy = compile_policy('rule "Read" condition: any m in assistant_messages where ' + body)
    if restored:
        raw = policy.to_bytes()
        policy = load_compiled_policy(raw)
        assert policy.to_bytes() == raw
    result = policy.scan(DATA, explain=True)
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == (["Read"] if matched else [])


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [True, "support"])
@pytest.mark.parametrize("count", ["len", "count"])
@pytest.mark.parametrize("quantifier", ["any", "all"])
def test_indexed_quantifier_with_nested_count_reports_its_boolean(
    tmp_path, restored, explain, count, quantifier
):
    source = "assistant_messages[-1].tool_calls"
    if quantifier == "all":
        source = '[c for c in assistant_messages[-1].tool_calls if c.name == "lookup"]'
    condition = (
        f"{quantifier} c in {source} where "
        f"{count}(assistant_messages[-1].tool_calls) == 2 and {SCORE}"
    )
    policy, adapter = local_policy(tmp_path, condition, restored=restored, max_calls=1)
    result = policy.scan(DATA, explain=explain)
    assert result.is_complete and result.rule_matches == ["Read"], result.evaluation_errors
    assert adapter.score_messages == [[{"role": "user", "content": "lookup"}]]
    (evidence,) = result.detailed_matches[0].model_results
    assert evidence.operation == "boolean_expression"
    assert evidence.expression_value is True
    assert evidence.operator is evidence.threshold is evidence.input_span is None
    assert evidence.score_summary is None
    assert evidence._call_count == 1


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [True, "support"])
def test_outer_indexed_count_retains_numeric_measurement(tmp_path, restored, explain):
    condition = (
        'count([c for c in assistant_messages[-1].tool_calls if c.name == "lookup" '
        f"and len(assistant_messages[-1].tool_calls) == 2 and {SCORE}]) == 1"
    )
    policy, adapter = local_policy(tmp_path, condition, restored=restored, max_calls=1)
    result = policy.scan(DATA, explain=explain)
    assert result.is_complete and result.rule_matches == ["Read"], result.evaluation_errors
    assert adapter.score_messages == [[{"role": "user", "content": "lookup"}]]
    (evidence,) = result.detailed_matches[0].model_results
    assert evidence.operation == "numeric_expression"
    assert evidence.expression_value == 1 and type(evidence.expression_value) is int
    assert evidence.operator == "=" and evidence.threshold == 1
    assert evidence.input_span is None


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [True, "support"])
def test_outer_count_owns_evidence_when_its_filter_contains_a_quantifier(
    tmp_path, restored, explain
):
    condition = (
        'count([c for c in assistant_messages[-1].tool_calls if c.name == "lookup" and '
        "any x in assistant_messages[-1].tool_calls where x.name == c.name and "
        'model("reader").score(message("user", x.name), "synthetic") > 0.5]) == 1'
    )
    policy, adapter = local_policy(tmp_path, condition, restored=restored, max_calls=1)
    result = policy.scan(DATA, explain=explain)
    assert result.is_complete and result.rule_matches == ["Read"], result.evaluation_errors
    assert adapter.score_messages == [[{"role": "user", "content": "lookup"}]]
    (evidence,) = result.detailed_matches[0].model_results
    assert evidence.operation == "numeric_expression"
    assert evidence.expression_value == 1 and type(evidence.expression_value) is int
    assert evidence.operator == "=" and evidence.threshold == 1
    assert evidence.input_span is None


@pytest.mark.parametrize("restored", [False, True])
def test_indexed_tool_call_argument_arrays_keep_field_text_matching(restored):
    condition = 'any m in assistant_messages where m.tool_calls[0].arguments.items ~ "^a"'
    policy = compile_policy('rule "Read" condition: ' + condition)
    if restored:
        wire = policy.to_bytes()
        policy = load_compiled_policy(wire)
        assert policy.to_bytes() == wire
    result = policy.scan(
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"items":["a","b"]}',
                            },
                        }
                    ],
                }
            ]
        }
    )
    assert result.is_complete and result.rule_matches == ["Read"]
