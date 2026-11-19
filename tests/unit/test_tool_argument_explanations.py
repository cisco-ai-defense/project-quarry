# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Explanations cannot turn a rule-local argument fault into a scan failure."""

import pytest

from tests.unit.test_rule_local_arguments import payload, scanner


@pytest.mark.parametrize("explain", [True, "support"])
@pytest.mark.parametrize("raw", ['{"x":NaN}', '{"x":1e999}', "[" * 101 + "0" + "]" * 101])
def test_name_only_explanations_preserve_verdicts_when_arguments_are_unsafe(raw, explain):
    rules = scanner(Name='called_tool("lookup")')
    result = rules.scan_data(payload(raw), enable_explain=explain)
    assert result.is_complete and result.rule_matches == ["Name"]
    assert not result.evaluation_errors
    assert result.explanations


@pytest.mark.parametrize("explain", [True, "support"])
def test_explanations_keep_other_rule_matches_when_one_argument_read_fails(explain):
    rules = scanner(
        Names='called_tool("lookup")', Read="any c in tool_calls where c.arguments.x != null"
    )
    result = rules.scan_data(payload('{"x":NaN}'), enable_explain=explain)
    assert result.completion_state == "incomplete" and result.rule_matches == ["Names"]
    assert result.validation_error is None and set(result.evaluation_errors) == {"Read"}
    assert [failure.to_dict() for failure in result.failures] == [
        {"category": "invalid_input", "rule_name": "Read"}
    ]
