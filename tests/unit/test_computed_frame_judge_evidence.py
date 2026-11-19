# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Computed children retain judge attribution through public-rule summaries."""

import pytest

from tests.unit.test_judge_reasons import JudgeAdapter, _scanner
from tests.unit.test_nested_rule_runtime_contracts import _policy, _wrappers


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("explain", [True, "support"])
def test_computed_child_judges_count_toward_called_rule_ordinals(
    tmp_path, restored, depth, explain
):
    adapter = JudgeAdapter()
    scanner = _scanner(tmp_path, adapter)
    source = (
        'rule "Leaf" condition: '
        "(any m in (messages[:1] + messages[:0]) "
        'where model("judge").score(message("user", m.content), "inner") > 0.5) '
        'and model("judge").score(messages, "outer") > 0.5'
    )
    source, name = _wrappers(source, depth=depth)
    policy = _policy(source, scanner.config, restored, scanner.resource_manager.inference_backend)
    result = policy.scan({"messages": [{"role": "user", "content": "needle"}]}, explain=explain)
    assert result.is_complete and result.rule_matches == [name]
    assert adapter.calls == 2
    (summary,) = result.detailed_matches[0].model_results
    assert summary.models == ("judge",)
    assert summary._call_count == 2
    assert [ordinal for ordinal, _, _ in summary.judge_reasons] == [0, 1]
    assert [alias for _, alias, _ in summary.judge_reasons] == ["judge", "judge"]
    assert summary.omitted_judge_reasons == 0


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("called", [False, True])
def test_computed_children_bound_reasons_without_losing_logical_call_counts(
    tmp_path, restored, called
):
    adapter = JudgeAdapter()
    scanner = _scanner(tmp_path, adapter)
    body = " and ".join(
        f'model("judge").score(message("user", m.content), "criterion-{index}") > 0.5'
        for index in range(12)
    )
    source = 'rule "Leaf" condition: any m in (messages + messages[:0]) where ' + body
    source, name = _wrappers(source, depth=int(called))
    policy = _policy(source, scanner.config, restored, scanner.resource_manager.inference_backend)
    result = policy.scan(
        {"messages": [{"role": "user", "content": "needle"}]},
        explain=True,
    )
    assert result.is_complete and result.rule_matches == [name]
    assert adapter.calls == 12
    (summary,) = result.detailed_matches[0].model_results
    assert summary._call_count == 12
    assert len(summary.judge_reasons) == 8 and summary.omitted_judge_reasons == 4
    assert [ordinal for ordinal, _, _ in summary.judge_reasons] == list(range(8))
