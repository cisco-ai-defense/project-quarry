# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Public rule calls preserve newer judge explanation contracts."""

import asyncio
import json
from dataclasses import asdict, replace

import pytest

from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.safety.limits import SafetyLimits
from quarry.safety.timer import QuarryTimeoutError
from quarry.scanner import DSLScanner
from quarry.types.exceptions import ModelError
from quarry.types.failures import ScanFailureCategory
from quarry.types.results import JudgeResult
from tests.unit.model_fakes import passthrough_builder
from tests.unit.test_judge_reasons import JudgeAdapter, _model


@pytest.mark.parametrize("depth", [1, 2])
def test_nested_public_rule_keeps_requested_judge_reason(tmp_path, depth):
    reason = "The synthetic request meets the criterion."
    adapter = JudgeAdapter(reason=reason)
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"judge": _model()}}, tmp_path, require_rules=False
    )
    backend = ModelInferenceBackend(
        config, adapters={"fake": adapter}, message_builders={"test": passthrough_builder}
    )
    source = 'rule "Judge" condition: model("judge").score(messages, "criterion") > 0.5\n'
    previous = "Judge"
    for number in range(depth):
        name = f"Caller{number}"
        source += f'rule "{name}" imports: check = "{previous}" condition: check(messages)\n'
        previous = name
    scanner = DSLScanner(
        replace(config, quarry_rule_texts=(source,)), model_inference_backend=backend
    )
    result = scanner.scan_data(
        {"messages": [{"role": "user", "content": "owned text"}]},
        rules_subset={previous},
        enable_explain=True,
    )
    assert result.is_complete and result.rule_matches == [previous]
    assert reason in json.dumps([e.to_dict() for e in result.explanations])


def scanner_for(tmp_path, condition, *, depth=1, adapter=None, models=None, limits=None):
    adapter = adapter or JudgeAdapter()
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": models or {"judge": _model()}},
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(
        config, adapters={"fake": adapter}, message_builders={"test": passthrough_builder}
    )
    source = f'rule "Judge" condition: {condition}\n'
    previous = "Judge"
    for number in range(depth):
        name = f"Caller{number}"
        source += f'rule "{name}" imports: check = "{previous}" condition: check(messages)\n'
        previous = name
    return (
        DSLScanner(
            replace(config, quarry_rule_texts=(source,)),
            model_inference_backend=backend,
            limits=limits,
        ),
        previous,
    )


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("count", [2, 12])
def test_called_reason_ordinals_and_omissions_are_preserved(tmp_path, depth, count):
    condition = " and ".join(
        f'model("judge").score(messages, "criterion {number}") > 0.5' for number in range(count)
    )
    scanner, name = scanner_for(tmp_path, condition, depth=depth)
    result = scanner.scan_data(
        {"messages": [{"role": "user", "content": "owned text"}]},
        rules_subset={name},
        enable_explain=True,
    )
    assert result.is_complete and result.rule_matches == [name]
    evidence = result.explanations[0].model_results[0]
    assert [ordinal for ordinal, _, _ in evidence.judge_reasons] == list(range(min(8, count)))
    assert evidence.omitted_judge_reasons == max(0, count - 8)
    assert "_call_count" not in evidence.to_dict()


def test_calls_without_reasons_still_occupy_their_logical_ordinal(tmp_path):
    class RecordingJudge(JudgeAdapter):
        def __init__(self):
            super().__init__()
            self.names = []

        async def score(self, config, messages, *, timeout=None):
            self.names.append(config.model)
            return JudgeResult(0.8, "owned reason")

    adapter = RecordingJudge()
    mute, live = _model(explanation=False), _model()
    mute["model"], live["model"] = "muted-model", "live-model"
    scanner, name = scanner_for(
        tmp_path,
        'model("mute").score(messages, "muted") > 0.5 and '
        '(any m in user_messages where model("judge").score(m, "live") > 0.5)',
        adapter=adapter,
        models={"mute": mute, "judge": live},
    )
    result = scanner.scan_data(
        {"messages": [{"role": "user", "content": "owned text"}]},
        rules_subset={name},
        enable_explain=True,
    )
    assert result.is_complete and result.rule_matches == [name]
    evidence = result.explanations[0].model_results[0]
    assert len(adapter.names) == 2
    assert evidence.judge_reasons == ((adapter.names.index("live-model"), "judge", "owned reason"),)


@pytest.mark.parametrize("explain", [False, True])
def test_called_reasons_stay_out_of_decision_records(tmp_path, explain):
    reason = "SYNTHETIC_PRIVATE_JUDGE_REASON"
    scanner, name = scanner_for(
        tmp_path,
        'model("judge").score(messages, "criterion") > 0.5',
        depth=2,
        adapter=JudgeAdapter(reason=reason),
    )
    result = asyncio.run(
        scanner._scan_data_with_parameters_async(
            {"messages": [{"role": "user", "content": "owned text"}]},
            {},
            rules_subset={name},
            enable_explain=explain,
            decision_paths=True,
        )
    )
    assert result.is_complete and result.rule_matches == [name]
    assert reason not in json.dumps(asdict(result.decision_paths[name]))
    if explain:
        assert reason in json.dumps([item.to_dict() for item in result.explanations])


def test_disabled_called_explanations_do_not_retain_adapter_reason(tmp_path):
    reason = "SYNTHETIC_DISABLED_REASON"
    scanner, name = scanner_for(
        tmp_path,
        'model("judge").score(messages, "criterion") > 0.5',
        adapter=JudgeAdapter(reason=reason),
        models={"judge": _model(explanation=False)},
    )
    result = scanner.scan_data(
        {"messages": [{"role": "user", "content": "owned text"}]},
        rules_subset={name},
        enable_explain=True,
    )
    assert result.is_complete and result.rule_matches == [name]
    assert reason not in json.dumps([item.to_dict() for item in result.explanations])


def test_called_model_failure_keeps_the_existing_typed_category(tmp_path):
    class Broken(JudgeAdapter):
        async def score(self, config, messages, *, timeout=None):
            raise ModelError("owned model failure")

    scanner, name = scanner_for(
        tmp_path, 'model("judge").score(messages, "criterion") > 0.5', adapter=Broken()
    )
    result = scanner.scan_data(
        {"messages": [{"role": "user", "content": "owned text"}]}, rules_subset={name}
    )
    assert result.completion_state == "incomplete"
    assert result.failures[0].category is ScanFailureCategory.MODEL


@pytest.mark.parametrize("fail_on_timeout", [False, True])
def test_called_model_deadline_keeps_timeout_policy(tmp_path, fail_on_timeout):
    class Expired(JudgeAdapter):
        async def score(self, config, messages, *, timeout=None):
            raise QuarryTimeoutError("owned model deadline")

    scanner, name = scanner_for(
        tmp_path,
        'model("judge").score(messages, "criterion") > 0.5',
        adapter=Expired(),
        limits=SafetyLimits(fail_on_timeout=fail_on_timeout),
    )
    data = {"messages": [{"role": "user", "content": "owned text"}]}
    if fail_on_timeout:
        with pytest.raises(QuarryTimeoutError):
            scanner.scan_data(data, rules_subset={name})
    else:
        result = scanner.scan_data(data, rules_subset={name})
        assert result.completion_state == "incomplete"
        assert result.failures[0].category is ScanFailureCategory.TIMEOUT
