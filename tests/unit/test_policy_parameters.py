# Copyright 2025 Cisco Systems, Inc. and its affilitates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Public behavior for compiled-policy parameters and batch execution.

A ``policy`` parameter is bound once at compile time and reported only by digest; a ``scan``
parameter is supplied per scan and validated exactly. Batches share one deadline, and a scan that
finishes after it is reported as timed out rather than trusted.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

import quarry.policy as policy_module
from quarry.config import QuarryConfig
from quarry.integrations import model_usage
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.policy import compile_policy
from quarry.safety import timer
from quarry.safety.limits import SafetyLimits
from quarry.safety.timer import QuarryTimeoutError
from quarry.scanner import DSLScanner
from quarry.telemetry import merge_model_usage
from quarry.types.exceptions import RuleParseError, RuleValidationError
from quarry.types.parameters import ParameterValueError
from quarry.types.rules import ScanResult


class _ScoreAdapter:
    """Fake judge: 0.9 for text containing "high", 0.2 for "low", 0.8 otherwise; "slow" waits."""

    def __init__(self) -> None:
        self.calls = 0

    async def score(self, _config, messages, *, timeout=None):
        self.calls += 1
        text = " ".join(str(m.get("content")) for m in messages if isinstance(m, dict))
        if "slow" in text:
            await asyncio.sleep(0.5)
        return 0.9 if "high" in text else 0.2 if "low" in text else 0.8

    async def tokenize(self, config, messages, _tools, *, timeout=None):
        count = sum(len(str(m.get("content") or "")) for m in messages if isinstance(m, dict))
        return {config.tokens.count_field: count}


class _PermitFirstLimiter:
    """Permit one physical call, then keep later calls waiting before dispatch."""

    def __init__(self) -> None:
        self.entries = 0
        self.waiting = asyncio.Event()

    @asynccontextmanager
    async def limit(self, _model):
        self.entries += 1
        if self.entries > 1:
            self.waiting.set()
            await asyncio.Future()
        yield 0.0


def _runtime(tmp_path, model: str = "judge-v1", **document):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "judge": {
                    "adapter": "fake",
                    "model": model,
                    "score": {"message_builder": "test", "boundary_seed": 7},
                    "tokens": {
                        "adapter": "fake",
                        "endpoint": None,
                        "extraction": {"count": "count"},
                    },
                }
            },
            **document,
        },
        tmp_path,
        require_rules=False,
    )
    adapter = _ScoreAdapter()
    instructions_seen: list[str] = []

    def builder(constitution, messages, _tools, _boundary, _score_config):
        instructions_seen.append(constitution)
        return messages

    backend = ModelInferenceBackend(
        config, adapters={"fake": adapter}, message_builders={"test": builder}
    )
    return config, backend, adapter, instructions_seen


def _policy(tmp_path, source: str, **options):
    config, backend, _, _ = _runtime(tmp_path)
    return compile_policy(source, runtime_config=config, model_inference_backend=backend, **options)


def _user(content: str) -> dict:
    return {"messages": [{"role": "user", "content": content}]}


_GUARDRAIL = """rule "Customer guardrail"
parameters:
    instructions: text policy
    threshold: number
condition:
    model("judge").score(messages, params.instructions) >= params.threshold
"""
_SECRET = "Customer-only evaluation instructions"


def test_a_policy_parameter_is_bound_once_and_a_scan_parameter_per_scan(tmp_path) -> None:
    config, backend, adapter, instructions_seen = _runtime(tmp_path)
    policy = compile_policy(
        _GUARDRAIL,
        policy_parameters={"instructions": _SECRET},
        runtime_config=config,
        model_inference_backend=backend,
    )

    matching = policy.scan_many([_user("first"), _user("second")], {"threshold": 0.7}, explain=True)
    non_matching = policy.scan(_user("first"), {"threshold": 0.9}, explain=True)

    assert [result.rule_matches for result in matching] == [
        ["Customer guardrail"],
        ["Customer guardrail"],
    ]
    assert non_matching.rule_matches == []
    # The second scan of "first" changes only the local threshold, so it reuses the
    # same model response without sending another request.
    assert adapter.calls == 2
    assert instructions_seen == [_SECRET] * 2
    assert matching[0].policy_digest == policy.policy_digest
    assert matching[0].parameter_digest == matching[1].parameter_digest
    assert matching[0].parameter_digest != non_matching.parameter_digest
    assert matching[0].parameter_evidence == {
        "instructions": {"type": "text", "binding": "policy", "redacted": True},
        "threshold": {"type": "number", "binding": "scan", "value": 0.7},
    }
    evidence = json.dumps([match.to_dict() for match in matching[0].detailed_matches])
    assert _SECRET not in evidence
    assert "sha256:" not in evidence
    assert _SECRET not in repr(policy)


def test_the_policy_digest_changes_with_policy_parameters_and_models(tmp_path) -> None:
    config, backend, _, _ = _runtime(tmp_path)
    changed_config, changed_backend, _, _ = _runtime(tmp_path, model="judge-v2")

    def digest(instructions: str, runtime_config, model_inference_backend) -> str:
        return compile_policy(
            _GUARDRAIL,
            policy_parameters={"instructions": instructions},
            runtime_config=runtime_config,
            model_inference_backend=model_inference_backend,
        ).policy_digest

    original = digest(_SECRET, config, backend)
    assert digest("Different instructions", config, backend) != original
    assert digest(_SECRET, changed_config, changed_backend) != original


def test_text_digests_depend_on_the_private_evidence_key(tmp_path) -> None:
    first_key = b"first-private-evidence-key-material"
    second_key = b"second-private-evidence-key-material"

    first = _policy(
        tmp_path,
        _GUARDRAIL,
        policy_parameters={"instructions": _SECRET},
        evidence_key=first_key,
    )
    same_key = _policy(
        tmp_path,
        _GUARDRAIL,
        policy_parameters={"instructions": _SECRET},
        evidence_key=first_key,
    )
    second = _policy(
        tmp_path,
        _GUARDRAIL,
        policy_parameters={"instructions": _SECRET},
        evidence_key=second_key,
    )

    conversation = _user("high risk")
    parameters = {"threshold": 0.7}
    first_result = first.scan(conversation, parameters)
    same_key_result = same_key.scan(conversation, parameters)
    second_result = second.scan(conversation, parameters)

    assert first.policy_digest == same_key.policy_digest
    assert first_result.parameter_digest == same_key_result.parameter_digest
    assert first.policy_digest != second.policy_digest
    assert first_result.parameter_digest != second_result.parameter_digest
    assert _SECRET not in repr(first_result)
    assert _SECRET not in repr(second_result)

    with pytest.raises(ValueError, match="evidence_key must contain at least 16 bytes"):
        _policy(
            tmp_path,
            _GUARDRAIL,
            policy_parameters={"instructions": _SECRET},
            evidence_key=b"too short",
        )


def test_policy_parameters_are_required_at_compile_and_refused_at_scan(tmp_path) -> None:
    config, backend, _, _ = _runtime(tmp_path)

    with pytest.raises(ParameterValueError, match="missing: instructions"):
        compile_policy(
            _GUARDRAIL, policy_parameters={}, runtime_config=config, model_inference_backend=backend
        )
    policy = compile_policy(
        _GUARDRAIL,
        policy_parameters={"instructions": _SECRET},
        runtime_config=config,
        model_inference_backend=backend,
    )
    with pytest.raises(ParameterValueError, match="unknown: instructions"):
        policy.scan(_user("first"), {"threshold": 0.7, "instructions": "runtime override"})


def _scanner(tmp_path, source: str) -> DSLScanner:
    config, backend, _, _ = _runtime(tmp_path)
    return DSLScanner(replace(config, quarry_rule_texts=(source,)), model_inference_backend=backend)


def test_a_plain_scanner_refuses_a_parameterized_rule(tmp_path) -> None:
    with pytest.raises(RuleValidationError, match="declares parameters; use compile_policy"):
        _scanner(tmp_path, _GUARDRAIL)


_THRESHOLD_ONLY = """rule "Parameterized"
parameters:
    threshold: number
condition:
    model("judge").score(messages, "instructions") >= params.threshold
"""


@pytest.mark.parametrize(
    ("values", "message"),
    [
        pytest.param({}, "missing", id="missing"),
        pytest.param({"threshold": True}, "parameter 'threshold' must be a number", id="boolean"),
        pytest.param(
            {"threshold": "0.7"}, "parameter 'threshold' must be a number", id="numeric-string"
        ),
        pytest.param(
            {"threshold": float("inf")}, "parameter 'threshold' must be finite", id="infinite"
        ),
        pytest.param({"threshold": 0.7, "extra": 1}, "unknown", id="undeclared"),
    ],
)
def test_scan_parameters_are_validated_exactly_without_coercion(tmp_path, values, message):
    policy = _policy(tmp_path, _THRESHOLD_ONLY)

    with pytest.raises(ParameterValueError, match=message):
        policy.scan(_user("x"), values)


def test_a_batch_must_not_be_empty(tmp_path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _policy(tmp_path, _THRESHOLD_ONLY).scan_many([], {"threshold": 0.7})


def _parameters(*declarations: str, condition: str) -> str:
    block = "".join(f"    {declaration}\n" for declaration in declarations)
    return f'rule "P"\nparameters:\n{block}condition:\n    {condition}\n'


@pytest.mark.parametrize(
    ("source", "error", "message"),
    [
        pytest.param(
            'rule "P"\ncondition:\n    model("judge").score(messages, "c") >= params.threshold\n',
            RuleValidationError,
            "undeclared parameter 'threshold'",
            id="undeclared-reference",
        ),
        pytest.param(
            _parameters("threshold: number", condition="count(user_messages) >= params.threshold"),
            RuleValidationError,
            "could not be compiled",
            id="structural-position",
        ),
        pytest.param(
            _parameters("threshold: number", condition="params"),
            RuleValidationError,
            "could not be compiled",
            id="bare-params-root",
        ),
        pytest.param(
            _parameters("threshold: number", condition="true"),
            RuleValidationError,
            "declared parameters are not used: threshold",
            id="declared-but-unused",
        ),
        pytest.param(
            _parameters("threshold: number", "threshold: number policy", condition="false"),
            RuleParseError,
            "duplicate definition",
            id="duplicate-declaration",
        ),
        pytest.param(
            _parameters(
                "threshold: number",
                condition='model("judge").score(messages, "c") >= params.threshold.value',
            ),
            RuleParseError,
            "chained field access",
            id="chained-field-access",
        ),
        pytest.param(
            _parameters("threshold: number request", condition="true"),
            RuleParseError,
            "'condition') at line 4, column 1",
            id="unknown-binding",
        ),
        pytest.param(
            _parameters("params: number", condition="true"),
            RuleParseError,
            "'params') at line 3, column 5",
            id="parameter-named-params",
        ),
        pytest.param(
            _parameters("policy: text", condition="true"),
            RuleParseError,
            "'policy') at line 3, column 5",
            id="parameter-named-policy",
        ),
    ],
)
def test_a_parameter_outside_the_supported_shape_fails_to_load(
    tmp_path, source, error, message
) -> None:
    config, backend, _, _ = _runtime(tmp_path)
    with pytest.raises(error, match=re.escape(message)):
        compile_policy(source, runtime_config=config, model_inference_backend=backend)


def test_variables_and_parameters_combine_in_one_rule(tmp_path) -> None:
    source = """rule "Both"
variables:
    enabled = any m in user_messages where m.content contains "x"
parameters:
    threshold: number
condition:
    enabled and model("judge").score(messages, "c") >= params.threshold
"""
    policy = _policy(tmp_path, source)

    assert policy.scan(_user("x"), {"threshold": 0.7}).rule_matches == ["Both"]
    assert policy.scan(_user("y"), {"threshold": 0.7}).rule_matches == []


def test_parameter_bindings_are_reported_per_declaration(tmp_path) -> None:
    source = """rule "Bindings"
parameters:
    first: number
    instructions: text policy
    second: number
condition:
    model("judge").score(messages, params.instructions) >= params.first and
    model("judge").score(messages, params.instructions) >= params.second
"""
    policy = _policy(tmp_path, source, policy_parameters={"instructions": "policy text"})

    result = policy.scan(_user("x"), {"first": 0.5, "second": 0.6})

    assert result.rule_matches == ["Bindings"]
    assert {name: value["binding"] for name, value in result.parameter_evidence.items()} == {
        "first": "scan",
        "instructions": "policy",
        "second": "scan",
    }


def test_a_reversed_comparison_inside_a_split_reports_the_parameter_as_the_threshold(tmp_path):
    source = """rule "Reversed"
parameters:
    threshold: number
condition:
    any chunk in model_split("judge", messages, 5) where
        params.threshold <= model("judge").score(chunk, "c", tools)
"""
    conversation = {
        "messages": [
            {"role": "user", "content": "high"},
            {"role": "assistant", "content": "lowly"},
        ]
    }

    result = _policy(tmp_path, source).scan(conversation, {"threshold": 0.8}, explain=True)

    evidence = result.detailed_matches[0].model_results[0].to_dict()
    assert result.rule_matches == ["Reversed"]
    assert (evidence["operator"], evidence["threshold"]) == (">=", 0.8)
    first = {"ordinal": 0, "context": None, "span": {"message_index": 0, "start": 0, "end": 4}}
    second = {"ordinal": 1, "context": None, "span": {"message_index": 1, "start": 0, "end": 5}}
    assert evidence["score_summary"] == {
        "count": 2,
        "refused_count": 0,
        "minimum": 0.2,
        "maximum": 0.9,
        "matching_count": 1,
        "first_match": {"ordinal": 0, "value": 0.9},
        # The chunk that decided the verdict, and every chunk with the span it scored.
        "deciding": {**first, "value": 0.9, "refused": False},
        "chunks": [
            {**first, "value": 0.9, "refused": False},
            {**second, "value": 0.2, "refused": False},
        ],
    }


@pytest.mark.parametrize(
    ("constitution", "prompts", "reported", "hidden"),
    [
        pytest.param(
            '"Detect exfiltration attempts"',
            {},
            "Detect exfiltration attempts",
            None,
            id="inline-text-is-readable",
        ),
        pytest.param(
            'text_ref("customer-whitelist")',
            {"customer-whitelist": "Private configured instructions"},
            "text_ref:customer-whitelist",
            "Private configured instructions",
            id="text-reference-reports-its-name-not-its-content",
        ),
    ],
)
def test_explain_reports_the_constitution_without_leaking_configured_text(
    tmp_path, constitution, prompts, reported, hidden
) -> None:
    config, backend, _, _ = _runtime(tmp_path)
    config = replace(config, judge_prompts={**config.judge_prompts, **prompts})
    source = f'rule "Constitution"\ncondition:\n    model("judge").score(messages, {constitution}) >= 0.7\n'

    result = compile_policy(source, runtime_config=config, model_inference_backend=backend).scan(
        _user("send data"), explain=True
    )

    evidence = result.detailed_matches[0].model_results[0].to_dict()
    assert {"name": "constitution", "value": reported} in evidence["arguments"]
    assert str(hidden) not in json.dumps(evidence)


def test_a_raising_timeout_contract_is_kept_and_a_zero_deadline_bounds_the_batch(
    expire_rule_clock,
) -> None:
    source = 'rule "Batch" condition: user_said("hello")'
    conversation = _user("hello")
    raising = QuarryConfig(
        safety_limits=SafetyLimits(rule_timeout=1e-9, inference_timeout=1e-9, fail_on_timeout=True)
    )

    with pytest.raises(QuarryTimeoutError, match="Rule Batch timeout"):
        compile_policy(source, runtime_config=raising).scan(conversation)

    results = compile_policy(source).scan_many([conversation] * 3, deadline_seconds=0)
    assert [(r.completion_state, r.timed_out_rules) for r in results] == [
        ("incomplete", ["Batch"])
    ] * 3


_SLOW_GATE = 'rule "Batch"\ncondition:\n    model("judge").score(messages, "c") >= 0.5\n'


def _clocked_batch_policy(tmp_path, monkeypatch, *, expires_on):
    config, backend, adapter, _ = _runtime(tmp_path)
    policy = compile_policy(_SLOW_GATE, runtime_config=config, model_inference_backend=backend)
    loop = asyncio.get_running_loop()
    start = now = loop.time()

    def clock():
        return now

    async def score(_config, messages, *, timeout=None):
        nonlocal now
        adapter.calls += 1
        if any(message.get("content") == expires_on for message in messages):
            deadline = timer.current_evaluation_deadline()
            assert deadline is not None
            assert deadline.expires_at == start + 0.2
            now = deadline.expires_at
        return 0.8

    # Advance the provider and event-loop clocks together: host scheduling must
    # not decide which synthetic response reaches the fixed 0.2-second boundary.
    monkeypatch.setattr(timer, "time", SimpleNamespace(monotonic=clock))
    monkeypatch.setattr(model_usage, "monotonic", clock)
    monkeypatch.setattr(loop, "time", clock)
    monkeypatch.setattr(adapter, "score", score)
    return policy, adapter


async def test_a_batch_keeps_results_finished_before_its_deadline(tmp_path, monkeypatch) -> None:
    policy, adapter = _clocked_batch_policy(tmp_path, monkeypatch, expires_on="slow")
    results = await policy.scan_many_async(
        [_user("fast"), _user("slow"), _user("fast")], deadline_seconds=0.2
    )

    assert adapter.calls == 2
    assert [(r.rule_matches, r.completion_state, r.timed_out_rules) for r in results] == [
        (["Batch"], "complete", None),
        ([], "incomplete", ["Batch"]),
        ([], "incomplete", ["Batch"]),
    ]


async def test_a_batch_rejects_a_first_result_at_the_deadline(tmp_path, monkeypatch) -> None:
    policy, adapter = _clocked_batch_policy(tmp_path, monkeypatch, expires_on="fast")
    results = await policy.scan_many_async(
        [_user("fast"), _user("slow"), _user("fast")], deadline_seconds=0.2
    )

    assert adapter.calls == 1
    assert [(r.rule_matches, r.completion_state, r.timed_out_rules) for r in results] == [
        ([], "incomplete", ["Batch"]),
        ([], "incomplete", ["Batch"]),
        ([], "incomplete", ["Batch"]),
    ]


async def test_batch_deadline_replacement_preserves_completed_item_telemetry(
    tmp_path, monkeypatch
) -> None:
    policy = _policy(tmp_path, _SLOW_GATE)
    completed = ScanResult(
        file_path=None,
        rule_matches=["Batch"],
        scan_time=0.12,
        rule_timings={"Batch": 0.1},
        model_usage={
            "calls": 1,
            "records": [{"model": "judge", "operation": "score", "calls": 1}],
            "truncated": False,
        },
    )

    async def scan(*_args, **_kwargs):
        return completed

    remaining = iter((1.0, 0.0))
    monkeypatch.setattr(policy._scanner, "_scan_data_with_parameters_async", scan)
    monkeypatch.setattr(
        policy_module,
        "remaining_evaluation_time",
        lambda: next(remaining),
    )

    (timed_out,) = await policy.scan_many_async([_user("high")], deadline_seconds=10)

    assert timed_out.rule_matches == []
    assert timed_out.timed_out_rules == ["Batch"]
    assert timed_out.scan_time == completed.scan_time
    assert timed_out.rule_timings == completed.rule_timings
    assert timed_out.model_usage == completed.model_usage
    assert merge_model_usage([timed_out.model_usage]) == timed_out.model_usage


async def test_a_batch_deadline_raises_when_timeouts_are_configured_to_fail(
    tmp_path, monkeypatch
) -> None:
    config, backend, adapter, _ = _runtime(tmp_path, fail_on_timeout=True)
    policy = compile_policy(_SLOW_GATE, runtime_config=config, model_inference_backend=backend)
    now = timer.time.monotonic()
    score = adapter.score

    def clock():
        return now

    async def score_and_expire(*args, **kwargs):
        nonlocal now
        result = await score(*args, **kwargs)
        if adapter.calls == 2:
            deadline = timer.current_evaluation_deadline()
            assert deadline is not None
            now = deadline.expires_at
        return result

    monkeypatch.setattr(timer, "time", SimpleNamespace(monotonic=clock))
    monkeypatch.setattr(model_usage, "monotonic", clock)
    monkeypatch.setattr(adapter, "score", score_and_expire)

    with pytest.raises(QuarryTimeoutError, match="Policy batch deadline exceeded") as exc_info:
        await policy.scan_many_async([_user("first"), _user("second")], deadline_seconds=10)

    aggregate = exc_info.value.scan_result
    assert adapter.calls == aggregate.model_usage["calls"] == 2
    assert aggregate.model_usage["records"] == [
        {"model": "judge", "operation": "score", "calls": 2}
    ]
    assert aggregate.rule_timings["Batch"] >= 0.0


async def test_later_limiter_timeout_keeps_prefix_usage_and_adds_no_attempt(
    tmp_path,
) -> None:
    config, backend, adapter, _ = _runtime(tmp_path, fail_on_timeout=True)
    limiter = _PermitFirstLimiter()
    backend.model_usage.rate_limiter = limiter
    policy = compile_policy(_SLOW_GATE, runtime_config=config, model_inference_backend=backend)

    with pytest.raises(QuarryTimeoutError, match="Policy batch deadline exceeded") as exc_info:
        await policy.scan_many_async(
            [_user("first"), _user("waiting")],
            deadline_seconds=0.2,
        )

    aggregate = exc_info.value.scan_result
    assert limiter.entries == 2
    assert limiter.waiting.is_set()
    assert adapter.calls == aggregate.model_usage["calls"] == 1
    assert aggregate.model_usage["records"] == [
        {"model": "judge", "operation": "score", "calls": 1}
    ]


async def test_raising_batch_deadline_carries_completed_item_telemetry(
    tmp_path, monkeypatch
) -> None:
    config, backend, _, _ = _runtime(tmp_path, fail_on_timeout=True)
    policy = compile_policy(_SLOW_GATE, runtime_config=config, model_inference_backend=backend)
    completed = ScanResult(
        file_path=None,
        rule_matches=["Batch"],
        scan_time=0.08,
        rule_timings={"Batch": 0.07},
        model_usage={
            "calls": 1,
            "records": [{"model": "judge", "operation": "score", "calls": 1}],
            "truncated": False,
        },
    )

    async def scan(*_args, **_kwargs):
        return completed

    remaining = iter((1.0, 0.0))
    monkeypatch.setattr(policy._scanner, "_scan_data_with_parameters_async", scan)
    monkeypatch.setattr(
        policy_module,
        "remaining_evaluation_time",
        lambda: next(remaining),
    )

    def expired() -> None:
        raise QuarryTimeoutError("Policy batch deadline exceeded")

    monkeypatch.setattr(policy_module, "raise_if_evaluation_expired", expired)

    with pytest.raises(QuarryTimeoutError, match="Policy batch deadline exceeded") as exc_info:
        await policy.scan_many_async([_user("high")], deadline_seconds=10)

    assert exc_info.value.scan_result.model_usage == completed.model_usage
    assert exc_info.value.scan_result.rule_timings == completed.rule_timings


async def test_raising_batch_aggregates_prefix_and_retried_current_once(
    tmp_path, monkeypatch
) -> None:
    config, backend, _, _ = _runtime(tmp_path, fail_on_timeout=True)
    policy = compile_policy(_SLOW_GATE, runtime_config=config, model_inference_backend=backend)
    prefix = ScanResult(
        None,
        ["Batch"],
        0.04,
        rule_timings={"Batch": 0.03},
        model_usage={
            "calls": 1,
            "records": [{"model": "judge", "operation": "score", "calls": 1}],
            "truncated": False,
        },
    )
    current = ScanResult(
        None,
        [],
        0.08,
        timed_out_rules=["Batch"],
        rule_timings={"Batch": 0.07},
        model_usage={
            "calls": 2,
            "records": [{"model": "judge", "operation": "score", "calls": 2}],
            "truncated": False,
        },
    )
    scans = 0

    async def scan(*_args, **_kwargs):
        nonlocal scans
        scans += 1
        if scans == 1:
            return prefix
        raise QuarryTimeoutError("Rule Batch timeout", scan_result=current)

    remaining = iter((1.0, 1.0, 1.0))
    monkeypatch.setattr(policy._scanner, "_scan_data_with_parameters_async", scan)
    monkeypatch.setattr(policy_module, "remaining_evaluation_time", lambda: next(remaining))

    with pytest.raises(QuarryTimeoutError, match="Rule Batch timeout") as exc_info:
        await policy.scan_many_async([_user("first"), _user("second")], deadline_seconds=10)

    aggregate = exc_info.value.scan_result
    assert aggregate.scan_time == pytest.approx(0.12)
    assert aggregate.rule_timings == {"Batch": pytest.approx(0.05)}
    assert aggregate.model_usage == {
        "calls": 3,
        "records": [{"model": "judge", "operation": "score", "calls": 3}],
        "truncated": False,
    }


def test_sync_batch_predispatch_timeout_retains_completed_prefix_telemetry(
    tmp_path, monkeypatch
) -> None:
    config, backend, _, _ = _runtime(tmp_path, fail_on_timeout=True)
    policy = compile_policy(_SLOW_GATE, runtime_config=config, model_inference_backend=backend)
    prefix = ScanResult(
        None,
        ["Batch"],
        0.05,
        rule_timings={"Batch": 0.04},
        model_usage={
            "calls": 1,
            "records": [{"model": "judge", "operation": "score", "calls": 1}],
            "truncated": False,
        },
    )

    async def scan(*_args, **_kwargs):
        return prefix

    remaining = iter((1.0, 1.0, 0.0))
    monkeypatch.setattr(policy._scanner, "_scan_data_with_parameters_async", scan)
    monkeypatch.setattr(policy_module, "remaining_evaluation_time", lambda: next(remaining))

    def expired() -> None:
        raise QuarryTimeoutError("Policy batch deadline exceeded")

    monkeypatch.setattr(policy_module, "raise_if_evaluation_expired", expired)

    with pytest.raises(QuarryTimeoutError, match="Policy batch deadline exceeded") as exc_info:
        policy.scan_many([_user("first"), _user("second")], deadline_seconds=10)

    aggregate = exc_info.value.scan_result
    assert aggregate.scan_time == prefix.scan_time
    assert aggregate.rule_timings == prefix.rule_timings
    assert aggregate.model_usage == prefix.model_usage


async def test_rule_traces_are_returned_only_when_requested() -> None:
    policy = compile_policy('rule "Bad"\ncondition:\n  user_said("bad")\n')
    conversation = _user("bad")

    assert (await policy.scan_async(conversation, explain=True)).rule_traces is None
    (traced,) = await policy.scan_many_async([conversation], record_traces=True)
    assert set(traced.rule_traces) == {"Bad"}
