# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Scan deadlines bound configured model work and preserve incomplete verdicts."""

from __future__ import annotations

import asyncio
import math
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.safety.limits import SafetyLimits
from quarry.safety.timer import QuarryTimeoutError
from quarry.scanner import DSLScanner
from tests.unit.model_fakes import classification

_MODEL_RULE = (
    'rule "m" condition: any msg in user_messages where model_scan("pi", msg.content, 0.5, 1)'
)
_FAST_RULE = 'rule "fast" condition: len(user_messages) > 0'
_ATTACK = {"messages": [{"role": "user", "content": "attack"}]}


@pytest.mark.parametrize("stage", ["arguments", "inputs"])
@pytest.mark.parametrize("api", ["sync", "async"])
@pytest.mark.parametrize("fail_on_timeout", [False, True])
def test_preflight_timeouts_preserve_zero_call_telemetry(monkeypatch, stage, api, fail_on_timeout):
    import quarry.scanner as scanner_module

    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            quarry_rule_texts=(_FAST_RULE,),
            safety_limits=SafetyLimits(rule_timeout=None, fail_on_timeout=fail_on_timeout),
        )
    )
    failure = QuarryTimeoutError("controlled preflight timeout")

    def expire(*_args, **_kwargs):
        raise failure

    if stage == "arguments":
        monkeypatch.setattr(scanner_module, "validate_conversation_arguments", expire)
    else:
        monkeypatch.setattr(type(scanner.input_registry), "prepare", expire)

    def scan():
        if api == "sync":
            return scanner.scan_data(_ATTACK)
        return asyncio.run(scanner.scan_data_async(_ATTACK))

    if fail_on_timeout:
        with pytest.raises(QuarryTimeoutError, match="controlled preflight timeout") as raised:
            scan()
        assert raised.value is failure
        result = raised.value.scan_result
        assert result is not None
        assert raised.value.result is result
    else:
        result = scan()

    assert result.completion_state == "incomplete"
    assert result.error_category == "timeout"
    assert result.rule_matches == []
    assert result.rule_timings is None
    assert result.model_usage == {"calls": 0, "records": [], "truncated": False}
    assert math.isfinite(result.scan_time) and result.scan_time >= 0


class _SlowAdapter:
    """A configured provider fake with an input-specific delay."""

    def __init__(self, delay: float, *, fast_delay: float | None = None):
        self.delay = delay
        self.fast_delay = fast_delay
        self.calls = 0

    async def classify_messages(self, _config, messages, _tools, *, timeout=None):
        self.calls += 1
        content = messages[0]["content"]
        delay = self.fast_delay if "fast" in content and self.fast_delay is not None else self.delay
        await asyncio.sleep(delay)
        return classification(0.02, 0.98)


def _scanner(tmp_path, *rules, adapter: _SlowAdapter, limits: SafetyLimits, limiter=None):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "pi": {
                    "adapter": "fake",
                    "model": "test-classifier",
                    "classify": {"class_index": 1},
                }
            },
        },
        tmp_path,
        require_rules=False,
    )
    config = replace(config, quarry_rule_texts=tuple(rules))
    backend = ModelInferenceBackend(config, adapters={"fake": adapter}, rate_limiter=limiter)
    return DSLScanner(config, model_inference_backend=backend, limits=limits)


@pytest.mark.parametrize(
    ("limits", "delay", "deadline", "timed_out", "reached_the_model"),
    [
        pytest.param(SafetyLimits(rule_timeout=0.05), 0.5, None, True, True, id="rule-timeout"),
        pytest.param(SafetyLimits(rule_timeout=5.0), 0.1, None, False, True, id="within-timeout"),
        pytest.param(
            SafetyLimits(rule_timeout=10.0), 1.0, 0.05, True, True, id="deadline-caps-timeout"
        ),
        pytest.param(SafetyLimits(), 1.0, 0.0, True, False, id="expired-deadline-calls-nothing"),
    ],
)
def test_slow_model_work_is_bounded_by_rule_and_scan_deadlines(
    tmp_path, limits, delay, deadline, timed_out, reached_the_model
) -> None:
    adapter = _SlowAdapter(delay)
    scanner = _scanner(tmp_path, _MODEL_RULE, adapter=adapter, limits=limits)

    result = asyncio.run(scanner.scan_data_async(_ATTACK, deadline_seconds=deadline))

    assert (result.timed_out_rules or []) == (["m"] if timed_out else [])
    assert result.rule_matches == ([] if timed_out else ["m"])
    assert bool(adapter.calls) is reached_the_model


def test_a_local_rule_completes_while_a_model_rule_times_out(tmp_path) -> None:
    scanner = _scanner(
        tmp_path,
        _MODEL_RULE,
        _FAST_RULE,
        adapter=_SlowAdapter(0.5),
        limits=SafetyLimits(rule_timeout=0.05),
    )

    result = asyncio.run(scanner.scan_data_async(_ATTACK))

    assert result.rule_matches == ["fast"]
    assert result.timed_out_rules == ["m"]


def test_concurrent_scans_have_isolated_timeouts(tmp_path) -> None:
    scanner = _scanner(
        tmp_path,
        _MODEL_RULE,
        adapter=_SlowAdapter(1.0, fast_delay=0.01),
        limits=SafetyLimits(rule_timeout=0.3),
    )

    async def run_both():
        return await asyncio.gather(
            scanner.scan_data_async({"messages": [{"role": "user", "content": "attack slow"}]}),
            scanner.scan_data_async({"messages": [{"role": "user", "content": "attack fast"}]}),
        )

    slow, fast = asyncio.run(run_both())

    assert (slow.rule_matches, slow.timed_out_rules) == ([], ["m"])
    assert (fast.rule_matches, fast.timed_out_rules) == (["m"], None)


@pytest.mark.parametrize("api", ["sync", "async"])
def test_fail_on_timeout_raises_with_partial_telemetry(tmp_path, api) -> None:
    adapter = _SlowAdapter(0.5)
    scanner = _scanner(
        tmp_path,
        _MODEL_RULE,
        adapter=adapter,
        limits=SafetyLimits(rule_timeout=0.05, fail_on_timeout=True),
    )

    with pytest.raises(QuarryTimeoutError, match="Rule m timeout") as exc_info:
        if api == "sync":
            scanner.scan_data(_ATTACK)
        else:
            asyncio.run(scanner.scan_data_async(_ATTACK))

    partial = exc_info.value.scan_result
    assert str(exc_info.value) == "Rule m timeout"
    assert adapter.calls == partial.model_usage["calls"] == 1
    assert partial.model_usage["records"] == [{"model": "pi", "operation": "classify", "calls": 1}]
    assert partial.rule_timings["m"] >= 0.0


def test_expired_scan_deadline_raises_with_zero_predispatch_usage(tmp_path) -> None:
    adapter = _SlowAdapter(0.5)
    scanner = _scanner(
        tmp_path,
        _MODEL_RULE,
        adapter=adapter,
        limits=SafetyLimits(rule_timeout=5.0, fail_on_timeout=True),
    )

    with pytest.raises(QuarryTimeoutError, match="Scan deadline exceeded") as exc_info:
        asyncio.run(scanner.scan_data_async(_ATTACK, deadline_seconds=0))

    assert adapter.calls == 0
    assert exc_info.value.scan_result.model_usage == {
        "calls": 0,
        "records": [],
        "truncated": False,
    }


class _BlockingLimiter:
    def __init__(self) -> None:
        self.waiting = asyncio.Event()

    @asynccontextmanager
    async def limit(self, _model):
        self.waiting.set()
        await asyncio.Future()
        yield 0.0


def test_timeout_while_waiting_for_limiter_records_no_adapter_attempt(tmp_path) -> None:
    adapter = _SlowAdapter(0.0)
    limiter = _BlockingLimiter()
    scanner = _scanner(
        tmp_path,
        _MODEL_RULE,
        adapter=adapter,
        limits=SafetyLimits(rule_timeout=0.05, fail_on_timeout=True),
        limiter=limiter,
    )

    with pytest.raises(QuarryTimeoutError, match="Rule m timeout") as exc_info:
        asyncio.run(scanner.scan_data_async(_ATTACK))

    assert limiter.waiting.is_set()
    assert adapter.calls == 0
    assert exc_info.value.scan_result.model_usage["calls"] == 0


def test_file_and_batch_wrappers_keep_raised_timeout_telemetry(tmp_path) -> None:
    conversation = tmp_path / "conversation.json"
    conversation.write_text('{"messages":[{"role":"user","content":"attack"}]}')

    file_adapter = _SlowAdapter(0.5)
    file_scanner = _scanner(
        tmp_path,
        _MODEL_RULE,
        adapter=file_adapter,
        limits=SafetyLimits(rule_timeout=0.05, fail_on_timeout=True),
    )
    file_result = file_scanner.scan_file(conversation)

    assert file_adapter.calls == file_result.model_usage["calls"] == 1
    assert file_result.rule_timings["m"] >= 0.0
    assert file_result.error == "Rule m timeout"

    batch_adapter = _SlowAdapter(0.5)
    batch_scanner = _scanner(
        tmp_path,
        _MODEL_RULE,
        adapter=batch_adapter,
        limits=SafetyLimits(rule_timeout=0.05, fail_on_timeout=True),
    )
    batch_result = batch_scanner.scan_batch_data([_ATTACK], tmp_path / "batch.json")

    assert batch_adapter.calls == batch_result.model_usage["calls"] == 1
    assert batch_result.rule_timings["m"] >= 0.0
    assert batch_result.evaluation_errors == {"conv_1": "Rule m timeout"}


@pytest.mark.parametrize(
    ("timeout", "reason"),
    [
        (0, "positive"),
        (-1, "positive"),
        (math.nan, "finite"),
        (math.inf, "finite"),
        (-math.inf, "finite"),
    ],
)
def test_a_rule_timeout_must_be_positive_and_finite(timeout, reason) -> None:
    with pytest.raises(ValueError, match=reason):
        SafetyLimits(rule_timeout=timeout)


def test_fail_on_timeout_raises_instead_of_reporting(tmp_path) -> None:
    scanner = _scanner(
        tmp_path,
        _MODEL_RULE,
        adapter=_SlowAdapter(0.5),
        limits=SafetyLimits(rule_timeout=0.05, fail_on_timeout=True),
    )

    with pytest.raises(QuarryTimeoutError):
        asyncio.run(scanner.scan_data_async(_ATTACK))
