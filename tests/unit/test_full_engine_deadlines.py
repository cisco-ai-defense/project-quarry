# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Scan deadlines bound configured model work and preserve incomplete verdicts."""

from __future__ import annotations

import asyncio
import math
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


def _scanner(tmp_path, *rules, adapter: _SlowAdapter, limits: SafetyLimits):
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
    backend = ModelInferenceBackend(config, adapters={"fake": adapter})
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


def test_fail_on_timeout_raises_instead_of_reporting(tmp_path) -> None:
    scanner = _scanner(
        tmp_path,
        _MODEL_RULE,
        adapter=_SlowAdapter(0.5),
        limits=SafetyLimits(rule_timeout=0.05, fail_on_timeout=True),
    )

    with pytest.raises(QuarryTimeoutError):
        asyncio.run(scanner.scan_data_async(_ATTACK))


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
