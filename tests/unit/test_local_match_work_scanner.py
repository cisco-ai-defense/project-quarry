# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Selected local work owns actual scanner decisions, not diagnostic verdict guesses."""

import asyncio
import itertools
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

import quarry.scanner as scanner_module
from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.engine.local_work import current_local_work, local_work_scope
from quarry.integrations import model_usage
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.safety import local_match_work as work
from quarry.safety import timer
from quarry.safety.limits import SafetyLimits
from quarry.safety.timer import QuarryTimeoutError, evidence_deadline
from quarry.scanner import DSLScanner
from quarry.types.exceptions import ConfigurationError
from tests.unit.model_fakes import classification

RULE = 'rule "Selected" condition: any msg in user_messages where msg.content contains "needle"'
OTHER = 'rule "Other" condition: true'
DATA = {"messages": [{"role": "user", "content": "synthetic needle private marker"}]}


def config(*, mode="shadow", allowance=1, rules=("Selected",), texts=(RULE, OTHER)):
    return QuarryConfig(
        config_version=2,
        quarry_rule_texts=texts,
        local_match_work=work.LocalMatchWorkConfig(
            (work.LocalMatchWorkGroup("selected", rules, allowance, mode),)
        ),
    )


@pytest.fixture
def deterministic_cpu(monkeypatch):
    monkeypatch.setattr(work, "thread_time_ns", itertools.count(step=10).__next__)


@pytest.mark.parametrize("raw", [None, False, {}, {"groups": []}, {"groups": True}])
def test_explicit_configuration_is_strict_even_when_falsey(tmp_path, raw):
    with pytest.raises(ValueError):
        QuarryConfig.from_mapping({"local_match_work": raw}, tmp_path, require_rules=False)


def test_mapping_detaches_selected_groups_and_direct_construction_revalidates(tmp_path):
    group = {"label": "selected", "rules": ["Selected"], "allowance_cpu_ns": 100}
    parsed = QuarryConfig.from_mapping(
        {"local_match_work": {"groups": [group]}}, tmp_path, require_rules=False
    )
    group["rules"].clear()
    assert parsed.local_match_work.groups[0].rules == ("Selected",)
    assert parsed.local_match_work.groups[0].mode == "shadow"
    with pytest.raises(ConfigurationError):
        QuarryConfig(local_match_work={"groups": []})
    object.__setattr__(parsed.local_match_work, "groups", ())
    with pytest.raises(ValueError):
        replace(parsed)


def test_unknown_rule_is_rejected_before_provider_construction(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("provider constructed before validating selected rules")

    monkeypatch.setattr(scanner_module, "ResourceManager", unexpected)
    with pytest.raises(ValueError, match="unknown rule"):
        DSLScanner(config(rules=("Missing",)))


@pytest.mark.parametrize("api", ["sync", "async", "compiled"])
@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_actual_verdicts_preserve_shadow_and_make_enforcing_cuts_incomplete(
    api, mode, deterministic_cpu
):
    runtime = config(mode=mode, texts=(RULE,))
    if api == "compiled":
        result = compile_policy(RULE, runtime_config=runtime).scan(DATA)
    else:
        scanner = DSLScanner(runtime)
        result = (
            scanner.scan_data(DATA) if api == "sync" else asyncio.run(scanner.scan_data_async(DATA))
        )
    assert result.is_complete is (mode == "shadow")
    assert result.rule_matches == (["Selected"] if mode == "shadow" else [])
    if mode == "enforce":
        assert result.evaluation_error_categories == {"Selected": "evaluation_limit"}
        assert result.completion_state == "incomplete"
    report = result.local_match_work
    assert report["complete"]  # Measurement completion is distinct from scan completion.
    group = report["groups"][0]
    assert group["closed"] and group["exhausted"] and group["active_segments"] == 0
    assert group["value_cpu_ns"] >= 10 and group["overshoot_cpu_ns"] >= 9
    serialized = json.dumps(report)
    assert "needle" not in serialized and "private marker" not in serialized


def test_unselected_rules_and_distinct_scans_keep_their_own_work(deterministic_cpu):
    scanner = DSLScanner(config(mode="enforce", rules=("Selected",)))
    first, second = scanner.scan_data(DATA), scanner.scan_data(DATA)
    for result in (first, second):
        assert result.rule_matches == ["Other"]
        assert set(result.evaluation_errors) == {"Selected"}
        assert result.local_match_work["groups"][0]["value_cpu_ns"] == 10
    assert first.local_match_work is not second.local_match_work


def test_caught_work_denial_cannot_become_a_completed_nonmatch(monkeypatch, deterministic_cpu):
    async def returns_fallback(*args, **kwargs):
        try:
            current_local_work().call("match", lambda: False)
        except work.LocalMatchWorkError:
            return SimpleNamespace(matched=False, output=None, refine=None, explore=None)
        raise AssertionError("synthetic engine failed to reach its budget cut")

    monkeypatch.setattr(scanner_module, "evaluate_async", returns_fallback)
    runtime = config(mode="enforce", allowance=25, texts=('rule "Selected" condition: true',))
    result = DSLScanner(runtime).scan_data(DATA)
    assert not result.is_complete, "caught work denial became a completed nonmatch"
    assert result.evaluation_error_categories == {"Selected": "evaluation_limit"}


def test_default_nested_scan_clears_and_restores_the_outer_allowance(deterministic_cpu):
    outer = work.LocalMatchWorkOwner(work.LocalMatchWorkGroup("outer", ("Selected",), 1, "enforce"))
    scanner = DSLScanner(QuarryConfig(config_version=2, quarry_rule_texts=(RULE,)))
    with local_work_scope(outer):
        result = scanner.scan_data(DATA)
        assert current_local_work() is outer
    outer.close()
    assert result.rule_matches == ["Selected"] and result.is_complete
    assert result.local_match_work is None
    assert outer.report()["value_cpu_ns"] == 0


def test_disabled_scanner_never_allocates_or_reads_a_cpu_clock(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("disabled scan allocated local work or read the CPU clock")

    monkeypatch.setattr(scanner_module, "LocalMatchWorkSession", unexpected)
    monkeypatch.setattr(work, "thread_time_ns", unexpected)
    result = DSLScanner(QuarryConfig(config_version=2, quarry_rule_texts=(RULE,))).scan_data(DATA)
    assert result.rule_matches == ["Selected"] and result.local_match_work is None


def test_saved_policy_remains_runtime_configured_and_default_bytes_are_stable():
    baseline = compile_policy(RULE, evidence_key=b"synthetic evidence key")
    configured = compile_policy(
        RULE,
        runtime_config=config(allowance=1_000_000_000, texts=(RULE,)),
        evidence_key=b"synthetic evidence key",
    )
    assert configured.to_bytes() == baseline.to_bytes()
    restored = load_compiled_policy(configured.to_bytes(), runtime_config=config(texts=()))
    result = restored.scan(DATA)
    assert result.local_match_work is not None and result.rule_matches == ["Selected"]


async def test_deferred_evidence_keeps_the_owner_until_actual_finish(deterministic_cpu):
    scanner = DSLScanner(config(allowance=1_000_000, texts=(RULE,)))
    deferred = []
    result = await scanner._scan_data_async(
        DATA,
        file_path=None,
        enable_explain=True,
        record_traces=True,
        rules_subset=None,
        parameters=None,
        deferred_evidence=deferred,
    )
    report = result.local_match_work
    assert len(deferred) == 1 and not report["complete"]
    before = report["groups"][0]["known_cpu_ns"]
    with evidence_deadline(10):
        finished = await deferred[0](result)
    assert finished.rule_matches == result.rule_matches == ["Selected"]
    assert finished.local_match_work is report and report["complete"]
    assert report["groups"][0]["value_cpu_ns"] >= before
    with pytest.raises(RuntimeError, match="already finished"):
        await deferred[0](result)


@pytest.mark.parametrize("cancel", [False, True])
async def test_provider_waits_preserve_deadline_cancellation_and_owner_cleanup(
    tmp_path, monkeypatch, cancel
):
    await _assert_provider_wait_cleanup(tmp_path, monkeypatch, cancel, 1024.0)


@pytest.mark.parametrize("clock_start", [1024.0, 1023.9999999999999])
@pytest.mark.parametrize("cancel", [False, True])
async def test_provider_waits_preserve_cleanup_across_clock_origins(
    tmp_path, monkeypatch, cancel, clock_start
):
    await _assert_provider_wait_cleanup(tmp_path, monkeypatch, cancel, clock_start)


def _provider_wait_scanner(tmp_path, adapter, limits):
    model_rule = 'rule "Selected" condition: model("judge").classify(messages) > 0.5'
    runtime = replace(
        config(allowance=1_000_000_000, texts=(model_rule,)),
        models=QuarryConfig.from_mapping(
            {
                "config_version": 2,
                "models": {
                    "judge": {
                        "adapter": "fake",
                        "model": "synthetic",
                        "classify": {"class_index": 1},
                    }
                },
            },
            tmp_path,
            require_rules=False,
        ).models,
    )
    return DSLScanner(runtime, model_adapters={"fake": adapter}, limits=limits)


async def _assert_provider_wait_cleanup(tmp_path, monkeypatch, cancel, clock_start):
    now = clock_start
    sessions, armed = [], []

    def clock():
        return now

    class Recorded(work.LocalMatchWorkSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sessions.append(self)

    class AdmittedAsyncio:
        def __init__(self, kind):
            self.kind = kind

        def timeout(self, seconds):
            owner = timer.current_evaluation_deadline()
            if owner is None:
                assert self.kind == "provider" and len(armed) == 1
                owner = armed[0][3]
            assert owner is not None and owner.error_message == "Rule Selected timeout"
            assert seconds == owner.expires_at - now and seconds > 0
            # This fixture exercises an admitted provider. Preserve its finite
            # configured budget, then expire a real timer after physical admission.
            native = asyncio.timeout(None)
            armed.append((self.kind, asyncio.get_running_loop(), native, owner))
            return native

        def __getattr__(self, name):
            return getattr(asyncio, name)

    monkeypatch.setattr(timer, "time", SimpleNamespace(monotonic=clock))
    monkeypatch.setattr(model_usage, "monotonic", clock)
    monkeypatch.setattr(scanner_module, "asyncio", AdmittedAsyncio("rule"))
    monkeypatch.setattr(model_usage, "asyncio", AdmittedAsyncio("provider"))
    monkeypatch.setattr(scanner_module, "LocalMatchWorkSession", Recorded)
    entered, stopped = asyncio.Event(), asyncio.Event()
    calls = 0

    class Waiting:
        async def classify_messages(self, _config, _messages, _tools, *, timeout=None):
            nonlocal calls
            calls += 1
            assert current_local_work() is None
            assert len(armed) == 2 and {item[0] for item in armed} == {"rule", "provider"}
            assert timeout == armed[0][3].expires_at - now
            entered.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()
            return classification(0.02, 0.98)

    scanner = _provider_wait_scanner(
        tmp_path,
        Waiting(),
        SafetyLimits(rule_timeout=10 if cancel else 0.05, fail_on_timeout=True),
    )
    task = asyncio.create_task(scanner.scan_data_async(DATA))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        _, loop, native, owner = next(item for item in armed if item[0] == "rule")
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not native.expired()
        else:
            now = owner.expires_at
            loop.call_soon(native.reschedule, loop.time())
            with pytest.raises(QuarryTimeoutError, match="Rule Selected timeout") as caught:
                await task
            assert native.expired()
            assert caught.value.scan_result.local_match_work["complete"]
        await asyncio.wait_for(stopped.wait(), 10)
        assert calls == 1 and len(sessions) == 1 and sessions[0].report()["complete"]
        assert sessions[0].report()["groups"][0]["active_segments"] == 0
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("clock_start", [1024.0, 1023.9999999999999])
async def test_expired_provider_admission_finalizes_local_work_without_dispatch(
    tmp_path, monkeypatch, clock_start
):
    now = clock_start
    sessions, reached = [], []

    class Recorded(work.LocalMatchWorkSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sessions.append(self)

    class Forbidden:
        async def classify_messages(self, *args, **kwargs):
            pytest.fail("expired local-work evaluation dispatched a provider")

    def arm(seconds):
        owner = timer.current_evaluation_deadline()
        assert owner is not None and seconds == owner.expires_at - now
        return asyncio.timeout(None)

    class ScannerAsyncio:
        timeout = staticmethod(arm)

        def __getattr__(self, name):
            return getattr(asyncio, name)

    original = ModelInferenceBackend.model_classify_messages

    async def expire_before_admission(self, *args, **kwargs):
        nonlocal now
        owner = timer.current_evaluation_deadline()
        assert owner is not None and owner.error_message == "Rule Selected timeout"
        now = owner.expires_at
        reached.append(True)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(timer, "time", SimpleNamespace(monotonic=lambda: now))
    monkeypatch.setattr(model_usage, "monotonic", lambda: now)
    monkeypatch.setattr(scanner_module, "asyncio", ScannerAsyncio())
    monkeypatch.setattr(scanner_module, "LocalMatchWorkSession", Recorded)
    monkeypatch.setattr(ModelInferenceBackend, "model_classify_messages", expire_before_admission)
    scanner = _provider_wait_scanner(
        tmp_path, Forbidden(), SafetyLimits(rule_timeout=0.05, fail_on_timeout=True)
    )
    with pytest.raises(QuarryTimeoutError, match="Rule Selected timeout") as caught:
        await scanner.scan_data_async(DATA)
    assert reached == [True] and caught.value.scan_result.local_match_work["complete"]
    assert len(sessions) == 1 and sessions[0].report()["complete"]
    assert sessions[0].report()["groups"][0]["active_segments"] == 0
