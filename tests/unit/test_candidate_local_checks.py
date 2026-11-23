# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Caller registrations, all-channel pre-check bounds and actual native execution."""

import re
import threading

import pytest

from quarry.integrations.candidate_checks import CandidateChannelChecks, LocalCandidatePattern
from quarry.integrations.candidate_providers import HostCandidateProvider, LocalCandidateProvider
from quarry.integrations.candidate_worker import CandidateWorkLimits
from quarry.types.candidates import CandidateBatchIdentity, CandidateReplyError
from tests.unit.test_candidate_batch_runtime import DATA, policy, response

IDENTITY = CandidateBatchIdentity("synthetic-candidates", "1", "heads")


def local(*patterns, checks=None, work=None):
    return LocalCandidateProvider(
        IDENTITY, ("left", "right"), patterns, checks=checks, work=work or CandidateWorkLimits()
    )


@pytest.mark.parametrize("restored", [False, True])
def test_local_aliases_share_one_native_batch_with_zero_billable_calls(
    tmp_path, monkeypatch, restored
):
    seen = []
    provider = local(
        LocalCandidatePattern("left", "abc", ("accept",)),
        LocalCandidatePattern("right", "cd", ("accept",)),
        checks={"accept": lambda text: seen.append((text, threading.get_ident())) or True},
    )
    physical = []
    run = provider._worker.run

    async def counted(*args, **kwargs):
        physical.append(True)
        return await run(*args, **kwargs)

    monkeypatch.setattr(provider._worker, "run", counted)
    try:
        result = policy(tmp_path, None, provider=provider, restored=restored).scan(DATA)
        assert result.is_complete, result.evaluation_errors
        assert result.rule_matches == ["Read"] and len(physical) == 1
        assert [item[0] for item in seen] == ["abc", "cd"]
        assert all(thread != threading.get_ident() for _, thread in seen)
        assert result.model_usage["calls"] == 0 and result.model_usage["records"] == []
        span = result.outputs["Read"].items[0].value
        assert (span.start, span.end, span.precision) == (2, 4, "exact")
    finally:
        provider.close()


@pytest.mark.parametrize("restored", [False, True])
def test_local_counts_all_candidates_before_even_rejecting_checks(tmp_path, restored):
    checked = []
    provider = local(
        LocalCandidatePattern("left", ".", ("reject",)),
        checks={"reject": lambda text: checked.append(text) or False},
        work=CandidateWorkLimits(max_candidates_per_channel=1),
    )
    try:
        result = policy(tmp_path, None, provider=provider, restored=restored).scan(DATA)
        assert not result.is_complete and not result.outputs
        assert result.evaluation_error_categories["Read"] == "evaluation_limit"
        assert checked == []
    finally:
        provider.close()


@pytest.mark.parametrize("restored", [False, True])
def test_host_checks_validate_every_channel_before_invoking_a_check(tmp_path, restored):
    checked = []
    provider = HostCandidateProvider(
        IDENTITY,
        ("left", "right"),
        "0" * 64,
        lambda request, **kwargs: response(request, invalid=True),
        channel_checks=(CandidateChannelChecks("left", ("accept",)),),
        checks={"accept": lambda text: checked.append(text) or True},
    )
    try:
        result = policy(tmp_path, None, provider=provider, restored=restored).scan(DATA)
        assert not result.is_complete and not result.outputs
        assert result.evaluation_error_categories["Read"] == "model_invalid_output"
        assert checked == []
    finally:
        provider.close()


def test_host_execution_candidate_bound_precedes_rejecting_checks(tmp_path):
    checked = []
    provider = HostCandidateProvider(
        IDENTITY,
        ("left", "right"),
        "0" * 64,
        lambda request, **kwargs: response(request, excessive=True),
        channel_checks=(CandidateChannelChecks("left", ("reject",)),),
        checks={"reject": lambda text: checked.append(text) or False},
        work=CandidateWorkLimits(max_candidates_per_channel=1),
    )
    try:
        result = policy(tmp_path, None, provider=provider).scan(DATA)
        assert not result.is_complete
        assert result.evaluation_error_categories["Read"] == "evaluation_limit" and checked == []
    finally:
        provider.close()


@pytest.mark.parametrize("restored", [False, True])
def test_host_named_checks_filter_native_ranges_on_worker_before_admission(tmp_path, restored):
    checked = []

    def accept(text):
        checked.append((text, threading.get_ident()))
        return text == "abc"

    provider = HostCandidateProvider(
        IDENTITY,
        ("left", "right"),
        "0" * 64,
        lambda request, **kwargs: response(request),
        channel_checks=(
            CandidateChannelChecks("left", ("synthetic",)),
            CandidateChannelChecks("right", ("synthetic",)),
        ),
        checks={"synthetic": accept},
    )
    try:
        result = policy(tmp_path, None, provider=provider, restored=restored).scan(DATA)
        # The native collect contract omits a rule with no collected survivors.
        assert result.is_complete and result.rule_matches == [] and result.outputs == {}
        assert [text for text, _ in checked] == ["abc", "c"]
        assert all(thread != threading.get_ident() for _, thread in checked)
        assert result.model_usage["calls"] == 1
    finally:
        provider.close()


@pytest.mark.parametrize("kind", ["local", "host"])
@pytest.mark.parametrize("failure", ["raises", "not_bool"])
def test_check_failures_are_sanitized_and_never_authoritative(tmp_path, kind, failure):
    def check(text):
        if failure == "raises":
            raise RuntimeError("synthetic private callback detail")
        return 1

    checks = {"synthetic": check}
    provider = (
        local(LocalCandidatePattern("left", "abc", ("synthetic",)), checks=checks)
        if kind == "local"
        else HostCandidateProvider(
            IDENTITY,
            ("left", "right"),
            "0" * 64,
            lambda request, **kwargs: response(request),
            channel_checks=(CandidateChannelChecks("left", ("synthetic",)),),
            checks=checks,
        )
    )
    try:
        result = policy(tmp_path, None, provider=provider).scan(DATA)
        assert not result.is_complete and not result.outputs and not result.rule_matches
        assert result.evaluation_error_categories["Read"] == "model_provider"
        assert "private callback detail" not in str(result.evaluation_errors)
    finally:
        provider.close()


@pytest.mark.parametrize("kind", ["local", "host"])
def test_effective_settings_include_checks_and_execution_bounds(kind):
    def provider(*, names=(), work=None):
        checks = {"synthetic": lambda text: True}
        if kind == "local":
            return local(LocalCandidatePattern("left", "abc", names), checks=checks, work=work)
        return HostCandidateProvider(
            IDENTITY,
            ("left", "right"),
            "0" * 64,
            lambda request, **kwargs: response(request),
            channel_checks=(CandidateChannelChecks("left", names),),
            checks=checks,
            work=work or CandidateWorkLimits(),
        )

    providers = [
        provider(),
        provider(names=("synthetic",)),
        provider(work=CandidateWorkLimits(max_work=1000)),
    ]
    try:
        assert len({p.settings_digest for p in providers}) == 3
    finally:
        for item in providers:
            item.close()


def test_local_native_multipart_and_unicode_ranges_do_not_join_sources(tmp_path):
    provider = local(
        LocalCandidatePattern("left", "雪", flags=int(re.IGNORECASE)),
        LocalCandidatePattern("right", "🙂"),
        LocalCandidatePattern("right", "雪🙂"),
    )
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a雪"},
                    {"type": "text", "text": "🙂b"},
                ],
            }
        ]
    }
    try:
        result = policy(tmp_path, None, provider=provider).scan(data)
        assert result.is_complete and result.rule_matches == ["Read"]
        spans = [item.value for item in result.outputs["Read"].items]
        assert len(spans) == 1
        assert (spans[0].start, spans[0].end) == (2, 3)
        assert spans[0].source_ranges[0].part_index == 1
    finally:
        provider.close()


@pytest.mark.parametrize("pattern", ["(?<=a)b", "(a)\\1", "(?=a)"])
def test_local_patterns_never_fall_back_from_re2(tmp_path, pattern):
    provider = local(LocalCandidatePattern("left", pattern))
    try:
        result = policy(tmp_path, None, provider=provider).scan(DATA)
        assert not result.is_complete and not result.rule_matches
    finally:
        provider.close()


@pytest.mark.parametrize("kind", ["local", "host"])
def test_missing_named_check_is_refused_at_registration(kind):
    with pytest.raises(CandidateReplyError, match="unregistered"):
        if kind == "local":
            local(LocalCandidatePattern("left", "abc", ("missing",)))
        else:
            HostCandidateProvider(
                IDENTITY,
                ("left", "right"),
                "0" * 64,
                lambda *args, **kwargs: None,
                channel_checks=(CandidateChannelChecks("left", ("missing",)),),
            )
