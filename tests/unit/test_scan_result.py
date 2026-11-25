# Copyright 2026 Cisco Systems, Inc. and its affiliates
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

"""Tests for authoritative, invalid, and incomplete scan results."""

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import pytest

from quarry import ConversationValidationError, IncompleteScanError, ScanResult


def _uninitialized_scan_result() -> ScanResult:
    return ScanResult.__new__(ScanResult)


class _LegacyScanResultPickle:
    """Emit a historical ScanResult state through pickle's normal state hook."""

    def __init__(self, state: dict) -> None:
        self.state = state

    def __reduce__(self):
        return (_uninitialized_scan_result, (), self.state)


def result_with(**overrides) -> ScanResult:
    values = {
        "file_path": None,
        "rule_matches": [],
        "scan_time": 0.01,
    }
    values.update(overrides)
    return ScanResult(**values)


def test_complete_result_is_authoritative() -> None:
    result = result_with(rule_matches=["Matched Rule"])

    assert result.completion_state == "complete"
    assert result.is_complete is True
    assert result.completion_issues == []
    assert result.raise_for_status() is None


def test_invalid_result_raises_typed_validation_error_with_result() -> None:
    result = result_with(
        error="Conversation validation error: missing messages",
        validation_error="missing messages",
    )

    assert result.completion_state == "invalid"
    with pytest.raises(ConversationValidationError) as exc_info:
        result.raise_for_status()

    assert exc_info.value.scan_result is result
    assert exc_info.value.result is result
    assert "missing messages" in str(exc_info.value)


def test_telemetry_defaults_preserve_legacy_construction_and_are_independent() -> None:
    first = ScanResult(None, [], 0.0)
    second = ScanResult(None, [], 0.0)

    assert first.rule_timings is second.rule_timings is None
    assert (
        first.model_usage
        == second.model_usage
        == {
            "calls": 0,
            "records": [],
            "truncated": False,
        }
    )
    first.model_usage["records"].append({"model": "judge", "operation": "score", "calls": 1})
    assert second.model_usage["records"] == []


def test_scan_result_telemetry_round_trips_through_json_and_pickle() -> None:
    result = result_with(
        rule_timings={"Rule": 0.005},
        model_usage={
            "calls": 1,
            "records": [{"model": "judge", "operation": "score", "calls": 1}],
            "truncated": False,
        },
    )

    json_payload = json.loads(json.dumps(asdict(result), sort_keys=True))
    assert json_payload["rule_timings"] == {"Rule": 0.005}
    assert json_payload["model_usage"] == result.model_usage

    restored = pickle.loads(pickle.dumps(result))
    assert restored == result
    assert restored.model_usage is not result.model_usage
    assert restored.model_usage["records"] is not result.model_usage["records"]


def test_legacy_pickle_without_telemetry_receives_independent_defaults() -> None:
    state = dict(result_with().__dict__)
    state.pop("model_usage")
    state.pop("rule_timings")

    first = pickle.loads(pickle.dumps(_LegacyScanResultPickle(state)))
    second = pickle.loads(pickle.dumps(_LegacyScanResultPickle(state)))

    assert isinstance(first, ScanResult)
    assert first.rule_timings is second.rule_timings is None
    assert (
        first.model_usage
        == second.model_usage
        == {
            "calls": 0,
            "records": [],
            "truncated": False,
        }
    )
    first.model_usage["records"].append({"model": "judge", "operation": "score", "calls": 1})
    assert second.model_usage["records"] == []


@pytest.mark.parametrize(
    "missing_observations",
    [
        ("logical_reads",),
        ("window_observations",),
        ("logical_reads", "window_observations"),
    ],
    ids=["before-read-reports", "before-window-reports", "before-both-reports"],
)
@pytest.mark.parametrize("before_telemetry", [False, True])
def test_older_pickles_restore_optional_reports_without_losing_results(
    missing_observations, before_telemetry
) -> None:
    usage = {
        "calls": 2,
        "records": [{"model": "judge", "operation": "score", "calls": 2}],
        "truncated": False,
    }
    state = dict(
        result_with(
            rule_matches=["Read"],
            rule_timings={"Read": 0.005},
            model_usage=usage,
            logical_reads={"synthetic_read": 1},
            window_observations={"synthetic_window": 2},
            local_match_work={"synthetic_work": 3},
        ).__dict__
    )
    for name in missing_observations:
        state.pop(name)
    if before_telemetry:
        state.pop("model_usage")
        state.pop("local_match_work")
    wire = pickle.dumps(_LegacyScanResultPickle(state))

    first = pickle.loads(wire)
    second = pickle.loads(wire)

    expected_usage = {"calls": 0, "records": [], "truncated": False} if before_telemetry else usage
    for restored in (first, second):
        assert restored.rule_matches == ["Read"]
        assert restored.is_complete
        assert restored.rule_timings == {"Read": 0.005}
        assert restored.model_usage == expected_usage
        assert restored.logical_reads == (
            None if "logical_reads" in missing_observations else {"synthetic_read": 1}
        )
        assert restored.window_observations == (
            None if "window_observations" in missing_observations else {"synthetic_window": 2}
        )
        assert restored.local_match_work == (None if before_telemetry else {"synthetic_work": 3})
    first.model_usage["records"].append({"model": "other", "operation": "score", "calls": 1})
    assert second.model_usage == expected_usage


@pytest.mark.parametrize("missing_required", ["rule_matches", "error", "validation_error"])
def test_older_pickles_still_require_verdict_and_failure_fields(missing_required) -> None:
    state = dict(result_with().__dict__)
    for name in ("logical_reads", "window_observations", "local_match_work", "model_usage"):
        state.pop(name)
    state.pop(missing_required)
    wire = pickle.dumps(_LegacyScanResultPickle(state))

    with pytest.raises(ValueError, match="missing required fields"):
        pickle.loads(wire)


def test_direct_legacy_state_rejects_unknown_fields() -> None:
    state = dict(result_with().__dict__)
    state.pop("model_usage")
    state["unexpected"] = "not restored"

    with pytest.raises(ValueError, match="unknown fields"):
        _uninitialized_scan_result().__setstate__(state)


def test_python_api_documents_scan_result_telemetry_contract() -> None:
    documentation = (Path(__file__).parents[2] / "docs" / "python-api.md").read_text(
        encoding="utf-8"
    )

    assert all(
        term in documentation
        for term in (
            "ScanResult.rule_timings",
            "ScanResult.model_usage",
            "QuarryTimeoutError",
            "[scan telemetry](PLUGIN.md#scan-telemetry)",
        )
    )


@pytest.mark.parametrize(
    "incomplete_fields",
    [
        {"error": "file read failed"},
        {"was_truncated": True},
        {"timed_out_rules": ["Slow Rule"]},
        {"skipped_rules": ["Broken Rule"]},
        {"evaluation_errors": {"Broken Rule": "TypeError: invalid value"}},
        {"evaluation_warnings": ["AnyComprehension truncated"]},
    ],
)
def test_incomplete_signals_are_never_authoritative(incomplete_fields) -> None:
    result = result_with(**incomplete_fields)

    assert result.completion_state == "incomplete"
    assert result.is_complete is False
    with pytest.raises(IncompleteScanError) as exc_info:
        result.raise_for_status()
    assert exc_info.value.scan_result is result
