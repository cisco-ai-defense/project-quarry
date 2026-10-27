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

import pytest

from quarry import ConversationValidationError, IncompleteScanError, ScanResult


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
