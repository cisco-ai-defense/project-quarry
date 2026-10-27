# Copyright 2026 Cisco Systems, Inc. and its affilitates
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

"""The public result contracts: scan completion and typed collection outputs.

Completion (``completion_state`` / ``is_complete`` / ``raise_for_status``) is what lets a guard
fail closed on an incomplete scan instead of returning a partial match list that reads as a
clean pass. Collection results serialize through one explicit ``to_dict`` wire shape.
"""

from dataclasses import dataclass

import pytest

from quarry import (
    CollectionResult,
    ConversationValidationError,
    IncompleteScanError,
    ProducerRef,
    ResultItem,
    ScanResult,
    Score,
    Span,
)


def result_with(**overrides) -> ScanResult:
    return ScanResult(**{"file_path": None, "rule_matches": [], "scan_time": 0.01, **overrides})


def test_a_complete_result_is_authoritative() -> None:
    result = result_with(rule_matches=["Matched Rule"])

    assert (result.completion_state, result.is_complete, result.completion_issues) == (
        "complete",
        True,
        [],
    )
    assert result.raise_for_status() is None


def test_the_positional_error_argument_remains_compatible() -> None:
    result = ScanResult(None, [], 0.01, "file read failed")

    assert (result.error, result.outputs, result.completion_state) == (
        "file read failed",
        {},
        "incomplete",
    )


def test_an_invalid_result_raises_a_validation_error_carrying_it() -> None:
    result = result_with(validation_error="missing messages")

    assert result.completion_state == "invalid"
    with pytest.raises(ConversationValidationError, match="missing messages") as error:
        result.raise_for_status()
    assert error.value.scan_result is result


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

    assert (result.completion_state, result.is_complete) == ("incomplete", False)
    with pytest.raises(IncompleteScanError) as error:
        result.raise_for_status()
    assert error.value.scan_result is result


@dataclass(frozen=True)
class _Value:
    coordinate: int

    def to_dict(self) -> dict[str, int]:
        return {"coordinate": self.coordinate}


@pytest.mark.parametrize(
    ("value", "wire"),
    [
        (_Value(7), {"coordinate": 7}),
        (
            Span(message_index=2, start=4, end=11, score=Score(0.875)),
            {"message_index": 2, "start": 4, "end": 11, "score": {"value": 0.875}},
        ),
    ],
    ids=["custom-value", "span"],
)
def test_a_collection_result_serializes_to_its_wire_shape(value, wire) -> None:
    result = CollectionResult(
        output_type="test_collection",
        category="TEST",
        source=ProducerRef(kind="model", name="test-model", run_id="run_123"),
        items=(ResultItem(item_id="item_123", value=value),),
    )

    assert result.to_dict() == {
        "type": "test_collection",
        "category": "TEST",
        "source": {"kind": "model", "name": "test-model", "run_id": "run_123"},
        "items": [{"item_id": "item_123", "value": wire}],
    }


@pytest.mark.parametrize("value", [-0.01, 1.01, float("inf"), float("nan"), True])
def test_a_score_outside_the_probability_range_is_rejected(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        Score(value)


@pytest.mark.parametrize(
    ("message_index", "start", "end"),
    [(-1, 0, 1), (0, -1, 1), (0, 1, 1), (0, 2, 1), (True, 0, 1)],
    ids=["negative-message", "negative-start", "empty", "reversed", "bool-index"],
)
def test_a_span_with_invalid_coordinates_is_rejected(message_index, start, end) -> None:
    with pytest.raises((TypeError, ValueError)):
        Span(message_index=message_index, start=start, end=end, score=Score(0.5))
