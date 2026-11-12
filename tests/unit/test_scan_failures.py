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

"""Failure taxonomy contract and exhaustive completion-state invariants."""

import asyncio
import csv
import itertools
import json
from dataclasses import FrozenInstanceError
from io import StringIO
from pathlib import Path

import pytest

from quarry import (
    ConversationValidationError,
    IncompleteScanError,
    ScanFailure,
    ScanResult,
)
from quarry import (
    ScanFailureCategory as Category,
)
from quarry.cli import format_results
from quarry.config import QuarryConfig
from quarry.plugin.service import QuarryPlugin
from quarry.safety.limits import SafetyLimits
from quarry.safety.timer import QuarryTimeoutError
from quarry.scanner import DSLScanner, _failure_category
from quarry.types.exceptions import (
    EvaluationError,
    EvaluationTimeoutError,
    LLMError,
    ModelError,
    ResourceError,
)

CONVERSATION = {"messages": [{"role": "user", "content": "hello"}]}


def result_with(**fields):
    return ScanResult(file_path=None, rule_matches=[], scan_time=0.0, **fields)


def scanner():
    return DSLScanner(
        QuarryConfig(
            quarry_rule_texts=('rule "Check"\ncondition: true\n',),
            safety_limits=SafetyLimits(rule_timeout=None),
        )
    )


def test_failure_category_wire_values_remain_stable():
    """Member refactors must preserve the public serialization contract."""
    assert [category.value for category in Category] == [
        "invalid_input",
        "rule_selection",
        "input_read",
        "preparation",
        "model",
        "resource",
        "evaluation",
        "timeout",
        "skipped",
        "truncated_input",
        "evaluation_limit",
        "unknown",
    ]


@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=7)))
def test_exhaustive_completion_state_and_failure_presence(flags):
    """Check every subset of the seven legacy completion signals (128 cases)."""
    invalid, error, evaluation, timeout, skipped, truncated, limited = flags
    result = result_with(
        validation_error="" if invalid else None,
        error="" if error else None,
        evaluation_errors={"Check": ""} if evaluation else None,
        timed_out_rules=["Check"] if timeout else None,
        skipped_rules=["Check"] if skipped else None,
        was_truncated=truncated,
        evaluation_warnings=[""] if limited else None,
    )
    expected = "invalid" if invalid else "incomplete" if any(flags) else "complete"
    assert result.completion_state == expected
    assert bool(result.failures) is (expected != "complete")
    assert all(isinstance(failure.category, Category) for failure in result.failures)
    if expected == "complete":
        assert result.raise_for_status() is None
    else:
        exception = ConversationValidationError if invalid else IncompleteScanError
        with pytest.raises(exception) as caught:
            result.raise_for_status()
        assert caught.value.details["failures"] == [f.to_dict() for f in result.failures]


@pytest.mark.parametrize(
    "error,expected",
    [
        (LLMError("provider text", {"code": "timeout"}), Category.MODEL),
        (ModelError("arbitrary"), Category.MODEL),
        (ResourceError("arbitrary"), Category.RESOURCE),
        (EvaluationError("arbitrary"), Category.EVALUATION),
        (ConversationValidationError("arbitrary"), Category.INVALID_INPUT),
        (QuarryTimeoutError("arbitrary"), Category.TIMEOUT),
        (TimeoutError("arbitrary"), Category.TIMEOUT),
        (EvaluationTimeoutError("Check", 1), Category.TIMEOUT),
        (ValueError("authentication timeout resource limit"), Category.UNKNOWN),
        (Exception("arbitrary"), Category.UNKNOWN),
    ],
)
def test_classification_uses_exception_types_only(error, expected):
    assert _failure_category(error) is expected


def test_failure_record_is_immutable_and_serialization_is_detached():
    failure = ScanFailure(Category.MODEL, "Check")
    with pytest.raises(FrozenInstanceError):
        failure.rule_name = "Other"
    data = failure.to_dict()
    data["category"] = "changed"
    assert failure.to_dict() == {"category": "model", "rule_name": "Check"}
    assert json.loads(json.dumps(failure.to_dict())) == failure.to_dict()
    with pytest.raises(ValueError):
        ScanFailure("provider-defined-category")
    with pytest.raises(TypeError):
        ScanFailure(Category.UNKNOWN, 3)


def test_legacy_results_and_later_mutation_keep_failure_projection_current():
    result = result_with()
    assert result.failures == ()
    result.evaluation_errors = {"Check": "provider content"}
    assert result.failures == (ScanFailure(Category.EVALUATION, "Check"),)
    result.evaluation_error_categories = {"Check": Category.MODEL}
    assert result.failures == (ScanFailure(Category.MODEL, "Check"),)
    result.evaluation_errors.clear()
    result.error_category = Category.MODEL
    assert result.is_complete
    assert result.failures == ()


@pytest.mark.parametrize(
    "error,expected",
    [
        (LLMError("synthetic provider reply", {"endpoint": "not returned"}), Category.MODEL),
        (ResourceError("synthetic failure"), Category.RESOURCE),
        (ValueError("synthetic failure"), Category.UNKNOWN),
        (QuarryTimeoutError("synthetic timeout"), Category.TIMEOUT),
        (EvaluationTimeoutError("Check", 1), Category.TIMEOUT),
    ],
)
def test_rule_failures_keep_type_without_raw_details(monkeypatch, error, expected):
    async def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr("quarry.scanner.evaluate_async", fail)
    result = scanner().scan_data(CONVERSATION)
    assert result.completion_state == "incomplete"
    assert result.failures == (ScanFailure(expected, "Check"),)
    assert "synthetic" not in json.dumps([f.to_dict() for f in result.failures])
    assert "endpoint" not in json.dumps([f.to_dict() for f in result.failures])


def test_cancellation_is_not_converted_to_a_scan_failure(monkeypatch):
    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr("quarry.scanner.evaluate_async", cancel)
    with pytest.raises(asyncio.CancelledError):
        scanner().scan_data(CONVERSATION)


def test_rule_selection_and_invalid_input_are_distinct():
    subject = scanner()
    invalid = subject.scan_data({"messages": []})
    assert invalid.failures == (ScanFailure(Category.INVALID_INPUT),)
    selection = asyncio.run(subject.scan_data_async(CONVERSATION, rules_subset={"Missing"}))
    assert selection.failures == (ScanFailure(Category.RULE_SELECTION),)


def test_file_read_and_json_decode_failures(tmp_path):
    missing = scanner().scan_file(tmp_path / "missing.json")
    assert missing.failures == (ScanFailure(Category.INPUT_READ),)
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    assert scanner().scan_file(malformed).failures == (ScanFailure(Category.INPUT_READ),)


def test_batch_preserves_child_categories_and_provenance(monkeypatch):
    subject = scanner()
    children = iter(
        [
            result_with(validation_error="invalid"),
            result_with(
                evaluation_errors={"Check": "failure"},
                evaluation_error_categories={"Check": Category.MODEL},
            ),
            result_with(timed_out_rules=["Check"]),
            result_with(error="selection failed", error_category=Category.RULE_SELECTION),
        ]
    )
    monkeypatch.setattr(subject, "scan_data", lambda *args: next(children))
    result = subject.scan_batch_data([CONVERSATION] * 4, Path("batch.json"))
    assert not result.is_complete
    failures = set(result.failures)
    assert ScanFailure(Category.INVALID_INPUT, "conv_1") in failures
    assert ScanFailure(Category.MODEL, "conv_2:Check") in failures
    assert ScanFailure(Category.TIMEOUT, "conv_3:Check") in failures
    assert ScanFailure(Category.RULE_SELECTION, "conv_4") in failures


def test_cli_json_and_csv_expose_same_categories():
    result = result_with(
        evaluation_errors={"Check": "failure"},
        evaluation_error_categories={"Check": Category.MODEL},
    )
    expected = [{"category": "model", "rule_name": "Check"}]
    assert json.loads(format_results([result], "json"))[0]["failures"] == expected
    row = next(csv.DictReader(StringIO(format_results([result], "csv"))))
    assert json.loads(row["failures"]) == expected


@pytest.mark.parametrize("field", ["validation_error", "error"])
@pytest.mark.parametrize("message", ["", "failure"])
def test_batch_cannot_turn_a_failed_child_into_a_complete_result(monkeypatch, field, message):
    subject = scanner()
    child = result_with(**{field: message})
    monkeypatch.setattr(subject, "scan_data", lambda *args: child)
    batch = subject.scan_batch_data([CONVERSATION], Path("batch.json"))
    assert batch.completion_state == "incomplete"
    expected = Category.INVALID_INPUT if field == "validation_error" else Category.UNKNOWN
    assert ScanFailure(expected, "conv_1") in batch.failures


def test_plugin_scan_payload_uses_the_public_failure_contract():
    subject = scanner()
    result = result_with(
        evaluation_errors={"Check": "failure"},
        evaluation_error_categories={"Check": Category.MODEL},
    )
    plugin = QuarryPlugin()
    payload = plugin._scan_payload(subject, result, None)
    assert payload["failures"] == [{"category": "model", "rule_name": "Check"}]
    assert payload["completion_state"] == "incomplete"
    assert "matched" not in payload


def test_new_attribution_fields_preserve_positional_constructor_compatibility():
    result = ScanResult(None, ["Check"], 0.25, "failure", None, False)
    assert result.rule_matches == ["Check"]
    assert result.failures == (ScanFailure(Category.UNKNOWN),)


def test_empty_batch_is_typed_invalid_input():
    result = scanner().scan_batch_data([], Path("batch.json"))
    assert result.completion_state == "invalid"
    assert result.failures == (ScanFailure(Category.INVALID_INPUT),)


@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=7)))
def test_exhaustive_batch_completion_preserves_every_child_failure(monkeypatch, flags):
    """Finite-state check: a batch is complete iff its only child is complete."""
    invalid, error, evaluation, timeout, skipped, truncated, limited = flags
    child = result_with(
        validation_error="" if invalid else None,
        error="" if error else None,
        evaluation_errors={"Check": ""} if evaluation else None,
        timed_out_rules=["Check"] if timeout else None,
        skipped_rules=["Check"] if skipped else None,
        was_truncated=truncated,
        evaluation_warnings=[""] if limited else None,
    )
    subject = scanner()
    monkeypatch.setattr(subject, "scan_data", lambda *args: child)
    batch = subject.scan_batch_data([CONVERSATION], Path("batch.json"))
    assert batch.is_complete is child.is_complete
    assert bool(batch.failures) is (not child.is_complete)
    for failure in child.failures:
        assert failure.category in {f.category for f in batch.failures}
