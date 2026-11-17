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

import orjson
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
from quarry.integrations.resources import ResourcePreparationError
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
        (ResourcePreparationError("arbitrary", {}), Category.RESOURCE),
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
        (ResourcePreparationError("synthetic failure", {}), Category.RESOURCE),
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


_JSON_METADATA = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(max_size=30),
    lambda values: (
        st.lists(values, max_size=3) | st.dictionaries(st.text(max_size=10), values, max_size=3)
    ),
    max_leaves=8,
)


@pytest.fixture(scope="module")
def payload_serializers():
    return scanner(), QuarryPlugin()


def assert_failure_serializers_agree(result, subject, plugin):
    expected = [failure.to_dict() for failure in result.failures]
    assert result.completion_state != "complete"
    with pytest.raises(IncompleteScanError) as caught:
        result.raise_for_status()
    assert caught.value.details["failures"] == expected
    assert json.loads(format_results([result], "json"))[0]["failures"] == expected
    row = next(csv.DictReader(StringIO(format_results([result], "csv"))))
    assert json.loads(row["failures"]) == expected
    assert plugin._scan_payload(subject, result, None)["failures"] == expected


@settings(max_examples=50, deadline=None)
@given(metadata=_JSON_METADATA | st.sampled_from([category.value for category in Category]))
def test_mutated_error_metadata_cannot_break_fail_closed_serialization(
    payload_serializers, metadata
):
    subject, plugin = payload_serializers
    result = result_with(error="")
    result.error_category = metadata
    assert_failure_serializers_agree(result, subject, plugin)
    expected = (
        Category(metadata)
        if type(metadata) is str and metadata in {category.value for category in Category}
        else Category.UNKNOWN
    )
    assert result.failures == (ScanFailure(expected),)
    result.error = None
    assert result.is_complete
    assert result.failures == ()


@settings(max_examples=50, deadline=None)
@given(metadata=_JSON_METADATA | st.sampled_from([category.value for category in Category]))
def test_mutated_rule_metadata_cannot_break_fail_closed_serialization(
    payload_serializers, metadata
):
    subject, plugin = payload_serializers
    result = result_with(evaluation_errors={"Check": ""})
    result.evaluation_error_categories = {"Check": metadata}
    assert_failure_serializers_agree(result, subject, plugin)
    expected = (
        Category(metadata)
        if type(metadata) is str and metadata in {category.value for category in Category}
        else Category.UNKNOWN
    )
    assert result.failures == (ScanFailure(expected, "Check"),)
    result.evaluation_errors.clear()
    assert result.is_complete
    assert result.failures == ()


@pytest.mark.parametrize("metadata", [[], ["model"], True, 3, "model", {"Other": "model"}, None])
def test_malformed_category_mapping_preserves_fail_closed_batch(monkeypatch, metadata):
    child = result_with(evaluation_errors={"Check": ""})
    child.evaluation_error_categories = metadata
    expected = (
        Category.EVALUATION if metadata is None or isinstance(metadata, dict) else Category.UNKNOWN
    )
    assert child.failures == (ScanFailure(expected, "Check"),)
    subject = scanner()
    monkeypatch.setattr(subject, "scan_data", lambda *args: child)
    batch = subject.scan_batch_data([CONVERSATION], Path("batch.json"))
    assert batch.completion_state == "incomplete"
    assert batch.failures == (ScanFailure(expected, "conv_1:Check"),)


def test_failure_projection_never_coerces_arbitrary_metadata():
    class NoCoercion:
        @property
        def __class__(self):
            pytest.fail("category metadata must not inspect a spoofable class attribute")

        def __str__(self):
            pytest.fail("category metadata must not be stringified")

        def __bool__(self):
            pytest.fail("category metadata must not be tested for truthiness")

        def __hash__(self):
            pytest.fail("category metadata must not be hashed")

    class NoOverrides(dict):
        def __bool__(self):
            pytest.fail("category dictionaries must not be tested for truthiness")

        def get(self, *args):
            pytest.fail("category dictionaries must not execute overridden get")

        def __contains__(self, key):
            pytest.fail("category dictionaries must not execute overridden contains")

        def __getitem__(self, key):
            pytest.fail("category dictionaries must not execute overridden getitem")

    result = result_with(error="", evaluation_errors={"Check": ""})
    result.error_category = NoCoercion()
    result.evaluation_error_categories = NoOverrides(Check=NoCoercion())
    assert result.failures == (
        ScanFailure(Category.UNKNOWN),
        ScanFailure(Category.UNKNOWN, "Check"),
    )


def test_batch_keeps_top_level_category_alongside_rule_error_category(monkeypatch):
    child = result_with(
        error="preparation failed",
        error_category=Category.PREPARATION,
        evaluation_errors={"conversation_preparation": "failure"},
        evaluation_error_categories={"conversation_preparation": Category.RESOURCE},
    )
    subject = scanner()
    monkeypatch.setattr(subject, "scan_data", lambda *args: child)
    batch = subject.scan_batch_data([CONVERSATION], Path("batch.json"))
    assert ScanFailure(Category.PREPARATION, "conv_1") in batch.failures
    assert ScanFailure(Category.RESOURCE, "conv_1:conversation_preparation") in batch.failures


@pytest.mark.parametrize("field", ["validation_error", "error"])
def test_plugin_keeps_empty_diagnostics_for_present_failure_fields(field):
    result = result_with(**{field: ""})
    payload = QuarryPlugin()._scan_payload(scanner(), result, None)
    assert payload["diagnostics"] == [
        {
            "code": "invalid_conversation" if field == "validation_error" else "scan_error",
            "message": "",
        }
    ]
    assert bool(payload["failures"])


@pytest.mark.parametrize(
    "exception,expected",
    [
        (OSError("runtime failure"), Category.UNKNOWN),
        (ResourceError("runtime failure"), Category.RESOURCE),
        (ResourcePreparationError("runtime failure", {}), Category.RESOURCE),
        (QuarryTimeoutError("runtime failure"), Category.TIMEOUT),
    ],
)
def test_file_execution_failure_is_not_misreported_as_a_read_failure(
    monkeypatch, tmp_path, exception, expected
):
    file_path = tmp_path / "conversation.json"
    file_path.write_text(json.dumps(CONVERSATION))
    subject = scanner()

    def fail(*args):
        raise exception

    monkeypatch.setattr(subject, "scan_data", fail)
    result = subject.scan_file(file_path)
    assert result.failures == (ScanFailure(expected),)


_BATCH_PROFILES = [
    {},
    {"validation_error": ""},
    {"error": "", "error_category": Category.PREPARATION},
    {"evaluation_errors": {"Check": ""}, "evaluation_error_categories": {"Check": Category.MODEL}},
    {"timed_out_rules": ["Check"]},
    {"skipped_rules": ["Check"]},
    {"was_truncated": True},
    {"evaluation_warnings": [""]},
    {
        "error": "",
        "error_category": Category.RESOURCE,
        "evaluation_errors": {"Check": ""},
        "evaluation_error_categories": {"Check": Category.MODEL},
    },
]


@pytest.mark.parametrize("profiles", itertools.product(_BATCH_PROFILES, repeat=2))
def test_batch_pairwise_composition_preserves_categories_and_provenance(monkeypatch, profiles):
    children = [result_with(**profile) for profile in profiles]
    pending = iter(children)
    subject = scanner()
    monkeypatch.setattr(subject, "scan_data", lambda *args: next(pending))
    batch = subject.scan_batch_data([CONVERSATION] * len(children), Path("batch.json"))
    assert batch.is_complete is all(child.is_complete for child in children)
    for index, child in enumerate(children, start=1):
        label = f"conv_{index}"
        for failure in child.failures:
            if failure.rule_name is not None:
                assert (
                    ScanFailure(failure.category, f"{label}:{failure.rule_name}") in batch.failures
                )
            elif failure.category in {
                Category.INVALID_INPUT,
                Category.PREPARATION,
                Category.RESOURCE,
            }:
                assert ScanFailure(failure.category, label) in batch.failures
            else:
                assert failure.category in {entry.category for entry in batch.failures}


def test_failure_projection_preserves_legacy_signal_order_and_repeated_rule_names():
    result = result_with(
        error="",
        error_category=Category.RESOURCE,
        evaluation_errors={"second": "", "first": ""},
        evaluation_error_categories={"second": Category.MODEL, "first": Category.TIMEOUT},
        timed_out_rules=["first", "first"],
        skipped_rules=["second"],
        was_truncated=True,
        evaluation_warnings=[""],
    )
    expected = (
        ScanFailure(Category.RESOURCE),
        ScanFailure(Category.MODEL, "second"),
        ScanFailure(Category.TIMEOUT, "first"),
        ScanFailure(Category.TIMEOUT, "first"),
        ScanFailure(Category.TIMEOUT, "first"),
        ScanFailure(Category.SKIPPED, "second"),
        ScanFailure(Category.TRUNCATED_INPUT),
        ScanFailure(Category.EVALUATION_LIMIT),
    )
    assert result.failures == expected
    assert result.failures == expected
    result.evaluation_error_categories.clear()
    assert result.failures[1:3] == (
        ScanFailure(Category.EVALUATION, "second"),
        ScanFailure(Category.EVALUATION, "first"),
    )


def test_spoofed_class_metadata_cannot_break_failure_serialization(payload_serializers):
    class FakeCategory:
        @property
        def __class__(self):
            return Category

    class FakeDict:
        @property
        def __class__(self):
            return dict

    subject, plugin = payload_serializers
    result = result_with(error="", evaluation_errors={"Check": ""})
    result.error_category = FakeCategory()
    result.evaluation_error_categories = FakeDict()
    assert result.failures == (
        ScanFailure(Category.UNKNOWN),
        ScanFailure(Category.UNKNOWN, "Check"),
    )
    assert_failure_serializers_agree(result, subject, plugin)


def test_spoofed_category_value_in_real_mapping_stays_unknown(payload_serializers):
    class FakeCategory:
        @property
        def __class__(self):
            return Category

    subject, plugin = payload_serializers
    result = result_with(evaluation_errors={"Check": ""})
    result.evaluation_error_categories = {"Check": FakeCategory()}
    assert result.failures == (ScanFailure(Category.UNKNOWN, "Check"),)
    assert_failure_serializers_agree(result, subject, plugin)


@pytest.mark.parametrize("spoof_class", [True, False])
def test_rule_name_rejects_non_strings_without_reading_class_attribute(spoof_class):
    class FakeName:
        @property
        def __class__(self):
            if spoof_class:
                return str
            pytest.fail("rule name validation must use the concrete runtime type")

    with pytest.raises(TypeError, match="failure rule_name must be a string or None"):
        ScanFailure(Category.MODEL, FakeName())


def test_genuine_string_subclass_rule_names_serialize_as_plain_strings(payload_serializers):
    class RuleName(str):
        @property
        def __class__(self):
            pytest.fail("rule name normalization must not inspect a spoofable class attribute")

        def __str__(self):
            pytest.fail("rule name normalization must not execute a string override")

    name = RuleName("Check Ω")
    failure = ScanFailure(Category.TIMEOUT, name)
    assert type(failure.rule_name) is str
    assert failure.rule_name == "Check Ω"
    assert orjson.loads(orjson.dumps(failure.to_dict())) == {
        "category": "timeout",
        "rule_name": "Check Ω",
    }
    result = result_with(timed_out_rules=[name])
    assert result.failures == (failure,)
    assert_failure_serializers_agree(result, *payload_serializers)


@pytest.mark.parametrize(
    "pretend_class",
    [
        QuarryTimeoutError,
        TimeoutError,
        EvaluationTimeoutError,
        ConversationValidationError,
        LLMError,
        ModelError,
        ResourceError,
        ResourcePreparationError,
        EvaluationError,
    ],
)
def test_exception_category_ignores_spoofed_class_attributes(monkeypatch, pretend_class):
    class FakeError(Exception):
        @property
        def __class__(self):
            return pretend_class

    error = FakeError("synthetic failure")
    assert _failure_category(error) is Category.UNKNOWN

    async def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr("quarry.scanner.evaluate_async", fail)
    result = scanner().scan_data(CONVERSATION)
    assert result.failures == (ScanFailure(Category.UNKNOWN, "Check"),)


@pytest.mark.parametrize(
    "base,args,expected",
    [
        (QuarryTimeoutError, ("synthetic timeout",), Category.TIMEOUT),
        (EvaluationTimeoutError, ("Check", 1), Category.TIMEOUT),
        (ConversationValidationError, ("synthetic invalid input",), Category.INVALID_INPUT),
        (LLMError, ("synthetic model failure",), Category.MODEL),
        (ModelError, ("synthetic model failure",), Category.MODEL),
        (ResourceError, ("synthetic resource failure",), Category.RESOURCE),
        (ResourcePreparationError, ("synthetic resource failure", {}), Category.RESOURCE),
        (EvaluationError, ("synthetic evaluation failure",), Category.EVALUATION),
    ],
)
def test_exception_category_preserves_concrete_subclasses_without_reading_class_attribute(
    base, args, expected
):
    class Derived(base):
        @property
        def __class__(self):
            pytest.fail("exception classification must use the concrete runtime type")

    assert _failure_category(Derived(*args)) is expected


def test_documented_string_category_metadata_matches_public_annotations(payload_serializers):
    result = ScanResult(
        file_path=None,
        rule_matches=[],
        scan_time=0.0,
        error="synthetic scan error",
        error_category="resource",
        evaluation_errors={"Check": "synthetic rule error"},
        evaluation_error_categories={"Check": "model"},
    )
    assert result.failures == (ScanFailure(Category.RESOURCE), ScanFailure(Category.MODEL, "Check"))
    assert_failure_serializers_agree(result, *payload_serializers)
