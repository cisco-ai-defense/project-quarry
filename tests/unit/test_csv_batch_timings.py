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

"""CSV timing association stays correct without repeating whole timing maps."""

import csv
import json
import pickle
from io import StringIO
from pathlib import Path

import pytest

from quarry import ScanResult
from quarry.cli import format_results
from quarry.config import QuarryConfig
from quarry.safety.limits import SafetyLimits
from quarry.scanner import DSLScanner


def scanner_for(source: str, *, batch_mode: bool = False) -> DSLScanner:
    return DSLScanner(
        QuarryConfig(
            quarry_rule_texts=tuple(source.splitlines()),
            safety_limits=SafetyLimits(rule_timeout=None),
        ),
        batch_mode=batch_mode,
    )


def conversations() -> list[dict]:
    return [
        {"messages": [{"role": "user", "content": "first synthetic input"}]},
        {"messages": [{"role": "user", "content": "second synthetic input"}]},
    ]


@pytest.mark.parametrize(
    ("source", "matched_rule"),
    [
        ('rule "Read" condition: true', "Read"),
        (
            'rule "Read" condition: true\nrule "conv_1:Read" condition: false',
            "Read",
        ),
        (
            'rule "Read" condition: false\nrule "conv_1:Read" condition: true',
            "conv_1:Read",
        ),
    ],
    ids=["ordinary-rule", "batch-label-is-another-rule", "literal-prefixed-rule"],
)
def test_batch_csv_retains_the_original_rules_average(source, matched_rule):
    result = scanner_for(source).scan_batch_data(conversations(), Path("batch.json"))
    assert result.is_complete
    assert result.rule_matches == [f"conv_1:{matched_rule}", f"conv_2:{matched_rule}"]
    rows = list(csv.DictReader(StringIO(format_results([result], "csv"))))
    assert [row["rule"] for row in rows] == result.rule_matches
    assert all(
        json.loads(row["timings"])
        == {
            "total_seconds": result.scan_time,
            "rules": {matched_rule: result.rule_timings[matched_rule]},
        }
        for row in rows
    )
    assert (
        json.loads(format_results([result], "json"))[0]["timings"]["rules"] == result.rule_timings
    )
    assert result.batch_size == 2


def test_scalar_csv_keeps_a_literal_batch_like_name_even_with_batch_mode_enabled():
    scanner = scanner_for(
        'rule "Read" condition: false\nrule "conv_1:Read" condition: true',
        batch_mode=True,
    )
    result = scanner.scan_data(conversations()[0], Path("conversation.json"))
    (row,) = csv.DictReader(StringIO(format_results([result], "csv")))
    assert row["rule"] == "conv_1:Read"
    assert json.loads(row["timings"])["rules"] == {
        "conv_1:Read": result.rule_timings["conv_1:Read"]
    }
    assert getattr(result, "batch_size", None) is None


@pytest.mark.parametrize("timings", [None, {"Read": 0.25, "Other": 0.5}])
def test_unmatched_csv_preserves_its_timing_map_in_the_single_output_row(timings):
    result = ScanResult(
        Path("conversation.json"), [], 1.0, rule_timings=timings, timed_out_rules=["Read"]
    )
    (row,) = csv.DictReader(StringIO(format_results([result], "csv")))
    assert row["rule"] == ""
    assert row["completion_state"] == "incomplete"
    assert json.loads(row["timed_out_rules"]) == ["Read"]
    assert json.loads(row["timings"]) == {
        "total_seconds": 1.0,
        "rules": dict(sorted((timings or {}).items())),
    }


def test_batch_provenance_and_timing_association_survive_pickle():
    result = scanner_for('rule "Read" condition: true').scan_batch_data(
        conversations(), Path("batch.json")
    )
    restored = pickle.loads(pickle.dumps(result))
    rows = list(csv.DictReader(StringIO(format_results([restored], "csv"))))
    assert all(json.loads(row["timings"])["rules"] == result.rule_timings for row in rows)
    assert restored.batch_size == result.batch_size == 2
    assert restored.model_usage == result.model_usage


def test_older_result_state_without_batch_provenance_gets_an_unknown_default():
    original = ScanResult(None, ["Read"], 1.0, rule_timings={"Read": 0.25})
    state = dict(original.__dict__)
    state.pop("batch_size", None)
    restored = ScanResult.__new__(ScanResult)
    restored.__setstate__(state)
    assert restored.batch_size is None
    assert restored.rule_timings == {"Read": 0.25}
    assert restored.rule_matches == ["Read"]


def test_many_batch_matches_keep_bounded_row_local_timing_output():
    names = [f"conv_{index}:Read" for index in range(1, 1_001)]
    result = ScanResult(Path("batch.json"), names, 1.0, rule_timings={"Read": 0.25})
    result.batch_size = 1_000
    output = format_results([result], "csv")
    rows = list(csv.DictReader(StringIO(output)))
    assert [row["rule"] for row in rows] == names
    assert all(json.loads(row["timings"])["rules"] == {"Read": 0.25} for row in rows)
    assert len(output) < 250_000


def test_empty_batch_retains_invalid_status_and_zero_batch_size():
    result = scanner_for('rule "Read" condition: true').scan_batch_data([], Path("empty.json"))
    assert result.completion_state == "invalid"
    assert result.model_usage == {"calls": 0, "records": [], "truncated": False}
    assert result.batch_size == 0


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "2"])
def test_batch_size_rejects_invalid_constructor_and_saved_state_values(invalid):
    with pytest.raises(ValueError, match="batch_size"):
        ScanResult(None, [], 0.0, batch_size=invalid)
    state = dict(ScanResult(None, [], 0.0).__dict__)
    state["batch_size"] = invalid
    with pytest.raises(ValueError, match="batch_size"):
        ScanResult.__new__(ScanResult).__setstate__(state)
