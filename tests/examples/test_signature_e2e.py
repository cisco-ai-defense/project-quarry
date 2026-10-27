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

"""Regression coverage for authoritative signature E2E verdicts."""

from quarry import ScanResult
from scripts import e2e_signature_test
from scripts.e2e_signature_test import _incomplete_scan_detail


def test_evaluation_error_is_not_accepted_as_a_clean_result() -> None:
    result = ScanResult(
        file_path=None,
        rule_matches=[],
        scan_time=0.01,
        evaluation_errors={"Model rule": "adapter failed"},
    )

    assert result.error is None
    assert _incomplete_scan_detail(result) == (
        "incomplete scan: rule evaluation failed: Model rule"
    )


def test_runner_options_use_dataset_paths_when_names_repeat(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data" / "same.json"
    drift = tmp_path / "drift" / "same.json"
    for path in (data, drift):
        path.parent.mkdir()
        path.write_text('{"messages": []}')

    monkeypatch.setattr(e2e_signature_test, "EXAMPLES_DIR", tmp_path)
    monkeypatch.setattr(
        e2e_signature_test,
        "EXAMPLE_OPTIONS",
        {"data/same.json": {"config_version": 1}, "drift/same.json": {"config_version": 2}},
    )
    assert e2e_signature_test._load_dataset(data)[1] == {"config_version": 1}
    assert e2e_signature_test._load_dataset(drift)[1] == {"config_version": 2}
