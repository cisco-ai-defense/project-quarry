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

"""Frozen synthetic rules scanned against the legacy interpreter's recorded verdicts.

The frozen corpus lives under ``tests/fixtures`` so public examples can evolve without
silently reducing parity coverage. Drift fixtures use the documented keyword stub probers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quarry.config import QuarryConfig
from quarry.scanner import DSLScanner
from tests.engine.golden import frozen_verdict

_REPO = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO / "tests" / "fixtures"
_FROZEN_CONVERSATIONS = (
    Path(__file__).resolve().parent / "snapshots" / "interpreter_conversations.json"
)
# The recorded corpus: 28 rules representable by the interpreter × 31 curated datasets.
_RECORDED_BASELINE = 28 * 31


def _datasets() -> dict[str, dict]:
    datasets = {
        path.name: json.loads(path.read_text())
        for path in sorted((_FIXTURES / "interpreter_data").glob("*.json"))
    }
    datasets.update(json.loads(_FROZEN_CONVERSATIONS.read_text()))
    return datasets


def test_every_frozen_rule_matches_the_interpreter_golden_on_every_dataset():
    datasets = _datasets()
    compared = 0
    mismatches = []
    for rule_file in sorted((_FIXTURES / "interpreter_rules").glob("*.rule")):
        text = rule_file.read_text()
        # The frozen interpreter snapshot hashes the original rule source, which used
        # `=` for equality. Compare current `==` rules against those recorded verdicts.
        legacy_text = text.replace(" == ", " = ")
        recorded = {
            name: record
            for name, conversation in datasets.items()
            if (record := frozen_verdict(legacy_text, conversation)) is not None
        }
        assert recorded, f"no frozen verdicts for {rule_file.name}; check the legacy spelling"
        scanner = DSLScanner(QuarryConfig(quarry_rule_texts=(text,)))
        for name, (verdict, error) in recorded.items():
            assert error is None, (rule_file.name, name, error)
            result = scanner.scan_data(datasets[name])
            assert result.is_complete, (rule_file.name, name, result.completion_issues)
            compared += 1
            if bool(result.rule_matches) != verdict:
                mismatches.append((rule_file.name, name, result.rule_matches, verdict))
    assert not mismatches
    assert compared == _RECORDED_BASELINE


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _call(name: str, call_id: str = "c1") -> dict:
    return {
        "role": "assistant",
        "content": "ok",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
        ],
    }


def _response(name: str, call_id: str = "c1") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": f"{name} rows"}


@pytest.mark.parametrize(
    ("rule", "messages", "expected"),
    [
        pytest.param(
            "topic_drift",
            [
                _user("please summarize the billing support ticket"),
                _assistant("buying bitcoin ethereum crypto wallet mining rig now"),
            ],
            True,
            id="topic-drift-off-topic-reply",
        ),
        pytest.param(
            "topic_drift",
            [
                _user("please summarize the billing support ticket"),
                _assistant("here is the billing support ticket summary"),
            ],
            False,
            id="topic-drift-on-topic-reply",
        ),
        pytest.param(
            "scope_creep",
            [_user("summarize the ticket"), _call("http_post")],
            True,
            id="scope-creep-external-call",
        ),
        pytest.param(
            "scope_creep",
            [_user("summarize the ticket"), _call("get_ticket")],
            False,
            id="scope-creep-read-only-call",
        ),
        pytest.param(
            "task_substitution",
            [_user("summarize the support ticket"), _call("issue_refund")],
            True,
            id="task-substitution-unrequested-refund",
        ),
        pytest.param(
            "task_substitution",
            [_user("please issue a refund for my order"), _call("issue_refund")],
            False,
            id="task-substitution-requested-refund",
        ),
        pytest.param(
            "task_substitution",
            [_user("summarize the support ticket"), _call("get_ticket")],
            False,
            id="task-substitution-read-only-call",
        ),
        pytest.param(
            "exfil_chain",
            [
                _user("look up the customer"),
                _call("query_db", "c1"),
                _response("query_db", "c1"),
                _call("http_post", "c2"),
            ],
            True,
            id="exfil-chain-read-then-send",
        ),
        pytest.param(
            "exfil_chain",
            [
                _user("look up the customer"),
                _call("http_post", "c2"),
                _call("query_db", "c1"),
                _response("query_db", "c1"),
            ],
            False,
            id="exfil-chain-send-before-read",
        ),
        pytest.param(
            "exfil_chain",
            [_user("look up the customer"), _call("query_db", "c1"), _response("query_db", "c1")],
            False,
            id="exfil-chain-no-send",
        ),
    ],
)
def test_drift_rules_ground_through_the_stub_probers(rule, messages, expected):
    text = (_FIXTURES / "drift_rules" / f"{rule}.rule").read_text()
    scanner = DSLScanner(QuarryConfig(quarry_rule_texts=(text,), allow_stub_probers=True))
    result = scanner.scan_data({"messages": messages})
    assert result.is_complete, result.completion_issues
    assert bool(result.rule_matches) is expected
