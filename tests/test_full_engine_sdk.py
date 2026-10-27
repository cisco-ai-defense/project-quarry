# Copyright 2025 Cisco Systems, Inc. and its affilitates
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

"""
Tests for Python SDK (QuarryGuard).
"""

import asyncio
from pathlib import Path

import pytest
import yaml

from quarry import ConversationValidationError, IncompleteScanError
from quarry import scanner as scanner_module
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.sdk import QuarryGuard

from .conftest import EMAIL_SPAN_RULE, EmailSpanAdapter, email_span_models

_READS_ARGS_RULE = (
    'rule "ReadsArgs"\ncondition:\n'
    '    any call in get_tool_calls("lookup") where call.arguments.q contains "x"\n'
)
_GREETING_RULE = 'rule "Greeting"\ncondition:\n    user_said("hello")\n'
_UNPARSEABLE_ARGUMENTS = {
    "messages": [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{not valid json"},
                }
            ],
        },
    ]
}
_VALID_TOOL_ARGUMENTS = {
    "messages": [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q": "safe"}'},
                }
            ],
        },
    ]
}


def _inject_reads_args_rule_error(monkeypatch) -> None:
    """Exercise SDK reporting of a rule failure with a valid conversation."""

    evaluate = scanner_module.evaluate_async

    async def fail_reads_args(artifact, *args, **kwargs):
        if artifact.rule_name == "ReadsArgs":
            raise RuntimeError("injected rule failure")
        return await evaluate(artifact, *args, **kwargs)

    monkeypatch.setattr(scanner_module, "evaluate_async", fail_reads_args)


def _config(tmp_path: Path, *rules: str, **settings) -> Path:
    """Write ``rules`` and a config that loads them; return the config path."""

    rule_paths = []
    for index, rule_text in enumerate(rules):
        rule_file = tmp_path / f"rule_{index}.rule"
        rule_file.write_text(rule_text)
        rule_paths.append(str(rule_file))
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"rules": rule_paths, **settings}))
    return config_file


def test_guard_requires_an_existing_config():
    with pytest.raises(FileNotFoundError):
        QuarryGuard("/nonexistent/config.yaml")


def test_guard_scan_reports_matches_and_isolates_results(
    tmp_path,
    sample_openai_conversation,
    sample_privesc_conversation,
    sample_data_exfil_rule,
    sample_privesc_rule,
):
    guard = QuarryGuard(_config(tmp_path, sample_data_exfil_rule, sample_privesc_rule))

    matches = guard.scan(sample_openai_conversation)
    assert [(match["rule"], match["metadata"]) for match in matches] == [
        (
            "Sensitive Data Exfiltration",
            {
                "description": "Detects reading sensitive files followed by external transmission",
                "severity": "CRITICAL",
            },
        )
    ]
    assert matches[0]["source_file"] == str(tmp_path / "rule_0.rule")
    assert matches[0]["timing"] >= 0.0

    # A caller editing one result must not change what later scans report.
    matches[0]["metadata"]["severity"] = "tampered"
    assert [m["rule"] for m in guard.scan(sample_privesc_conversation)] == [
        "Privilege Escalation Attempt"
    ]
    assert guard.scan(sample_openai_conversation)[0]["metadata"]["severity"] == "CRITICAL"
    assert guard.scan({"messages": [{"role": "user", "content": "Hello world"}]}) == []


async def test_guard_scan_async_returns_matches(
    tmp_path, sample_openai_conversation, sample_data_exfil_rule
):
    guard = QuarryGuard(_config(tmp_path, sample_data_exfil_rule))

    matches = await guard.scan_async(sample_openai_conversation)

    assert [match["rule"] for match in matches] == ["Sensitive Data Exfiltration"]


async def test_guard_sync_scan_rejects_an_active_event_loop(tmp_path):
    guard = QuarryGuard(_config(tmp_path, _GREETING_RULE))

    with pytest.raises(RuntimeError, match="await QuarryGuard.scan_async"):
        guard.scan({"messages": [{"role": "user", "content": "hello"}]})


def test_guard_match_carries_its_typed_collection_output(tmp_path):
    config_path = tmp_path / "config.yaml"
    (tmp_path / "email.rule").write_text(EMAIL_SPAN_RULE)
    config_path.write_text(
        yaml.dump(
            {
                "config_version": 2,
                "rules": [str(tmp_path / "email.rule")],
                "models": email_span_models(),
            }
        )
    )
    backend = ModelInferenceBackend(
        QuarryConfig.from_yaml(config_path), adapters={"fake": EmailSpanAdapter()}
    )
    guard = QuarryGuard(config_path, model_inference_backend=backend)

    (match,) = guard.scan({"messages": [{"role": "user", "content": "write to alice@example.com"}]})

    assert (match["rule"], match["output"]["type"]) == ("Email", "span_collection")
    assert [item["value"] for item in match["output"]["items"]] == [
        {"message_index": 0, "start": 9, "end": 26, "score": {"value": 0.9}}
    ]


@pytest.mark.parametrize(
    ("conversation", "message"),
    [
        pytest.param("not a dict", "Expected conversation dictionary", id="not-a-mapping"),
        pytest.param({"messages": [], "tools": []}, "list is empty", id="empty-messages"),
        pytest.param({"tools": []}, "messages", id="missing-messages"),
    ],
)
def test_guard_fails_closed_on_invalid_input(tmp_path, conversation, message):
    """Invalid input raises rather than returning matches indistinguishable from a clean pass."""

    guard = QuarryGuard(_config(tmp_path, _GREETING_RULE))

    with pytest.raises(ConversationValidationError, match=message):
        guard.scan(conversation)


@pytest.mark.parametrize(
    "scan",
    [
        pytest.param(lambda guard, conversation: guard.scan(conversation), id="sync"),
        pytest.param(
            lambda guard, conversation: asyncio.run(guard.scan_async(conversation)), id="async"
        ),
    ],
)
def test_guard_rejects_malformed_arguments_before_evaluating_any_rule(tmp_path, scan):
    guard = QuarryGuard(_config(tmp_path, _GREETING_RULE, _READS_ARGS_RULE))

    with pytest.raises(ConversationValidationError, match="malformed JSON"):
        scan(guard, _UNPARSEABLE_ARGUMENTS)


@pytest.mark.parametrize(
    "scan",
    [
        pytest.param(lambda guard, conversation: guard.scan(conversation), id="sync"),
        pytest.param(
            lambda guard, conversation: asyncio.run(guard.scan_async(conversation)), id="async"
        ),
    ],
)
def test_guard_raises_incomplete_scan_with_the_partial_matches_when_a_rule_errors(
    tmp_path, scan, monkeypatch
):
    _inject_reads_args_rule_error(monkeypatch)
    guard = QuarryGuard(_config(tmp_path, _GREETING_RULE, _READS_ARGS_RULE))

    with pytest.raises(IncompleteScanError) as exc_info:
        scan(guard, _VALID_TOOL_ARGUMENTS)

    assert sorted(exc_info.value.scan_result.evaluation_errors) == ["ReadsArgs"]
    assert exc_info.value.details["partial_matches"] == ["Greeting"]


@pytest.mark.parametrize("fail_on_timeout", [False, True])
def test_guard_raises_incomplete_scan_on_timeout_for_either_setting(tmp_path, fail_on_timeout):
    class _SlowAdapter:
        async def classify_messages(self, _config, _messages, _tools, *, timeout=None):
            await asyncio.sleep(5)

    config_path = _config(
        tmp_path,
        'rule "Slow"\ncondition:\n    model("head").classify(messages) > 0.5\n',
        _GREETING_RULE,
        config_version=2,
        rule_timeout=0.1,
        fail_on_timeout=fail_on_timeout,
        models={"head": {"adapter": "slow", "model": "m", "classify": {"class_index": 0}}},
    )
    backend = ModelInferenceBackend(
        QuarryConfig.from_yaml(config_path), adapters={"slow": _SlowAdapter()}
    )
    guard = QuarryGuard(config_path, model_inference_backend=backend)

    with pytest.raises(IncompleteScanError) as exc_info:
        guard.scan({"messages": [{"role": "user", "content": "hello"}]})

    assert exc_info.value.scan_result.timed_out_rules == ["Slow"]
    assert exc_info.value.details["partial_matches"] == ["Greeting"]
