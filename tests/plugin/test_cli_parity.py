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

"""Shared verdict, completion, and diagnostic contracts for the CLI and plugin."""

import asyncio
from types import SimpleNamespace

import orjson
import pytest

from quarry import cli
from quarry import scanner as scanner_module
from quarry.config import QuarryConfig
from quarry.plugin.service import QuarryPlugin
from quarry.safety.limits import SafetyLimits

RULE = 'rule "Greeting"\ncondition:\n    user_said("hello")\n'
TEST_LIMITS = SafetyLimits(rule_timeout=None)


def _cli_diagnostics(result: dict) -> list[dict]:
    """Normalize the CLI's stable diagnostic fields to the plugin vocabulary."""

    diagnostics = []
    if result["validation_error"] is not None:
        diagnostics.append({"code": "invalid_conversation", "message": result["validation_error"]})
    diagnostics.extend(
        {"code": "evaluation_error", "rule_name": name, "message": message}
        for name, message in (result["evaluation_errors"] or {}).items()
    )
    return diagnostics


@pytest.mark.parametrize(
    ("conversation", "evaluation_failure", "exit_code", "completion_state", "verdict"),
    [
        pytest.param(
            {"messages": [{"role": "user", "content": "hello"}]},
            False,
            1,
            "complete",
            True,
            id="match",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": "goodbye"}]},
            False,
            0,
            "complete",
            False,
            id="clean",
        ),
        pytest.param({"messages": []}, False, 2, "invalid", None, id="invalid"),
        pytest.param(
            {"messages": [{"role": "user", "content": "hello"}]},
            True,
            2,
            "incomplete",
            None,
            id="evaluation-failure",
        ),
    ],
)
async def test_cli_and_plugin_agree_on_scan_outcomes(
    tmp_path,
    monkeypatch,
    capsys,
    conversation,
    evaluation_failure,
    exit_code,
    completion_state,
    verdict,
):
    """Presentation may differ, but both front ends must report the same outcome."""

    configured_rule_timeouts = []
    scanner_init = scanner_module.DSLScanner.__init__

    def init_without_rule_deadline(self, config, *args, **kwargs):
        limits = kwargs.get("limits")
        configured_rule_timeouts.append(
            (limits if limits is not None else config.get_safety_limits()).rule_timeout
        )
        scanner_init(self, config, *args, **kwargs)

    monkeypatch.setattr(scanner_module.DSLScanner, "__init__", init_without_rule_deadline)

    if evaluation_failure:

        async def fail_evaluation(*_args, **_kwargs):
            raise RuntimeError("synthetic evaluation failure")

        monkeypatch.setattr(scanner_module, "evaluate_async", fail_evaluation)

    plugin_result = await QuarryPlugin(QuarryConfig(safety_limits=TEST_LIMITS)).scan(
        rule_text=RULE,
        conversation=conversation,
        explain=False,
    )

    rule_path = tmp_path / "greeting.rule"
    rule_path.write_text(RULE)
    config_path = tmp_path / "config.yaml"
    cli_config = QuarryConfig(
        quarry_rules_paths=(rule_path,),
        safety_limits=TEST_LIMITS,
    )
    monkeypatch.setattr(
        cli.QuarryConfig,
        "from_yaml",
        classmethod(lambda _cls, _path: cli_config),
    )
    target_path = tmp_path / "conversation.json"
    target_path.write_bytes(orjson.dumps(conversation))
    cli_exit = await asyncio.to_thread(
        cli.scan_command,
        SimpleNamespace(
            config=config_path,
            target=target_path,
            no_limits=False,
            recursive=False,
            verbose=False,
            batch=False,
            validate_only=False,
            output_format="json",
            explain=False,
            show_timing=False,
            output=None,
        ),
    )
    cli_result = orjson.loads(capsys.readouterr().out)[0]
    cli_verdict = bool(cli_result["matches"]) if cli_result["complete"] else None

    assert cli_exit == exit_code
    assert cli_result["completion_state"] == plugin_result["completion_state"] == completion_state
    assert cli_verdict == plugin_result.get("matched") == verdict
    assert cli_result["failures"] == plugin_result["failures"]
    assert _cli_diagnostics(cli_result) == plugin_result["diagnostics"]
    assert configured_rule_timeouts == [None, None]
