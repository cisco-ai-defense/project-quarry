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

"""Host-side local file inputs for the values-only Quarry MCP tools."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from quarry.plugin.service import MAX_RULE_TEXT_CHARS, MAX_SCAN_BATCH
from quarry.safety.limits import SafetyLimits

REPO_ROOT = Path(__file__).parents[2]
HOOK_SCRIPT = REPO_ROOT / "hooks" / "hydrate_quarry_inputs.py"
HOOK_CONFIG = REPO_ROOT / "hooks" / "hooks.json"
SCAN_TOOL = "mcp__plugin_quarry_quarry__quarry_scan"
COMPILE_TOOL = "mcp__quarry__quarry_compile"
RULE = 'rule "Check"\ncondition:\n    user_said("hello")\n'
INLINE_RULE = 'rule "Check"\ncondition:\n    true\n'
LONGEST_RULE = 'rule "Long"\ncondition:\n    true\n'.ljust(MAX_RULE_TEXT_CHARS, " ")


def _conversation(content: str) -> dict:
    return {"messages": [{"role": "user", "content": content}]}


def _hook(event: dict) -> dict | None:
    """Run the hook as the host does: one JSON event on stdin, an optional decision on stdout."""

    completed = subprocess.run(
        [sys.executable, str(HOOK_SCRIPT)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    return json.loads(completed.stdout) if completed.stdout else None


def _pre_tool_use(root: Path, arguments: dict, tool_name: str = SCAN_TOOL) -> dict | None:
    return _hook(
        {
            "session_id": "test-session",
            "cwd": str(root),
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_use_id": "test-call",
            "tool_input": arguments,
        }
    )


def _files(files: dict[str, str | bytes]) -> Callable[[Path], None]:
    """Workspace setup that writes each relative path with its content."""

    def write(root: Path) -> None:
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8")

    return write


def _allow(updated_input: dict) -> dict:
    """The decision that lets the tool call run with ``updated_input``."""

    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": updated_input,
        }
    }


def _directory(name: str) -> Callable[[Path], None]:
    return lambda root: (root / name).mkdir()


def _sparse_file(name: str, size: int) -> Callable[[Path], None]:
    """A file of ``size`` NUL bytes that costs no disk space to create."""

    def write(root: Path) -> None:
        with (root / name).open("wb") as handle:
            handle.truncate(size)

    return write


def _symlink_outside(name: str) -> Callable[[Path], None]:
    def write(root: Path) -> None:
        outside = root.parent / f"{root.name}-outside.json"
        outside.write_text('{"messages": []}', encoding="utf-8")
        try:
            (root / name).symlink_to(outside)
        except OSError:
            pytest.skip("the test environment cannot create symlinks")

    return write


@pytest.mark.parametrize(
    ("setup", "tool_name", "arguments", "output"),
    [
        pytest.param(
            _files({"rules/check.rule": RULE}),
            COMPILE_TOOL,
            {"rule_text": "quarry+file://rules/check.rule", "config_version": 2},
            _allow({"rule_text": RULE, "config_version": 2}),
            id="compile-rule-file",
        ),
        pytest.param(
            _files({"rules/check.rule": RULE, "samples/one.json": json.dumps(_conversation("hi"))}),
            SCAN_TOOL,
            {
                "rule_text": "quarry+file://rules/check.rule",
                "conversation": {"file": "samples/one.json"},
                "parameters": {"threshold": 0.8},
                "explain": False,
            },
            _allow(
                {
                    "rule_text": RULE,
                    "conversation": _conversation("hi"),
                    "parameters": {"threshold": 0.8},
                    "explain": False,
                }
            ),
            id="rule-and-conversation-files-leave-other-inputs-alone",
        ),
        pytest.param(
            _files({"rules/check.rule": RULE}),
            SCAN_TOOL,
            {"rule_text": "quarry+file://rules/check.rule", "conversation": _conversation("x")},
            _allow({"rule_text": RULE, "conversation": _conversation("x")}),
            id="rule-file-keeps-an-inline-conversation",
        ),
        pytest.param(
            _files(
                {
                    "one.json": json.dumps(_conversation("first")),
                    "two.json": json.dumps(_conversation("second")),
                    "three.json": json.dumps(_conversation("third")),
                }
            ),
            SCAN_TOOL,
            {
                "rule_text": INLINE_RULE,
                "conversations": [
                    {"file": "one.json"},
                    {"file": "two.json"},
                    {"file": "three.json"},
                ],
            },
            _allow(
                {
                    "rule_text": INLINE_RULE,
                    "conversations": [
                        _conversation("first"),
                        _conversation("second"),
                        _conversation("third"),
                    ],
                }
            ),
            id="conversation-files-keep-their-order",
        ),
        pytest.param(
            _files({"batch.json": json.dumps([_conversation("first"), _conversation("second")])}),
            SCAN_TOOL,
            {"rule_text": INLINE_RULE, "conversations": [{"batch_file": "batch.json"}]},
            _allow(
                {
                    "rule_text": INLINE_RULE,
                    "conversations": [_conversation("first"), _conversation("second")],
                }
            ),
            id="batch-file-expands-into-conversations",
        ),
        pytest.param(
            _files({"batch.json": json.dumps([{}] * MAX_SCAN_BATCH)}),
            SCAN_TOOL,
            {"rule_text": INLINE_RULE, "conversations": [{"batch_file": "batch.json"}]},
            _allow({"rule_text": INLINE_RULE, "conversations": [{}] * MAX_SCAN_BATCH}),
            id="batch-at-the-plugin-limit",
        ),
        pytest.param(
            _files({"long.rule": LONGEST_RULE}),
            COMPILE_TOOL,
            {"rule_text": "quarry+file://long.rule"},
            _allow({"rule_text": LONGEST_RULE}),
            id="rule-at-the-plugin-limit",
        ),
        pytest.param(
            _files({"large.json": json.dumps(_conversation("x" * 1_000_000))}),
            SCAN_TOOL,
            {"rule_text": INLINE_RULE, "conversation": {"file": "large.json"}},
            _allow({"rule_text": INLINE_RULE, "conversation": _conversation("x" * 1_000_000)}),
            id="large-conversation-is-not-truncated",
        ),
        pytest.param(
            _files({}),
            SCAN_TOOL,
            {"rule_text": INLINE_RULE, "conversation": _conversation("hello")},
            None,
            id="value-only-call-passes-through",
        ),
        pytest.param(
            _files({"rules/check.rule": RULE}),
            "mcp__unrelated__quarry_compile",
            {"rule_text": "quarry+file://rules/check.rule"},
            None,
            id="similarly-named-server-is-not-rewritten",
        ),
    ],
)
def test_hook_replaces_file_markers_with_their_contents(
    tmp_path, setup, tool_name, arguments, output
):
    setup(tmp_path)

    assert _pre_tool_use(tmp_path, arguments, tool_name) == output


@pytest.mark.parametrize(
    ("setup", "tool_name", "arguments", "reason", "secrets"),
    [
        pytest.param(
            _files({}),
            SCAN_TOOL,
            {"conversation": {"file": "../outside.json"}},
            "must not contain '..'",
            (),
            id="parent-path",
        ),
        pytest.param(
            _files({}),
            SCAN_TOOL,
            {"conversation": {"file": "/tmp/outside.json"}},
            "must be relative",
            (),
            id="absolute-path",
        ),
        pytest.param(
            _files({}),
            SCAN_TOOL,
            {"conversation": {"file": "~/.config/private.json"}},
            "must not start with '~'",
            (),
            id="home-path",
        ),
        pytest.param(
            _files({}),
            SCAN_TOOL,
            {"conversation": {"file": "samples\\one.json"}},
            "must use '/'",
            (),
            id="backslash-path",
        ),
        pytest.param(
            _files({}),
            SCAN_TOOL,
            {"conversation": {"file": "missing.json"}},
            "must resolve inside",
            (),
            id="missing-file",
        ),
        pytest.param(
            _symlink_outside("outside.json"),
            SCAN_TOOL,
            {"conversation": {"file": "outside.json"}},
            "must resolve inside the workspace",
            (),
            id="symlink-escaping-the-workspace",
        ),
        pytest.param(
            _directory("conversation.json"),
            SCAN_TOOL,
            {"conversation": {"file": "conversation.json"}},
            "must identify a regular file",
            (),
            id="directory-marker",
        ),
        pytest.param(
            _files({"rule.pem": "not a rule file"}),
            SCAN_TOOL,
            {"rule_text": "quarry+file://rule.pem"},
            "rule file path must end with '.rule'",
            (),
            id="rule-file-of-another-format",
        ),
        pytest.param(
            _files({"conversation.txt": "{}"}),
            SCAN_TOOL,
            {"conversation": {"file": "conversation.txt"}},
            "conversation file path must end with '.json'",
            (),
            id="conversation-file-of-another-format",
        ),
        pytest.param(
            _files({"conversation.json": "not JSON"}),
            SCAN_TOOL,
            {"conversation": {"file": "conversation.json"}},
            "contains invalid JSON",
            ("not JSON",),
            id="invalid-json",
        ),
        pytest.param(
            _files({"conversation.json": "[]"}),
            SCAN_TOOL,
            {"conversation": {"file": "conversation.json"}},
            "must contain one JSON object",
            (),
            id="conversation-that-is-not-an-object",
        ),
        pytest.param(
            _files({"rule.rule": b"\xff\xfe"}),
            COMPILE_TOOL,
            {"rule_text": "quarry+file://rule.rule"},
            "rule file must be valid UTF-8",
            (),
            id="non-utf8-rule",
        ),
        pytest.param(
            _files({}),
            SCAN_TOOL,
            {"conversation": {"file": "conversation.json", "messages": []}},
            "must contain only 'file'",
            (),
            id="marker-with-other-fields",
        ),
        pytest.param(
            _files({"batch.json": json.dumps([{}] * (MAX_SCAN_BATCH + 1))}),
            SCAN_TOOL,
            {"conversations": [{"batch_file": "batch.json"}]},
            f"between 1 and {MAX_SCAN_BATCH}",
            (),
            id="batch-above-the-plugin-limit",
        ),
        pytest.param(
            _files({"batch.json": "[{}]"}),
            SCAN_TOOL,
            {"conversations": [{"batch_file": "batch.json"}, _conversation("inline")]},
            "must be the only item",
            (),
            id="batch-marker-mixed-with-conversations",
        ),
        pytest.param(
            _files({"long.rule": LONGEST_RULE + " "}),
            COMPILE_TOOL,
            {"rule_text": "quarry+file://long.rule"},
            f"must not exceed {MAX_RULE_TEXT_CHARS} characters",
            (),
            id="rule-above-the-plugin-limit",
        ),
        # A conversation at the scanner's size limit passes the size check and fails only as
        # JSON; one byte more is refused before it is read.
        pytest.param(
            _sparse_file("conversation.json", SafetyLimits().max_total_size),
            SCAN_TOOL,
            {"conversation": {"file": "conversation.json"}},
            "contains invalid JSON",
            (),
            id="conversation-at-the-scanner-limit-is-read",
        ),
        pytest.param(
            _sparse_file("conversation.json", SafetyLimits().max_total_size + 1),
            SCAN_TOOL,
            {"conversation": {"file": "conversation.json"}},
            f"exceeds its {SafetyLimits().max_total_size}-byte limit",
            (),
            id="conversation-above-the-scanner-limit",
        ),
    ],
)
def test_hook_denies_an_unsafe_or_invalid_file_without_leaking_the_workspace(
    tmp_path, setup, tool_name, arguments, reason, secrets
):
    setup(tmp_path)

    result = _pre_tool_use(tmp_path, {"rule_text": INLINE_RULE, **arguments}, tool_name)

    output = result["hookSpecificOutput"]
    assert (output["hookEventName"], output["permissionDecision"]) == ("PreToolUse", "deny")
    assert reason in output["permissionDecisionReason"]
    leaks = [
        text for text in (str(tmp_path), *secrets) if text in output["permissionDecisionReason"]
    ]
    assert leaks == []


def test_session_start_supplies_local_markers_without_changing_the_shared_skill(tmp_path):
    result = _hook(
        {
            "session_id": "test-session",
            "cwd": str(tmp_path),
            "hook_event_name": "SessionStart",
            "source": "startup",
        }
    )

    output = result["hookSpecificOutput"]
    assert output["hookEventName"] == "SessionStart"
    assert "quarry+file://" in output["additionalContext"]
    assert '{"file":' in output["additionalContext"]
    skill = (REPO_ROOT / "skills" / "quarry-policy" / "SKILL.md").read_text(encoding="utf-8")
    assert "quarry+file://" not in skill
    assert "PreToolUse" not in skill


def test_hook_rejects_a_file_replaced_after_validation(tmp_path):
    # The swap has to land between validation and the read, which a single stdin event
    # cannot schedule, so this one check drives the hook's two steps directly.
    spec = importlib.util.spec_from_file_location("hydrate_quarry_inputs", HOOK_SCRIPT)
    hook = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = hook
    spec.loader.exec_module(hook)
    path = tmp_path / "conversation.json"
    path.write_text('{"messages": []}', encoding="utf-8")
    resolved = hook._resolve_file(
        tmp_path.resolve(), hook._FileRequest("conversation", "conversation.json")
    )
    path.unlink()
    path.write_text(json.dumps(_conversation("changed")), encoding="utf-8")

    with pytest.raises(hook.HydrationError, match="changed after validation"):
        hook._read_utf8(resolved)


def test_both_plugin_hosts_discover_the_same_hook_and_script():
    config = json.loads(HOOK_CONFIG.read_text(encoding="utf-8"))
    group = config["hooks"]["PreToolUse"][0]
    command = group["hooks"][0]["command"]
    session_command = config["hooks"]["SessionStart"][0]["hooks"][0]["command"]

    assert [
        bool(re.fullmatch(group["matcher"], name))
        for name in (
            "mcp__quarry__quarry_scan",
            "mcp__quarry__quarry_compile",
            "mcp__plugin_quarry_quarry__quarry_scan",
            "mcp__unrelated__quarry_scan",
        )
    ] == [True, True, True, False]
    assert "${CLAUDE_PLUGIN_ROOT}" in command
    assert "hooks/hydrate_quarry_inputs.py" in command
    assert session_command == command
    assert HOOK_SCRIPT.is_file()


async def test_hydrated_files_pass_the_unchanged_mcp_contract(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("QUARRY_PLUGIN_CONFIG", "")
    _files(
        {
            "check.rule": RULE,
            "matching.json": json.dumps(_conversation("hello")),
            "clean.json": json.dumps(_conversation("goodbye")),
        }
    )(tmp_path)
    result = _pre_tool_use(
        tmp_path,
        {
            "rule_text": "quarry+file://check.rule",
            "conversations": [{"file": "matching.json"}, {"file": "clean.json"}],
            "explain": False,
        },
    )

    from quarry.plugin.server import mcp

    scan = (
        await mcp.call_tool("quarry_scan", result["hookSpecificOutput"]["updatedInput"])
    ).structured_content

    assert scan["completion_state"] == "complete"
    assert scan["count"] == 2
    assert [item["matched"] for item in scan["results"]] == [True, False]
