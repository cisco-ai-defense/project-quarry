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
Tests for CLI functionality.
"""

import csv
import subprocess
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
import yaml

from quarry import cli
from quarry import scanner as scanner_module
from quarry.integrations import model_inference

from .conftest import EMAIL_SPAN_RULE, EmailSpanAdapter, email_span_models

_GREETING_RULE = 'rule "Greeting"\ncondition:\n    user_said("hello")\n'
_READS_ARGS_RULE = (
    'rule "ReadsArgs"\ncondition:\n'
    '    any call in get_tool_calls("lookup") where call.arguments.q contains "x"\n'
)
_HELLO = {"messages": [{"role": "user", "content": "hello"}]}
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
    """Exercise CLI reporting of a rule failure with a valid conversation."""

    evaluate = scanner_module.evaluate_async

    async def fail_reads_args(artifact, *args, **kwargs):
        if artifact.rule_name == "ReadsArgs":
            raise RuntimeError("injected rule failure")
        return await evaluate(artifact, *args, **kwargs)

    monkeypatch.setattr(scanner_module, "evaluate_async", fail_reads_args)


@pytest.fixture
def quarry(monkeypatch, capsys):
    """Run ``quarry <args>`` through the CLI entry point and capture what it prints."""

    # Wide enough that human output never wraps a temporary path.
    monkeypatch.setenv("COLUMNS", "400")

    def run(*args) -> SimpleNamespace:
        monkeypatch.setattr(sys, "argv", ["quarry", *map(str, args)])
        returncode = cli.main()
        captured = capsys.readouterr()
        return SimpleNamespace(returncode=returncode, stdout=captured.out, stderr=captured.err)

    return run


def _cli_config(tmp_path: Path, *rules: str, **settings) -> Path:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    for index, rule in enumerate(rules):
        (rules_dir / f"rule_{index}.rule").write_text(rule)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.dump({"rules": [str(rules_dir)], **settings}))
    return config


def _target(tmp_path: Path, payload, name: str = "conversation.json") -> Path:
    target = tmp_path / name
    target.write_bytes(orjson.dumps(payload))
    return target


def test_cli_module_prints_help_without_a_command_and_rejects_an_unknown_one():
    def run(*args) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "quarry.cli", *args], capture_output=True, text=True, check=False
        )

    result = run()

    assert result.returncode == 1
    assert "Available commands" in result.stdout
    assert run("invalid_command").returncode == 2


def test_cli_verbose_reports_progress_on_stderr(tmp_path):
    config = _cli_config(tmp_path, _GREETING_RULE)
    target = _target(tmp_path, _HELLO)

    # The progress log is written to the process's own stderr, so this runs the module.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "quarry.cli",
            "scan",
            "-c",
            config,
            "-t",
            target,
            "-v",
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "[STATUS] Compiling rules to DFA artifacts..." in result.stderr
    assert f"[STATUS] Scanning file: {target}" in result.stderr
    assert '[MATCH] Rule matched: name="Greeting"' in result.stderr


@pytest.mark.parametrize(
    ("payload", "flags", "exit_code", "matches", "completion_state", "errors", "validation_error"),
    [
        pytest.param(
            {"messages": [{"role": "user", "content": "hello there"}]},
            [],
            1,
            [{"rule": "Greeting"}],
            "complete",
            [],
            None,
            id="match-exits-one",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": "bye"}]},
            [],
            0,
            [],
            "complete",
            [],
            None,
            id="clean-exits-zero",
        ),
        pytest.param(
            _UNPARSEABLE_ARGUMENTS,
            [],
            2,
            [],
            "invalid",
            [],
            "malformed JSON",
            id="malformed-tool-arguments-are-invalid",
        ),
        pytest.param(
            {"messages": []},
            [],
            2,
            [],
            "invalid",
            [],
            "'messages' list is empty. Nothing to scan.",
            id="invalid-conversation-exits-two",
        ),
        pytest.param(
            [],
            ["--batch"],
            2,
            [],
            "invalid",
            [],
            "Batch must contain at least one conversation",
            id="empty-batch-exits-two",
        ),
    ],
)
def test_cli_scan_exit_code_reports_match_clean_incomplete_or_invalid(
    quarry, tmp_path, payload, flags, exit_code, matches, completion_state, errors, validation_error
):
    config = _cli_config(tmp_path, _GREETING_RULE, _READS_ARGS_RULE)

    result = quarry("scan", "-c", config, "-t", _target(tmp_path, payload), "-o", "json", *flags)

    assert result.returncode == exit_code
    [output] = orjson.loads(result.stdout)
    assert output["matches"] == matches
    assert output["completion_state"] == completion_state
    assert output["complete"] is (completion_state == "complete")
    assert sorted(output["evaluation_errors"] or {}) == errors
    if validation_error is None:
        assert output["validation_error"] is None
    else:
        assert validation_error in output["validation_error"]


def test_cli_rule_error_keeps_partial_matches_incomplete(quarry, tmp_path, monkeypatch):
    _inject_reads_args_rule_error(monkeypatch)
    config = _cli_config(tmp_path, _GREETING_RULE, _READS_ARGS_RULE)

    result = quarry(
        "scan", "-c", config, "-t", _target(tmp_path, _VALID_TOOL_ARGUMENTS), "-o", "json"
    )

    assert result.returncode == 2
    [output] = orjson.loads(result.stdout)
    assert output["completion_state"] == "incomplete"
    assert output["matches"] == [{"rule": "Greeting"}]
    assert sorted(output["evaluation_errors"]) == ["ReadsArgs"]


def test_cli_csv_and_output_file_carry_the_same_result(quarry, tmp_path):
    config = _cli_config(tmp_path, _GREETING_RULE)
    target = _target(tmp_path, _HELLO)

    csv_result = quarry("scan", "-c", config, "-t", target, "-o", "csv")
    assert csv_result.returncode == 1
    [row] = list(csv.DictReader(StringIO(csv_result.stdout)))
    assert (row["file"], row["rule"], row["completion_state"]) == (
        str(target),
        "Greeting",
        "complete",
    )

    output_file = tmp_path / "out" / "results.json"
    written = quarry("scan", "-c", config, "-t", target, "-o", "json", "--output", output_file)
    assert written.returncode == 1
    assert written.stdout == ""
    assert f"Results written to {output_file}" in written.stderr
    assert orjson.loads(output_file.read_bytes())[0]["matches"] == [{"rule": "Greeting"}]


def test_cli_batch_explain_reports_prefixed_evidence(quarry, tmp_path):
    config = _cli_config(tmp_path, _GREETING_RULE)
    target = _target(
        tmp_path, [{"messages": [{"role": "user", "content": "bye"}]}, _HELLO], "batch.json"
    )

    result = quarry("scan", "-c", config, "-t", target, "--batch", "--explain", "-o", "json")

    assert result.returncode == 1
    [output] = orjson.loads(result.stdout)
    assert output["matches"] == [{"rule": "conv_2:Greeting"}]
    assert [match["rule_name"] for match in output["detailed_matches"]] == ["conv_2:Greeting"]


def test_cli_human_output_explains_the_match_and_charts_rule_timing(quarry, tmp_path):
    config = _cli_config(
        tmp_path,
        'rule "Leak"\nmetadata:\n    description = "Token echoed by a tool"\ncondition:\n'
        '  any m in user_messages where ("TOKEN" in m.content as tok)\n'
        "  and any r in tool_responses where r.content contains tok\n",
    )
    target = _target(
        tmp_path,
        {
            "messages": [
                {"role": "user", "content": "my TOKEN here"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "x", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "echoing TOKEN"},
            ]
        },
    )

    result = quarry("scan", "-c", config, "-t", target, "--explain", "--show-timing")

    assert result.returncode == 1
    assert "Scan Summary" in result.stdout
    assert "Rule Performance" in result.stdout
    assert "Avg (ms)" in result.stdout
    assert "reason: value capture `tok`" in result.stdout
    assert f"[MATCH] {target} (1 matches)" in result.stdout
    assert "[Leak] (rule_0.rule)" in result.stdout
    assert "description: Token echoed by a tool" in result.stdout
    assert "Why it matched" in result.stdout
    assert 'captured: "TOKEN"' in result.stdout
    assert "witness (load-bearing turns)" in result.stdout
    assert '[0] user_messages[0]: "my TOKEN here"' in result.stdout
    assert '[3] tool_responses[0]: "echoing TOKEN"' in result.stdout


def test_cli_human_output_marks_invalid_tool_arguments_without_partial_matches(quarry, tmp_path):
    config = _cli_config(tmp_path, _GREETING_RULE, _READS_ARGS_RULE)
    target = _target(tmp_path, _UNPARSEABLE_ARGUMENTS)

    result = quarry("scan", "-c", config, "-t", target)

    assert result.returncode == 2
    assert f"[INVALID] {target}" in result.stdout
    assert "Invalid/incomplete files:" in result.stdout
    assert "malformed JSON" in result.stdout
    assert "Partial matches" not in result.stdout


def test_cli_human_output_marks_an_incomplete_partial_match_non_authoritative(
    quarry, tmp_path, monkeypatch
):
    _inject_reads_args_rule_error(monkeypatch)
    config = _cli_config(tmp_path, _GREETING_RULE, _READS_ARGS_RULE)
    target = _target(tmp_path, _VALID_TOOL_ARGUMENTS)

    result = quarry("scan", "-c", config, "-t", target)

    assert result.returncode == 2
    assert f"[INCOMPLETE] {target}" in result.stdout
    assert "Partial matches (not an authoritative decision): Greeting" in result.stdout


@pytest.mark.parametrize(
    ("output_format", "timed_out_rules"),
    [
        pytest.param("json", lambda stdout: orjson.loads(stdout)[0]["timed_out_rules"], id="json"),
        pytest.param(
            "csv",
            lambda stdout: orjson.loads(next(csv.DictReader(StringIO(stdout)))["timed_out_rules"]),
            id="csv",
        ),
    ],
)
def test_cli_timeout_is_incomplete_and_serializes_the_timed_out_rules(
    quarry, tmp_path, output_format, timed_out_rules
):
    config = _cli_config(tmp_path, _GREETING_RULE, rule_timeout=1e-9)

    result = quarry("scan", "-c", config, "-t", _target(tmp_path, _HELLO), "-o", output_format)

    assert result.returncode == 2
    assert timed_out_rules(result.stdout) == ["Greeting"]


def test_cli_one_invalid_file_makes_a_directory_scan_exit_two(quarry, tmp_path):
    config = _cli_config(tmp_path, _GREETING_RULE, _READS_ARGS_RULE)
    target = tmp_path / "conversations"
    target.mkdir()
    _target(target, _HELLO, "complete.json")
    _target(target, _UNPARSEABLE_ARGUMENTS, "invalid.json")

    result = quarry("scan", "-c", config, "-t", target, "-o", "json")

    assert result.returncode == 2
    assert [item["completion_state"] for item in orjson.loads(result.stdout)] == [
        "complete",
        "invalid",
    ]


def test_cli_no_limits_lifts_configured_input_limits(quarry, tmp_path):
    config = _cli_config(tmp_path, _GREETING_RULE, max_messages=1)
    target = _target(
        tmp_path,
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        },
    )

    limited = quarry("scan", "-c", config, "-t", target, "-o", "json")
    assert limited.returncode == 2
    [limited_output] = orjson.loads(limited.stdout)
    assert limited_output["completion_state"] == "invalid"
    assert "max_messages" in limited_output["validation_error"]

    unlimited = quarry("scan", "-c", config, "-t", target, "-o", "json", "--no-limits")
    assert unlimited.returncode == 1
    assert orjson.loads(unlimited.stdout)[0]["matches"] == [{"rule": "Greeting"}]


_BROKEN_RULE = 'rule "Broken"\ncondition:\n    user_said(\n'


@pytest.mark.parametrize(
    ("rule", "arguments", "stderr"),
    [
        pytest.param(
            _GREETING_RULE,
            lambda root, config, target: ["-c", root / "missing.yaml", "-t", target],
            "[ERROR] Failed to load config",
            id="missing-config",
        ),
        pytest.param(
            _BROKEN_RULE,
            lambda root, config, target: ["-c", config, "-t", target],
            "[ERROR] Failed to initialize scanner",
            id="rule-syntax-error",
        ),
        pytest.param(
            _GREETING_RULE,
            lambda root, config, target: ["-c", config, "-t", root / "missing.json"],
            "[ERROR] Target not found",
            id="missing-target",
        ),
        pytest.param(
            _GREETING_RULE,
            lambda root, config, target: ["-c", config, "-t", root / "empty"],
            "[ERROR] No JSON files found",
            id="empty-directory",
        ),
        pytest.param(
            _GREETING_RULE,
            lambda root, config, target: ["-c", config, "-t", target, "--output", root / "out.txt"],
            "--output is only supported with json or csv output formats",
            id="human-output-file",
        ),
        pytest.param(
            _GREETING_RULE,
            lambda root, config, target: [
                "-c",
                config,
                "-t",
                target,
                "-o",
                "json",
                "--output",
                root,
            ],
            "[ERROR] Output path is a directory",
            id="output-is-a-directory",
        ),
    ],
)
def test_cli_startup_and_usage_errors_exit_two(quarry, tmp_path, rule, arguments, stderr):
    config = _cli_config(tmp_path, rule)
    target = _target(tmp_path, _HELLO)
    (tmp_path / "empty").mkdir()

    result = quarry("scan", *arguments(tmp_path, config, target))

    assert result.returncode == 2
    assert stderr in result.stderr
    assert "Scan Summary" not in result.stdout
    assert not (tmp_path / "out.txt").exists()


def test_cli_recursive_flag_controls_nested_file_discovery(quarry, tmp_path):
    config = _cli_config(tmp_path, _GREETING_RULE)
    target = tmp_path / "conversations"
    (target / "nested").mkdir(parents=True)
    _target(target / "nested", _HELLO)

    flat = quarry("scan", "-c", config, "-t", target, "-o", "json")
    assert flat.returncode == 2
    assert "[ERROR] No JSON files found" in flat.stderr

    recursive = quarry("scan", "-c", config, "-t", target, "-o", "json", "--recursive")
    assert recursive.returncode == 1
    assert orjson.loads(recursive.stdout)[0]["matches"] == [{"rule": "Greeting"}]


def test_cli_validate_only_reports_rules_without_reading_the_target(quarry, tmp_path):
    config = _cli_config(tmp_path, _GREETING_RULE)

    result = quarry("scan", "-c", config, "-t", "/nonexistent/file", "--validate-only")

    assert result.returncode == 0
    assert "● Greeting" in result.stdout
    assert "Validated 1 rules" in result.stdout
    assert "All rules passed validation" in result.stdout


@pytest.mark.parametrize(
    ("rule", "exit_code", "warning", "summary"),
    [
        (
            'rule "Broad"\ncondition: has_tool("delete_file")\n',
            0,
            "[MEDIUM] Broad: Rule 'Broad' has no specific event or message-content anchor.",
            "No blocking warnings",
        ),
        (
            'rule "Impossible"\ncondition: len(messages) > 5 and len(messages) < 3\n',
            1,
            "[HIGH] Impossible: Rule 'Impossible' has a condition that is always false",
            "Total warnings: 1",
        ),
        (
            'rule "Unbounded"\ncondition: called_tool("b") after called_tool("a")\n',
            1,
            "[MEDIUM] Unbounded: Unbounded 'after' operator detected",
            "Total warnings: 1",
        ),
    ],
)
def test_cli_validate_only_reports_quality_and_preserves_existing_exit_behavior(
    quarry, tmp_path, rule, exit_code, warning, summary
):
    config = _cli_config(tmp_path, rule)

    result = quarry("scan", "-c", config, "-t", "/nonexistent/file", "--validate-only")

    assert result.returncode == exit_code
    assert warning in result.stdout
    assert summary in result.stdout


def test_cli_validate_only_quality_analysis_is_best_effort(quarry, tmp_path, monkeypatch):
    config = _cli_config(tmp_path, _GREETING_RULE)

    def unavailable_quality_analysis(_rule, _artifact):
        raise RuntimeError("quality analysis unavailable")

    monkeypatch.setattr(cli, "quality_warnings", unavailable_quality_analysis)
    result = quarry("scan", "-c", config, "-t", "/nonexistent/file", "--validate-only")

    assert result.returncode == 0
    assert "All rules passed validation" in result.stdout

    monkeypatch.setattr(cli, "quality_warnings", lambda _rule, _artifact: [object()])
    malformed = quarry("scan", "-c", config, "-t", "/nonexistent/file", "--validate-only")

    assert malformed.returncode == 0
    assert "All rules passed validation" in malformed.stdout


def test_cli_validate_only_packaged_tool_availability_examples_do_not_fail(quarry, tmp_path):
    examples = Path(__file__).parents[1] / "examples" / "rules"
    config = _cli_config(
        tmp_path,
        (examples / "has_tool.rule").read_text(),
        (examples / "has_tools.rule").read_text(),
    )

    result = quarry("scan", "-c", config, "-t", "/nonexistent/file", "--validate-only")

    assert result.returncode == 0
    assert "Total warnings: 2" in result.stdout
    assert "No blocking warnings" in result.stdout
    assert "[MEDIUM] Offered Tool:" in result.stdout
    assert "[MEDIUM] Multiple Offered Tools:" in result.stdout


@pytest.mark.parametrize(
    ("payload", "flags", "present", "absent"),
    [
        pytest.param(
            {"messages": [{"role": "user", "content": "write to alice@example.com"}]},
            ["-o", "json"],
            ['"type": "span_collection"', '"category": "EMAIL"', '"start": 9', '"end": 26'],
            [],
            id="json",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": "write to alice@example.com"}]},
            ["-o", "csv"],
            ['""type"":""span_collection""', '""start"":9,""end"":26'],
            [],
            id="csv",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": "write to alice@example.com"}]},
            [],
            ["output: span_collection [EMAIL] (1 item)"],
            ["characters 9:26"],
            id="human-summarizes-the-collection",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": "write to alice@example.com"}]},
            ["--explain"],
            ["output: span_collection [EMAIL] (1 item)", "message 0, characters 9:26, score 0.9"],
            [],
            id="human-explain-lists-each-span",
        ),
        pytest.param(
            [
                {"messages": [{"role": "user", "content": "nothing here"}]},
                {"messages": [{"role": "user", "content": "write to alice@example.com"}]},
            ],
            ["--batch", "--explain"],
            ["conv_2", "Email", "message 0, characters 9:26, score 0.9"],
            ["conv_1"],
            id="human-batch-groups-by-conversation",
        ),
    ],
)
def test_cli_reports_a_typed_collection_output_in_every_format(
    quarry, tmp_path, monkeypatch, payload, flags, present, absent
):
    # The CLI builds its own inference backend, so the provider adapter registry is the
    # closest boundary at which a test can stand in for the model provider.
    monkeypatch.setattr(model_inference, "_default_adapters", lambda: {"fake": EmailSpanAdapter()})
    (tmp_path / "email.rule").write_text(EMAIL_SPAN_RULE)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.dump(
            {
                "config_version": 2,
                "rules": [str(tmp_path / "email.rule")],
                "models": email_span_models(),
            }
        )
    )

    result = quarry("scan", "-c", config, "-t", _target(tmp_path, payload), *flags)

    assert result.returncode == 1
    assert [fragment in result.stdout for fragment in present] == [True] * len(present)
    assert [fragment in result.stdout for fragment in absent] == [False] * len(absent)
