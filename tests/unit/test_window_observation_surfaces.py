# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Public entry points opt in explicitly and isolate conversation snapshots."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from quarry import ReadUsageSession, model_read_usage
from quarry.cli import EXIT_INCOMPLETE, scan_command
from tests.plugin.test_plugin import _plugin
from tests.unit.model_fakes import ModelAdapter, model_entry, v2_scanner
from tests.unit.test_model_window_evidence import EvidenceAdapter, policy
from tests.unit.test_model_window_observations import DATA


def scanner(tmp_path):
    adapter = EvidenceAdapter()
    direct = v2_scanner(
        tmp_path,
        {"R": 'model("classes").classify(messages) > 0.5'},
        {
            "classes": {
                **model_entry(classify=0),
                "windows": {"unit": "characters", "size": 3, "step": 2, "scope": "last_turn"},
            }
        },
        adapter,
    )
    return direct, adapter


@pytest.mark.parametrize("entry", ["scan", "scan_async", "scan_many", "scan_many_async"])
@pytest.mark.parametrize("logical", [False, True])
async def test_policy_opt_in_is_separate_and_each_conversation_has_its_own_rows(
    tmp_path, entry, logical
):
    compiled, adapter = policy(tmp_path, 'model("classes").classify(messages) > 0.5')
    method = getattr(compiled, entry)
    data = [DATA, DATA] if "many" in entry else DATA
    options = {"window_observations": True, "logical_reads": logical}
    result = (
        await method(data, **options)
        if entry.endswith("async")
        else await asyncio.to_thread(method, data, **options)
    )
    results = result if isinstance(result, list) else [result]
    for index, result in enumerate(results):
        assert result.is_complete and result.rule_matches == ["R"]
        assert (result.logical_reads is not None) is logical
        report = result.window_observations
        assert report["enabled"] and report["finalized"] and report["state"] == "collected"
        assert report["scopes"] == [index + 1]
        (row,) = report["rows"]
        assert row["scope"] == index + 1 and row["rule"] == "R"
        assert row["source_characters"] == {"state": "measured", "value": 6}
    assert len(adapter.requests) == 1
    if len(results) == 2:
        results[0].window_observations["rows"][0]["probabilities"][0][0] = 999
        assert results[1].window_observations["rows"][0]["probabilities"][0][0] == 0.3


@pytest.mark.parametrize(
    "entry", ["scan_data", "scan_data_async", "scan_batch_data", "scan_file", "scan_directory"]
)
async def test_scanner_file_and_directory_surfaces(tmp_path, entry):
    direct, adapter = scanner(tmp_path)
    directory = tmp_path / "inputs"
    directory.mkdir()
    path = directory / "synthetic.json"
    path.write_text(json.dumps(DATA), encoding="utf-8")
    args = {
        "scan_data": (DATA,),
        "scan_data_async": (DATA,),
        "scan_batch_data": ([DATA, DATA], path),
        "scan_file": (path,),
        "scan_directory": (directory,),
    }[entry]
    method = getattr(direct, entry)
    result = (
        await method(*args, window_observations=True)
        if entry.endswith("async")
        else await asyncio.to_thread(method, *args, window_observations=True)
    )
    (result,) = result if isinstance(result, list) else [result]
    report = result.window_observations
    assert result.is_complete and report["finalized"]
    assert result.logical_reads is None
    assert report["scopes"] == ([1, 2] if entry == "scan_batch_data" else [1])
    assert len(report["rows"]) == (2 if entry == "scan_batch_data" else 1)
    assert all(row["source_characters"]["value"] == 6 for row in report["rows"])
    assert len(adapter.requests) == 1


async def test_opt_in_borrows_existing_metadata_ownership(tmp_path):
    compiled, _ = policy(tmp_path, 'model("classes").classify(messages) > 0.5')
    with model_read_usage() as session:
        result = await compiled.scan_async(DATA, window_observations=True, logical_reads=True)
        assert not result.window_observations["finalized"]
        assert not result.logical_reads["finalized"]
        assert result.window_observations["rows"][0]["probabilities"][0][0] == 0.3
    assert session.window_observation_report()["finalized"]
    assert not result.window_observations["finalized"]


async def test_disabled_public_option_does_not_create_a_session(tmp_path, monkeypatch):
    compiled, _ = policy(tmp_path, 'model("classes").classify(messages) > 0.5')

    def unexpected(*args, **kwargs):
        pytest.fail("disabled observation created accounting")

    monkeypatch.setattr(ReadUsageSession, "__init__", unexpected)
    result = await compiled.scan_async(DATA)
    assert result.is_complete and result.window_observations is None


@pytest.mark.parametrize("entry", ["scan", "scan_async", "scan_many", "scan_many_async"])
async def test_public_booleans_are_strict(tmp_path, entry):
    compiled, _ = policy(tmp_path, 'model("classes").classify(messages) > 0.5')
    method = getattr(compiled, entry)
    data = [DATA] if "many" in entry else DATA
    with pytest.raises(ValueError, match="window_observations must be a boolean"):
        if entry.endswith("async"):
            await method(data, window_observations="yes")
        else:
            await asyncio.to_thread(method, data, window_observations="yes")


@pytest.mark.parametrize("mode", ["single", "batch", "directory"])
def test_real_cli_flag_preserves_default_shape_and_attaches_requested_report(tmp_path, mode):
    rule = tmp_path / "synthetic.rule"
    rule.write_text('rule "Synthetic" condition: true\n', encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("config_version: 2\nrules:\n  - " + str(rule) + "\n", encoding="utf-8")
    directory = tmp_path / "inputs"
    directory.mkdir()
    data = directory / "synthetic.json"
    data.write_text(json.dumps([DATA, DATA] if mode == "batch" else DATA), encoding="utf-8")
    argv = [
        sys.executable,
        "-m",
        "quarry.cli",
        "scan",
        "--config",
        str(config),
        "--target",
        str(directory if mode == "directory" else data),
        "--output-format",
        "json",
    ]
    if mode == "batch":
        argv.append("--batch")
    for enabled in (False, True):
        process = subprocess.run(
            argv + (["--window-observations"] if enabled else []),
            cwd=Path(__file__).parents[2],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert process.returncode == 1, process.stderr
        (result,) = json.loads(process.stdout)
        assert result["complete"]
        assert ("window_observations" in result) is enabled
        assert "logical_reads" not in result
        if enabled:
            assert result["window_observations"]["finalized"]
            assert result["window_observations"]["rows"] == []


@pytest.mark.parametrize("output_format", ["human", "csv"])
def test_cli_rejects_formats_that_cannot_serialize_report(output_format, capsys):
    assert (
        scan_command(SimpleNamespace(window_observations=True, output_format=output_format))
        == EXIT_INCOMPLETE
    )
    assert "--window-observations requires --output-format json" in capsys.readouterr().err


async def test_real_mcp_strict_input_and_structured_response(tmp_path, monkeypatch):
    from mcp.server.mcpserver.exceptions import ToolError

    from quarry.plugin import server

    plugin = _plugin(tmp_path, monkeypatch, ModelAdapter(), {})
    monkeypatch.setattr(server, "_service", plugin)
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    assert tools["quarry_scan"].input_schema["properties"]["window_observations"] == {
        "default": False,
        "title": "Window Observations",
        "type": "boolean",
    }
    arguments = {"rule_text": 'rule "R" condition: true', "conversation": DATA, "explain": False}
    before = await server.mcp.call_tool("quarry_scan", arguments)
    assert "window_observations" not in before.structured_content
    response = await server.mcp.call_tool("quarry_scan", {**arguments, "window_observations": True})
    payload = response.structured_content
    assert json.loads(response.content[0].text) == payload
    assert payload["window_observations"]["finalized"]
    assert payload["window_observations"]["rows"] == []
    with pytest.raises(ToolError):
        await server.mcp.call_tool("quarry_scan", {**arguments, "window_observations": "yes"})
