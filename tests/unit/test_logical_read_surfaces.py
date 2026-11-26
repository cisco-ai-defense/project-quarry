# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Real file, CLI and MCP surfaces share the versioned logical-read contract."""

import asyncio
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator, ValidationError

from quarry import DSLScanner, ReadUsageLimits, logical_read_report_schema, model_read_usage
from quarry.cli import EXIT_INCOMPLETE, scan_command
from quarry.integrations.read_usage import capture_read_usage, read_usage_scope
from quarry.types.messages import Messages
from quarry.types.rules import ScanResult
from tests.plugin.test_plugin import SCORE_MODEL, _plugin
from tests.unit.model_fakes import ModelAdapter
from tests.unit.test_public_logical_reads import characters, scanner, summary
from tests.unit.test_read_usage import finish, measure
from tests.unit.test_scanner_read_usage_lifetimes import CHECK, DATA


def validate(report):
    schema = logical_read_report_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)


def test_file_batch_preserves_all_conversation_namespaces_and_excludes_outer_reads(tmp_path):
    direct, adapter = scanner(tmp_path)
    with model_read_usage() as session:
        with read_usage_scope("unrelated"):
            finish(session, model="unrelated-model", chars=measure())
        result = direct.scan_batch_data([DATA] * 8, tmp_path / "synthetic.json", logical_reads=True)
        report = result.logical_reads
        assert report["scopes"] == list(range(1, 9))
        assert characters(result) == 8 * 31
        assert summary(result)["observations"] == summary(result)["evaluations"] == 8
        assert summary(result)["models"] == ["judge"]
        assert {row["rule"] for row in report["rows"]} == {"Outer"}
        assert not report["finalized"]
        validate(report)
    assert result.is_complete and adapter.score_calls == 1


def test_combined_batch_retains_known_lower_bounds_after_scope_admission_refusal(tmp_path):
    direct, _ = scanner(tmp_path)
    with model_read_usage(ReadUsageLimits(max_scopes=2)):
        result = direct.scan_batch_data([DATA] * 3, tmp_path / "synthetic.json", logical_reads=True)
    assert result.is_complete
    assert result.logical_reads["scopes"] == [1, 2]
    assert characters(result) is None
    assert summary(result)["coverage"]["characters"]["known_units"] == 62
    assert result.logical_reads["omitted_observations"] > 0
    validate(result.logical_reads)


@pytest.mark.parametrize("batch", [False, True])
def test_scan_file_and_directory_attach_final_reports(tmp_path, batch):
    direct, adapter = scanner(tmp_path)
    direct.batch_mode = batch
    directory = tmp_path / "inputs"
    directory.mkdir()
    path = directory / "synthetic.json"
    path.write_text(json.dumps([DATA, DATA] if batch else DATA))
    first = direct.scan_file(path, logical_reads=True)
    (second,) = direct.scan_directory(directory, logical_reads=True)
    for result in [first, second]:
        assert result.is_complete and result.logical_reads["finalized"]
        assert characters(result) == (62 if batch else 31)
        validate(result.logical_reads)
    assert adapter.score_calls == 1


@pytest.mark.parametrize("mode", ["single", "batch", "directory"])
def test_real_cli_flag_serializes_only_requested_reports(tmp_path, mode):
    root = Path(__file__).parents[2]
    rule = tmp_path / "synthetic.rule"
    rule.write_text('rule "Synthetic" condition: true\n')
    config = tmp_path / "config.yaml"
    config.write_text("config_version: 2\nrules:\n  - " + str(rule) + "\n")
    directory = tmp_path / "inputs"
    directory.mkdir()
    data = directory / "synthetic.json"
    data.write_text(json.dumps([DATA, DATA] if mode == "batch" else DATA))
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
    for enabled in [False, True]:
        process = subprocess.run(
            argv + (["--logical-reads"] if enabled else []),
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert process.returncode == 1, process.stderr
        (result,) = json.loads(process.stdout)
        assert result["complete"]
        assert ("logical_reads" in result) is enabled
        if enabled:
            assert result["logical_reads"]["finalized"]
            assert (
                result["logical_reads"]["aggregates"]["totals"]["coverage"]["characters"]["value"]
                == 0
            )
            validate(result["logical_reads"])


@pytest.mark.parametrize("output_format", ["human", "csv"])
def test_cli_flag_requires_json_without_silently_dropping_the_report(output_format, capsys):
    assert (
        scan_command(SimpleNamespace(logical_reads=True, output_format=output_format))
        == EXIT_INCOMPLETE
    )
    assert "--logical-reads requires --output-format json" in capsys.readouterr().err


@pytest.mark.parametrize("batch", [False, True])
async def test_plugin_reports_preserve_wire_work_and_boundary_exclusion(
    tmp_path, monkeypatch, batch
):
    adapter = ModelAdapter()
    plugin = _plugin(
        tmp_path, monkeypatch, adapter, {"judge": {**SCORE_MODEL, "model": "synthetic-score"}}
    )
    options = {"conversations": [DATA, DATA]} if batch else {"conversation": DATA}
    source = f'rule "Outer" condition: {CHECK}'
    before = await plugin.scan(rule_text=source, explain=False, **options)
    response = await plugin.scan(rule_text=source, explain=False, logical_reads=True, **options)
    assert adapter.score_calls == 1 and adapter.tokenizations == 0
    old = before["results"] if batch else [before]
    payloads = response["results"] if batch else [response]
    for previous, payload in zip(old, payloads, strict=True):
        assert "logical_reads" not in previous
        assert payload["completion_state"] == previous["completion_state"] == "complete"
        assert payload["matched"] == previous["matched"]
        report = payload["logical_reads"]
        assert report["finalized"]
        assert report["aggregates"]["totals"]["coverage"]["characters"]["value"] == 31
        assert report["aggregates"]["totals"]["coverage"]["text_tokens"]["value"] is None
        validate(report)


async def test_real_mcp_schema_and_call_tool_return_equal_text_and_structured_reports(
    tmp_path, monkeypatch
):
    from mcp.server.mcpserver.exceptions import ToolError

    from quarry.plugin import server

    adapter = ModelAdapter()
    plugin = _plugin(
        tmp_path, monkeypatch, adapter, {"judge": {**SCORE_MODEL, "model": "synthetic-score"}}
    )
    monkeypatch.setattr(server, "_service", plugin)
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    assert tools["quarry_scan"].input_schema["properties"]["logical_reads"] == {
        "default": False,
        "title": "Logical Reads",
        "type": "boolean",
    }
    arguments = {
        "rule_text": f'rule "Outer" condition: {CHECK}',
        "conversation": DATA,
        "logical_reads": True,
        "explain": False,
    }
    response = await server.mcp.call_tool("quarry_scan", arguments)
    payload = response.structured_content
    assert json.loads(response.content[0].text) == payload
    assert payload["logical_reads"]["finalized"]
    validate(payload["logical_reads"])
    assert adapter.score_calls == 1
    with pytest.raises(ToolError):
        await server.mcp.call_tool("quarry_scan", {**arguments, "logical_reads": "yes"})


@pytest.mark.parametrize("mode", ["default", "pending", "manual", "missing_scope", "limited"])
def test_schema_covers_full_reports_and_unknown_measurements(mode):
    with model_read_usage(
        ReadUsageLimits(max_aggregate_work=1) if mode == "limited" else None
    ) as session:
        if mode == "pending":
            session.begin("reader", "score")
        elif mode in {"manual", "limited"}:
            finish(session, chars=measure(), tokens=measure("text_tokens", end=3))
        report = session.scope_report(None) if mode == "missing_scope" else session.report()
        validate(report)
    validate(session.report())


def test_schema_is_owned_and_rejects_raw_unknown_report_fields():
    schema = logical_read_report_schema()
    schema["properties"].clear()
    with model_read_usage() as session:
        report = session.report()
    validate(report)
    with pytest.raises(ValidationError, match="Additional properties"):
        Draft202012Validator(logical_read_report_schema()).validate(
            {**report, "raw_prompt": "synthetic"}
        )


@pytest.mark.parametrize(
    "entry,kind",
    [
        ("batch", "unscoped"),
        ("file", "unscoped"),
        ("batch", "after-native"),
        ("batch", "replayed"),
        ("batch", "explicit-scope"),
    ],
)
def test_custom_reads_without_registered_conversation_identity_never_become_zero(
    tmp_path, entry, kind
):
    original, adapter = scanner(tmp_path)
    replay = None

    class Custom(DSLScanner):
        def scan_data(self, data, file_path=None, enable_explain=False):
            result = (
                super().scan_data(data, file_path, enable_explain)
                if kind == "after-native"
                else ScanResult(file_path, [], 0.0)
            )
            if kind == "replayed":
                replay.replay()
            elif kind == "explicit-scope":
                with read_usage_scope("custom", scope=session.new_scope()):
                    asyncio.run(self.read(data))
            else:
                asyncio.run(self.read(data))
            return result

        async def read(self, data):
            await self.resource_manager.model_inference_backend.model_score_messages(
                "judge", "synthetic policy", Messages.from_openai(data["messages"])
            )

    custom = Custom(
        original.config, model_inference_backend=original.resource_manager.model_inference_backend
    )
    with model_read_usage() as session:
        if kind == "replayed":

            async def prime():
                with capture_read_usage() as captured:
                    await custom.read(DATA)
                return captured.result()

            replay = asyncio.run(prime())
            assert replay is not None
        path = tmp_path / "synthetic.json"
        if entry == "file":
            path.write_text(json.dumps(DATA))
            result = custom.scan_file(path, logical_reads=True)
        else:
            result = custom.scan_batch_data([DATA], path, logical_reads=True)
        assert session.report()["aggregates"]["totals"]["coverage"]["characters"]["value"] > 0
    assert result.is_complete and adapter.score_calls == 1
    assert result.logical_reads["state"] == "scope_unavailable"
    assert result.logical_reads["rows"] == []
    assert characters(result) is None
    assert not summary(result)["coverage"]["characters"]["complete"]
    validate(result.logical_reads)


async def test_native_preparation_reads_use_their_conversation_before_rule_attribution(tmp_path):
    original, adapter = scanner(tmp_path)

    class Preparing(DSLScanner):
        async def _scan_data_with_values_async(self, data, **kwargs):
            await self.resource_manager.model_inference_backend.model_score_messages(
                "judge", "synthetic policy", Messages.from_openai(data["messages"])
            )
            return await super()._scan_data_with_values_async(data, **kwargs)

    config = replace(
        original.config, quarry_rules_paths=(), quarry_rule_texts=('rule "Outer" condition: false',)
    )
    native = Preparing(
        config, model_inference_backend=original.resource_manager.model_inference_backend
    )
    result = await native.scan_data_async(DATA, logical_reads=True)
    assert result.is_complete and result.rule_matches == [] and adapter.score_calls == 1
    assert result.logical_reads["scope"] == 1
    assert characters(result) == 31
    assert {row["scope"] for row in result.logical_reads["rows"]} == {1}
    assert {row["rule"] for row in result.logical_reads["rows"]} == {None}
    validate(result.logical_reads)
