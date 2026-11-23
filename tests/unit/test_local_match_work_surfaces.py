# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Opt-in report delivery preserves actual file, CLI, compiled batch and MCP semantics."""

import asyncio
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from quarry import compile_policy
from quarry.cli import EXIT_INCOMPLETE, EXIT_MATCH
from quarry.plugin.contracts import LocalMatchWorkReport
from quarry.plugin.service import QuarryPlugin
from quarry.safety.local_work_reporting import aggregate_local_match_work
from quarry.scanner import DSLScanner
from tests.unit.test_local_match_work_scanner import DATA, RULE, config
from tests.unit.test_local_match_work_scanner import deterministic_cpu as deterministic_cpu

pytestmark = pytest.mark.usefixtures("deterministic_cpu")


@pytest.mark.parametrize(
    "entry", ["scan_data", "scan_data_async", "scan_file", "scan_directory", "scan_batch_data"]
)
async def test_scanner_surfaces_deliver_detached_bounded_reports(tmp_path, entry):
    scanner = DSLScanner(config())
    directory = tmp_path / "input"
    directory.mkdir()
    path = directory / "synthetic.json"
    path.write_text(json.dumps(DATA), encoding="utf-8")
    args = {
        "scan_data": (DATA,),
        "scan_data_async": (DATA,),
        "scan_file": (path,),
        "scan_directory": (directory,),
        "scan_batch_data": ([DATA, DATA], path),
    }[entry]
    method = getattr(scanner, entry)
    result = (
        await method(*args) if entry.endswith("async") else await asyncio.to_thread(method, *args)
    )
    if isinstance(result, list):
        (result,) = result
    assert result.is_complete
    report = result.local_match_work
    assert report["complete"] and len(report["groups"]) == 1
    assert report["conversations"] == (2 if entry == "scan_batch_data" else 1)
    assert report["scope"] == ("batch" if entry == "scan_batch_data" else "conversation")
    TypeAdapter(LocalMatchWorkReport).validate_python(report)


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
@pytest.mark.parametrize("entry", ["scan_many", "scan_many_async"])
async def test_compiled_ordered_batches_keep_separate_per_conversation_allowances(mode, entry):
    policy = compile_policy(RULE, runtime_config=config(mode=mode, texts=()))
    method = getattr(policy, entry)
    results = (
        await method([DATA, DATA], explain=True)
        if entry.endswith("async")
        else await asyncio.to_thread(method, [DATA, DATA], explain=True)
    )
    assert len(results) == 2
    assert all(result.is_complete is (mode == "shadow") for result in results)
    reports = [result.local_match_work for result in results]
    assert reports[0] is not reports[1]
    assert all(report["complete"] and report["conversations"] == 1 for report in reports)


@pytest.mark.parametrize("mode", ["single", "batch", "directory"])
@pytest.mark.parametrize("enforce", [False, True])
def test_actual_cli_configuration_reports_work_without_changing_disabled_shape(
    tmp_path, mode, enforce
):
    rules = tmp_path / "synthetic.quarry"
    rules.write_text(RULE, encoding="utf-8")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    path = inputs / "synthetic.json"
    path.write_text(json.dumps([DATA, DATA] if mode == "batch" else DATA), encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    document = {"config_version": 2, "rules": [str(rules)]}
    for enabled in (False, True):
        if enabled:
            document["local_match_work"] = {
                "groups": [
                    {
                        "label": "selected",
                        "rules": ["Selected"],
                        "allowance_cpu_ns": 1,
                        "mode": "enforce" if enforce else "shadow",
                    }
                ]
            }
        cfg.write_text(json.dumps(document), encoding="utf-8")
        argv = [
            sys.executable,
            "-c",
            # Run the actual CLI module in a fresh process with a deterministic
            # CPU clock, just as the in-process scanner/compiled/MCP fixtures do.
            # A1ns allowance cannot assume every platform records a positive tick
            # for a tiny real matcher. No production entry point is replaced.
            "import itertools, runpy; import quarry.safety.local_match_work as work; "
            "work.thread_time_ns = itertools.count(step=10).__next__; "
            "runpy.run_module('quarry.cli', run_name='__main__')",
            "scan",
            "--config",
            str(cfg),
            "--target",
            str(inputs if mode == "directory" else path),
            "--output-format",
            "json",
        ]
        if mode == "batch":
            argv.append("--batch")
        process = subprocess.run(
            argv, cwd=Path(__file__).parents[2], capture_output=True, text=True, timeout=30
        )
        assert process.returncode == (EXIT_INCOMPLETE if enabled and enforce else EXIT_MATCH), (
            process.stderr
        )
        (result,) = json.loads(process.stdout)
        assert result["complete"] is not (enabled and enforce)
        assert ("local_match_work" in result) is enabled
        if enabled:
            report = result["local_match_work"]
            assert report["complete"] and report["conversations"] == (2 if mode == "batch" else 1)
            TypeAdapter(LocalMatchWorkReport).validate_python(report)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("mode", ["shadow", "enforce"])
async def test_actual_mcp_preserves_structured_reports_and_completion(monkeypatch, batch, mode):
    from quarry.plugin import server

    plugin = QuarryPlugin(runtime_config=config(mode=mode, texts=()))
    monkeypatch.setattr(server, "_service", plugin)
    arguments = {"rule_text": RULE, "explain": False}
    arguments["conversations" if batch else "conversation"] = [DATA, DATA] if batch else DATA
    response = await server.mcp.call_tool("quarry_scan", arguments)
    payload = response.structured_content
    assert json.loads(response.content[0].text) == payload
    assert payload["completion_state"] == ("complete" if mode == "shadow" else "incomplete")
    report = payload["local_match_work"]
    assert report["complete"] and report["conversations"] == (2 if batch else 1)
    TypeAdapter(LocalMatchWorkReport).validate_python(report)
    if batch:
        assert len(payload["results"]) == 2
        assert all(item["local_match_work"]["conversations"] == 1 for item in payload["results"])


@pytest.mark.parametrize(
    "change", ["negative", "bool", "oversize_groups", "oversize_label", "overflow"]
)
def test_transport_schema_rejects_non_native_or_unbounded_measurements(change):
    report = DSLScanner(config()).scan_data(DATA).local_match_work
    if change == "oversize_groups":
        report["groups"] *= 65
    elif change == "oversize_label":
        report["groups"][0]["label"] = "x" * 65
    else:
        report["groups"][0]["known_cpu_ns"] = {"negative": -1, "bool": True, "overflow": 1 << 63}[
            change
        ]
    with pytest.raises(ValidationError):
        TypeAdapter(LocalMatchWorkReport).validate_python(report)


def test_aggregate_counts_each_exhausted_conversation_without_pooling_allowances():
    first = DSLScanner(config()).scan_data(DATA).local_match_work
    saved = copy.deepcopy(first)
    result = aggregate_local_match_work(first for _ in range(10_000))
    assert first == saved
    assert len(result["groups"]) == 1 and result["conversations"] == 10_000
    row = result["groups"][0]
    assert row["allowance_cpu_ns"] == 1
    assert row["known_cpu_ns"] == first["groups"][0]["known_cpu_ns"] * 10_000
    assert row["exhausted_conversations"] == 10_000
    assert row["complete"] and result["complete"]


def test_saturated_or_unmetered_aggregate_never_promotes_a_known_value():
    first = DSLScanner(config()).scan_data(DATA).local_match_work
    first["groups"][0]["known_cpu_ns"] = (1 << 63) - 1
    first["groups"][0]["value_cpu_ns"] = (1 << 63) - 1
    result = aggregate_local_match_work((first, first))
    row = result["groups"][0]
    assert row["known_cpu_ns"] == (1 << 63) - 1
    assert row["value_cpu_ns"] is None and row["overshoot_cpu_ns"] is None
    assert row["counter_saturated"] and not row["complete"] and not result["complete"]


def test_completed_shared_group_rule_stays_complete_when_next_rule_is_denied(deterministic_cpu):
    texts = ('rule "First" condition: true', RULE.replace('"Selected"', '"Second"'))
    scanner = DSLScanner(
        config(mode="enforce", allowance=20, texts=texts, rules=("First", "Second"))
    )
    result = scanner.scan_data(DATA)
    assert result.rule_matches == ["First"]
    assert set(result.evaluation_errors) == {"Second"}
    assert result.evaluation_error_categories == {"Second": "evaluation_limit"}
    assert not result.is_complete
    row = result.local_match_work["groups"][0]
    assert row["value_cpu_ns"] == 20 and row["overshoot_cpu_ns"] == 0
    assert row["denied_segments"] == 1
