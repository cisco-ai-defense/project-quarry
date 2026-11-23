# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Trusted host registrations reach real frontends without loading user-supplied code."""

import json
from dataclasses import replace

import pytest
import yaml
from pydantic import TypeAdapter

from quarry.cli import EXIT_INCOMPLETE, EXIT_MATCH, format_results, main
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.integrations.resources import ResourceManager
from quarry.plugin import server
from quarry.plugin.contracts import ScanResponse
from quarry.plugin.service import QuarryPlugin
from quarry.safety.limits import SafetyLimits
from quarry.scanner import DSLScanner
from quarry.types.exceptions import ConfigurationError
from quarry.types.external_spans import ExternalReadRange, ExternalSpanRange, ExternalSpanReply
from tests.unit.test_external_span_adapter import COLLECT, DATA


class Host:
    def __init__(self, kind):
        self.kind = kind
        self.calls = 0

    async def external_spans(self, config, text, *, timeout=None):
        self.calls += 1
        if self.kind == "provider":
            raise RuntimeError("SYNTHETIC_PRIVATE_HOST_DETAIL")
        if self.kind == "invalid":
            return {"raw": "SYNTHETIC_PRIVATE_HOST_DETAIL"}
        return ExternalSpanReply.for_input(
            text,
            (ExternalSpanRange(text.index("abcdef"), text.index("abcdef") + 1, 0.9),)
            if self.kind == "full"
            else (),
            (ExternalReadRange(0, len(text)),) if self.kind == "full" else (),
            complete=self.kind == "full",
        )


def configuration(tmp_path, renderer):
    mapping = {
        "config_version": 2,
        "models": {
            "reader": {
                "adapter": "external",
                "model": "synthetic-v1",
                "category": "SYNTHETIC",
                "spans": {"backend": "external", "renderer": renderer},
            },
            # Lazy scanner preparation must not try to load an unused adapter.
            "unused": {"adapter": "unregistered", "model": "synthetic-unused"},
        },
    }
    config = QuarryConfig.from_mapping(mapping, tmp_path, require_rules=False)
    return replace(
        config, quarry_rule_texts=(COLLECT,), safety_limits=SafetyLimits(rule_timeout=None)
    ), mapping


def assert_reply(payload, kind, *, plugin):
    complete = payload["completion_state"] == "complete" if plugin else payload["complete"]
    assert complete is (kind == "full")
    if kind != "full":
        assert {row["category"] for row in payload["failures"]} == {
            {
                "partial": "model_partial_read",
                "invalid": "model_invalid_output",
                "provider": "model_provider",
            }[kind]
        }
        coverage = payload["logical_reads"]["rows"][0]["coverage"]["characters"]
        assert coverage["value"] is None and coverage["known_units"] == 0
    wire = json.dumps(payload)
    assert "SYNTHETIC_PRIVATE_HOST_DETAIL" not in wire
    assert "abcdef" not in wire


@pytest.mark.parametrize("renderer", ["join", "json"])
@pytest.mark.parametrize("kind", ["full", "partial", "invalid", "provider"])
def test_scanner_uses_a_snapshot_of_the_public_host_registry(tmp_path, renderer, kind):
    config, _ = configuration(tmp_path, renderer)
    host = Host(kind)
    registry = {"external": host}
    scanner = DSLScanner(config, model_adapters=registry)
    registry.clear()
    assert scanner.loaded_rules
    result = scanner.scan_data(DATA, logical_reads=True)
    payload = json.loads(format_results([result], format_type="json"))[0]
    assert_reply(payload, kind, plugin=False)
    assert host.calls == 1


@pytest.mark.parametrize("renderer", ["join", "json"])
@pytest.mark.parametrize("kind", ["full", "partial", "invalid", "provider"])
async def test_actual_mcp_uses_the_public_service_host_registry(
    tmp_path, monkeypatch, renderer, kind
):
    config, _ = configuration(tmp_path, renderer)
    host = Host(kind)
    registry = {"external": host}
    service = QuarryPlugin(config, model_adapters=registry)
    registry.clear()
    monkeypatch.setattr(server, "_service", service)
    packet = await server.mcp.call_tool(
        "quarry_scan",
        {
            "rule_text": COLLECT,
            "conversation": DATA,
            "explain": False,
            "logical_reads": True,
            "verbosity": "compact",
        },
    )
    payload = packet.structured_content
    TypeAdapter(ScanResponse).validate_python(payload)
    assert json.loads(packet.content[0].text) == payload
    assert_reply(payload, kind, plugin=True)
    assert host.calls == 1


@pytest.mark.parametrize("renderer", ["join", "json"])
@pytest.mark.parametrize("kind", ["full", "partial", "invalid", "provider"])
def test_actual_cli_accepts_trusted_host_startup_registration(tmp_path, capsys, renderer, kind):
    _, mapping = configuration(tmp_path, renderer)
    rule = tmp_path / "synthetic.rule"
    rule.write_text(COLLECT, encoding="utf-8")
    mapping["rules"] = [str(rule)]
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    data_path = tmp_path / "synthetic.json"
    data_path.write_text(json.dumps(DATA), encoding="utf-8")
    host = Host(kind)
    exit_code = main(
        [
            "scan",
            "--config",
            str(config_path),
            "--target",
            str(data_path),
            "--output-format",
            "json",
            "--logical-reads",
        ],
        model_adapters={"external": host},
    )
    assert exit_code == (EXIT_MATCH if kind == "full" else EXIT_INCOMPLETE)
    (payload,) = json.loads(capsys.readouterr().out)
    assert_reply(payload, kind, plugin=False)
    assert host.calls == 1


@pytest.mark.parametrize("entry", [DSLScanner, ResourceManager])
def test_host_registry_and_prebuilt_backend_are_mutually_exclusive(tmp_path, entry):
    config, _ = configuration(tmp_path, "join")
    backend = ModelInferenceBackend(config, models_to_load=[])
    with pytest.raises(ValueError, match="mutually exclusive"):
        entry(config, model_inference_backend=backend, model_adapters={"external": Host("full")})


def test_configuration_cannot_register_or_import_a_host_callback(tmp_path):
    config, _ = configuration(tmp_path, "join")
    with pytest.raises(ConfigurationError, match="unregistered adapter 'external'"):
        DSLScanner(config)


async def test_mcp_tool_input_cannot_register_a_host_callback(tmp_path, monkeypatch):
    config, _ = configuration(tmp_path, "join")
    host = Host("full")
    monkeypatch.setattr(server, "_service", QuarryPlugin(config, model_adapters={"external": host}))
    with pytest.raises(Exception, match="model_adapters|Extra inputs|Unknown"):
        await server.mcp.call_tool(
            "quarry_scan",
            {
                "rule_text": COLLECT,
                "conversation": DATA,
                "model_adapters": {"external": "synthetic.module.callback"},
            },
        )
    assert host.calls == 0
