# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Real formatters and MCP calls preserve external read failure categories."""

import json

import pytest
from pydantic import TypeAdapter

from quarry.cli import format_results
from quarry.plugin import server
from quarry.plugin.contracts import ScanResponse
from quarry.plugin.service import QuarryPlugin
from quarry.types.external_spans import ExternalSpanReply
from tests.unit.test_external_span_adapter import COLLECT, DATA, ExternalOnly, policy


@pytest.mark.parametrize("kind", ["partial", "invalid", "provider"])
async def test_cli_and_real_mcp_preserve_typed_failures_without_host_payloads(
    tmp_path, monkeypatch, kind
):
    class Host(ExternalOnly):
        async def external_spans(self, config, text, *, timeout=None):
            if kind == "provider":
                raise RuntimeError("SYNTHETIC_PRIVATE_HOST_DETAIL")
            if kind == "invalid":
                return {"raw": "SYNTHETIC_PRIVATE_HOST_DETAIL"}
            return ExternalSpanReply.for_input(text, (), (), complete=False)

    compiled, adapter = policy(tmp_path, Host())
    native = await compiled.scan_async(DATA, logical_reads=True)
    expected = {
        "partial": "model_partial_read",
        "invalid": "model_invalid_output",
        "provider": "model_provider",
    }[kind]
    cli = json.loads(format_results([native], format_type="json"))[0]
    assert {f["category"] for f in cli["failures"]} == {expected}
    plugin = QuarryPlugin(
        runtime_config=compiled._scanner.config, model_adapters={"external": adapter}
    )
    monkeypatch.setattr(server, "_service", plugin)
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
    assert json.loads(packet.content[0].text) == payload
    TypeAdapter(ScanResponse).validate_python(payload)
    assert payload["completion_state"] == "incomplete"
    assert {f["category"] for f in payload["failures"]} == {expected}
    for response in (cli, payload):
        coverage = response["logical_reads"]["rows"][0]["coverage"]["characters"]
        assert coverage["value"] is None
        assert coverage["known_units"] == 0
        assert coverage["measurement_states"] == {
            "partial" if kind == "partial" else "unavailable": 1
        }
    for wire in (json.dumps(cli), json.dumps(payload)):
        assert "SYNTHETIC_PRIVATE_HOST_DETAIL" not in wire
        assert "abcdef" not in wire


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("restored", [False, True])
def test_empty_rendered_text_still_requires_a_validated_completeness_reply(
    tmp_path, complete, restored
):
    adapter = ExternalOnly(
        lambda text: ExternalSpanReply.for_input(text, (), (), complete=complete)
    )
    compiled, _ = policy(
        tmp_path,
        adapter,
        restored=restored,
    )
    result = compiled.scan({"messages": [{"role": "user", "content": ""}]}, logical_reads=True)
    assert result.is_complete is complete
    assert result.outputs == {} and adapter.calls == 1
    if not complete:
        assert result.evaluation_error_categories["Read"] == "model_partial_read"


def test_empty_conversation_keeps_the_existing_framework_validation_gate(tmp_path):
    compiled, adapter = policy(tmp_path)
    result = compiled.scan({"messages": []})
    assert not result.is_complete and result.validation_error
    assert adapter.calls == 0


def test_empty_message_selection_keeps_native_missing_input_semantics(tmp_path):
    compiled, adapter = policy(
        tmp_path, source=COLLECT.replace("spans(messages,", "spans(messages[:0],")
    )
    result = compiled.scan(DATA)
    assert not result.is_complete and not result.outputs
    assert adapter.calls == 0


def test_complete_external_result_does_not_publish_its_input_digest(tmp_path):
    import hashlib

    compiled, _ = policy(tmp_path)
    result = compiled.scan(DATA, logical_reads=True)
    wire = format_results([result], format_type="json")
    assert result.is_complete
    assert "abcdef" not in wire
    assert hashlib.sha256(b"abcdef").hexdigest() not in wire
