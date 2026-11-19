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

"""Published plugin response schemas and compatibility with existing serializers."""

from __future__ import annotations

import json
from importlib import resources
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter, ValidationError

from quarry import CollectionResult, ProducerRef, ResultItem, ScanResult, Span
from quarry.cli import format_results
from quarry.decision_report import decision_flows
from quarry.decision_schema import DecisionPathPayload, DecisionTreePayload, ScoreLocation
from quarry.plugin import server as plugin_server
from quarry.plugin.contracts import (
    CompileResponse,
    DecisionFlowResponse,
    DecisionTreeResponse,
    ModelUsageSummary,
    ParameterEvidence,
    ReferenceResponse,
    RuleOutput,
    ScanEntry,
    ScanResponse,
    TimingSummary,
    published_output_schemas,
    write_tool_schemas,
)
from quarry.plugin.server import mcp
from quarry.plugin.service import QuarryPlugin

RULE = 'rule "Synthetic boundary"\ncondition:\n    true'
CONVERSATION = {"messages": [{"role": "user", "content": "sample text"}]}
HEX = "a" * 64


def _tree() -> dict:
    return {
        "schema_version": 1,
        "root": "t:root",
        "nodes": {
            "t:root": {
                "id": "t:root",
                "kind": "check",
                "label": "synthetic check",
                "source_nodes": ["n0"],
                "true": "match",
                "false": "no_match",
            }
        },
        "tree_sha256": HEX,
    }


def _path() -> dict:
    return {
        "tree_sha256": HEX,
        "config_sha256": HEX,
        "parameter_digest": HEX,
        "conversation_sha256": HEX,
        "leaf": "match",
        "path": [{"node": "t:root", "outcome": True}],
    }


@pytest.mark.asyncio
async def test_mcp_and_checked_in_schemas_are_the_same_deterministic_document(tmp_path) -> None:
    published = published_output_schemas()
    live = {tool.name: tool.output_schema for tool in await mcp.list_tools()}
    packaged = json.loads(
        resources.files("quarry.plugin").joinpath("tool_schemas.json").read_text(encoding="utf-8")
    )

    assert live == packaged == published
    assert all(schema["additionalProperties"] is True for schema in published.values())
    assert published["quarry_scan"]["properties"]["completion_state"]["enum"] == [
        "complete",
        "invalid",
        "incomplete",
    ]
    flow_schema = published["quarry_decision_flows"]
    assert "unmapped_paths" in flow_schema["required"]
    assert flow_schema["properties"]["unmapped_paths"]["additionalProperties"] == {
        "type": "integer"
    }

    path = write_tool_schemas(tmp_path)
    first = path.read_bytes()
    write_tool_schemas(tmp_path)
    assert path.read_bytes() == first
    assert first.endswith(b"\n")


@pytest.mark.asyncio
async def test_existing_service_payloads_satisfy_all_five_response_contracts() -> None:
    plugin = QuarryPlugin()
    reference = plugin.reference(action="index", limit=1)
    rejected = plugin.reference(action="rejected")
    compilation = plugin.compile(rule_text=RULE)
    tree_response = plugin.decision_tree(rule_text=RULE)
    single = await plugin.scan(rule_text=RULE, conversation=CONVERSATION, explain=False)
    compact = await plugin.scan(
        rule_text=RULE,
        conversations=[CONVERSATION, {"messages": "invalid"}],
    )
    full = await plugin.scan(
        rule_text=RULE,
        conversations=[CONVERSATION],
        verbosity="full",
        explain=False,
    )

    TypeAdapter(ReferenceResponse).validate_python(reference)
    TypeAdapter(ReferenceResponse).validate_python(rejected)
    TypeAdapter(CompileResponse).validate_python(compilation)
    TypeAdapter(DecisionTreeResponse).validate_python(tree_response)
    TypeAdapter(ScanResponse).validate_python(single)
    compact_validated = TypeAdapter(ScanResponse).validate_python(compact)
    TypeAdapter(ScanResponse).validate_python(full)

    (tree,) = tree_response["decision_tree"].values()
    flow = decision_flows(tree, [single["decision_path"]])
    assert flow["unmapped_paths"] == {}
    TypeAdapter(DecisionFlowResponse).validate_python(flow)

    assert compact_validated["verbosity"] == "compact"
    assert "decision_path" not in compact_validated["results"][0]
    assert compact_validated["results"][1]["completion_state"] == "invalid"
    assert "matched" not in compact_validated["results"][1]
    assert full["results"][0]["decision_path"]

    adapter = TypeAdapter(ScanResponse)
    serialized = adapter.dump_json(compact_validated)
    assert adapter.validate_json(serialized) == compact_validated
    assert adapter.dump_json(compact_validated) == serialized


def test_parameter_evidence_contract_matches_both_runtime_shapes() -> None:
    adapter = TypeAdapter(ParameterEvidence)
    number = {"type": "number", "binding": "scan", "value": 0.7}
    text = {"type": "text", "binding": "policy", "redacted": True}

    assert adapter.validate_python(number) == number
    assert adapter.validate_python(text) == text
    assert json.loads(adapter.dump_json(adapter.validate_python(number))) == number
    assert json.loads(adapter.dump_json(adapter.validate_python(text))) == text

    extended = {**text, "future_field": {"kept": True}}
    assert adapter.validate_python(extended)["future_field"] == {"kept": True}


@pytest.mark.parametrize(
    "evidence",
    [
        pytest.param(
            {"type": "text", "binding": "policy"},
            id="text-without-redacted-marker",
        ),
        pytest.param(
            {"type": "text", "binding": "policy", "redacted": False},
            id="text-with-false-redacted-marker",
        ),
        pytest.param(
            {"type": "text", "binding": "policy", "sha256": HEX},
            id="text-with-only-nonexistent-hash",
        ),
        pytest.param(
            {"type": "number", "binding": "scan"},
            id="number-without-value",
        ),
        pytest.param(
            {"type": "number", "binding": "scan", "value": True},
            id="number-with-boolean-value",
        ),
        pytest.param(
            {"type": "binary", "binding": "scan", "value": 1},
            id="unknown-parameter-type",
        ),
        pytest.param(
            {"type": "number", "binding": "request", "value": 1},
            id="unknown-parameter-binding",
        ),
    ],
)
def test_parameter_evidence_contract_rejects_malformed_shapes(evidence) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ParameterEvidence).validate_python(evidence)


def test_parameter_evidence_schema_states_presence_and_absence_semantics() -> None:
    definitions = published_output_schemas()["quarry_scan"]["$defs"]
    number = definitions["NumberParameterEvidence"]
    text = definitions["TextParameterEvidence"]

    assert set(number["required"]) == {"type", "binding", "value"}
    assert set(number["properties"]) == {"type", "binding", "value"}
    assert number["properties"]["type"]["const"] == "number"
    assert number["properties"]["binding"]["enum"] == ["policy", "scan"]

    assert set(text["required"]) == {"type", "binding", "redacted"}
    assert set(text["properties"]) == {"type", "binding", "redacted"}
    assert text["properties"]["type"]["const"] == "text"
    assert text["properties"]["binding"]["enum"] == ["policy", "scan"]
    assert text["properties"]["redacted"]["const"] is True

    assert all("sha256" not in definition["properties"] for definition in (number, text))


@pytest.mark.asyncio
async def test_typed_mcp_boundary_preserves_existing_structured_json(monkeypatch) -> None:
    plugin = QuarryPlugin()
    monkeypatch.setattr(plugin_server, "_service", plugin)

    reference_arguments = {"action": "index", "limit": 1}
    expected_reference = plugin.reference(**reference_arguments)
    actual_reference = (
        await plugin_server.mcp.call_tool("quarry_reference", reference_arguments)
    ).structured_content
    assert actual_reference == expected_reference

    expected_compile = plugin.compile(rule_text=RULE)
    actual_compile = (
        await plugin_server.mcp.call_tool("quarry_compile", {"rule_text": RULE})
    ).structured_content
    assert actual_compile == expected_compile

    expected_tree = plugin.decision_tree(rule_text=RULE)
    actual_tree = (
        await plugin_server.mcp.call_tool("quarry_decision_tree", {"rule_text": RULE})
    ).structured_content
    assert actual_tree == expected_tree

    expected_scan = await plugin.scan(
        rule_text='rule "Malformed"\ncondition:\n    called_tool(',
        conversation=CONVERSATION,
    )
    actual_scan = (
        await plugin_server.mcp.call_tool(
            "quarry_scan",
            {
                "rule_text": 'rule "Malformed"\ncondition:\n    called_tool(',
                "conversation": CONVERSATION,
            },
        )
    ).structured_content
    assert actual_scan == expected_scan

    (tree,) = expected_tree["decision_tree"].values()
    scanned = await plugin.scan(rule_text=RULE, conversation=CONVERSATION, explain=False)
    flow_arguments = {"tree": tree, "paths": [scanned["decision_path"]]}
    expected_flow = decision_flows(**flow_arguments)
    actual_flow = (
        await plugin_server.mcp.call_tool("quarry_decision_flows", flow_arguments)
    ).structured_content
    assert actual_flow == expected_flow


def test_decision_schemas_reject_malformed_and_non_json_boundaries() -> None:
    tree_adapter = TypeAdapter(DecisionTreePayload)
    path_adapter = TypeAdapter(DecisionPathPayload)

    with_extension = _tree()
    with_extension["future_field"] = {"kept": True}
    with_extension["nodes"]["t:root"]["future_node_field"] = 7
    validated = tree_adapter.validate_python(with_extension)
    assert validated["future_field"] == {"kept": True}
    assert validated["nodes"]["t:root"]["future_node_field"] == 7

    missing_edge = _tree()
    del missing_edge["nodes"]["t:root"]["false"]
    with pytest.raises(ValidationError):
        tree_adapter.validate_python(missing_edge)

    future_version = _tree()
    future_version["schema_version"] = 2
    with pytest.raises(ValidationError):
        tree_adapter.validate_python(future_version)

    malformed_digest = _tree()
    malformed_digest["tree_sha256"] = "short"
    with pytest.raises(ValidationError):
        tree_adapter.validate_python(malformed_digest)

    invalid_leaf = _path()
    invalid_leaf["leaf"] = "allow"
    with pytest.raises(ValidationError):
        path_adapter.validate_python(invalid_leaf)

    numeric_outcome = _path()
    numeric_outcome["path"][0]["outcome"] = 1
    with pytest.raises(ValidationError):
        path_adapter.validate_python(numeric_outcome)

    non_finite = _path()
    non_finite["path"][0]["score"] = {
        "value": float("inf"),
        "operator": ">",
        "threshold": 0.5,
        "evaluations": 1,
    }
    with pytest.raises(ValidationError):
        path_adapter.validate_python(non_finite)


def test_unique_step_ids_survive_typed_serialization() -> None:
    entry = {
        "completion_state": "complete",
        "failures": [],
        "timings": {"total_seconds": 0.01, "rules": {}},
        "model_usage": {"calls": 0, "records": [], "truncated": False},
        "models_called": [],
        "diagnostics": [],
        "decision_scores": {"t:score": 0.2, "t:score~1": 0.8},
        "matched": False,
        "matches": [],
    }
    adapter = TypeAdapter(ScanEntry)

    serialized = adapter.dump_json(adapter.validate_python(entry))

    assert list(json.loads(serialized)["decision_scores"]) == ["t:score", "t:score~1"]


def test_timing_and_model_usage_contracts_reject_malformed_or_unbounded_values() -> None:
    timing = TypeAdapter(TimingSummary)
    usage = TypeAdapter(ModelUsageSummary)
    valid_usage = {
        "calls": 1,
        "records": [{"model": "judge", "operation": "score", "calls": 1}],
        "truncated": False,
    }

    assert usage.validate_python(valid_usage) == valid_usage
    assert timing.validate_python({"total_seconds": 0.0, "rules": {"r": 0.1}})
    for malformed in (
        {**valid_usage, "calls": -1},
        {**valid_usage, "calls": True},
        {**valid_usage, "calls": 1_000_001},
        {**valid_usage, "records": valid_usage["records"] * 513},
        {
            **valid_usage,
            "records": [{"model": "x" * 201, "operation": "score", "calls": 1}],
        },
        {
            **valid_usage,
            "records": [{"model": "judge", "operation": "provider_call", "calls": 1}],
        },
    ):
        with pytest.raises(ValidationError):
            usage.validate_python(malformed)
    for malformed in (
        {"total_seconds": -0.1},
        {"total_seconds": float("inf")},
        {"total_seconds": True},
    ):
        with pytest.raises(ValidationError):
            timing.validate_python(malformed)


def test_cli_and_plugin_keep_shared_failure_and_typed_output_shapes() -> None:
    output = CollectionResult(
        output_type="span_collection",
        category="synthetic",
        source=ProducerRef(kind="collector", name="sample", run_id="run-1"),
        items=(ResultItem("item-1", Span(message_index=0, start=0, end=6)),),
    )
    complete = ScanResult(
        file_path=None,
        rule_matches=["Synthetic boundary"],
        scan_time=0.01,
        rule_timings={"Synthetic boundary": 0.005},
        model_usage={
            "calls": 1,
            "records": [
                {"model": "judge", "operation": "score", "calls": 1},
            ],
            "truncated": False,
        },
        outputs={"Synthetic boundary": output},
    )
    scanner = SimpleNamespace(
        loaded_rules={
            "Synthetic boundary": SimpleNamespace(
                name="Synthetic boundary", metadata={"category": "synthetic"}
            )
        }
    )
    plugin_payload = QuarryPlugin()._scan_payload(scanner, complete, None)
    cli_payload = json.loads(format_results([complete], "json"))[0]

    assert plugin_payload["matches"][0]["output"] == cli_payload["matches"][0]["output"]
    assert TypeAdapter(RuleOutput).validate_python(plugin_payload["matches"][0]["output"])
    assert plugin_payload["failures"] == cli_payload["failures"] == []
    assert plugin_payload["timings"] == cli_payload["timings"]
    assert plugin_payload["model_usage"] == cli_payload["model_usage"]
    assert plugin_payload["models_called"] == cli_payload["models_called"] == ["judge"]

    invalid = ScanResult(
        file_path=None,
        rule_matches=[],
        scan_time=0.0,
        validation_error="messages must be a list",
    )
    plugin_invalid = QuarryPlugin()._scan_payload(scanner, invalid, None)
    cli_invalid = json.loads(format_results([invalid], "json"))[0]

    assert "matched" not in plugin_invalid
    assert plugin_invalid["failures"] == cli_invalid["failures"]
    assert plugin_invalid["failures"] == [{"category": "invalid_input"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "condition",
    [
        'model("judge").score(message("user", "owned text"), "criterion") > 0.5',
        'model("judge").score(message("user", messages[0].content), "criterion") > 0.5',
        'any chunk in model("judge").split(message("user", "owned constructed text"), 40) where model("judge").score(chunk, "criterion") > 0.5',
    ],
)
@pytest.mark.parametrize("verbosity", ["compact", "full"])
@pytest.mark.parametrize("explain", [False, True])
async def test_constructed_model_locations_retain_their_existing_wire_shape(
    tmp_path, monkeypatch, condition, verbosity, explain
):
    from dataclasses import replace

    from quarry.config import QuarryConfig
    from quarry.integrations import model_inference
    from quarry.safety.limits import SafetyLimits

    class Adapter:
        async def score(self, config, messages, *, timeout=None):
            return 0.8

        async def tokenize(self, config, messages, tools, *, timeout=None):
            return {config.tokens.count_field: 8}

    monkeypatch.setattr(model_inference, "_default_adapters", lambda: {"owned": Adapter()})
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "llm_rate_limits": {"default": {"requests_per_interval": None}},
            "models": {
                "judge": {
                    "adapter": "owned",
                    "model": "owned/model",
                    "score": {
                        "message_builder": "constitution",
                        "boundary_seed": 7,
                        "tokens": {"negative": "0", "positive": "1"},
                    },
                    "tokens": {"adapter": "owned", "extraction": {"count": "count"}},
                }
            },
        },
        tmp_path,
        require_rules=False,
    )
    config = replace(config, safety_limits=SafetyLimits(rule_timeout=None))
    plugin = QuarryPlugin(config)
    monkeypatch.setattr(plugin_server, "_service", plugin)
    source = 'rule "Read" condition: ' + condition
    result = await plugin.scan(
        rule_text=source, conversation=CONVERSATION, verbosity=verbosity, explain=explain
    )
    assert result["completion_state"] == "complete" and result["matched"]
    assert TypeAdapter(ScanResponse).validate_python(result) == result
    if verbosity == "full":
        scores = [step["score"] for step in result["decision_path"]["path"] if "score" in step]
        assert scores
        for score in scores:
            location = score.get("span") or score["deciding"]["span"]
            assert location["constructed"] is True
            assert "message_index" not in location
        (tree,) = plugin.decision_tree(rule_text=source)["decision_tree"].values()
        assert decision_flows(tree, [result["decision_path"]])["leaves"]["match"] == {"all": 1}
    response = await mcp.call_tool(
        "quarry_scan",
        {
            "rule_text": source,
            "conversation": CONVERSATION,
            "verbosity": verbosity,
            "explain": explain,
        },
    )
    assert response.structured_content["completion_state"] == "complete"
    assert response.structured_content["matched"]
    assert (
        json.loads("".join(block.text for block in response.content if block.type == "text"))
        == response.structured_content
    )


@pytest.mark.parametrize(
    "location",
    [
        {"constructed": False},
        {"constructed": True, "pieces": "invalid"},
        {"constructed": True, "pieces": [{"position": "0", "field": "content"}]},
        {"constructed": True, "pieces": [{"position": 0}]},
    ],
)
def test_score_locations_reject_malformed_constructed_origins(location):
    with pytest.raises(ValidationError):
        TypeAdapter(ScoreLocation).validate_python(location)


def test_score_locations_preserve_constructed_origin_extensions():
    location = {
        "constructed": True,
        "pieces": [{"position": 0, "field": "content", "future_piece": [1, 2]}],
        "future_location": "retained",
    }
    assert TypeAdapter(ScoreLocation).validate_python(location) == location
