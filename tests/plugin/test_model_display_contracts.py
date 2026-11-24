# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""The current typed transport preserves the public display-metadata producer."""

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from quarry.config import QuarryConfig
from quarry.plugin import service
from quarry.plugin.contracts import ModelAliasSummary, ReferenceResponse, published_output_schemas
from quarry.plugin.service import QuarryPlugin


@pytest.mark.parametrize("label", ["Policy Judge", "界" * 100])
def test_reference_display_metadata_has_a_named_bounded_schema_and_keeps_extensions(label):
    payload = {
        "operations": ["score"],
        "display_name": label,
        "description": "Purpose",
        "future_alias_field": {"kept": True},
    }
    adapter = TypeAdapter(ModelAliasSummary)
    assert adapter.validate_json(adapter.dump_json(adapter.validate_python(payload))) == payload
    schema = published_output_schemas()["quarry_reference"]["$defs"]["ModelAliasSummary"]
    assert schema["properties"]["display_name"] == {
        "minLength": 1,
        "maxLength": 100,
        "title": "Display Name",
        "type": "string",
    }
    assert "display_name" not in schema["required"]
    assert adapter.validate_python({"operations": []}) == {"operations": []}


@pytest.mark.parametrize("label", ["", "x" * 101, 7, True, None, b"bytes"])
def test_display_metadata_rejects_invalid_known_values(label):
    with pytest.raises(ValidationError, match="display_name"):
        TypeAdapter(ModelAliasSummary).validate_python({"operations": [], "display_name": label})


def _server(plugin, monkeypatch):
    monkeypatch.setattr(service, "load_runtime_config", lambda *_args, **_kwargs: None)
    from quarry.plugin import server

    monkeypatch.setattr(server, "_service", plugin)
    return server


async def test_real_mcp_reference_preserves_the_optional_display_contract(monkeypatch):
    plugin = QuarryPlugin()
    response = plugin.reference(action="index", limit=1)
    response["model_aliases"] = {"judge": {"operations": ["score"], "display_name": "Policy Judge"}}
    assert TypeAdapter(ReferenceResponse).validate_python(response) == response
    monkeypatch.setattr(plugin, "reference", lambda **_kwargs: response)
    server = _server(plugin, monkeypatch)
    result = await server.mcp.call_tool("quarry_reference", {"action": "index", "limit": 1})
    assert result.structured_content == response
    assert json.loads("".join(x.text for x in result.content if x.type == "text")) == response


async def test_real_display_producer_reaches_the_typed_mcp_transport(tmp_path, monkeypatch):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "judge": {
                    "adapter": "synthetic",
                    "model": "synthetic-score",
                    "display_name": "Policy Judge",
                }
            },
        },
        tmp_path,
        require_rules=False,
    )
    plugin = QuarryPlugin(config)
    expected = plugin.reference(action="index", limit=1)
    assert expected["model_aliases"]["judge"]["display_name"] == "Policy Judge"
    server = _server(plugin, monkeypatch)
    result = await server.mcp.call_tool("quarry_reference", {"action": "index", "limit": 1})
    assert result.structured_content == expected
    assert json.loads("".join(x.text for x in result.content if x.type == "text")) == expected
