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

"""Output contracts across input rejection and model-reference metadata features."""

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from quarry.decision_schema import DecisionPathPayload
from quarry.plugin import server
from quarry.plugin.contracts import ModelAliasSummary, ReferenceResponse, published_output_schemas
from quarry.plugin.service import QuarryPlugin

HEX = "a" * 64
RULE = 'rule "Read" condition: user_said("hello")'


def _rejected_path():
    return {
        "tree_sha256": HEX,
        "config_sha256": HEX,
        "parameter_digest": HEX,
        "conversation_sha256": None,
        "leaf": None,
        "path": [],
        "future_path_field": {"kept": True},
    }


def test_rejected_path_retains_null_digest_and_extensions_through_json():
    adapter = TypeAdapter(DecisionPathPayload)
    payload = _rejected_path()
    assert adapter.validate_python(payload) == payload
    assert adapter.validate_json(adapter.dump_json(payload)) == payload
    schema = published_output_schemas()["quarry_scan"]["$defs"]["DecisionPathPayload"]
    assert "conversation_sha256" in schema["required"]
    assert schema["properties"]["conversation_sha256"]["anyOf"] == [
        {"pattern": "^[0-9a-f]{64}$", "type": "string"},
        {"type": "null"},
    ]


@pytest.mark.parametrize("field", ["tree_sha256", "config_sha256", "parameter_digest"])
def test_null_conversation_digest_does_not_relax_other_digest_fields(field):
    with pytest.raises(ValidationError, match=field):
        TypeAdapter(DecisionPathPayload).validate_python({**_rejected_path(), field: None})


@pytest.mark.parametrize("digest", ["", "short", "A" * 64, True, 0, []])
def test_nullable_conversation_digest_still_refuses_malformed_present_values(digest):
    with pytest.raises(ValidationError, match="conversation_sha256"):
        TypeAdapter(DecisionPathPayload).validate_python(
            {**_rejected_path(), "conversation_sha256": digest}
        )


def test_nullable_digest_remains_required():
    payload = _rejected_path()
    del payload["conversation_sha256"]
    with pytest.raises(ValidationError, match="conversation_sha256"):
        TypeAdapter(DecisionPathPayload).validate_python(payload)


@pytest.mark.parametrize("label", ["Policy Judge", "界" * 100])
def test_reference_display_metadata_has_named_bounded_schema_and_preserves_extensions(label):
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


async def test_real_mcp_transport_preserves_rejected_path_without_a_digest(monkeypatch):
    plugin = QuarryPlugin()
    response = await plugin.scan(rule_text=RULE, conversation={"messages": []})
    assert response["completion_state"] == "invalid" and "matched" not in response
    # Model the producer that refuses input before hashing. The transport must keep its
    # explicit null, rather than rejecting it or replacing it with an invented digest.
    response["decision_path"]["conversation_sha256"] = None

    async def scan(**_kwargs):
        return response

    monkeypatch.setattr(plugin, "scan", scan)
    monkeypatch.setattr(server, "_service", plugin)
    result = await server.mcp.call_tool(
        "quarry_scan", {"rule_text": RULE, "conversation": {"messages": []}}
    )
    assert result.structured_content == response
    assert (
        json.loads("".join(item.text for item in result.content if item.type == "text")) == response
    )


async def test_real_mcp_reference_preserves_optional_display_metadata(monkeypatch):
    plugin = QuarryPlugin()
    response = plugin.reference(action="index", limit=1)
    response["model_aliases"] = {"judge": {"operations": ["score"], "display_name": "Policy Judge"}}
    assert TypeAdapter(ReferenceResponse).validate_python(response) == response
    monkeypatch.setattr(plugin, "reference", lambda **_kwargs: response)
    monkeypatch.setattr(server, "_service", plugin)
    result = await server.mcp.call_tool("quarry_reference", {"action": "index", "limit": 1})
    assert result.structured_content == response
    assert (
        json.loads("".join(item.text for item in result.content if item.type == "text")) == response
    )
