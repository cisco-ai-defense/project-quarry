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

"""A toy adapter for confirmed writes to an external memory store."""

from pydantic import BaseModel, ConfigDict

from quarry import ExtensionSpec


class MemoryWrite(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    namespace: list[str]
    key: str
    tool_call_id: str | None = None
    source_tool: str | None = None


extension = ExtensionSpec("memory", {"write_committed": MemoryWrite})


def confirmed_write_event(
    *,
    event_id: str,
    run_id: str,
    namespace: list[str],
    key: str,
    tool_call_id: str | None = None,
    source_tool: str | None = None,
) -> dict:
    """Adapt a confirmed store write to Quarry's JSON event envelope."""
    payload = {"namespace": namespace, "key": key}
    if tool_call_id is not None:
        payload["tool_call_id"] = tool_call_id
    if source_tool is not None:
        payload["source_tool"] = source_tool
    return {
        "id": event_id,
        "type": "memory.write_committed",
        "run_id": run_id,
        "payload": payload,
    }
