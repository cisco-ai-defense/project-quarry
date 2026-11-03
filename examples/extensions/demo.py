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

"""Register two unrelated domains and scan their JSON facts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from quarry import DSLScanner, ExtensionSpec
from quarry.config import QuarryConfig


class MemoryWrite(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    key: str
    source_tool: str | None = None


class FileRead(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    allowed: bool


class SandboxPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    mode: Literal["observe", "enforce"] = "enforce"


MEMORY_RULE = """rule "Memory write after tool A"
condition:
    (any write in memory.write_committed where write.payload.source_tool == "tool_a")
    right_after
    called_tool("tool_a")"""

SANDBOX_RULE = '''rule "Sandbox denied file read"
condition:
    any read in sandbox.file_read where
        read.payload.allowed == false and sandbox.state.mode == "enforce"'''


def run_demo() -> tuple[list[str], list[str]]:
    scanner = DSLScanner(
        QuarryConfig(config_version=2, quarry_rule_texts=(MEMORY_RULE, SANDBOX_RULE)),
        extensions=[
            ExtensionSpec("memory", {"write_committed": MemoryWrite}),
            ExtensionSpec("sandbox", {"file_read": FileRead}, state=SandboxPolicy),
        ],
    )

    conversation = {
        "messages": [
            {"role": "user", "content": "Run tool A"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-7",
                        "type": "function",
                        "function": {"name": "tool_a", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-7", "content": "done"},
        ]
    }
    memory_event = {
        "id": "write-1",
        "type": "memory.write_committed",
        "run_id": "run-1",
        "payload": {"key": "preference", "source_tool": "tool_a"},
    }
    timeline = [
        {"message_index": 0},
        {"message_index": 1},
        {"tool_call_id": "call-7"},
        {"event_id": "write-1"},
        {"message_index": 2},
    ]
    memory_result = scanner.scan_data(
        conversation,
        events=[memory_event],
        timeline=timeline,
        rules_subset={"Memory write after tool A"},
    )
    memory_result.raise_for_status()

    sandbox_event = {
        "id": "read-1",
        "type": "sandbox.file_read",
        "run_id": "run-2",
        "payload": {"path": "/workspace/blocked.txt", "allowed": False},
    }
    sandbox_result = scanner.scan_data(
        None,
        events=[sandbox_event],
        extension_state={"sandbox": {}},
        rules_subset={"Sandbox denied file read"},
    )
    sandbox_result.raise_for_status()
    return memory_result.rule_matches, sandbox_result.rule_matches


if __name__ == "__main__":
    memory_matches, sandbox_matches = run_demo()
    print("memory:", memory_matches)
    print("sandbox:", sandbox_matches)
