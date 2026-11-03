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

"""Run two independent Quarry extension domains without external services."""

from pathlib import Path

from examples.extensions.memory_extension import confirmed_write_event
from examples.extensions.memory_extension import extension as memory_extension
from examples.extensions.sandbox_extension import extension as sandbox_extension
from examples.extensions.sandbox_extension import file_read_event
from quarry import DSLScanner
from quarry.config import QuarryConfig

RULES = Path(__file__).resolve().parent


def run_demo() -> tuple[list[str], list[str]]:
    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            quarry_rule_texts=tuple(
                (RULES / name).read_text(encoding="utf-8")
                for name in ("memory.dsl", "memory_temporal.dsl", "sandbox.dsl")
            ),
        ),
        extensions=[memory_extension, sandbox_extension],
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
    memory_event = confirmed_write_event(
        event_id="write-1",
        run_id="run-1",
        namespace=["demo"],
        key="preference",
        tool_call_id="call-7",
        source_tool="tool_a",
    )
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
        rules_subset={
            "Tool A committed persistent memory",
            "Confirmed memory write immediately after tool A",
        },
    )
    memory_result.raise_for_status()

    sandbox_event = file_read_event(
        event_id="read-1",
        run_id="run-2",
        path="/workspace/blocked.txt",
        allowed=False,
    )
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
