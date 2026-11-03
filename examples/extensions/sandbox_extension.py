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

"""A toy adapter for an external sandbox's file-read decisions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from quarry import ExtensionSpec


class FileRead(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    allowed: bool


class SandboxPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    mode: Literal["observe", "enforce"] = "enforce"


extension = ExtensionSpec("sandbox", {"file_read": FileRead}, state=SandboxPolicy)


def file_read_event(*, event_id: str, run_id: str, path: str, allowed: bool) -> dict:
    """Adapt the sandbox runtime's authoritative file-read decision."""
    return {
        "id": event_id,
        "type": "sandbox.file_read",
        "run_id": run_id,
        "payload": {"path": path, "allowed": allowed},
    }
