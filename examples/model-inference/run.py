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

"""Run the classification example with a deterministic local adapter."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from quarry import DSLScanner, ModelInferenceBackend
from quarry.config import QuarryConfig

HERE = Path(__file__).resolve().parent


class ExampleAdapter:
    """Return a fixed classifier response without network access."""

    async def classify_messages(self, _config, _messages, _tools, *, timeout=None):
        del timeout
        return SimpleNamespace(chunks=[SimpleNamespace(probabilities=[0.1, 0.9])])


def main() -> int:
    config = replace(
        QuarryConfig.from_yaml(HERE / "config.example.yaml"),
        quarry_rules_paths=[HERE / "classify.rule"],
    )
    backend = ModelInferenceBackend(config, adapters={"example": ExampleAdapter()})
    scanner = DSLScanner(config, model_inference_backend=backend)
    conversation = json.loads((HERE / "conversation.json").read_text(encoding="utf-8"))
    result = scanner.scan_data(conversation)
    result.raise_for_status()
    print(result.rule_matches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
