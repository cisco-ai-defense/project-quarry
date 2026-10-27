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

"""Run a parameterized policy with a deterministic local score adapter."""

from __future__ import annotations

import json
from pathlib import Path

from quarry import ModelInferenceBackend, compile_policy
from quarry.config import QuarryConfig

HERE = Path(__file__).resolve().parent


class ExampleAdapter:
    """Return a fixed score without network access."""

    async def score(self, _config, _messages, *, timeout=None):
        del timeout
        return 0.2


def main() -> int:
    source = (HERE / "policy.rule").read_text(encoding="utf-8")
    config = QuarryConfig.from_yaml(HERE / "config.yaml")
    backend = ModelInferenceBackend(config, adapters={"example": ExampleAdapter()})
    policy = compile_policy(
        source,
        policy_parameters={"instructions": "Score requests for protected credentials highly."},
        runtime_config=config,
        model_inference_backend=backend,
    )
    conversation = json.loads((HERE / "conversation.json").read_text(encoding="utf-8"))
    result = policy.scan(conversation, parameters={"threshold": 0.8})
    result.raise_for_status()
    print(result.rule_matches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
