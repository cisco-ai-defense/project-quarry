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

"""Public compile complexity metadata stays tied to validated rules."""

from __future__ import annotations

import pytest

from quarry.config import QuarryConfig
from quarry.plugin.service import QuarryPlugin

LOW_RULE = 'rule "Simple"\ncondition: true\n'
MEDIUM_RULE = (
    'rule "Two flows"\ncondition:\n'
    '    (called_tool("b") after(within: 3) called_tool("a")) and\n'
    '    (called_tool("d") after(within: 3) called_tool("c"))\n'
)
INVALID_RULE = 'rule "Invalid"\ncondition: made_up_function("x")\n'
CONVERSATION = {"messages": [{"role": "user", "content": "hello"}]}


def test_compile_reports_the_single_rules_validator_complexity():
    compilation = QuarryPlugin().compile(rule_text=LOW_RULE)

    assert compilation["valid"] is True
    assert compilation["rule_names"] == ["Simple"]
    assert compilation["estimated_complexity"] == "low"
    assert compilation["rule_complexities"] == {"Simple": "low"}


def test_compile_reports_each_rule_when_a_generation_contains_multiple_rules(monkeypatch):
    plugin = QuarryPlugin()
    monkeypatch.setattr(
        plugin,
        "_rule_config",
        lambda _rule_text, _config_version: QuarryConfig(quarry_rule_texts=(MEDIUM_RULE, LOW_RULE)),
    )

    compilation = plugin.compile(rule_text=LOW_RULE)

    assert compilation["valid"] is True
    assert compilation["rule_names"] == ["Simple", "Two flows"]
    assert compilation["estimated_complexity"] == "medium"
    assert compilation["rule_complexities"] == {"Simple": "low", "Two flows": "medium"}


def test_failed_compile_does_not_claim_an_estimated_complexity():
    compilation = QuarryPlugin().compile(rule_text=INVALID_RULE)

    assert compilation["valid"] is False
    assert "estimated_complexity" not in compilation
    assert "rule_complexities" not in compilation


def test_empty_scanner_generation_fails_instead_of_reporting_no_complexity(monkeypatch):
    class EmptyScanner:
        loaded_rules = {}
        resource_requirements = {}

        def __init__(self, _config, *, llm_rate_limiter, limits, _allow_parameters):
            pass

    monkeypatch.setattr("quarry.plugin.service.DSLScanner", EmptyScanner)

    compilation = QuarryPlugin().compile(rule_text=LOW_RULE)

    assert compilation["valid"] is False
    assert compilation["stage"] == "validate"
    assert "estimated_complexity" not in compilation
    assert "rule_complexities" not in compilation


@pytest.mark.asyncio
@pytest.mark.parametrize("visualize", [False, True])
async def test_scan_compilation_preserves_complexity_with_or_without_visualization(visualize):
    result = await QuarryPlugin().scan(
        rule_text=LOW_RULE, conversation=CONVERSATION, visualize=visualize
    )

    assert result["completion_state"] == "complete"
    assert result["compilation"]["estimated_complexity"] == "low"
    assert result["compilation"]["rule_complexities"] == {"Simple": "low"}


@pytest.mark.asyncio
async def test_failed_scan_compilation_does_not_claim_an_estimated_complexity():
    result = await QuarryPlugin().scan(rule_text=INVALID_RULE, conversation=CONVERSATION)

    assert result["completion_state"] == "invalid"
    assert result["compilation"]["valid"] is False
    assert "estimated_complexity" not in result["compilation"]
    assert "rule_complexities" not in result["compilation"]
