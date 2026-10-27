# Copyright 2025 Cisco Systems, Inc. and its affilitates
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

"""Named-variable expression sugar (``variables:`` block).

A variable is inlined into the condition before compilation, so a rule written with variables
answers exactly as the same rule written inline. Inlining is bounded: an expansion that would
exceed the node budget is rejected at parse time instead of being built. Load-time errors for
cycles, duplicates and reserved names live with the other admission errors.
"""

import asyncio
from dataclasses import replace

import pytest

from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.scanner import DSLScanner
from quarry.types.exceptions import RuleParseError

from ..conftest import rule_matches

_NAMED = """rule "R"
variables:
    strong   = any m in user_messages where m.content contains "alpha"
    weak     = any m in user_messages where m.content contains "beta"
    flagged  = any m in user_messages where m.content contains "gamma"
    combined = weak and flagged
condition: strong or combined
"""
_INLINE = """rule "R"
condition:
    (any m in user_messages where m.content contains "alpha")
    or ((any m in user_messages where m.content contains "beta")
        and (any m in user_messages where m.content contains "gamma"))
"""


@pytest.mark.parametrize(
    ("contents", "matched"),
    [
        pytest.param(["alpha"], True, id="first-alternative"),
        pytest.param(["beta", "gamma"], True, id="variable-built-from-other-variables"),
        pytest.param(["beta"], False, id="half-of-a-conjunction"),
        pytest.param(["delta"], False, id="no-variable-holds"),
    ],
)
def test_a_rule_with_variables_answers_like_the_same_rule_written_inline(contents, matched):
    conversation = {"messages": [{"role": "user", "content": c} for c in contents]}

    assert rule_matches(_NAMED, conversation) is matched
    assert rule_matches(_INLINE, conversation) is matched


@pytest.mark.parametrize(
    ("rule", "content", "matched"),
    [
        pytest.param(
            'rule "R"\nvariables:\n    msg = user_said("zzz")\n'
            'condition: any msg in user_messages where msg.content contains "attack"\n',
            "attack",
            True,
            id="a-loop-variable-shadows-a-declared-variable",
        ),
        pytest.param(
            'rule "R"\nvariables:\n    hit = user_said("x")\n'
            'condition: any m in user_messages where m.content contains "y" or hit\n',
            "x",
            True,
            id="an-unshadowed-variable-is-still-inlined-in-the-body",
        ),
    ],
)
def test_variables_respect_comprehension_scope(rule, content, matched) -> None:
    assert rule_matches(rule, {"messages": [{"role": "user", "content": content}]}) is matched


@pytest.mark.parametrize(
    "rule",
    [
        pytest.param(
            'rule "collision"\nvariables:\n    hit = false\n'
            'condition: called_tool("danger") as hit and hit\n',
            id="adjacent-scalar-alias",
        ),
        pytest.param(
            'rule "collision"\nvariables:\n    hit = false\n'
            'condition: (called_tool("danger") as hit) and true and hit\n',
            id="nested-left-conjunction",
        ),
        pytest.param(
            'rule "collision"\nstrings:\n    $id = "ID-[A-Z]" regex\n'
            "variables:\n    hit = false\n"
            "condition: (any m in user_messages where ($id in m.content as hit)) "
            "and (any a in assistant_messages where a.content contains hit)\n",
            id="capture-across-comprehensions",
        ),
        pytest.param(
            'rule "collision"\nvariables:\n    hit = false\n'
            '    guarded = called_tool("danger") as hit and hit\n'
            "condition: guarded\n",
            id="alias-in-named-expression",
        ),
        pytest.param(
            'rule "collision"\nvariables:\n    hit = called_tool("danger")\n'
            "condition: hit as hit and hit\n",
            id="alias-expression-reads-declaration",
        ),
        pytest.param(
            'rule "collision"\nvariables:\n    hit = false\n'
            'condition: (called_tool("danger") as hit and hit) or hit\n',
            id="alias-and-sibling-variable",
        ),
        pytest.param(
            'rule "collision"\nvariables:\n    hit = false\n'
            "condition: true\n"
            'collect: [m for m in messages if called_tool("danger") as hit and hit]\n',
            id="alias-in-collect-filter",
        ),
    ],
)
def test_variable_name_colliding_with_as_binding_is_rejected_at_load(rule) -> None:
    with pytest.raises(RuleParseError, match=r"variables: 'hit' conflicts with `as hit`"):
        DSLScanner(QuarryConfig(quarry_rule_texts=(rule,)))


def test_distinct_as_binding_and_named_variable_remain_accepted() -> None:
    rule = (
        'rule "safe"\nvariables:\n    configured = false\n'
        'condition: (called_tool("danger") as local and local) or configured\n'
    )
    scanner = DSLScanner(QuarryConfig(quarry_rule_texts=(rule,)))
    assert "safe" in scanner.loaded_rules
    result = scanner.scan_data(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "danger", "arguments": "{}"},
                        }
                    ],
                }
            ]
        }
    )
    assert result.is_complete, result.completion_issues
    assert result.rule_matches == ["safe"]


def _fan_out_bomb() -> str:
    """Each binding references the one below it three times: naive inlining is ~3**17 nodes."""
    lines = ["variables:", '    v0 = user_said("x")']
    lines += [f"    v{i} = v{i - 1} and v{i - 1} or v{i - 1}" for i in range(1, 18)]
    return "\n".join(lines) + "\n"


def _balanced_or(terms: list[str]) -> str:
    """Join ``terms`` with ``or`` as a balanced tree, so a large expression stays shallow."""
    if len(terms) == 1:
        return terms[0]
    middle = len(terms) // 2
    return f"({_balanced_or(terms[:middle])} or {_balanced_or(terms[middle:])})"


# Each is ~600-800 nodes: one hundred copies exceed the budget, two copies do not.
_LARGE_DISJUNCTION = _balanced_or([f'user_said("x{i}")' for i in range(256)])
_TWO_HUNDRED_PREDICATES = " or ".join(f'content contains "p{i}"' for i in range(200))


@pytest.mark.parametrize(
    "rule",
    [
        pytest.param(f'rule "Bomb"\n{_fan_out_bomb()}condition:\n    v17\n', id="fan-out-chain"),
        pytest.param(
            f'rule "Repeated"\nvariables:\n    v = {_LARGE_DISJUNCTION}\n'
            f"condition: {' or '.join(['v'] * 100)}\n",
            id="a-small-variable-repeated-past-the-budget",
        ),
        pytest.param(
            f'rule "Patterns"\nvariables:\n'
            f"    p = any m in messages where m matches {{ {_TWO_HUNDRED_PREDICATES} }}\n"
            f"condition: {' or '.join(['p'] * 100)}\n",
            id="every-predicate-of-a-pattern-block-counts",
        ),
    ],
)
def test_an_expansion_over_the_node_budget_is_rejected_at_parse(rule) -> None:
    with pytest.raises(RuleParseError, match="exceeds 50000 expression nodes"):
        DSLScanner(QuarryConfig(quarry_rule_texts=(rule,)))


def test_a_variable_repeated_within_the_budget_loads() -> None:
    rule = f'rule "Repeated"\nvariables:\n    v = {_LARGE_DISJUNCTION}\ncondition: v or v\n'

    assert rule_matches(rule, {"messages": [{"role": "user", "content": "x7"}]}) is True


def test_variables_reading_one_model_output_share_one_inference(tmp_path):
    calls: list[str] = []

    class _Adapter:
        async def classify_messages(self, _config, messages, _tools, *, timeout=None):
            calls.append(messages[0]["content"])
            return type("R", (), {"chunks": [type("C", (), {"probabilities": [0.7, 0.65]})()]})()

    rule = (
        'rule "pi"\n'
        "  variables:\n"
        '    pi       = model_score("m", msg.content, 0) >= 0.8\n'
        '    pi_lower = model_score("m", msg.content, 0) >= 0.5\n'
        '    toxic    = model_score("m", msg.content, 1) >= 0.6\n'
        "  condition: any msg in user_messages where pi or (pi_lower and toxic)\n"
    )
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "m": {"adapter": "fake", "model": "test-classifier", "classify": {"class_index": 0}}
            },
        },
        tmp_path,
        require_rules=False,
    )
    config = replace(config, quarry_rule_texts=(rule,))
    scanner = DSLScanner(
        config,
        model_inference_backend=ModelInferenceBackend(config, adapters={"fake": _Adapter()}),
    )

    result = asyncio.run(scanner.scan_data_async({"messages": [{"role": "user", "content": "hi"}]}))

    # [0.7, 0.65]: pi is false, but pi_lower and toxic both hold.
    assert result.rule_matches == ["pi"]
    # Three thresholds over two class indices of the same input share one provider call.
    assert calls == ["hi"]
