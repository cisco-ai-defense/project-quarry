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

"""Equivalence gate: a decomposed rule decides as its fused automaton does, grounding less.

A rule compiled into its decision tree is walked one check at a time along its root's boot
guard, cheapest first (``quarry.engine.decision``). The walk must give the fused automaton's
verdict whenever both complete, must fail only where the fused rule also fails, and must
never ground a model check more often. The first-order property tests in
``test_semantic_properties.py`` already check decomposed verdicts of local rules against
their meaning; this gate adds model checks, which can fail or cost a call.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from quarry.decision_report import Digests, report_tree, tree_report
from quarry.dsl.parser import DSLParser
from quarry.engine.compiler import compile_rule
from quarry.engine.contracts import ScanRecord
from quarry.engine.events import build_evaluation_input
from quarry.engine.executor import evaluate_async
from quarry.types.context import ScanContext

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
CONVERSATIONS = sorted((EXAMPLES / "data").glob("*.json"))

# Local and model checks mixed under nested and/or/not, with models that cost more than
# the local checks beside them and checks that appear twice.
RULES = [
    '(any m in user_messages where m.content contains "list") '
    'and model("judge").score(messages, "fail-a") > 0.5',
    'model("judge").score(messages, "fail-a") > 0.5 '
    'or (any m in messages where m.content contains "sorted")',
    'not (any c in tool_calls where c.name == "send_email") '
    'or (model("judge").score(messages[-2:], "b") >= 0.5 '
    'and model("judge").score(messages, "fail-c") < 0.4)',
    '((any m in user_messages where m.content contains "a") '
    'and model("judge").score(messages, "b") > 0.5) '
    'or ((any m in assistant_messages where m.content contains "e") '
    'and model("judge").score(messages, "b") > 0.5)',
    '(any m in user_messages where m.content contains "Python") '
    'and not (model("judge").score(messages, "fail-a") > 0.5 '
    'or (any m in assistant_messages where m.content contains "slice"))',
]
_PARSER = DSLParser()


def _constitution(spec) -> str:
    compare = spec.model_expression
    score = next(child for child in compare.children if child.kind == "model_call")
    return str(score.children[-1].value)


def _run(artifact, conversation, failing: str | None):
    """Evaluate with a judge that is deterministic per check and raises for ``failing``."""

    calls: Counter[str] = Counter()

    async def judge(spec, _text) -> bool:
        calls[spec.atom_id] += 1
        if failing is not None and _constitution(spec) == failing:
            raise RuntimeError(f"judge failed on {failing}")
        return hashlib.sha256(spec.atom_id.encode()).digest()[0] % 2 == 0

    context = ScanContext(messages=conversation["messages"], tools=conversation.get("tools", []))
    recorder = ScanRecord()
    try:
        result = asyncio.run(
            evaluate_async(
                artifact,
                build_evaluation_input(context),
                async_detector=judge,
                minimize_witness=False,
                record=recorder,
            )
        )
    except RuntimeError:
        return None, calls, recorder
    return result.matched, calls, recorder


@pytest.mark.parametrize("condition", RULES)
def test_a_decomposed_rule_decides_as_its_fused_automaton_with_fewer_calls(condition):
    rule = _PARSER.parse(f'rule "R"\ncondition:\n    {condition}\n')
    fused = compile_rule(rule, decompose=False).artifact
    tree = compile_rule(rule).artifact
    assert tree.decision_tree.decomposed
    walked = report_tree(tree, Digests(str, str, str))
    failures = [None, "fail-a", "fail-c"]
    for path in CONVERSATIONS:
        conversation = json.loads(path.read_text())
        for failing in failures:
            fused_verdict, fused_calls, _ = _run(fused, conversation, failing)
            tree_verdict, tree_calls, recorder = _run(tree, conversation, failing)
            case = (condition, path.name, failing)
            if tree_verdict is None:
                assert fused_verdict is None, case  # fails only where the fused rule fails
            elif fused_verdict is not None:
                assert tree_verdict == fused_verdict, case
            # A walk grounds each check at most once, and only checks the fused rule reads.
            for atom_id, count in tree_calls.items():
                assert count <= fused_calls[atom_id], case
            # The recorded checks walk the tree from its root to the verdict's leaf.
            key = walked.root
            for check in recorder.checks:
                step = walked.steps[key]
                assert step.node.atom_id == check.atom_id, case
                key = step.on_true if check.value else step.on_false
            if tree_verdict is not None:
                assert key == ("match" if tree_verdict else "no_match"), case


def test_exploration_grounds_skipped_checks_without_changing_the_verdict():
    # Without "story" the path stops at the local check; exploration also scores the
    # skipped model check, and a model that fails there leaves the decided verdict.
    from quarry.types.exceptions import ModelError

    rule = _PARSER.parse(
        'rule "R"\ncondition:\n    (any m in user_messages where m.content contains "story")\n'
        '    and model("judge").score(messages, "fiction") > 0.5\n'
    )
    tree = compile_rule(rule).artifact
    context = ScanContext(messages=[{"role": "user", "content": "hello"}], tools=[])

    async def passing(_spec, _text) -> bool:
        return True

    async def failing(_spec, _text) -> bool:
        raise ModelError("judge unavailable")

    async def explored(detector) -> tuple:
        recorder = ScanRecord()
        result = await evaluate_async(
            tree,
            build_evaluation_input(context),
            async_detector=detector,
            minimize_witness=False,
            record=recorder,
        )
        await result.explore(False, None)
        return result, recorder

    for detector, off_path in ((passing, (True, None)), (failing, (None, "model_failed"))):
        result, recorder = asyncio.run(explored(detector))
        assert result.matched is False
        on_path, skipped = recorder.checks
        assert (on_path.value, on_path.origin) == (False, "verdict")
        assert (skipped.value, skipped.reason, skipped.origin) == (*off_path, "explored")


def test_a_capability_prober_is_not_a_model_of_its_node():
    # A prober's first argument is a tag, not a model alias. With no model configured at
    # all, keying the node by that tag as a model would fail to report the tree.
    rule = _PARSER.parse(
        'rule "R"\ncondition:\n    (any c in tool_calls where capability_tags("code_execution", c.name))\n'
        '    and (any m in user_messages where m.content contains "run")\n'
    )
    configured: dict[str, str] = {}
    digests = Digests(configured.__getitem__, configured.__getitem__, configured.__getitem__)
    tree = tree_report(compile_rule(rule).artifact, digests)
    assert [node["kind"] for node in tree["nodes"].values()] == ["check", "check"]
