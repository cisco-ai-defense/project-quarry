# Copyright 2026 Cisco Systems, Inc. and its affilitates
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

"""Rules the compiler cannot represent fail to load with a reason; nearby shapes still load.

A rule that would need a construct the engine does not express is rejected when the policy is
compiled, never run with a silently different meaning. Every model the rules name is configured
(served by a fake adapter that is never called), so the compiler, not resource preparation,
decides each row.
"""

from __future__ import annotations

import re

import pytest

from quarry import compile_policy
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.types.exceptions import RuleValidationError


class _UncalledAdapter:
    """Every adapter operation the configured models declare; compilation calls none of them."""

    async def classify_messages(self, config, messages, tools, *, timeout=None):
        raise AssertionError("compiling a policy ran a model")

    async def decode_spans(self, config, raw_result, messages, threshold):
        raise AssertionError("compiling a policy ran a model")

    async def score(self, config, messages, *, timeout=None):
        raise AssertionError("compiling a policy ran a model")

    async def tokenize(self, config, messages, tools, *, timeout=None):
        raise AssertionError("compiling a policy ran a model")


_MODEL = {
    "adapter": "fake",
    "model": "physical-model",
    "category": "PII",
    "classify": {"class_index": 0},
    "spans": {"class_index": 1},
    "score": {"message_builder": "passthrough", "boundary_seed": 7},
    "tokens": {"adapter": "fake", "endpoint": None, "extraction": {"count": "count"}},
}


@pytest.fixture(scope="module")
def compile_rule_policy(tmp_path_factory):
    """``compile_policy`` under a version 2 configuration that serves every model name used."""
    directory = tmp_path_factory.mktemp("policy")
    (directory / "placeholder.rule").write_text('rule "Placeholder"\ncondition: true\n')
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "rules": [str(directory / "placeholder.rule")],
            "models": {name: dict(_MODEL) for name in ("privacy", "other", "judge", "guard", "m")},
        },
        directory,
    )
    backend = ModelInferenceBackend(
        config,
        adapters={"fake": _UncalledAdapter()},
        message_builders={"passthrough": lambda _constitution, messages, *_: messages},
    )

    def compile_rule(source: str):
        return compile_policy(source, runtime_config=config, model_inference_backend=backend)

    return compile_rule


class _UncalledV1Adapter:
    """The version 1 adapter operations a scorer with a tokenizer declares; none is called."""

    async def score(self, config, messages, *, timeout=None):
        raise AssertionError("compiling a policy ran a model")

    async def tokenize(self, config, messages, tools, *, timeout=None):
        raise AssertionError("compiling a policy ran a model")

    async def complete(self, config, messages, tools, *, timeout=None):
        raise AssertionError("compiling a policy ran a model")


@pytest.fixture(scope="module")
def compile_v1_policy(tmp_path_factory):
    """``compile_policy`` under a version 1 configuration with the ``gemma4`` scorer."""
    directory = tmp_path_factory.mktemp("policy_v1")
    (directory / "placeholder.rule").write_text('rule "Placeholder"\ncondition: true\n')
    scorer = {
        "adapter": "fake",
        "model": "test/gemma4",
        "endpoint": "https://model.example",
        "tokens": {"adapter": "fake", "endpoint": "/tokenize", "extraction": {"count": "count"}},
        "score": {
            "message_builder": "builder",
            "boundary_seed": 7,
            "tokens": {"negative": "0", "positive": "1"},
            "extraction": {"probabilities": "probabilities"},
            "calibration": "calibration.joblib",
        },
    }
    config = QuarryConfig.from_mapping(
        {"rules": [str(directory / "placeholder.rule")], "models": {"gemma4": scorer}}, directory
    )
    backend = ModelInferenceBackend(
        config,
        adapters={"fake": _UncalledV1Adapter()},
        message_builders={"builder": lambda *_: []},
        calibrator_loader=lambda _path: lambda score: score,
    )

    def compile_rule(source: str):
        return compile_policy(source, runtime_config=config, model_inference_backend=backend)

    return compile_rule


def _rule(condition: str, *strings: str) -> str:
    block = "strings:\n" + "".join(f"    {s}\n" for s in strings) if strings else ""
    return f'rule "R"\n{block}condition:\n    {condition}\n'


def _collect(predicate: str, source: str = 'model("privacy").spans(messages, 0.70)') -> str:
    return (
        'rule "R"\nstrings:\n    $allowed = "example.com"\ncondition: true\ncollect:\n'
        f"    [span for span in {source} if {predicate}]\n"
    )


_FORTY_STRINGS = tuple(f'$e{i} = "w{i}x"' for i in range(40))
_TWENTY_OF_FORTY = "any m in user_messages where 20 of ({}) in m.content".format(
    ", ".join(f"$e{i}" for i in range(40))
)


_REJECTED = [
    pytest.param(
        _rule('count([t for t in tools if t.description ~ "always"]) >= 2'),
        "static surface 'tools' is not supported",
        id="filter-count-over-tool-definitions",
    ),
    pytest.param(
        _rule('count([m for m in user_messages if called_tool("x")]) >= 2'),
        "needs a filter that only reads the element itself",
        id="filter-count-with-a-conversation-wide-filter",
    ),
    pytest.param(
        _rule('count([m for m in user_messages if m.content ~ "a"]) >= 2.5'),
        "2.5 is not a meaningful bound",
        id="filter-count-with-a-fractional-bound",
    ),
    pytest.param(
        _rule('count([m for m in user_messages if m.content ~ "a"]) >= 1000'),
        "nested too deeply to compile",
        id="filter-count-with-a-runaway-bound",
    ),
    pytest.param(
        _rule(
            'count([m for m in user_messages if (m.content ~ "a") after (m.content ~ "b")]) >= 2'
        ),
        "a temporal flow inside count([...]) is not supported",
        id="filter-count-over-a-flow",
    ),
    pytest.param(
        _rule("any t in tools where any m in user_messages where m.content contains t.name"),
        "cannot be bound from `tools`",
        id="join-bound-from-a-tool-definition",
    ),
    pytest.param(
        _rule(
            'any t in tools where ("zz" in t.description as v) and any m in user_messages where m.content contains v'
        ),
        "cannot be bound from `tools`",
        id="capture-bound-from-a-tool-definition",
    ),
    pytest.param(
        _rule(
            'any m in user_messages where ("TOKEN" in m.content as v) and any a in assistant_messages where ("TOKEN" in a.content as v) and any t in tool_responses where t.content contains v'
        ),
        "multiple distinct value-capture bind sites",
        id="capture-with-two-bind-sites",
    ),
    pytest.param(
        _rule(
            'any msg in user_messages where model_score("m", msg.content, len(msg.content)) >= 0.8'
        ),
        "'class_index' must be written as a literal value",
        id="model-output-index-not-literal",
    ),
    pytest.param(
        _rule('any msg in user_messages where model_scan("m", msg.content, len(msg.content), 1)'),
        "'threshold' must be written as a literal value",
        id="model-threshold-not-literal",
    ),
    pytest.param(
        _rule('any c in tool_calls where c.arguments.paths[c.arguments.i] contains "x"'),
        "needs a literal integer index",
        id="field-index-not-literal",
    ),
    pytest.param(
        _rule('any c in tool_calls where count(c.name, "x") + count(c.arguments, "y") > 2'),
        "not expressible in a step",
        id="count-sum-across-two-fields",
    ),
    pytest.param(
        _rule("any m in user_messages where @a* in m.content < 5", '$a1 = ["x", "y"]'),
        "multi-pattern (unsupported)",
        id="wildcard-position-over-a-pattern-list",
    ),
    pytest.param(
        _rule('count(tools, "x") > 2'),
        "unsupported top-level condition",
        id="occurrence-count-over-tool-definitions",
    ),
    pytest.param(
        _rule('model_score("m", tools, 1) > 0.8'),
        "unsupported top-level condition",
        id="model-score-over-tool-definitions",
    ),
    pytest.param(
        _rule(
            'any m in user_messages where not (@a in m.content) after(within: 3) any n in user_messages where "x" in n.content',
            '$a = "a"',
        ),
        "could not be compiled",
        id="negated-position-as-a-flow-operand",
    ),
    pytest.param(
        _rule(_TWENTY_OF_FORTY, *_FORTY_STRINGS),
        "exceeded its work budget of 5,000,000 formula-node visits",
        id="n-of-expansion-past-the-work-budget",
    ),
    pytest.param(
        _rule('model("privacy").spans(messages, 0.70) != model("privacy").spans(messages, 0.70)'),
        "supported only as the direct collect source",
        id="spans-in-the-boolean-condition",
    ),
    pytest.param(
        _collect('model("other").spans(messages, 0.70) == model("other").spans(messages, 0.70)'),
        "supported only as the direct collect source",
        id="spans-in-the-collect-filter",
    ),
    pytest.param(
        _collect('model("other").split(messages, 200) == model("other").split(messages, 200)'),
        "supported only as comprehension sources",
        id="split-in-the-collect-filter",
    ),
    pytest.param(
        _collect("true", 'model("privacy").spans(message("user", "synthetic"), 0.70)'),
        "must be a view of the original conversation",
        id="spans-over-a-synthetic-message",
    ),
    pytest.param(
        _collect("true", 'model("privacy").spans(messages, 1.1)'),
        "threshold must be at most 1.0",
        id="spans-threshold-above-one",
    ),
    pytest.param(
        _collect("true", 'model("guard").split(messages, 200)'),
        "source must be a direct model().spans() call",
        id="collect-over-chunks",
    ),
    pytest.param(
        _collect('span.score.value >= 0.60 or model("judge").classify(messages) >= 0.80'),
        "may be combined only with top-level 'and'",
        id="collect-model-filter-under-or",
    ),
    pytest.param(
        _collect("messages.extract(span) in $allowed"),
        "a string set is supported only as '$set in text'",
        id="collect-string-set-on-the-right",
    ),
    pytest.param(
        _rule('join([c.name for c in tool_calls], ",") contains "x"'),
        "join() is supported only as message() or add_message() content in a model input",
        id="join-in-a-condition",
    ),
    pytest.param(
        _rule("[c.name for c in tool_calls]"),
        "a map comprehension is supported only as the first argument of join()",
        id="map-comprehension-as-a-condition",
    ),
    pytest.param(
        _rule("len([c.name for c in tool_calls]) > 1"),
        "a map comprehension is supported only as the first argument of join()",
        id="length-of-a-map-comprehension",
    ),
    pytest.param(
        _rule("[c.name for c in messages[-1].tool_calls]"),
        "a map comprehension is supported only as the first argument of join()",
        id="map-comprehension-over-the-newest-message-as-a-condition",
    ),
    pytest.param(
        _rule("len([c.name for c in messages[-1].tool_calls]) > 1"),
        "a map comprehension is supported only as the first argument of join()",
        id="length-of-a-map-comprehension-over-the-newest-message",
    ),
    pytest.param(
        _rule(
            'model("judge").score(message("user", [c.name for c in messages[-1].tool_calls]), '
            '"policy") > 0.9'
        ),
        "a map comprehension is supported only as the first argument of join()",
        id="map-comprehension-as-message-content",
    ),
    pytest.param(
        _rule(
            'any c in tool_calls where model("judge").score(message("user", join([c.name for c '
            'in messages[-1].tool_calls], ",")), "policy") > 0.9'
        ),
        "every model call in an any comprehension must depend on its bound item",
        id="map-variable-that-hides-the-element-is-not-a-read-of-it",
    ),
    pytest.param(
        _rule('any m in [c.name for c in messages[-1].tool_calls] where m contains "a"'),
        "a map comprehension is supported only as the first argument of join()",
        id="map-comprehension-as-the-collection-of-any",
    ),
    pytest.param(
        _rule('all m in [c.name for c in messages[-1].tool_calls] where m contains "a"'),
        "a map comprehension is supported only as the first argument of join()",
        id="map-comprehension-as-the-collection-of-all",
    ),
    pytest.param(
        _rule(
            'count([m for m in [c.name for c in messages[-1].tool_calls] if m contains "a"]) > 0'
        ),
        "a map comprehension is supported only as the first argument of join()",
        id="map-comprehension-as-the-collection-of-a-filter",
    ),
    pytest.param(
        _rule(
            "any m in [c.arguments.command for c in messages[-1].tool_calls] where "
            'model("judge").score(message("user", m), "policy") > 0.9'
        ),
        "a map comprehension is supported only as the first argument of join()",
        id="a-judge-of-each-value-of-a-map-comprehension",
    ),
    pytest.param(
        _rule('any m in join([c.name for c in tool_calls], ",") where m contains "a"'),
        "join() is supported only as message() or add_message() content in a model input",
        id="join-as-the-collection-of-any",
    ),
]


@pytest.mark.parametrize(("source", "reason"), _REJECTED)
def test_rule_is_rejected_at_load_with_its_reason(compile_rule_policy, source, reason):
    with pytest.raises(RuleValidationError, match=re.escape(reason)):
        compile_rule_policy(source)


_LOADED = [
    pytest.param(
        _collect("true", 'model("privacy").spans(messages[1:], 0.70)'),
        id="spans-over-a-message-slice",
    ),
    pytest.param(
        'rule "R"\nstrings:\n    $allowed = "^[a-z]+@example\\\\.com$" regex\n'
        'condition: model("privacy").classify(messages) >= 0.60\ncollect:\n'
        '    [span for span in model("privacy").spans(messages, 0.70) '
        "if not $allowed in messages.extract(span)]\n",
        id="model-gate-with-a-regex-removal-filter",
    ),
    pytest.param(
        _collect("true", 'model("privacy").spans(user_messages[:2], 0.70)'),
        id="spans-over-a-message-view",
    ),
    pytest.param(
        _collect("$allowed in messages.extract(span, 4, 8)"),
        id="collect-filter-on-bounded-span-context",
    ),
    pytest.param(
        _collect(
            'span.score.value >= 0.60 and model("judge").classify(message("user", messages.extract(span))) >= 0.80'
        ),
        id="collect-local-and-model-filter",
    ),
    pytest.param(
        _rule('called_tool("x") after(within: 2) called_tools("a")'), id="flow-after-one-named-call"
    ),
    pytest.param(
        _rule(
            'any c in model("guard").split(messages, 200) where model("guard").score(c, "policy") > 0.9 and (called_tool("b") after called_tool("a")) and (called_tool("d") after called_tool("c"))'
        ),
        id="chunked-body-with-two-flows",
    ),
    pytest.param(
        _rule(
            '(any m in user_messages where m.content contains "a" and (called_tool("x") after(within:30) called_tool("y"))) and (any n in assistant_messages where n.content contains "b" and (called_tool("z") after(within:30) called_tool("w")))'
        ),
        id="two-closed-flows-within-the-default-budget",
    ),
    pytest.param(
        _rule(
            'any m in user_messages where (called_tool("a") after(within: 50) m.content contains "x") and (called_tool("b") after(within: 50) m.content contains "x")'
        ),
        id="element-in-two-long-flows",
    ),
    pytest.param(
        'rule "R"\nstrings:\n    $wipe = "rm -rf" regex\n    $pipe = "curl .*sh" regex\n'
        "variables:\n    risky = $wipe in c.arguments.* or $pipe in c.arguments.*\ncondition:\n    "
        + "\n    or ".join(
            f'model("judge").score(message("user", join([c.arguments.{field} for c in '
            'messages[-1].tool_calls if risky], "\\n")), "policy") > 0.9'
            for field in ("command", "cmd", "CommandLine")
        )
        + "\n",
        id="one-joined-judge-for-each-command-field-of-the-newest-message",
    ),
]


@pytest.mark.parametrize("source", _LOADED)
def test_rule_loads(compile_rule_policy, source):
    compile_rule_policy(source)  # raises RuleValidationError if the rule does not load


_LOADED_V1 = [
    pytest.param(
        'any c in llm_split_conversation("gemma4", conversation(), 200) where '
        'llm_score("gemma4", "policy", c) > 0.9 and (called_tool("b") after called_tool("a")) '
        'and (called_tool("d") after called_tool("c"))',
        id="chunked-score-beside-two-flows",
    ),
    pytest.param(
        'count([c for c in llm_split_conversation("gemma4", conversation(), 200) '
        'if llm_score("gemma4", "policy", c) > 0.9]) >= 2',
        id="chunked-count",
    ),
    pytest.param(
        'llm_score("gemma4", "policy", messages[1:]) > 0.5', id="score-over-a-message-slice"
    ),
]


@pytest.mark.parametrize("condition", _LOADED_V1)
def test_version_1_model_rule_loads(compile_v1_policy, condition):
    compile_v1_policy(_rule(condition))  # raises RuleValidationError if the rule does not load
