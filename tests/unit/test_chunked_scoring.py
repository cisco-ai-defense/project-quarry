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

"""Chunked measurement over ``llm_split_conversation(...)``.

A rule quantifies (``any``/``all``) or counts (``count([...])``) a per-chunk measurement::

    any chunk in llm_split_conversation(m, conversation(), N) where llm_score(m, p, chunk) > t
    count([c for c in llm_split_conversation(m, conversation(), N) if <scalar>(..., c) > t]) >= K

The input is split into token-bounded chunks, every chunk is measured, and the measurements
decide the rule. The direction of the reduction follows both the quantifier and the comparator:
``any chunk ... < t`` holds when ONE chunk is low. The fake tokenizer counts wire bytes, so the
long conversation below splits into 16 chunks of 181 and 177 "tokens" at a 200-token budget.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.policy import compile_policy
from quarry.safety.limits import SafetyLimits
from quarry.scanner import DSLScanner
from quarry.types.exceptions import RuleValidationError
from tests.unit.model_fakes import (
    POLICY_TEXT,
    V1_MODEL,
    FakeAdapter,
    ModelAdapter,
    model_entry,
    passthrough_builder,
    set_config_environment,
    v1_backend,
)


class ByteTokenAdapter(FakeAdapter):
    """Count tokens as wire bytes; score chunks in order from ``scores`` (default 0.0)."""

    def __init__(self, scores: list[float] | None = None) -> None:
        super().__init__(scores=scores or [0.0] * 64)
        self.active = 0
        self.max_concurrent = 0

    async def tokenize(self, config, messages, tools, *, timeout=None):
        payload = json.dumps(
            {"messages": messages, "tools": tools}, separators=(",", ":"), sort_keys=True
        )
        return {config.tokens.count_field: len(payload.encode("utf-8"))}

    async def score(self, config, messages, *, timeout=None):
        self.active += 1
        self.max_concurrent = max(self.max_concurrent, self.active)
        try:
            return await super().score(config, messages, timeout=timeout)
        finally:
            self.active -= 1


class MarkedChunkAdapter(ByteTokenAdapter):
    """Score 0.85 only a chunk that holds the marker turn and is judged against POLICY_TEXT."""

    MARKER = "Turn 7: review"

    async def score(self, config, messages, *, timeout=None):
        await super().score(config, messages, timeout=timeout)
        text = json.dumps(messages)
        return 0.85 if self.MARKER in text and POLICY_TEXT in text else 0.0


def _long_conversation() -> dict:
    prefix = [
        {"role": "system", "content": "Keep context."},
        {"role": "developer", "content": "Keep order."},
    ]
    turns = [
        message
        for turn in range(8)
        for message in (
            {"role": "user", "content": f"Turn {turn}: review deployment constraints."},
            {"role": "assistant", "content": f"Turn {turn}: constraints recorded."},
        )
    ]
    return {"messages": [*prefix, *turns]}


_OTHER_MODEL = [line.replace("gemma4:", "other:") for line in V1_MODEL[1:21]]


@pytest.fixture(scope="module")
def config(tmp_path_factory) -> QuarryConfig:
    """Version 1 config: the ``gemma4`` scorer/tokenizer plus an ``other`` model like it."""

    directory = tmp_path_factory.mktemp("chunked")
    placeholder = directory / "placeholder.rule"
    placeholder.write_text('rule "placeholder"\ncondition: true\n', encoding="utf-8")
    config_path = directory / "config.yaml"
    config_path.write_text(
        "\n".join(["rules:", f"  - {placeholder}", *V1_MODEL[:21], *_OTHER_MODEL, *V1_MODEL[21:]]),
        encoding="utf-8",
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        set_config_environment(monkeypatch)
        return QuarryConfig.from_yaml(config_path)


def _scan(config, rules: dict[str, str], adapter, conversation=None, scored=None, limits=None):
    """Scan ``rules`` (name -> condition) in one scanner over ``conversation``."""

    texts = tuple(f'rule "{name}"\ncondition: {condition}\n' for name, condition in rules.items())
    scanner = DSLScanner(
        replace(config, quarry_rules_paths=(), quarry_rule_texts=texts),
        limits=limits,
        model_inference_backend=v1_backend(config, adapter, scored),
    )
    return scanner.scan_data(conversation or _long_conversation())


_SPLIT = 'llm_split_conversation("gemma4", conversation(), 200)'
_SCORE = 'llm_score("gemma4", "policy", c)'
_ONE_HIGH = [0.1] * 7 + [0.8] + [0.1] * 8
_ONE_LOW = [0.8] * 3 + [0.1] + [0.8] * 12
_ALL_HIGH = [0.8] * 16
_ALL_LOW = [0.1] * 16


@pytest.mark.parametrize(
    ("condition", "scores", "matched"),
    [
        # The calibrator adds 0.1: 0.8 scores 0.9 and 0.1 scores 0.2.
        pytest.param(f"any c in {_SPLIT} where {_SCORE} > 0.85", _ONE_HIGH, True, id="any-high"),
        pytest.param(f"any c in {_SPLIT} where {_SCORE} > 0.85", _ALL_LOW, False, id="any-none"),
        pytest.param(f"any c in {_SPLIT} where {_SCORE} >= 0.9", _ONE_HIGH, True, id="any-ge"),
        pytest.param(f"all c in {_SPLIT} where {_SCORE} > 0.85", _ONE_LOW, False, id="all-one-low"),
        pytest.param(f"all c in {_SPLIT} where {_SCORE} > 0.85", _ALL_HIGH, True, id="all-high"),
        pytest.param(f"any c in {_SPLIT} where {_SCORE} < 0.5", _ONE_LOW, True, id="any-lt-one"),
        pytest.param(f"any c in {_SPLIT} where {_SCORE} < 0.5", _ALL_HIGH, False, id="any-lt-none"),
        pytest.param(f"any c in {_SPLIT} where {_SCORE} <= 0.2", _ONE_LOW, True, id="any-le"),
        pytest.param(f"all c in {_SPLIT} where {_SCORE} < 0.5", _ONE_HIGH, False, id="all-lt-one"),
        pytest.param(f"all c in {_SPLIT} where {_SCORE} < 0.5", _ALL_LOW, True, id="all-lt-every"),
        pytest.param(f"all c in {_SPLIT} where {_SCORE} >= 0.9", _ALL_HIGH, True, id="all-ge"),
        pytest.param(
            f"any c in {_SPLIT} where 0.5 > {_SCORE}", _ONE_LOW, True, id="flipped-less-than"
        ),
        pytest.param(
            f"any c in {_SPLIT} where 0.85 < {_SCORE}", _ONE_HIGH, True, id="flipped-greater-than"
        ),
        pytest.param(
            f"not (any c in {_SPLIT} where {_SCORE} > 0.85)", _ALL_LOW, True, id="negated"
        ),
    ],
)
def test_a_chunk_quantifier_reduces_in_the_direction_of_its_comparison(
    config, condition, scores, matched
) -> None:
    adapter = ByteTokenAdapter(scores)

    result = _scan(config, {"R": condition}, adapter)

    assert (result.rule_matches, result.is_complete) == ((["R"] if matched else []), True)
    assert len(adapter.builder_messages) == 16
    assert adapter.max_concurrent > 1


def _count(op_and_bound: str, inner: str = f"{_SCORE} > 0.85", collection: str = _SPLIT) -> str:
    return f"count([c for c in {collection} if {inner}]) {op_and_bound}"


@pytest.mark.parametrize(
    ("condition", "positives", "matched"),
    [
        (_count(">= 2"), 3, True),
        (_count(">= 3"), 3, True),
        (_count(">= 4"), 3, False),
        # The case `any` would wrongly match: one tripping chunk is not two.
        (_count(">= 2"), 1, False),
        (_count(">= 1"), 0, False),
        (_count("== 1"), 1, True),
        (_count("== 1"), 2, False),
        (f"len([c for c in {_SPLIT} if {_SCORE} > 0.85]) >= 2", 3, True),
        (f"2 <= count([c for c in {_SPLIT} if {_SCORE} > 0.85])", 3, True),
    ],
    ids=[
        "three-of-two",
        "three-of-three",
        "three-of-four",
        "one-of-two",
        "none-of-one",
        "exactly-one",
        "two-is-not-exactly-one",
        "len-synonym",
        "flipped-bound",
    ],
)
def test_a_chunk_count_counts_every_chunk_that_passes(
    config, condition, positives, matched
) -> None:
    adapter = ByteTokenAdapter([0.8] * positives)

    result = _scan(config, {"R": condition}, adapter)

    assert (result.rule_matches, result.is_complete) == ((["R"] if matched else []), True)
    assert len(adapter.builder_messages) == 16


_TOKENS = 'llm_input_tokens("gemma4", c)'


@pytest.mark.parametrize(
    ("condition", "matched"),
    [
        (f"any c in {_SPLIT} where {_TOKENS} > 180", True),
        (f"all c in {_SPLIT} where {_TOKENS} > 180", False),
        (f"all c in {_SPLIT} where {_TOKENS} <= 200", True),
        (_count("> 3", inner=f"{_TOKENS} > 100"), True),
    ],
    ids=["largest-chunk", "smallest-chunk", "every-chunk-within-budget", "count"],
)
def test_a_token_count_reduction_measures_chunks_without_scoring_them(
    config, condition, matched
) -> None:
    adapter = ByteTokenAdapter()

    result = _scan(config, {"R": condition}, adapter)

    assert (result.rule_matches, result.is_complete) == ((["R"] if matched else []), True)
    assert adapter.builder_messages == []


def test_rules_that_differ_in_one_chunk_setting_get_their_own_verdicts(config) -> None:
    """Budget, reduction, scalar, text, inner threshold, and count-vs-any each change the answer."""

    named = 'llm_score("gemma4", text_ref("unsafe_shell"), c)'
    split = 'llm_split_conversation("gemma4", conversation(), {})'
    rules = {
        "AnySmall": f"any c in {split.format(200)} where {named} > 0.9",
        "AllSmall": f"all c in {split.format(200)} where {named} > 0.9",
        "AllWhole": f"all c in {split.format(100000)} where {named} > 0.9",
        "LiteralText": f'any c in {split.format(200)} where llm_score("gemma4", "unsafe_shell", c) > 0.9',
        "ScoreAboveTop": f"any c in {split.format(200)} where {named} > 0.96",
        "TokensAboveTop": f"any c in {split.format(200)} where {_TOKENS} > 0.96",
        "CountTwo": _count(">= 2", inner=f"{named} > 0.9"),
        "CountLowTwo": _count(">= 2", inner=f"{named} > 0.05"),
    }

    # Eight rules each split and measure the transcript: allow more than the default calls.
    result = _scan(config, rules, MarkedChunkAdapter(), limits=SafetyLimits(max_model_calls=1000))

    assert result.is_complete
    assert sorted(result.rule_matches) == ["AllWhole", "AnySmall", "CountLowTwo", "TokensAboveTop"]


_WITH_RUN = {
    "messages": [
        *_long_conversation()["messages"],
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "r1", "type": "function", "function": {"name": "run", "arguments": "{}"}}
            ],
        },
    ]
}


@pytest.mark.parametrize(
    ("conversation", "matched"), [(_long_conversation(), False), (_WITH_RUN, True)]
)
def test_a_conversation_wide_conjunct_in_a_chunk_body_is_checked_once(
    config, conversation, matched
) -> None:
    split = 'llm_split_conversation("gemma4", conversation(), 400)'
    condition = f'any c in {split} where {_SCORE} > 0.85 and called_tool("run")'

    result = _scan(config, {"R": condition}, ByteTokenAdapter(_ALL_HIGH + [0.8]), conversation)

    assert (result.rule_matches, result.is_complete) == ((["R"] if matched else []), True)


_EXCHANGE = {
    "messages": [
        {"role": "user", "content": "Read the deployment file and summarize it."},
        {
            "role": "assistant",
            "content": "Reading it now.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "/srv/deploy.yaml"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "replicas: 3, region: us-east-1"},
        {"role": "user", "content": "Now compare it with the staging configuration."},
        {"role": "assistant", "content": "Staging runs one replica in us-west-2."},
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
}


@pytest.mark.parametrize(
    ("collection", "expected_chunks"),
    [
        (
            'llm_split_conversation("gemma4", user_messages, 120)',
            [[_EXCHANGE["messages"][0]], [_EXCHANGE["messages"][3]]],
        ),
        (
            'llm_split_conversation("gemma4", messages, 400)',
            [_EXCHANGE["messages"][:3], _EXCHANGE["messages"][3:]],
        ),
    ],
    ids=["role-view", "tool-exchange-stays-whole"],
)
def test_a_messages_selection_splits_without_the_offered_tools(
    config, collection, expected_chunks
) -> None:
    adapter = ByteTokenAdapter()
    scored: list = []

    result = _scan(
        config,
        {"R": f'any c in {collection} where llm_score("gemma4", "p", c) > 0.5'},
        adapter,
        _EXCHANGE,
        scored,
    )

    assert result.is_complete
    assert scored == expected_chunks
    assert all(tools == [] for tools in adapter.tools)


@pytest.mark.parametrize(
    ("conversation", "reason"),
    [
        (
            {"messages": [{"role": "user", "content": "x" * 500}]},
            "A conversation turn exceeds max_input_tokens",
        ),
        (
            {
                "messages": [
                    {"role": "system", "content": "s" * 500},
                    {"role": "user", "content": "short"},
                ]
            },
            "System and developer messages exceed max_input_tokens",
        ),
    ],
    ids=["one-turn", "system-prefix"],
)
def test_a_conversation_that_cannot_be_split_fails_the_scan_closed(
    config, conversation, reason
) -> None:
    adapter = ByteTokenAdapter()

    result = _scan(config, {"R": f"any c in {_SPLIT} where {_SCORE} > 0.5"}, adapter, conversation)

    assert (result.rule_matches, result.completion_state) == ([], "incomplete")
    assert reason in result.evaluation_errors["R"]
    assert adapter.builder_messages == []


@pytest.mark.parametrize(
    ("condition", "reason"),
    [
        (f"any c in {_SPLIT} where {_SCORE} == 0.5", "needs an ordering comparison"),
        (f"any c in {_SPLIT} where {_SCORE} != 0.5", "needs an ordering comparison"),
        (_count(">= 1", inner=f"{_SCORE} == 0.9"), "per-chunk filter needs an ordering"),
        (
            f'any c in {_SPLIT} where llm_complete("gemma4", c) > 0.5',
            "must compare one of llm_input_tokens, llm_score",
        ),
        (
            f'any c in {_SPLIT} where llm_input_tokens(c, "gemma4") > 150',
            "must be written as a literal value",
        ),
        (
            f'any c in {_SPLIT} where llm_input_tokens("other", c) > 150',
            "splits and measures with ONE model",
        ),
        (
            f'any c in {_SPLIT} where llm_score("other", "p", c) > 0.5',
            "splits and measures with ONE model",
        ),
        (f'any c in {_SPLIT} where c.content ~ "ignore previous"', "a chunked body must"),
        (_count(">= 1", inner='c.content ~ "ignore previous"'), "a chunked body must"),
        (
            _count(">= 1", inner=f'{_SCORE} > 0.9 and llm_score("gemma4", "b", c) > 0.9'),
            "a chunked body must",
        ),
        (
            f'any c in {_SPLIT} where llm_score("gemma4", "p", conversation()) > 0.5',
            "a chunked body must",
        ),
        (
            f"any m in user_messages where any c in {_SPLIT} where {_SCORE} > 0.5",
            "cannot be nested inside another comprehension",
        ),
        (
            f"any m in user_messages where {_count('>= 1', inner=f'{_SCORE} > 0.9')}",
            "cannot be nested inside another comprehension",
        ),
        (
            'any c in llm_split_conversation("gemma4", conversation(), len(messages)) '
            f"where {_SCORE} > 0.5",
            "must be written as a literal value",
        ),
        (
            f'any c in llm_split_conversation("gemma4", "nope", 200) where {_SCORE} > 0.5',
            "accepts ['model_input'], got string",
        ),
        (
            f'any c in llm_split_conversation("gemma4", conversation(), 0) where {_SCORE} > 0.5',
            "must be at least 1",
        ),
        (f"len({_SPLIT}) == 2", "unsupported top-level condition"),
    ],
)
def test_a_chunk_shape_that_cannot_be_measured_is_rejected_at_load(
    config, condition, reason
) -> None:
    with pytest.raises(RuleValidationError) as error:
        compile_policy(
            f'rule "R"\ncondition: {condition}\n',
            runtime_config=config,
            model_inference_backend=v1_backend(config, FakeAdapter()),
        )

    assert reason in str(error.value)


@pytest.mark.parametrize(
    ("condition", "reason"),
    [
        (
            'any c in model("gemma4").split(messages, 200) where model("other").score(c, "p") > 0.5',
            "model split and score must use the same model",
        ),
        (
            'any c in model("gemma4").split(messages, 200) where '
            'model("gemma4").score(messages, c[0].content) > 0.5',
            "must use the bound chunk as its messages argument",
        ),
        (
            'model("gemma4").split(messages, 200) == model("gemma4").split(messages, 200)',
            "supported only as an any-comprehension collection",
        ),
    ],
)
def test_a_model_split_shape_that_cannot_be_measured_is_rejected_at_load(
    tmp_path: Path, condition, reason
) -> None:
    v2 = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "gemma4": model_entry(score=True, tokens=True),
                "other": model_entry(score=True, tokens=True),
            },
        },
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(
        v2, adapters={"fake": ModelAdapter()}, message_builders={"test": passthrough_builder}
    )

    with pytest.raises(RuleValidationError) as error:
        compile_policy(
            f'rule "R"\ncondition: {condition}\n',
            runtime_config=v2,
            model_inference_backend=backend,
        )

    assert reason in str(error.value)
