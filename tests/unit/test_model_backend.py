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

"""ModelInferenceBackend behavior seen by a scan: model calls, caching, spans, budgets, errors.

Rules run through ``DSLScanner``; the adapters in ``model_fakes`` stand in for the provider and
record exactly what each one was sent.
"""

from __future__ import annotations

import asyncio
import json
import math
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from quarry.cli import format_results
from quarry.config import QuarryConfig
from quarry.core.conversation import Conversation
from quarry.integrations.model_inference import ModelInferenceBackend, build_constitution_messages
from quarry.integrations.model_usage import model_call_budget
from quarry.safety.limits import SafetyLimits
from quarry.scanner import DSLScanner
from quarry.types.exceptions import ConfigurationError, LLMError, ModelError, RuleValidationError
from quarry.types.messages import Messages
from quarry.types.results import Score, Span
from tests.unit.model_fakes import (
    FakeAdapter,
    ModelAdapter,
    NoWaitLimiter,
    classification,
    model_entry,
    passthrough_builder,
    set_config_environment,
    v1_backend,
    v1_scanner,
    v2_scanner,
    write_v1_config,
)

_TOOL_CALL_CONVERSATION = {
    "messages": [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "a" * 110},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": '{"command":"rm -rf /tmp/example","reason":"cleanup"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "denied"},
        {"role": "user", "content": "b" * 110},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a command",
                "parameters": {"type": "object"},
            },
        }
    ],
}


def test_every_model_operation_runs_through_the_public_scanner(tmp_path: Path) -> None:
    """Direct, weighted, structured, event-bound, and concurrent split calls share one API."""

    rules = {
        "Score": 'model("gemma4").score(messages, text_ref("policy"), tools) > 0.9',
        "Weighted": '0.65 * model("head0").classify(messages) + 0.35 * model("head1").classify(messages) > 0.6',
        "Complete": 'model("gemma4").complete(messages).content ~ "blocked" and model("gemma4").complete(messages).refusal == null and model("gemma4").complete(messages).tool_calls[0].function.name == "shell"',
        "NoRefusal": 'model("gemma4").complete(messages).refusal ~ ".+"',
        "MissingTool": 'model("text_only").complete(messages).tool_calls[0].function.name == "shell"',
        "Tokens": 'model("gemma4").input_tokens(messages, tools) < 100000000000000000000',
        "Split": 'any chunk in model("gemma4").split(messages, 200) where 0.65 * model("gemma4").score(chunk, text_ref("policy"), tools) + 0.35 * model("gemma4").score(chunk, text_ref("policy"), tools) > 0.96',
        "TransitionalSplit": 'any chunk in model_split("gemma4", messages, 200) where 0.65 * model("gemma4").score(chunk, text_ref("policy"), tools) + 0.35 * model("gemma4").score(chunk, text_ref("policy"), tools) > 0.96',
        "Event": 'any call in tool_calls where model("head0").classify(message("user", call.arguments.command)) > 0.7',
        "SlicedConstructed": 'any call in tool_calls[:1] where model("gemma4").score(message("user", call.arguments.command).add_message("assistant", call.arguments.reason), text_ref("policy")) > 0.9',
        "AssistantTools": 'any msg in assistant_messages where model("head0").classify(message("user", msg.tool_calls[0].function.name)) > 0.7',
        "Slices": '(any msg in user_messages[:1] where model("head0").classify(message("user", msg.content)) > 0.7) or (any msg in user_messages[1:] where model("head0").classify(message("user", msg.content)) > 0.7)',
        "Constructed": 'model("head0").classify(add_message(message("system", text_ref("policy")), "user", "b")) > 0.7',
        "LegacyClassify": 'any msg in user_messages where model_score("legacy", msg.content, 0) > 0.7',
    }
    adapter = ModelAdapter()
    scanner = v2_scanner(
        tmp_path,
        rules,
        {
            "gemma4": model_entry(score=True, tokens=True),
            "head0": model_entry(classify=0),
            "head1": model_entry(classify=1),
            "text_only": model_entry(physical_model="text-model"),
            "legacy": model_entry(physical_model="legacy-model"),
        },
        adapter,
        texts={"policy": "Detect unsafe commands."},
        llm_rate_limits={"default": {"requests_per_interval": None, "max_concurrent": 8}},
    )

    result = scanner.scan_data(_TOOL_CALL_CONVERSATION, enable_explain=True)

    assert result.is_complete
    assert set(result.rule_matches) == set(rules) - {"NoRefusal", "MissingTool"}
    # One classification per distinct input and physical model, each under the rule deadline.
    assert sorted(model for model, _ in adapter.classification_calls) == [
        *["legacy-model"] * 2,
        *["physical-model"] * 6,
    ]
    assert all(timeout is not None and timeout > 0 for _, timeout in adapter.classification_calls)
    assert [
        {"role": "system", "content": "Detect unsafe commands."},
        {"role": "user", "content": "b"},
    ] in adapter.classification_messages
    assert adapter.maximum_score_concurrency > 1
    evidence = {item.rule_name: item for item in result.detailed_matches or []}
    assert evidence["Weighted"].model_results[0].expression_value == 0.66
    split_summary = evidence["Split"].model_results[0].score_summary
    assert split_summary["count"] > 1
    # Split and its transitional spelling share chunk scores; the whole-input score is one more.
    assert adapter.score_calls == 2 + split_summary["count"]
    assert adapter.completions == 2
    assert [
        {"role": "user", "content": "rm -rf /tmp/example"},
        {"role": "assistant", "content": "cleanup"},
    ] in adapter.score_messages
    assert [(item.source_path, item.matched) for item in evidence["Event"].model_results] == [
        ("tool_calls[0]", True)
    ]
    assert evidence["Tokens"].model_results[0].threshold == "100000000000000000000"
    assert format_results([result], "json")


def _v2_backend(tmp_path: Path, models: dict, adapter, **kwargs) -> ModelInferenceBackend:
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "rules": ["unused.rule"],
            "models": models,
            "llm_rate_limits": {"default": {"requests_per_interval": None, "max_concurrent": 8}},
        },
        tmp_path,
    )
    return ModelInferenceBackend(
        config, adapters={"fake": adapter}, message_builders={"test": passthrough_builder}, **kwargs
    )


_INPUT = Messages.from_openai([{"role": "user", "content": "input"}])


class _Completes:
    """A completion adapter answering every request with ``wire`` (v2) or ``text`` (v1)."""

    def __init__(self, wire=None, text=None) -> None:
        self.wire = wire
        self.text = text

    async def complete_message(self, _config, _messages, _tools, *, timeout=None):
        return self.wire

    async def complete(self, _config, _messages, _tools, *, timeout=None):
        return self.text


def _calls(count: int, arguments: str = "{}") -> list[dict]:
    return [
        {"id": f"c{index}", "type": "function", "function": {"name": "f", "arguments": arguments}}
        for index in range(count)
    ]


@pytest.mark.parametrize(
    ("wire", "limits", "reason"),
    [
        pytest.param(
            {"role": "assistant", "content": "x" * 100},
            {"max_message_size": 64},
            "Completion exceeds max_message_size 64",
            id="message-size",
        ),
        pytest.param(
            {"role": "assistant", "content": object()},
            {},
            "Completion is not valid JSON",
            id="not-json",
        ),
        pytest.param(
            {"role": "assistant", "tool_calls": _calls(2)},
            {"max_tool_calls": 1},
            "Completion has more than max_tool_calls 1",
            id="tool-calls",
        ),
        pytest.param(
            {"role": "assistant", "tool_calls": _calls(1, '{"a": 1}')},
            {"max_tool_argument_size": 4},
            "Completion tool call 0 arguments exceed max_tool_argument_size 4",
            id="argument-size",
        ),
        pytest.param(
            {"role": "assistant", "tool_calls": _calls(1)},
            {"max_timeline_items": 1},
            "Completion exceeds max_timeline_items 1",
            id="timeline-items",
        ),
        pytest.param(
            {"role": "assistant", "content": "hi"},
            {"max_total_size": 40},
            "Completion exceeds max_total_size 40",
            id="total-size",
        ),
        pytest.param(
            {"role": "assistant", "tool_calls": ["c0"]},
            {},
            r"Completion message is invalid: .*tool_calls\[0\] is not an object",
            id="string-tool-call",
        ),
        pytest.param(
            {"role": "user", "content": "hi"},
            {},
            "Completion message must have assistant role",
            id="user-role",
        ),
        pytest.param("hi", {}, "Completion response must be a message", id="bare-string"),
    ],
)
def test_a_completion_outside_the_limits_or_the_contract_is_refused(
    tmp_path: Path, wire, limits, reason
) -> None:
    backend = _v2_backend(tmp_path, {"gemma4": model_entry()}, _Completes(wire))

    with pytest.raises(LLMError, match=reason) as error:
        asyncio.run(backend.model_complete_message("gemma4", _INPUT, limits=SafetyLimits(**limits)))

    assert error.value.details == {"code": "invalid_output", "model": "gemma4"}


def _text_backend(tmp_path: Path, config_version: int, adapter) -> ModelInferenceBackend:
    config = QuarryConfig.from_mapping(
        {"config_version": config_version, "models": {"m": {"adapter": "fake", "model": "x"}}},
        tmp_path,
        require_rules=False,
    )
    return ModelInferenceBackend(config, adapters={"fake": adapter})


_TEXT_INPUT = Conversation([{"role": "user", "content": "input"}])


@pytest.mark.parametrize(
    ("config_version", "adapter", "error", "reason"),
    [
        pytest.param(
            2,
            _Completes({"role": "assistant", "content": None, "refusal": "no"}),
            LLMError,
            "Completion response must be text",
            id="v2-refusal",
        ),
        pytest.param(
            2,
            _Completes({"role": "assistant", "tool_calls": _calls(1)}),
            LLMError,
            "Completion response is a tool call",
            id="v2-tool-call",
        ),
        pytest.param(
            2,
            _Completes({"role": "assistant", "content": [{"type": "text", "text": "x"}]}),
            LLMError,
            "Completion response must be text",
            id="v2-content-parts",
        ),
        pytest.param(
            2, object(), ConfigurationError, "adapter 'fake' cannot complete", id="v2-no-completion"
        ),
        pytest.param(
            1, object(), ConfigurationError, "adapter 'fake' cannot complete", id="v1-no-completion"
        ),
        pytest.param(
            1, _Completes(text=7), LLMError, "Completion response must be text", id="v1-not-text"
        ),
    ],
)
def test_a_text_completion_that_is_not_plain_text_is_refused(
    tmp_path: Path, config_version, adapter, error, reason
) -> None:
    backend = _text_backend(tmp_path, config_version, adapter)

    with pytest.raises(error, match=reason):
        asyncio.run(backend.model_text_completion("m", _TEXT_INPUT))


@pytest.mark.parametrize(
    ("config_version", "adapter"),
    [
        (2, _Completes({"role": "assistant", "content": "safe", "refusal": None})),
        (1, _Completes(text="safe")),
    ],
    ids=["v2", "v1"],
)
def test_a_text_completion_returns_the_assistant_text(
    tmp_path: Path, config_version, adapter
) -> None:
    backend = _text_backend(tmp_path, config_version, adapter)

    assert asyncio.run(backend.model_text_completion("m", _TEXT_INPUT)) == "safe"


@pytest.mark.asyncio
async def test_repeated_model_operations_reach_the_adapter_once_per_scan_cache(tmp_path) -> None:
    adapter = ModelAdapter()
    backend = _v2_backend(
        tmp_path,
        {"guard": model_entry(score=True, tokens=True)},
        adapter,
        rate_limiter=NoWaitLimiter(),
    )

    def twice(operation):
        return asyncio.gather(operation(), operation())

    scores = await twice(
        lambda: backend.model_score_messages("guard", "policy", _INPUT, timeout=1.0)
    )
    completions = await twice(lambda: backend.model_complete_message("guard", _INPUT, timeout=1.0))
    token_counts = await twice(
        lambda: backend.model_input_tokens_messages("guard", _INPUT, timeout=1.0)
    )
    splits = await twice(lambda: backend.model_split_messages("guard", _INPUT, 100, timeout=1.0))
    huge_budget_split = await backend.model_split_messages(
        "guard", _INPUT, 1 << 20_000, timeout=1.0
    )

    assert scores == [0.99, 0.99]
    assert completions[0] == completions[1]
    assert token_counts == [5, 5]
    assert splits[0] == splits[1] == huge_budget_split == [_INPUT]
    assert (adapter.score_calls, adapter.completions, adapter.tokenizations) == (1, 1, 1)


class _TwoSpanAdapter(ModelAdapter):
    """Decode ``values`` (in order) as spans of the first message, scoring 0.9, 0.8, ..."""

    def __init__(self, *values: str) -> None:
        super().__init__()
        self.values = values

    async def decode_spans(self, config, raw_result, messages, threshold):
        self.decoded_spans.append((config, raw_result, messages, threshold))
        content = messages[0].content
        return [
            Span(0, content.index(value), content.index(value) + len(value), Score(0.9 - i / 10))
            for i, value in enumerate(self.values)
        ]


@pytest.mark.asyncio
async def test_classifier_aliases_share_one_response_and_decode_their_own_spans(
    tmp_path: Path,
) -> None:
    adapter = ModelAdapter()
    backend = _v2_backend(
        tmp_path,
        {
            "email": model_entry(category="EMAIL", classify=0, spans=4),
            "phone": model_entry(category="PHONE", classify=1, spans=5),
        },
        adapter,
    )
    messages = Messages.from_openai(
        [{"role": "user", "content": "Contact alice@example.com today"}]
    )

    scores = [
        await backend.model_classify_messages(alias, messages) for alias in ("email", "phone")
    ]
    email = await backend.model_span_messages("email", messages, 0.7)
    phone = await backend.model_span_messages("phone", messages, 0.9)

    assert adapter.classifications == 1
    assert scores == [0.8, 0.4]
    assert [
        (config.category, config.spans.class_index, raw.chunks[0].probabilities, threshold)
        for config, raw, _, threshold in adapter.decoded_spans
    ] == [("EMAIL", 4, [0.8, 0.4], 0.7), ("PHONE", 5, [0.8, 0.4], 0.9)]
    assert (email.output_type, email.category, phone.category) == (
        "span_collection",
        "EMAIL",
        "PHONE",
    )
    assert (email.source.kind, email.source.name) == ("model", "email")
    assert [messages.extract(item.value) for item in email.items] == ["alice@example.com"]
    assert "alice@example.com" not in str(email.to_dict())


@pytest.mark.asyncio
async def test_duplicate_span_coordinates_get_unique_ids_that_differ_per_run(tmp_path) -> None:
    class DuplicateSpanAdapter(ModelAdapter):
        async def decode_spans(self, _config, _raw_result, _messages, _threshold):
            span = Span(0, 0, 5, Score(0.9))
            return [span, span]

    backend = _v2_backend(
        tmp_path, {"email": model_entry(category="EMAIL", spans=4)}, DuplicateSpanAdapter()
    )
    messages = Messages.from_openai([{"role": "user", "content": "email"}])

    first = await backend.model_span_messages("email", messages, 0.7, timeout=1.0)
    second = await backend.model_span_messages("email", messages, 0.7, timeout=1.0)

    first_ids = {item.item_id for item in first.items}
    assert len(first_ids) == 2
    assert first.to_dict() == first.to_dict()
    assert second.source.run_id != first.source.run_id
    assert first_ids.isdisjoint(item.item_id for item in second.items)


@pytest.mark.asyncio
async def test_span_coordinates_outside_the_message_are_rejected(tmp_path: Path) -> None:
    class InvalidSpanAdapter(ModelAdapter):
        async def decode_spans(self, _config, _raw_result, _messages, _threshold):
            return [Span(0, 0, 100, Score(0.91))]

    backend = _v2_backend(
        tmp_path, {"email": model_entry(category="EMAIL", spans=4)}, InvalidSpanAdapter()
    )

    with pytest.raises(ValueError, match="outside message content"):
        await backend.model_span_messages(
            "email", Messages.from_openai([{"role": "user", "content": "short"}]), 0.7
        )


def _collect_scanner(tmp_path: Path, rule_text: str, adapter, **config) -> DSLScanner:
    rule_path = tmp_path / "collect.rule"
    rule_path.write_text(rule_text, encoding="utf-8")
    quarry_config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "rules": [str(rule_path)],
            "models": {
                "email": model_entry(category="EMAIL", classify=0, spans=4),
                "judge": model_entry(classify=0, physical_model="judge-model"),
            },
            "rule_timeout": 1.0,
            **config,
        },
        tmp_path,
    )
    backend = ModelInferenceBackend(quarry_config, adapters={"fake": adapter})
    return DSLScanner(quarry_config, model_inference_backend=backend)


def test_collect_filters_every_span_and_keeps_the_ones_it_selects(tmp_path: Path) -> None:
    rule = r"""rule "Email"
strings:
    $allowed = "^[A-Za-z0-9._%+-]+@example\.com$" regex
condition:
    model("email").classify(messages) >= 0.60
collect:
    [span for span in model("email").spans(messages, 0.70) if not $allowed in messages.extract(span)]
"""
    adapter = _TwoSpanAdapter("allowed@example.com", "outside@elsewhere.net")
    scanner = _collect_scanner(tmp_path, rule, adapter)

    result = scanner.scan_data(
        {
            "messages": [
                {"role": "user", "content": "Use allowed@example.com, not outside@elsewhere.net"}
            ]
        }
    )

    assert (result.rule_matches, result.is_complete) == (["Email"], True)
    assert (adapter.classifications, [entry[3] for entry in adapter.decoded_spans]) == (1, [0.7])
    output = result.outputs["Email"]
    assert output.category == "EMAIL"
    assert [
        (item.value.message_index, item.value.start, item.value.end, item.value.score)
        for item in output.items
    ] == [(0, 29, 50, Score(0.8))]
    assert output.to_dict()["type"] == "span_collection"
    assert "outside@elsewhere.net" not in str(output.to_dict())


def test_collect_runs_the_local_filter_before_the_cached_model_filter(tmp_path: Path) -> None:
    rule = """rule "Judged email"
condition: true
collect:
    [span for span in model("email").spans(messages, 0.70) if model("judge").classify(message("user", messages.extract(span))) >= 0.70 and span.score.value >= 0.80 and model("judge").classify(message("user", messages.extract(span))) >= 0.60]
"""

    class JudgedSpanAdapter(ModelAdapter):
        async def decode_spans(self, config, raw_result, messages, threshold):
            content = messages[0].content
            return [
                Span(0, content.index(value), content.index(value) + len(value), Score(score))
                for value, score in (("aaaa", 0.5), ("alice@example.com", 0.9))
            ]

    adapter = JudgedSpanAdapter()
    scanner = _collect_scanner(tmp_path, rule, adapter)

    result = scanner.scan_data(
        {"messages": [{"role": "user", "content": "aaaa alice@example.com"}]}
    )

    assert (result.rule_matches, result.is_complete) == (["Judged email"], True)
    # One classification for the span model, one for the judge on the span that passed locally.
    assert adapter.classifications == 2
    assert [item.value.start for item in result.outputs["Judged email"].items] == [5]


def test_collect_model_calls_spend_the_scan_model_call_budget(tmp_path: Path) -> None:
    rule = """rule "Budgeted spans"
condition: true
collect:
    [span for span in model("email").spans(messages, 0.70) if model("judge").classify(message("user", messages.extract(span))) >= 0.70]
"""
    adapter = _TwoSpanAdapter("first@example.com", "second@example.com")
    scanner = _collect_scanner(tmp_path, rule, adapter, max_model_calls=1)

    result = scanner.scan_data(
        {"messages": [{"role": "user", "content": "first@example.com second@example.com"}]}
    )

    assert (result.rule_matches, result.completion_state) == ([], "incomplete")
    assert "max_model_calls=1" in result.evaluation_errors["Budgeted spans"]


def test_a_false_condition_never_decodes_spans(tmp_path: Path) -> None:
    rule = """rule "Email"
condition:
    model("email").classify(messages) >= 0.90
collect:
    [span for span in model("email").spans(messages, 0.70) if true]
"""
    adapter = ModelAdapter()
    scanner = _collect_scanner(tmp_path, rule, adapter)

    result = scanner.scan_data({"messages": [{"role": "user", "content": "no match"}]})

    assert (result.rule_matches, result.outputs, result.is_complete) == ([], {}, True)
    assert (adapter.classifications, adapter.decoded_spans) == (1, [])


def test_spans_over_a_constructed_input_are_rejected_at_load(tmp_path: Path) -> None:
    rule = """rule "Synthetic"
condition: true
collect:
    [span for span in model("email").spans(message("user", "synthetic"), 0.70) if true]
"""

    with pytest.raises(RuleValidationError, match="view of the original conversation"):
        _collect_scanner(tmp_path, rule, ModelAdapter())


@pytest.mark.asyncio
async def test_one_aliases_exhausted_budget_or_failure_does_not_leak_into_another(tmp_path) -> None:
    models = {"exhausted": model_entry(classify=0), "available": model_entry(classify=0)}
    adapter = ModelAdapter()
    backend = _v2_backend(tmp_path, models, adapter)
    shared = Messages.from_openai([{"role": "user", "content": "shared"}])
    with model_call_budget(1):
        await backend.model_classify_messages(
            "exhausted", Messages.from_openai([{"role": "user", "content": "first"}])
        )
        exhausted, available = await asyncio.gather(
            backend.model_classify_messages("exhausted", shared),
            backend.model_classify_messages("available", shared),
            return_exceptions=True,
        )

    assert exhausted.details["code"] == "model_call_limit"
    assert available == 0.8
    assert adapter.classifications == 2

    class FailingAdapter(ModelAdapter):
        async def classify_messages(self, config, messages, tools, *, timeout=None):
            raise LLMError("physical-model rejected the request", {"code": "invalid_output"})

    failing = _v2_backend(tmp_path, models, FailingAdapter())
    failures = await asyncio.gather(
        failing.model_classify_messages("exhausted", shared),
        failing.model_classify_messages("available", shared),
        return_exceptions=True,
    )

    # Each failure names its own alias, never the shared physical model.
    assert [(failure.details, "physical-model" in str(failure)) for failure in failures] == [
        ({"code": "invalid_output", "model": "exhausted"}, False),
        ({"code": "invalid_output", "model": "available"}, False),
    ]


def test_the_model_call_limit_is_per_configured_model_and_resets_every_scan(tmp_path) -> None:
    rules = {
        "Budgeted": 'model("safety").complete(message("user", "a")).content == "blocked" and '
        'model("safety").complete(message("user", "b")).content == "blocked" and '
        'model("safety").complete(message("user", "b")).content == "blocked" and '
        'model("safety").complete(message("user", "c")).content == "blocked"',
        "Separate": 'model("classifier").complete(message("user", "a")).content == "blocked"',
    }
    # Different physical models send different requests, though both aliases use one adapter.
    models = {"safety": model_entry(), "classifier": model_entry(physical_model="classifier")}
    adapter = ModelAdapter()
    scanner = v2_scanner(tmp_path, rules, models, adapter, max_model_calls=2)
    data = {"messages": [{"role": "user", "content": "input"}]}

    results = [scanner.scan_data(data), scanner.scan_data(data)]

    assert [(r.completion_state, r.rule_matches) for r in results] == [
        ("incomplete", ["Separate"]),
        ("complete", ["Budgeted", "Separate"]),
    ]
    assert "max_model_calls=2" in results[0].evaluation_errors["Budgeted"]
    # The second scan reuses the first scan's three responses and pays for only the final one.
    assert adapter.completions == 4

    unlimited = v2_scanner(
        tmp_path,
        rules,
        models,
        ModelAdapter(),
        max_model_calls=2,
        enforce_limits=False,
    ).scan_data(data)

    assert (unlimited.completion_state, unlimited.rule_matches) == (
        "complete",
        ["Budgeted", "Separate"],
    )


class _WordTokenAdapter(ModelAdapter):
    """Count one token per word plus four per message."""

    async def tokenize(self, config, messages, tools, *, timeout=None):
        self.tokenizations += 1
        words = sum(len(str(message.get("content") or "").split()) for message in messages)
        return {config.tokens.count_field: words + 4 * len(messages)}


class _FramedTokenAdapter(ModelAdapter):
    """Count text and tool-call argument words, four per message and ten per request."""

    async def tokenize(self, config, messages, tools, *, timeout=None):
        self.tokenizations += 1
        words = sum(
            len(str(message.get("content") or "").split())
            + sum(
                len(call["function"]["arguments"].split())
                for call in message.get("tool_calls") or ()
            )
            for message in messages
        )
        return {config.tokens.count_field: words + 4 * len(messages) + 10}


def _split_backend(tmp_path, adapter) -> ModelInferenceBackend:
    return _v2_backend(
        tmp_path, {"gemma4": model_entry(tokens=True)}, adapter, rate_limiter=NoWaitLimiter()
    )


def _indexes(chunks) -> list[list[int]]:
    return [[message.message_index for message in chunk] for chunk in chunks]


def test_a_split_partitions_the_messages_and_repeats_only_the_context(tmp_path) -> None:
    backend = _split_backend(tmp_path, ModelAdapter())
    raw = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "a" * 120},
        *_TOOL_CALL_CONVERSATION["messages"][2:4],
        {"role": "user", "content": "b" * 120},
    ]
    messages = Messages.from_openai(raw)

    chunks = asyncio.run(
        backend.model_split_messages("gemma4", messages, 140, context=messages[:1])
    )

    assert all(
        asyncio.run(backend.model_input_tokens_messages("gemma4", chunk)) <= 140 for chunk in chunks
    )
    # The tool response shares its call's chunk only because both fit; nothing keeps them
    # together.
    assert _indexes(chunks) == [[0, 1, 2, 3], [0, 4]]
    assert [m for chunk in chunks for m in chunk.to_openai_messages()[1:]] == raw[1:]


def test_a_split_keeps_whole_messages_when_overlap_cannot_fit(tmp_path) -> None:
    backend = _split_backend(tmp_path, _WordTokenAdapter())
    messages = Messages.from_openai(
        [
            {"role": "user", "content": "bb a bb bb dddd"},
            {"role": "user", "content": "ccc"},
        ]
    )

    chunks = asyncio.run(backend.model_split_messages("gemma4", messages, 9, overlap=5))

    assert [[message.content for message in chunk] for chunk in chunks] == [
        ["bb a bb bb dddd"],
        ["ccc"],
    ]
    assert chunks[0][0] is messages[0]
    assert chunks[1][0] is messages[1]


def test_a_split_shortens_overlap_to_fit_the_next_chunk(tmp_path) -> None:
    backend = _split_backend(tmp_path, _WordTokenAdapter())
    messages = Messages.from_openai(
        [
            {"role": "user", "content": "one two three four five"},
            {"role": "user", "content": "six"},
        ]
    )

    chunks = asyncio.run(backend.model_split_messages("gemma4", messages, 10, overlap=6))

    assert [[message.content for message in chunk] for chunk in chunks] == [
        ["one two three four five"],
        [" five", "six"],
    ]


def test_a_split_keeps_a_message_it_cannot_cut_when_its_chunk_fits(tmp_path) -> None:
    backend = _split_backend(tmp_path, _FramedTokenAdapter())
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "read", "arguments": json.dumps({"path": " ".join(["dir"] * 19)})},
    }
    messages = Messages.from_openai(
        [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "read the file for me"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
        ]
    )

    # Counted alone the call is 34 tokens against a budget of 45 - 15 = 30, because it pays
    # the request framing again; with the context its chunk is exactly 39.
    chunks = asyncio.run(backend.model_split_messages("gemma4", messages, 45, context=messages[:1]))

    assert _indexes(chunks) == [[0, 1], [0, 2]]
    assert [asyncio.run(backend.model_input_tokens_messages("gemma4", c)) for c in chunks] == [
        24,
        39,
    ]


def test_a_split_cuts_a_long_message_into_ranges_that_do_not_shrink(tmp_path) -> None:
    adapter = _WordTokenAdapter()
    backend = _split_backend(tmp_path, adapter)
    text = " ".join(["word"] * 20_000)
    messages = Messages.from_openai([{"role": "user", "content": text}])

    chunks = asyncio.run(backend.model_split_messages("gemma4", messages, 200))

    # 112 ranges, not the 1,707 of repeatedly shrinking pieces.
    assert len(chunks) == 112
    spans = [chunk.span for chunk in chunks]
    assert spans[0].start == 0 and spans[-1].end == len(text)
    assert all(left.end == right.start for left, right in zip(spans, spans[1:], strict=False))
    assert all(len(chunk[0].content.split()) + 4 <= 200 for chunk in chunks)
    # Equal ranges send the same tokenizer request, which the shared cache counts once.
    assert adapter.tokenizations <= 2 + 112


_SPLIT_THEN_SCORE = (
    'any chunk in model("gemma4").split(messages, 140) where '
    'model("gemma4").score(chunk, text_ref("policy")) > 0.5'
)


class _LowScoreAdapter(ModelAdapter):
    async def score(self, _config, messages, *, timeout=None):
        self.score_messages.append(messages)
        return 0.1


def test_a_split_divides_a_message_longer_than_its_budget(tmp_path) -> None:
    # No chunk passes, so every chunk is scored and the scores show what the split made.
    adapter = _LowScoreAdapter()
    scanner = v2_scanner(
        tmp_path,
        {"R": _SPLIT_THEN_SCORE},
        {"gemma4": model_entry(score=True, tokens=True)},
        adapter,
        texts={"policy": "Detect unsafe commands."},
    )

    text = "a" * 75 + "b" * 75
    result = scanner.scan_data({"messages": [{"role": "user", "content": text}]})

    assert (result.rule_matches, result.completion_state) == ([], "complete")
    scored = ["".join(m["content"] for m in messages) for messages in adapter.score_messages]
    assert len(scored) > 1 and "".join(sorted(scored)) == text


def test_a_split_fails_the_scan_closed_when_one_character_cannot_fit(tmp_path) -> None:
    adapter = _FramedTokenAdapter()
    scanner = v2_scanner(
        tmp_path,
        {"R": _SPLIT_THEN_SCORE.replace("split(messages, 140)", "split(messages, 12)")},
        {"gemma4": model_entry(score=True, tokens=True)},
        adapter,
        texts={"policy": "Detect unsafe commands."},
    )

    result = scanner.scan_data({"messages": [{"role": "user", "content": "unsafe"}]})

    assert (result.rule_matches, result.completion_state) == ([], "incomplete")
    assert "A chunk exceeds max_input_tokens" in result.evaluation_errors["R"]
    assert adapter.score_calls == 0


class _SlowAdapter(ModelAdapter):
    async def classify_messages(self, config, messages, tools, *, timeout=None):
        await asyncio.sleep(1.0)
        return await super().classify_messages(config, messages, tools, timeout=timeout)


@pytest.mark.asyncio
async def test_an_empty_classification_input_is_refused_before_the_adapter(tmp_path) -> None:
    adapter = _SlowAdapter()
    backend = _v2_backend(tmp_path, {"head": model_entry(classify=0)}, adapter)

    with pytest.raises(ModelError, match="empty"):
        await backend.model_classify_messages("head", Messages())
    assert adapter.classifications == 0


@pytest.mark.asyncio
async def test_a_slow_classification_times_out_under_its_alias_name(tmp_path) -> None:
    backend = _v2_backend(tmp_path, {"head": model_entry(classify=0)}, _SlowAdapter())

    with pytest.raises(LLMError) as error:
        await backend.model_classify_messages("head", _INPUT, timeout=0.01)

    assert error.value.details == {"code": "timeout", "model": "head"}
    assert "physical-model" not in str(error.value)


class _EventLimiter:
    """Record when each limiter ticket is held, alongside the adapter attempts."""

    def __init__(self, events: list) -> None:
        self.events = events

    @asynccontextmanager
    async def limit(self, bucket):
        self.events.append(("acquire", bucket))
        try:
            yield 0.0
        finally:
            self.events.append(("release", bucket))


class _FailsOnceAdapter(ModelAdapter):
    def __init__(self, events: list) -> None:
        super().__init__()
        self.events = events
        self.timeouts = []

    async def classify_messages(self, config, messages, tools, *, timeout=None):
        self.events.append(("attempt", config.model))
        self.timeouts.append(timeout)
        if len(self.timeouts) == 1:
            raise LLMError("try again", {"code": "connection"})
        return await super().classify_messages(config, messages, tools, timeout=timeout)


@pytest.mark.asyncio
async def test_a_retry_takes_a_new_limiter_ticket_and_the_remaining_time(tmp_path) -> None:
    events: list = []
    adapter = _FailsOnceAdapter(events)
    backend = _v2_backend(
        tmp_path, {"head": model_entry(classify=0)}, adapter, rate_limiter=_EventLimiter(events)
    )

    score = await backend.model_classify_messages("head", _INPUT, timeout=1.0)

    assert score == 0.8
    # The backoff sleeps between tickets, never while holding one.
    assert events == [
        ("acquire", "physical-model"),
        ("attempt", "physical-model"),
        ("release", "physical-model"),
        ("acquire", "physical-model"),
        ("attempt", "physical-model"),
        ("release", "physical-model"),
    ]
    assert 0 < adapter.timeouts[1] < adapter.timeouts[0] <= 1.0


def test_a_retried_attempt_spends_the_model_call_budget(tmp_path) -> None:
    adapter = _FailsOnceAdapter([])
    scanner = v2_scanner(
        tmp_path,
        {"Retried": 'model("head").classify(messages) > 0.5'},
        {"head": model_entry(classify=0)},
        adapter,
        max_model_calls=1,
    )

    result = scanner.scan_data({"messages": [{"role": "user", "content": "input"}]})

    assert (result.rule_matches, result.completion_state) == ([], "incomplete")
    assert "max_model_calls=1" in result.evaluation_errors["Retried"]


# ── version 1: llm_score / llm_input_tokens / llm_complete / model_scan ───────────────────


@pytest.mark.parametrize(
    ("condition", "matched"),
    [
        # The adapter scores 0.8 and the calibrator adds 0.1.
        ('llm_score("gemma4", "policy", conversation()) > 0.85', True),
        ('llm_score("gemma4", "policy", conversation()) > 0.95', False),
        ('llm_score("gemma4", text_ref("unsafe_shell"), conversation()) > 0.85', True),
        # Six content characters.
        ('llm_input_tokens("gemma4", conversation()) > 3', True),
        ('llm_input_tokens("gemma4", conversation()) > 6', False),
        # The adapter completes "completed".
        ('llm_complete("gemma4", conversation()) == "completed"', True),
        ('llm_complete("gemma4", conversation()) == "NEVER"', False),
        ('llm_complete("gemma4", conversation()) != "NEVER"', True),
        ('llm_complete("gemma4", conversation()) ~ "compl"', True),
        ('llm_complete("gemma4", conversation()) ~ "^nope$"', False),
        # The adapter classifies every text as (0.1, 0.9, 0.2, 0.8).
        ('any m in user_messages where model_scan("gemma4", m.content, 0.5, 1)', True),
        ('any m in user_messages where model_scan("gemma4", m.content, 0.95, 1)', False),
    ],
)
def test_a_version_1_model_comparison_decides_the_rule(
    tmp_path, monkeypatch, condition, matched
) -> None:
    set_config_environment(monkeypatch)
    scanner = v1_scanner(tmp_path, condition, FakeAdapter())

    result = scanner.scan_data({"messages": [{"role": "user", "content": "abcdef"}]})

    assert (result.rule_matches, result.is_complete) == ((["R"] if matched else []), True)


def test_a_vllm_token_model_with_an_absolute_endpoint_loads(tmp_path) -> None:
    model = {**model_entry(), "tokens": _tokens("vllm", "https://tokens.example/tokenize")}
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"m": model}}, tmp_path, require_rules=False
    )

    backend = ModelInferenceBackend(config, adapters={"fake": ModelAdapter()})

    assert backend.model_ids == {"m"}


def test_a_missing_named_text_stops_scanner_startup(tmp_path, monkeypatch) -> None:
    set_config_environment(monkeypatch)

    with pytest.raises(RuntimeError, match="missing"):
        v1_scanner(
            tmp_path,
            'llm_score("gemma4", text_ref("missing"), conversation()) > 0.5',
            FakeAdapter(),
        )


def test_identical_verdict_tokens_are_rejected_at_config_load(tmp_path, monkeypatch) -> None:
    set_config_environment(monkeypatch)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "rules: [unused.rule]\nmodels:\n  gemma4:\n    adapter: fake\n    model: test/gemma4\n"
        "    score:\n      message_builder: constitution\n      boundary_seed: 7\n"
        '      tokens:\n        negative: "1"\n        positive: "1"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="differ"):
        QuarryConfig.from_yaml(config_path)


def test_a_constitution_scorer_without_verdict_tokens_stops_backend_startup(tmp_path) -> None:
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "rules": ["unused.rule"],
            "models": {
                "guard": {
                    "adapter": "fake",
                    "model": "test/guard",
                    "score": {"message_builder": "constitution", "boundary_seed": 7},
                }
            },
        },
        tmp_path,
    )

    with pytest.raises(ConfigurationError, match="requires positive and negative tokens"):
        ModelInferenceBackend(config, adapters={"fake": FakeAdapter()})


def _tokens(adapter: str, endpoint: str | None = None) -> dict:
    return {"adapter": adapter, "endpoint": endpoint, "extraction": {"count": "count"}}


_CALIBRATED = {
    "adapter": "fake",
    "model": "physical-model",
    "score": {"message_builder": "test", "boundary_seed": 7, "calibration": "cal.joblib"},
}


@pytest.mark.parametrize(
    ("model", "options", "reason"),
    [
        pytest.param(
            model_entry(),
            {"models_to_load": ["ghost"]},
            "Inference models not configured: ghost. Available models: m",
            id="unknown-model-to-load",
        ),
        pytest.param(
            {"adapter": "nope", "model": "x"}, {}, "unregistered adapter 'nope'", id="adapter"
        ),
        pytest.param(
            {**model_entry(), "tokens": _tokens("nope")},
            {},
            "unregistered tokens adapter 'nope'",
            id="tokens-adapter",
        ),
        pytest.param(
            {**model_entry(), "tokens": _tokens("litellm")},
            {},
            "LiteLLM completions are not an input-token API",
            id="litellm-tokens",
        ),
        pytest.param(
            {**model_entry(), "tokens": _tokens("plain")},
            {"adapters": {"fake": ModelAdapter(), "plain": object()}},
            "tokens adapter 'plain' cannot tokenize",
            id="tokens-adapter-without-tokenize",
        ),
        pytest.param(
            {**model_entry(), "tokens": _tokens("vllm")},
            {},
            "vLLM token counting requires tokens.endpoint",
            id="vllm-without-endpoint",
        ),
        pytest.param(
            {**model_entry(), "tokens": _tokens("vllm", "/tokenize")},
            {},
            "A relative tokens.endpoint requires the model endpoint",
            id="vllm-relative-endpoint",
        ),
        pytest.param(
            model_entry(score=True),
            {"adapters": {"fake": object()}},
            "adapter 'fake' cannot score",
            id="adapter-without-score",
        ),
        pytest.param(
            {**model_entry(), "score": {"message_builder": "nope", "boundary_seed": 7}},
            {},
            "unregistered message builder 'nope'",
            id="message-builder",
        ),
        pytest.param(
            _CALIBRATED,
            {},
            "configures calibration 'cal.joblib' but no calibrator loader was provided",
            id="calibration-without-loader",
        ),
        pytest.param(
            _CALIBRATED,
            {"calibrator_loader": lambda _path: 0.5},
            "calibration loader must return a callable",
            id="calibration-not-callable",
        ),
    ],
)
def test_a_model_the_backend_cannot_serve_stops_backend_startup(
    tmp_path, model, options, reason
) -> None:
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"m": model}}, tmp_path, require_rules=False
    )
    options = {"adapters": {"fake": ModelAdapter()}, **options}

    with pytest.raises(ConfigurationError, match=reason):
        ModelInferenceBackend(config, message_builders={"test": passthrough_builder}, **options)


def _v1_backend_with(tmp_path, monkeypatch, calibrator_loader) -> ModelInferenceBackend:
    set_config_environment(monkeypatch)
    rule = tmp_path / "unused.rule"
    rule.write_text('rule "unused"\ncondition: true\n', encoding="utf-8")
    return ModelInferenceBackend(
        write_v1_config(tmp_path, [rule]),
        models_to_load=["gemma4"],
        adapters={"fake": FakeAdapter()},
        message_builders={"test_builder": build_constitution_messages},
        calibrator_loader=calibrator_loader,
    )


def test_a_calibrated_score_outside_the_unit_interval_is_an_error(tmp_path, monkeypatch) -> None:
    backend = _v1_backend_with(tmp_path, monkeypatch, lambda _path: lambda _score: math.nan)

    with pytest.raises(LLMError, match=r"Calibrated score is outside \[0, 1\]"):
        asyncio.run(
            backend.model_text_score(
                "gemma4", "policy", Conversation([{"role": "user", "content": "input"}])
            )
        )


def test_an_unreadable_calibration_artifact_stops_backend_startup(tmp_path, monkeypatch) -> None:
    def fail_to_load(_path: str):
        raise OSError("unreadable artifact")

    with pytest.raises(ConfigurationError, match="could not load calibration"):
        _v1_backend_with(tmp_path, monkeypatch, fail_to_load)


# ── configured adapter classification (model_score / model_scan) ─────


class _ClassificationAdapter:
    """Configured provider fake answering each classification with ``response``."""

    def __init__(self, response) -> None:
        self.response = response
        self.calls = []

    async def classify_messages(self, config, messages, _tools, *, timeout=None):
        self.calls.append((config.model, messages[0]["content"]))
        return self.response


def _classification_scan(tmp_path: Path, condition: str, response):
    adapter = _ClassificationAdapter(response)
    scanner = v2_scanner(
        tmp_path,
        {"R": f"any m in user_messages where {condition}"},
        {"m": model_entry(classify=0, physical_model="configured-model")},
        adapter,
    )
    result = scanner.scan_data({"messages": [{"role": "user", "content": "hello"}]})
    return result, adapter


_TWO_CHUNKS = SimpleNamespace(
    chunks=[
        SimpleNamespace(probabilities=[0.1, 0.2, 0.9]),
        SimpleNamespace(probabilities=[0.3, 0.4, 0.5]),
    ]
)


@pytest.mark.parametrize(
    ("condition", "response", "matched"),
    [
        # The configured index is read as its maximum across chunks.
        ('model_score("m", m.content, 2) >= 0.8', _TWO_CHUNKS, True),
        ('model_score("m", m.content, 0) >= 0.35', _TWO_CHUNKS, False),
        ('model_scan("m", m.content, 0.5, 0)', classification(0.7, 0.65), True),
        ('model_scan("m", m.content, 0.8, 0)', classification(0.7, 0.65), False),
        ('model_score("m", m.content, 1) >= 0.6', classification(0.7, 0.65), True),
    ],
)
def test_a_configured_classification_reads_its_class_index_and_threshold(
    tmp_path, condition, response, matched
) -> None:
    result, adapter = _classification_scan(tmp_path, condition, response)

    assert (result.rule_matches, result.is_complete) == ((["R"] if matched else []), True)
    assert adapter.calls == [("configured-model", "hello")]


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(SimpleNamespace(chunks=[]), id="no-chunks"),
        pytest.param(classification(), id="empty-probabilities"),
        pytest.param(classification(0.1), id="index-out-of-range"),
        pytest.param(classification(0.1, math.nan), id="nan"),
        pytest.param(classification(0.1, -0.5), id="negative"),
        pytest.param(classification(0.1, 1.7), id="above-one"),
        pytest.param(classification(0.1, True), id="bool"),
        pytest.param(classification(0.1, "0.9"), id="not-a-number"),
    ],
)
def test_a_malformed_classification_fails_the_scan_closed(tmp_path, response) -> None:
    result, _ = _classification_scan(tmp_path, 'model_score("m", m.content, 1) >= 0.5', response)

    assert (result.rule_matches, result.completion_state) == ([], "incomplete")
    assert "R" in result.evaluation_errors


class _RefusingAdapter(FakeAdapter):
    """Refuse every score request, as a guard model does with content it will not judge."""

    async def score(self, config, messages, *, timeout=None):
        raise LLMError(
            "Model refused the request", {"code": "model_refused", "reason": "provider_stop"}
        )


def test_a_refused_version_1_score_fails_the_scan_closed(tmp_path, monkeypatch) -> None:
    set_config_environment(monkeypatch)
    rule = tmp_path / "r.rule"
    rule.write_text(
        'rule "R"\ncondition: llm_score("gemma4", "policy", conversation()) < 0.5\n',
        encoding="utf-8",
    )
    config = write_v1_config(tmp_path, [rule])
    scanner = DSLScanner(config, model_inference_backend=v1_backend(config, _RefusingAdapter()))

    result = scanner.scan_data({"messages": [{"role": "user", "content": "abcdef"}]})

    assert (result.rule_matches, result.completion_state) == ([], "incomplete")
    assert "provider_stop" in result.evaluation_errors["R"]


class _RefusalAdapter(ModelAdapter):
    """``blocked`` has every request rejected; ``malformed`` scores no verdict and says no in
    free text; ``answers`` scores 0.2 and completes "ok"."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[tuple[str, str]] = []

    async def score(self, config, messages, *, timeout=None):
        self.requests.append((config.model, "score"))
        if config.model == "answers":
            return 0.2
        reason = "invalid_output" if config.model == "malformed" else "request_blocked"
        raise LLMError("Model refused the request", {"code": "model_refused", "reason": reason})

    async def complete_message(self, config, messages, tools, *, timeout=None):
        self.requests.append((config.model, "complete"))
        if config.model == "answers":
            return {"role": "assistant", "content": "ok"}
        if config.model == "malformed":
            return {"role": "assistant", "content": None, "refusal": "I can't help with that."}
        raise LLMError(
            "Provider rejected the request", {"code": "model_refused", "reason": "request_blocked"}
        )


def test_a_rule_handles_a_refused_score_or_completion_without_failing(tmp_path) -> None:
    """refused() and if_refused() read a call's own request; an unguarded refusal fails closed."""

    adapter = _RefusalAdapter()
    score = 'model("{}").score(messages, "policy")'
    rules = {
        "RefusedOrHigh": f"if_refused({score.format('blocked')}, 1) > 0.5",
        "AnsweredAndHigh": f"if_refused({score.format('blocked')}, 0) > 0.5",
        "Malformed": f"refused({score.format('malformed')})",
        "AnsweredScore": f"not refused({score.format('answers')})",
        "AnsweredFallbackUnused": f"if_refused({score.format('answers')}, 1) < 0.5",
        "CompleteRefused": 'refused(model("blocked").complete(messages))',
        "CompleteText": 'if_refused(model("blocked").complete(messages).content, "x") == "x"',
        "CompleteAnswered": 'if_refused(model("answers").complete(messages).content, "x") == "ok"',
        # Saying no in free text is the model's answer, readable from `.refusal`.
        "FreeTextNo": 'not refused(model("malformed").complete(messages)) and '
        'if_refused(model("malformed").complete(messages).refusal, "") ~ "can.t"',
        "InLoop": "any m in user_messages where "
        'refused(model("blocked").score(message("user", m.content), "policy"))',
        "Unguarded": f"{score.format('blocked')} > 0.5",
    }
    models = {
        name: model_entry(score=True, physical_model=name)
        for name in ("blocked", "malformed", "answers")
    }
    scanner = v2_scanner(tmp_path, rules, models, adapter)

    result = scanner.scan_data(
        {"messages": [{"role": "user", "content": "hello"}]}, enable_explain=True
    )

    assert sorted(result.rule_matches) == sorted(set(rules) - {"AnsweredAndHigh", "Unguarded"})
    assert (result.completion_state, list(result.evaluation_errors)) == (
        "incomplete",
        ["Unguarded"],
    )
    assert "(request_blocked)" in result.evaluation_errors["Unguarded"]
    # One request per distinct input, however many rules and guards read it: the loop's
    # one-message input is the whole conversation here, so it shares the top-level request.
    assert sorted(adapter.requests) == sorted(
        (name, operation) for name in models for operation in ("score", "complete")
    )
    evidence = {item.rule_name: item for item in result.detailed_matches or []}
    assert evidence["Malformed"].model_results[0].refusals == (
        ("malformed", "score", "invalid_output"),
    )
    assert evidence["CompleteRefused"].model_results[0].refusals == (
        ("blocked", "complete", "request_blocked"),
    )
    # The refused call stays in the evidence alongside the fallback the rule compared.
    [blocked] = evidence["RefusedOrHigh"].model_results
    assert (blocked.models, blocked.expression_value) == (("blocked",), 1)


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        ("refused(1)", "must read a model"),
        ('refused(model("m").classify(messages))', "must read a model"),
        (
            'if_refused(model("m").score(messages, "p"), model("m").score(messages, "q")) > 0.5',
            "literal or a parameter",
        ),
        ('if_refused(model("m").score(messages, "p"), "high") > 0.5', "does not match"),
        (
            'refused(model("m").score(messages, "p")) or model("m").score(messages, "p") > 0.5',
            "also read outside one",
        ),
        (
            'model("m").score(messages, "p") as s and (s > 0.9 or if_refused(s, 1) > 0.8)',
            "also read outside one",
        ),
        (
            'refused(model("m").score(messages, "p")) and '
            'model("m").score(messages, "p", tools) > 0.5',
            "also read outside one",
        ),
        (
            'any chunk in model("m").split(messages, 100) where '
            'refused(model("m").score(chunk, "p"))',
            "only if_refused",
        ),
    ],
    ids=[
        "not-a-call",
        "classify",
        "model-fallback",
        "mismatched-fallback",
        "mixed",
        "mixed-through-alias",
        "mixed-with-tools",
        "split-refused",
    ],
)
def test_a_refusal_guard_outside_its_contract_is_rejected_at_load(
    tmp_path, condition, message
) -> None:
    models = {"m": model_entry(classify=0, score=True, tokens=True)}

    with pytest.raises(RuleValidationError, match=message):
        v2_scanner(tmp_path, {"R": condition}, models, ModelAdapter())


def test_a_split_predicate_scores_a_refused_chunk_with_its_fallback(tmp_path) -> None:
    class HalfRefuses(ModelAdapter):
        async def score(self, config, messages, *, timeout=None):
            self.score_messages.append(messages)
            if len(self.score_messages) == 1:
                raise LLMError("refused", {"code": "model_refused", "reason": "provider_stop"})
            return 0.2

    rule = (
        'any chunk in model("m").split(messages, 3) where '
        'if_refused(model("m").score(chunk, "policy"), 1) > 0.5'
    )
    scanner = v2_scanner(
        tmp_path, {"Split": rule}, {"m": model_entry(score=True, tokens=True)}, HalfRefuses()
    )

    result = scanner.scan_data(
        {"messages": [{"role": "user", "content": "abc"}, {"role": "user", "content": "def"}]},
        enable_explain=True,
    )

    assert (result.rule_matches, result.is_complete) == (["Split"], True)
    [summary] = [item.model_results[0].score_summary for item in result.detailed_matches or []]
    assert summary["refused_count"] == 1
    assert sorted(chunk["refused"] for chunk in summary["chunks"]) == [False, True]


@pytest.mark.parametrize(
    ("details", "refused", "reason"),
    [
        ({"code": "guardrail_blocked"}, True, "refusal_field"),
        (
            {"code": "model_refused", "reason": "I will not judge: model text"},
            True,
            "refusal_field",
        ),
        ({"code": "invalid_output"}, False, None),
    ],
    ids=["legacy-code", "unknown-reason", "not-a-refusal"],
)
def test_an_adapter_reports_a_refusal_only_with_a_refusal_code_and_a_fixed_reason(
    tmp_path, details, refused, reason
) -> None:
    class Reports(ModelAdapter):
        async def score(self, config, messages, *, timeout=None):
            raise LLMError("adapter failure: provider text", details)

    scanner = v2_scanner(
        tmp_path,
        {
            "R": 'refused(model("m").score(messages, "p"))',
            "Unguarded": 'model("m").score(messages, "p") > 0.5',
        },
        {"m": model_entry(score=True)},
        Reports(),
    )

    result = scanner.scan_data(
        {"messages": [{"role": "user", "content": "hello"}]}, enable_explain=True
    )

    assert result.rule_matches == (["R"] if refused else [])
    if refused:
        [read] = result.detailed_matches[0].model_results
        assert read.refusals == (("m", "score", reason),)
        # An unguarded read reports the refusal with a fixed message, not the adapter's.
        assert result.evaluation_errors == {"Unguarded": f"Model refused the request ({reason})"}
        assert "provider text" not in repr(result) and "model text" not in repr(result)
    else:
        assert "R" in result.evaluation_errors


def test_a_classifier_cannot_be_refused(tmp_path) -> None:
    class RefusingClassifier(ModelAdapter):
        async def classify_messages(self, config, messages, tools, *, timeout=None):
            raise LLMError("refused", {"code": "model_refused", "reason": "provider_stop"})

    rule = 'refused(model("m").score(messages, "p") + model("m").classify(messages))'
    scanner = v2_scanner(
        tmp_path, {"R": rule}, {"m": model_entry(classify=0, score=True)}, RefusingClassifier()
    )

    result = scanner.scan_data({"messages": [{"role": "user", "content": "hello"}]})

    assert (result.rule_matches, result.completion_state) == ([], "incomplete")
    assert "cannot be refused" in result.evaluation_errors["R"]


_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["safe", "unsafe"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"category": {"type": "string"}},
                "required": ["category"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["label", "confidence", "findings"],
    "additionalProperties": False,
}
_VERDICT = {"label": "unsafe", "confidence": 0.9, "findings": [{"category": "pii"}]}


class _Verdicts(ModelAdapter):
    """Complete each conversation with the reply its last message names."""

    def __init__(self, replies: dict[str, dict]) -> None:
        super().__init__()
        self.replies = replies
        self.schemas: list[str | None] = []

    async def complete_message(self, config, messages, tools, *, timeout=None):
        self.completions += 1
        self.schemas.append(config.complete and config.complete.response_schema)
        return self.replies[messages[-1]["content"]]


def _verdict_scanner(tmp_path: Path, rules: dict[str, str], adapter) -> DSLScanner:
    entry = {**model_entry(), "complete": {"response_schema": _VERDICT_SCHEMA}}
    return v2_scanner(tmp_path, rules, {"verdict": entry, "plain": model_entry()}, adapter)


def test_a_rule_reads_a_schema_completion_by_field_and_handles_a_mismatch(tmp_path) -> None:
    """Fields of a matching reply are typed values; a mismatching reply is a refusal."""

    completion = 'model("verdict").complete(messages)'
    adapter = _Verdicts(
        {
            "check": {"role": "assistant", "content": json.dumps(_VERDICT)},
            "garbled": {"role": "assistant", "content": "unsafe, probably"},
        }
    )
    rules = {
        "Label": f'{completion}.json.label == "unsafe"',
        "Confident": f"{completion}.json.confidence > 0.8",
        "Finding": f'{completion}.json.findings[0].category == "pii"',
        "Raw": f'{completion}.content ~ "\\"label\\""',
        "Guarded": f'if_refused({completion}.json.label, "unsafe") == "unsafe"',
        "Refused": f"refused({completion})",
    }
    scanner = _verdict_scanner(tmp_path, rules, adapter)

    matching = scanner.scan_data({"messages": [{"role": "user", "content": "check"}]})
    garbled = scanner.scan_data(
        {"messages": [{"role": "user", "content": "garbled"}]}, enable_explain=True
    )

    assert sorted(matching.rule_matches) == ["Confident", "Finding", "Guarded", "Label", "Raw"]
    assert matching.is_complete
    assert sorted(garbled.rule_matches) == ["Guarded", "Refused"]
    assert set(garbled.evaluation_errors) == {"Label", "Confident", "Finding", "Raw"}
    assert all(
        error.endswith("does not match the response schema (invalid_output)")
        for error in garbled.evaluation_errors.values()
    )
    assert "probably" not in repr(garbled)
    # One request per conversation, sent with the alias's schema.
    assert adapter.completions == 2
    assert [json.loads(schema) for schema in adapter.schemas] == [_VERDICT_SCHEMA] * 2
    evidence = {item.rule_name: item for item in garbled.detailed_matches or []}
    assert evidence["Refused"].model_results[0].refusals == (
        ("verdict", "complete", "invalid_output"),
    )


@pytest.mark.parametrize(
    "reply",
    [
        {"role": "assistant", "content": json.dumps({**_VERDICT, "confidence": "high"})},
        {"role": "assistant", "content": json.dumps({"label": "unsafe", "confidence": 0.9})},
        {"role": "assistant", "tool_calls": _calls(1)},
        {"role": "assistant", "content": None, "refusal": "I can't help with that."},
    ],
    ids=["wrong-type", "missing-field", "tool-call-only", "refusal-field"],
)
def test_a_reply_outside_the_response_schema_is_refused_after_one_request(tmp_path, reply) -> None:
    adapter = _Verdicts({"check": reply})
    scanner = _verdict_scanner(
        tmp_path, {"R": 'refused(model("verdict").complete(messages))'}, adapter
    )

    result = scanner.scan_data({"messages": [{"role": "user", "content": "check"}]})

    assert (result.rule_matches, adapter.completions) == (["R"], 1)


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        ('model("verdict").complete(messages).json.lable == "unsafe"', "has no field 'lable'"),
        (
            'model("verdict").complete(messages).json.label > 0.5',
            "ordered model comparisons require numeric operands",
        ),
        ('model("plain").complete(messages).json.label == "unsafe"', "requires a model alias"),
        (
            'any m in assistant_messages where m.json.label == "unsafe"',
            "Unknown field 'json'",
        ),
    ],
    ids=["typo", "ordered-string", "no-schema", "transcript-message"],
)
def test_a_json_read_outside_the_response_schema_is_rejected_at_load(
    tmp_path, condition, message
) -> None:
    with pytest.raises(RuleValidationError, match=message):
        _verdict_scanner(tmp_path, {"R": condition}, _Verdicts({}))


def test_a_schema_reply_over_the_completion_limits_is_an_error_not_a_refusal(tmp_path) -> None:
    entry = {**model_entry(), "complete": {"response_schema": _VERDICT_SCHEMA}}
    backend = _v2_backend(
        tmp_path, {"verdict": entry}, _Completes({"role": "assistant", "content": "x" * 1000})
    )

    with pytest.raises(LLMError) as error:
        asyncio.run(
            backend.model_complete("verdict", _INPUT, limits=SafetyLimits(max_message_size=64))
        )

    # A breached limit stays an error no refusal guard can catch.
    assert error.value.details == {"code": "invalid_output", "model": "verdict"}


def test_a_schema_integer_keeps_its_exact_value(tmp_path) -> None:
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
        "additionalProperties": False,
    }
    adapter = _Verdicts({"check": {"role": "assistant", "content": '{"n": 18446744073709551617}'}})
    read = 'model("verdict").complete(messages).json.n'
    scanner = v2_scanner(
        tmp_path,
        {"Exact": f"{read} == 18446744073709551617", "Rounded": f"{read} == 18446744073709551616"},
        {"verdict": {**model_entry(), "complete": {"response_schema": schema}}},
        adapter,
    )

    result = scanner.scan_data({"messages": [{"role": "user", "content": "check"}]})

    assert (result.rule_matches, result.is_complete) == (["Exact"], True)
