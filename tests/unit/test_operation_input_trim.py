# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Operation-aware trim selects one bounded input without dispatching discarded input."""

import asyncio
import json

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.integrations.resources import ResourcePreparationError
from quarry.types.results import Score, Span
from tests.unit.model_fakes import ModelAdapter, NoWaitLimiter, model_entry, passthrough_builder


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("operation", ["score", "classify"])
@pytest.mark.parametrize("direction,expected", [("start", "first"), ("end", "third")])
def test_trim_selects_one_oldest_or_newest_whole_message(
    tmp_path, restored, operation, direction, expected
):
    adapter = ModelAdapter()
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"reader": model_entry(score=True, classify=0, tokens=True)},
        },
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(
        config,
        adapters={"fake": adapter},
        message_builders={"test": passthrough_builder},
        rate_limiter=NoWaitLimiter(),
    )
    selected = f'trim(messages, 6, from="{direction}", cut_keep="none")'
    arguments = selected + (', "synthetic policy"' if operation == "score" else "")
    policy = compile_policy(
        f'rule "Read" condition: model("reader").{operation}({arguments}) > 0.5',
        runtime_config=config,
        model_inference_backend=backend,
    )
    if restored:
        raw = policy.to_bytes()
        policy = load_compiled_policy(raw, runtime_config=config, model_inference_backend=backend)
        assert policy.to_bytes() == raw
    result = policy.scan(
        {"messages": [{"role": "user", "content": text} for text in ("first", "second", "third")]}
    )
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]
    payloads = adapter.score_messages if operation == "score" else adapter.classification_messages
    assert payloads == [[{"role": "user", "content": expected}]]
    assert adapter.score_calls + adapter.classifications == 1
    assert adapter.tokenizations > 0


class TrimAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()
        self.complete_inputs = []
        self.token_inputs = []

    async def complete_message(self, config, messages, tools, *, timeout=None):
        self.completions += 1
        self.complete_inputs.append(messages)
        return {"role": "assistant", "content": '{"ok":true}'}

    async def decode_spans(self, config, raw_result, messages, threshold):
        return [Span(0, 0, 1, Score(0.9))]

    async def tokenize(self, config, messages, tools, *, timeout=None):
        self.token_inputs.append((messages, tools))
        return await super().tokenize(config, messages, tools, timeout=timeout)


def trim_policy(
    tmp_path,
    operation,
    selected,
    *,
    limit=None,
    restored=False,
    adapter=None,
    builder=None,
    policy_text='"policy"',
    offered_tools=False,
):
    adapter = adapter or TrimAdapter()
    entry = model_entry(score=True, classify=0, tokens=True, spans=0, category="SYNTHETIC")
    entry["complete"] = {
        "response_schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
    }
    if limit is not None:
        entry[operation]["max_input_tokens"] = limit
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"reader": entry}}, tmp_path, require_rules=False
    )
    backend = ModelInferenceBackend(
        config,
        adapters={"fake": adapter},
        message_builders={"test": builder or passthrough_builder},
        rate_limiter=NoWaitLimiter(),
    )
    tools = ", tools" if offered_tools else ""
    if operation == "score":
        body = f'condition: model("reader").score({selected}, {policy_text}{tools}) > 0.5'
    elif operation == "classify":
        body = f'condition: model("reader").classify({selected}{tools}) > 0.5'
    elif operation == "complete":
        body = f'condition: model("reader").complete({selected}{tools}).json.ok'
    else:
        body = (
            f'condition: true collect: [s for s in model("reader").spans({selected}, 0.5) if true]'
        )
    compiled = compile_policy(
        'rule "Read" ' + body, runtime_config=config, model_inference_backend=backend
    )
    if restored:
        raw = compiled.to_bytes()
        compiled = load_compiled_policy(raw, runtime_config=config, model_inference_backend=backend)
        assert compiled.to_bytes() == raw
    return compiled, adapter, backend


DATA = {"messages": [{"role": "user", "content": text} for text in ("first", "second", "third")]}


@pytest.mark.parametrize("operation", ["complete", "spans"])
@pytest.mark.parametrize("direction,expected,index", [("start", "first", 0), ("end", "third", 2)])
@pytest.mark.parametrize("restored", [False, True])
def test_trim_completion_and_span_inputs_preserve_native_coordinates(
    tmp_path, operation, direction, expected, index, restored
):
    policy, adapter, _ = trim_policy(
        tmp_path,
        operation,
        f'trim(messages, 6, from="{direction}", cut_keep="none")',
        restored=restored,
    )
    result = policy.scan(DATA, explain=True)
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]
    payloads = (
        adapter.complete_inputs if operation == "complete" else adapter.classification_messages
    )
    assert payloads == [[{"role": "user", "content": expected}]]
    if operation == "spans":
        span = result.outputs["Read"].items[0].value
        assert (span.message_index, span.start, span.end) == (index, 0, 1)


@pytest.mark.parametrize("operation", ["score", "classify", "complete", "spans"])
@pytest.mark.parametrize("restored", [False, True])
def test_implicit_trim_uses_each_consuming_operations_limit(tmp_path, operation, restored):
    policy, adapter, _ = trim_policy(
        tmp_path,
        operation,
        'trim(messages, from="start", cut_keep="none")',
        limit=6,
        restored=restored,
    )
    result = policy.scan(DATA)
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]
    payloads = (
        adapter.score_messages
        if operation == "score"
        else adapter.complete_inputs
        if operation == "complete"
        else adapter.classification_messages
    )
    assert payloads == [[{"role": "user", "content": "first"}]]


def test_implicit_score_measures_rendered_policy_and_framing(tmp_path):
    def framed(policy, messages, tools, boundary, config):
        return [{"role": "system", "content": policy}, *messages]

    policy, adapter, _ = trim_policy(
        tmp_path,
        "score",
        'trim(messages, from="start", cut_keep="none")',
        limit=13,
        policy_text='"criteria"',
        builder=framed,
    )
    result = policy.scan(DATA)
    assert result.is_complete, result.evaluation_errors
    assert adapter.score_messages == [
        [{"role": "system", "content": "criteria"}, {"role": "user", "content": "first"}]
    ]
    assert all(
        messages[0] == {"role": "system", "content": "criteria"} and tools == []
        for messages, tools in adapter.token_inputs
    )


@pytest.mark.parametrize(
    "context,expected",
    [("messages[-1:]", ["third", "first"]), ('message("system", "prefix")', ["prefix", "first"])],
)
def test_trim_moves_native_or_constructed_context_without_changing_the_body(
    tmp_path, context, expected
):
    policy, adapter, _ = trim_policy(
        tmp_path,
        "classify",
        f'trim(messages, 11, context={context}, from="start", cut_keep="none")',
    )
    result = policy.scan(DATA)
    assert result.is_complete, result.evaluation_errors
    assert [m["content"] for m in adapter.classification_messages[0]] == expected


@pytest.mark.parametrize("direction,expected", [("start", ["first", "s"]), ("end", ["s", "third"])])
def test_partial_cut_keeps_only_a_fitting_text_prefix(tmp_path, direction, expected):
    policy, adapter, _ = trim_policy(tmp_path, "classify", f'trim(messages, 6, from="{direction}")')
    result = policy.scan(DATA, explain=True)
    assert result.is_complete, result.evaluation_errors
    assert [m["content"] for m in adapter.classification_messages[0]] == expected
    arguments = dict(result.detailed_matches[0].model_results[0].arguments)
    assert arguments["trim_omitted_messages"] == 1
    assert arguments["trim_omitted_characters"] == 5


def test_written_text_token_arithmetic_sets_the_message_budget(tmp_path):
    policy, adapter, _ = trim_policy(
        tmp_path,
        "classify",
        'trim(messages, 9 - model("reader").input_tokens(message("user", "pad")), from="start", cut_keep="none")',
    )
    result = policy.scan(DATA)
    assert result.is_complete, result.evaluation_errors
    assert adapter.classification_messages == [[{"role": "user", "content": "first"}]]


@pytest.mark.parametrize("operation", ["score", "classify", "complete", "spans"])
def test_empty_trim_input_dispatches_no_tokenizer_or_model(tmp_path, operation):
    policy, adapter, _ = trim_policy(tmp_path, operation, "trim(messages[:0], 6)")
    result = policy.scan(DATA)
    assert result.is_complete, result.evaluation_errors
    assert (
        adapter.tokenizations
        == adapter.score_calls
        == adapter.classifications
        == adapter.completions
        == 0
    )


def test_missing_computed_policy_short_circuits_trim_before_measurement(tmp_path):
    policy, adapter, _ = trim_policy(
        tmp_path, "score", "trim(messages, 6)", policy_text="messages[50].content"
    )
    result = policy.scan(DATA)
    assert result.is_complete and result.rule_matches == []
    assert adapter.tokenizations == adapter.score_calls == 0


def test_native_input_is_not_limited_to_the_constructed_message_count(tmp_path):
    policy, adapter, _ = trim_policy(tmp_path, "classify", "trim(messages, 1000)")
    result = policy.scan({"messages": [{"role": "user", "content": "x"} for _ in range(300)]})
    assert result.is_complete, result.evaluation_errors
    assert len(adapter.classification_messages[0]) == 300
    assert adapter.tokenizations == adapter.classifications == 1


@pytest.mark.parametrize("mutation", ["alias", "operation", "budget", "size", "context"])
def test_prepared_trim_cannot_forge_its_owner_budget_or_operand_kind(tmp_path, mutation):
    policy, adapter, backend = trim_policy(tmp_path, "classify", "trim(messages, 6)")
    document = json.loads(policy.to_bytes())
    pending = [document]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if value.get("kind") == "trim":
                selected = value
                break
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    if mutation in {"alias", "operation"}:
        parts = selected["value"].split("\x1f")
        parts[0 if mutation == "alias" else 1] = "other" if mutation == "alias" else "score"
        selected["value"] = "\x1f".join(parts)
    elif mutation == "budget":
        selected["children"].append({"kind": "literal", "value": "forged", "children": []})
    elif mutation == "context":
        selected["children"][2] = {"kind": "literal", "value": "not messages", "children": []}
    else:
        selected["children"][1] = {
            "kind": "model_call",
            "value": "reader\x1fcomplete",
            "children": [selected["children"][0]],
        }
    with pytest.raises(ValueError, match="trim"):
        load_compiled_policy(
            json.dumps(document).encode(),
            runtime_config=backend.config,
            model_inference_backend=backend,
        )
    assert adapter.tokenizations == adapter.classifications == adapter.completions == 0


@pytest.mark.parametrize("operation", ["score", "classify", "complete", "spans"])
def test_implicit_trim_requires_a_configured_operation_limit(tmp_path, operation):
    with pytest.raises(ResourcePreparationError, match="max_input_tokens"):
        trim_policy(tmp_path, operation, "trim(messages)")


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "6"])
@pytest.mark.parametrize("operation", ["score", "classify", "complete", "spans"])
def test_operation_limits_are_closed_positive_integer_configuration(tmp_path, operation, limit):
    with pytest.raises(ValueError, match="max_input_tokens"):
        trim_policy(tmp_path, operation, "trim(messages)", limit=limit)


def test_implicit_trim_confirms_nonlinear_tool_and_framing_costs(tmp_path):
    class Nonlinear(TrimAdapter):
        async def tokenize(self, config, messages, tools, *, timeout=None):
            self.tokenizations += 1
            self.token_inputs.append((messages, tools))
            return {
                "count": sum(len(m.get("content", "")) for m in messages)
                + 7 * len(tools)
                + len(messages) ** 2
            }

    policy, adapter, _ = trim_policy(
        tmp_path,
        "classify",
        'trim(messages, from="start", cut_keep="none")',
        limit=13,
        offered_tools=True,
        adapter=Nonlinear(),
    )
    result = policy.scan(
        {
            **DATA,
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ],
        }
    )
    assert result.is_complete, result.evaluation_errors
    assert adapter.classification_messages == [[{"role": "user", "content": "first"}]]
    assert all(len(tools) == 1 for _, tools in adapter.token_inputs)
    assert sum(len(m["content"]) for m in adapter.classification_messages[0]) + 7 + 1 <= 13


def test_logical_read_collection_does_not_repeat_trim_or_provider_work(tmp_path):
    policy, adapter, _ = trim_policy(tmp_path, "classify", 'trim(messages, 6, cut_keep="none")')
    first = policy.scan(DATA)
    before = (adapter.classifications, adapter.tokenizations)
    second = policy.scan(DATA, logical_reads=True)
    assert first.is_complete and second.is_complete
    assert (adapter.classifications, adapter.tokenizations) == before
    assert {row["operation"] for row in second.logical_reads["rows"]} == {
        "classify",
        "input_tokens",
    }
    assert second.logical_reads["aggregates"]["totals"]["coverage"]["characters"]["complete"]


@pytest.mark.parametrize(
    "setting", ['from="middle"', 'cut_keep="end"', "unknown=true", 'from="end", from="start"']
)
def test_trim_rejects_unknown_or_repeated_selection_settings(tmp_path, setting):
    with pytest.raises(Exception, match="trim"):
        trim_policy(tmp_path, "classify", f"trim(messages, 6, {setting})")


async def test_cancellation_of_trim_wait_does_not_dispatch_the_consuming_model(tmp_path):
    entered = asyncio.Event()
    released = asyncio.Event()

    class Waiting(TrimAdapter):
        async def tokenize(self, config, messages, tools, *, timeout=None):
            self.tokenizations += 1
            entered.set()
            try:
                await asyncio.Future()
            finally:
                released.set()

    policy, adapter, _ = trim_policy(tmp_path, "classify", "trim(messages, 6)", adapter=Waiting())
    task = asyncio.create_task(policy.scan_async(DATA))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(released.wait(), timeout=2)
    assert adapter.classifications == 0


@pytest.mark.parametrize(
    "selection",
    [
        'trim(messages[:0], model("reader").input_tokens(message("user", "pad")))',
        'trim(messages[50], model("reader").input_tokens(message("user", "pad")))',
    ],
)
def test_missing_trim_operands_skip_dynamic_token_budget_work(tmp_path, selection):
    policy, adapter, _ = trim_policy(tmp_path, "classify", selection)
    result = policy.scan(DATA)
    assert result.is_complete and result.rule_matches == []
    assert adapter.classifications == adapter.tokenizations == 0


def test_a_genuinely_empty_trim_read_is_measured_zero_without_a_request(tmp_path):
    policy, adapter, _ = trim_policy(tmp_path, "classify", "trim(messages[:0], 6)")
    result = policy.scan(DATA, logical_reads=True)
    assert result.is_complete and result.rule_matches == []
    row = result.logical_reads["rows"][0]
    assert row["outcome"] == "not_sent"
    assert row["coverage"]["characters"]["value"] == 0
    assert adapter.classifications == adapter.tokenizations == 0


@pytest.mark.parametrize("context", ["system_messages", 'message("system", messages[50].content)'])
def test_an_empty_optional_context_does_not_remove_the_input(tmp_path, context):
    policy, adapter, _ = trim_policy(
        tmp_path, "classify", f'trim(messages, 6, context={context}, cut_keep="none")'
    )
    result = policy.scan(DATA)
    assert result.is_complete and result.rule_matches == ["Read"]
    assert adapter.classification_messages == [[{"role": "user", "content": "third"}]]


@pytest.mark.parametrize("direction", ["start", "end"])
def test_span_over_moved_context_keeps_its_original_source_coordinate(tmp_path, direction):
    policy, _, _ = trim_policy(
        tmp_path,
        "spans",
        f'trim(messages, 11, context=messages[-1:], from="{direction}", cut_keep="none")',
    )
    result = policy.scan(DATA)
    assert result.is_complete, result.evaluation_errors
    span = result.outputs["Read"].items[0].value
    assert (span.message_index, span.start, span.end) == (2, 0, 1)


def test_span_over_a_partial_trim_keeps_the_boundary_message_coordinates(tmp_path):
    policy, _, _ = trim_policy(tmp_path, "spans", 'trim(messages, 6, from="end")')
    result = policy.scan(DATA)
    assert result.is_complete, result.evaluation_errors
    span = result.outputs["Read"].items[0].value
    assert (span.message_index, span.start, span.end) == (1, 0, 1)


async def test_trim_wait_respects_the_inspection_deadline(tmp_path):
    released = asyncio.Event()

    class Waiting(TrimAdapter):
        async def tokenize(self, config, messages, tools, *, timeout=None):
            self.tokenizations += 1
            try:
                await asyncio.Future()
            finally:
                released.set()

    policy, adapter, _ = trim_policy(tmp_path, "classify", "trim(messages, 6)", adapter=Waiting())
    result = await asyncio.wait_for(policy.scan_async(DATA, deadline_seconds=0.02), timeout=2)
    assert not result.is_complete
    assert result.timed_out_rules == ["Read"]
    await asyncio.wait_for(released.wait(), timeout=2)
    assert adapter.classifications == 0


def test_trim_integer_arithmetic_is_bounded_before_large_intermediate_products(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("quarry.integrations.model_expression.MAX_TRIM_ARITHMETIC_WORK", 64)
    policy, adapter, _ = trim_policy(
        tmp_path,
        "classify",
        "trim(messages, 1000000000 * 1000000000 - 1000000000 * 1000000000 + 6)",
    )
    result = policy.scan(DATA, explain=True)
    assert not result.is_complete
    assert adapter.classifications == adapter.tokenizations == 0
    assert "work bound" in result.evaluation_errors["Read"]


def test_invalid_token_budget_count_does_not_enter_explanation_json(tmp_path):
    class Invalid(TrimAdapter):
        async def tokenize(self, config, messages, tools, *, timeout=None):
            self.tokenizations += 1
            return {"count": True}

    policy, adapter, _ = trim_policy(
        tmp_path,
        "classify",
        'trim(messages, model("reader").input_tokens(message("user", "pad")))',
        adapter=Invalid(),
    )
    result = policy.scan(DATA, explain=True)
    assert not result.is_complete and adapter.classifications == 0
    assert adapter.tokenizations == 1
    assert "no valid input-token count" in result.evaluation_errors["Read"]


@pytest.mark.parametrize("selection", ["trim(messages, 1000000001)", "trim(messages)"])
def test_large_positive_input_limits_preserve_split_size_units(tmp_path, selection):
    policy, adapter, _ = trim_policy(tmp_path, "classify", selection, limit=1000000001)
    result = policy.scan(DATA)
    assert result.is_complete and result.rule_matches == ["Read"]
    assert adapter.classification_messages == [DATA["messages"]]
