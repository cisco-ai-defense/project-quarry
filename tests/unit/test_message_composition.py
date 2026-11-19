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

"""Typed conditional, selection, and composition behavior for message model inputs."""

from __future__ import annotations

import asyncio

import orjson
import pytest

import quarry.integrations.model_expression as model_expression_module
import quarry.scanner as scanner_module
import quarry.types.messages as messages_module
from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.dsl.parser import DSLParser
from quarry.engine.atoms import (
    MAX_CONSTRUCTED_MESSAGES,
    MAX_MODEL_EXPRESSION_DEPTH,
    MAX_MODEL_EXPRESSION_FANOUT,
    Cost,
    ModelExpression,
    validate_constructed_message_graph,
    validate_model_expression,
)
from quarry.engine.compiler import compile_rule
from quarry.engine.events import build_evaluation_input
from quarry.engine.executor import evaluate_async
from quarry.integrations.model_expression import evaluate_model_expression
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.plugin.visualization import build_rule_visualization
from quarry.safety.limits import SafetyLimits
from quarry.scanner import DSLScanner
from quarry.types.context import ScanContext
from quarry.types.dsl import (
    BinaryOp,
    ConditionalValue,
    FilterComprehension,
    Identifier,
    Literal,
    ModelCall,
    walk_condition_nodes,
)
from quarry.types.dsl.render import render_condition
from quarry.types.messages import AssistantMessage, Messages, ToolMessage
from quarry.types.results import Score, Span
from tests.conftest import rule_matches
from tests.unit.model_fakes import (
    FakeAdapter,
    ModelAdapter,
    model_entry,
    set_config_environment,
    v1_scanner,
    v2_scanner,
)

_CONVERSATION = {
    "messages": [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "middle"},
        {"role": "user", "content": "last"},
    ]
}

_TOOL_CONVERSATION = {
    "messages": [
        {"role": "user", "content": "run"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "done"},
    ]
}

_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a synthetic record.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _compile_expression(expression: str) -> list[str]:
    rule = DSLParser().parse(f'rule "R"\ncondition: {expression}\n')
    result = compile_rule(rule)
    return [error.message for error in result.diagnostics.errors]


def test_conditional_values_are_right_associative_and_looser_than_or() -> None:
    rule = DSLParser().parse(
        'rule "R"\ncondition: messages if true or false else user_messages '
        "if false else assistant_messages\n"
    )

    assert isinstance(rule.condition, ConditionalValue)
    assert isinstance(rule.condition.predicate, BinaryOp)
    assert rule.condition.predicate.operator == "or"
    assert isinstance(rule.condition.fallback, ConditionalValue)


def test_else_remains_available_as_a_contextual_variable_name() -> None:
    rule = DSLParser().parse('rule "R"\nvariables:\n    else = true\ncondition:\n    else\n')

    assert compile_rule(rule).ok


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("true if true else false", True),
        ("(1 if false else 2) > 1", True),
        ("true if false else false", False),
        (
            "any item in messages where "
            '(item.content if item.role == "user" else "") contains "first"',
            True,
        ),
    ],
)
def test_conditional_values_work_without_a_model_call(condition, expected) -> None:
    assert rule_matches(f'rule "R"\ncondition: {condition}\n', _CONVERSATION) is expected


@pytest.mark.parametrize(
    ("model_input", "expected"),
    [
        (
            '[m for m in messages if m.role == "user"]',
            [_CONVERSATION["messages"][0], _CONVERSATION["messages"][2]],
        ),
        (
            "messages[-1:] + messages[:1]",
            [_CONVERSATION["messages"][2], _CONVERSATION["messages"][0]],
        ),
        (
            "(messages[-2:] + messages[:1]).user_messages",
            [_CONVERSATION["messages"][2], _CONVERSATION["messages"][0]],
        ),
        (
            "messages[:1] + messages[:1]",
            [_CONVERSATION["messages"][0], _CONVERSATION["messages"][0]],
        ),
        (
            'messages[:1] + message("assistant", "constructed")',
            [_CONVERSATION["messages"][0], {"role": "assistant", "content": "constructed"}],
        ),
        (
            'message("system", "context") + messages[-1:]',
            [{"role": "system", "content": "context"}, _CONVERSATION["messages"][2]],
        ),
        (
            '[item for item in message("user", "kept") + messages[:2] if item.role == "user"]',
            [{"role": "user", "content": "kept"}, _CONVERSATION["messages"][0]],
        ),
        (
            'message("assistant", "") + messages[:1] if true else messages[99:]',
            [{"role": "assistant", "content": ""}, _CONVERSATION["messages"][0]],
        ),
        (
            'messages[:1] if messages[-1].role == "user" else messages[99:]',
            [_CONVERSATION["messages"][0]],
        ),
    ],
)
def test_model_receives_selected_messages_in_deterministic_written_order(
    tmp_path, model_input, expected
) -> None:
    adapter = ModelAdapter()
    scanner = v2_scanner(
        tmp_path,
        {"R": f'model("judge").classify({model_input}) > 0.5'},
        {"judge": model_entry(classify=0)},
        adapter,
    )

    result = scanner.scan_data(_CONVERSATION)

    assert (result.rule_matches, result.evaluation_errors) == (["R"], None)
    assert adapter.classification_messages == [expected]


def test_message_combination_normally_reuses_identity_and_preserves_values() -> None:
    source = Messages.from_openai(_CONVERSATION["messages"])

    combined = Messages.combine((source[-1:], source[:1]))

    assert combined.to_openai_messages() == [
        _CONVERSATION["messages"][2],
        _CONVERSATION["messages"][0],
    ]
    assert combined[0] is source[2]
    assert combined[1] is source[0]
    assert [message.message_index for message in combined] == [2, 0]
    assert combined.user_messages.to_openai_messages() == combined.to_openai_messages()
    with pytest.raises(AttributeError, match="immutable"):
        combined._ordinals = ()  # type: ignore[misc]


def test_message_combination_documents_relinked_tool_identity_exception() -> None:
    documentation = Messages.combine.__doc__ or ""

    assert "normally reuses immutable message objects" in documentation
    assert "must not rely on object identity" in documentation
    assert "Value, content, provenance" in documentation


def test_python_composition_preserves_multimodal_source_and_tool_graphs() -> None:
    raw = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "unit-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "unit-1", "content": "ready"},
    ]
    source = Messages.from_openai(raw)
    built = source[:0].add_message("developer", "").add_message("user", "review this")

    combined = Messages.combine((source[:2], built, source[2:]))
    combined.assert_self_contained()

    assert combined.to_openai_messages() == [
        raw[0],
        raw[1],
        {"role": "developer", "content": ""},
        {"role": "user", "content": "review this"},
        raw[2],
    ]
    assert [message.message_index for message in combined] == [0, 1, 3, 4, 2]
    assert combined[0].content_parts[1].type == "image_url"
    assistant = combined[1]
    response = combined[4]
    assert isinstance(assistant, AssistantMessage)
    assert isinstance(response, ToolMessage)
    assert assistant.tool_calls[0].response is response
    assert response.call is assistant.tool_calls[0]
    assert source[1].tool_calls[0].response is source[2]
    with pytest.raises(ValueError, match="inserted text"):
        combined.source_ranges(4, 0, 1)
    copied = source[:0]._add_synthetic_message(
        "user",
        source[0].content,
        identity=(73, 0),
        provenance=source[0]._content_provenance(),
    )
    assert [item.to_dict() for item in copied.source_ranges(3, 0, 4)] == [
        {"message_index": 0, "part_index": 0, "start": 0, "end": 4}
    ]
    built_span = combined._bind_span(Span(3, 0, 6, Score(0.8)))
    assert built_span.to_dict() == {
        "message_index": 4,
        "start": 0,
        "end": 6,
        "score": {"value": 0.8},
        "source_unmapped": True,
    }
    assert combined.extract(built_span) == "review"


def test_repeated_and_independently_constructed_units_have_stable_private_identity() -> None:
    source = Messages.from_openai(_CONVERSATION["messages"])
    once = source[:0]._add_synthetic_message(
        "user",
        "same",
        identity=(41, 0),
    )
    again = source[:0]._add_synthetic_message(
        "user",
        "same",
        identity=(41, 0),
    )

    repeated = Messages.combine((once, once))
    independent = Messages.combine((once, again))

    expected = [
        {"role": "user", "content": "same"},
        {"role": "user", "content": "same"},
    ]
    assert repeated.to_openai_messages() == expected
    assert independent.to_openai_messages() == expected
    assert repeated[0] is repeated[1]
    assert independent[0] is independent[1]
    conflicting = source[:0]._add_synthetic_message(
        "user",
        "different",
        identity=(41, 0),
    )
    with pytest.raises(ValueError, match="different content variants"):
        Messages.combine((once, conflicting))


@pytest.mark.parametrize("message_count", [1_000, 4_000])
def test_mixed_root_grafting_has_exact_linear_metered_work(message_count) -> None:
    source = Messages.from_openai(
        [{"role": "user", "content": str(index)} for index in range(message_count)]
    )
    built = Messages().add_message("assistant", "review")
    metrics = messages_module._CombineMetrics()

    combined = Messages.combine((source[:1], built), _metrics=metrics)

    assert combined.to_openai_messages() == [
        {"role": "user", "content": "0"},
        {"role": "assistant", "content": "review"},
    ]
    assert metrics.canonical_message_visits == 2 * message_count + 1
    assert metrics.selection_visits == 4
    assert metrics.selected_message_visits == 2
    assert metrics.provenance_visits == 1
    assert metrics.segment_visits == 2
    assert metrics.message_clone_visits == 2
    assert metrics.work_units == 2 * message_count + 12


def test_mixed_root_grafting_checks_deadlines_before_aggregate_allocation(
    monkeypatch,
) -> None:
    source = Messages.from_openai(
        [{"role": "user", "content": str(index)} for index in range(4_000)]
    )
    built = Messages().add_message("assistant", "review")
    metrics = messages_module._CombineMetrics()

    def expire() -> None:
        if metrics.canonical_message_visits >= 512:
            raise RuntimeError("mixed composition deadline")

    monkeypatch.setattr(messages_module, "_raise_if_combine_expired", expire)
    with pytest.raises(RuntimeError, match="mixed composition deadline"):
        Messages.combine((source[:1], built), _metrics=metrics)
    assert 512 <= metrics.canonical_message_visits <= 768


@pytest.mark.asyncio
async def test_constructed_message_batch_is_cancellable_at_bounded_intervals(
    monkeypatch,
) -> None:
    built = 'message("user", "0")' + "".join(
        f'.add_message("assistant", "{index}")' for index in range(1, 17)
    )
    parsed = DSLParser().parse(f'rule "R"\ncondition: model("judge").classify({built}) > 0.5\n')
    expression = next(iter(compile_rule(parsed).artifact.atom_specs.values())).model_expression
    assert expression is not None

    def find_kind(node, kind):
        if node.kind == kind:
            return node
        for child in node.children:
            found = find_kind(child, kind)
            if found is not None:
                return found
        return None

    constructed = find_kind(expression, "add_message")
    assert constructed is not None

    async def cancel(_delay) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(model_expression_module.asyncio, "sleep", cancel)
    with pytest.raises(asyncio.CancelledError):
        await evaluate_model_expression(
            constructed,
            ScanContext(messages=_CONVERSATION["messages"], tools=[]),
        )


@pytest.mark.asyncio
async def test_mixed_root_grafting_is_cancellable_during_source_indexing(
    monkeypatch,
) -> None:
    source = Messages.from_openai(
        [{"role": "user", "content": str(index)} for index in range(8_000)]
    )
    built = Messages().add_message("assistant", "review")
    metrics = messages_module._CombineMetrics()

    async def cancel() -> None:
        if metrics.canonical_message_visits >= 4_096:
            raise asyncio.CancelledError

    monkeypatch.setattr(messages_module, "_yield_combine_cancellation", cancel)
    with pytest.raises(asyncio.CancelledError):
        await Messages._combine_async((source[:1], built), _metrics=metrics)
    assert metrics.canonical_message_visits == 4_096


@pytest.mark.parametrize(
    ("model_input", "message"),
    [
        ('messages if true else "text"', "branches of a conditional value"),
        ("messages if 1 else messages", "predicate must be Boolean"),
        ("messages + 1", "message combination requires message values"),
        (
            "[other for item in messages if true]",
            "must keep its loop variable",
        ),
        (
            '[item for item in messages if model("judge").classify(item) > 0.5]',
            "predicate cannot call a model",
        ),
        (
            "(messages[:1] + messages[-1:]).unknown_messages",
            "has no field 'unknown_messages'",
        ),
        (
            "[item for item in messages if item.message_index == 0]",
            "message_index is provenance metadata",
        ),
    ],
)
def test_impossible_composed_values_fail_during_compilation(model_input, message) -> None:
    errors = _compile_expression(f'model("judge").classify({model_input}) > 0.5')

    assert errors
    assert message in " ".join(errors)


def test_mismatched_pure_conditional_fails_during_compilation() -> None:
    errors = _compile_expression('(1 if true else "text") > 0')

    assert "both branches of a conditional value" in " ".join(errors)


def test_composed_expression_depth_is_bounded_at_compile_time() -> None:
    rule = DSLParser().parse('rule "R"\ncondition: model("judge").classify(messages) > 0.5\n')
    assert isinstance(rule.condition, BinaryOp)
    assert isinstance(rule.condition.left, ModelCall)
    value = rule.condition.left.args[0]
    for _ in range(MAX_MODEL_EXPRESSION_DEPTH + 1):
        value = ConditionalValue(value, Literal(True), Identifier("messages"))
    rule.condition.left.args[0] = value

    result = compile_rule(rule)

    assert not result.ok
    assert "nested levels" in " ".join(error.message for error in result.diagnostics.errors)


def test_combine_fanout_and_expression_cycles_are_rejected() -> None:
    source = Messages.from_openai(_CONVERSATION["messages"])
    with pytest.raises(ValueError, match="cannot combine more than"):
        Messages.combine(tuple(source[:1] for _ in range(MAX_MODEL_EXPRESSION_FANOUT + 1)))

    expression_text = " + ".join("messages[:0]" for _ in range(MAX_MODEL_EXPRESSION_FANOUT + 1))
    errors = _compile_expression(f'model("judge").classify({expression_text}) > 0.5')
    assert "message combination exceeds 256 operands" in " ".join(errors)

    expression = ModelExpression("not", None, ())
    object.__setattr__(expression, "children", (expression,))
    with pytest.raises(ValueError, match="cycle"):
        validate_model_expression(expression)


def test_constructed_message_occurrences_and_aggregate_text_are_bounded(
    monkeypatch,
) -> None:
    built = ModelExpression(
        "message",
        None,
        (
            ModelExpression("literal", "user"),
            ModelExpression("literal", "x"),
        ),
    )
    accepted = ModelExpression(
        "concat",
        None,
        (built,) * MAX_CONSTRUCTED_MESSAGES,
    )
    validate_constructed_message_graph(accepted)

    with pytest.raises(ValueError, match="exceeds 256 constructed messages"):
        validate_constructed_message_graph(
            accepted,
            additional_roots=(built,),
        )
    oversized = ModelExpression(
        "message",
        None,
        (
            ModelExpression("literal", "user"),
            ModelExpression("literal", "x" * 1_000_001),
        ),
    )
    with pytest.raises(ValueError, match="constructed input text exceeds"):
        validate_constructed_message_graph(oversized)

    source = Messages.from_openai(_CONVERSATION["messages"])
    specifications = tuple(
        ("user", str(index), (index, 0), None) for index in range(MAX_CONSTRUCTED_MESSAGES + 1)
    )
    with pytest.raises(ValueError, match="exceeds 256 messages"):
        source[:0]._add_synthetic_messages(specifications)
    private_text = "private-prefix-" + "x" * 1_000_000
    with pytest.raises(ValueError) as oversized_error:
        Messages().add_message("user", private_text)
    assert "private-prefix" not in str(oversized_error.value)

    def expire() -> None:
        raise RuntimeError("constructor deadline")

    monkeypatch.setattr(messages_module, "_raise_if_combine_expired", expire)
    with pytest.raises(RuntimeError, match="constructor deadline"):
        Messages().add_message("user", "x" * 5_000)


def test_prepared_policy_round_trip_preserves_composed_model_input(tmp_path) -> None:
    source = """rule "Prepared composition"
variables:
    selected = [item for item in messages if item.role == "user"]
    input = messages[:1] + message("assistant", "review") + selected[-1:] if messages[-1].role == "user" else selected
condition:
    model("judge").classify(input) > 0.5
"""
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"judge": model_entry(classify=0)},
        },
        tmp_path,
        require_rules=False,
    )
    adapter = ModelAdapter()
    backend = ModelInferenceBackend(config, adapters={"fake": adapter})
    compiled = compile_policy(
        source,
        runtime_config=config,
        model_inference_backend=backend,
    )

    encoded = compiled.to_bytes()
    loaded = load_compiled_policy(
        encoded,
        runtime_config=config,
        model_inference_backend=backend,
    )

    assert loaded.to_bytes() == encoded
    assert loaded.scan(_CONVERSATION).rule_matches == ["Prepared composition"]
    assert adapter.classification_messages[-1] == [
        _CONVERSATION["messages"][0],
        {"role": "assistant", "content": "review"},
        _CONVERSATION["messages"][2],
    ]


def test_mixed_composition_order_changes_model_cache_identity() -> None:
    def atom_id(model_input: str) -> str:
        parsed = DSLParser().parse(
            f'rule "R"\ncondition: model("judge").classify({model_input}) > 0.5\n'
        )
        artifact = compile_rule(parsed).artifact
        return next(iter(artifact.atom_specs))

    assert atom_id('messages[:1] + message("assistant", "review")') != atom_id(
        'message("assistant", "review") + messages[:1]'
    )


def test_model_spans_report_constructed_and_copied_source_provenance(tmp_path) -> None:
    class MixedSpanAdapter(ModelAdapter):
        async def decode_spans(self, config, raw_result, messages, threshold):
            self.decoded_spans.append((config, raw_result, messages, threshold))
            target = len(messages) - 1
            return [Span(target, 1, 4, Score(0.9))]

    adapter = MixedSpanAdapter()
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "finder": model_entry(category="TEXT", classify=0, spans=0),
            },
        },
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(config, adapters={"fake": adapter})

    literal = compile_policy(
        'rule "Literal"\ncondition: true\ncollect:\n'
        '    [span for span in model("finder").spans('
        'messages[:1] + message("assistant", "note"), 0.5) if true]\n',
        runtime_config=config,
        model_inference_backend=backend,
    )
    loaded = load_compiled_policy(
        literal.to_bytes(),
        runtime_config=config,
        model_inference_backend=backend,
    )
    literal_result = loaded.scan(_CONVERSATION)
    literal_span = literal_result.outputs["Literal"].items[0].value
    assert literal_span.to_dict() == {
        "message_index": 3,
        "start": 1,
        "end": 4,
        "score": {"value": 0.9},
        "source_unmapped": True,
    }

    copied = compile_policy(
        'rule "Copied"\ncondition: true\ncollect:\n'
        '    [span for span in model("finder").spans('
        'messages[:1] + message("assistant", messages[0].content), 0.5) if true]\n',
        runtime_config=config,
        model_inference_backend=backend,
    )
    copied_result = copied.scan(_CONVERSATION)
    copied_span = copied_result.outputs["Copied"].items[0].value
    assert copied_span.to_dict() == {
        "message_index": 3,
        "start": 1,
        "end": 4,
        "score": {"value": 0.9},
        "source_ranges": [{"message_index": 0, "start": 1, "end": 4}],
    }


def test_prepared_policy_rejects_an_unknown_composed_expression_kind(tmp_path) -> None:
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"judge": model_entry(classify=0)},
        },
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(config, adapters={"fake": ModelAdapter()})
    policy = compile_policy(
        'rule "R"\ncondition: model("judge").classify(messages[:1] + messages[-1:]) > 0.5\n',
        runtime_config=config,
        model_inference_backend=backend,
    )
    document = orjson.loads(policy.to_bytes())
    document["rule"]["artifact"]["atom_specs"][0]["model_expression"]["kind"] = "unknown"

    with pytest.raises(ValueError, match="invalid compiled policy"):
        load_compiled_policy(
            orjson.dumps(document),
            runtime_config=config,
            model_inference_backend=backend,
        )


@pytest.mark.parametrize(
    "malformation",
    ["role", "content", "receiver", "oversized"],
)
def test_prepared_policy_rejects_malformed_constructed_messages(
    tmp_path,
    malformation,
) -> None:
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"judge": model_entry(classify=0)},
        },
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(config, adapters={"fake": ModelAdapter()})
    source = (
        'message("user", "first").add_message("assistant", "second")'
        if malformation == "receiver"
        else 'message("user", "first")'
    )
    policy = compile_policy(
        f'rule "R"\ncondition: model("judge").classify({source}) > 0.5\n',
        runtime_config=config,
        model_inference_backend=backend,
    )
    document = orjson.loads(policy.to_bytes())
    root = document["rule"]["artifact"]["atom_specs"][0]["model_expression"]

    def find_kind(node, kind):
        if node["kind"] == kind:
            return node
        for child in node["children"]:
            found = find_kind(child, kind)
            if found is not None:
                return found
        return None

    constructed = find_kind(
        root,
        "add_message" if malformation == "receiver" else "message",
    )
    assert constructed is not None
    if malformation == "role":
        constructed["children"][0]["value"] = "tool"
    elif malformation == "content":
        constructed["children"][1] = {
            "kind": "tools",
            "value": None,
            "children": [],
        }
    elif malformation == "receiver":
        constructed["children"][0] = {
            "kind": "messages",
            "value": None,
            "children": [],
        }
    else:
        constructed["children"][1]["value"] = "x" * 1_000_001

    with pytest.raises(ValueError, match="invalid compiled policy"):
        load_compiled_policy(
            orjson.dumps(document),
            runtime_config=config,
            model_inference_backend=backend,
        )


def test_prepared_policy_rejects_the_previous_compiler_identity(tmp_path) -> None:
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"judge": model_entry(classify=0)},
        },
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(config, adapters={"fake": ModelAdapter()})
    policy = compile_policy(
        'rule "R"\ncondition: model("judge").classify('
        'messages[:1] + message("assistant", "review")) > 0.5\n',
        runtime_config=config,
        model_inference_backend=backend,
    )
    document = orjson.loads(policy.to_bytes())
    document["rule"]["artifact"]["compiler_version"] = "compiler-7"

    with pytest.raises(ValueError, match="compiler-7.*unsupported"):
        load_compiled_policy(
            orjson.dumps(document),
            runtime_config=config,
            model_inference_backend=backend,
        )


@pytest.mark.parametrize("shape", ["depth", "fanout"])
def test_prepared_policy_rejects_excessive_expression_shapes(tmp_path, shape) -> None:
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"judge": model_entry(classify=0)},
        },
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(config, adapters={"fake": ModelAdapter()})
    policy = compile_policy(
        'rule "R"\ncondition: model("judge").classify(messages) > 0.5\n',
        runtime_config=config,
        model_inference_backend=backend,
    )
    document = orjson.loads(policy.to_bytes())
    if shape == "depth":
        expression = {"kind": "literal", "value": True, "children": []}
        for _ in range(MAX_MODEL_EXPRESSION_DEPTH):
            expression = {"kind": "not", "value": None, "children": [expression]}
    else:
        child = {"kind": "messages", "value": None, "children": []}
        expression = {
            "kind": "concat",
            "value": None,
            "children": [child] * (MAX_MODEL_EXPRESSION_FANOUT + 1),
        }
    document["rule"]["artifact"]["atom_specs"][0]["model_expression"] = expression

    with pytest.raises(ValueError, match="invalid compiled policy"):
        load_compiled_policy(
            orjson.dumps(document),
            runtime_config=config,
            model_inference_backend=backend,
        )


def test_message_selection_obeys_the_existing_comprehension_budget(tmp_path) -> None:
    adapter = ModelAdapter()
    scanner = v2_scanner(
        tmp_path,
        {"R": 'model("judge").classify([m for m in messages if true]) > 0.5'},
        {"judge": model_entry(classify=0)},
        adapter,
        max_comprehension_size=2,
    )

    result = scanner.scan_data(_CONVERSATION)

    assert result.rule_matches == []
    assert result.evaluation_errors == {
        "R": "message selection source has 3 items, exceeding max_comprehension_size=2"
    }
    assert adapter.classification_messages == []


def test_empty_selection_skips_the_model_call(tmp_path) -> None:
    adapter = ModelAdapter()
    scanner = v2_scanner(
        tmp_path,
        {"R": 'model("judge").classify([m for m in messages if m.role == "system"]) > 0.5'},
        {"judge": model_entry(classify=0)},
        adapter,
    )

    result = scanner.scan_data(_CONVERSATION)

    assert (result.rule_matches, result.is_complete) == ([], True)
    assert adapter.classification_messages == []


def test_composition_rejects_a_tool_result_before_its_call(tmp_path) -> None:
    conversation = {
        "messages": [
            {"role": "user", "content": "run"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "done"},
        ]
    }
    adapter = ModelAdapter()
    scanner = v2_scanner(
        tmp_path,
        {"R": 'model("judge").classify(messages[-1:] + messages[1:2]) > 0.5'},
        {"judge": model_entry(classify=0)},
        adapter,
    )

    result = scanner.scan_data(conversation)

    assert result.rule_matches == []
    assert result.evaluation_errors == {
        "R": "tool-result message selected before its originating assistant tool call"
    }
    assert adapter.classification_messages == []


@pytest.mark.parametrize(
    ("model_input", "message"),
    [
        pytest.param(
            "messages[1:2] + messages[1:2]",
            "message selection contains duplicate tool-call ID 'call-1'",
            id="duplicate-call",
        ),
        pytest.param(
            "messages[1:] + messages[-1:]",
            "message selection contains duplicate tool-result for tool-call ID 'call-1'",
            id="duplicate-response",
        ),
        pytest.param(
            "messages[1:] + messages[1:]",
            "message selection contains duplicate tool-call ID 'call-1'",
            id="repeated-call-response-unit",
        ),
        pytest.param(
            'messages[1:2] + message("user", "separator") + messages[1:]',
            "message selection contains duplicate tool-call ID 'call-1'",
            id="synthetic-between-duplicate-tool-units",
        ),
        pytest.param(
            "(messages[1:2] + messages[1:2]).assistant_messages",
            "message selection contains duplicate tool-call ID 'call-1'",
            id="filtered-duplicate-call",
        ),
        pytest.param(
            "messages[1:2] + (messages[1:] + messages[-1:]).tool_responses",
            "message selection contains duplicate tool-result for tool-call ID 'call-1'",
            id="filtered-duplicate-response",
        ),
    ],
)
def test_composed_selections_reject_repeated_tool_units_before_dispatch(
    tmp_path, model_input, message
) -> None:
    adapter = ModelAdapter()
    scanner = v2_scanner(
        tmp_path,
        {"R": f'model("judge").classify({model_input}) > 0.5'},
        {"judge": model_entry(classify=0)},
        adapter,
    )

    result = scanner.scan_data(_TOOL_CONVERSATION)

    assert result.rule_matches == []
    assert result.evaluation_errors == {"R": message}
    assert adapter.classification_messages == []


def test_failed_self_containment_check_does_not_mutate_source_links() -> None:
    source = Messages.from_openai(_TOOL_CONVERSATION["messages"])
    assistant = source[1]
    response = source[2]
    assert isinstance(assistant, AssistantMessage)
    assert isinstance(response, ToolMessage)
    call = assistant.tool_calls[0]

    with pytest.raises(ValueError, match="duplicate tool-call ID 'call-1'"):
        Messages.combine((source[1:], source[1:]))

    source.assert_self_contained()
    assert call.response is response
    assert response.call is call
    assert source.to_openai_messages() == _TOOL_CONVERSATION["messages"]


def test_distinct_tool_units_can_be_reordered_without_losing_links(tmp_path) -> None:
    conversation = {
        "messages": [
            {"role": "user", "content": "run both"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "first", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "first result"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {"name": "second", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-2", "content": "second result"},
        ]
    }
    adapter = ModelAdapter()
    scanner = v2_scanner(
        tmp_path,
        {"R": 'model("judge").classify(messages[3:] + messages[1:3]) > 0.5'},
        {"judge": model_entry(classify=0)},
        adapter,
    )

    result = scanner.scan_data(conversation)

    expected = conversation["messages"][3:] + conversation["messages"][1:3]
    assert (result.rule_matches, result.evaluation_errors) == (["R"], None)
    assert adapter.classification_messages == [expected]
    sent = adapter.classification_messages[0]
    assert [message.get("tool_call_id") for message in sent if message["role"] == "tool"] == [
        "call-2",
        "call-1",
    ]


@pytest.mark.asyncio
async def test_pure_conditional_does_not_increment_external_call_accounting() -> None:
    rule = DSLParser().parse('rule "R"\ncondition: true if true else false\n')
    artifact = compile_rule(rule).artifact
    (spec,) = artifact.atom_specs.values()
    assert spec.cost is Cost.PURE
    detector_calls = 0

    async def local_expression(_spec, _text) -> bool:
        nonlocal detector_calls
        detector_calls += 1
        return True

    context = ScanContext(messages=_CONVERSATION["messages"], tools=[])
    result = await evaluate_async(
        artifact,
        build_evaluation_input(context),
        async_detector=local_expression,
        minimize_witness=False,
    )

    assert result.matched is True
    assert detector_calls == 1
    assert result.external_calls == 0


def test_version_one_pure_conditional_keeps_the_rule_timeout(monkeypatch) -> None:
    config = QuarryConfig(
        config_version=1,
        quarry_rule_texts=('rule "R"\ncondition: true if true else false\n',),
    )
    scanner = DSLScanner(
        config,
        limits=SafetyLimits(rule_timeout=0.01, inference_timeout=1.0),
    )
    original_evaluate = scanner_module.evaluate_async

    async def delayed_evaluate(*args, **kwargs):
        await asyncio.sleep(0.1)
        return await original_evaluate(*args, **kwargs)

    monkeypatch.setattr(scanner_module, "evaluate_async", delayed_evaluate)

    result = scanner.scan_data(_CONVERSATION)

    assert scanner._rule_grounds_external("R") is False
    assert result.timed_out_rules == ["R"]


@pytest.mark.parametrize(
    ("condition", "expected_external", "expected_timeout", "expected_calls"),
    [
        pytest.param(
            'any tool in tools where model_scan("gemma4", tool.description, 0.5, 1)',
            True,
            False,
            1,
            id="external-tool-quantifier",
        ),
        pytest.param(
            'any tool in tools where tool.name == "lookup" and '
            'model_scan("gemma4", tool.description, 0.5, 1)',
            True,
            False,
            1,
            id="composed-external-tool-quantifier",
        ),
        pytest.param(
            'any outer in tools where outer.name == "lookup" and '
            "(any inner in tools where "
            'model_scan("gemma4", inner.description, 0.5, 1))',
            True,
            False,
            1,
            id="nested-external-tool-quantifier",
        ),
        pytest.param(
            'any tool in tools where tool.name == "lookup"',
            False,
            True,
            0,
            id="pure-tool-quantifier",
        ),
        pytest.param(
            'model_scan("gemma4", user_messages[0].content, 0.5, 1)',
            True,
            False,
            1,
            id="top-level-external",
        ),
    ],
)
def test_version_one_timeout_classification_reaches_tool_quantifier_bodies(
    tmp_path,
    monkeypatch,
    condition,
    expected_external,
    expected_timeout,
    expected_calls,
) -> None:
    set_config_environment(monkeypatch)
    adapter = FakeAdapter()
    scanner = v1_scanner(tmp_path, condition, adapter)
    scanner.limits = SafetyLimits(
        rule_timeout=0.01,
        inference_timeout=1.0,
        max_model_calls=1,
    )
    original_evaluate = scanner_module.evaluate_async

    async def delayed_evaluate(*args, **kwargs):
        await asyncio.sleep(0.1)
        return await original_evaluate(*args, **kwargs)

    monkeypatch.setattr(scanner_module, "evaluate_async", delayed_evaluate)

    result = scanner.scan_data(
        {
            "messages": [{"role": "user", "content": "synthetic input"}],
            "tools": _TOOL_DEFINITIONS,
        }
    )

    assert scanner._rule_grounds_external("R") is expected_external
    assert bool(result.timed_out_rules) is expected_timeout
    assert result.rule_matches == ([] if expected_timeout else ["R"])
    assert len(adapter.classification_calls) == expected_calls
    assert result.evaluation_errors is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("condition", "expected_calls"),
    [
        pytest.param(
            'any tool in tools where model_scan("gemma4", tool.description, 0.5, 1)',
            1,
            id="external-tool-quantifier",
        ),
        pytest.param(
            'any outer in tools where outer.name == "lookup" and '
            "(any inner in tools where "
            'model_scan("gemma4", inner.description, 0.5, 1))',
            1,
            id="nested-external-tool-quantifier",
        ),
        pytest.param(
            'any tool in tools where tool.name == "lookup"',
            0,
            id="pure-tool-quantifier",
        ),
        pytest.param(
            'model_scan("gemma4", user_messages[0].content, 0.5, 1)',
            1,
            id="top-level-external",
        ),
    ],
)
async def test_tool_quantifier_external_accounting_counts_only_executed_body_calls(
    condition, expected_calls
) -> None:
    artifact = compile_rule(DSLParser().parse(f'rule "R"\ncondition: {condition}\n')).artifact
    detector_calls = 0

    async def detector(_spec, _text) -> bool:
        nonlocal detector_calls
        detector_calls += 1
        return True

    context = ScanContext(
        messages=[{"role": "user", "content": "synthetic input"}],
        tools=_TOOL_DEFINITIONS,
    )
    result = await evaluate_async(
        artifact,
        build_evaluation_input(context),
        async_detector=detector,
        minimize_witness=False,
    )

    assert result.matched is True
    assert detector_calls == expected_calls
    assert result.external_calls == expected_calls


@pytest.mark.parametrize(
    ("condition", "expected_calls"),
    [
        pytest.param(
            'true if true else model("judge").classify(messages) > 0.5',
            0,
            id="unselected-model-branch",
        ),
        pytest.param(
            'true if false else model("judge").classify(messages) > 0.5',
            1,
            id="selected-model-branch",
        ),
        pytest.param(
            'model("judge").classify(messages) > 0.5 '
            'if true else model("judge").classify(messages) > 0.5',
            1,
            id="one-of-two-model-branches",
        ),
    ],
)
def test_model_conditional_branches_are_lazy_and_share_the_call_budget(
    tmp_path, condition, expected_calls
) -> None:
    adapter = ModelAdapter()
    scanner = v2_scanner(
        tmp_path,
        {"R": condition},
        {"judge": model_entry(classify=0)},
        adapter,
        max_model_calls=1,
    )

    result = scanner.scan_data(_CONVERSATION)

    assert scanner._rule_grounds_external("R") is True
    assert (result.rule_matches, result.evaluation_errors) == (["R"], None)
    assert adapter.classifications == expected_calls


def test_filter_comprehension_render_parse_round_trip_and_source_graph() -> None:
    source = (
        'rule "R"\ncondition: model("judge").classify('
        '[selected for item in messages if item.role == "user"]) > 0.5\n'
    )
    rule = DSLParser().parse(source)
    filtered = next(
        node
        for node in walk_condition_nodes(rule.condition)
        if isinstance(node, FilterComprehension)
    )

    rendered_filter = render_condition(filtered)
    rendered_condition = render_condition(rule.condition)
    reparsed = DSLParser().parse(f'rule "R"\ncondition: {rendered_condition}\n')
    reparsed_filter = next(
        node
        for node in walk_condition_nodes(reparsed.condition)
        if isinstance(node, FilterComprehension)
    )
    graph = build_rule_visualization(rule)
    filter_node = next(node for node in graph.nodes if node.node_type == "FilterComprehension")

    assert rendered_filter == '[selected for item in messages if (item.role == "user")]'
    assert reparsed_filter == filtered
    assert filter_node.label == rendered_filter


def test_mixed_constructor_rendering_round_trips_as_authored_syntax() -> None:
    rule = DSLParser().parse(
        'rule "R"\ncondition: model("judge").classify('
        'messages[:1] + message("assistant", "review")) > 0.5\n'
    )

    rendered = render_condition(rule.condition)
    reparsed = DSLParser().parse(f'rule "R"\ncondition: {rendered}\n')

    assert reparsed.condition == rule.condition
    assert 'messages[:1] + message("assistant", "review")' in rendered
