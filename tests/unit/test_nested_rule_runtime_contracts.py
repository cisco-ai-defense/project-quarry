# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Nested calls retain selected messages, source clocks and model-input failures."""

from dataclasses import replace

import pytest

from quarry import DSLScanner, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.integrations.resources import requirements_from_artifact
from quarry.policy import CompiledPolicy
from quarry.safety.limits import SafetyLimits
from tests.unit.model_fakes import FakeAdapter, set_config_environment, v1_backend, write_v1_config

CHAT = {
    "messages": [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "needle"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "other"},
    ]
}


def _policy(source, config, restored, backend=None):
    scanner = DSLScanner(
        replace(config, quarry_rules_paths=(), quarry_rule_texts=(source,)),
        model_inference_backend=backend,
    )
    # The policy API stores one entry point; its artifact carries its nested checks.
    name = next(reversed(scanner.compiled_rules))
    entry = DSLScanner._from_compiled_policy(
        config,
        rule=scanner.loaded_rules[name],
        artifact=scanner.compiled_rules[name],
        resource_requirements=requirements_from_artifact(scanner.compiled_rules[name]),
        model_inference_backend=backend,
    )
    policy = CompiledPolicy._from_scanner(entry, source)
    if restored:
        wire = policy.to_bytes()
        policy = load_compiled_policy(wire, runtime_config=config, model_inference_backend=backend)
        assert policy.to_bytes() == wire
    return policy


def _wrappers(source, *, depth, selected="messages", negated=False):
    previous = "Leaf"
    for number in range(depth):
        name = f"Caller{number}"
        argument = selected if number == depth - 1 else "messages"
        negation = "not " if negated and number == depth - 1 else ""
        source += (
            f'\nrule "{name}" imports: check = "{previous}" condition: {negation}check({argument})'
        )
        previous = name
    return source, previous


@pytest.mark.parametrize(
    "selection",
    ["messages.user_messages", "messages[1:2]", "messages[1:].user_messages", "messages[1:3]"],
)
@pytest.mark.parametrize("inner", ["leaf(messages[0])", "any m in messages where leaf(m)"])
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True])
def test_nested_single_message_promotion_uses_its_root_ordinal(selection, inner, restored, explain):
    source = (
        'rule "Leaf" condition: len(messages) == 1 and messages[0].content == "needle"\n'
        f'rule "Middle" imports: leaf = "Leaf" condition: {inner}\n'
        f'rule "Outer" imports: middle = "Middle" condition: middle({selection})'
    )
    policy = _policy(source, QuarryConfig(config_version=2), restored)
    result = policy.scan(CHAT, explain=explain)
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Outer"]


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True])
def test_indexed_constructed_message_promotion_retains_its_own_text(restored, explain):
    source = (
        'rule "Leaf" condition: messages[0].content == "constructed"\n'
        'rule "Outer" imports: leaf = "Leaf" condition: leaf(message("user", "constructed")[0])'
    )
    policy = _policy(source, QuarryConfig(config_version=2), restored)
    result = policy.scan(CHAT, explain=explain)
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Outer"]


@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize(
    ("relation", "entries", "state", "matched"),
    [
        ("right_after", ["chat", "note"], "complete", True),
        ("right_after", ["note", "chat"], "complete", False),
        ("right_after", None, "incomplete", False),
        ("after(within: 1)", ["chat", "audit", "note"], "complete", False),
        ("after(within: 2)", ["chat", "audit", "note"], "complete", True),
    ],
)
def test_intermediate_calls_preserve_the_descendant_shared_clock(
    depth, restored, explain, relation, entries, state, matched
):
    source = f'rule "Leaf" condition: (any w in memory.notes where w.id == "n") {relation} user_said("go")'
    source, name = _wrappers(source, depth=depth)
    config = QuarryConfig(
        config_version=2,
        inputs={
            "memory.notes": {"content": ["body"]},
            "audit.records": {"facts": {"kind": {"type": "string"}}},
        },
        safety_limits=SafetyLimits(rule_timeout=None),
    )
    data = {
        "messages": [{"role": "user", "content": "go"}],
        "memory": {"notes": [{"id": "n", "body": "go"}]},
        "audit": {"records": [{"id": "a", "kind": "ok"}]},
    }
    if entries is not None:
        references = {
            "chat": {"message_index": 0},
            "note": {"input": "memory.notes", "id": "n"},
            "audit": {"input": "audit.records", "id": "a"},
        }
        data["timeline"] = [references[item] for item in entries]
        if "audit" not in entries:
            data["audit"]["records"] = []
    policy = _policy(source, config, restored)
    result = policy.scan(data, explain=explain)
    assert result.completion_state == state, result.evaluation_errors
    assert result.rule_matches == ([name] if matched else [])


@pytest.fixture
def model_config(tmp_path, monkeypatch):
    placeholder = tmp_path / "placeholder.rule"
    placeholder.write_text('rule "Placeholder" condition: true')
    set_config_environment(monkeypatch)
    return write_v1_config(tmp_path, [placeholder])


@pytest.mark.parametrize("index", [99, -5, 1])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("negated", [False, True])
@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True])
def test_called_dynamic_index_failure_cannot_become_a_false_or_negated_match(
    model_config, index, depth, negated, restored, explain
):
    source = f'rule "Leaf" condition: llm_score("gemma4", "p", message("user", messages[{index}].content)) > 0.5'
    source, name = _wrappers(
        source, depth=depth, selected="messages.user_messages[0:1]", negated=negated
    )
    adapter = FakeAdapter()
    policy = _policy(source, model_config, restored, v1_backend(model_config, adapter))
    result = policy.scan(CHAT, explain=explain)
    assert result.completion_state == "incomplete"
    assert result.rule_matches == []
    assert "out of range" in result.evaluation_errors[name]
    assert not adapter.called()


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize(
    ("condition", "matched"),
    [
        ('false and llm_score("gemma4", "p", message("user", messages[99].content)) > 0.5', False),
        ('true or llm_score("gemma4", "p", message("user", messages[99].content)) > 0.5', True),
    ],
)
def test_called_dead_model_inputs_remain_folded_away(model_config, depth, condition, matched):
    source, name = _wrappers(f'rule "Leaf" condition: {condition}', depth=depth)
    adapter = FakeAdapter()
    policy = _policy(source, model_config, True, v1_backend(model_config, adapter))
    result = policy.scan(CHAT)
    assert result.is_complete
    assert result.rule_matches == ([name] if matched else [])
    assert not adapter.called()


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True])
def test_negative_index_promotion_is_relative_to_selected_view(restored, explain):
    source = (
        'rule "Leaf" condition: messages[0].content == "needle"\n'
        'rule "Middle" imports: leaf = "Leaf" condition: leaf(messages[-1])\n'
        'rule "Outer" imports: middle = "Middle" condition: middle(messages[1:2])'
    )
    result = _policy(source, QuarryConfig(config_version=2), restored).scan(CHAT, explain=explain)
    assert result.is_complete
    assert result.rule_matches == ["Outer"]


@pytest.mark.asyncio
async def test_bound_message_promotion_preserves_partial_text_and_rejects_foreign_roots():
    from types import SimpleNamespace

    from quarry.engine.atoms import ModelExpression
    from quarry.integrations.model_expression import evaluate_model_expression
    from quarry.types.messages import Messages

    root = Messages.from_openai(CHAT["messages"])
    selected = root._select_ranges((1,), {1: (1, 5)})
    context = SimpleNamespace(conversation=SimpleNamespace(messages=selected))
    expression = ModelExpression("as_messages", None, (ModelExpression("binding", "m"),))
    promoted = await evaluate_model_expression(expression, context, bindings={"m": selected[0]})
    assert promoted.shares_root_with(root)
    assert promoted._ordinals == (1,)
    assert promoted[0] is selected[0]
    assert promoted[0].content == "eedl"
    foreign = Messages.from_openai(CHAT["messages"])[1]
    with pytest.raises(ValueError, match="owning selection"):
        await evaluate_model_expression(expression, context, bindings={"m": foreign})


@pytest.mark.parametrize("index", [0, -1])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("restored", [False, True])
def test_valid_nested_model_index_scores_only_the_selected_message(
    model_config, index, depth, restored
):
    source = f'rule "Leaf" condition: llm_score("gemma4", "p", message("user", messages[{index}].content)) > 0.5'
    source, name = _wrappers(source, depth=depth, selected="messages.user_messages[0:1]")
    adapter = FakeAdapter()
    scored = []
    policy = _policy(source, model_config, restored, v1_backend(model_config, adapter, scored))
    result = policy.scan(CHAT)
    assert result.is_complete
    assert result.rule_matches == [name]
    assert len(adapter.builder_messages) == 1
    assert scored == [[{"role": "user", "content": "needle"}]]
