# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Local calls keep rule isolation, native text coordinates and scan ownership."""

import asyncio

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.types.messages import ToolCall
from tests.unit.model_fakes import ModelAdapter
from tests.unit.test_message_local_tool_conditions import DATA
from tests.unit.test_message_local_tool_paid_conditions import SCORE, local_policy
from tests.unit.test_nested_rule_runtime_contracts import _policy, _wrappers
from tests.unit.test_unreadable_tool_calls import transcript


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("filtered", [False, True])
def test_native_cardinality_does_not_spend_an_iteration_allowance(tmp_path, restored, filtered):
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "max_comprehension_size": 1}, tmp_path, require_rules=False
    )
    value = "[c for c in m.tool_calls if true]" if filtered else "m.tool_calls"
    source = f'rule "Read" condition: any m in assistant_messages where len({value}) == 2'
    policy = compile_policy(source, runtime_config=config)
    if restored:
        policy = load_compiled_policy(policy.to_bytes(), runtime_config=config)
    result = policy.scan(DATA)
    assert result.is_complete is (not filtered)
    assert result.rule_matches == ([] if filtered else ["Read"])
    if filtered:
        assert "comprehension limit" in result.evaluation_errors["Read"]


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("explain", [False, True, "support"])
@pytest.mark.parametrize(
    "body",
    [
        "len(m.tool_calls) == 1",
        "count([c for c in m.tool_calls if true]) == 1",
        "any c in m.tool_calls where true",
        "all c in m.tool_calls where true",
    ],
)
def test_native_opaque_call_sizes_do_not_read_call_fields(restored, explain, body):
    policy = compile_policy('rule "Read" condition: any m in assistant_messages where ' + body)
    if restored:
        policy = load_compiled_policy(policy.to_bytes())
    result = policy.scan({"messages": transcript()}, explain=explain)
    assert result.is_complete, result.evaluation_errors
    assert result.rule_matches == ["Read"]


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize(
    "body",
    [
        'any c in m.tool_calls where c.name == "lookup"',
        'any c in m.tool_calls where c.response.result contains "reply"',
        'count([c for c in m.tool_calls if c.arguments.value == "private"]) == 1',
    ],
)
def test_native_opaque_field_faults_remain_rule_local(restored, body):
    source = (
        'rule "Control" condition: true\nrule "Read" condition: '
        "any m in assistant_messages where " + body
    )
    from quarry import DSLScanner

    scanner = DSLScanner(QuarryConfig(quarry_rule_texts=(source,)))
    if restored:
        # The single entry's artifact carries the same rule-local guard contract.
        policy = compile_policy('rule "Read" condition: any m in assistant_messages where ' + body)
        policy = load_compiled_policy(policy.to_bytes())
        result = policy.scan({"messages": transcript()})
        assert result.rule_matches == []
    else:
        result = scanner.scan_data({"messages": transcript()})
        assert result.rule_matches == ["Control"]
    assert not result.is_complete
    assert set(result.evaluation_errors) == {"Read"}
    assert result.failures[0].category == "invalid_input"
    assert "{opaque" not in str(result.evaluation_errors)


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize(
    "selection",
    [
        "messages",
        "(messages[:0] + messages)",
        '[item for item in messages if item.role == "assistant"]',
    ],
)
@pytest.mark.parametrize("called", [False, True])
def test_paid_local_conditions_follow_called_and_computed_frames(
    tmp_path, restored, selection, called
):
    body = f'any m in {selection} where count([c for c in m.tool_calls if c.name == "lookup" and {SCORE}]) == 1'
    base, adapter = local_policy(tmp_path, "true")
    source, name = _wrappers('rule "Leaf" condition: ' + body, depth=int(called))
    policy = _policy(
        source, base._scanner.config, restored, base._scanner.resource_manager.inference_backend
    )
    result = policy.scan(DATA, explain=True)
    assert result.is_complete and result.rule_matches == [name], result.evaluation_errors
    assert adapter.score_messages == [[{"role": "user", "content": "lookup"}]]
    for detail in result.detailed_matches:
        for evidence in detail.model_results:
            assert not isinstance(evidence.expression_value, (int, float)) or isinstance(
                evidence.expression_value, bool
            )
            assert evidence.operator is evidence.threshold is None


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("field", ["result", "content"])
def test_linked_response_spans_keep_original_native_coordinates(restored, field):
    source = (
        'rule "Read" strings: $p="found" condition: true collect: '
        f"[s for s in $p.spans(assistant_messages[-1].tool_calls[0].response.{field}) if true]"
    )
    policy = compile_policy(source)
    if restored:
        policy = load_compiled_policy(policy.to_bytes())
    result = policy.scan(DATA, explain=True)
    assert result.is_complete and result.rule_matches == ["Read"], result.evaluation_errors
    (item,) = result.outputs["Read"].items
    assert (item.value.message_index, item.value.start, item.value.end) == (2, 0, 5)


@pytest.mark.parametrize("restored", [False, True])
def test_pure_filtered_mapping_keeps_the_consuming_judges_own_score(tmp_path, restored):
    condition = (
        'model("reader").score(message("user", '
        "join([c.name for c in [x for x in assistant_messages[-1].tool_calls "
        'if x.name == "lookup"]], "\\n")), "synthetic") > 0.5'
    )
    policy, adapter = local_policy(tmp_path, condition, restored=restored)
    result = policy.scan(DATA, explain=True)
    assert result.is_complete and result.rule_matches == ["Read"], result.evaluation_errors
    assert adapter.score_messages == [[{"role": "user", "content": "lookup"}]]
    (evidence,) = result.detailed_matches[0].model_results
    assert evidence.operation == "score" and evidence.expression_value == 0.99
    assert evidence.operator == ">" and evidence.threshold == 0.5


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("quantifier", ["any", "all"])
def test_cheap_membership_excludes_unselected_field_reads(monkeypatch, restored, quantifier):
    original = ToolCall.guard_read

    def guard(call, field):
        if field == "response" and call.function.name == "other":
            raise AssertionError("Unselected response was read")
        original(call, field)

    monkeypatch.setattr(ToolCall, "guard_read", guard)
    source = (
        'rule "Read" condition: any m in assistant_messages where '
        f'{quantifier} c in [c for c in m.tool_calls if c.name == "lookup"] '
        'where c.response.result contains "absent"'
    )
    policy = compile_policy(source)
    if restored:
        policy = load_compiled_policy(policy.to_bytes())
    result = policy.scan(DATA, explain=True)
    assert result.is_complete and result.rule_matches == [], result.evaluation_errors


@pytest.mark.parametrize("restored", [False, True])
def test_local_calls_and_declared_inputs_do_not_read_unselected_bad_sections(tmp_path, restored):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "inputs": {
                "memory.notes": {"facts": {"kind": {"type": "string"}}},
                "audit.unread": {"facts": {"kind": {"type": "string"}}},
            },
        },
        tmp_path,
        require_rules=False,
    )
    source = (
        'rule "Read" condition: '
        '(any m in assistant_messages where count([c for c in m.tool_calls if c.name == "lookup"]) == 1) '
        'and (any n in memory.notes where n.kind == "synthetic")'
    )
    policy = compile_policy(source, runtime_config=config)
    if restored:
        policy = load_compiled_policy(policy.to_bytes(), runtime_config=config)
    result = policy.scan(
        {
            **DATA,
            "memory": {"notes": [{"id": "n", "kind": "synthetic"}]},
            "audit": {"unread": "malformed-unused"},
        }
    )
    assert result.is_complete and result.rule_matches == ["Read"], result.evaluation_errors


@pytest.mark.parametrize("restored", [False, True])
async def test_cancelled_local_paid_work_does_not_poison_the_next_scan(tmp_path, restored):
    entered, released = asyncio.Event(), asyncio.Event()

    class Waiting(ModelAdapter):
        waiting = True

        async def score(self, config, messages, *, timeout=None):
            if self.waiting:
                entered.set()
                try:
                    await asyncio.Future()
                finally:
                    released.set()
            return await super().score(config, messages, timeout=timeout)

    adapter = Waiting()
    policy, _ = local_policy(
        tmp_path,
        f"any m in assistant_messages where any c in m.tool_calls where {SCORE}",
        restored=restored,
        adapter=adapter,
    )
    task = asyncio.create_task(policy.scan_async(DATA))
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(released.wait(), 5)
    adapter.waiting = False
    result = await policy.scan_async(DATA, explain=True)
    assert result.is_complete and result.rule_matches == ["Read"]


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_excluded_opaque_call_does_not_fault_a_paid_quantifier(tmp_path, restored, fused):
    filter_condition = 'model("reader").score(message("user", "filter"), "synthetic") < 0.1'
    filtered = f"[c for c in assistant_messages[-1].tool_calls if {filter_condition}]"
    body = 'c.name == "lookup"'
    condition = (
        f"any c in {filtered} where {body}" if fused else f"any c in ({filtered})[:1] where {body}"
    )
    policy, adapter = local_policy(tmp_path, condition, restored=restored)

    result = policy.scan({"messages": transcript()})

    assert result.is_complete and result.rule_matches == [], result.evaluation_errors
    assert len(adapter.score_messages) == 1
