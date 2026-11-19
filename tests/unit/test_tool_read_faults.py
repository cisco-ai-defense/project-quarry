# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Offered definitions are preserved; only semantic reads of faulty tools fail."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from quarry import IncompleteScanError, InvalidInputError, compile_policy, load_compiled_policy
from quarry.cli import scan_command
from quarry.config import QuarryConfig
from quarry.core.conversation import Conversation
from quarry.engine.events import build_static_environment
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.integrations.model_usage import model_call_budget
from quarry.safety.timer import QuarryTimeoutError
from quarry.scanner import DSLScanner
from quarry.sdk import QuarryGuard
from quarry.types import messages as message_types
from quarry.types.context import ScanContext
from quarry.types.messages import Messages, Tool
from tests.unit.model_fakes import ModelAdapter, model_entry, passthrough_builder, v2_scanner


def tool(name="lookup", kind="function", **function):
    return {"type": kind, "function": {"name": name, **function}}


FAULTS = [
    pytest.param([tool(kind="other")], id="unknown-type"),
    pytest.param([tool(kind="")], id="empty-type"),
    pytest.param([tool(name="")], id="empty-name"),
    pytest.param([tool(), tool()], id="duplicate-name"),
]


def payload(tools):
    return {"messages": [{"role": "user", "content": "held"}], "tools": tools}


def scanner(**conditions):
    return DSLScanner(
        QuarryConfig(
            quarry_rule_texts=tuple(
                f'rule "{name}" condition: {condition}' for name, condition in conditions.items()
            )
        )
    )


@pytest.mark.parametrize("tools", FAULTS)
@pytest.mark.parametrize("explain", [False, True])
def test_faulty_definitions_do_not_poison_unrelated_or_skipped_rules(tools, explain):
    rules = scanner(
        Always="true",
        Text='any m in messages where m.content contains "held"',
        Count="len(messages) == 1",
        Skip='false and has_tool("lookup")',
        SkipQuantifier='false and (any t in tools where t.name == "lookup")',
        Alternate='true or has_tool("lookup")',
    )
    result = rules.scan_data(payload(tools), enable_explain=explain)
    assert result.is_complete and set(result.rule_matches) == {
        "Always",
        "Text",
        "Count",
        "Alternate",
    }
    assert not result.evaluation_errors
    if explain:
        assert len(result.explanations) == 6


@pytest.mark.parametrize("tools", FAULTS)
@pytest.mark.parametrize(
    "condition",
    [
        'has_tool("lookup")',
        'has_tools("lookup", "other")',
        "len(tools) > 0",
        'any t in tools where t.name == "lookup"',
        'all t in tools where t.name != "absent"',
    ],
)
def test_every_tools_surface_attributes_the_fault_to_its_reading_rule(tools, condition):
    result = scanner(
        Text='any m in messages where m.content contains "held"', Read=condition
    ).scan_data(payload(tools), enable_explain=True)
    assert result.completion_state == "incomplete" and result.rule_matches == ["Text"]
    assert result.validation_error is None and set(result.evaluation_errors) == {"Read"}
    assert [f.to_dict() for f in result.failures] == [
        {"category": "invalid_input", "rule_name": "Read"}
    ]


@pytest.mark.parametrize("tools", FAULTS)
def test_wire_inspection_stays_lossless_while_semantic_host_reads_fail(tools):
    source = payload(tools)
    conversation = Conversation.from_openai(source)
    assert conversation.tools_fault is not None
    assert conversation.raw_tools == tools and conversation.to_openai_dict() == source
    assert json.loads(conversation.to_json()) == source
    for read in [
        lambda: conversation.tools,
        lambda: conversation.has_tool("lookup"),
        lambda: conversation.has_tools("lookup"),
        lambda: conversation.tool_by_name("lookup"),
    ]:
        with pytest.raises(InvalidInputError) as error:
            read()
        assert error.value.details["code"] == "invalid_input"
    projected = conversation.raw_tools
    projected[0]["function"]["name"] = "changed"
    assert conversation.raw_tools == tools
    with pytest.raises(AttributeError):
        conversation._tools_fault = None


def test_static_setup_defers_tools_and_each_surface_size_separately():
    context = ScanContext(payload([tool(), tool()])["messages"], [tool(), tool()])
    static = build_static_environment(context)
    assert static.tool_count == 2
    assert static.surface_sizes["messages"] == 1 and static.surface_sizes["user_messages"] == 1
    for read in [
        lambda: static.tools["lookup"],
        lambda: len(static.tools),
        lambda: static.surface_sizes["tools"],
    ]:
        with pytest.raises(InvalidInputError):
            read()
    assert static.surface_sizes["messages"] == 1


def test_shared_deferred_mapping_retries_faults_without_storing_exceptions():
    static = build_static_environment(
        ScanContext(payload([tool(name="")])["messages"], [tool(name="")])
    )
    errors = []
    for _ in range(2):
        with pytest.raises(InvalidInputError) as failure:
            _ = static.tools["lookup"]
        errors.append(failure.value)
    assert errors[0] is not errors[1]
    assert static.tools._resolved is None


@pytest.mark.parametrize("tools", [[tool()], [tool(name=" ")], [tool(), tool(name="other")]])
def test_named_function_tools_keep_existing_semantics(tools):
    source = payload(tools)
    conversation = Conversation.from_openai(source)
    assert conversation.tools_fault is None and len(conversation.tools) == len(tools)
    first = tools[0]["function"]["name"]
    assert conversation.has_tool(first) and conversation.tool_by_name(first).name == first
    result = scanner(Read="len(tools) > 0").scan_data(source)
    assert result.is_complete and result.rule_matches == ["Read"]


@pytest.mark.parametrize("prepared", [False, True])
def test_prepared_tool_fault_does_not_poison_later_valid_scan(prepared):
    policy = compile_policy('rule "Read" condition: has_tool("lookup")')
    if prepared:
        policy = load_compiled_policy(policy.to_bytes())
    bad = policy.scan(payload([tool(), tool()]))
    assert (
        bad.completion_state == "incomplete" and bad.failures[0].category.value == "invalid_input"
    )
    good = policy.scan(payload([tool()]))
    assert good.is_complete and good.rule_matches == ["Read"]


def test_models_only_read_offered_tools_when_the_rule_passes_them(tmp_path):
    adapter = ModelAdapter()
    rules = v2_scanner(
        tmp_path,
        {
            "Plain": 'model("reader").classify(messages) > 0.5',
            "Offered": 'model("reader").classify(messages, tools) > 0.5',
        },
        {"reader": model_entry(classify=0)},
        adapter,
    )
    result = rules.scan_data(payload([tool(), tool()]))
    assert result.rule_matches == ["Plain"] and result.completion_state == "incomplete"
    assert set(result.evaluation_errors) == {"Offered"}
    assert len(adapter.classification_messages) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["score", "classify", "complete", "tokens"])
async def test_direct_tool_sequence_is_checked_before_cache_and_budget(
    tmp_path, operation, monkeypatch
):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"reader": model_entry(classify=0, score=True, tokens=True)},
        },
        tmp_path,
        require_rules=False,
    )
    adapter = ModelAdapter()
    backend = ModelInferenceBackend(
        config, adapters={"fake": adapter}, message_builders={"test": passthrough_builder}
    )
    messages = Messages.from_openai(payload([])["messages"])
    tools = [Tool.model_validate(tool()), Tool.model_validate(tool())]

    async def cache(*args, **kwargs):
        pytest.fail("faulty tools reached the cache boundary")

    monkeypatch.setattr(backend, "_call", cache)
    with model_call_budget(1) as budget:
        with pytest.raises(InvalidInputError):
            if operation == "score":
                await backend.model_score_messages("reader", "policy", messages, tools)
            elif operation == "classify":
                await backend.model_classify_messages("reader", messages, tools)
            elif operation == "complete":
                await backend.model_complete("reader", messages, tools)
            else:
                await backend.model_input_tokens_messages("reader", messages, tools)
        assert budget.calls_by_alias == {}
    assert (
        adapter.classifications
        == adapter.score_calls
        == adapter.completions
        == adapter.tokenizations
        == 0
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [None],
        [tool(name=4)],
        [{"type": "function", "function": {}}],
        [{"type": "custom"}],
        [{"function": {"name": "lookup"}}],
        [tool(kind=4)],
        [tool(description=None)],
        [tool(parameters=[])],
        [dict(tool(), other="held")],
        [tool(other="held")],
        [tool(parameters={"x": float("nan")})],
    ],
)
def test_malformed_offered_tools_still_reject_the_whole_input(value):
    result = scanner(Always="true").scan_data(payload(value))
    assert result.completion_state == "invalid" and not result.rule_matches


def test_schema_depth_and_cycles_remain_structural_faults():
    nested = {}
    for _ in range(101):
        nested = {"x": nested}
    cyclic = {}
    cyclic["x"] = cyclic
    for schema in [nested, cyclic]:
        result = scanner(Always="true").scan_data(payload([tool(parameters=schema)]))
        assert result.completion_state == "invalid"


def test_fault_diagnostic_does_not_expose_tool_names_types_or_schema():
    marker = "private-marker-" * 1000
    result = scanner(Read='has_tool("lookup")').scan_data(
        payload([tool(name=marker, kind=marker, description=marker)])
    )
    assert result.completion_state == "incomplete"
    assert marker not in str(result.evaluation_errors)
    assert len(result.evaluation_errors["Read"]) < 200


def test_fault_classification_checks_deadlines_during_large_collections(monkeypatch):
    tools = [Tool.model_validate(tool(str(i))) for i in range(1000)]
    calls = 0

    def deadline():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise QuarryTimeoutError("synthetic tool deadline")

    monkeypatch.setattr(message_types, "_raise_if_combine_expired", deadline)
    with pytest.raises(QuarryTimeoutError, match="synthetic tool deadline"):
        message_types.tools_fault(tools)


def surface_config(tmp_path):
    paths = []
    for name, condition in {
        "Text": 'any m in messages where m.content contains "held"',
        "Read": 'has_tool("lookup")',
    }.items():
        path = tmp_path / (name + ".rule")
        path.write_text(f'rule "{name}" condition: {condition}')
        paths.append(str(path))
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"rules": paths}))
    return path


def test_sdk_tool_fault_retains_partial_match_and_fails_closed(tmp_path):
    guard = QuarryGuard(surface_config(tmp_path))
    with pytest.raises(IncompleteScanError) as error:
        guard.scan(payload([tool(), tool()]))
    result = error.value.scan_result
    assert result.rule_matches == ["Text"] and result.failures[0].category.value == "invalid_input"


def test_cli_tool_fault_has_non_success_exit_and_partial_match(tmp_path, capsys):
    config = surface_config(tmp_path)
    target = tmp_path / "input.json"
    target.write_text(json.dumps(payload([tool(), tool()])))
    status = scan_command(
        SimpleNamespace(
            config=config,
            target=target,
            no_limits=False,
            recursive=False,
            verbose=False,
            batch=False,
            validate_only=False,
            output_format="json",
            explain=False,
            show_timing=False,
            output=None,
        )
    )
    [result] = json.loads(capsys.readouterr().out)
    assert status == 2 and result["completion_state"] == "incomplete"
    assert result["matches"] == [{"rule": "Text"}]
    assert result["failures"] == [{"category": "invalid_input", "rule_name": "Read"}]


@pytest.mark.asyncio
async def test_async_sdk_tool_fault_retains_partial_match(tmp_path):
    guard = QuarryGuard(surface_config(tmp_path))
    with pytest.raises(IncompleteScanError) as failure:
        await guard.scan_async(payload([tool(), tool()]))
    result = failure.value.scan_result
    assert result.rule_matches == ["Text"]
    assert result.failures[0].category.value == "invalid_input"


def test_custom_backend_does_not_receive_a_faulty_explicit_tools_argument(tmp_path):
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"reader": model_entry(classify=0)}},
        tmp_path,
        require_rules=False,
    )
    backend = ModelInferenceBackend(config, adapters={"fake": ModelAdapter()})

    async def custom(*args, **kwargs):
        pytest.fail("faulty offered tools reached custom backend")

    backend.model_classify_messages = custom
    policy = compile_policy(
        'rule "Read" condition: model("reader").classify(messages, tools) > 0.5',
        runtime_config=config,
        model_inference_backend=backend,
    )
    result = policy.scan(payload([tool(), tool()]))
    assert result.completion_state == "incomplete"
    assert result.failures[0].category.value == "invalid_input"


@pytest.mark.asyncio
@pytest.mark.parametrize("explain", [False, True, "support"])
@pytest.mark.parametrize(
    "condition,matched",
    [
        ('user_said("absent") and has_tool("lookup")', False),
        ('user_said("held") or has_tool("lookup")', True),
        ('user_said("absent") and(any t in tools where t.name=="lookup")', False),
        ('user_said("held") or(any t in tools where t.name=="lookup")', True),
    ],
)
async def test_exploration_of_skipped_tool_reads_preserves_the_deciding_verdict(
    explain, condition, matched
):
    policy = compile_policy('rule "Read" condition:' + condition)
    results = await policy.scan_many_async(
        [payload([tool(), tool()])],
        explain=explain,
        record_traces=True,
        decision_paths=True,
        explore_rate=1.0,
    )
    assert len(results) == 1 and results[0].is_complete
    assert results[0].rule_matches == (["Read"] if matched else [])
    assert not results[0].evaluation_errors


@pytest.mark.asyncio
@pytest.mark.parametrize("explain", [False, True, "support"])
async def test_model_evidence_does_not_read_omitted_faulty_offered_tools(tmp_path, explain):
    class ObservedAdapter(ModelAdapter):
        def __init__(self):
            super().__init__()
            self.sent_tools = []

        async def classify_messages(self, config, messages, tools, *, timeout=None):
            self.sent_tools.append(tools)
            return await super().classify_messages(config, messages, tools, timeout=timeout)

    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"reader": model_entry(classify=0)}},
        tmp_path,
        require_rules=False,
    )
    adapter = ObservedAdapter()
    backend = ModelInferenceBackend(config, adapters={"fake": adapter})
    policy = compile_policy(
        'rule "Read" condition:model("reader").classify(messages)>.5',
        runtime_config=config,
        model_inference_backend=backend,
    )
    results = await policy.scan_many_async(
        [payload([tool(), tool()])],
        explain=explain,
        record_traces=True,
        decision_paths=True,
        explore_rate=1.0,
    )
    assert len(results) == 1 and results[0].is_complete
    assert results[0].rule_matches == ["Read"]
    assert adapter.classification_messages == [[{"role": "user", "content": "held"}]]
    assert adapter.sent_tools == [[]]
