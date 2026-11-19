# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Rule-local sender faults and causal links survive views without inventing data."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from quarry import IncompleteScanError, InvalidInputError, compile_policy, load_compiled_policy
from quarry.cli import scan_command
from quarry.engine.events import build_static_environment
from quarry.engine.matchers import MISSING, field_access
from quarry.integrations.content_transforms import apply_content_transform
from quarry.integrations.message_readability import require_model_messages
from quarry.integrations.model_usage import model_call_budget
from quarry.safety.timer import QuarryTimeoutError
from quarry.safety.validator import validate_conversation
from quarry.sdk import QuarryGuard
from quarry.types.context import ScanContext
from quarry.types.exceptions import ConversationValidationError
from quarry.types.messages import Messages
from quarry.types.transforms import ContentTransform, TransformOperation
from tests.unit.test_unreadable_tool_calls import model_backend, scanner, surface_config


def call(identity="c", name="lookup"):
    return {"id": identity, "type": "function", "function": {"name": name, "arguments": "{}"}}


def assistant(identity="c", **extra):
    return {"role": "assistant", "content": "HELD", "tool_calls": [call(identity)], **extra}


def reply(identity="c", content="REPLY", **extra):
    return {"role": "tool", "tool_call_id": identity, "content": content, **extra}


SENDERS = [
    pytest.param({"role": "alien", "content": "opaque"}, id="unknown-role"),
    pytest.param({"role": "system", "content": "opaque", "refusal": "no"}, id="system-refusal"),
    pytest.param({"role": "user", "content": "opaque", "refusal": "no"}, id="user-refusal"),
    pytest.param(assistant(tool_call_id="foreign"), id="assistant-result-id"),
    pytest.param(reply(refusal="no"), id="result-refusal"),
]


@pytest.mark.parametrize("sender", SENDERS)
@pytest.mark.parametrize("explain", [False, True])
def test_only_sender_field_reads_fail_and_evidence_does_not_read_unused_text(sender, explain):
    result = scanner(
        Always="true",
        Size="len(messages) == 2",
        Last='any m in messages[-1:] where m.content contains "visible"',
        Read='any m in messages[:1] where m.content contains "opaque"',
    ).scan_data(
        {"messages": [sender, {"role": "user", "content": "visible"}]}, enable_explain=explain
    )
    assert result.rule_matches == ["Always", "Size", "Last"]
    assert result.completion_state == "incomplete" and result.validation_error is None
    assert [failure.to_dict() for failure in result.failures] == [
        {"category": "invalid_input", "rule_name": "Read"}
    ]


@pytest.mark.parametrize("sender", SENDERS)
def test_unreadable_senders_keep_role_position_wire_and_unknown_field_semantics(sender):
    messages = Messages.from_openai([sender])
    value = messages[0]
    assert type(value).__name__ == "UnreadableMessage"
    assert field_access(value, "role") == sender["role"]
    assert field_access(value, "message_index") == 0
    assert field_access(value, "unknown_field") is MISSING
    for field in [
        "content",
        "content_parts",
        "openai_content",
        "refusal",
        "tool_calls",
        "tool_call_id",
        "id",
        "name",
        "call",
        "result",
    ]:
        with pytest.raises(InvalidInputError) as error:
            field_access(value, field)
        assert "opaque" not in str(error.value) and "foreign" not in str(error.value)
    assert messages.to_openai_messages() == [sender]
    projected = messages.to_openai_messages()
    projected[0]["content"] = "changed"
    assert messages.to_openai_messages() == [sender]


def test_user_call_id_is_readable_sender_data_and_does_not_create_a_result():
    raw = {"role": "user", "content": "visible", "tool_call_id": "c"}
    messages = Messages.from_openai([raw])
    assert type(messages[0]).__name__ == "UserMessage"
    assert field_access(messages[0], "tool_call_id") == "c"
    assert messages[0].content == "visible" and len(messages.tool_responses) == 0
    assert messages.to_openai_messages() == [raw]


def test_system_prompt_is_deferred_and_its_fault_is_rule_local():
    raw = [{"role": "system", "content": "opaque", "refusal": "no"}]
    static = build_static_environment(ScanContext(raw))
    assert static.surface_sizes["messages"] == 1
    with pytest.raises(InvalidInputError):
        _ = static.system_prompt
    result = scanner(Always="true", Prompt='system_prompt contains "opaque"').scan_data(
        {"messages": raw}
    )
    assert result.rule_matches == ["Always"] and result.failures[0].rule_name == "Prompt"


@pytest.mark.parametrize("identity", [None, "", "missing"])
def test_unusable_call_ids_do_not_poison_names_counts_or_wire(identity):
    raw = assistant(None if identity == "missing" else identity)
    if identity == "missing":
        raw["tool_calls"][0].pop("id")
    messages = Messages.from_openai([raw])
    value = messages.tool_calls[0]
    assert value.name == "lookup" and value.arguments == {}
    for field in ["id", "response"]:
        with pytest.raises(InvalidInputError):
            field_access(value, field)
    result = scanner(
        Name='called_tool("lookup")',
        Size="len(tool_calls) == 1",
        MissingResults='len(get_tool_responses("lookup")) == 0',
        Read='any c in tool_calls where c.id == "c"',
    ).scan_data({"messages": [raw]})
    assert result.rule_matches == ["Name", "Size", "MissingResults"]
    assert result.failures[0].category.value == "invalid_input"
    assert messages.to_openai_messages() == [raw]


@pytest.mark.parametrize("identity", [None, "", "unknown"])
def test_unassigned_results_have_readable_text_but_no_invented_call(identity):
    raw = reply(identity)
    messages = Messages.from_openai([raw])
    value = messages[0]
    assert value.result == "REPLY"
    for field in ["id", "tool_call_id", "call", "name"]:
        with pytest.raises(InvalidInputError):
            field_access(value, field)
    result = scanner(
        Text='any r in tool_responses where r.result contains "REPLY"',
        Size="len(tool_responses) == 1",
        Name='len(get_tool_responses("lookup")) > 0',
    ).scan_data({"messages": [raw]})
    assert result.rule_matches == ["Text", "Size"] and result.failures[0].rule_name == "Name"


def test_pending_collisions_consume_replies_and_later_resolved_reuse_is_exact():
    raw = [assistant(), assistant(), reply(), reply(), assistant(), reply(content="third")]
    messages = Messages.from_openai(raw)
    first, second, third = messages.tool_calls
    for ambiguous in [first, second]:
        assert ambiguous.name == "lookup"
        with pytest.raises(InvalidInputError):
            field_access(ambiguous, "id")
        with pytest.raises(InvalidInputError):
            field_access(ambiguous, "response")
    assert field_access(third, "id") == "c" and third.response is messages[-1]
    assert messages[-1].call is third
    assert messages[4:].tool_calls[0] is third
    with pytest.raises(InvalidInputError):
        field_access(messages[:1].tool_calls[0], "id")
    assert messages.to_openai_messages() == raw


@pytest.mark.parametrize("after", [False, True])
def test_unassigned_result_uncertainty_only_affects_preceding_calls(after):
    raw = [assistant(), reply("unknown")] if after else [reply("unknown"), assistant()]
    messages = Messages.from_openai(raw)
    value = messages.tool_calls[0]
    assert field_access(value, "id") == "c"
    if after:
        with pytest.raises(InvalidInputError):
            field_access(value, "response")
    else:
        assert field_access(value, "response") is None


def test_sequential_reuse_has_distinct_graph_nodes_in_slices_and_composition():
    raw = [assistant(), reply(content="first"), assistant(), reply(content="second")]
    messages = Messages.from_openai(raw)
    first, second = messages.tool_calls
    assert first.response is messages[1] and second.response is messages[3]
    assert messages[:1].tool_calls[0].response is messages[1]
    assert messages[1:2][0].call is first
    combined = Messages.combine((messages[:2], messages[2:]))
    assert combined.to_openai_messages() == raw
    assert combined.tool_calls[0].response is combined[1]
    assert combined.tool_calls[1].response is combined[3]
    with pytest.raises(ValueError, match="duplicate tool-call"):
        combined.assert_self_contained()
    assert first.response is messages[1] and second.response is messages[3]


@pytest.mark.asyncio
async def test_content_transform_preserves_reused_call_slots_and_source_links():
    raw = [assistant(), reply(content="FIRST"), assistant(), reply(content="SECOND")]
    source = Messages.from_openai(raw)
    transformed = await apply_content_transform(
        source, ContentTransform((TransformOperation("casefold"),))
    )
    assert transformed[1].content == "first" and transformed[3].content == "second"
    assert transformed.tool_calls[0].response is transformed[1]
    assert transformed.tool_calls[1].response is transformed[3]
    assert transformed[1].call is not transformed[3].call
    combined = Messages.combine((transformed[:2], transformed[2:]))
    assert combined.tool_calls[0].response is combined[1]
    assert combined.tool_calls[1].response is combined[3]
    assert source.to_openai_messages() == raw
    assert source.tool_calls[0].response is source[1]


@pytest.mark.asyncio
async def test_content_transform_keeps_unassigned_result_text_without_creating_links():
    source = Messages.from_openai([reply(None)])
    transformed = await apply_content_transform(
        source, ContentTransform((TransformOperation("casefold"),))
    )
    assert transformed[0].result == "reply"
    with pytest.raises(InvalidInputError):
        field_access(transformed[0], "call")
    combined = Messages.combine((transformed,))
    assert combined[0].result == "reply"
    with pytest.raises(InvalidInputError):
        field_access(combined[0], "call")


@pytest.mark.parametrize("prepared", [False, True])
def test_faulty_scan_does_not_poison_reused_policy_or_later_valid_scan(prepared):
    policy = compile_policy(
        'rule "R" condition: any m in messages where m.content contains "visible"'
    )
    if prepared:
        policy = load_compiled_policy(policy.to_bytes())
    bad = policy.scan({"messages": [{"role": "unknown", "content": "visible"}]})
    good = policy.scan({"messages": [{"role": "user", "content": "visible"}]})
    assert (
        bad.completion_state == "incomplete" and bad.failures[0].category.value == "invalid_input"
    )
    assert good.completion_state == "complete" and good.rule_matches == ["R"]


@pytest.mark.parametrize("role", ["system", "developer", "assistant", "unknown"])
@pytest.mark.parametrize("identity", [None, "", "c"])
def test_named_result_aggregate_only_reads_a_claimed_nonempty_sender_id(role, identity):
    result = scanner(
        Any='any r in get_tool_responses("lookup") where true',
        Count='len(get_tool_responses("lookup")) > 0',
    ).scan_data({"messages": [{"role": role, "content": "text", "tool_call_id": identity}]})
    # System/developer optional IDs must be nonempty even for otherwise unreadable data.
    if role in {"system", "developer"} and not identity:
        assert result.validation_error is not None
    else:
        assert not result.rule_matches and result.validation_error is None
        assert bool(result.failures) is bool(identity)
        if identity:
            assert result.failures[0].rule_name == "Count"


@pytest.mark.parametrize(
    "messages",
    [
        [assistant(None)],
        [reply()],
        [assistant(), assistant()],
        [assistant(), reply(), assistant(), reply()],
        [{"role": "unknown", "content": "text"}],
    ],
)
@pytest.mark.asyncio
async def test_bad_model_input_is_refused_before_cache_adapter_or_budget(
    tmp_path, monkeypatch, messages
):
    backend, adapter = model_backend(tmp_path)
    source = Messages.from_openai(messages)

    async def forbidden(*args, **kwargs):
        pytest.fail("unreadable model input reached the cache")

    monkeypatch.setattr(backend, "_call", forbidden)
    with model_call_budget(1) as budget:
        with pytest.raises(InvalidInputError):
            await backend.model_classify_messages("reader", source)
        assert budget.calls_by_alias == {}
    assert adapter.classifications == 0


@pytest.mark.asyncio
async def test_shared_model_guard_rejects_repeated_ids_before_custom_backend_dispatch():
    source = Messages.from_openai([assistant(), reply(), assistant(), reply()])
    with pytest.raises(InvalidInputError):
        await require_model_messages(source)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_guard_preserves_partial_matches_and_fails_closed_on_missing_call_id(
    tmp_path, asynchronous
):
    import asyncio

    guard = QuarryGuard(surface_config(tmp_path))
    with pytest.raises(IncompleteScanError) as error:
        if asynchronous:
            await guard.scan_async({"messages": [dict(assistant(None), content="held")]})
        else:
            await asyncio.to_thread(
                guard.scan, {"messages": [dict(assistant(None), content="held")]}
            )
    result = error.value.scan_result
    assert result.rule_matches == ["Text"] and result.validation_error is None
    assert result.failures[0].category.value == "invalid_input"


def test_cli_reports_missing_call_id_as_an_incomplete_rule_read(tmp_path, capsys):
    config = surface_config(tmp_path)
    target = tmp_path / "input.json"
    target.write_text(json.dumps({"messages": [dict(assistant(None), content="held")]}))
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


def test_link_resolution_checks_deadlines_with_many_pending_collisions():
    from quarry.types.message_links import LinkEvent, resolve_links

    events = [LinkEvent("call", object(), "c") for _ in range(4096)]
    checks = 0

    def checkpoint():
        nonlocal checks
        checks += 1
        if checks == 4:
            raise QuarryTimeoutError("stop")

    with pytest.raises(QuarryTimeoutError):
        resolve_links(events, checkpoint)
    assert checks == 4


@pytest.mark.parametrize(
    "raw",
    [
        [{"role": "user", "content": "ok", "extra": True}],
        [{"role": "user", "content": []}],
        [
            {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": {"arguments": "{}"}}],
            }
        ],
    ],
)
def test_structural_errors_remain_global_even_when_the_rule_is_constant(raw):
    result = compile_policy('rule "R" condition: true').scan({"messages": raw})
    assert result.completion_state == "invalid" and result.validation_error
    assert not result.rule_matches


def test_explicit_allowed_roles_remains_a_validation_constraint():
    payload = {"messages": [{"role": "alien", "content": "text"}]}
    validate_conversation(payload, parse_arguments=False)
    with pytest.raises(ConversationValidationError):
        validate_conversation(payload, allowed_roles={"user"}, parse_arguments=False)
    assert json.loads(Messages.from_openai(payload["messages"]).to_json()) == payload["messages"]
