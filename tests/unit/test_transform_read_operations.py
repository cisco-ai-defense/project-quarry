# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Actual selected-part requests, layered adoption and owned transform input metadata."""

import asyncio
import gc
import weakref
from types import SimpleNamespace

import pytest

from quarry import load_compiled_policy
from quarry.engine.atoms import ModelExpression
from quarry.input_registry import InputObject
from quarry.integrations.adapter_message_transforms import apply_adapter_transform
from quarry.integrations.model_expression import evaluate_model_expression
from quarry.integrations.model_usage import model_call_budget
from quarry.integrations.read_coverage import ReadCoverage
from quarry.integrations.read_operations import current_model_read
from quarry.integrations.read_usage import ReadUsageLimits, model_read_usage
from quarry.types.adapter_transforms import TransformCacheIdentity, adapter_transform_reference
from quarry.types.exceptions import LLMError
from quarry.types.messages import Messages
from tests.unit.test_adapter_transform_dsl import Adapter, compile_condition, configured
from tests.unit.test_located_read_identity import disjoint, project


def text(value):
    return {"type": "text", "text": value}


def messages(value="same"):
    return Messages.from_openai([{"role": "user", "content": value}])


def rows(session):
    return session.snapshot()["rows"]


@pytest.mark.parametrize(
    "method", ["transform_unit", "transform_message_unit", "transform_units", "apply"]
)
async def test_direct_model_transform_boundaries_adopt_one_logical_request(tmp_path, method):
    _, service, adapter = configured(tmp_path, model=True)
    with model_call_budget(10) as physical, model_read_usage() as session:
        if method == "apply":
            result = await apply_adapter_transform(
                messages(), service, "convert", _message_value=True
            )
            assert result[0].content == "SAME"
        elif method == "transform_units":
            result = await service.transform_units("convert", [text("same")])
            assert result[0].text == "SAME"
        else:
            result = await getattr(service, method)("convert", text("same"))
            assert result.text == "SAME"
        assert session._owners == 1
    assert len(adapter.parts) == 1 and physical.calls_by_alias == {"extractor": 1}
    row = rows(session)[0]
    assert len(rows(session)) == row["observations"] == row["unique_cuts"] == 1
    assert (row["model"], row["operation"], row["outcome"]) == (
        "extractor",
        "transform",
        "succeeded",
    )
    assert row["coverage"]["characters"]["value"] == 4
    assert row["coverage"]["text_tokens"]["value"] is None


@pytest.mark.parametrize("prepared", [False, True])
def test_model_expression_transform_chain_preserves_requesting_rule_and_provider_work(
    tmp_path, prepared
):
    config, service, adapter = configured(tmp_path, model=True)
    policy = compile_condition(
        config, service, 'model("judge").classify(messages.transform("convert")) > 0.5'
    )
    if prepared:
        policy = load_compiled_policy(
            policy.to_bytes(), runtime_config=config, model_inference_backend=service
        )
    with model_read_usage() as session:
        result = policy.scan({"messages": [{"role": "user", "content": "same"}]})
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(adapter.parts) == len(adapter.inputs) == 1
    assert {
        (row["rule"], row["model"], row["operation"], row["observations"]) for row in rows(session)
    } == {("R", "extractor", "transform", 1), ("R", "judge", "classify", 1)}
    assert {row["coverage"]["characters"]["value"] for row in rows(session)} == {4}


@pytest.mark.parametrize("selected", ["text", "image_url", "input_audio"])
async def test_only_selected_parts_are_read_and_binary_payloads_are_not_text(tmp_path, selected):
    _, service, adapter = configured(
        tmp_path, definitions={"convert": {"model": "extractor", "parts": [selected]}}
    )
    value = messages(
        [
            text("ab"),
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            text("cd"),
            {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
        ]
    )
    with model_read_usage() as session:
        await apply_adapter_transform(value, service, "convert")
    assert [part["type"] for part in adapter.parts] == (
        ["text", "text"] if selected == "text" else [selected]
    )
    row = rows(session)[0]
    assert row["observations"] == 1 and row["outcome"] == "succeeded"
    assert row["coverage"]["characters"]["value"] == (4 if selected == "text" else 0)
    assert row["cut_identity_complete"]


async def test_overlapping_native_cuts_and_located_disjoint_fields_keep_source_coverage(tmp_path):
    _, service, _ = configured(tmp_path, model=True)
    original = messages("abcdefghij")
    owner = InputObject({"body": "abcdefghij"})
    with model_read_usage() as session:
        for start, end in [(1, 5), (3, 7)]:
            await apply_adapter_transform(
                original._select_ranges((0,), {0: (start, end)}), service, "convert"
            )
        located = project(["ab", "ij"], disjoint(owner))
        await apply_adapter_transform(located, service, "convert")
    row = rows(session)[0]
    assert row["observations"] == 3 and row["unique_cuts"] == 3
    assert (
        row["coverage"]["characters"]["value"] == 10
    )  # six overlapping native units + four located field units


async def test_presliced_native_input_created_before_collection_stays_unavailable(tmp_path):
    _, service, _ = configured(tmp_path, model=True)
    value = messages("abcdef")._select_ranges((0,), {0: (1, 4)})
    with model_read_usage() as session:
        await apply_adapter_transform(value, service, "convert")
    measured = rows(session)[0]["coverage"]["characters"]
    assert measured["value"] is None and measured["measurement_states"] == {"unavailable": 1}


@pytest.mark.parametrize("shape", ["empty_messages", "unselected", "empty_batch"])
async def test_successful_empty_selection_is_requested_but_unsent_validated_zero(tmp_path, shape):
    _, service, adapter = configured(
        tmp_path, definitions={"convert": {"model": "extractor", "parts": ["image_url"]}}
    )
    with model_read_usage() as session:
        if shape == "empty_batch":
            await service.transform_units("convert", [])
        else:
            await apply_adapter_transform(
                Messages() if shape == "empty_messages" else messages(), service, "convert"
            )
    row = rows(session)[0]
    assert row["outcome"] == "not_sent" and row["coverage"]["characters"]["value"] == 0
    assert adapter.parts == []


@pytest.mark.parametrize("container", [False, True])
async def test_failure_keeps_complete_single_part_or_explicit_partial_enumeration(
    tmp_path, container
):
    class Failing(Adapter):
        def transform(self, config, part, *, timeout=None):
            self.parts.append(part)
            raise LLMError("synthetic transform failure", {"code": "invalid_output"})

    _, service, _ = configured(tmp_path, model=True, adapter=Failing())
    with model_read_usage() as session, pytest.raises(LLMError):
        if container:
            await apply_adapter_transform(
                messages([text("first"), text("unseen")]), service, "convert"
            )
        else:
            await service.transform_unit("convert", text("first"))
    row = rows(session)[0]
    measured = row["coverage"]["characters"]
    assert row["outcome"] == "failed" and measured["known_units"] == 5
    assert measured["value"] == (None if container else 5)
    assert measured["measurement_states"] == ({"partial": 1} if container else {"measured": 1})


async def test_invalid_expression_source_records_a_failed_request_before_arguments(tmp_path):
    config, service, _ = configured(tmp_path, model=True)
    expression = ModelExpression(
        "model_transform",
        adapter_transform_reference("convert", config.transforms["convert"]),
        (ModelExpression("literal", 42),),
    )
    context = SimpleNamespace(resource_manager=SimpleNamespace(inference_backend=service))
    with model_read_usage() as session, pytest.raises(TypeError, match="source must be Messages"):
        await evaluate_model_expression(expression, context)
    row = rows(session)[0]
    assert row["operation"] == "transform" and row["outcome"] == "failed"
    assert row["coverage"]["characters"]["value"] is None


async def test_injected_single_unit_hook_retains_native_input_identity(tmp_path, monkeypatch):
    _, service, _ = configured(tmp_path, model=True)
    seen = []

    async def injected(name, unit):
        seen.append((name, dict(unit)))
        return unit["text"].upper()

    monkeypatch.setattr(service, "transform_unit", injected)
    value = messages("abcdef")
    with model_read_usage() as session:
        await apply_adapter_transform(
            value._select_ranges((0,), {0: (1, 4)}), service, "convert", _message_value=True
        )
        await apply_adapter_transform(
            value._select_ranges((0,), {0: (2, 5)}), service, "convert", _message_value=True
        )
    assert seen == [("convert", text("bcd")), ("convert", text("cde"))]
    assert rows(session)[0]["coverage"]["characters"]["value"] == 4


async def test_cached_repeated_transform_requests_release_native_owners_and_buffers(tmp_path):
    _, service, adapter = configured(tmp_path, model=True)
    with model_read_usage() as session:
        value = messages()
        reference = weakref.ref(value[0])
        await apply_adapter_transform(value, service, "convert")
        await apply_adapter_transform(value, service, "convert")
        del value
        gc.collect()
        assert reference() is None and session._owners == 1
    assert len(adapter.parts) == 1
    assert (
        rows(session)[0]["observations"] == 2
        and rows(session)[0]["coverage"]["characters"]["value"] == 4
    )


async def test_cancelled_transform_keeps_partial_input_and_closes_all_metadata_buffers(tmp_path):
    entered, stopped = asyncio.Event(), asyncio.Event()
    reads = []

    class Waiting(Adapter):
        async def transform(self, config, part, *, timeout=None):
            reads.append(current_model_read())
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    _, service, _ = configured(tmp_path, model=True, adapter=Waiting())
    with model_read_usage() as session:
        task = asyncio.create_task(apply_adapter_transform(messages(), service, "convert"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(stopped.wait(), 1)
        assert session._owners == 1
        assert reads and all(
            read._inputs == {}
            and read._characters.intervals == ()
            and read._complete_characters.intervals == ()
            and read._cut is None
            and read._complete_cut is None
            for read in reads
        )
    assert rows(session)[0]["outcome"] == "cancelled"
    assert rows(session)[0]["coverage"]["characters"]["value"] is None


@pytest.mark.parametrize("limit", ["events", "ranges", "input_work"])
async def test_transform_metadata_limits_do_not_change_native_work(tmp_path, limit):
    _, service, adapter = configured(tmp_path, model=True)
    with model_read_usage(ReadUsageLimits(**{"max_" + limit: 1})) as session:
        result = await apply_adapter_transform(
            messages([text("ab"), text("cd")]), service, "convert"
        )
    assert result[0].content == "ABCD" and len(adapter.parts) == 2
    assert all(row["outcome"] == "succeeded" for row in rows(session))


async def test_local_transforms_create_no_model_read_events(tmp_path):
    _, service, adapter = configured(tmp_path)
    with model_read_usage() as session:
        await apply_adapter_transform(messages(), service, "convert")
        await service.transform_message_unit("convert", text("local"))
        await service.transform_units("convert", [text("batch")])
    assert len(adapter.parts) == 3 and rows(session) == []


async def test_native_part_and_full_message_share_text_coordinates_across_operations(tmp_path):
    _, service, _ = configured(tmp_path, model=True)
    value = messages("abcdef")
    with model_read_usage() as session:
        await service.model_classify_messages("judge", value)
        await apply_adapter_transform(value._select_ranges((0,), {0: (1, 4)}), service, "convert")
        union = ReadCoverage()
        for event in session._events.values():
            union.add(event.characters.intervals)
        assert union.snapshot().characters == 6


async def test_reentrant_provider_request_is_independent_after_part_handoff_is_consumed(tmp_path):
    class Reentrant(Adapter):
        async def transform(self, config, part, *, timeout=None):
            self.parts.append(part)
            if part["text"] == "outer":
                await self.service.transform_unit("convert", text("inner"))
            return part["text"].upper()

    adapter = Reentrant()
    _, service, _ = configured(tmp_path, model=True, adapter=adapter)
    adapter.service = service
    with model_read_usage() as session:
        await apply_adapter_transform(messages("outer"), service, "convert")
    assert [part["text"] for part in adapter.parts] == ["outer", "inner"]
    assert rows(session)[0]["observations"] == 2
    assert rows(session)[0]["coverage"]["characters"]["value"] == 10


async def test_empty_text_is_a_sent_part_not_an_empty_selection(tmp_path):
    _, service, adapter = configured(tmp_path, model=True)
    with model_read_usage() as session:
        await apply_adapter_transform(messages(""), service, "convert")
    assert adapter.parts == [text("")]
    assert rows(session)[0]["outcome"] == "succeeded"
    assert rows(session)[0]["coverage"]["characters"]["value"] == 0


async def test_pending_container_reserves_ranges_and_keeps_an_explicit_limit_lower_bound(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class Waiting(Adapter):
        async def transform(self, config, part, *, timeout=None):
            self.parts.append(part)
            entered.set()
            await release.wait()
            return part["text"].upper()

    _, service, adapter = configured(tmp_path, model=True, adapter=Waiting())
    with model_read_usage(ReadUsageLimits(max_ranges=1)) as session:
        task = asyncio.create_task(
            apply_adapter_transform(messages([text("ab"), text("cd")]), service, "convert")
        )
        try:
            await entered.wait()
            assert session._range_count == 1
            assert all(event.outcome == "pending" for event in session._events.values())
        finally:
            release.set()
            result = await task
        assert result[0].content == "ABCD" and session._range_count <= 1
    measured = rows(session)[0]["coverage"]["characters"]
    assert len(adapter.parts) == 2 and rows(session)[0]["outcome"] == "succeeded"
    assert measured["value"] is None and measured["known_units"] == 2


async def test_identity_work_exhausted_while_awaiting_does_not_erase_validated_input(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class Waiting(Adapter):
        async def transform(self, config, part, *, timeout=None):
            entered.set()
            await release.wait()
            return part["text"].upper()

    _, service, _ = configured(tmp_path, model=True, adapter=Waiting())
    with model_read_usage() as session:
        task = asyncio.create_task(apply_adapter_transform(messages(), service, "convert"))
        try:
            await entered.wait()
            session._text_inputs.registry._work = session._text_inputs.registry.max_work
        finally:
            release.set()
            await task
    assert rows(session)[0]["outcome"] == "succeeded"
    assert rows(session)[0]["coverage"]["characters"]["value"] == 4


async def test_custom_backend_receives_identical_selected_units_and_is_observed(tmp_path):
    config, service, _ = configured(tmp_path, model=True)
    calls = []

    async def transform(name, unit):
        calls.append((name, dict(unit)))
        return unit["text"].upper()

    custom = SimpleNamespace(
        config=config, transform_parts=service.transform_parts, transform_unit=transform
    )
    with model_read_usage() as session:
        result = await apply_adapter_transform(messages(), custom, "convert", _message_value=True)
    assert result[0].content == "SAME" and calls == [("convert", text("same"))]
    assert (
        rows(session)[0]["observations"] == 1
        and rows(session)[0]["coverage"]["characters"]["value"] == 4
    )


async def test_mutating_injected_forwarder_does_not_falsely_validate_old_source_as_new_payload(
    tmp_path, monkeypatch
):
    _, service, adapter = configured(tmp_path, model=True)
    original = service.transform_unit

    async def changed(name, unit):
        return await original(name, text("diff"))

    monkeypatch.setattr(service, "transform_unit", changed)
    with model_read_usage() as session:
        await apply_adapter_transform(messages(), service, "convert", _message_value=True)
    measured = rows(session)[0]["coverage"]["characters"]
    assert adapter.parts == [text("diff")]
    assert measured["value"] is None and measured["known_units"] == 4


async def test_text_part_transform_does_not_read_opaque_call_fields(tmp_path):
    _, service, adapter = configured(
        tmp_path, definitions={"convert": {"model": "extractor", "parts": ["text"]}}
    )
    value = Messages.from_openai(
        [
            {
                "role": "assistant",
                "content": "same",
                "tool_calls": [
                    {
                        "type": "other",
                        "id": "synthetic-call",
                        "function": {"name": "lookup", "arguments": "{opaque"},
                    }
                ],
            }
        ]
    )
    with model_read_usage() as session:
        await apply_adapter_transform(value, service, "convert")
    assert adapter.parts == [text("same")]
    assert rows(session)[0]["coverage"]["characters"]["value"] == 4


def test_unchosen_transform_branch_creates_no_requested_read(tmp_path):
    config, service, adapter = configured(tmp_path, model=True)
    policy = compile_condition(
        config, service, 'false and messages.transform("convert")[0].content contains "SAME"'
    )
    with model_read_usage() as session:
        result = policy.scan({"messages": [{"role": "user", "content": "same"}]})
    assert (
        result.is_complete
        and result.rule_matches == []
        and adapter.parts == []
        and rows(session) == []
    )


async def test_model_batch_host_hook_preserves_native_part_sources_across_owned_children(
    tmp_path, monkeypatch
):
    _, service, adapter = configured(tmp_path, model=True)
    # Preserve the existing host override hook; built-in model transforms normally run
    # sequentially. The native batch runner owns and drains these explicit children.
    monkeypatch.setattr(
        service,
        "transform_cache_identity",
        lambda name: TransformCacheIdentity("synthetic.batch", "1"),
    )
    with model_read_usage() as session:
        result = await apply_adapter_transform(
            messages([text("ab"), text("ab")]), service, "convert", _message_value=True
        )
        assert session._owners == 1
    assert result[0].content == "ABAB" and adapter.parts == [text("ab")]
    assert len(rows(session)) == rows(session)[0]["observations"] == 1
    assert rows(session)[0]["coverage"]["characters"]["value"] == 4


async def test_custom_batch_hook_observes_selected_native_input_without_false_unsent_zero(tmp_path):
    config, service, _ = configured(tmp_path, model=True)
    calls = []

    async def batch(name, units, *, _message_value=False):
        calls.append((name, [dict(unit) for unit in units], _message_value))
        return [unit["text"].upper() for unit in units]

    custom = SimpleNamespace(
        config=config,
        transform_parts=service.transform_parts,
        transform_cache_identity=lambda name: TransformCacheIdentity("synthetic.batch", "1"),
        transform_units=batch,
        transform_unit=service.transform_unit,
    )
    with model_read_usage() as session:
        result = await apply_adapter_transform(
            messages([text("ab"), text("cd")]), custom, "convert", _message_value=True
        )
    assert result[0].content == "ABCD" and calls == [("convert", [text("ab"), text("cd")], True)]
    assert len(rows(session)) == rows(session)[0]["observations"] == 1
    assert rows(session)[0]["outcome"] == "succeeded"
    assert rows(session)[0]["coverage"]["characters"]["value"] == 4


async def test_versioned_local_batch_keeps_owned_work_and_has_no_model_events():
    from tests.unit.test_shared_transform_cache import backend as shared_backend

    service, adapter = shared_backend()
    source = messages([text(str(index % 4)) for index in range(20)])
    with model_read_usage() as session:
        result = await apply_adapter_transform(source, service, "clean", _message_value=True)
    assert result[0].content == source[0].content and len(adapter.calls) == 4
    assert rows(session) == []
