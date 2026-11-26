# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Production operation observation at engine and backend boundaries."""

import asyncio
import gc
import weakref
from types import SimpleNamespace

import pytest

from quarry.config import QuarryConfig
from quarry.engine.atoms import ModelExpression
from quarry.integrations.model_expression import evaluate_model_expression
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.integrations.model_usage import model_call_budget
from quarry.integrations.read_operations import (
    current_model_read,
    handoff_model_read,
    model_read,
    record_model_input,
)
from quarry.integrations.read_usage import ReadUsageLimits, model_read_usage, read_usage_scope
from quarry.types.context import ScanContext
from quarry.types.exceptions import MODEL_REFUSED, LLMError
from quarry.types.messages import Messages, Tool
from tests.unit.model_fakes import (
    ModelAdapter,
    NoWaitLimiter,
    model_entry,
    passthrough_builder,
    v2_scanner,
)

TEXT = "alice@example.com"
POLICY = "policy"


def backend(tmp_path, adapter=None):
    adapter = adapter or ModelAdapter()
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "reader": model_entry(
                    score=True,
                    classify=0,
                    tokens=True,
                    spans=0,
                    category="EMAIL",
                    physical_model="text-model",
                )
            },
        },
        tmp_path,
        require_rules=False,
    )
    return ModelInferenceBackend(
        config,
        adapters={"fake": adapter},
        message_builders={"test": passthrough_builder},
        rate_limiter=NoWaitLimiter(),
    ), adapter


def messages(text=TEXT):
    return Messages.from_openai([{"role": "user", "content": text}])


def rows(session):
    return session.snapshot()["rows"]


async def invoke(service, method, value):
    function = getattr(service, method)
    if method in {"model_score_messages", "model_judge_messages", "model_text_score"}:
        return await function("reader", POLICY, value)
    if method == "model_span_messages":
        return await function("reader", value, 0.5)
    if method in {"model_split_messages", "model_split_conversation"}:
        return await function("reader", value, 1000)
    if method == "model_classification":
        return await function("reader", TEXT)
    return await function("reader", value)


@pytest.mark.parametrize(
    ("method", "operation"),
    [
        ("model_score_messages", "score"),
        ("model_judge_messages", "score"),
        ("model_text_score", "score"),
        ("model_classify_messages", "classify"),
        ("model_classification", "classify"),
        ("model_span_messages", "spans"),
        ("model_complete_message", "complete"),
        ("model_complete", "complete"),
        ("model_text_completion", "complete"),
        ("model_input_tokens_messages", "input_tokens"),
        ("model_input_tokens", "input_tokens"),
        ("model_split_messages", "split"),
        ("model_split_conversation", "split"),
    ],
)
async def test_each_public_backend_request_has_one_logical_observation_and_one_physical_attempt(
    tmp_path, method, operation
):
    service, _ = backend(tmp_path)
    with model_call_budget(5) as physical, model_read_usage() as session:
        await invoke(service, method, messages())
        assert session._owners == 1
    observed = rows(session)
    assert len(observed) == 1
    row = observed[0]
    assert (row["model"], row["operation"], row["outcome"], row["observations"]) == (
        "reader",
        operation,
        "succeeded",
        1,
    )
    assert row["cut_identity_complete"] and row["unique_cuts"] == 1
    assert row["coverage"]["characters"]["value"] == len(TEXT) + (
        len(POLICY) if operation == "score" else 0
    )
    assert row["coverage"]["text_tokens"]["value"] is None
    assert row["coverage"]["text_tokens"]["measurement_states"] == {"unavailable": 1}
    assert physical.calls_by_alias == {"reader": 1}
    assert session.snapshot()["finalized"]


@pytest.mark.parametrize(
    ("condition", "operation"),
    [
        ('model("reader").score(messages, "policy") > 0.5', "score"),
        ('model("reader").classify(messages) > 0.5', "classify"),
        ('model("reader").complete(messages).content == "safe"', "complete"),
        ('model("reader").input_tokens(messages) > 0', "input_tokens"),
        ('any s in model("reader").spans(messages, 0.5) where true', "spans"),
    ],
)
async def test_engine_request_and_builtin_backend_adopt_one_observation(
    tmp_path, condition, operation
):
    scanner = v2_scanner(
        tmp_path,
        {"Outer": condition},
        {
            "reader": model_entry(
                score=True,
                classify=0,
                tokens=True,
                spans=0,
                category="EMAIL",
                physical_model="text-model",
            )
        },
        ModelAdapter(),
    )
    with model_read_usage() as session:
        result = await scanner.scan_data_async({"messages": [{"role": "user", "content": TEXT}]})
    assert result.is_complete and result.rule_matches == ["Outer"]
    observed = rows(session)
    assert len(observed) == 1
    assert (observed[0]["rule"], observed[0]["operation"], observed[0]["observations"]) == (
        "Outer",
        operation,
        1,
    )


@pytest.mark.parametrize("shape", ["missing", "empty", "invalid"])
async def test_request_is_observed_before_argument_resolution_or_native_guard_dispatch(
    tmp_path, shape
):
    service, adapter = backend(tmp_path)
    context = ScanContext(
        [{"role": "user", "content": TEXT}],
        resource_manager=SimpleNamespace(inference_backend=service),
    )

    def literal(value):
        return ModelExpression("literal", value)

    source = ModelExpression("messages")
    if shape == "missing":
        source = ModelExpression("index", 99, (source,))
    elif shape == "empty":
        source = ModelExpression("slice", children=(source, literal(0), literal(0)))
    else:
        source = ModelExpression("message", children=(literal("user"), literal(123)))
    call = ModelExpression("model_call", "reader\x1fscore", (source, literal(POLICY)))
    with model_read_usage() as session:
        if shape == "invalid":
            with pytest.raises(TypeError, match="must resolve to a string"):
                await evaluate_model_expression(call, context)
        else:
            await evaluate_model_expression(call, context)
    observed = rows(session)
    assert len(observed) == 1 and observed[0]["observations"] == 1
    assert observed[0]["outcome"] == ("failed" if shape == "invalid" else "not_sent")
    assert observed[0]["coverage"]["characters"]["value"] == (0 if shape == "empty" else None)
    assert adapter.score_calls == 0


@pytest.mark.parametrize("failure", ["error", "refusal", "invalid"])
async def test_failed_and_refused_requests_preserve_logical_input_and_native_failure(
    tmp_path, failure
):
    class Failing(ModelAdapter):
        async def score(self, config, wire, *, timeout=None):
            self.score_messages.append(wire)
            if failure == "error":
                raise LLMError("synthetic failure", {"code": "invalid_output"})
            if failure == "refusal":
                raise LLMError(
                    "synthetic refusal", {"code": MODEL_REFUSED, "reason": "refusal_field"}
                )
            return float("nan")

    service, adapter = backend(tmp_path, Failing())
    with model_read_usage() as session, pytest.raises(LLMError):
        await service.model_score_messages("reader", POLICY, messages())
    row = rows(session)[0]
    assert row["outcome"] == ("refused" if failure == "refusal" else "failed")
    assert row["coverage"]["characters"]["value"] == len(TEXT) + len(POLICY)
    assert row["observations"] == 1 and adapter.score_calls == 1


async def test_cancelled_real_request_releases_metadata_owner_and_keeps_cancellation(tmp_path):
    entered = asyncio.Event()
    stopped = asyncio.Event()

    class Waiting(ModelAdapter):
        async def score(self, config, wire, *, timeout=None):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    service, _ = backend(tmp_path, Waiting())
    with model_read_usage() as session:
        task = asyncio.create_task(service.model_score_messages("reader", POLICY, messages()))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(stopped.wait(), 1)
        assert session._owners == 1
    assert rows(session)[0]["outcome"] == "cancelled"


async def test_reentrant_adapter_call_is_a_new_request_after_handoff_was_consumed(tmp_path):
    class Reentrant(ModelAdapter):
        async def score(self, config, wire, *, timeout=None):
            self.score_messages.append(wire)
            if wire[0]["content"] == "outer":
                await self.backend.model_score_messages("reader", POLICY, messages("inner"))
            return 0.99

    adapter = Reentrant()
    service, _ = backend(tmp_path, adapter)
    adapter.backend = service
    with model_read_usage() as session, model_read("reader", "score"), handoff_model_read():
        await service.model_score_messages("reader", POLICY, messages("outer"))
    assert adapter.score_calls == 2
    row = rows(session)[0]
    assert row["observations"] == 2 and row["unique_cuts"] == 2
    assert row["coverage"]["characters"]["value"] == 10 + len(POLICY)


async def test_handoff_is_single_use_and_does_not_cross_task_or_reader_boundaries(tmp_path):
    service, _ = backend(tmp_path)
    with model_read_usage() as session:
        with read_usage_scope("Parent"), model_read("reader", "score"), handoff_model_read():
            await service.model_score_messages("reader", POLICY, messages())
            await service.model_score_messages("reader", POLICY, messages())
        with read_usage_scope("TaskParent"), model_read("reader", "score"), handoff_model_read():
            await asyncio.create_task(service.model_score_messages("reader", POLICY, messages()))
        with (
            read_usage_scope("OldReader"),
            model_read("reader", "score"),
            handoff_model_read(),
            read_usage_scope("NewReader"),
        ):
            await service.model_score_messages("reader", POLICY, messages())
    assert {row["rule"]: row["observations"] for row in rows(session)} == {
        "Parent": 2,
        "TaskParent": 2,
        "OldReader": 1,
        "NewReader": 1,
    }
    assert current_model_read() is None


async def test_completed_operation_metadata_does_not_retain_native_messages(tmp_path):
    service, _ = backend(tmp_path)
    with model_read_usage() as session:
        value = messages()
        reference = weakref.ref(value[0])
        await service.model_score_messages("reader", POLICY, value)
        del value
        gc.collect()
        assert reference() is None
        assert session._owners == 1


@pytest.mark.parametrize("limit", ["events", "owners", "input_work"])
async def test_metadata_admission_limits_do_not_repeat_provider_work_or_change_values(
    tmp_path, limit
):
    service, adapter = backend(tmp_path)
    with model_read_usage(ReadUsageLimits(**{f"max_{limit}": 1})) as session:
        assert await service.model_score_messages("reader", POLICY, messages()) == 0.99
        assert await service.model_score_messages("reader", POLICY, messages()) == 0.99
    assert adapter.score_calls == 1
    assert (
        not session.snapshot()["observations_complete"]
        or not session.snapshot()["measurements_complete"]
    )


async def test_offered_schema_text_is_measured_without_projecting_tool_parameter_properties(
    tmp_path, monkeypatch
):
    service, _ = backend(tmp_path)
    tool = Tool.model_validate(
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "search",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        }
    )
    original = Tool.parameters
    # Native serialization uses frozen source storage. Optional accounting must do so too.
    monkeypatch.setattr(
        Tool,
        "parameters",
        property(lambda _self: (_ for _ in ()).throw(AssertionError("defensive schema property"))),
    )
    with model_read_usage() as session:
        await service.model_classify_messages("reader", messages(), [tool])
    expected = len(TEXT) + sum(
        map(
            len,
            [
                "lookup",
                "search",
                "type",
                "properties",
                "required",
                "object",
                "q",
                "type",
                "string",
                "q",
            ],
        )
    )
    assert rows(session)[0]["coverage"]["characters"]["value"] == expected
    assert original is not None


def test_public_async_argument_binding_still_rejects_invalid_calls_immediately(tmp_path):
    service, _ = backend(tmp_path)
    with pytest.raises(TypeError):
        service.model_score_messages("reader")


@pytest.mark.parametrize(
    ("condition", "operation"),
    [
        ('all m in user_messages where model_scan("reader", m.content, 0.5, 0)', "classify"),
        ('all m in user_messages where model_score("reader", m.content, 0) > 0.5', "classify"),
        ('llm_score("reader", "policy", conversation()) > 0.5', "score"),
        ('llm_input_tokens("reader", conversation()) > 0', "input_tokens"),
        ('llm_complete("reader", conversation()) == "safe"', "complete"),
    ],
)
async def test_legacy_builtins_keep_distinct_field_occurrences_and_one_layered_read(
    tmp_path, condition, operation
):
    adapter = ModelAdapter()
    scanner = v2_scanner(
        tmp_path,
        {"Outer": condition},
        {"reader": model_entry(score=True, classify=0, tokens=True, physical_model="text-model")},
        adapter,
    )
    with model_read_usage() as session:
        result = await scanner.scan_data_async(
            {"messages": [{"role": "user", "content": "same"}] * 2}
        )
    assert result.is_complete and result.rule_matches == ["Outer"]
    row = rows(session)[0]
    assert len(rows(session)) == 1 and row["operation"] == operation
    assert row["observations"] == (2 if operation == "classify" else 1)
    assert row["coverage"]["characters"]["value"] == 8 + (
        len(POLICY) if operation == "score" else 0
    )
    if operation == "classify":
        assert row["unique_cuts"] == 2
        assert adapter.classifications == 1  # Native physical cache remains shared.


async def test_real_split_and_score_observe_requests_without_observing_internal_token_probes(
    tmp_path,
):
    scanner = v2_scanner(
        tmp_path,
        {
            "Outer": 'any chunk in model("reader").split(messages, 1000) where '
            'model("reader").score(chunk, "policy") > 0.5'
        },
        {"reader": model_entry(score=True, tokens=True)},
        ModelAdapter(),
    )
    with model_read_usage() as session:
        result = await scanner.scan_data_async({"messages": [{"role": "user", "content": TEXT}]})
    assert result.is_complete and result.rule_matches == ["Outer"]
    assert {row["operation"]: row["observations"] for row in rows(session)} == {
        "split": 1,
        "score": 1,
    }
    assert {row["coverage"]["characters"]["value"] for row in rows(session)} == {
        len(TEXT),
        len(TEXT) + len(POLICY),
    }


async def test_pending_operations_reserve_the_global_range_quota_before_dispatch(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class Waiting(ModelAdapter):
        async def score(self, config, wire, *, timeout=None):
            self.score_messages.append(wire)
            entered.set()
            await release.wait()
            return 0.99

    service, adapter = backend(tmp_path, Waiting())
    with model_read_usage(ReadUsageLimits(max_ranges=2)) as session:
        tasks = [
            asyncio.create_task(service.model_score_messages("reader", POLICY, messages()))
            for _ in range(5)
        ]
        try:
            await entered.wait()
            for _ in range(10):
                if len(session._events) == 5:
                    break
                await asyncio.sleep(0)
            assert len(session._events) == 5 and session._range_count == 2
            assert sum(event.range_count for event in session._events.values()) == 2
            assert all(event.outcome == "pending" for event in session._events.values())
        finally:
            release.set()
            results = await asyncio.gather(*tasks)
        assert results == [0.99] * 5 and session._owners == 1
    assert adapter.score_calls == 1
    assert sum(row["observations"] for row in rows(session)) == 5
    assert {row["outcome"] for row in rows(session)} == {"succeeded"}
    assert not session.snapshot()["measurements_complete"]


def test_completed_operation_uses_validated_staging_and_clears_its_range_buffers():
    with model_read_usage() as session:
        with model_read("reader", "score") as read:
            record_model_input(messages(), constitution=POLICY)
            assert read._characters.intervals
            # Another live request can exhaust identity work while this one is awaiting.
            session._text_inputs.registry._work = session._text_inputs.registry.max_work
        assert read._inputs == {} and read._characters.intervals == () and read._cut is None
    assert rows(session)[0]["outcome"] == "succeeded"
    assert rows(session)[0]["coverage"]["characters"]["value"] == len(TEXT) + len(POLICY)


async def test_assistant_refusal_text_is_part_of_validated_logical_input(tmp_path):
    service, _ = backend(tmp_path)
    value = Messages.from_openai([{"role": "assistant", "content": "body", "refusal": "refused"}])
    with model_read_usage() as session:
        await service.model_classify_messages("reader", value)
    measured = rows(session)[0]["coverage"]["characters"]
    assert measured["value"] == 11 and measured["known_units"] == 11
    assert measured["measurement_states"] == {"measured": 1}
