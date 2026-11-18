# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Grounding replay preserves original values, guards, failures and detector work."""

import asyncio
import gc
import weakref
from types import SimpleNamespace

import pytest

from quarry.dsl.parser import DSLParser
from quarry.engine.atoms import AtomDomain, Cost, make_atom
from quarry.engine.compiler import compile_rule
from quarry.engine.events import build_evaluation_input
from quarry.engine.executor import evaluate_async
from quarry.engine.oracle import BOOT, Grounded, Oracle, _ObservedGrounded
from quarry.integrations.read_usage import (
    ReadUsageLimits,
    current_read_usage,
    grounding_read_observer,
    model_read_usage,
    read_usage_scope,
)
from quarry.types.context import ScanContext
from tests.unit.test_message_value_read_usage import measured_read


def atom(*, tool=False, **overrides):
    return make_atom(
        **{
            "domain": AtomDomain.STATIC if tool else AtomDomain.EVENT,
            "selector": "tools" if tool else "messages",
            "value_expr_key": "field:description" if tool else "field:content",
            "predicate": "external",
            "cost": Cost.MODEL,
            **overrides,
        }
    )


def event():
    return SimpleNamespace(
        position=0, source={"content": "synthetic text"}, surface_tags=frozenset({"messages"})
    )


@pytest.mark.parametrize("tool", [False, True])
@pytest.mark.parametrize("sync_hit", [False, True])
@pytest.mark.parametrize("value", [False, True])
async def test_cached_external_reads_replay_for_each_reader_preserving_value_identity(
    tool, sync_hit, value
):
    calls = 0
    expected = Grounded(value, {"kind": "synthetic evidence"})

    async def detector(_spec, _text, _event=None):
        nonlocal calls
        calls += 1
        measured_read(current_read_usage())
        return expected

    spec = atom(tool=tool)
    where = {"description": "synthetic tool"} if tool else event()
    with model_read_usage() as session:
        oracle = Oracle(async_detector=detector, read_observer=grounding_read_observer())
        compute = oracle.ground_tool_async if tool else oracle.ground_async
        with read_usage_scope("first"):
            assert await compute(spec, where) is expected
        with read_usage_scope("second"):
            for _ in range(2):
                if sync_hit:
                    assert (oracle.ground_tool if tool else oracle.ground)(spec, where) is expected
                else:
                    assert await compute(spec, where) is expected
        with read_usage_scope("evidence", phase="evidence"):
            assert (oracle.evidence_for_tool if tool else oracle.evidence_for)(
                spec, where
            ) is expected.evidence
        rows = session.snapshot()["rows"]
    assert calls == oracle.external_calls == 1
    assert [row["rule"] for row in rows] == ["first", "second"]
    assert [row["observations"] for row in rows] == [1, 1]
    assert all(row["coverage"]["characters"]["value"] == 12 for row in rows)
    assert oracle.evidence_for(spec, BOOT) is None


async def test_local_predicates_and_evidence_lookups_never_invoke_an_observer():
    class Forbidden:
        def capture(self):
            raise AssertionError("a local predicate must not capture model reads")

        def replay(self, _reads):
            raise AssertionError("a local/evidence read must not replay model reads")

    oracle = Oracle(read_observer=Forbidden())
    spec = atom(predicate="contains_any", patterns=("synthetic",), cost=Cost.PURE)
    where = event()
    first = oracle.ground(spec, where)
    assert first.value
    assert await oracle.ground_async(spec, where) is first
    assert oracle.evidence_for(spec, where) is first.evidence
    tool = {"description": "synthetic tool"}
    spec = atom(tool=True, predicate="contains_any", patterns=("synthetic",), cost=Cost.PURE)
    first = oracle.ground_tool(spec, tool)
    assert first.value
    assert await oracle.ground_tool_async(spec, tool) is first
    assert oracle.evidence_for_tool(spec, tool) is first.evidence


@pytest.mark.parametrize("reason", ["surface", "boot"])
async def test_guarded_external_atoms_capture_a_valid_empty_footprint_without_detector_work(reason):
    async def forbidden(*_):
        raise AssertionError("guarded input reached the detector")

    spec = atom(selector="tool_calls" if reason == "surface" else "messages")
    where = event() if reason == "surface" else BOOT
    with model_read_usage() as session:
        oracle = Oracle(async_detector=forbidden, read_observer=grounding_read_observer())
        with read_usage_scope("first"):
            first = await oracle.ground_async(spec, where)
        with read_usage_scope("second"):
            assert await oracle.ground_async(spec, where) is first
        report = session.snapshot()
    assert first.value is False and oracle.external_calls == 0
    assert report["observations_complete"] is True and report["rows"] == []


@pytest.mark.parametrize("tool", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_failed_or_cancelled_grounding_never_caches_a_partial_replay(tool, cancelled):
    started = asyncio.Event()
    calls = 0

    async def detector(*_):
        nonlocal calls
        calls += 1
        measured_read(current_read_usage())
        if calls > 1:
            return True
        started.set()
        if cancelled:
            await asyncio.Event().wait()
        raise ValueError("synthetic detector failure")

    spec = atom(tool=tool)
    where = {"description": "synthetic tool"} if tool else event()
    with model_read_usage() as session:
        oracle = Oracle(async_detector=detector, read_observer=grounding_read_observer())
        compute = oracle.ground_tool_async if tool else oracle.ground_async
        task = asyncio.create_task(compute(spec, where))
        try:
            await asyncio.wait_for(started.wait(), 2)
            if cancelled:
                task.cancel()
            with pytest.raises(asyncio.CancelledError if cancelled else ValueError):
                await task
            assert oracle._cache == {}
            with read_usage_scope("retry"):
                result = await compute(spec, where)
            with read_usage_scope("retry-reader"):
                assert await compute(spec, where) is result
            rows = session.snapshot()["rows"]
            assert [row["rule"] for row in rows] == [None, "retry", "retry-reader"]
            assert all(row["observations"] == 1 for row in rows)
            assert calls == oracle.external_calls == 2
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_capture_budget_bounds_empty_metadata_without_rejecting_or_repeating_values():
    calls = 0

    async def detector(*_):
        nonlocal calls
        calls += 1
        return True

    where = event()
    specs = [atom(detector_arg=str(i)) for i in range(12)]
    with model_read_usage(ReadUsageLimits(max_captures=2)) as session:
        oracle = Oracle(async_detector=detector, read_observer=grounding_read_observer())
        results = [await oracle.ground_async(spec, where) for spec in specs]
        for spec, expected in zip(specs, results, strict=True):
            assert expected.value is True
            assert await oracle.ground_async(spec, where) is expected
        assert sum(isinstance(value, _ObservedGrounded) for value in oracle._cache.values()) == 2
        assert session._captures == 2
        report = session.snapshot()
    assert calls == oracle.external_calls == 12
    assert not report["observations_complete"] and report["rows"] == []


async def test_owner_closed_or_missing_metadata_is_explicitly_incomplete_without_recompute():
    calls = 0

    async def detector(*_):
        nonlocal calls
        calls += 1
        measured_read(current_read_usage())
        return True

    with model_read_usage() as source:
        reference = weakref.ref(source)
        oracle = Oracle(async_detector=detector, read_observer=grounding_read_observer())
        spec, where = atom(), event()
        original = await oracle.ground_async(spec, where)
    del source
    gc.collect()
    assert reference() is None
    with model_read_usage() as destination:
        assert await oracle.ground_async(spec, where) is original
        report = destination.snapshot()
    assert calls == 1
    assert report["rows"] == [] and not report["observations_complete"]


async def test_concurrent_misses_preserve_existing_dispatch_and_replay_the_stored_result_only():
    first_started, second_started = asyncio.Event(), asyncio.Event()
    first_release, second_release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def detector(*_):
        nonlocal calls
        calls += 1
        number = calls
        measured_read(current_read_usage(), bytes([number]) * 32)
        (first_started if number == 1 else second_started).set()
        await (first_release if number == 1 else second_release).wait()
        return Grounded(number == 1, number)

    with model_read_usage() as session:
        oracle = Oracle(async_detector=detector, read_observer=grounding_read_observer())
        spec, where = atom(), event()

        async def read(rule):
            with read_usage_scope(rule):
                return await oracle.ground_async(spec, where)

        first = asyncio.create_task(read("first"))
        second = None
        try:
            await asyncio.wait_for(first_started.wait(), 2)
            second = asyncio.create_task(read("second"))
            await asyncio.wait_for(second_started.wait(), 2)
            first_release.set()
            assert (await first).value is True
            second_release.set()
            stored = await second
            assert stored.value is False
            assert await read("third") is stored
            rows = session.snapshot()["rows"]
            identities = {reader.rule: identity for reader, identity in session._bindings}
            assert identities["first"] != identities["second"] == identities["third"]
            assert [row["observations"] for row in rows] == [1, 1, 1]
            assert calls == oracle.external_calls == 2
        finally:
            for task in [first, second]:
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in [first, second] if task is not None), return_exceptions=True
            )


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize("gate", [False, True])
async def test_executor_preserves_guarded_verdict_and_work_with_generic_observer(observed, gate):
    rule = DSLParser().parse(
        'rule "R" condition: user_said("go") and model("judge").score(messages, "synthetic") > 0.5'
    )
    artifact = compile_rule(rule).artifact
    context = ScanContext(messages=[{"role": "user", "content": "go" if gate else "stop"}])
    calls = 0

    async def detector(*_):
        nonlocal calls
        calls += 1
        measured_read(current_read_usage())
        return True

    with model_read_usage() as session:
        result = await evaluate_async(
            artifact,
            build_evaluation_input(context),
            async_detector=detector,
            minimize_witness=False,
            read_observer=grounding_read_observer() if observed else None,
        )
        report = session.snapshot()
    assert result.matched == gate and calls == int(gate)
    if gate:
        assert report["rows"][0]["coverage"]["characters"]["value"] == 12
    else:
        assert report["rows"] == []
    assert report["observations_complete"]


async def test_default_cache_keeps_original_objects_without_optional_metadata():
    expected = Grounded(True, {"kind": "synthetic evidence"})

    async def detector(*_):
        return expected

    assert grounding_read_observer() is None
    oracle = Oracle(async_detector=detector)
    spec, where = atom(), event()
    assert await oracle.ground_async(spec, where) is expected
    assert await oracle.ground_async(spec, where) is expected
    assert list(oracle._cache.values()) == [expected]
    assert all(type(value) is Grounded for value in oracle._cache.values())


def test_generic_observation_contract_remains_pure():
    from pathlib import Path

    import quarry.engine.observations as observations
    from tests.engine.test_async_runtime import _quarry_imports

    contract = Path(observations.__file__)
    assert _quarry_imports(contract) == set()
