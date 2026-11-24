# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Bounded host snapshots and finalizers cannot reenter a held metadata mutex."""

from types import SimpleNamespace

import pytest

from quarry import WindowObservationLimits, observe_model_windows
from quarry.cache.async_cache import result_cache, scan_scope
from quarry.integrations import window_observations
from quarry.integrations.read_usage import read_usage_scope
from quarry.types.messages import Messages
from tests.unit.model_fakes import model_entry
from tests.unit.test_model_token_head_contracts import backend
from tests.unit.test_model_window_evidence import policy
from tests.unit.test_model_window_read_usage import MappedAdapter

DATA = {"messages": [{"role": "user", "content": "ab"}]}


def service(tmp_path, adapter):
    return backend(
        tmp_path,
        {
            "segments": {
                **model_entry(spans=0, category="SYNTHETIC"),
                "windows": {"unit": "tokens", "size": 3},
            }
        },
        adapter,
    )


async def test_growing_native_probability_storage_cannot_extend_the_admitted_snapshot(
    tmp_path, monkeypatch
):
    raw_vectors = []
    grown = set()
    original = window_observations.isfinite

    class Adapter(MappedAdapter):
        async def classify_window_batch(self, *args, **kwargs):
            raw = await super().classify_window_batch(*args, **kwargs)
            vector = [0.8, 0.2]
            raw_vectors.append(vector)
            raw.chunks = [SimpleNamespace(probabilities=vector)]
            return raw

    def concurrent_growth(value):
        # Scheduling control: mutate the provider-owned list after copying begins.
        # Ordinary list iteration would walk these extra values without admission.
        for vector in raw_vectors:
            if id(vector) not in grown:
                grown.add(id(vector))
                vector.extend([0.5] * 100)
        return original(value)

    monkeypatch.setattr(window_observations, "isfinite", concurrent_growth)
    adapter = Adapter()
    runtime = service(tmp_path, adapter)
    result_cache.clear()
    with (
        observe_model_windows(WindowObservationLimits(max_probabilities=2)) as observations,
        scan_scope(),
        read_usage_scope("R"),
    ):
        result = await runtime.model_span_messages(
            "segments", Messages.from_openai(DATA["messages"]), 0.5
        )
    (row,) = observations.report()["rows"]
    assert len(result.items) == 1 and len(adapter.requests) == 1
    assert len(raw_vectors[0]) == 102
    assert row["state"] == "measured" and row["probabilities"] == [[0.8, 0.2]]


async def test_unused_host_finalizers_are_released_outside_the_session_lock(tmp_path, monkeypatch):
    raw_vectors, acquired, reentrant_reports = [], [], []
    original = window_observations.isfinite

    class Unused:
        def __del__(self):
            # The nonblocking check makes a regression fail rather than hang pytest.
            available = observations._session._lock.acquire(blocking=False)
            acquired.append(available)
            if available:
                observations._session._lock.release()
                reentrant_reports.append(observations.report())

    class Adapter(MappedAdapter):
        async def classify_window_batch(self, *args, **kwargs):
            raw = await super().classify_window_batch(*args, **kwargs)
            vector = [0.8, Unused()]
            raw_vectors.append(vector)
            raw.chunks = [SimpleNamespace(probabilities=vector)]
            return raw

    def concurrent_release(value):
        for vector in raw_vectors:
            vector.clear()
        return original(value)

    monkeypatch.setattr(window_observations, "isfinite", concurrent_release)
    adapter = Adapter()
    runtime = service(tmp_path, adapter)
    result_cache.clear()
    with observe_model_windows() as observations, scan_scope(), read_usage_scope("R"):
        result = await runtime.model_span_messages(
            "segments", Messages.from_openai(DATA["messages"]), 0.5
        )
    (row,) = observations.report()["rows"]
    assert len(result.items) == 1 and len(adapter.requests) == 1
    assert acquired == [True] and len(reentrant_reports) == 1
    assert row["state"] == "partial" and row["probabilities"] == [None]
    assert row["probability_states"] == ["invalid_probability_vector"]


@pytest.mark.parametrize("unit", ["characters", "tokens"])
async def test_empty_selected_input_remains_measured_zero_at_the_minimum_work_budget(
    tmp_path, unit
):
    compiled, adapter = policy(tmp_path, 'model("classes").classify(messages) == 0', unit=unit)
    result_cache.clear()
    with observe_model_windows(WindowObservationLimits(max_work=1)) as observations:
        result = await compiled.scan_async({"messages": [{"role": "user", "content": ""}]})
    (row,) = observations.report()["rows"]
    assert result.is_complete and row["state"] == "not_sent"
    assert row["source_characters"] == row["covered_tokens"] == {"state": "measured", "value": 0}
    assert row["probabilities"] == [] and adapter.requests == adapter.encodings == []
