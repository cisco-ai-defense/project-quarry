# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Computed collection sizes stay scalar, with normal source effects and numeric bounds."""

from __future__ import annotations

from operator import eq, ge, gt, le, lt, ne

import pytest

from quarry import load_compiled_policy
from tests.unit.test_computed_message_conditions import PAYLOAD, prepare


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("function", ["len", "count"])
@pytest.mark.parametrize(
    "operator,compare", [("==", eq), ("!=", ne), (">", gt), (">=", ge), ("<", lt), ("<=", le)]
)
@pytest.mark.parametrize("bound", [10000, 1000000000, -0.5, 2.5])
def test_size_checks_accept_large_signed_and_fractional_bounds(
    tmp_path, prepared, function, operator, compare, bound
):
    policy, adapter = prepare(
        tmp_path, f'{function}(messages.transform("convert")) {operator} {bound}', prepared
    )
    result = policy.scan(PAYLOAD, explain=True)
    assert result.is_complete
    assert result.rule_matches == (["R"] if compare(2, bound) else [])
    assert len(adapter.parts) == 2


@pytest.mark.parametrize("function", ["len", "count"])
def test_threshold_magnitude_does_not_expand_the_compiled_graph(tmp_path, function):
    lengths = []
    for bound in [1, 10000, 1000000000]:
        policy, _ = prepare(tmp_path, f'{bound} < {function}(messages.transform("convert"))', False)
        lengths.append(len(policy.to_bytes()))
    assert max(lengths) - min(lengths) < 1000


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("function", ["len", "count"])
def test_size_of_a_bound_value_stays_at_its_own_element(tmp_path, prepared, function):
    policy, adapter = prepare(
        tmp_path,
        f'all outer in messages where {function}(message("user", outer.content).transform("convert")) < 10000',
        prepared,
    )
    result = policy.scan(PAYLOAD)
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(adapter.parts) == 2


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("function", ["len", "count"])
def test_size_of_the_same_value_inside_its_body_uses_the_child_frame(tmp_path, prepared, function):
    source = 'messages[1:].transform("convert")'
    policy, adapter = prepare(
        tmp_path,
        f'any item in {source} where {function}({source}) == 1 and item.content contains "safe"',
        prepared,
    )
    result = policy.scan(PAYLOAD, explain=True)
    assert result.is_complete and result.rule_matches == ["R"]
    assert len(adapter.parts) == 1


@pytest.mark.parametrize("function", ["len", "count"])
def test_size_skipped_by_a_cheap_gate_keeps_resource_preparation_without_dispatch(
    tmp_path, function
):
    policy, adapter = prepare(
        tmp_path, f'true or {function}(messages.transform("convert")) > 10000', True
    )
    assert policy.scan(PAYLOAD).rule_matches == ["R"]
    assert adapter.parts == []


@pytest.mark.parametrize("function", ["len", "count"])
def test_large_size_check_does_not_read_opaque_message_fields(tmp_path, function):
    policy, adapter = prepare(tmp_path, f'{function}(messages.transform("convert")) < 10000', True)
    for explain in [False, True]:
        result = policy.scan(
            {"messages": [{"role": "alien", "content": "opaque"}]}, explain=explain
        )
        assert result.is_complete and result.rule_matches == ["R"]
    assert adapter.parts == []


@pytest.mark.parametrize("function", ["len", "count"])
def test_prepared_scalar_size_does_not_need_the_source_parser(tmp_path, function, monkeypatch):
    from quarry.dsl.parser import DSLParser

    policy, _ = prepare(tmp_path, f'{function}(messages.transform("convert")) < 10000', False)
    data = policy.to_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared size loading must not parse or recompile a rule")

    monkeypatch.setattr(DSLParser, "parse", forbidden)
    scanner = policy._scanner
    loaded = load_compiled_policy(
        data, runtime_config=scanner.config, model_inference_backend=scanner.model_inference_backend
    )
    assert loaded.scan(PAYLOAD).rule_matches == ["R"]
