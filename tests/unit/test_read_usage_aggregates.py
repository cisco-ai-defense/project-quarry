# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Cross-operation unions with independent finite-set and native backend controls."""

import gc
import weakref

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quarry.integrations import read_usage
from quarry.integrations.read_operations import model_read, record_model_input
from quarry.integrations.read_tokens import TextTokenBasis
from quarry.integrations.read_usage import (
    ReadUsageLimits,
    capture_read_usage,
    model_read_usage,
    read_usage_scope,
)
from tests.unit.test_model_read_operations import backend, messages
from tests.unit.test_read_usage import finish, measure

BASIS = TextTokenBasis("synthetic-pairs", "1")
PAIRS = ((0, 2), (2, 4), (4, 6))


def totals(session):
    return session.report()["aggregates"]["totals"]


def value(group, unit="characters"):
    return group["coverage"][unit]["value"]


def observe(source, *, model="reader", operation="classify"):
    with model_read(model, operation):
        record_model_input(source)


async def test_native_cross_operation_union_preserves_provider_and_tokenizer_work(tmp_path):
    service, adapter = backend(tmp_path)
    source = messages("abcdef")
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        assert session.provide_text_tokens("reader", "abcdef", PAIRS)
        assert session.provide_text_tokens("reader", "p", ((0, 1),))
        await service.model_score_messages("reader", "p", source)
        await service.model_classify_messages("reader", source)
        result = totals(session)
        assert value(result) == 7
        assert value(result, "text_tokens") == 4
        assert result["operations"] == {"classify": 1, "score": 1}
        assert result["observations"] == result["evaluations"] == 2
        assert session.report()["aggregates"]["complete"]
    assert adapter.score_calls == 1 and adapter.tokenizations == 0
    assert totals(session) == result


def test_memo_readers_are_separately_attributed_but_model_input_is_unioned_once():
    with model_read_usage() as session:
        with read_usage_scope("first"), capture_read_usage() as captured:
            finish(session, chars=measure(end=6), tokens=measure("text_tokens", end=3))
        with read_usage_scope("second"):
            captured.result().replay()
            captured.result().replay()
        report = session.report()["aggregates"]
    assert [row["rule"] for row in report["by_reader"]] == ["first", "second"]
    assert [value(row) for row in report["by_reader"]] == [6, 6]
    assert value(report["totals"]) == 6
    assert report["totals"]["observations"] == 2
    assert report["totals"]["evaluations"] == 1


def test_conversation_scopes_add_even_when_caller_reuses_the_same_source_axis():
    with model_read_usage() as session:
        scopes = [session.new_scope(), session.new_scope()]
        for scope in scopes:
            with read_usage_scope("reader", scope=scope):
                finish(session, chars=measure(end=6), tokens=measure("text_tokens", end=3))
        assert value(totals(session)) == 12
        for scope in scopes:
            assert value(session.scope_report(scope)["aggregates"]["totals"]) == 6
        assert totals(session)["evaluations"] == 2


@pytest.mark.parametrize("same_basis", [True, False])
def test_token_basis_compatibility_is_explicit_across_aliases(same_basis):
    other = BASIS if same_basis else TextTokenBasis("synthetic-whole", "2")
    source = messages("abcdef")
    with model_read_usage(token_bases={"reader": BASIS, "other": other}) as session:
        assert session.provide_text_tokens("reader", "abcdef", PAIRS)
        assert session.provide_text_tokens("other", "abcdef", PAIRS if same_basis else ((0, 6),))
        with read_usage_scope("first"):
            observe(source)
        with read_usage_scope("second", phase="evidence"):
            observe(source, model="other", operation="complete")
        result = totals(session)
    assert value(result) == 6
    token = result["coverage"]["text_tokens"]
    assert token["basis_compatible"] is same_basis
    assert token["complete"] is same_basis
    assert token["value"] == (3 if same_basis else None)
    assert token["known_units"] == (3 if same_basis else None)
    assert sorted(group["coverage"]["value"] for group in result["token_bases"]) == (
        [3] if same_basis else [1, 3]
    )


@given(
    st.lists(
        st.tuples(st.integers(0, 7), st.integers(1, 8)).filter(lambda pair: pair[0] < pair[1]),
        min_size=1,
        max_size=15,
    )
)
def test_cross_operation_coverage_matches_independent_finite_character_sets(cuts):
    expected = set()
    with model_read_usage() as session:
        for index, (start, end) in enumerate(cuts):
            ticket = session.begin("reader", "score" if index % 2 else "classify")
            assert ticket.finish("succeeded", characters=measure(start=start, end=end))
            expected.update(range(start, end))
        assert value(totals(session)) == len(expected)


@pytest.mark.parametrize("outcome", ["succeeded", "refused", "failed", "cancelled", "not_sent"])
def test_outcome_summary_keeps_measurement_availability_independent(outcome):
    with model_read_usage() as session:
        finish(session, outcome=outcome)
        result = totals(session)
    assert result["outcomes"] == {outcome: 1}
    assert result["observations_complete"]
    assert not result["measurements_complete"]
    assert value(result) is None
    assert result["coverage"]["characters"]["known_units"] == 0


@pytest.mark.parametrize(
    "limit",
    ["max_aggregate_groups", "max_aggregate_keys", "max_aggregate_ranges", "max_aggregate_work"],
)
def test_summary_budget_cannot_change_detail_counts_or_validate_zero(limit):
    with model_read_usage(ReadUsageLimits(**{limit: 1})) as session:
        finish(session, chars=measure(end=6), tokens=measure("text_tokens", end=3))
        finish(session, chars=measure(start=10, end=16), tokens=measure("text_tokens", end=4))
        report = session.report()
        detail = session.snapshot()
    assert value(detail["rows"][0]) == 12
    assert {k: v for k, v in report.items() if k != "aggregates"} == detail
    assert not report["aggregates"]["complete"]
    assert report["aggregates"]["coverage_limited"]
    assert value(report["aggregates"]["totals"]) is None


def test_missing_scope_never_leaks_other_scan_data_or_becomes_validated_zero():
    with model_read_usage(token_bases={"secret-model": BASIS}) as session:
        empty = session.new_scope()
        with read_usage_scope("private-rule"):
            finish(session, model="secret-model", chars=measure())
        for scope in [None, True, -1, empty + 1]:
            report = session.scope_report(scope)
            assert report["state"] == "scope_unavailable"
            assert not report["rows"] and "token_bases" not in report
            assert value(report["aggregates"]["totals"]) is None
        report = session.scope_report(empty)
        assert report["rows"] == report["token_bases"] == []
        assert report["aggregates"]["complete"]
        assert value(report["aggregates"]["totals"]) == 0


def test_rich_reports_own_their_containers_and_retain_no_native_input():
    with model_read_usage(token_bases={"reader": BASIS}) as session:
        source = messages("abcdef")
        owner = weakref.ref(source[0])
        assert session.provide_text_tokens("reader", "abcdef", PAIRS)
        observe(source)
        del source
        gc.collect()
        assert owner() is None
        report = session.report()
        report["aggregates"]["by_reader"].clear()
        assert len(session.report()["aggregates"]["by_reader"]) == 1
    report = session.report()
    report["aggregates"]["totals"]["coverage"]["characters"]["value"] = 999
    assert value(totals(session)) == 6
    assert not session._events and not session._bindings


def test_pending_and_abandoned_reads_keep_aggregate_observations_incomplete():
    with model_read_usage() as session:
        assert session.begin("reader", "classify") is not None
        pending = totals(session)
        assert pending["outcomes"] == {"pending": 1}
        assert not pending["observations_complete"]
        assert value(pending) is None
    abandoned = totals(session)
    assert abandoned["outcomes"] == {"unknown": 1}
    assert not abandoned["observations_complete"]
    assert abandoned["coverage"]["characters"]["measurement_states"] == {"abandoned": 1}


def test_scope_omission_prevents_an_empty_aggregate_from_claiming_complete_zero():
    with model_read_usage(ReadUsageLimits(max_scopes=1)) as session:
        scope = session.new_scope()
        assert session.new_scope() is None
        report = session.scope_report(scope)
    assert not report["aggregates"]["complete"]
    assert value(report["aggregates"]["totals"]) is None
    assert report["omitted_observations"] == 1


def test_foreign_process_reports_before_touching_inherited_locks(monkeypatch):
    class Poison:
        def __enter__(self):
            pytest.fail("inherited lock touched")

    with model_read_usage() as session:
        scope = session.new_scope()
    session._lock = Poison()
    monkeypatch.setattr(read_usage.os, "getpid", lambda: session._pid + 1)
    report = session.scope_report(scope)
    assert report["state"] == "foreign_process"
    assert value(report["aggregates"]["totals"]) is None
