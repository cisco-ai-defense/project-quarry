# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Coverage is an exact set union; failed optional measurements cannot partially commit."""

import concurrent.futures
from dataclasses import FrozenInstanceError

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quarry.integrations.read_coverage import (
    CoverageDelta,
    CoverageLimitError,
    CoverageOwnerError,
    ReadCoverage,
    ReadInterval,
)

A, B = b"a" * 32, b"b" * 32


def counts(coverage):
    value = coverage.snapshot()
    return value.characters, value.text_tokens, value.buckets, value.intervals


def test_repeated_overlapping_and_adjacent_cuts_count_their_union():
    coverage = ReadCoverage()
    assert coverage.add([ReadInterval(A, "characters", 0, 4)]) == CoverageDelta(4, 0)
    assert coverage.add([ReadInterval(A, "characters", 2, 8)]) == CoverageDelta(4, 0)
    assert coverage.add([ReadInterval(A, "characters", 0, 8)]) == CoverageDelta(0, 0)
    coverage.add([ReadInterval(A, "characters", 8, 10)])
    assert counts(coverage) == (10, 0, 1, 1)
    assert coverage.add([ReadInterval(A, "characters", 14, 20)]) == CoverageDelta(6, 0)
    assert counts(coverage) == (16, 0, 1, 2)
    assert coverage.add([ReadInterval(A, "characters", 5, 18)]) == CoverageDelta(4, 0)
    assert counts(coverage) == (20, 0, 1, 1)


def test_distinct_inputs_and_measurement_units_never_share_offsets():
    coverage = ReadCoverage()
    coverage.add(
        [
            ReadInterval(A, "characters", 0, 10),
            ReadInterval(B, "characters", 0, 10),
            ReadInterval(A, "text_tokens", 0, 3),
            ReadInterval(B, "text_tokens", 0, 3),
        ]
    )
    assert counts(coverage) == (20, 6, 4, 4)
    with pytest.raises(FrozenInstanceError):
        coverage.snapshot().characters = 1


def test_empty_intervals_retain_no_identity_or_units():
    coverage = ReadCoverage()
    assert coverage.add([]) == CoverageDelta(0, 0)
    coverage.add([ReadInterval(A, "characters", 0, 0), ReadInterval(B, "text_tokens", 7, 7)])
    assert counts(coverage) == (0, 0, 0, 0)


@given(
    st.lists(
        st.tuples(st.integers(0, 3), st.booleans(), st.integers(0, 30), st.integers(0, 30)),
        max_size=40,
    )
)
def test_union_matches_an_independent_finite_set_reference(rows):
    intervals = []
    reference = set()
    for source, tokens, first, last in rows:
        begin, end = sorted([first, last])
        unit = "text_tokens" if tokens else "characters"
        key = bytes([source]) * 32
        intervals.append(ReadInterval(key, unit, begin, end))
        reference.update((key, unit, offset) for offset in range(begin, end))
    expected = tuple(
        sum(item[1] == unit for item in reference) for unit in ["characters", "text_tokens"]
    )
    for values in [intervals, list(reversed(intervals))]:
        coverage = ReadCoverage(max_work=1_000_000)
        delta = coverage.add(values)
        assert (delta.characters, delta.text_tokens) == expected
        assert counts(coverage)[:2] == expected
        assert coverage.add(values) == CoverageDelta(0, 0)


@pytest.mark.parametrize("kind", ["keys", "intervals", "work"])
def test_exhausted_batch_never_leaves_partially_validated_totals(kind):
    coverage = ReadCoverage(
        max_buckets=1 if kind == "keys" else 20,
        max_intervals=2 if kind == "intervals" else 20,
        max_work=10 if kind == "work" else 100_000,
    )
    coverage.add([ReadInterval(A, "characters", 0, 2)])
    before = counts(coverage)
    if kind == "keys":
        batch = [ReadInterval(A, "characters", 1, 4), ReadInterval(B, "characters", 0, 2)]
    elif kind == "intervals":
        batch = [ReadInterval(A, "characters", 4, 6), ReadInterval(A, "characters", 8, 10)]
    else:
        batch = [ReadInterval(A, "characters", 2, 3)]
    with pytest.raises(CoverageLimitError):
        coverage.add(batch)
    assert counts(coverage) == before


def test_forged_or_invalid_measurement_does_not_commit_an_earlier_valid_range():
    coverage = ReadCoverage()
    forged = object.__new__(ReadInterval)
    object.__setattr__(forged, "source", A)
    object.__setattr__(forged, "unit", "characters")
    object.__setattr__(forged, "start", 4)
    object.__setattr__(forged, "end", -1)
    with pytest.raises(ValueError, match="invalid logical input coverage interval"):
        coverage.add([ReadInterval(A, "characters", 0, 2), forged])
    assert counts(coverage) == (0, 0, 0, 0)


def test_incomplete_forged_measurement_has_the_same_fixed_error():
    coverage = ReadCoverage()
    with pytest.raises(ValueError, match="^invalid logical input coverage interval$"):
        coverage.add([object.__new__(ReadInterval)])
    assert counts(coverage) == (0, 0, 0, 0)


def test_interval_repr_keeps_opaque_input_identity_private():
    identifier = b"sensitive-identity".ljust(32, b"x")
    assert "sensitive" not in repr(ReadInterval(identifier, "characters", 0, 1))


@pytest.mark.parametrize("name", ["max_buckets", "max_intervals", "max_work"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, 1 << 64])
def test_coverage_budget_configuration_requires_bounded_exact_positive_integers(name, value):
    with pytest.raises(ValueError, match="^invalid logical input coverage limit$"):
        ReadCoverage(**{name: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", b"short"),
        ("source", "secret source value"),
        ("unit", "secret unit value"),
        ("start", True),
        ("end", False),
        ("start", -1),
        ("end", 1 << 64),
        ("end", 0.5),
    ],
)
def test_invalid_range_errors_are_fixed_and_text_free(field, value):
    fields = {"source": A, "unit": "characters", "start": 0, "end": 5, field: value}
    with pytest.raises(ValueError) as error:
        ReadInterval(**fields)
    assert str(error.value) == "invalid logical input coverage interval"


def test_exact_native_metadata_validation_does_not_invoke_host_hooks():
    class String(str):
        def __eq__(self, other):
            raise AssertionError("untrusted equality hook")

    class Items(list):
        def __iter__(self):
            raise AssertionError("untrusted iteration hook")

    with pytest.raises(ValueError):
        ReadInterval(A, String("characters"), 0, 1)
    with pytest.raises(ValueError):
        ReadCoverage().add(Items([ReadInterval(A, "characters", 0, 1)]))


def test_atomic_deltas_remain_exact_across_concurrent_readers():
    coverage = ReadCoverage(max_work=1_000_000)

    def observe(index):
        return coverage.add([ReadInterval(A, "characters", index, index + 12)])

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        deltas = list(pool.map(observe, range(64)))
    assert sum(delta.characters for delta in deltas) == 75
    assert counts(coverage) == (75, 0, 1, 1)


def test_foreign_process_is_rejected_before_inherited_lock_is_touched():
    coverage = ReadCoverage()

    class Held:
        def __enter__(self):
            raise AssertionError("inherited lock touched")

        def __exit__(self, *args):
            return False

    coverage._pid = -1
    coverage._lock = Held()
    for operation in [coverage.snapshot, coverage.clear, lambda: coverage.add([])]:
        with pytest.raises(CoverageOwnerError):
            operation()


def test_clear_releases_all_metadata_and_work_for_the_next_lifetime():
    coverage = ReadCoverage()
    coverage.add([ReadInterval(A, "characters", 0, 4), ReadInterval(A, "text_tokens", 0, 2)])
    coverage.clear()
    assert counts(coverage) == (0, 0, 0, 0)
    assert coverage.snapshot().work == 0
    assert coverage.add([ReadInterval(B, "characters", 7, 11)]) == CoverageDelta(4, 0)
