# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Source range indexing preserves merges and avoids new hooks on extension values."""

import pytest

from quarry.types.transforms import ContentSourceRange, ProvenanceSegment, source_ranges_for


def source(start, end, part=None):
    return ContentSourceRange(0, part, start, end, start, end)


def mapped(values):
    return tuple(ProvenanceSegment(i, i + 1, (value,), "whole") for i, value in enumerate(values))


def test_merged_duplicate_ranges_keep_counts_of_current_values():
    a, b, c, extension = source(0, 2), source(0, 1), source(1, 2), source(2, 3)
    values = [a, b, c, extension, a]
    assert source_ranges_for(mapped(values), 0, len(values)) == (a, source(0, 3))


def test_distinct_native_range_deduplication_does_not_compare_every_earlier_range(monkeypatch):
    count, comparisons = 256, [0]
    original = ContentSourceRange.__eq__

    def observed(left, right):
        comparisons[0] += 1
        return original(left, right)

    monkeypatch.setattr(ContentSourceRange, "__eq__", observed)
    values = [source(0, 1, index) for index in range(count)]
    result = source_ranges_for(mapped(values), 0, count)
    assert result == tuple(values)
    assert comparisons[0] <= 2 * count, f"quadratic source comparisons: {comparisons[0]}"


@pytest.mark.parametrize("extension", ["range", "integer"])
def test_non_native_source_values_keep_the_original_equality_contract_without_hashing(extension):
    class HostRange(ContentSourceRange):
        __hash__ = None

    class HostInteger(int):
        def __hash__(self):
            raise AssertionError("source extension integers must not be hashed")

    native = source(0, 1)
    custom = (
        HostRange(0, None, 0, 1, 0, 1)
        if extension == "range"
        else ContentSourceRange(HostInteger(0), None, 0, 1, 0, 1)
    )
    result = source_ranges_for(mapped([native, custom]), 0, 2)
    expected = (native, custom) if extension == "range" else (native,)
    assert result == expected
