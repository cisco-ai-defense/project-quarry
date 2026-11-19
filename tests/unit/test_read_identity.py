# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Opaque text lineage is a weakly owned coordinate system, not copied input storage."""

import concurrent.futures
import gc
import subprocess
import sys
import weakref

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quarry.integrations.read_coverage import ReadCoverage
from quarry.integrations.read_identity import (
    ReadIdentityLimitError,
    ReadIdentityOwnerError,
    ReadIdentityUnavailable,
    ReadTextRegistry,
)


class Owner:
    pass


def test_equal_text_from_distinct_occurrences_stays_separate_but_preserving_copies_share():
    index = ReadTextRegistry()
    one, two, clone = Owner(), Owner(), Owner()
    first, second = index.observe(one, "same"), index.observe(two, "same")
    index.copied(one, clone)
    assert first == index.observe(clone, "same")
    assert first.source != second.source


def test_copy_before_first_read_preserves_occurrence_and_does_not_retain_original():
    index = ReadTextRegistry()
    first, second = Owner(), Owner()
    ref = weakref.ref(first)
    index.copied(first, second)
    interval = index.observe(first, "synthetic")
    del first
    gc.collect()
    assert ref() is None
    assert index.observe(second, "synthetic") == interval


def test_transformed_variants_share_copies_but_keep_their_own_text_axis():
    index = ReadTextRegistry()
    original, folded, same, other = (Owner() for _ in range(4))
    raw = index.observe(original, "ß")
    index.derived(original, folded, "ss")
    index.derived(original, same, "ss")
    index.derived(original, other, "sS")
    assert index.observe(folded, "ss") == index.observe(same, "ss")
    assert index.observe(other, "sS").source != index.observe(folded, "ss").source != raw.source
    assert index.observe(folded, "ss").end == 2


def test_nested_slices_and_repeated_observations_keep_parent_coordinates():
    index = ReadTextRegistry()
    root, first, nested, overlap = (Owner() for _ in range(4))
    text = "0123456789"
    whole = index.observe(root, text)
    index.sliced(root, first, 2, 8, text)
    index.sliced(first, nested, 1, 4, text[2:8])
    index.sliced(root, overlap, 4, 9, text)
    small = index.observe(nested, "345")
    assert small.source == whole.source and (small.start, small.end) == (3, 6)
    coverage = ReadCoverage()
    coverage.add([small, index.observe(overlap, "45678"), small])
    assert coverage.snapshot().characters == 6


def test_equal_transformed_text_from_distinct_source_windows_keeps_distinct_identity():
    index = ReadTextRegistry()
    root, left, right, one, two = (Owner() for _ in range(5))
    index.sliced(root, left, 0, 4, "aaaaaaaa")
    index.sliced(root, right, 4, 8, "aaaaaaaa")
    index.derived(left, one, "Xaaaa")
    index.derived(right, two, "Xaaaa")
    assert index.observe(one, "Xaaaa").source != index.observe(two, "Xaaaa").source


def test_equivalent_direct_and_nested_windows_share_the_same_transformed_identity():
    index = ReadTextRegistry()
    root, direct, outer, inner, one, two = (Owner() for _ in range(6))
    index.sliced(root, direct, 2, 6, "0123456789")
    index.sliced(root, outer, 1, 9, "0123456789")
    index.sliced(outer, inner, 1, 5, "12345678")
    index.derived(direct, one, "X2345")
    index.derived(inner, two, "X2345")
    assert index.observe(one, "X2345") == index.observe(two, "X2345")


def test_whole_window_and_original_share_the_same_transformed_identity():
    index = ReadTextRegistry()
    root, whole, one, two = (Owner() for _ in range(4))
    index.sliced(root, whole, 0, 3, "abc")
    index.derived(root, one, "Xabc")
    index.derived(whole, two, "Xabc")
    assert index.observe(one, "Xabc") == index.observe(two, "Xabc")


@given(st.lists(st.tuples(st.integers(0, 40), st.integers(0, 40)), max_size=30))
def test_native_slice_axis_unions_match_an_independent_character_set(ranges):
    index = ReadTextRegistry()
    root = Owner()
    text = "0123456789" * 4
    owners, intervals, expected = [], [], set()
    for first, last in ranges:
        low, high = sorted((first, last))
        owner = Owner()
        owners.append(owner)
        index.sliced(root, owner, low, high, text)
        intervals.append(index.observe(owner, text[low:high]))
        expected.update(range(low, high))
    coverage = ReadCoverage(max_work=1_000_000)
    coverage.add(intervals)
    assert coverage.snapshot().characters == len(expected)


def test_concatenated_parts_share_character_axis_without_changing_unicode_or_surrogates():
    index = ReadTextRegistry()
    owner = Owner()
    assert index.observe_parts(owner, ("😀", "e\u0301", "\ud800")) == index.observe(
        owner, "😀e\u0301\ud800"
    )
    assert index.observe(owner, "😀e\u0301\ud800").end == 4


def test_changed_visible_text_does_not_rewrite_an_older_immutable_interval():
    index = ReadTextRegistry()
    owner = Owner()
    first = index.observe(owner, "old")
    assert index.observe(owner, "new").source != first.source
    assert index.observe(owner, "old") == first


def test_exported_frozen_interval_cannot_rewrite_later_measurements_or_copies():
    index = ReadTextRegistry()
    owner, clone = Owner(), Owner()
    exported = index.observe(owner, "abc")
    source = exported.source
    index.copied(owner, clone)
    object.__setattr__(exported, "end", 100)
    assert index.observe(owner, "abc").end == 3
    assert index.observe_parts(clone, ("a", "bc")).end == 3
    assert index.observe(clone, "abc").source == source


def test_unknown_rewrite_cannot_masquerade_as_a_validated_independent_input():
    index = ReadTextRegistry()
    root, rewritten, child = Owner(), Owner(), Owner()
    index.unavailable(rewritten)
    with pytest.raises(ReadIdentityUnavailable):
        index.observe(rewritten, "synthetic")
    index.copied(rewritten, child)
    with pytest.raises(ReadIdentityUnavailable):
        index.observe(child, "synthetic")
    index.derived(root, rewritten, "synthetic")
    assert index.observe(rewritten, "synthetic").end == 9


def test_live_index_keeps_neither_inputs_nor_owners_and_is_itself_collectable():
    index = ReadTextRegistry()
    owner = Owner()
    ref, index_ref = weakref.ref(owner), weakref.ref(index)
    index.observe(owner, "sensitive input")
    del owner
    gc.collect()
    assert ref() is None and index.snapshot().owners == 0
    owner = Owner()
    index.observe(owner, "other input")
    del index
    gc.collect()
    assert index_ref() is None


def test_gc_triggered_by_identity_allocation_can_reenter_weak_cleanup_without_deadlock():
    code = """
import gc
from quarry.integrations import read_identity

class Owner:
    pass

gc.disable()
index = read_identity.ReadTextRegistry()
old = Owner()
old.cycle = old
index.observe(old, "synthetic old")
del old
original = read_identity.secrets.token_bytes
def nonce(size):
    gc.collect()
    return original(size)
read_identity.secrets.token_bytes = nonce
new = Owner()
assert index.observe(new, "synthetic new").end == 13
assert index.snapshot().owners == 1
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0, result.stderr


def test_reclaimed_owner_reservations_remain_charged_and_cannot_recycle_an_identity():
    index = ReadTextRegistry(max_owners=2)
    owner = Owner()
    first = index.observe(owner, "same")
    del owner
    gc.collect()
    owner = Owner()
    assert index.observe(owner, "same").source != first.source
    with pytest.raises(ReadIdentityLimitError):
        index.observe(Owner(), "same")
    with pytest.raises(ReadIdentityLimitError):
        index.observe(owner, "same")
    assert index.snapshot().reservations == 2


def test_work_exhaustion_precedes_text_encoding_and_disables_later_partial_admission():
    class Text(str):
        def encode(self, *args, **kwargs):
            raise AssertionError("host encoding hook")

    index = ReadTextRegistry(max_work=10)
    owner = Owner()
    with pytest.raises(ValueError, match="^logical input identity needs exact text$"):
        index.observe(owner, Text("private"))
    with pytest.raises(ReadIdentityLimitError):
        index.observe(owner, "x" * 1000)
    with pytest.raises(ReadIdentityLimitError):
        index.observe(owner, "x")
    assert index.snapshot().work == 10


def test_nonweak_owners_do_not_enter_a_global_strong_retention_fallback():
    index = ReadTextRegistry()
    with pytest.raises(
        ReadIdentityUnavailable, match="^logical input identity needs a weak owner$"
    ):
        index.observe([], "private")
    assert index.snapshot().owners == index.snapshot().reservations == 0


def test_identity_never_hashes_compares_or_reads_host_owner_attributes():
    class Host:
        def __hash__(self):
            raise AssertionError("host hash")

        def __eq__(self, _other):
            raise AssertionError("host equality")

        def __getattribute__(self, _name):
            raise AssertionError("host attribute")

    index = ReadTextRegistry()
    one, two = Host(), Host()
    first = index.observe(one, "synthetic")
    index.copied(one, two)
    assert index.observe(two, "synthetic") == first


def test_concurrent_observations_of_one_occurrence_have_one_stable_axis():
    index = ReadTextRegistry()
    owner = Owner()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        intervals = list(pool.map(lambda _: index.observe(owner, "synthetic"), range(64)))
    assert len({value.source for value in intervals}) == 1
    assert index.snapshot().owners == index.snapshot().reservations == 1


def test_foreign_pid_is_rejected_before_inherited_registry_lock():
    class Held:
        def __enter__(self):
            raise AssertionError("inherited identity lock touched")

        def __exit__(self, *args):
            return False

    index = ReadTextRegistry()
    owner = Owner()
    index._pid = -1
    index._lock = Held()
    for operation in [
        lambda: index.observe(owner, "x"),
        lambda: index.copied(owner, owner),
        lambda: index.inspect(1),
        index.snapshot,
    ]:
        with pytest.raises(ReadIdentityOwnerError):
            operation()
    index.close()


def test_close_drops_weak_entries_and_irreversibly_seals_the_axis():
    index = ReadTextRegistry()
    owner = Owner()
    index.observe(owner, "private")
    index.close()
    index.close()
    assert index.snapshot().closed and index.snapshot().owners == 0
    with pytest.raises(ReadIdentityOwnerError, match="^logical input identity is closed$"):
        index.observe(owner, "private")


@pytest.mark.parametrize("field", ["max_owners", "max_work"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, 1 << 64])
def test_limits_require_bounded_exact_positive_integers(field, value):
    with pytest.raises(ValueError, match="^invalid logical input identity limit$"):
        ReadTextRegistry(**{field: value})


@pytest.mark.parametrize("start,end", [(True, 1), (0, False), (-1, 1), (2, 1), (0, 4)])
def test_invalid_slice_metadata_is_text_free_and_keeps_inputs_unregistered(start, end):
    index = ReadTextRegistry()
    one, two = Owner(), Owner()
    with pytest.raises(ValueError, match="^invalid logical input identity slice$"):
        index.sliced(one, two, start, end, "abc")
    assert index.snapshot().owners == 0
