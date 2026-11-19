# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Native lineage must not change input fields, decision storage or optional failures."""

import gc
import weakref
from pathlib import Path

import pytest

from quarry.cache.message_values import _retained_bytes
from quarry.integrations.content_transforms import apply_content_transform
from quarry.integrations.read_coverage import ReadCoverage
from quarry.integrations.read_usage import ReadUsageLimits, model_read_usage, resume_read_usage
from quarry.types.messages import Messages
from quarry.types.read_observations import _text_read_observer
from quarry.types.transforms import ContentTransform, TransformOperation
from tests.engine.test_async_runtime import _quarry_imports


def messages(text="0123456789"):
    return Messages.from_openai([{"role": "user", "content": text}])


def measured(session, message):
    result = session.text_measurement(message, message.source_content)
    assert result.state == "measured"
    return result.intervals[0]


def test_native_split_nested_split_and_preserving_copy_share_exact_parent_coordinates():
    root = messages()
    with model_read_usage() as session:
        first = root._select_ranges((0,), {0: (2, 8)})
        nested = first._select_ranges((0,), {0: (1, 4)})
        second = root._select_ranges((0,), {0: (4, 9)})
        copied = nested[0].model_copy()
        interval = measured(session, copied)
        assert (interval.start, interval.end) == (3, 6)
        assert interval.source == measured(session, root[0]).source
        coverage = ReadCoverage()
        coverage.add([interval, measured(session, second[0]), measured(session, nested[0])])
        assert coverage.snapshot().characters == 6


async def test_transformed_full_variants_and_their_slices_keep_separate_character_axes():
    root = messages("Straße")
    plan = ContentTransform((TransformOperation("casefold"),))
    with model_read_usage() as session:
        one = await apply_content_transform(root, plan)
        two = await apply_content_transform(root, plan)
        raw = measured(session, root[0])
        folded = measured(session, one[0])
        assert folded == measured(session, two[0]) and folded.source != raw.source
        assert (raw.end, folded.end) == (6, 7)
        left = one._select_ranges((0,), {0: (1, 5)})
        right = two._select_ranges((0,), {0: (3, 7)})
        coverage = ReadCoverage()
        coverage.add([measured(session, left[0]), measured(session, right[0])])
        assert coverage.snapshot().characters == 6


async def test_authored_inserted_text_is_counted_as_part_of_the_input_variant():
    root = messages("abc")
    with model_read_usage() as session:
        result = await apply_content_transform(
            root, ContentTransform((TransformOperation("prefix", "XX"),))
        )
        interval = measured(session, result[0])
        assert interval.end - interval.start == 5
        cut = result._select_ranges((0,), {0: (0, 3)})
        assert measured(session, cut[0]).source == interval.source
        assert measured(session, cut[0]).end == 3


def test_role_and_part_layout_rewrites_preserve_character_identity():
    root = messages("synthetic")
    with model_read_usage() as session:
        first = measured(session, root[0])
        rewritten = root.rewritten(
            [{"role": "assistant", "content": [{"type": "text", "text": "synthetic"}]}]
        )
        assert rewritten[0].role == "assistant"
        assert measured(session, rewritten[0]) == first


def test_reordered_repeated_and_grafted_messages_preserve_occurrences():
    root = Messages.from_openai(
        [{"role": "user", "content": "same"}, {"role": "assistant", "content": "same"}]
    )
    with model_read_usage() as session:
        first, second = (measured(session, message) for message in root)
        assert first.source != second.source
        combined = Messages.combine(
            (root[1:], root[:1], root[:1], Messages().add_message("user", "new"))
        )
        assert measured(session, combined[0]) == second
        assert measured(session, combined[1]) == measured(session, combined[2]) == first
        assert measured(session, combined[3]).source not in {first.source, second.source}


async def test_multimodal_metadata_walk_counts_only_text_without_filling_content_caches():
    root = Messages.from_openai(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "ab"},
                    {"type": "image_url", "image_url": {"url": "data:synthetic"}},
                    {"type": "text", "text": "cd"},
                ],
            }
        ]
    )
    owner = root[0]
    before = await _retained_bytes(root, 10**9)
    private_before = dict(owner.__pydantic_private__)
    with model_read_usage() as session:
        assert measured(session, owner).end == 4
        assert await _retained_bytes(root, 10**9) == before
        assert owner.__pydantic_private__ == private_before


async def test_live_metadata_does_not_retain_finished_native_values_or_their_text():
    with model_read_usage() as session:
        root = messages("sensitive synthetic input")
        one = root[0]
        two = one.model_copy()
        references = [weakref.ref(one), weakref.ref(two)]
        interval = measured(session, one)
        assert measured(session, two) == interval
        del root, one, two
        gc.collect()
        assert all(reference() is None for reference in references)
        assert session._text_inputs.registry.snapshot().owners == 0
        assert not session.snapshot()["finalized"]


def test_unknown_host_content_update_is_unavailable_until_a_native_derivation_supplies_a_map():
    root = messages("old")
    with model_read_usage() as session:
        changed = root[0].model_copy(update={"source_content": "new"})
        assert changed.content == "new"
        result = session.text_measurement(changed, changed.source_content)
        assert result.state == "unavailable" and not result.intervals


def test_optional_metadata_failure_never_changes_a_native_copy_or_retains_error_text(monkeypatch):
    root = messages("synthetic")
    with model_read_usage() as session:

        def fail(*_args):
            raise RuntimeError("sensitive synthetic diagnostic")

        monkeypatch.setattr(session._text_inputs.registry, "copied", fail)
        copied = root[0].model_copy()
        assert copied.content == "synthetic"
        result = session.text_measurement(copied, copied.source_content)
        assert result.state == "failed" and result.intervals == ()
        assert session._text_inputs._failure == "failed"


def test_unknown_host_update_mappings_are_not_walked_by_accounting():
    class Host:
        def __len__(self):
            raise AssertionError("host length")

        def __iter__(self):
            raise AssertionError("host iteration")

    root, target = messages(), messages()
    with model_read_usage() as session:
        session._text_inputs.copy_updated(root[0], target[0], Host())
        assert session.text_measurement(target[0], target[0].source_content).state == "unavailable"


@pytest.mark.parametrize("limit", ["owners", "work"])
async def test_optional_identity_exhaustion_never_changes_transform_or_slice_results(limit):
    root = messages("abcdef")
    limits = ReadUsageLimits(**{f"max_input_{limit}": 1})
    with model_read_usage(limits) as session:
        copied = root[0].model_copy()
        clipped = root._select_ranges((0,), {0: (1, 4)})
        transformed = await apply_content_transform(
            root, ContentTransform((TransformOperation("prefix", "XX"),))
        )
        assert copied.content == "abcdef"
        assert clipped[0].content == "bcd" and transformed[0].content == "XXabcdef"
        result = session.text_measurement(clipped[0], clipped[0].source_content)
        assert result.state == "limit_exceeded" and not result.intervals


def test_reactivating_owners_restores_the_native_observer_without_borrowing_foreign_state():
    with model_read_usage() as source:
        lease = source.retain()
        original = _text_read_observer.get()
    with model_read_usage() as current:
        foreign = _text_read_observer.get()
        with resume_read_usage(lease):
            assert _text_read_observer.get() is original
        assert _text_read_observer.get() is foreign
        with resume_read_usage(None):
            assert _text_read_observer.get() is None
        assert _text_read_observer.get() is foreign
    lease.close()
    assert source._closed and current._closed and _text_read_observer.get() is None


def test_native_observation_contract_remains_pure():
    import quarry.types.read_observations as contract

    assert _quarry_imports(Path(contract.__file__)) == set()
