# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Token windows preserve exact IDs and full-cut positions, including endpoints."""

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from quarry.config import ModelWindowsConfig
from quarry.integrations import model_windows
from quarry.integrations.model_windows import TokenEncoding, selected_token_scores, token_windows
from quarry.safety import timer
from quarry.safety.timer import QuarryTimeoutError, evaluation_deadline


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("suffix", [False, True])
@pytest.mark.parametrize("scope", ["all", "last_turn"])
async def test_exact_window_batches_preserve_full_origin_positions(prefix, suffix, scope):
    ids = ((100,) if prefix else ()) + (10, 11, 12, 13, 14, 15) + ((200,) if suffix else ())
    offset = int(prefix)
    encoding = TokenEncoding(ids, prefix, suffix, (offset + 3, offset + 6))
    windows = await token_windows(
        encoding, ModelWindowsConfig("tokens", 2, 1, scope), max_windows=10
    )
    expected = (
        [(10, 11), (11, 12), (12, 13), (13, 14), (14, 15)]
        if scope == "all"
        else [(13, 14), (14, 15)]
    )
    assert [w.token_ids for w in windows] == [
        ((100,) if prefix else ()) + tokens + ((200,) if suffix else ()) for tokens in expected
    ]
    expected_starts = range(offset if scope == "all" else offset + 3, offset + 5)
    assert [w.positions for w in windows] == [
        ((0,) if prefix else ()) + (start, start + 1) + ((len(ids) - 1,) if suffix else ())
        for start in expected_starts
    ]
    assert all(w.token_ids == tuple(ids[position] for position in w.positions) for w in windows)


@pytest.mark.asyncio
async def test_a_full_turn_does_not_duplicate_endpoints_or_deduplicate_equal_ids():
    encoding = TokenEncoding((7, 7, 7), True, True, (0, 3))
    windows = await token_windows(
        encoding, ModelWindowsConfig("tokens", 4, scope="last_turn"), max_windows=1
    )
    assert len(windows) == 1
    assert windows[0].token_ids == (7, 7, 7)
    assert windows[0].positions == (0, 1, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "encoding",
    [
        TokenEncoding(()),
        TokenEncoding((100, 200), True, True),
        TokenEncoding((100, 1, 200), True, True, (1, 1)),
    ],
)
async def test_empty_selected_content_has_no_batches(encoding):
    scope = "all" if encoding.last_turn is None else "last_turn"
    assert (
        await token_windows(encoding, ModelWindowsConfig("tokens", 8, scope=scope), max_windows=1)
        == ()
    )


def test_encoding_detaches_the_host_arrays_and_is_immutable():
    ids, turn = [100, 10, 11, 200], [1, 3]
    encoding = TokenEncoding(ids, True, True, turn)
    ids[1] = 99
    turn[0] = 2
    assert encoding.token_ids == (100, 10, 11, 200)
    assert encoding.last_turn == (1, 3)
    assert encoding.cache_lifetime == 300
    with pytest.raises(FrozenInstanceError):
        encoding.last_turn = (1, 2)


@pytest.mark.parametrize("length", [0, 8192])
def test_native_encoding_validation_honors_an_expired_evaluation(length, monkeypatch):
    with evaluation_deadline(1), monkeypatch.context() as clock:
        clock.setattr(timer.time, "monotonic", lambda: float("inf"))
        with pytest.raises(QuarryTimeoutError):
            TokenEncoding((1,) * length)


def test_large_native_encoding_stops_at_deadline_before_reaching_a_late_bad_token(monkeypatch):
    with evaluation_deadline(1) as deadline, monkeypatch.context() as clock:
        moments = iter((deadline.expires_at - 0.8, deadline.expires_at - 0.4))
        clock.setattr(timer.time, "monotonic", lambda: next(moments, deadline.expires_at))
        with pytest.raises(QuarryTimeoutError):
            TokenEncoding((1,) * 8191 + ("invalid",))


def test_optional_character_offsets_detach_and_preserve_overlapping_token_geometry():
    offsets = [None, [0, 2], [1, 3], None]
    encoded = TokenEncoding((100, 10, 11, 200), True, True, character_offsets=offsets)
    offsets[1][0] = 99
    assert encoded.character_offsets == (None, (0, 2), (1, 3), None)
    assert TokenEncoding((10, 11)).character_offsets is None


def test_zero_width_special_token_offsets_do_not_reverse_content_geometry():
    offsets = ((0, 0), (0, 1), (0, 0), (1, 2), (0, 0))
    value = TokenEncoding((100, 10, 150, 11, 200), True, True, character_offsets=offsets)
    assert value.character_offsets == offsets


@pytest.mark.parametrize(
    "offsets",
    [
        [],
        [None],
        [None, (0, 1), (1, 2)],
        [None, (True, 1), None, None],
        [None, (-1, 1), None, None],
        [None, (2, 1), None, None],
        [None, (2, 3), (1, 4), None],
        [(0, 1), (1, 2), (2, 3), None],
        [None, (0, 1), (1, 2), (2, 3)],
    ],
)
def test_invalid_or_inconsistent_token_character_offsets_are_rejected(offsets):
    with pytest.raises(ValueError, match="character_offsets"):
        TokenEncoding((100, 10, 11, 200), True, True, character_offsets=offsets)


@pytest.mark.parametrize("ids", [(True,), (-1,), (1.5,), ("1",), "123", None])
def test_invalid_token_ids_fail_before_cutting(ids):
    with pytest.raises(ValueError, match="encoding"):
        TokenEncoding(ids)


@pytest.mark.parametrize("turn", [(-1, 1), (1, 0), (0, 4), (True, 2), (0,), "12"])
def test_invalid_turn_ranges_fail_before_cutting(turn):
    with pytest.raises(ValueError, match="last_turn"):
        TokenEncoding((100, 1, 200), True, True, turn)


@pytest.mark.asyncio
async def test_a_last_turn_cut_requires_host_geometry():
    with pytest.raises(ValueError, match="last_turn"):
        await token_windows(
            TokenEncoding((1, 2)), ModelWindowsConfig("tokens", 2, scope="last_turn"), max_windows=1
        )


@pytest.mark.asyncio
async def test_overlapping_work_and_count_are_bounded_before_batch_allocation(monkeypatch):
    encoding = TokenEncoding((100, 1, 2, 3, 4, 5, 6, 200), True, True)
    config = ModelWindowsConfig("tokens", 2, 1)
    with pytest.raises(ValueError, match="count limit"):
        await token_windows(encoding, config, max_windows=4)
    monkeypatch.setattr(model_windows, "MAX_WINDOW_TOKEN_WORK", 19)
    with pytest.raises(ValueError, match="work limit"):
        await token_windows(encoding, config, max_windows=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [True, -0.1, 1.1, float("nan"), float("inf"), "0.9", 10**1000])
async def test_the_entire_selected_head_is_validated_even_after_an_early_positive(invalid):
    windows = await token_windows(
        TokenEncoding((1, 2, 3, 4)), ModelWindowsConfig("tokens", 2), max_windows=2
    )
    head = [[[0.9], [0.9]], [[0.9], [invalid]]]
    with pytest.raises(ValueError, match="probability"):
        selected_token_scores(head, windows, 0)


@pytest.mark.asyncio
async def test_selected_head_is_detached_without_reading_other_classes():
    windows = await token_windows(
        TokenEncoding((1, 2)), ModelWindowsConfig("tokens", 2), max_windows=1
    )
    head = [[[float("nan"), 0.8], [None, 1]]]
    scores = selected_token_scores(head, windows, 1)
    head[0][0][1] = 0
    assert scores == ((0.8, 1.0),)


@pytest.mark.asyncio
@pytest.mark.parametrize("head", [[], [[[0.9]]], [[[0.9], []]], [[[0.9], [0.9]], [[0.9]]]])
async def test_head_shape_must_match_every_window_and_token(head):
    windows = await token_windows(
        TokenEncoding((1, 2)), ModelWindowsConfig("tokens", 2), max_windows=1
    )
    with pytest.raises(ValueError, match="token head"):
        selected_token_scores(head, windows, 0)


@pytest.mark.asyncio
async def test_expired_cuts_and_heads_fail_before_returning_work(monkeypatch):
    windows = await token_windows(
        TokenEncoding((1, 2)), ModelWindowsConfig("tokens", 2), max_windows=1
    )
    with evaluation_deadline(1) as deadline, monkeypatch.context() as clock:
        clock.setattr(timer.time, "monotonic", lambda: deadline.expires_at + 1)
        with pytest.raises(QuarryTimeoutError):
            await token_windows(
                TokenEncoding((1, 2)), ModelWindowsConfig("tokens", 2), max_windows=1
            )
        with pytest.raises(QuarryTimeoutError):
            selected_token_scores([[[0.9], [0.9]]], windows, 0)


@pytest.mark.asyncio
async def test_large_overlapping_cuts_cooperate_with_cancellation():
    task = asyncio.create_task(
        token_windows(
            TokenEncoding(tuple(range(5000))), ModelWindowsConfig("tokens", 2, 1), max_windows=5000
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_large_selected_head_validation_cooperates_with_cancellation(monkeypatch):
    validate = getattr(model_windows, "selected_token_scores_async", None)
    assert callable(validate), "native head validation must support cooperative cancellation"
    encoding = TokenEncoding(tuple(range(5000)))
    windows = await token_windows(encoding, ModelWindowsConfig("tokens", 5000), max_windows=1)
    head = [[[0.9] for _ in range(5000)]]
    entered, count = asyncio.Event(), [0]
    original = model_windows.math.isfinite

    def observed(value):
        count[0] += 1
        if count[0] == 1024:
            entered.set()
        return original(value)

    monkeypatch.setattr(model_windows.math, "isfinite", observed)
    task = asyncio.create_task(validate(head, windows, 0))
    await entered.wait()
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
