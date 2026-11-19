# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Character cuts retain exact originals and conservative rendered projections."""

from array import array

import pytest

from quarry.config import ModelWindowsConfig
from quarry.integrations import model_windows
from quarry.integrations.model_windows import character_windows
from quarry.types.messages import Messages
from quarry.types.origins import WHOLE, Derived, Place, SourceText, joined


def located(text="0123456789"):
    message = Messages.from_openai([{"role": "user", "content": text}])[0]
    return SourceText(text, Place(0, "", 0, message), 0)


@pytest.mark.asyncio
async def test_window_local_offsets_map_through_the_full_originating_cut():
    whole = located()
    cut = SourceText("234567", Place(0, "", 2, whole.origin.owner), 0)
    windows = await character_windows(cut, ModelWindowsConfig("characters", 4, 2), max_windows=2)
    assert [window.text for window in windows] == ["2345", "4567"]
    assert all(window.source is cut for window in windows)
    first, second = windows[0].span(1, 3), windows[1].span()
    assert (first.message_index, first.start, first.end, first.precision) == (0, 3, 5, "exact")
    assert (second.message_index, second.start, second.end, second.precision) == (0, 4, 8, "exact")


@pytest.mark.asyncio
async def test_a_joined_window_maps_both_original_messages():
    messages = Messages.from_openai(
        [{"role": "user", "content": "abc"}, {"role": "assistant", "content": "def"}]
    )
    origin = joined(
        [("abc", Place(0, "", 0, messages[0])), ("\n", None), ("def", Place(1, "", 0, messages[1]))]
    )
    source = SourceText(origin.text, origin, 0)
    windows = await character_windows(source, ModelWindowsConfig("characters", 3, 2), max_windows=3)
    assert [window.text for window in windows] == ["abc", "c\nd", "def"]
    crossed = windows[1].span()
    assert (crossed.message_index, crossed.start, crossed.end_message_index, crossed.end) == (
        0,
        2,
        1,
        1,
    )
    last = windows[-1].span()
    assert (last.message_index, last.start, last.end) == (1, 0, 3)


@pytest.mark.asyncio
async def test_opaque_json_projection_stays_enclosing_after_a_window_cut():
    original = located('a"b')
    rendered = '{"content":"a\\"b"}'
    origin = Derived(
        rendered,
        (original.origin,),
        array("q", (0, len(rendered), 0, 0, len(original.text), WHOLE)),
    )
    source = SourceText(rendered, origin, 0)
    windows = await character_windows(source, ModelWindowsConfig("characters", 5), max_windows=10)
    span = windows[2].span(1, 3)
    assert (span.message_index, span.start, span.end, span.precision) == (0, 0, 3, "enclosing")
    assert windows[2].source is source


@pytest.mark.asyncio
async def test_last_turn_uses_the_full_rendered_input_coordinates():
    source = located("first\nlast")
    windows = await character_windows(
        source,
        ModelWindowsConfig("characters", 3, 2, "last_turn"),
        max_windows=2,
        last_turn=(6, 10),
    )
    assert [window.text for window in windows] == ["las", "st"]
    assert [(window.span().start, window.span().end) for window in windows] == [(6, 9), (8, 10)]


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", [None, (-1, 2), (2, 1), (0, 11), (True, 2)])
async def test_last_turn_requires_a_valid_host_range(selection):
    with pytest.raises(ValueError, match="last_turn"):
        await character_windows(
            located(),
            ModelWindowsConfig("characters", 3, scope="last_turn"),
            max_windows=10,
            last_turn=selection,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("start,end", [(-1, 1), (0, 4), (True, 1), (2, 1)])
async def test_decoded_window_ranges_cannot_escape_the_window(start, end):
    windows = await character_windows(located(), ModelWindowsConfig("characters", 3), max_windows=4)
    with pytest.raises(ValueError, match="within the window"):
        windows[0].span(start, end)


@pytest.mark.asyncio
async def test_empty_windows_and_bounded_overlapping_character_work(monkeypatch):
    assert (
        await character_windows(located(""), ModelWindowsConfig("characters", 3), max_windows=1)
        == ()
    )
    config = ModelWindowsConfig("characters", 4, 1)
    with pytest.raises(ValueError, match="count limit"):
        await character_windows(located(), config, max_windows=6)
    monkeypatch.setattr(model_windows, "MAX_WINDOW_CHARACTER_WORK", 27)
    with pytest.raises(ValueError, match="work limit"):
        await character_windows(located(), config, max_windows=7)
