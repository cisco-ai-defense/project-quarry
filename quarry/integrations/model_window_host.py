# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Bounded admission of explicit host projections and opaque enclosing renderings."""

from __future__ import annotations

import asyncio
import inspect
from array import array
from bisect import bisect_right
from copy import deepcopy

from quarry.integrations.model_window_input import (
    WindowInput,
    join_window_input,
    message_window_origin,
)
from quarry.integrations.model_window_ranges import WindowRangeBudget, origin_intervals
from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.origins import WHOLE, Derived, Place, SourceText
from quarry.types.results import InputText

_MAX_GRAPH_WORK = 4_000_000
_MAX_GRAPH_NODES = 100_000


def _place_key(place):
    if (
        type(place.field) is not str
        or type(place.position) is not int
        or place.position < 0
        or type(place.start) is not int
        or place.start < 0
    ):
        raise ValueError("invalid host origin place")
    source = place.source
    if source is not None and (
        type(source) is not InputText
        or type(source.input) is not str
        or (source.item is not None and type(source.item) is not str)
        or type(source.fields) is not tuple
        or any(type(field) is not str for field in source.fields)
    ):
        raise ValueError("invalid host input owner")
    return id(place.owner), place.position, place.field, source


async def _allowed_ranges(messages):
    ranges = {}
    work = 0
    budget = WindowRangeBudget()
    for index, message in enumerate(messages):
        origin = await message_window_origin(messages, index, budget)
        for place, low, high in origin_intervals(origin, 0, len(message.content), budget):
            work += 1
            if work > _MAX_GRAPH_WORK:
                raise ValueError("window input origin exceeds its work limit")
            ranges.setdefault(_place_key(place), []).append((low, high))
            if work % 256 == 0:
                await asyncio.sleep(0)
                raise_if_evaluation_expired()
        if index % 256 == 0:
            await asyncio.sleep(0)
            raise_if_evaluation_expired()
    merged = {}
    for key, values in ranges.items():
        result = []
        for low, high in sorted(values):
            work += 1
            if work > _MAX_GRAPH_WORK:
                raise ValueError("window input origin exceeds its work limit")
            if result and low <= result[-1][1]:
                result[-1] = (result[-1][0], max(result[-1][1], high))
            else:
                result.append((low, high))
            if work % 256 == 0:
                await asyncio.sleep(0)
                raise_if_evaluation_expired()
        merged[key] = (tuple(low for low, _ in result), tuple(result))
    return merged


async def validate_host_input(value: WindowInput, messages):
    allowed = await _allowed_ranges(messages)
    root = value.source.origin
    stack = [(root, False, 0)]
    active, lengths = set(), {}
    nodes = work = 0
    while stack:
        node, finish, depth = stack.pop()
        identity = id(node)
        if not finish and identity in lengths:
            continue
        if not finish and identity in active:
            raise ValueError("host window origin contains a cycle")
        if depth > 100:
            raise ValueError("host window origin exceeds its depth limit")
        if not finish:
            nodes += 1
            if nodes + len(stack) > _MAX_GRAPH_NODES:
                raise ValueError("host window origin exceeds its node limit")
            if nodes % 256 == 0:
                await asyncio.sleep(0)
                raise_if_evaluation_expired()
        if type(node) is Place:
            entry = allowed.get(_place_key(node))
            if entry is None or node.start > entry[1][-1][1]:
                raise ValueError("host window origin names an unselected owner")
            lengths[identity] = entry[1][-1][1] - node.start
            continue
        if type(node) is not Derived:
            raise ValueError("host window origin must use closed Place/Derived nodes")
        if (
            type(node.text) is not str
            or type(node.sources) is not tuple
            or type(node.runs) is not array
            or node.runs.typecode != "q"
            or len(node.runs) % 6
        ):
            raise ValueError("invalid host window origin graph")
        if not finish:
            if nodes + len(stack) + len(node.sources) > _MAX_GRAPH_NODES:
                raise ValueError("host window origin exceeds its node limit")
            active.add(identity)
            stack.append((node, True, depth))
            stack.extend((source, False, depth + 1) for source in reversed(node.sources))
            continue
        previous_end = 0
        for offset in range(0, len(node.runs), 6):
            start, end, source, low, high, kind = node.runs[offset : offset + 6]
            work += 1
            if work > _MAX_GRAPH_WORK:
                raise ValueError("host window origin exceeds its work limit")
            if (
                not 0 <= start < end <= len(node.text)
                or start < previous_end
                or not 0 <= source < len(node.sources)
                or not 0 <= low <= high <= lengths[id(node.sources[source])]
                or kind not in (0, 1)
            ):
                raise ValueError("host window origin contains invalid runs")
            previous_end = end
            if work % 256 == 0:
                await asyncio.sleep(0)
                raise_if_evaluation_expired()
        active.remove(identity)
        lengths[identity] = len(node.text)
    if type(root) is Derived and root.text != value.source.text:
        raise ValueError("host window origin text differs from its input")
    if len(value.source.text) > lengths[id(root)]:
        raise ValueError("host window input is outside its origin")
    budget = WindowRangeBudget()
    for place, low, high in origin_intervals(root, 0, len(value.source.text), budget):
        key = _place_key(place)
        starts, intervals = allowed[key]
        index = bisect_right(starts, low) - 1
        if index < 0 or high > intervals[index][1]:
            raise ValueError("host window origin is outside the selected input")
        work += 1
        if work > _MAX_GRAPH_WORK:
            raise ValueError("host window origin exceeds its work limit")
        if work % 256 == 0:
            await asyncio.sleep(0)
            raise_if_evaluation_expired()
    raise_if_evaluation_expired()


async def render_host_input(adapter, config, messages, tools):
    raise_if_evaluation_expired()
    value = adapter.render_window_input(config, messages, deepcopy(tools))
    if inspect.isawaitable(value):
        value = await value
    raise_if_evaluation_expired()
    if type(value) is str:
        original = await join_window_input(messages, tools)
        runs = array("q")
        if value and original.source.text:
            runs.extend((0, len(value), 0, 0, len(original.source.text), WHOLE))
        origin = Derived(value, (original.source.origin,), runs)
        value = WindowInput(
            SourceText(value, origin, original.source.message_index), original.context
        )
    if type(value) is not WindowInput:
        raise ValueError("host window renderer must return WindowInput or an opaque string")
    if config.windows.scope == "last_turn" and value.last_turn is None:
        raise ValueError("host last-turn rendering requires an explicit interval")
    await validate_host_input(value, messages)
    return value
