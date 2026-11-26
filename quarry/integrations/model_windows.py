# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Bounded, immutable token cuts that keep full-encoding coordinates for hosts."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import ClassVar

from quarry.config import ModelWindowsConfig
from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.origins import SourceText
from quarry.types.results import Span

MAX_ENCODING_TOKENS = 1_000_000
MAX_WINDOW_TOKEN_WORK = 4_000_000
MAX_WINDOW_CHARACTER_WORK = 4_000_000
MAX_MODEL_WINDOWS = 10_000


@dataclass(frozen=True, slots=True)
class CharacterWindow:
    """A local character range that keeps its full originating rendered input."""

    source: SourceText
    start: int
    end: int

    @property
    def text(self) -> str:
        return self.source.text[self.start : self.end]

    def span(self, start: int = 0, end: int | None = None) -> Span:
        end = self.end - self.start if end is None else end
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= self.end - self.start
        ):
            raise ValueError("decoded window range must lie within the window")
        return self.source.span(self.start + start, self.start + end)


async def character_windows(
    source: SourceText,
    config: ModelWindowsConfig,
    *,
    max_windows: int,
    last_turn: tuple[int, int] | None = None,
) -> tuple[CharacterWindow, ...]:
    """Cut rendered characters without replacing their full source projection."""
    raise_if_evaluation_expired()
    if config.unit != "characters":
        raise ValueError("character cuts require character window units")
    if type(max_windows) is not int or max_windows <= 0:
        raise ValueError("max_windows must be a positive integer")
    low, high = 0, len(source.text)
    if config.scope == "last_turn":
        if (
            last_turn is None
            or type(last_turn) not in (tuple, list)
            or len(last_turn) != 2
            or any(type(position) is not int for position in last_turn)
            or not 0 <= last_turn[0] <= last_turn[1] <= high
        ):
            raise ValueError("last-turn cuts require a valid full-input last_turn interval")
        low, high = last_turn
    if low == high:
        return ()
    step = config.step
    assert step is not None
    count = 1 + (max(0, high - low - config.size) + step - 1) // step
    if count > max_windows:
        raise ValueError("model windows exceed the window count limit")
    if count * min(config.size, high - low) > MAX_WINDOW_CHARACTER_WORK:
        raise ValueError("model windows exceed the character work limit")
    windows = []
    for ordinal in range(count):
        start = low + ordinal * step
        windows.append(CharacterWindow(source, start, min(start + config.size, high)))
        if ordinal % 256 == 0:
            await asyncio.sleep(0)
            raise_if_evaluation_expired()
    return tuple(windows)


@dataclass(frozen=True, slots=True)
class TokenEncoding:
    """One host encoding, including optional endpoint tokens and a last-turn interval.

    ``last_turn`` uses half-open positions in the full ``token_ids``, before any cut.
    Endpoint flags identify exactly one special token at their respective end. Window
    size counts content tokens; actual request limits also count preserved endpoints.
    """

    token_ids: tuple[int, ...]
    prefix_token: bool = False
    suffix_token: bool = False
    last_turn: tuple[int, int] | None = None
    context_tokens: int = 0
    # Optional half-open Python character offsets in the full rendered input. None for
    # the whole table means unavailable; a None row is a token with no input characters.
    character_offsets: tuple[tuple[int, int] | None, ...] | None = None
    cache_lifetime: ClassVar[int] = 300

    def __post_init__(self):
        raise_if_evaluation_expired()
        tokens = self.token_ids
        if type(tokens) not in (tuple, list) or len(tokens) > MAX_ENCODING_TOKENS:
            raise ValueError("encoding must contain a bounded list or tuple of token IDs")
        for ordinal, token in enumerate(tokens):
            if ordinal % 256 == 0:
                raise_if_evaluation_expired()
            if type(token) is not int or token < 0:
                raise ValueError("encoding token IDs must be non-negative integers")
        object.__setattr__(self, "token_ids", tuple(tokens))
        if type(self.context_tokens) is not int or self.context_tokens < 0:
            raise ValueError("encoding context_tokens must be a non-negative integer")
        if type(self.prefix_token) is not bool or type(self.suffix_token) is not bool:
            raise ValueError("encoding endpoint flags must be booleans")
        if int(self.prefix_token) + int(self.suffix_token) > len(tokens):
            raise ValueError("encoding endpoint tokens must occupy distinct positions")
        turn = self.last_turn
        if turn is not None:
            if (
                type(turn) not in (tuple, list)
                or len(turn) != 2
                or any(type(position) is not int for position in turn)
                or not 0 <= turn[0] <= turn[1] <= len(tokens)
            ):
                raise ValueError("encoding last_turn must be an interval in the full encoding")
            object.__setattr__(self, "last_turn", tuple(turn))
        offsets = self.character_offsets
        if offsets is not None:
            if type(offsets) not in (tuple, list) or len(offsets) != len(tokens):
                raise ValueError("encoding character_offsets must match the token IDs")
            detached = []
            previous = (0, 0)
            for ordinal, row in enumerate(offsets):
                if row is not None:
                    if (
                        type(row) not in (tuple, list)
                        or len(row) != 2
                        or any(type(value) is not int for value in row)
                        or not 0 <= row[0] <= row[1]
                        or row[0] != row[1]
                        and (row[0] < previous[0] or row[1] < previous[1])
                        or (
                            ordinal == 0
                            and self.prefix_token
                            or ordinal == len(tokens) - 1
                            and self.suffix_token
                        )
                        and row[0] != row[1]
                    ):
                        raise ValueError(
                            "encoding character_offsets must be ordered input intervals"
                        )
                    value = tuple(row)
                    if row[0] != row[1]:
                        previous = value
                    detached.append(value)
                else:
                    detached.append(None)
                if ordinal % 256 == 0:
                    raise_if_evaluation_expired()
            object.__setattr__(self, "character_offsets", tuple(detached))
        raise_if_evaluation_expired()


def token_character_ranges(encoding, positions, charge):
    """Coalesce actual host offsets without inferring character widths from token IDs."""
    offsets = encoding.character_offsets
    if offsets is None:
        raise ValueError("encoding character_offsets are unavailable")
    current = None
    for position in positions:
        charge()
        row = offsets[position]
        if row is None or row[0] == row[1]:
            continue
        if current is not None and row[0] <= current[1]:
            current = current[0], max(current[1], row[1])
        else:
            if current is not None:
                yield current
            current = row
    if current is not None:
        yield current


@dataclass(frozen=True, slots=True)
class TokenWindow:
    """Exact provider token IDs and their positions in the full originating encoding."""

    token_ids: tuple[int, ...]
    positions: tuple[int, ...]


async def token_windows(
    encoding: TokenEncoding,
    config: ModelWindowsConfig,
    *,
    max_windows: int,
) -> tuple[TokenWindow, ...]:
    """Cut content while preserving each applicable endpoint once in every batch."""
    raise_if_evaluation_expired()
    if config.unit != "tokens":
        raise ValueError("token cuts require token window units")
    if type(max_windows) is not int or max_windows <= 0:
        raise ValueError("max_windows must be a positive integer")
    low, high = int(encoding.prefix_token), len(encoding.token_ids) - int(encoding.suffix_token)
    if config.scope == "last_turn":
        if encoding.last_turn is None:
            raise ValueError("last-turn cuts require the host's last_turn interval")
        low = max(low, encoding.last_turn[0])
        high = min(high, encoding.last_turn[1])
    if low >= high:
        return ()
    step = config.step
    assert step is not None
    count = 1 + (max(0, high - low - config.size) + step - 1) // step
    if count > max_windows:
        raise ValueError("model windows exceed the window count limit")
    endpoints = (0,) if encoding.prefix_token else ()
    tail = (len(encoding.token_ids) - 1,) if encoding.suffix_token else ()
    # Bound allocations for heavily overlapping windows before constructing any batches.
    if count * (min(config.size, high - low) + len(endpoints) + len(tail)) > MAX_WINDOW_TOKEN_WORK:
        raise ValueError("model windows exceed the token work limit")
    windows = []
    for ordinal in range(count):
        start = low + ordinal * step
        positions = (*endpoints, *range(start, min(start + config.size, high)), *tail)
        windows.append(TokenWindow(tuple(encoding.token_ids[p] for p in positions), positions))
        if ordinal % 256 == 0:
            await asyncio.sleep(0)
            raise_if_evaluation_expired()
    return tuple(windows)


def _selected_token_values(
    head: object,
    windows: tuple[TokenWindow, ...],
    class_index: int,
):
    """Yield validated probabilities, with None separating native window rows."""
    if type(class_index) is not int or class_index < 0:
        raise ValueError("token head class_index must be a non-negative integer")
    if (
        not isinstance(head, (list, tuple))
        or type(head) not in (list, tuple)
        or len(head) != len(windows)
    ):
        raise ValueError("token head must contain one result per window")
    work = 0
    for rows, window in zip(head, windows, strict=True):
        if type(rows) not in (list, tuple) or len(rows) != len(window.token_ids):
            raise ValueError("token head must contain one row per requested token")
        for row in rows:
            work += 1
            if work > MAX_WINDOW_TOKEN_WORK:
                raise ValueError("token head exceeds its work limit")
            if work % 256 == 0:
                raise_if_evaluation_expired()
            if type(row) not in (list, tuple) or class_index >= len(row):
                raise ValueError("token head row has no selected class")
            value = row[class_index]
            if type(value) not in (int, float):
                raise ValueError("selected token probability must be a finite number in [0, 1]")
            try:
                value = float(value)
            except OverflowError as exc:
                raise ValueError(
                    "selected token probability must be a finite number in [0, 1]"
                ) from exc
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("selected token probability must be a finite number in [0, 1]")
            yield value
        yield None
    raise_if_evaluation_expired()


def selected_token_scores(head, windows, class_index) -> tuple[tuple[float, ...], ...]:
    """Validate the whole selected head for synchronous host consumers."""
    scores, selected = [], []
    for value in _selected_token_values(head, windows, class_index):
        if value is None:
            scores.append(tuple(selected))
            selected = []
        else:
            selected.append(value)
    return tuple(scores)


async def selected_token_scores_async(head, windows, class_index) -> tuple[tuple[float, ...], ...]:
    """Validate the whole selected head with checkpoints before any decoder can run."""
    scores, selected = [], []
    for ordinal, value in enumerate(_selected_token_values(head, windows, class_index)):
        if ordinal % 256 == 0:
            await asyncio.sleep(0)
            raise_if_evaluation_expired()
        if value is None:
            scores.append(tuple(selected))
            selected = []
        else:
            selected.append(value)
    return tuple(scores)
