# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Task-owned, bounded window decisions; no input text, token IDs or host reply owners."""

from __future__ import annotations

import asyncio
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from quarry.integrations.model_windows import CharacterWindow, TokenWindow, token_character_ranges
from quarry.types.messages import (
    AssistantMessage,
    DeveloperMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
    _synthetic_key,
)
from quarry.types.origins import ENCLOSING, PRECISIONS, WHOLE, Derived, Place, _mapped
from quarry.types.results import InputText
from quarry.types.transforms import ContentSourcePart

MAX_WINDOW_SAMPLES = 8
MAX_WINDOW_PIECES = 8
MAX_WINDOW_EVIDENCE_WORK = 4096
_MESSAGES = (SystemMessage, DeveloperMessage, UserMessage, AssistantMessage, ToolMessage)


@dataclass(slots=True)
class _Capture:
    pid: int
    thread: int
    task: int
    value: dict | None = None


_CAPTURE: ContextVar[_Capture | None] = ContextVar("window_decision_capture", default=None)


@contextmanager
def capture_window_result(enabled=True):
    if not enabled:
        yield None
        return
    capture = _Capture(os.getpid(), threading.get_ident(), id(asyncio.current_task()))
    token = _CAPTURE.set(capture)
    try:
        yield capture
    finally:
        _CAPTURE.reset(token)


class _GeometryLimit(ValueError):
    pass


def _context_parts(owner, start, end, work):
    parts = {}
    for segment in owner._provenance or ():
        work[0] += 1
        if work[0] > MAX_WINDOW_EVIDENCE_WORK:
            raise _GeometryLimit
        if segment.start >= end:
            break
        if segment.end <= start:
            continue
        for part in segment.parts:
            work[0] += 1
            if work[0] > MAX_WINDOW_EVIDENCE_WORK or type(part) is not ContentSourcePart:
                raise _GeometryLimit
            parts[(part.message_index, part.part_index)] = part.to_dict()
            if len(parts) > MAX_WINDOW_PIECES:
                raise _GeometryLimit
    return list(parts.values())


def _piece_ranges(origin, low, high, work, precision=0, depth=0):
    work[0] += 1
    if work[0] > MAX_WINDOW_EVIDENCE_WORK or depth > 100:
        raise _GeometryLimit
    if type(origin) is Place:
        if type(origin.field) is not str or len(origin.field) > 256:
            raise _GeometryLimit
        piece = {
            "position": origin.position,
            "field": origin.field,
            "start": origin.start + low,
            "end": origin.start + high,
            "precision": PRECISIONS[precision],
        }
        if type(origin.owner) in _MESSAGES:
            if _synthetic_key(origin.owner) is not None:
                piece["constructed"] = True
            parts = _context_parts(origin.owner, origin.start + low, origin.start + high, work)
            if parts:
                piece["source_parts"] = parts
        source = origin.source
        if source is not None:
            if (
                type(source) is not InputText
                or type(source.input) is not str
                or len(source.input) > 256
                or type(source.item) not in (str, type(None))
                or source.item is not None
                and len(source.item) > 256
                or type(source.fields) is not tuple
                or len(source.fields) > 8
                or any(type(field) is not str or len(field) > 256 for field in source.fields)
            ):
                raise _GeometryLimit
            piece["source"] = source.to_dict()
        yield piece
    elif type(origin) is Derived:
        for fields in origin._touched(low, high):
            work[0] += 1
            if work[0] > MAX_WINDOW_EVIDENCE_WORK:
                raise _GeometryLimit
            start, start_precision = _mapped(fields, max(low, fields[0]), end=False)
            end, end_precision = _mapped(fields, min(high, fields[1]), end=True)
            if start < end:
                yield from _piece_ranges(
                    origin.sources[fields[2]],
                    start,
                    end,
                    work,
                    max(
                        precision,
                        start_precision,
                        end_precision,
                        ENCLOSING if fields[5] == WHOLE else 0,
                    ),
                    depth + 1,
                )
    else:
        raise _GeometryLimit


def _character_pieces(source, start, end, work):
    pieces, truncated = [], False
    try:
        for piece in _piece_ranges(source.origin, start, end, work):
            if len(pieces) == MAX_WINDOW_PIECES:
                truncated = True
                break
            pieces.append(piece)
    except (ValueError, TypeError, IndexError, KeyError):
        truncated = True
    return pieces, truncated


def _token_intervals(window, encoding):
    # Native plans have contiguous content positions plus at most one endpoint at each end.
    first, beyond = int(encoding.prefix_token), len(window.positions) - int(encoding.suffix_token)
    intervals = []
    candidates = []
    if first:
        candidates.append((window.positions[0], window.positions[0] + 1))
    if first < beyond:
        candidates.append((window.positions[first], window.positions[beyond - 1] + 1))
    if beyond < len(window.positions):
        candidates.append((window.positions[-1], window.positions[-1] + 1))
    for start, end in candidates:
        if intervals and start == intervals[-1][1]:
            intervals[-1][1] = end
        else:
            intervals.append([start, end])
    return intervals


def _token_pieces(source, window, encoding, work):
    if encoding.character_offsets is None:
        return [], False, "unavailable"
    pieces, truncated = [], False

    def charge():
        work[0] += 1
        if work[0] > MAX_WINDOW_EVIDENCE_WORK:
            raise _GeometryLimit

    try:
        for start, end in token_character_ranges(encoding, window.positions, charge):
            selected, partial = _character_pieces(source, start, end, work)
            remaining = MAX_WINDOW_PIECES - len(pieces)
            pieces.extend(selected[:remaining])
            truncated |= partial or len(selected) > remaining
            if truncated:
                break
    except (ValueError, TypeError, IndexError, KeyError):
        truncated = True
    return pieces, truncated, "partial" if truncated else "mapped"


def record_window_result(measured, windows, source, encoding):
    capture = _CAPTURE.get()
    if (
        capture is None
        or capture.pid != os.getpid()
        or capture.thread != threading.get_ident()
        or capture.task != id(asyncio.current_task())
    ):
        return
    values, maximum, deciding = measured
    count = len(values)
    ordinals = list(range(min(count, MAX_WINDOW_SAMPLES)))
    if deciding not in ordinals:
        ordinals[-1] = deciding
    work, samples = [0], []
    # Spend the small optional geometry budget on the deciding window first.
    for ordinal in (deciding, *(ordinal for ordinal in ordinals if ordinal != deciding)):
        window = windows[ordinal]
        if type(window) is CharacterWindow:
            unit, intervals = "characters", [[window.start, window.end]]
            pieces, truncated = _character_pieces(source, window.start, window.end, work)
            geometry = "partial" if truncated else "mapped"
        elif type(window) is TokenWindow:
            unit, intervals = "tokens", _token_intervals(window, encoding)
            pieces, truncated, geometry = _token_pieces(source, window, encoding, work)
        else:
            return
        if not truncated and geometry != "unavailable":
            kinds = {
                "context"
                if "source_parts" in piece
                else "constructed"
                if piece.get("constructed")
                else "mapped"
                for piece in pieces
            }
            geometry = next(iter(kinds)) if len(kinds) == 1 else "mixed" if kinds else "unavailable"
        samples.append(
            {
                "ordinal": ordinal,
                "value": values[ordinal],
                "range": {"unit": unit, "intervals": intervals},
                "pieces": pieces,
                "pieces_truncated": truncated,
                "source_geometry": geometry,
            }
        )
    samples.sort(key=lambda sample: sample["ordinal"])
    capture.value = {
        "unit": unit,
        "count": count,
        "minimum": min(values),
        "maximum": maximum,
        "deciding": next(sample for sample in samples if sample["ordinal"] == deciding),
        "samples": samples,
        "omitted_windows": count - len(samples),
    }
