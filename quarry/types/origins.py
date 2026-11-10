# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Where a text came from, so evidence found in any text points into the scanned conversation.

A ``Place`` is a text of the conversation: one field of one item, read from an offset. A
``Derived`` text is one Quarry made from others, such as a model's rendered input, with sorted
runs that say which range of which source each range of it copied. Text no run covers, such
as a role prefix or a separator, came from no source.
"""

from __future__ import annotations

from array import array
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

from quarry.types.results import PRECISIONS, InputText, Span, text_part

# How a run maps its output range to its source range: ``LINEAR`` character for character, or
# by interpolation when the lengths differ; ``WHOLE`` as a whole, so any offset inside it
# stands for the whole source range.
LINEAR, WHOLE = 0, 1
# The fields of one run, flat in ``Derived.runs``: output start and end, the index of its
# source, source start and end, and its kind.
RUN_FIELDS = 6
# The most runs one transformed unit keeps; a larger map is coarsened, never refused.
MAX_RUNS_PER_UNIT = 512
EXACT, INTERPOLATED, ENCLOSING = range(len(PRECISIONS))


@dataclass(frozen=True, slots=True)
class Place:
    """A text of the scanned conversation: ``field`` of the item at ``position`` (empty for a
    message's text, or a tool call's argument path), read from ``start`` of that field's whole
    text, as a ``split`` piece holds part of a message. ``owner`` is the canonical item a span
    of the text is bound to."""

    position: int
    field: str
    start: int
    owner: Any = field(compare=False)
    # The input item the text belongs to, or None for text of the conversation.
    source: InputText | None = None


@dataclass(frozen=True, slots=True, eq=False)
class Derived:
    """A text Quarry or an adapter made, and where each range of it came from.

    ``runs`` holds ``RUN_FIELDS`` integers per run: its output range, the index in ``sources``
    of the text it read, that text's range, and the run's kind. Output ranges are non-empty,
    sorted and do not overlap; so are each source's ranges. A source is the origin of the text
    it read, a place or another derived text, which is mapped back in turn when a span is
    built.
    """

    text: str
    sources: tuple[Origin, ...]
    runs: array

    def __eq__(self, other: object) -> bool:
        """Whether ``other`` is the same text made the same way from the same places, as each
        evaluation of one transform of one input makes it, so a span found in one is found in
        the other."""

        return self is other or (
            isinstance(other, Derived)
            and self.text == other.text
            and self.runs == other.runs
            and self.sources == other.sources
        )

    def __hash__(self) -> int:
        return hash(self.text)

    def places(self, start: int, end: int) -> tuple[Span, ...]:
        """The places of the conversation ``text[start:end]`` covers: one span for each
        stretch of message texts next to each other, from the first character it covers to
        the last, and one for each other text. Empty for text no source wrote."""

        groups: list[list[_Piece]] = []
        for piece in self._pieces(start, end):
            place = piece[0]
            last = groups[-1][-1] if groups else None
            if (
                last is not None
                and place.field == ""
                and last[0].field == ""
                and place.owner.message_index - last[0].owner.message_index in {0, 1}
                and _start(piece) >= _end(last)
            ):
                groups[-1].append(piece)
            else:
                groups.append([piece])
        return tuple(_span(group[0], group[-1]) for group in groups)

    def span(self, start: int, end: int) -> Span:
        """The one span of the conversation ``text[start:end]``, a non-empty range of a
        message's text, covers: from the first place it covers to the last in conversation
        order, across every text between. Every character of such a text came from a source.
        Text that reads its places out of conversation order, such as messages joined in
        another order or a message written twice, gives the span that holds them all, which
        is enclosing. A range whose ends lie in texts no span can join,
        such as a tool call's argument and a later message, fails."""

        pieces = self._pieces(start, end)
        if all(
            _start(piece) >= _end(before) for before, piece in zip(pieces, pieces[1:], strict=False)
        ):
            return _span(pieces[0], pieces[-1])
        first, last = min(pieces, key=_start), max(pieces, key=_end)
        return replace(_span(first, last), precision=PRECISIONS[ENCLOSING])

    def _pieces(self, start: int, end: int) -> list[_Piece]:
        """The ranges of places ``text[start:end]`` covers, in order, each with how closely
        its start and its end are known. One source's touched runs give one range, from the
        start of the first to the end of the last; a derived source is mapped in turn."""

        # Each touched source's range and precisions, in output order.
        covered: dict[int, list[int]] = {}
        for fields in self._touched(start, end):
            low, low_precision = _mapped(fields, max(start, fields[0]), end=False)
            high, high_precision = _mapped(fields, min(end, fields[1]), end=True)
            entry = covered.setdefault(fields[2], [low, high, low_precision, high_precision])
            entry[1], entry[3] = high, high_precision
        pieces: list[_Piece] = []
        for source, (low, high, low_precision, high_precision) in covered.items():
            origin = self.sources[source]
            if isinstance(origin, Place):
                pieces.append((origin, low, high, low_precision, high_precision))
                continue
            inner = origin._pieces(low, high)
            if inner:
                first = inner[0]
                inner[0] = (*first[:3], max(low_precision, first[3]), first[4])
                last = inner[-1]
                inner[-1] = (*last[:4], max(high_precision, last[4]))
                pieces.extend(inner)
        return pieces

    def _touched(self, start: int, end: int) -> Iterator[array]:
        """The fields of each run whose output range meets ``text[start:end]``, in order."""

        runs = self.runs
        count = len(runs) // RUN_FIELDS
        # The first run that ends after ``start``; output ends are sorted as starts are.
        index = bisect_right(range(count), start, key=lambda run: runs[run * RUN_FIELDS + 1])
        for run in range(index, count):
            fields = runs[run * RUN_FIELDS : (run + 1) * RUN_FIELDS]
            if fields[0] >= end:
                return
            yield fields


Origin = Place | Derived
# A range of one place, and how closely its start and its end are known.
_Piece = tuple[Place, int, int, int, int]
# A point of the conversation in its order: the input item it is in ("" for the chat), a
# position, the place of a text among the texts of its item, and an offset in that text.
Point = tuple[str, int, tuple[int, int, str], int]


def source_key(source: InputText | None) -> str:
    """The order of the texts of ``source``, an input item, among the scan's texts: the
    chat's first, then each input item's apart."""

    return "" if source is None else f"\x00{source.input}\x00{source.item or ''}"


def field_part(owner: Any, field: str, source: InputText | None) -> tuple[int, int, str]:
    """Where ``field`` orders among the texts of its item: an input item's content fields in
    declared order, or as :func:`text_part` orders a message's or a tool call's."""

    if source is not None:
        return (0, source.fields.index(field), field)
    return text_part(owner, field)


def reads(origin: Origin, start: int, end: int) -> Iterator[tuple[Point, Point]]:
    """The ranges of the conversation ``text[start:end]`` of the text at ``origin`` was made
    from, each from where it starts to where it ends, in the order the text holds them: one
    for a conversation text, and for a made text one for each place each of its runs reads,
    a derived source's in turn. When each range starts where the one before it ends or
    later, every span found in the text starts at or after the spans found before it."""

    if isinstance(origin, Place):
        key = source_key(origin.source)
        part = field_part(origin.owner, origin.field, origin.source)
        yield (
            (key, origin.position, part, origin.start + start),
            (key, origin.position, part, origin.start + end),
        )
        return
    for fields in origin._touched(start, end):
        low, _ = _mapped(fields, max(start, fields[0]), end=False)
        high, _ = _mapped(fields, min(end, fields[1]), end=True)
        yield from reads(origin.sources[fields[2]], low, high)


def _mapped(run: Sequence[int], at: int, *, end: bool) -> tuple[int, int]:
    """Where offset ``at`` of a run's output, as a span's start or its ``end``, lies in the
    run's source, and how closely: exactly on the run's boundary or inside an exact run, by
    rounding outward inside a run of unequal lengths, and at the whole source range inside a
    whole run or a run onto an empty source, such as a media part."""

    out_start, out_end, _, source_start, source_end, kind = run
    edge = source_end if end else source_start
    if source_start == source_end:
        return edge, ENCLOSING
    if at == (out_end if end else out_start):
        return edge, EXACT
    if kind == WHOLE:
        return edge, ENCLOSING
    out_length, source_length = out_end - out_start, source_end - source_start
    if out_length == source_length:
        return source_start + at - out_start, EXACT
    scaled = (at - out_start) * source_length
    # Floor for a start and ceiling for an end, so the true range lies inside more often.
    return source_start + (-(-scaled // out_length) if end else scaled // out_length), INTERPOLATED


def _start(piece: _Piece) -> Point:
    """Where a piece starts in conversation order: its input item, its position, its text's
    place among the texts of its item, and its offset in that text."""

    place = piece[0]
    part = field_part(place.owner, place.field, place.source)
    return source_key(place.source), place.position, part, place.start + piece[1]


def _end(piece: _Piece) -> Point:
    """Where a piece ends in conversation order, as :func:`_start` orders its start."""

    place = piece[0]
    part = field_part(place.owner, place.field, place.source)
    return source_key(place.source), place.position, part, place.start + piece[2]


def home(origin: Origin) -> Place:
    """The first place of the conversation ``origin`` reads. A message's origin reads at
    least one: each text Quarry makes of a message reads that message's texts."""

    while isinstance(origin, Derived):
        origin = origin.sources[0]
    return origin


def _span(first: _Piece, last: _Piece) -> Span:
    """The span from ``first``'s start to ``last``'s end, bound to ``first``'s owner, as
    precise as the coarser of its two ends."""

    start, low = first[0], first[1]
    finish, high = last[0], last[2]
    # Each text Quarry makes reads one input item's texts, or the chat's, never both.
    assert start.source == finish.source, "a span cannot join texts of two inputs"
    return Span(
        getattr(start.owner, "message_index", start.position),
        start.start + low,
        finish.start + high,
        end_message_index=getattr(finish.owner, "message_index", finish.position),
        source_field=start.field or None,
        end_source_field=finish.field or None,
        precision=PRECISIONS[max(first[3], last[4])],
        source=start.source,
        place_position=start.position,
        end_place_position=finish.position,
    )._bind_to_source(start.owner)


def checked_map(text: str, source_length: int, runs: Any) -> array:
    """The map of ``text``, which a transformation made from one unit of text of
    ``source_length`` characters (0 for a media part, whose text is empty), as the flat runs
    of a ``Derived`` text of that one source: the one rule every transformation's map follows.

    ``runs`` is the transformation's map, a list of runs each ``[out_start, out_end,
    source_start, source_end]`` for a linear run or with ``"whole"`` appended for a whole one:
    output ranges non-empty, in order and covering ``text`` exactly; source ranges inside the
    unit, in order and not overlapping, with gaps for deleted text, each non-empty unless the
    unit is. Without a map (None), the whole text stands for the whole unit. A map that breaks
    the rule fails with ``ValueError`` naming what it returned; adjacent exact runs are merged,
    and a map of more than ``MAX_RUNS_PER_UNIT`` runs is coarsened into whole runs, so the map
    kept is at most that many runs whatever the transformation returned.
    """

    if runs is None:
        runs = [[0, len(text), 0, source_length, "whole"]] if text else []
    if not isinstance(runs, list | tuple):
        raise ValueError("runs that are not a list")
    checked: list[list[int]] = []
    for run in runs:
        if (
            not isinstance(run, list | tuple)
            or len(run) not in {4, 5}
            or any(isinstance(value, bool) or not isinstance(value, int) for value in run[:4])
            or (len(run) == 5 and run[4] != "whole")
        ):
            raise ValueError('a run that is not four integers, or four integers and "whole"')
        out_start, out_end, source_start, source_end = cast(Sequence[int], run[:4])
        expected_start = checked[-1][1] if checked else 0
        if out_start != expected_start or not out_start < out_end <= len(text):
            raise ValueError("runs that do not cover its text in order without overlap")
        if not 0 <= source_start <= source_end <= source_length:
            raise ValueError("a run outside the part it read")
        if checked and source_start < checked[-1][4]:
            raise ValueError("runs whose source ranges move back or overlap")
        if source_start == source_end and source_length:
            raise ValueError("a run that reads no character of a part with text")
        kind = WHOLE if len(run) == 5 else LINEAR
        last = checked[-1] if checked else None
        if (
            last is not None
            and kind == LINEAR == last[5]
            and last[1] - last[0] == last[4] - last[3]
            and out_end - out_start == source_end - source_start
            and last[4] == source_start
        ):
            last[1], last[4] = out_end, source_end
        else:
            checked.append([out_start, out_end, 0, source_start, source_end, kind])
    if (checked[-1][1] if checked else 0) != len(text):
        raise ValueError("runs that do not cover its text")
    if len(checked) > MAX_RUNS_PER_UNIT:
        size = -(-len(checked) // MAX_RUNS_PER_UNIT)
        checked = [
            group[0]
            if len(group) == 1
            else [group[0][0], group[-1][1], 0, group[0][3], group[-1][4], WHOLE]
            for group in (checked[i : i + size] for i in range(0, len(checked), size))
        ]
    return array("q", (value for run in checked for value in run))


def joined(pieces: Iterable[tuple[str, Origin | None]]) -> Derived:
    """The origin of the texts of ``pieces`` written one after another, each exactly at its
    origin; a piece without one, such as a separator, came from no source. An empty piece
    copies nothing but stays a source, so the text still says which item it was read from."""

    texts: list[str] = []
    sources: list[Origin] = []
    runs = array("q")
    offset = 0
    for text, origin in pieces:
        if origin is not None:
            if text:
                runs.extend((offset, offset + len(text), len(sources), 0, len(text), LINEAR))
            sources.append(origin)
        texts.append(text)
        offset += len(text)
    text = "".join(texts)
    return Derived(text, tuple(sources), runs)


def excerpt(origin: Origin, start: int, end: int) -> Origin:
    """The origin of characters ``start`` to ``end`` of the text at ``origin``."""

    if isinstance(origin, Place):
        return replace(origin, start=origin.start + start)
    runs = array("q", (0, end - start, 0, start, end, LINEAR) if end > start else ())
    return Derived(origin.text[start:end], (origin,), runs)


@dataclass(frozen=True, slots=True)
class SourceText:
    """One text spans are found in or cut from, and where it came from: a place of the
    conversation, or a text a transformation made from it, whose spans map back through its
    origin. Each value of a list or an object argument is a text of its own, at its own path.
    ``message_index`` is the ``message_index`` of the item the text belongs to, which a span
    found in a made text reports as where it was found."""

    text: str
    origin: Origin
    message_index: int

    def span(self, begin: int, end: int) -> Span:
        """The span of ``text[begin:end]``: of the conversation text it is, or that it was
        made from, keeping where in this text it was found."""

        origin = self.origin
        if isinstance(origin, Place):
            span = Span(
                getattr(origin.owner, "message_index", origin.position),
                origin.start + begin,
                origin.start + end,
                source_field=origin.field or None,
                source=origin.source,
                place_position=origin.position,
            )
            return span._bind_to_source(origin.owner)
        found = (origin, self.message_index, begin, end)
        return replace(origin.span(begin, end), _found_in=found)

    def rest(self, span: Span) -> Span:
        """The span from where ``span`` starts to the end of the text."""

        offsets = self.offsets(span)
        if offsets is None:
            raise ValueError("a span is outside the text it was found in")
        return self.span(offsets[0], len(self.text))

    def offsets(self, span: Span) -> tuple[int, int] | None:
        """Where ``span`` starts and ends in ``text``, or None when it lies outside it: where
        it was found, for a span found in this made text, or its range of this conversation
        text."""

        origin = self.origin
        if isinstance(origin, Derived):
            found = span._found_in
            return (found[2], found[3]) if found is not None and found[0] == origin else None
        if (span.position, span.field, span.source) != (
            origin.position,
            origin.field,
            origin.source,
        ) or not (span.within_one_text):
            return None
        begin, end = span.start - origin.start, span.end - origin.start
        return (begin, end) if 0 <= begin <= end <= len(self.text) else None

    def pieces(self, begin: int, end: int) -> list[dict[str, Any]]:
        """``text[begin:end]`` as pieces of evidence, each a position, a field and a range:
        one for a conversation text, and for a made text the places it maps to."""

        origin = self.origin
        if isinstance(origin, Derived):
            return [span.to_dict() for span in origin.places(begin, end)]
        piece: dict[str, Any] = {
            "position": origin.position,
            "field": origin.field,
            "start": origin.start + begin,
            "end": origin.start + end,
        }
        if origin.source is not None:
            piece["source"] = origin.source.to_dict()
        return [piece]
