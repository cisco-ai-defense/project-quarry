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

"""Public contracts for one typed collection returned by a rule."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Generic, Protocol, TypeVar


class ResultValue(Protocol):
    """A typed result value with an explicit public serializer."""

    def to_dict(self) -> dict:
        """Return the public, JSON-compatible value representation."""


@dataclass(frozen=True, slots=True)
class Score:
    """One normalized model score."""

    value: float

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise TypeError("score must be a number")
        value = float(self.value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("score must be finite and between 0 and 1")
        object.__setattr__(self, "value", value)

    def to_dict(self) -> dict[str, float]:
        return {"value": self.value}


@dataclass(frozen=True, slots=True)
class InputText:
    """The input section and item that supplied a text field."""

    input: str
    item: str | None
    fields: tuple[str, ...] = field(compare=False)

    def to_dict(self) -> dict[str, str]:
        data = {"input": self.input}
        if self.item is not None:
            data["item"] = self.item
        return data


PRECISIONS = ("exact", "interpolated", "enclosing")


def text_part(owner: object, name: str) -> tuple[int, int, str]:
    """Order a message's text before other named text fields of that item."""

    if not name:
        return (0, 0, "")
    rank = getattr(owner, "field_rank", None)
    return (0, rank(name) if callable(rank) else 1, name)


@dataclass(frozen=True, slots=True)
class Span:
    """One location in a conversation: from ``start`` of one message to ``end`` of another.

    Offsets index each message's canonical text. Both end messages are inclusive and every
    message between them is covered whole; ``end_message_index`` defaults to
    ``message_index``, a range inside one message. A message without text has the single
    offset ``0``. ``score`` is set when a model scored the span.
    """

    message_index: int
    start: int
    end: int
    score: Score | None = None
    end_message_index: int | None = None
    _message: object | None = field(default=None, repr=False, compare=False)
    # Input text has its own place and canonical owner, independent of chat messages.
    source: InputText | None = None
    source_field: str | None = None
    end_source_field: str | None = None
    precision: str = "exact"
    label: str | None = None
    _source: object | None = field(default=None, repr=False, compare=False)
    _found_in: tuple[object, int, int, int] | None = field(default=None, repr=False, compare=False)
    _found: tuple[int, int] | None = field(init=False, default=None, repr=False)
    place_position: int | None = None
    end_place_position: int | None = None

    def __post_init__(self) -> None:
        found = self._found_in
        object.__setattr__(
            self, "_found", None if found is None or self.precision == "exact" else found[2:]
        )
        end_index = self.message_index if self.end_message_index is None else self.end_message_index
        object.__setattr__(self, "end_message_index", end_index)
        for name, value in (
            ("message_index", self.message_index),
            ("start", self.start),
            ("end", self.end),
            ("end_message_index", end_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"span {name} must be an integer")
        for name, value in (
            ("place_position", self.place_position),
            ("end_place_position", self.end_place_position),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise TypeError(f"span {name} must be an integer")
            if value is not None and value < 0:
                raise ValueError(f"span {name} must not be negative")
        if self.message_index < 0:
            raise ValueError("span message_index must not be negative")
        if self.start < 0 or self.end < 0:
            raise ValueError("span offsets must not be negative")
        if end_index < self.message_index:
            raise ValueError("span must not end before the message it starts in")
        if self.source is not None:
            if not isinstance(self.source, InputText):
                raise TypeError("span source must be an InputText")
            if self.source_field not in self.source.fields:
                raise ValueError("span source field is outside the input content fields")
            end_field = self.end_source_field or self.source_field
            if end_field not in self.source.fields:
                raise ValueError("span end source field is outside the input content fields")
            if self.source.fields.index(end_field) < self.source.fields.index(self.source_field):
                raise ValueError("span must end in the same or a later input field")
            if end_index != self.message_index:
                raise ValueError("an input span cannot cross items")
        # Only a message without text is covered by the empty range 0..0.
        if (
            end_index == self.message_index
            and self.source_field == (self.end_source_field or self.source_field)
            and self.end <= self.start
            and (self.start, self.end) != (0, 0)
        ):
            raise ValueError("span end must be greater than start")
        if self.score is not None and not isinstance(self.score, Score):
            raise TypeError("span score must be a Score")
        if self.precision not in PRECISIONS:
            raise ValueError("span precision must be exact, interpolated, or enclosing")
        if self.label is not None and (not isinstance(self.label, str) or not self.label):
            raise ValueError("span label must be a nonempty string")

    @property
    def within_one_message(self) -> bool:
        """Return whether the span is within one message."""
        return self.end_message_index == self.message_index

    @property
    def position(self) -> int:
        """The place of this text in its source."""

        return self.message_index if self.place_position is None else self.place_position

    @property
    def end_position(self) -> int:
        if self.end_place_position is not None:
            return self.end_place_position
        return self.end_message_index if self.place_position is None else self.place_position

    @property
    def field(self) -> str:
        return self.source_field or ""

    @property
    def end_field(self) -> str:
        return self.end_source_field or self.field

    @property
    def within_one_text(self) -> bool:
        return self.position == self.end_position and self.field == self.end_field

    def to_dict(self) -> dict:
        """Return the JSON-ready span shape."""
        if self.source is not None:
            data: dict = {
                "position": self.position,
                "field": self.source_field,
                "start": self.start,
                "end": self.end,
            }
            if (end_field := self.end_source_field or self.source_field) != self.source_field:
                data["end_position"] = self.end_position
                data["end_field"] = end_field
            if self.score is not None:
                data["score"] = self.score.to_dict()
            if self.precision != "exact":
                data["precision"] = self.precision
            if self.label is not None:
                data["label"] = self.label
            if self._found_in is not None:
                data["found_in"] = {
                    "message_index": self._found_in[1],
                    "start": self._found_in[2],
                    "end": self._found_in[3],
                }
            data["source"] = self.source.to_dict()
            return data
        if self.place_position is not None and self._message is None:
            data = {
                "position": self.position,
                "field": self.field,
                "start": self.start,
                "end": self.end,
            }
            if not self.within_one_text:
                data["end_position"] = self.end_position
                data["end_field"] = self.end_field
            if self.score is not None:
                data["score"] = self.score.to_dict()
            if self.precision != "exact":
                data["precision"] = self.precision
            if self.label is not None:
                data["label"] = self.label
            if self._found_in is not None:
                data["found_in"] = {
                    "message_index": self._found_in[1],
                    "start": self._found_in[2],
                    "end": self._found_in[3],
                }
            return data
        data: dict = {"message_index": self.message_index, "start": self.start}
        if not self.within_one_message:
            data["end_message_index"] = self.end_message_index
        data["end"] = self.end
        if self.score is not None:
            data["score"] = self.score.to_dict()
        if self.source_field is not None:
            data["field"] = self.source_field
        if self.end_source_field is not None and self.end_source_field != self.source_field:
            data["end_field"] = self.end_source_field
        if self.precision != "exact":
            data["precision"] = self.precision
        if self.label is not None:
            data["label"] = self.label
        return data

    def _bind_to_message(self, message: object) -> Span:
        """Return this span bound to its canonical source message."""

        return replace(self, _message=message)

    def _bind_to_source(self, owner: object) -> Span:
        """Return this span bound to the input object holding its text."""

        return replace(self, _source=owner)


@dataclass(frozen=True, slots=True)
class ProducerRef:
    """Scan-local provenance for one typed collection producer."""

    kind: str
    name: str
    run_id: str

    def to_dict(self) -> dict[str, str]:
        """Return the JSON-ready producer ref shape."""
        return {"kind": self.kind, "name": self.name, "run_id": self.run_id}


T = TypeVar("T", bound=ResultValue)


@dataclass(frozen=True, slots=True)
class ResultItem(Generic[T]):
    """One typed value and its scan-local identity."""

    item_id: str
    value: T

    def to_dict(self) -> dict:
        """Return the JSON-ready result item shape."""
        return {"item_id": self.item_id, "value": self.value.to_dict()}


@dataclass(frozen=True, slots=True)
class CollectionResult(Generic[T]):
    """One complete producer-owned collection returned by a rule."""

    output_type: str
    category: str | None
    source: ProducerRef
    items: tuple[ResultItem[T], ...]

    def __getitem__(self, key: int | slice) -> T | CollectionResult[T]:
        """Read one value or retain a slice of this collection."""

        if isinstance(key, slice):
            return replace(self, items=self.items[key])
        return self.items[key].value

    def to_dict(self) -> dict:
        """Return the JSON-ready collection result shape."""
        return {
            "type": self.output_type,
            "category": self.category,
            "source": self.source.to_dict(),
            "items": [item.to_dict() for item in self.items],
        }
