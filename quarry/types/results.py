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

    def __post_init__(self) -> None:
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
        if self.message_index < 0:
            raise ValueError("span message_index must not be negative")
        if self.start < 0 or self.end < 0:
            raise ValueError("span offsets must not be negative")
        if end_index < self.message_index:
            raise ValueError("span must not end before the message it starts in")
        # Only a message without text is covered by the empty range 0..0.
        if (
            end_index == self.message_index
            and self.end <= self.start
            and (self.start, self.end) != (0, 0)
        ):
            raise ValueError("span end must be greater than start")
        if self.score is not None and not isinstance(self.score, Score):
            raise TypeError("span score must be a Score")

    @property
    def within_one_message(self) -> bool:
        """Return whether the span is within one message."""
        return self.end_message_index == self.message_index

    def to_dict(self) -> dict:
        """Return the JSON-ready span shape."""
        data: dict = {"message_index": self.message_index, "start": self.start}
        if not self.within_one_message:
            data["end_message_index"] = self.end_message_index
        data["end"] = self.end
        if self.score is not None:
            data["score"] = self.score.to_dict()
        return data

    def _bind_to_message(self, message: object) -> Span:
        """Return this span bound to its canonical source message."""

        return replace(self, _message=message)


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

    def to_dict(self) -> dict:
        """Return the JSON-ready collection result shape."""
        return {
            "type": self.output_type,
            "category": self.category,
            "source": self.source.to_dict(),
            "items": [item.to_dict() for item in self.items],
        }
