# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Immutable configuration and provenance values for local content transforms."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

TransformKind = Literal["strip", "casefold", "replace", "prefix", "suffix"]
MappingKind = Literal["linear", "whole"]

TRANSFORM_KINDS = frozenset({"strip", "casefold", "replace", "prefix", "suffix"})
MAX_TRANSFORM_DEFINITIONS = 64
MAX_TRANSFORM_NAME_CHARACTERS = 128
MAX_TRANSFORM_OPERATIONS = 16
MAX_TRANSFORM_CHAIN = 16
MAX_TRANSFORM_LITERAL_CHARACTERS = 4_096
MAX_TRANSFORM_PLAN_BYTES = 140_000
MAX_TRANSFORM_INPUT_CHARACTERS = 1_000_000
MAX_TRANSFORM_OUTPUT_CHARACTERS = 2_000_000
MAX_TRANSFORM_EXPANSION_RATIO = 8
MAX_TRANSFORM_INSERTED_CHARACTERS = 4_096
MAX_TRANSFORM_MAPPING_SEGMENTS = 8_192
MAX_TRANSFORM_WORK = 4_000_000


def _canonical_operation_json(operation: TransformOperation) -> str:
    """Encode one operation exactly as it appears inside a canonical plan."""

    return json.dumps(
        operation.to_wire(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_plan_size(operations: tuple[TransformOperation, ...]) -> int:
    """Return the exact UTF-8 wire size, saturating immediately above the safety bound."""

    total = 2  # opening and closing brackets
    ceiling = MAX_TRANSFORM_PLAN_BYTES + 1
    for index, operation in enumerate(operations):
        try:
            encoded = _canonical_operation_json(operation).encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("content transform literals must contain valid Unicode") from exc
        total = min(ceiling, total + bool(index) + len(encoded))
        if total == ceiling:
            return ceiling
    return total


def _bounded_utf8_size(value: str) -> int:
    """Count UTF-8 bytes without materializing an untrusted encoded copy."""

    total = 0
    ceiling = MAX_TRANSFORM_PLAN_BYTES + 1
    for character in value:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise ValueError("compiled content transform must contain valid Unicode")
        width = (
            1 if codepoint <= 0x7F else 2 if codepoint <= 0x7FF else 3 if codepoint <= 0xFFFF else 4
        )
        total = min(ceiling, total + width)
        if total == ceiling:
            return ceiling
    return total


@dataclass(frozen=True, slots=True)
class TransformOperation:
    """One allowlisted operation in a reviewed transform definition."""

    kind: TransformKind
    value: str = ""
    replacement: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in TRANSFORM_KINDS:
            raise ValueError(f"unsupported content transform operation {self.kind!r}")
        if not isinstance(self.value, str) or not isinstance(self.replacement, str):
            raise TypeError("content transform operation values must be strings")
        if (
            len(self.value) > MAX_TRANSFORM_LITERAL_CHARACTERS
            or len(self.replacement) > MAX_TRANSFORM_LITERAL_CHARACTERS
        ):
            raise ValueError(
                "content transform literals must not exceed "
                f"{MAX_TRANSFORM_LITERAL_CHARACTERS} characters"
            )
        if self.kind in {"strip", "casefold"} and (self.value or self.replacement):
            raise ValueError(f"{self.kind} does not accept text values")
        if self.kind in {"prefix", "suffix"} and self.replacement:
            raise ValueError(f"{self.kind} accepts only its text value")
        if self.kind == "replace" and not self.value:
            raise ValueError("replace requires a non-empty literal match")

    def to_wire(self) -> dict[str, str]:
        """Return the canonical JSON-ready operation."""

        if self.kind in {"strip", "casefold"}:
            return {"op": self.kind}
        if self.kind in {"prefix", "suffix"}:
            return {"op": self.kind, "text": self.value}
        return {"op": self.kind, "old": self.value, "new": self.replacement}


@dataclass(frozen=True, slots=True)
class ContentTransform:
    """One named, bounded sequence of local content operations."""

    operations: tuple[TransformOperation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operations, tuple) or not self.operations:
            raise ValueError("a content transform requires a non-empty operation tuple")
        if len(self.operations) > MAX_TRANSFORM_OPERATIONS:
            raise ValueError(
                f"a content transform supports at most {MAX_TRANSFORM_OPERATIONS} operations"
            )
        if any(type(operation) is not TransformOperation for operation in self.operations):
            raise TypeError("content transform operations must be TransformOperation values")
        if _canonical_plan_size(self.operations) > MAX_TRANSFORM_PLAN_BYTES:
            raise ValueError(
                f"content transform plan exceeds {MAX_TRANSFORM_PLAN_BYTES} canonical JSON bytes"
            )

    def canonical_json(self) -> str:
        """Return a stable identity embedded in compiled model expressions."""

        return "[" + ",".join(_canonical_operation_json(op) for op in self.operations) + "]"

    @classmethod
    def from_canonical_json(cls, value: str) -> ContentTransform:
        """Decode and strictly validate one compiled transform identity."""

        if not isinstance(value, str):
            raise TypeError("compiled content transform must be a string")
        plan_bytes = _bounded_utf8_size(value)
        if plan_bytes > MAX_TRANSFORM_PLAN_BYTES:
            raise ValueError(
                f"compiled content transform exceeds {MAX_TRANSFORM_PLAN_BYTES} canonical JSON bytes"
            )
        try:
            raw = json.loads(value)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("compiled content transform is not valid JSON") from exc
        if not isinstance(raw, list) or not raw:
            raise ValueError("compiled content transform must be a non-empty operation list")
        if len(raw) > MAX_TRANSFORM_OPERATIONS:
            raise ValueError(
                f"compiled content transform exceeds {MAX_TRANSFORM_OPERATIONS} operations"
            )
        operations = []
        for index, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise TypeError(f"compiled content transform operation {index} must be a mapping")
            kind = entry.get("op")
            if kind in {"strip", "casefold"} and set(entry) == {"op"}:
                operations.append(TransformOperation(kind))
            elif kind in {"prefix", "suffix"} and set(entry) == {"op", "text"}:
                operations.append(TransformOperation(kind, entry["text"]))
            elif kind == "replace" and set(entry) == {"op", "old", "new"}:
                operations.append(TransformOperation(kind, entry["old"], entry["new"]))
            else:
                raise ValueError(
                    f"compiled content transform operation {index} has unsupported fields"
                )
        return cls(tuple(operations))


@dataclass(frozen=True, slots=True)
class ContentSourceRange:
    """One original message content range read by transformed characters."""

    message_index: int
    part_index: int | None
    start: int
    end: int
    message_start: int
    message_end: int

    def __post_init__(self) -> None:
        values = (
            self.message_index,
            self.start,
            self.end,
            self.message_start,
            self.message_end,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("content source range coordinates must be integers")
        if self.part_index is not None and (
            isinstance(self.part_index, bool)
            or not isinstance(self.part_index, int)
            or self.part_index < 0
        ):
            raise TypeError("content source range part_index must be a non-negative integer")
        if (
            self.message_index < 0
            or self.start < 0
            or self.end <= self.start
            or self.message_start < 0
            or self.message_end <= self.message_start
        ):
            raise ValueError("content source ranges must be non-empty and non-negative")

    def to_dict(self) -> dict[str, int]:
        """Return the public source coordinates."""

        value = {
            "message_index": self.message_index,
            "start": self.start,
            "end": self.end,
        }
        if self.part_index is not None:
            value["part_index"] = self.part_index
        return value


@dataclass(frozen=True, slots=True)
class ProvenanceSegment:
    """One transformed output range and the source ranges it represents."""

    start: int
    end: int
    sources: tuple[ContentSourceRange, ...]
    kind: MappingKind = "linear"

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError("provenance segments must cover a non-empty output range")
        if self.kind not in {"linear", "whole"}:
            raise ValueError(f"unsupported provenance mapping kind {self.kind!r}")
        if not isinstance(self.sources, tuple) or any(
            not isinstance(source, ContentSourceRange) for source in self.sources
        ):
            raise TypeError("provenance segment sources must be ContentSourceRange values")


def slice_provenance(
    segments: tuple[ProvenanceSegment, ...], start: int, end: int
) -> tuple[ProvenanceSegment, ...]:
    """Return one provenance slice rebased to output offset zero."""

    sliced = []
    for segment in segments:
        low, high = max(start, segment.start), min(end, segment.end)
        if low >= high:
            continue
        sources = segment.sources
        if (
            segment.kind == "linear"
            and len(sources) == 1
            and segment.end - segment.start == sources[0].end - sources[0].start
        ):
            source = sources[0]
            offset = low - segment.start
            length = high - low
            sources = (
                ContentSourceRange(
                    source.message_index,
                    source.part_index,
                    source.start + offset,
                    source.start + offset + length,
                    source.message_start + offset,
                    source.message_start + offset + length,
                ),
            )
        sliced.append(ProvenanceSegment(low - start, high - start, sources, segment.kind))
    return tuple(sliced)


def source_mapping_for(
    segments: tuple[ProvenanceSegment, ...],
    start: int,
    end: int,
) -> tuple[tuple[ContentSourceRange, ...], bool]:
    """Return exact mapped ranges plus whether any requested character is unmapped."""

    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise ValueError("transformed source ranges require non-negative start < end offsets")
    output_end = segments[-1].end if segments else 0
    if end > output_end:
        raise ValueError("transformed range is outside message content")
    ranges: list[ContentSourceRange] = []
    covered = start
    unmapped = False
    for segment in segments:
        low, high = max(start, segment.start), min(end, segment.end)
        if low >= high:
            continue
        if segment.start > covered:
            unmapped = True
        covered = max(covered, high)
        if not segment.sources:
            unmapped = True
        sources = segment.sources
        if (
            segment.kind == "linear"
            and len(sources) == 1
            and segment.end - segment.start == sources[0].end - sources[0].start
        ):
            source = sources[0]
            offset = low - segment.start
            length = high - low
            sources = (
                ContentSourceRange(
                    source.message_index,
                    source.part_index,
                    source.start + offset,
                    source.start + offset + length,
                    source.message_start + offset,
                    source.message_start + offset + length,
                ),
            )
        for source in sources:
            last = ranges[-1] if ranges else None
            if (
                last is not None
                and last.message_index == source.message_index
                and last.part_index == source.part_index
                and last.end == source.start
                and last.message_end == source.message_start
            ):
                ranges[-1] = ContentSourceRange(
                    last.message_index,
                    last.part_index,
                    last.start,
                    source.end,
                    last.message_start,
                    source.message_end,
                )
            elif source not in ranges:
                ranges.append(source)
    if covered < end:
        unmapped = True
    return tuple(ranges), unmapped


def source_ranges_for(
    segments: tuple[ProvenanceSegment, ...],
    start: int,
    end: int,
    *,
    require_mappable: bool = False,
) -> tuple[ContentSourceRange, ...]:
    """Map one transformed range to original content-part ranges."""

    ranges, unmapped = source_mapping_for(segments, start, end)
    if require_mappable and unmapped:
        raise ValueError("transformed range includes inserted text with no source position")
    if require_mappable and not ranges:
        raise ValueError("transformed range has no source position")
    return ranges
