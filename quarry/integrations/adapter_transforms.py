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

"""Validate and retain adapter transform replies without retaining copied input text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.adapter_transforms import TransformResult, TransformRun
from quarry.types.exceptions import ModelError
from quarry.types.transforms import (
    MAX_TRANSFORM_EXPANSION_RATIO,
    MAX_TRANSFORM_INPUT_CHARACTERS,
    MAX_TRANSFORM_INSERTED_CHARACTERS,
    MAX_TRANSFORM_MAPPING_SEGMENTS,
    MAX_TRANSFORM_OUTPUT_CHARACTERS,
)


def transform_error(name: str, code: str, reason: str) -> ModelError:
    """Name a transform failure without turning it into a provider refusal."""

    return ModelError(f"Transform {name!r}: {reason}", {"code": code, "transform": name})


def _unicode(text: str) -> None:
    for index, character in enumerate(text):
        if index % 256 == 0:
            raise_if_evaluation_expired()
        if 0xD800 <= ord(character) <= 0xDFFF:
            raise ValueError("text must contain valid Unicode")


def transform_unit_snapshot(unit: object, *, _check_unicode: bool = True) -> dict:
    """Copy one exact wire part, bounding its graph before any cache-key encoding.

    Only the three public content part shapes are accepted. Batch preflight may defer
    Unicode checking until each part's mandatory admission check, before hashing or dispatch.
    The adapter may mutate its payload, so cache keys and validation use a private snapshot.
    """

    if type(unit) is not dict:
        raise ValueError("transform input must be a wire content part")
    unit = cast(dict[str, Any], unit)
    if type(unit.get("type")) is not str:
        raise ValueError("transform input must be a wire content part")
    part_type = unit["type"]
    if part_type == "text" and set(unit) == {"type", "text"}:
        text = unit["text"]
        if type(text) is not str or len(text) > MAX_TRANSFORM_INPUT_CHARACTERS:
            raise ValueError("transform text input exceeds its character limit or is not text")
        if _check_unicode:
            _unicode(text)
        return {"type": "text", "text": text}
    if part_type == "image_url" and set(unit) == {"type", "image_url"}:
        image = unit["image_url"]
        if type(image) is not dict or set(image) not in (
            {"url"},
            {"url", "detail"},
        ):
            raise ValueError("transform image input requires url and optional detail")
        values = cast(tuple[str, ...], tuple(image.values()))
        if any(type(value) is not str for value in values):
            raise ValueError("transform image input fields must be strings")
        if image.get("detail", "auto") not in {"auto", "low", "high"}:
            raise ValueError("transform image detail must be auto, low or high")
        if sum(map(len, values)) > MAX_TRANSFORM_INPUT_CHARACTERS:
            raise ValueError("transform image input exceeds its character limit")
        if _check_unicode:
            for value in values:
                _unicode(value)
        return {"type": "image_url", "image_url": dict(image)}
    if part_type == "input_audio" and set(unit) == {"type", "input_audio"}:
        audio = unit["input_audio"]
        if type(audio) is not dict or set(audio) != {"data", "format"}:
            raise ValueError("transform audio input requires data and format")
        if any(type(value) is not str for value in audio.values()):
            raise ValueError("transform audio input fields must be strings")
        values = cast(tuple[str, ...], tuple(audio.values()))
        if sum(map(len, values)) > MAX_TRANSFORM_INPUT_CHARACTERS:
            raise ValueError("transform audio input exceeds its character limit")
        if _check_unicode:
            for value in values:
                _unicode(value)
        return {"type": "input_audio", "input_audio": dict(audio)}
    raise ValueError("transform input must be a supported text, image_url or input_audio part")


@dataclass(frozen=True, slots=True)
class KeptTransform:
    """Validated reply: only newly produced text, plus immutable mapping and copy markers."""

    inserted: str
    runs: tuple[TransformRun, ...]
    copied: tuple[bool, ...]


def keep_transform(raw: object, unit: dict, name: str) -> KeptTransform:
    """Validate against the input and compact the result before cache admission."""

    try:
        return _keep_transform(raw, unit)
    except (TypeError, ValueError) as exc:
        raise transform_error(name, "invalid_output", str(exc)) from exc


def _keep_transform(raw: object, unit: dict) -> KeptTransform:
    raise_if_evaluation_expired()
    source = unit.get("text") if unit["type"] == "text" else None
    if type(raw) is str:
        text = raw
        if not text:
            runs = ()
        elif source:
            runs = (
                TransformRun(0, len(text), 0, len(source), "linear" if text == source else "whole"),
            )
        else:
            runs = (TransformRun(0, len(text)),)
    elif type(raw) is TransformResult:
        text, runs = raw.text, raw.runs
    else:
        raise TypeError("adapter must return text or TransformResult")
    if type(text) is not str:
        raise TypeError("transform result text must be a string")
    if len(text) > MAX_TRANSFORM_OUTPUT_CHARACTERS:
        raise ValueError("transform output exceeds its character limit")
    if source is not None and len(text) > (
        len(source) * MAX_TRANSFORM_EXPANSION_RATIO + MAX_TRANSFORM_INSERTED_CHARACTERS
    ):
        raise ValueError("transform output exceeds its expansion limit")
    _unicode(text)
    if type(runs) is not tuple or len(runs) > MAX_TRANSFORM_MAPPING_SEGMENTS:
        raise ValueError("transform runs must be a bounded tuple")
    cursor = 0
    inserted: list[str] = []
    copied: list[bool] = []
    checked: list[TransformRun] = []
    for index, run in enumerate(runs):
        if index % 256 == 0:
            raise_if_evaluation_expired()
        if type(run) is not TransformRun:
            raise TypeError("transform runs must be TransformRun values")
        if type(run.start) is not int or type(run.end) is not int:
            raise TypeError("transform output offsets must be integers")
        if run.start != cursor or not cursor < run.end <= len(text):
            raise ValueError("transform runs must cover output in order without gaps or overlap")
        if type(run.kind) is not str or run.kind not in {"linear", "whole"}:
            raise ValueError("transform run kind must be linear or whole")
        if run.source_start is None and run.source_end is None:
            if run.kind != "whole":
                raise ValueError("an unmapped transform run cannot be linear")
        else:
            if type(run.source_start) is not int or type(run.source_end) is not int:
                raise TypeError("transform source offsets must both be integers or both absent")
            if source is None or not 0 <= run.source_start < run.source_end <= len(source):
                raise ValueError("transform source offsets must address the input text")
            if run.kind == "linear" and run.end - run.start != run.source_end - run.source_start:
                raise ValueError("linear transform runs must have equal input and output lengths")
        piece = text[run.start : run.end]
        is_copy = (
            run.kind == "linear"
            and source is not None
            and piece == source[run.source_start : run.source_end]
        )
        if not is_copy:
            inserted.append(piece)
        copied.append(is_copy)
        checked.append(run)
        cursor = run.end
    if cursor != len(text):
        raise ValueError("transform runs must cover all output text")
    raise_if_evaluation_expired()
    return KeptTransform("".join(inserted), tuple(checked), tuple(copied))


def read_transform(kept: KeptTransform, unit: dict) -> TransformResult:
    """Reconstruct a caller's immutable result from compact kept text and its own input."""

    source = unit.get("text", "")
    cursor = 0
    pieces: list[str] = []
    for index, (run, copied) in enumerate(zip(kept.runs, kept.copied, strict=True)):
        if index % 256 == 0:
            raise_if_evaluation_expired()
        if copied:
            pieces.append(source[run.source_start : run.source_end])
        else:
            end = cursor + run.end - run.start
            pieces.append(kept.inserted[cursor:end])
            cursor = end
    raise_if_evaluation_expired()
    return TransformResult("".join(pieces), kept.runs)
