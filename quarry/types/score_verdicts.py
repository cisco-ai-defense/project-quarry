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

"""Bounded, immutable score settings and strict JSON verdict extraction."""

import json
import math
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass

MAX_SCORE_LABELS = 16
MAX_SCORE_LABEL_CHARS = 256
MAX_SCORE_REPLY_CHARS = 65_536
MAX_SCORE_JSON_DEPTH = 16
MAX_SCORE_JSON_VALUES = 4096
ScoreLabels = str | tuple[str, ...]
ScoreArguments = tuple[tuple[str, int | float], ...]
ScorePath = tuple[str | int, ...]


def score_labels(value: object) -> ScoreLabels | None:
    """Keep legacy strings; canonicalize a declared label set without interpreting strings."""

    if value is None:
        return None
    single = type(value) is str
    if single:
        values = (value,)
    elif (type(value) is list or type(value) is tuple) and 1 <= len(value) <= MAX_SCORE_LABELS:
        values = value
    else:
        raise ValueError("score labels must be a string or a list of 1 through 16 strings")
    normalized = []
    for label in values:
        if type(label) is not str or not 1 <= len(label) <= MAX_SCORE_LABEL_CHARS:
            raise ValueError("each score label must contain 1 through 256 characters")
        label = label.strip()
        if not label or any(
            unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"} for char in label
        ):
            raise ValueError("score labels must be nonempty text without control characters")
        normalized.append(label)
    if len(set(normalized)) != len(normalized):
        raise ValueError("score label sets must not contain duplicates")
    return normalized[0] if single else tuple(sorted(normalized))


def label_set(value: ScoreLabels | None) -> tuple[str, ...]:
    return () if value is None else (value,) if isinstance(value, str) else value


def validate_score_sides(negative: ScoreLabels | None, positive: ScoreLabels | None) -> None:
    for left in label_set(negative):
        for right in label_set(positive):
            if left == right:
                raise ValueError("score positive and negative tokens must differ")
            if left.startswith(right) or right.startswith(left):
                raise ValueError("score labels on opposite sides must not prefix one another")


def score_arguments(value: object) -> ScoreArguments | None:
    """Copy only finite sampling settings; transport, credentials and arbitrary kwargs stay out."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        if len(value) > 4:
            raise ValueError("score arguments contain unsupported settings")
        entries = list(value.items())
    elif type(value) is tuple and len(value) <= 4:
        entries = list(value)
    else:
        raise ValueError("score arguments must be a mapping of supported numeric settings")
    result: dict[str, int | float] = {}
    for entry in entries:
        if not (type(entry) is tuple or type(entry) is list) or len(entry) != 2:
            raise ValueError("score arguments must contain named numeric settings")
        name, item = entry
        if type(name) is not str or name not in {"max_tokens", "temperature", "top_p", "seed"}:
            raise ValueError("score arguments contain an unsupported setting")
        if name in result or not (type(item) is int or type(item) is float):
            raise ValueError("score arguments must contain unique finite numeric settings")
        if name == "max_tokens":
            valid = type(item) is int and 1 <= item <= 4096
        elif name == "seed":
            valid = type(item) is int and -(2**63) <= item < 2**63
        else:
            try:
                valid = math.isfinite(item) and (
                    0 <= item <= 2 if name == "temperature" else 0 < item <= 1
                )
            except OverflowError:
                valid = False
        if not valid:
            raise ValueError("score argument is outside its finite supported range")
        result[name] = item
    return tuple(sorted(result.items())) or None


def score_path(value: object) -> ScorePath:
    """A dotted key path or explicit key/index list, with no expressions or attribute access."""

    if type(value) is str:
        if not 1 <= len(value) <= 512:
            raise ValueError("score JSON path exceeds its length limit")
        parts = value.split(".")
    elif type(value) is tuple or type(value) is list:
        parts = value
    else:
        raise ValueError("score JSON path must be a dotted string or key/index list")
    if not 1 <= len(parts) <= MAX_SCORE_JSON_DEPTH:
        raise ValueError("score JSON path must contain 1 through 16 components")
    result: list[str | int] = []
    total = 0
    for part in parts:
        if type(part) is str and 1 <= len(part) <= 128:
            total += len(part)
            if any(unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"} for char in part):
                raise ValueError("score JSON path contains an invalid key")
        elif type(part) is int and 0 <= part < MAX_SCORE_JSON_VALUES:
            total += 4
        else:
            raise ValueError(
                "score JSON path components must be bounded keys or nonnegative indexes"
            )
        result.append(part)
    if total > 512:
        raise ValueError("score JSON path exceeds its aggregate length limit")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ScoreVerdictConfig:
    """Read a strict JSON label and optionally the probability of that selected label."""

    label_path: ScorePath
    confidence_path: ScorePath | None = None
    max_chars: int = 16_384

    def __post_init__(self) -> None:
        object.__setattr__(self, "label_path", score_path(self.label_path))
        if self.confidence_path is not None:
            object.__setattr__(self, "confidence_path", score_path(self.confidence_path))
            common = min(len(self.label_path), len(self.confidence_path))
            if self.label_path[:common] == self.confidence_path[:common]:
                raise ValueError("score label and confidence paths must select separate values")
        if type(self.max_chars) is not int or not 16 <= self.max_chars <= MAX_SCORE_REPLY_CHARS:
            raise ValueError("score JSON max_chars must be an integer from 16 through 65536")


def json_verdict(
    content: str,
    config: ScoreVerdictConfig,
    negative: tuple[str, ...],
    positive: tuple[str, ...],
    check_deadline: Callable[[], None],
) -> float:
    """Reject ambiguous, oversized and nonfinite JSON before reading configured paths."""

    if len(content) > config.max_chars:
        raise ValueError("score JSON exceeds its character limit")
    # Bound recursive parsing before json.loads. Braces within escaped JSON strings do not
    # contribute depth; malformed quoting is subsequently rejected by the JSON decoder.
    depth = 0
    quoted = escaped = False
    for index, char in enumerate(content):
        if index % 256 == 0:
            check_deadline()
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_SCORE_JSON_DEPTH:
                raise ValueError("score JSON exceeds its depth limit")
        elif char in "]}":
            depth -= 1

    def integer(text: str) -> int:
        if len(text) > 128:
            raise ValueError("score JSON integer exceeds its digit limit")
        return int(text)

    def number(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("score JSON numbers must be finite")
        return value

    def constant(_text: str) -> None:
        raise ValueError("score JSON constants must be finite JSON values")

    pairs_seen = 0

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        nonlocal pairs_seen
        pairs_seen += len(pairs)
        if pairs_seen > MAX_SCORE_JSON_VALUES:
            raise ValueError("score JSON exceeds its value limit")
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("score JSON contains a duplicate key")
            result[key] = value
        return result

    value = json.loads(
        content,
        parse_int=integer,
        parse_float=number,
        parse_constant=constant,
        object_pairs_hook=object_pairs,
    )
    pending = [value]
    seen = 0
    while pending:
        item = pending.pop()
        seen += 1
        if seen > MAX_SCORE_JSON_VALUES:
            raise ValueError("score JSON exceeds its value limit")
        if seen % 256 == 0:
            check_deadline()
        if type(item) is dict:
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)

    def read(path: ScorePath) -> object:
        item = value
        for part in path:
            if (
                type(item) is dict
                and type(part) is str
                and part in item
                or type(item) is list
                and type(part) is int
                and part < len(item)
            ):
                item = item[part]
            else:
                raise ValueError("score JSON does not contain its configured path")
        return item

    label = read(config.label_path)
    if type(label) is not str or label not in negative + positive:
        raise ValueError("score JSON contains an unknown verdict label")
    confidence: float = 1.0
    if config.confidence_path is not None:
        supplied = read(config.confidence_path)
        if not (type(supplied) is int or type(supplied) is float) or not 0 <= supplied <= 1:
            raise ValueError("score JSON confidence must be a finite number from zero through one")
        confidence = float(supplied)
    check_deadline()
    return confidence if label in positive else 1.0 - confidence
