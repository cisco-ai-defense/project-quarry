# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Map decoded scalar text to character ranges in its original JSON document."""

import json
import re
from array import array
from bisect import bisect_right
from collections.abc import Iterator
from typing import Any, NamedTuple, cast

from quarry.safety.timer import raise_if_evaluation_expired

Path = tuple[str, ...]
_SPACE = re.compile(r"[ \t\n\r]*")
_SCALAR = re.compile(r"[^,\]} \t\n\r]+")
_ESCAPES = re.compile(
    r"\\(?:u[dD][89aAbB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2}"
    r'|u[0-9a-fA-F]{4}|["\\/bfnrt])'
)


class _Range(NamedTuple):
    start: int
    end: int
    string: bool


def _json_locations(text: str) -> dict[str, Any]:
    """Index a validated object's scalar tokens without recursively parsing containers."""

    root: dict[str, Any] = {}
    parents: list[dict[str, Any] | list[Any]] = [root]
    key: str | None = None
    string = json.JSONDecoder().raw_decode
    position = cast(re.Match[str], _SPACE.match(text)).end()
    assert text[position] == "{"
    position += 1
    while parents:
        raise_if_evaluation_expired()
        position = cast(re.Match[str], _SPACE.match(text, position)).end()
        start = position
        token = text[position]
        if token in ":,":
            position += 1
            continue
        if token in "}]":
            parents.pop()
            position += 1
            continue
        parent = parents[-1]
        value: Any
        if token == '"':
            decoded, position = string(text, position)
            if isinstance(parent, dict) and key is None:
                key = decoded
                continue
            value = _Range(start, position, True)
        elif token == "{":
            value = {}
            position += 1
        elif token == "[":
            value = []
            position += 1
        else:
            position = cast(re.Match[str], _SCALAR.match(text, position)).end()
            value = None if token == "n" else _Range(start, position, False)
        if isinstance(parent, dict):
            assert key is not None
            parent[key] = value
            key = None
        else:
            parent.append(value)
        if isinstance(value, (dict, list)):
            parents.append(value)
    return root


def _ranges(root: dict[str, Any]) -> Iterator[tuple[Path, _Range]]:
    pending: list[tuple[Path, Iterator[tuple[str | int, Any]]]] = [((), iter(root.items()))]
    while pending:
        path, children = pending[-1]
        for key, value in children:
            raise_if_evaluation_expired()
            child = (*path, str(key))
            if isinstance(value, dict):
                pending.append((child, iter(value.items())))
                break
            if isinstance(value, list):
                pending.append((child, enumerate(value)))
                break
            if value is not None:
                yield child, value
        else:
            pending.pop()


def _string_knots(raw: str, start: int, field_end: int) -> tuple[array, array]:
    """Boundaries after complete escapes of an already located JSON string."""
    base = start + 1
    decoded, source = array("q", [0]), array("q", [base])
    removed = 0
    for escaped in _ESCAPES.finditer(raw, base, field_end - 1):
        raise_if_evaluation_expired()
        end = escaped.end()
        removed += end - escaped.start() - 1
        decoded.append(end - base - removed)
        source.append(end)
    return decoded, source


class JSONSource:
    """A parsed value's raw character locations, and escape boundaries read on demand."""

    __slots__ = ("raw", "_fields", "_strings")

    def __init__(self, raw: str) -> None:
        self.raw = raw
        self._fields = dict(_ranges(_json_locations(raw)))
        self._strings: dict[Path, int | tuple[array, array]] = {}

    def bounds(self, path: Path, text: str, start: int, end: int) -> tuple[int, int]:
        """The raw characters that produced text[start:end] of one decoded scalar.

        A number whose spelling changed while decoding depends on its whole JSON token.
        String offsets follow the stdlib's decoder, including escapes and surrogate pairs.
        """
        field = self._fields[path]
        raw = self.raw
        if not field.string:
            return (
                (field.start + start, field.start + end)
                if raw[field.start : field.end] == text
                else (field.start, field.end)
            )

        def offset(at: int) -> int:
            # The decoder already located both ends. Neither needs an escape map.
            if at == 0:
                return field.start + 1
            if at == len(text):
                return field.end - 1
            knots = self._strings.get(path)
            if knots is None:
                knots = self._strings[path] = raw.find("\\", field.start + 1, field.end - 1)
            if isinstance(knots, int):
                if knots < 0 or at <= knots - field.start - 1:
                    return field.start + 1 + at
                knots = self._strings[path] = _string_knots(raw, field.start, field.end)
            decoded, source = knots
            index = bisect_right(decoded, at) - 1
            return source[index] + at - decoded[index]

        bounds = offset(start), offset(end)
        raise_if_evaluation_expired()
        return bounds
