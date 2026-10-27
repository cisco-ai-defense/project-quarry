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

"""The JSON Schema subset a model alias may declare for its completions.

A rule reads a schema completion's fields by name (``.json.label``), so the schema must make
every field path known and typed: each object lists all its properties as required and
closes itself with ``additionalProperties: false``, and only plain types appear. This is the
subset strict structured outputs accept. Keywords that make paths open or conditional
(``$ref``, ``anyOf``, ``oneOf``, open maps) are rejected, and so are ``pattern`` and
``format``, which a validator would run as a Python regular expression over model output.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from typing import Any

MAX_DEPTH = 10
# OpenAI's documented Structured Outputs limits, so a schema the provider would reject fails
# when the config loads instead of on the first request.
MAX_PROPERTIES = 5000
MAX_ENUM_VALUES = 1000
MAX_STRING_LENGTH = 120_000
MAX_LARGE_ENUM_LENGTH = 15_000
_LARGE_ENUM = 250
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# A name the rule grammar reserves, so `.json.<name>` could never read it.
_RESERVED_NAMES = frozenset({"right_after"})
# JSON Schema scalar types and the model-expression type a rule reads them as.
_SCALAR_TYPES = {"string": "string", "integer": "int", "number": "float", "boolean": "bool"}
_ANNOTATIONS = frozenset({"title", "description"})
_SCALAR_KEYWORDS = {
    "string": frozenset({"enum"}),
    "integer": frozenset({"minimum", "maximum"}),
    "number": frozenset({"minimum", "maximum"}),
    "boolean": frozenset(),
}
# The only list type: a rule compares a nullable string with a string or null, and a nullable
# number or boolean has no comparison a rule could write.
_NULLABLE_STRING = ["string", "null"]


def response_schema_text(schema: object) -> str:
    """Check one response schema and return it as compact JSON text.

    The text keeps property order, which strict structured outputs follow. Raises
    ``ValueError`` naming the JSON pointer of the first unsupported construct.
    """

    try:
        _require_json(schema, "")
    except RecursionError:
        raise ValueError("nests too deeply or refers to itself") from None
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("must be an object schema (type: object) at /")
    _check(schema, "", 0)
    _check_size(schema)
    return json.dumps(schema, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def response_schema_types(alias: str, schema_text: str) -> tuple[str, dict[str, dict[str, str]]]:
    """Return the model-expression type of a schema's root and the fields of each object.

    Each object node gets its own type name, so a field read is checked against exactly the
    object it reads from.
    """

    tables: dict[str, dict[str, str]] = {}
    ordinal = itertools.count()

    def type_of(node: dict[str, Any]) -> str:
        kind = node["type"]
        if kind == _NULLABLE_STRING:
            return "string?"
        if kind == "object":
            name = json_object_type(alias, next(ordinal))
            tables[name] = {field: type_of(child) for field, child in node["properties"].items()}
            return name
        if kind == "array":
            return f"{type_of(node['items'])}[]"
        return _SCALAR_TYPES[kind]

    return type_of(json.loads(schema_text)), tables


def json_object_type(alias: str, ordinal: int) -> str:
    """Name the model-expression type of one object in an alias's response schema."""

    return f"json object of model {alias!r} #{ordinal}"


def _require_json(value: object, pointer: str) -> None:
    """Reject YAML values with no JSON form, so the schema means what was written."""

    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"key {key!r} at {pointer or '/'} is not a string")
            _require_json(item, f"{pointer}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_json(item, f"{pointer}/{index}")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"value at {pointer or '/'} is not a finite number")
    elif not (value is None or isinstance(value, str | bool | int)):
        raise ValueError(f"value at {pointer or '/'} has no JSON form")


def _only(node: dict[str, Any], allowed: frozenset[str], pointer: str) -> None:
    unsupported = sorted(set(node) - allowed)
    if unsupported:
        raise ValueError(
            f"{unsupported[0]!r} at {pointer or '/'} is not supported; supported here: "
            + ", ".join(sorted(allowed))
        )


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _check(node: object, pointer: str, depth: int) -> None:
    where = pointer or "/"
    if depth > MAX_DEPTH:
        raise ValueError(f"nests deeper than {MAX_DEPTH} levels at {where}")
    if not isinstance(node, dict):
        raise ValueError(f"must be a schema object at {where}")
    _check_annotations(node, where)
    kind = node.get("type")
    if kind is None:
        keywords = sorted(set(node) - _ANNOTATIONS)
        if keywords:
            raise ValueError(
                f"{keywords[0]!r} at {where} is not supported; every schema needs a plain type"
            )
        raise ValueError(f"schema at {where} needs a type")
    if isinstance(kind, list):
        if kind != _NULLABLE_STRING:
            raise ValueError(
                f"type {kind!r} at {where} is not supported; the only list type is "
                '["string", "null"]'
            )
        _only(node, frozenset({"type"}) | _ANNOTATIONS, pointer)
        return
    if not isinstance(kind, str):
        raise ValueError(f"type at {where} must be a type name")
    if kind == "object":
        _only(
            node,
            frozenset({"type", "properties", "required", "additionalProperties"}) | _ANNOTATIONS,
            pointer,
        )
        properties = node.get("properties")
        if not isinstance(properties, dict) or not properties:
            raise ValueError(f"object at {where} needs at least one property")
        for name in properties:
            if not _IDENTIFIER.fullmatch(name):
                raise ValueError(f"property {name!r} at {where} is not a rule identifier")
            if name in _RESERVED_NAMES:
                raise ValueError(f"property {name!r} at {where} is reserved by the rule grammar")
        required = node.get("required")
        if (
            not isinstance(required, list)
            or not all(isinstance(name, str) for name in required)
            or sorted(required) != sorted(properties)
        ):
            raise ValueError(f"object at {where} must list every property in required")
        if node.get("additionalProperties") is not False:
            raise ValueError(f"object at {where} needs additionalProperties: false")
        for name, child in properties.items():
            _check(child, f"{pointer}/properties/{name}", depth + 1)
        return
    if kind == "array":
        _only(node, frozenset({"type", "items", "minItems", "maxItems"}) | _ANNOTATIONS, pointer)
        for bound in ("minItems", "maxItems"):
            value = node.get(bound)
            # Present means set: an explicit null is not a bound.
            if bound in node and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{bound} at {where} must be a non-negative integer")
        if node.get("minItems", 0) > node.get("maxItems", node.get("minItems", 0)):
            raise ValueError(f"minItems at {where} exceeds maxItems")
        if "items" not in node:
            raise ValueError(f"array at {where} needs items")
        _check(node["items"], f"{pointer}/items", depth + 1)
        return
    if kind in _SCALAR_TYPES:
        _only(node, frozenset({"type"}) | _ANNOTATIONS | _SCALAR_KEYWORDS[kind], pointer)
        if "enum" in node:
            values = node["enum"]
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(item, str) for item in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"enum at {where} must list one or more distinct strings")
            if len(values) > _LARGE_ENUM and sum(map(len, values)) > MAX_LARGE_ENUM_LENGTH:
                raise ValueError(
                    f"enum at {where} has more than {_LARGE_ENUM} values totalling over "
                    f"{MAX_LARGE_ENUM_LENGTH} characters"
                )
        _check_bounds(node, where)
        return
    raise ValueError(
        f"type {kind!r} at {where} is not supported; supported: object, array, "
        + ", ".join(_SCALAR_TYPES)
    )


def _check_bounds(node: dict[str, Any], where: str) -> None:
    for bound in ("minimum", "maximum"):
        if bound in node and not _number(node[bound]):
            raise ValueError(f"{bound} at {where} must be a number")
    if "minimum" in node and "maximum" in node and node["minimum"] > node["maximum"]:
        # No reply could match, so every completion would be refused.
        raise ValueError(f"minimum at {where} exceeds maximum")


def _check_annotations(node: dict[str, Any], where: str) -> None:
    for annotation in sorted(_ANNOTATIONS):
        if annotation in node and not isinstance(node[annotation], str):
            raise ValueError(f"{annotation} at {where} must be a string")


def _check_size(schema: dict[str, Any]) -> None:
    """Reject a schema larger than the provider accepts."""

    properties = enum_values = string_length = 0
    pending = [schema]
    while pending:
        node = pending.pop()
        names = node.get("properties", {})
        properties += len(names)
        string_length += sum(map(len, names))
        values = node.get("enum", ())
        enum_values += len(values)
        string_length += sum(map(len, values))
        pending.extend(names.values())
        if "items" in node:
            pending.append(node["items"])
    for count, limit, what in (
        (properties, MAX_PROPERTIES, "object properties"),
        (enum_values, MAX_ENUM_VALUES, "enum values"),
        (string_length, MAX_STRING_LENGTH, "characters of property names and enum values"),
    ):
        if count > limit:
            raise ValueError(f"has {count} {what}; the limit is {limit}")
