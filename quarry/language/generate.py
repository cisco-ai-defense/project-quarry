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

"""Generate Quarry's runtime catalog, reference, and examples from code."""

from __future__ import annotations

import inspect
import json
import re
import types
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from quarry import __version__
from quarry.core.conversation import Conversation
from quarry.dsl import functions
from quarry.dsl.grammar import (
    VARIABLE_RESERVED_KEYWORDS,
    grammar_literals,
    grammar_production_for_handler,
    operator_precedence,
    transformer_handler_names,
)
from quarry.dsl.syntax import SYNTAX_EXCLUSIONS, TOKEN_RECORDS
from quarry.dsl.transformer import DSLTransformer
from quarry.engine.compiler import _Compiler
from quarry.engine.contracts import COMPILER_VERSION
from quarry.language.compatibility import COMPATIBILITY_RECORDS, MAX_COMPATIBILITY_RECORDS
from quarry.language.example_records import EXAMPLE_RECORDS
from quarry.language.metadata import (
    COMPILER_TAGS,
    PUBLIC_TYPES,
    RESOURCE_GROUPS,
    SURFACE_FIELD_TYPES,
    TYPE_CONVERSIONS,
    CompilerFormMetadata,
    ConstructMetadata,
    DSLParameter,
    FunctionMetadata,
    SurfaceFieldExclusion,
    SurfaceFieldMetadata,
    SurfaceMetadata,
)
from quarry.types.messages import Messages
from quarry.types.text_resources import TextRef

FORMAT_VERSION = "1.1"

_SIMPLE_PYTHON_TYPES = {
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "number",
}
_COLLECTION_ELEMENT_TYPES = {
    "Message": "collection<message>",
    "MessageType": "collection<message>",
    "ToolCall": "collection<tool_call>",
    "ToolMessage": "collection<tool_response>",
    "Tool": "collection<tool_definition>",
}
_FORBIDDEN_PUBLIC_DESCRIPTION_WORDS = (
    "ctx",
    "selector",
    "Messages",
    "Conversation",
    "delegate",
    "FunctionCall",
    "Lark",
    "LTLf",
    "arg-shift",
    "_index(",
)


def _source_owners() -> tuple[Any, ...]:
    """Return the implementation owners inspected by the generator."""

    return (Conversation, Messages, TextRef, functions, DSLTransformer, _Compiler)


def _annotated_targets(owner: Any):
    """Yield metadata-bearing objects declared directly on one owner."""

    if hasattr(owner, "__quarry_language_metadata__"):
        yield owner
    for member in vars(owner).values():
        if isinstance(member, property):
            member = member.fget
        elif isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        if (
            isinstance(owner, types.ModuleType)
            and getattr(member, "__module__", None) != owner.__name__
        ):
            continue
        if member is not None and hasattr(member, "__quarry_language_metadata__"):
            yield member


def _annotation_parts(annotation: Any) -> tuple[Any, DSLParameter | None]:
    if get_origin(annotation) is Annotated:
        base, *extras = get_args(annotation)
        marker = next((extra for extra in extras if isinstance(extra, DSLParameter)), None)
        return base, marker
    return annotation, None


def _type_name(annotation: Any) -> str:
    """Map a Python annotation to one closed public Quarry type."""

    annotation, _ = _annotation_parts(annotation)
    if annotation in _SIMPLE_PYTHON_TYPES:
        return _SIMPLE_PYTHON_TYPES[annotation]
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        choices = [item for item in get_args(annotation) if item not in {None, type(None)}]
        if len(choices) == 1:
            return _type_name(choices[0])
        mapped = tuple(dict.fromkeys(_type_name(item) for item in choices))
        if len(mapped) == 1:
            return mapped[0]
        raise ValueError(f"Python union {annotation!r} needs explicit accepted_types metadata")
    if origin in {list, tuple, set, frozenset}:
        args = get_args(annotation)
        if not args:
            return "collection"
        element = args[0]
        element_name = getattr(element, "__name__", "")
        if element_name in _COLLECTION_ELEMENT_TYPES:
            return _COLLECTION_ELEMENT_TYPES[element_name]
        if element is str:
            return "collection<string>"
        return "collection"
    annotation_name = getattr(annotation, "__name__", "")
    if annotation_name == "Messages":
        return "collection<message>"
    raise ValueError(f"Python annotation {annotation!r} needs explicit public type metadata")


def _public_description(target: Any) -> str:
    """Return the public part of a docstring, excluding an ``Internal:`` appendix."""

    docstring = inspect.cleandoc(inspect.getdoc(target) or "")
    public = re.split(r"^Internal:\s*$", docstring, maxsplit=1, flags=re.MULTILINE)[0].strip()
    if not public:
        raise ValueError(f"{target.__module__}.{target.__qualname__} needs a public docstring")
    for word in _FORBIDDEN_PUBLIC_DESCRIPTION_WORDS:
        if word in public:
            raise ValueError(
                f"{target.__module__}.{target.__qualname__} public description contains "
                f"implementation word {word!r}; move that detail below an Internal: heading"
            )
    return re.sub(r"``([^`]+)``", r"`\1`", public)


def _summary(description: str) -> str:
    paragraph = description.split("\n\n", 1)[0]
    return " ".join(line.strip() for line in paragraph.splitlines()).strip()


def _hints(target: Any) -> dict[str, Any]:
    try:
        return get_type_hints(target, include_extras=True)
    except (NameError, TypeError):
        return dict(getattr(target, "__annotations__", {}))


def _parameters(target: Any) -> list[dict[str, Any]]:
    signature = inspect.signature(target)
    hints = _hints(target)
    records: list[dict[str, Any]] = []
    for parameter in signature.parameters.values():
        if parameter.name in {"self", "cls"}:
            continue
        annotation, marker = _annotation_parts(hints.get(parameter.name, parameter.annotation))
        if marker is not None and marker.hidden:
            continue
        variadic = parameter.kind is inspect.Parameter.VAR_POSITIONAL
        required = parameter.default is inspect.Parameter.empty and not variadic
        if marker is not None and marker.required is not None:
            required = marker.required
        accepted_types = (
            marker.accepted_types
            if marker is not None and marker.accepted_types
            else (_type_name(annotation),)
        )
        unknown = set(accepted_types) - set(PUBLIC_TYPES)
        if unknown:
            raise ValueError(f"{target.__qualname__}.{parameter.name} uses unknown types {unknown}")
        if marker is not None and marker.literal_only and marker.field_only:
            raise ValueError(
                f"{target.__qualname__}.{parameter.name} cannot be both literal-only and field-only"
            )
        default = None if parameter.default is inspect.Parameter.empty else parameter.default
        if default is not None and not isinstance(default, str | int | float | bool):
            raise ValueError(
                f"{target.__qualname__}.{parameter.name} has a non-JSON public default {default!r}"
            )
        records.append(
            {
                "name": marker.name if marker and marker.name else parameter.name,
                "accepted_types": list(accepted_types),
                "required": required,
                "variadic": variadic,
                "literal_only": bool(marker and marker.literal_only),
                "field_only": bool(marker and marker.field_only),
                "default": default,
                "minimum": marker.minimum if marker else None,
                "maximum": marker.maximum if marker else None,
                "allowed_values": list(marker.allowed_values) if marker else [],
            }
        )
    return records


def _render_types(types_: list[str]) -> str:
    return " | ".join(types_)


def _render_signature(name: str, parameters: list[dict[str, Any]], return_type: str) -> str:
    rendered: list[str] = []
    for parameter in parameters:
        prefix = "*" if parameter["variadic"] else ""
        value = f"{prefix}{parameter['name']}: {_render_types(parameter['accepted_types'])}"
        if not parameter["required"] and not parameter["variadic"]:
            value += f" = {json.dumps(parameter['default'])}"
        rendered.append(value)
    return f"{name}({', '.join(rendered)}) -> {return_type}"


def _source_name(target: Any) -> str:
    return f"{target.__module__}.{target.__qualname__}"


def _validate_metadata(
    item_id: str,
    resource_group: str | None,
    compiler_tags: tuple[str, ...],
    deprecated: bool,
    replacement: str | None,
) -> None:
    if resource_group is not None and resource_group not in RESOURCE_GROUPS:
        raise ValueError(f"{item_id} uses unknown resource group {resource_group!r}")
    unknown_tags = set(compiler_tags) - set(COMPILER_TAGS)
    if unknown_tags:
        raise ValueError(f"{item_id} uses unknown compiler tags {unknown_tags}")
    if replacement is not None and not deprecated:
        raise ValueError(f"{item_id} declares a replacement but is not deprecated")


def _function_record(target: Any, metadata: FunctionMetadata) -> dict[str, Any]:
    description = _public_description(target)
    parameters = _parameters(target)
    return_type = metadata.return_type
    if return_type is None:
        return_annotation = _hints(target).get(
            "return", inspect.signature(target).return_annotation
        )
        return_type = _type_name(return_annotation)
    if return_type not in PUBLIC_TYPES:
        raise ValueError(f"{metadata.id} uses unknown return type {return_type!r}")
    _validate_metadata(
        metadata.id,
        metadata.resource_group,
        metadata.compiler_tags,
        metadata.deprecated,
        metadata.replacement,
    )
    return {
        "id": metadata.id,
        "kind": "function",
        "name": metadata.name,
        "signature": _render_signature(metadata.name, parameters, return_type),
        "summary": _summary(description),
        "description": description,
        "parameters": parameters,
        "return_type": return_type,
        "allowed_uses": [item.value for item in metadata.allowed_uses],
        "resource_group": metadata.resource_group,
        "compiler_tags": list(metadata.compiler_tags),
        "accessor_surface": metadata.accessor_surface,
        "accessor_filter": metadata.accessor_filter,
        "implicit_receiver_count": metadata.implicit_receiver_count,
        "syntax": metadata.syntax,
        "deprecated": metadata.deprecated,
        "replacement": metadata.replacement,
        "source": _source_name(target),
    }


def _compiler_form_record(target: Any, metadata: CompilerFormMetadata) -> dict[str, Any]:
    description = _public_description(target)
    parameters = _parameters(target)
    _validate_metadata(
        metadata.id,
        metadata.resource_group,
        metadata.compiler_tags,
        metadata.deprecated,
        metadata.replacement,
    )
    return {
        "id": metadata.id,
        "kind": "compiler_form",
        "name": metadata.name,
        "signature": _render_signature(metadata.name, parameters, metadata.return_type),
        "summary": _summary(description),
        "description": description,
        "parameters": parameters,
        "return_type": metadata.return_type,
        "allowed_uses": [item.value for item in metadata.allowed_uses],
        "resource_group": metadata.resource_group,
        "compiler_tags": list(metadata.compiler_tags),
        "implicit_receiver_count": metadata.implicit_receiver_count,
        "syntax": metadata.syntax,
        "deprecated": metadata.deprecated,
        "replacement": metadata.replacement,
        "source": _source_name(target),
    }


def _surface_record(target: Any, metadata: SurfaceMetadata) -> dict[str, Any]:
    description = _public_description(target)
    fields: list[str] = []
    field_details: list[dict[str, str]] = []
    if metadata.fields_from is not None:
        declared_fields = getattr(metadata.fields_from, "DSL_FIELDS", None)
        if not isinstance(declared_fields, frozenset) or not all(
            isinstance(field, str) for field in declared_fields
        ):
            raise TypeError(
                f"{metadata.fields_from.__name__}.DSL_FIELDS must be a frozenset of strings"
            )
        declared_metadata = getattr(metadata.fields_from, "DSL_FIELD_METADATA", None)
        if not isinstance(declared_metadata, tuple) or not all(
            isinstance(field, SurfaceFieldMetadata) for field in declared_metadata
        ):
            raise TypeError(
                f"{metadata.fields_from.__name__}.DSL_FIELD_METADATA must be a tuple of "
                "SurfaceFieldMetadata records"
            )
        exclusions = getattr(metadata.fields_from, "DSL_FIELD_EXCLUSIONS", None)
        if not isinstance(exclusions, tuple) or not all(
            isinstance(field, SurfaceFieldExclusion) for field in exclusions
        ):
            raise TypeError(
                f"{metadata.fields_from.__name__}.DSL_FIELD_EXCLUSIONS must be a tuple of "
                "SurfaceFieldExclusion records"
            )
        metadata_by_name = {field.name: field for field in declared_metadata}
        exclusion_by_name = {field.name: field for field in exclusions}
        if len(metadata_by_name) != len(declared_metadata):
            raise ValueError(f"{metadata.fields_from.__name__} has duplicate field metadata")
        if len(exclusion_by_name) != len(exclusions):
            raise ValueError(f"{metadata.fields_from.__name__} has duplicate field exclusions")
        unknown_types = {
            field.public_type
            for field in declared_metadata
            if field.public_type not in SURFACE_FIELD_TYPES
        }
        if unknown_types:
            raise ValueError(
                f"{metadata.fields_from.__name__} uses unknown field types {sorted(unknown_types)}"
            )
        if any(not field.name or not field.description.strip() for field in declared_metadata):
            raise ValueError(
                f"{metadata.fields_from.__name__} field metadata needs names and descriptions"
            )
        if any(not field.name or not field.reason.strip() for field in exclusions):
            raise ValueError(
                f"{metadata.fields_from.__name__} field exclusions need names and reasons"
            )
        overlap = sorted(set(metadata_by_name) & set(exclusion_by_name))
        expected_fields = declared_fields - set(exclusion_by_name)
        missing = sorted(expected_fields - set(metadata_by_name))
        extra = sorted((set(metadata_by_name) | set(exclusion_by_name)) - declared_fields)
        if overlap or missing or extra:
            raise ValueError(
                f"{metadata.fields_from.__name__} field documentation mismatch; "
                f"overlap={overlap}, missing={missing}, extra={extra}"
            )
        fields = sorted(expected_fields)
        field_details = [
            {
                "name": field.name,
                "type": field.public_type,
                "description": field.description,
            }
            for field in sorted(declared_metadata, key=lambda item: item.name)
        ]
    return {
        "id": metadata.id,
        "kind": "surface",
        "name": metadata.name,
        "summary": _summary(description),
        "description": description,
        "public_type": metadata.public_type,
        "domain": metadata.domain,
        "allowed_uses": [item.value for item in metadata.allowed_uses],
        "fields": fields,
        "field_details": field_details,
        "model_input": metadata.model_input,
        "source": _source_name(target),
    }


def _construct_record(target: Any, metadata: ConstructMetadata) -> dict[str, Any]:

    description = _public_description(target)
    handler = target.__name__
    return {
        "id": metadata.id,
        "kind": "construct",
        "name": metadata.name,
        "summary": _summary(description),
        "description": description,
        "syntax": grammar_production_for_handler(metadata.grammar_rule or handler),
        "handler": handler,
        "source": _source_name(target),
    }


def _token_records() -> list[dict[str, Any]]:

    records = [
        {
            "id": item.id,
            "kind": "token",
            "spelling": item.spelling,
            "summary": item.summary,
            "category": item.category,
            "reserved_for_variables": item.reserved_for_variables,
            "source": "quarry.dsl.syntax.TOKEN_RECORDS",
        }
        for item in TOKEN_RECORDS
    ]
    spellings = [record["spelling"] for record in records]
    duplicates = sorted({value for value in spellings if spellings.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate token spellings: {duplicates}")
    # ``params`` parses as an identifier and is promoted to the parameter namespace by the
    # transformer. It remains a reserved public spelling even though it is not a grammar literal.
    expected = grammar_literals() | {"$", "//", "params"}
    missing = sorted(expected - set(spellings))
    extra = sorted(set(spellings) - expected)
    if missing or extra:
        raise ValueError(f"token coverage mismatch; missing={missing}, extra={extra}")
    reserved = {record["spelling"] for record in records if record["reserved_for_variables"]}
    expected_reserved = {
        "any",
        "all",
        "not",
        "and",
        "or",
        "in",
        "as",
        "where",
        "true",
        "false",
        "null",
        "params",
    }
    if reserved != expected_reserved:
        raise ValueError(
            f"reserved token mismatch; expected={sorted(expected_reserved)}, got={sorted(reserved)}"
        )
    return records


def _validate_construct_coverage(constructs: list[dict[str, Any]]) -> None:

    handlers = [item["handler"] for item in constructs]
    exclusions = [item.handler for item in SYNTAX_EXCLUSIONS]
    overlap = sorted(set(handlers) & set(exclusions))
    if overlap:
        raise ValueError(f"transformer handlers are both public and excluded: {overlap}")
    combined = (*handlers, *exclusions)
    duplicates = sorted({name for name in combined if combined.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate transformer handler records: {duplicates}")
    expected = transformer_handler_names()
    recorded = set(handlers) | set(exclusions)
    if expected != recorded:
        raise ValueError(
            "transformer handler coverage mismatch; "
            f"missing={sorted(expected - recorded)}, extra={sorted(recorded - expected)}"
        )
    rejected_forms = {record.source_form for record in COMPATIBILITY_RECORDS}
    for exclusion in SYNTAX_EXCLUSIONS:
        if exclusion.reason == "not_a_public_form" and exclusion.reference is not None:
            raise ValueError(f"{exclusion.handler} does not need an exclusion reference")
        if exclusion.reason == "rejected_spelling" and exclusion.reference not in rejected_forms:
            raise ValueError(f"{exclusion.handler} must reference a compatibility source_form")


_CALL_SPELLING = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
# A string literal or a line comment is text, not code, so a parenthesis inside one names no
# call. The grammar's STRING is ``common.ESCAPED_STRING`` and its comments run ``//`` to the
# end of the line; both are blanked before any call spelling is read out of a source form.
_NON_CODE_SPAN = re.compile(r'"(?:\\.|[^"\\\n])*"|//[^\r\n]*')


def _code_only(source_form: str) -> str:
    """Return ``source_form`` with string literals and comments blanked out.

    Each span becomes a single space rather than nothing, so blanking can never join two
    neighbouring tokens into a spelling that the author never wrote. A quote left over
    afterwards means the form has an unterminated literal, which no masking can interpret
    and no parser would accept, so it fails generation instead of being scanned raw."""

    masked = _NON_CODE_SPAN.sub(" ", source_form)
    if '"' in masked:
        raise ValueError(f"compatibility record has an unterminated string literal: {source_form}")
    return masked


def _rejected_spelling_catalog(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Publish rejected spellings so a caller can be corrected instead of guessing.

    The linker reports exactly one unknown call name, so a link-stage record is published
    with the one call spelling it contains that is not an installed callable. Deriving that
    name here keeps ``COMPATIBILITY_RECORDS`` the only place a rejected spelling is written
    down. The checks below make a future record that breaks the derivation fail generation,
    rather than silently publishing a record that can never be matched to the
    ``unknown_function`` error it exists to explain.
    """

    if len(COMPATIBILITY_RECORDS) > MAX_COMPATIBILITY_RECORDS:
        raise ValueError(
            f"the compatibility table holds {len(COMPATIBILITY_RECORDS)} records but the plugin "
            f"serves it whole up to {MAX_COMPATIBILITY_RECORDS}; raise "
            "MAX_COMPATIBILITY_RECORDS deliberately or paginate the rejected action"
        )

    callable_names = {
        item["name"] for item in items if item["kind"] in {"function", "compiler_form"}
    }
    records: list[dict[str, Any]] = []
    for record in COMPATIBILITY_RECORDS:
        rejected_call: str | None = None
        if record.rejected_by == "link":
            unknown = sorted(
                {
                    name
                    for name in _CALL_SPELLING.findall(_code_only(record.source_form))
                    if name not in callable_names
                }
            )
            if not unknown:
                raise ValueError(
                    "link-stage compatibility record calls only installed functions and would "
                    f"never reach the linker: {record.source_form}"
                )
            if len(unknown) > 1:
                raise ValueError(
                    f"link-stage compatibility record contains several unknown calls {unknown} "
                    f"but the linker reports only one: {record.source_form}"
                )
            rejected_call = unknown[0]
        records.append(
            {
                "source_form": record.source_form,
                "rejected_by": record.rejected_by,
                "replacement": record.replacement,
                "rejected_call": rejected_call,
            }
        )
    return records


def _example_catalog() -> list[dict[str, Any]]:
    """Return packaged example paths and declared checks in machine-readable form."""

    examples: list[dict[str, Any]] = []
    for record in sorted(EXAMPLE_RECORDS, key=lambda item: item.id):
        slug = record.id.removeprefix("example:")
        checks: list[dict[str, Any]] = []
        if record.matching is not None:
            checks.append(
                {
                    "input_path": f"examples/{slug}.matching.json",
                    "expected_match": True,
                }
            )
        if record.non_matching is not None:
            checks.append(
                {
                    "input_path": f"examples/{slug}.non_matching.json",
                    "expected_match": False,
                }
            )
        examples.append(
            {
                "id": record.id,
                "rule_path": f"examples/{slug}.rule",
                "covered_item_ids": sorted(record.covered_item_ids),
                "requires_backend": record.backend_required,
                "checks": checks,
            }
        )
    return examples


def _validate_examples(items: list[dict[str, Any]], examples: list[dict[str, Any]]) -> None:
    ids = [example["id"] for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("example IDs must be unique")
    if any(not item_id.startswith("example:") for item_id in ids):
        raise ValueError("example IDs must start with 'example:'")
    public_ids = {
        item["id"]
        for item in items
        if item["kind"] in {"function", "compiler_form", "surface", "construct"}
    }
    covered = {item_id for example in examples for item_id in example["covered_item_ids"]}
    unknown = sorted(covered - public_ids)
    missing = sorted(public_ids - covered)
    if unknown or missing:
        raise ValueError(f"example coverage mismatch; missing={missing}, unknown={unknown}")
    incomplete = [
        example["id"]
        for example in examples
        if not example["requires_backend"]
        and {check["expected_match"] for check in example["checks"]} != {True, False}
    ]
    if incomplete:
        raise ValueError(
            f"examples without a backend need matching and non-matching checks: {incomplete}"
        )


def generate_language_spec() -> dict[str, Any]:
    """Return the deterministic catalog generated from annotated implementation sources."""

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for owner in _source_owners():
        for target in _annotated_targets(owner):
            metadata = target.__quarry_language_metadata__
            if metadata.id in seen:
                raise ValueError(f"duplicate language item ID: {metadata.id}")
            seen.add(metadata.id)
            if isinstance(metadata, FunctionMetadata):
                record = _function_record(target, metadata)
            elif isinstance(metadata, CompilerFormMetadata):
                record = _compiler_form_record(target, metadata)
            elif isinstance(metadata, SurfaceMetadata):
                record = _surface_record(target, metadata)
            elif isinstance(metadata, ConstructMetadata):
                record = _construct_record(target, metadata)
            else:
                raise TypeError(f"unsupported metadata type {type(metadata).__name__}")
            items.append(record)

    constructs = [item for item in items if item["kind"] == "construct"]
    _validate_construct_coverage(constructs)
    items.extend(_token_records())
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("language item IDs must be unique")
    examples = _example_catalog()
    _validate_examples(items, examples)

    tokens = [item for item in items if item["kind"] == "token"]
    surfaces = [item for item in items if item["kind"] == "surface"]
    accessors = [
        item
        for item in items
        if item["kind"] == "function" and item["accessor_surface"] is not None
    ]

    reserved = (
        {item["spelling"] for item in tokens if item["reserved_for_variables"]}
        | {item["name"] for item in surfaces}
        | {item["name"] for item in accessors}
        | set(VARIABLE_RESERVED_KEYWORDS)
    )
    return {
        "format_version": FORMAT_VERSION,
        "language_version": __version__,
        "compiler_version": COMPILER_VERSION,
        "operator_precedence": [
            {
                "precedence": record.precedence,
                "family": record.family,
                "operators": list(record.operators),
                "associativity": record.associativity,
                "description": record.description,
            }
            for record in operator_precedence()
        ],
        "reserved_variable_names": sorted(reserved),
        "type_conversions": [
            {"from": source, "to": target, "description": description}
            for source, target, description in TYPE_CONVERSIONS
        ],
        "rejected_spellings": _rejected_spelling_catalog(items),
        "examples": examples,
        "items": sorted(items, key=lambda item: item["id"]),
    }


def _examples_by_item(spec: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in spec["examples"]:
        for item_id in example["covered_item_ids"]:
            result[item_id].append(example)
    return result


def _render_example_links(examples: list[dict[str, Any]]) -> str:
    links = []
    for example in examples:
        path = example["rule_path"]
        label = f"[{path}]({path})"
        if example["requires_backend"]:
            label += " (requires a configured backend)"
        links.append(label)
    return ", ".join(links)


def _markdown_table_cell(value: str) -> str:
    """Escape a value so Markdown does not treat its content as a new table column."""

    return value.replace("|", r"\|")


def render_reference(spec: dict[str, Any]) -> str:
    """Render the public Markdown reference from one generated catalog."""

    examples = _examples_by_item(spec)
    lines = [
        "# Quarry language reference",
        "",
        "This file is generated from the Python functions and grammar that Quarry runs.",
        "Do not edit it by hand. Run `python -m scripts.generate_language_spec` after changing",
        "a public function, surface, or grammar form.",
        "",
        "## Functions",
        "",
        "| Function | What it does |",
        "|---|---|",
    ]
    callables = [item for item in spec["items"] if item["kind"] in {"function", "compiler_form"}]
    for item in callables:
        signature = _markdown_table_cell(item["signature"])
        status = "Deprecated. " if item["deprecated"] else ""
        summary = _markdown_table_cell(status + item["summary"])
        lines.append(f"| `{signature}` | {summary} |")
    for item in callables:
        lines.extend(["", f"### `{item['name']}`", "", f"`{item['signature']}`", ""])
        if item["syntax"]:
            lines.extend([f"Rule syntax: `{item['syntax']}`", ""])
        lines.extend(item["description"].splitlines())
        restrictions: list[str] = []
        if item["deprecated"]:
            replacement = item["replacement"] or "No direct replacement"
            restrictions.append(f"Deprecated. Use `{replacement}` instead")
        if item["allowed_uses"]:
            restrictions.append("Allowed in: " + ", ".join(item["allowed_uses"]))
        if item["resource_group"]:
            restrictions.append(f"Requires: `{item['resource_group']}` backend support")
        for parameter in item["parameters"]:
            rules: list[str] = []
            if parameter["literal_only"]:
                if "text_reference" in parameter["accepted_types"]:
                    rules.append("must be literal text or a `text_ref(...)` call")
                else:
                    rules.append("must be written directly in the rule")
            if parameter["field_only"]:
                rules.append("must be a conversation field or surface")
            if parameter["allowed_values"]:
                rules.append("choices: " + ", ".join(parameter["allowed_values"]))
            if parameter["minimum"] is not None:
                rules.append(f"minimum: {parameter['minimum']}")
            if parameter["maximum"] is not None:
                rules.append(f"maximum: {parameter['maximum']}")
            if rules:
                restrictions.append(f"`{parameter['name']}`: " + "; ".join(rules))
        if restrictions:
            lines.extend(["", *[f"- {value}" for value in restrictions]])
        item_examples = examples.get(item["id"], [])
        if item_examples:
            lines.extend(["", "Examples: " + _render_example_links(item_examples)])

    lines.extend(["", "## Type conversions", "", "| From | To | Rule |", "|---|---|---|"])
    for record in spec["type_conversions"]:
        rule = _markdown_table_cell(record["description"])
        lines.append(f"| `{record['from']}` | `{record['to']}` | {rule} |")

    surfaces = [item for item in spec["items"] if item["kind"] == "surface"]
    lines.extend(["", "## Data surfaces", "", "Surfaces are bare names. Do not add `()`.", ""])
    lines.extend(
        [
            "| Surface | Type | Item fields | What it contains |",
            "|---|---|---|---|",
        ]
    )
    for item in surfaces:
        name = _markdown_table_cell(item["name"])
        public_type = _markdown_table_cell(item["public_type"])
        fields = ", ".join(f"`{field}`" for field in item["fields"]) or "—"
        summary = _markdown_table_cell(item["summary"])
        lines.append(f"| `{name}` | `{public_type}` | {fields} | {summary} |")

    lines.extend(
        [
            "",
            "### Surface fields",
            "",
            "Only the fields listed here are supported for rule authoring.",
            "",
            "| Surface | Field | Type | Meaning |",
            "|---|---|---|---|",
        ]
    )
    for item in surfaces:
        for field in item["field_details"]:
            name = _markdown_table_cell(item["name"])
            field_name = _markdown_table_cell(field["name"])
            public_type = _markdown_table_cell(field["type"])
            description = _markdown_table_cell(field["description"])
            lines.append(f"| `{name}` | `{field_name}` | `{public_type}` | {description} |")

    lines.extend(
        [
            "",
            "## Operator precedence",
            "",
            "Rows are ordered from lowest to highest precedence. Parentheses override this",
            "order. Add parentheses when mixing boolean, flow, comparison, or text operators",
            "so the intended grouping is explicit.",
            "",
            "| Precedence | Family | Operators | Associativity | Meaning |",
            "|---|---|---|---|---|",
        ]
    )
    for record in spec["operator_precedence"]:
        operators = ", ".join(f"`{item}`" for item in record["operators"])
        family = _markdown_table_cell(record["family"])
        description = _markdown_table_cell(record["description"])
        lines.append(
            f"| {record['precedence']} | {family} | {operators} | "
            f"{record['associativity']} | {description} |"
        )

    constructs = [item for item in spec["items"] if item["kind"] == "construct"]
    lines.extend(["", "## Language forms", ""])
    for item in constructs:
        lines.extend(
            [
                f"### {item['name']}",
                "",
                item["description"],
                "",
                f"Grammar: `{item['syntax']}`",
                "",
                "Examples: " + _render_example_links(examples.get(item["id"], [])),
                "",
            ]
        )

    tokens = [item for item in spec["items"] if item["kind"] == "token"]
    lines.extend(
        [
            "## Fixed words and symbols",
            "",
            "| Spelling | Category | Reserved as a variable name |",
            "|---|---|---|",
        ]
    )
    for item in tokens:
        reserved = "yes" if item["reserved_for_variables"] else "no"
        spelling = _markdown_table_cell(item["spelling"])
        category = _markdown_table_cell(item["category"])
        lines.append(f"| `{spelling}` | {category} | {reserved} |")

    lines.extend(
        [
            "",
            "## Rejected spellings",
            "",
            "These spellings are not alternate runtime functions. Quarry rejects them at the",
            "listed stage so the error is predictable.",
            "",
            "| Spelling | Rejected during | Use instead |",
            "|---|---|---|",
        ]
    )
    for record in spec["rejected_spellings"]:
        source_form = _markdown_table_cell(record["source_form"])
        rejected_by = _markdown_table_cell(record["rejected_by"])
        replacement = (
            f"`{_markdown_table_cell(record['replacement'])}`"
            if record["replacement"]
            else "No direct replacement"
        )
        lines.append(f"| `{source_form}` | {rejected_by} | {replacement} |")
    return "\n".join(lines).rstrip() + "\n"


def _write_examples(language_dir: Path) -> None:

    examples_dir = language_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    expected: set[Path] = set()
    for record in sorted(EXAMPLE_RECORDS, key=lambda item: item.id):
        slug = record.id.removeprefix("example:")
        rule_path = examples_dir / f"{slug}.rule"
        rule_path.write_text(record.rule_text.rstrip() + "\n", encoding="utf-8")
        expected.add(rule_path)
        for suffix, conversation in (
            ("matching", record.matching),
            ("non_matching", record.non_matching),
        ):
            if conversation is None:
                continue
            path = examples_dir / f"{slug}.{suffix}.json"
            path.write_text(
                json.dumps(conversation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            expected.add(path)
    for path in examples_dir.iterdir():
        if path.is_file() and path not in expected:
            path.unlink()


def write_language_outputs(repo_root: Path) -> dict[str, Any]:
    """Write runtime language artifacts below one repository root."""

    repo_root = repo_root.resolve()
    language_dir = repo_root / "quarry" / "language"
    spec = generate_language_spec()
    (language_dir / "spec.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (language_dir / "reference.md").write_text(render_reference(spec), encoding="utf-8")
    _write_examples(language_dir)
    return spec


def write_language_spec(path: Path) -> dict[str, Any]:
    """Compatibility entry point that writes the catalog to one explicit path."""

    spec = generate_language_spec()
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec


__all__ = [
    "FORMAT_VERSION",
    "generate_language_spec",
    "render_reference",
    "write_language_outputs",
    "write_language_spec",
]
