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

"""Declared inputs: data a conversation carries beside its messages.

The ``inputs:`` configuration section declares each section a scan may carry, such as
``memory.writes`` (a list of items) or ``sandbox.state`` (one object), with its content fields
(text) and its facts (typed fields). The ``adapters:`` section names the converters that turn
a native format carried under one key of the conversation into declared sections. The
declarations are data: a host passes them in its configuration, never as code.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, cast

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, create_model

from quarry.types.exceptions import ConfigurationError

FACT_TYPES = ("string", "integer", "number", "boolean", "list", "object")
SCALAR_TYPES = ("string", "integer", "number", "boolean")
ROLES = ("user", "system", "developer", "assistant")
# The key every list item carries, never declared.
ITEM_ID = "id"
_PART = re.compile(r"[a-z][a-z0-9_]*\Z")


@dataclass(frozen=True)
class FactType:
    """One declared fact: its type, and, as the type allows, its allowed values, list item
    type or object fields."""

    type: str
    values: tuple[Any, ...] | None = None
    optional: bool = False
    items: str | None = None
    fields: Mapping[str, FactType] | None = None


@dataclass(frozen=True)
class ContentField:
    """One content field: a string the rules read as text."""

    name: str
    optional: bool = False


@dataclass(frozen=True)
class InputDeclaration:
    """One declared section: ``namespace.name``, a list of items or (``single``) one object."""

    section: str
    single: bool = False
    role: str = "user"
    content: tuple[ContentField, ...] = ()
    facts: Mapping[str, FactType] = field(default_factory=dict)

    @property
    def namespace(self) -> str:
        return self.section.split(".", 1)[0]

    @property
    def name(self) -> str:
        return self.section.split(".", 1)[1]

    @property
    def content_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.content)

    def describe(self) -> dict[str, Any]:
        """The declaration as plain JSON values, in a canonical form: what the plugin lists and
        what a compiled policy records."""

        def fact(value: FactType) -> dict[str, Any]:
            described: dict[str, Any] = {"type": value.type}
            if value.values is not None:
                described["values"] = list(value.values)
            if value.items is not None:
                described["items"] = value.items
            if value.fields is not None:
                described["fields"] = {name: fact(item) for name, item in value.fields.items()}
            if value.optional:
                described["optional"] = True
            return described

        described: dict[str, Any] = {"section": self.section, "single": self.single}
        if self.content:
            described["role"] = self.role
            described["content"] = [
                {"name": item.name, "optional": item.optional} for item in self.content
            ]
        described["facts"] = {name: fact(value) for name, value in self.facts.items()}
        return described


@dataclass(frozen=True)
class AdapterConfig:
    """One ``adapters:`` entry: the conversation key it reads and the adapter that converts it."""

    key: str
    adapter: str


class InputAdapter(ABC):
    """Converts a native format into declared sections, and decides nothing else.

    ``sections`` are the sections it can fill: ``messages``, ``tools``, sections the
    configuration declares, or sections it declares itself in ``declarations`` (raw
    declarations, as an ``inputs:`` entry holds them). ``version`` changes whenever its output
    for the same native input does, so a compiled policy that reads its sections records it.
    ``convert`` receives only the sections a selected rule reads, as ``wanted``, and returns
    exactly those.
    """

    name: ClassVar[str]
    sections: ClassVar[tuple[str, ...]]
    version: ClassVar[str]
    declarations: ClassVar[Mapping[str, Mapping[str, Any]]] = MappingProxyType({})

    @abstractmethod
    def convert(self, native: Any, wanted: frozenset[str]) -> dict[str, Any]:
        """The ``wanted`` sections converted from ``native``, by name, and no others.
        ``messages`` and ``tools``, when wanted, are always returned; an input section the
        native value cannot fill is left out, and the scan then does not supply it."""


def _mapping(value: object, owner: str, allowed: set[str]) -> dict:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{owner} must be a mapping")
    unknown = sorted(str(key) for key in set(value) - allowed)
    if unknown:
        raise ConfigurationError(f"{owner} has unknown keys: {', '.join(unknown)}")
    return dict(value)


def _name(value: object, owner: str) -> str:
    if not isinstance(value, str) or not _PART.fullmatch(value):
        raise ConfigurationError(
            f"{owner} name {value!r} must be lowercase letters, digits and '_', "
            "starting with a letter"
        )
    return value


def _optional(fields: Mapping, owner: str) -> bool:
    value = fields.get("optional", False)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{owner} optional must be true or false")
    return value


def _parse_fact(value: object, owner: str) -> FactType:
    fields = _mapping(value, owner, {"type", "values", "optional", "items", "fields"})
    kind = fields.get("type")
    if kind not in FACT_TYPES:
        raise ConfigurationError(f"{owner} type must be one of {', '.join(FACT_TYPES)}")
    optional = _optional(fields, owner)
    values = fields.get("values")
    if values is not None:
        if kind not in SCALAR_TYPES:
            raise ConfigurationError(f"{owner} values apply only to a scalar type")
        if not isinstance(values, list) or not values:
            raise ConfigurationError(f"{owner} values must be a non-empty list")
        for item in values:
            if not _matches_scalar(kind, item):
                raise ConfigurationError(f"{owner} value {item!r} is not a {kind}")
            if isinstance(item, float) and not math.isfinite(item):
                # A scanned number is finite, and NaN never equals itself.
                raise ConfigurationError(f"{owner} value {item!r} is not a finite number")
        if len(set(map(repr, values))) != len(values):
            raise ConfigurationError(f"{owner} values repeat")
        values = tuple(values)
    items = fields.get("items")
    if (kind == "list") != (items is not None):
        raise ConfigurationError(f"{owner} 'items' names a list's item type, and only a list's")
    if items is not None and items not in SCALAR_TYPES:
        raise ConfigurationError(f"{owner} items must be one of {', '.join(SCALAR_TYPES)}")
    nested = fields.get("fields")
    if (kind == "object") != (nested is not None):
        raise ConfigurationError(f"{owner} 'fields' declares an object's fields, and only its")
    parsed_fields = None
    if nested is not None:
        if not isinstance(nested, Mapping) or not nested:
            raise ConfigurationError(f"{owner} fields must be a non-empty mapping")
        parsed_fields = MappingProxyType(
            {
                _name(name, f"{owner} field"): _parse_fact(item, f"{owner}.{name}")
                for name, item in nested.items()
            }
        )
    return FactType(kind, values, optional, items, parsed_fields)


def _matches_scalar(kind: str, value: object) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, int | float) and not isinstance(value, bool)


def parse_declaration(section: object, value: object) -> InputDeclaration:
    """One ``inputs:`` entry, checked against the declaration grammar."""

    if not isinstance(section, str) or section.count(".") != 1:
        raise ConfigurationError(
            f"Input {section!r} must be named '<namespace>.<name>', such as 'memory.writes'"
        )
    owner = f"Input '{section}'"
    namespace, name = section.split(".")
    _name(namespace, f"{owner} namespace")
    _name(name, owner)
    fields = _mapping(value, owner, {"single", "role", "content", "facts"})
    single = fields.get("single", False)
    if not isinstance(single, bool):
        raise ConfigurationError(f"{owner} single must be true or false")
    content: list[ContentField] = []
    raw_content = fields.get("content", [])
    if not isinstance(raw_content, list):
        raise ConfigurationError(f"{owner} content must be a list of field names")
    for entry in raw_content:
        if isinstance(entry, str):
            content.append(ContentField(_name(entry, f"{owner} content field")))
        else:
            entry_fields = _mapping(entry, f"{owner} content field", {"name", "optional"})
            content.append(
                ContentField(
                    _name(entry_fields.get("name"), f"{owner} content field"),
                    _optional(entry_fields, f"{owner} content field"),
                )
            )
    if "role" in fields and not content:
        raise ConfigurationError(f"{owner} role applies only to a section with content")
    role = fields.get("role", "user")
    if role not in ROLES:
        raise ConfigurationError(f"{owner} role must be one of {', '.join(ROLES)}")
    raw_facts = fields.get("facts", {})
    if not isinstance(raw_facts, Mapping):
        raise ConfigurationError(f"{owner} facts must be a mapping of name to type")
    facts = {
        _name(fact_name, f"{owner} fact"): _parse_fact(fact_value, f"{owner} fact '{fact_name}'")
        for fact_name, fact_value in raw_facts.items()
    }
    names = [item.name for item in content] + list(facts)
    if not single:
        names.append(ITEM_ID)
    repeated = sorted({item for item in names if names.count(item) > 1})
    if repeated:
        raise ConfigurationError(
            f"{owner} names {', '.join(repeated)} more than once (an item's 'id' is reserved)"
        )
    if not names or names == [ITEM_ID]:
        raise ConfigurationError(f"{owner} declares no content and no facts")
    return InputDeclaration(section, single, role, tuple(content), MappingProxyType(facts))


def parse_inputs(data: object) -> dict[str, InputDeclaration]:
    """The ``inputs:`` section."""

    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigurationError("'inputs' must be a mapping of section name to declaration")
    return {section: parse_declaration(section, value) for section, value in data.items()}


def parse_adapters(data: object) -> dict[str, AdapterConfig]:
    """The ``adapters:`` section: each conversation key it names maps to ``{adapter}``."""

    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigurationError("'adapters' must be a mapping of conversation key to adapter")
    adapters: dict[str, AdapterConfig] = {}
    for key, value in data.items():
        owner = f"Adapter key '{key}'"
        _name(key, owner)
        fields = _mapping(value, owner, {"adapter"})
        adapter = fields.get("adapter")
        if not isinstance(adapter, str) or not adapter:
            raise ConfigurationError(f"{owner} needs 'adapter', the name of an input adapter")
        adapters[key] = AdapterConfig(key, adapter)
    return adapters


_SCALAR_ANNOTATIONS: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float | int,
    "boolean": bool,
}


def _one_of(values: tuple[Any, ...]):
    """A check, after the scalar type's own strict check, that a value is one of ``values``:
    the type check keeps ``true`` and ``1.0`` from passing as ``1``."""

    def check(value: Any) -> Any:
        if not any(type(value) is type(item) and value == item for item in values):
            raise ValueError(f"must be one of {list(values)}")
        return value

    return AfterValidator(check)


def _annotation(value: FactType, label: str) -> Any:
    if value.type in _SCALAR_ANNOTATIONS:
        annotation: Any = _SCALAR_ANNOTATIONS[value.type]
        if value.values is not None:
            annotation = Annotated[annotation, _one_of(value.values)]
    elif value.type == "list":
        assert value.items is not None
        annotation = cast(Any, list)[_SCALAR_ANNOTATIONS[value.items]]
    else:
        assert value.fields is not None
        annotation = _model(label, value.fields, ())
    return annotation


# Validation is by the declared names (aliases); each field's name inside the schema is its
# position, so a declared name can never collide with an attribute of the schema itself.
_STRICT = ConfigDict(extra="forbid", validate_default=True, strict=True, validate_by_name=False)


def _field(annotation: Any, optional: bool, alias: str) -> tuple[Any, Any]:
    if optional:
        return (annotation | None, Field(default=None, alias=alias))
    return (annotation, Field(alias=alias))


def _model(
    label: str, facts: Mapping[str, FactType], content: tuple[ContentField, ...]
) -> type[BaseModel]:
    definitions: dict[str, Any] = {}
    for item in content:
        definitions[f"f{len(definitions)}"] = _field(str, item.optional, item.name)
    for name, value in facts.items():
        annotation = _annotation(value, f"{label}.{name}")
        definitions[f"f{len(definitions)}"] = _field(annotation, value.optional, name)
    model_name = "".join(part.capitalize() for part in re.split(r"[._]", label) if part)
    return create_model(model_name or "Input", __config__=_STRICT, **definitions)


def declaration_model(declaration: InputDeclaration) -> type[BaseModel]:
    """The strict schema one item (or the single object) of ``declaration`` validates against,
    built from the declaration, without its reserved ``id``."""

    return _model(declaration.section, declaration.facts, declaration.content)


# The entry-point group an installed package registers an input adapter under, by name.
INPUT_ADAPTER_GROUP = "quarry.input_adapters"


def adapter_fingerprints(
    adapters: Mapping[str, InputAdapter], read: frozenset[str]
) -> dict[str, str]:
    """The identity of each adapter, by conversation key, that fills a surface of ``read``: a
    section it fills, or the chat or the tools when it fills those. A compiled policy records
    them, and loads only where the same adapters fill what it reads."""

    from quarry.language.runtime import surface_names
    from quarry.policy import _canonical_sha256

    reads_chat = bool(read & (surface_names() - {"tools"}))
    return {
        key: _canonical_sha256({"adapter": adapter.name, "version": adapter.version})
        for key, adapter in sorted(adapters.items())
        if (set(adapter.sections) & read)
        or ("messages" in adapter.sections and reads_chat)
        or ("tools" in adapter.sections and "tools" in read)
    }


def load_input_adapters(
    configured: Mapping[str, AdapterConfig],
    declared: Mapping[str, InputDeclaration],
    host: Mapping[str, InputAdapter] | None = None,
) -> tuple[dict[str, InputAdapter], dict[str, InputDeclaration]]:
    """Load host or entry-point adapters and combine their section declarations.

    Reject duplicate sections or adapters that fill an undeclared section.
    """

    from importlib.metadata import entry_points

    host = dict(host or {})
    installed = {point.name: point for point in entry_points(group=INPUT_ADAPTER_GROUP)}
    loaded: dict[str, InputAdapter] = {}
    owners: dict[str, str] = {}
    declarations = dict(declared)
    for key, entry in configured.items():
        if key in {"messages", "tools", "timeline", "functions"}:
            raise ConfigurationError(f"Adapter key '{key}' is a key the conversation itself uses")
        name = entry.adapter
        if name in host:
            adapter = host[name]
        elif name in installed:
            target = installed[name].load()
            adapter = target() if isinstance(target, type) else target
        else:
            available = ", ".join(sorted({*host, *installed})) or "none"
            raise ConfigurationError(
                f"Adapter key '{key}' names input adapter '{name}', which is not available; "
                f"available: {available}"
            )
        if not isinstance(adapter, InputAdapter):
            raise ConfigurationError(f"Input adapter '{name}' must be an InputAdapter")
        for section, raw in adapter.declarations.items():
            if section in declarations:
                raise ConfigurationError(
                    f"Input adapter '{name}' declares '{section}', which is declared already"
                )
            declarations[section] = parse_declaration(section, raw)
        for section in adapter.sections:
            if section not in {"messages", "tools"} and section not in declarations:
                raise ConfigurationError(
                    f"Input adapter '{name}' fills '{section}', which no input declares"
                )
            if section in owners:
                raise ConfigurationError(
                    f"Adapter keys '{owners[section]}' and '{key}' both fill '{section}'"
                )
            owners[section] = key
        loaded[key] = adapter
    # A key a namespace holds would be read as that namespace's sections, never converted.
    namespaces = {declaration.namespace for declaration in declarations.values()}
    for key in loaded:
        if key in namespaces:
            raise ConfigurationError(f"Adapter key '{key}' is the name of an input namespace")
    return loaded, declarations
