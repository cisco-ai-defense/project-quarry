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

"""The declared inputs of one scanner: the sections a conversation carries beside its
messages, resolved in rules by name and validated when a selected rule reads them.

The host declares each section in its configuration (``inputs:``, see ``quarry.inputs``); rule
text never loads Python modules.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from types import MappingProxyType, UnionType
from typing import Annotated, Any, Union, cast, get_args, get_origin

import orjson
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from quarry.inputs import ITEM_ID, InputDeclaration, declaration_model
from quarry.safety.limits import SafetyLimits, json_size
from quarry.types.dsl import (
    AllComprehension,
    AnyComprehension,
    Assignment,
    BinaryOp,
    ConditionNode,
    ContainsAnyOp,
    FieldAccess,
    FilterComprehension,
    FlowOp,
    FunctionCall,
    Identifier,
    Literal,
    MapComprehension,
    PatternMatch,
    condition_child_fields,
    walk_condition_nodes,
)
from quarry.types.exceptions import ConfigurationError, ConversationValidationError
from quarry.types.messages import ToolCall

# Names the conversation itself uses, which no input namespace may take.
_RESERVED = frozenset({"conversation", "true", "false", "null", "params", "timeline", "functions"})


@dataclass(frozen=True)
class _ScopedName:
    """A lexical name binding and the scope in which its value was defined."""

    kind: str  # "element" for a comprehension, "alias" for an `as` binding
    expression: ConditionNode
    scope: Mapping[str, _ScopedName]


def _conjunction_scope(
    node: ConditionNode, scope: Mapping[str, _ScopedName]
) -> Mapping[str, _ScopedName]:
    """Carry aliases from the left conjunction into its following clause."""

    pending = [node]
    while pending:
        current = pending.pop()
        if (
            isinstance(current, BinaryOp)
            and not isinstance(current, FlowOp)
            and current.operator == "and"
        ):
            pending.extend((current.right, current.left))
        elif isinstance(current, Assignment):
            scope = {
                **scope,
                current.variable: _ScopedName("alias", current.expression, scope),
            }
    return scope


def _scoped_children(
    node: ConditionNode, scope: Mapping[str, _ScopedName]
) -> list[tuple[ConditionNode, Mapping[str, _ScopedName]]]:
    """Use the canonical AST child list, adding bindings only in their lexical body."""

    children: list[tuple[ConditionNode, Mapping[str, _ScopedName]]] = []
    comprehensions = (AnyComprehension, AllComprehension, FilterComprehension, MapComprehension)
    for field_name in condition_child_fields(node):
        child_scope = scope
        if isinstance(node, comprehensions) and field_name != "collection":
            child_scope = {
                **scope,
                node.variable: _ScopedName("element", node.collection, scope),
            }
        elif (
            isinstance(node, BinaryOp)
            and not isinstance(node, FlowOp)
            and node.operator == "and"
            and field_name == "right"
        ):
            child_scope = _conjunction_scope(node.left, scope)
        value = getattr(node, field_name)
        if isinstance(value, ConditionNode):
            children.append((value, child_scope))
        elif isinstance(value, list | tuple):
            children.extend(
                (child, child_scope) for child in value if isinstance(child, ConditionNode)
            )
    return children


def _scoped_visits(root: ConditionNode):
    """Walk authored scopes without treating unrelated `as` or loop names as globals."""

    stack: list[tuple[ConditionNode, Mapping[str, _ScopedName], tuple[ConditionNode, ...]]] = [
        (root, {}, ())
    ]
    while stack:
        node, scope, ancestors = stack.pop()
        yield node, scope, ancestors
        stack.extend(
            (child, child_scope, (*ancestors, node))
            for child, child_scope in reversed(_scoped_children(node, scope))
        )


@dataclass(frozen=True)
class _Namespace:
    """The sections one input namespace declares: its lists of items, and its single
    objects, each with the strict schema an item validates against."""

    events: Mapping[str, type[BaseModel]]
    singles: Mapping[str, type[BaseModel]]


class InputObject:
    """Immutable, allowlisted DSL view of a validated object."""

    __slots__ = ("_fields",)

    def __init__(self, fields: Mapping[str, Any]):
        object.__setattr__(
            self,
            "_fields",
            MappingProxyType({key: _freeze(value) for key, value in fields.items()}),
        )

    @property
    def _allowed_fields(self) -> frozenset[str]:
        return frozenset(self._fields)

    def __getattr__(self, name: str) -> Any:
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Extension objects are immutable")


@dataclass(frozen=True)
class MissingInputState:
    """Marks a single section the conversation did not carry, which must not silently behave
    like a complete no-match."""

    surface: str


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


class InputRegistry:
    """The declared inputs of one scanner generation, validated and frozen."""

    def __init__(self, declarations: Mapping[str, InputDeclaration] | None = None):
        from quarry.language.runtime import reserved_variable_names, surface_names

        reserved = _RESERVED | surface_names() | reserved_variable_names()
        grouped: dict[str, tuple[dict[str, type[BaseModel]], dict[str, type[BaseModel]]]] = {}
        self.declarations: Mapping[str, InputDeclaration] = MappingProxyType(
            dict(declarations or {})
        )
        for declaration in self.declarations.values():
            if declaration.namespace in reserved:
                raise ConfigurationError(
                    f"Input namespace {declaration.namespace!r} is a name the conversation "
                    "or the language uses"
                )
            events, singles = grouped.setdefault(declaration.namespace, ({}, {}))
            (singles if declaration.single else events)[declaration.name] = declaration_model(
                declaration
            )
        self.definitions: Mapping[str, _Namespace] = MappingProxyType(
            {
                namespace: _Namespace(MappingProxyType(events), MappingProxyType(singles))
                for namespace, (events, singles) in grouped.items()
            }
        )

    @staticmethod
    def surface_name(namespace: str, name: str) -> str:
        """Qualified name of a registered event or state surface."""
        return f"{namespace}.{name}"

    @cached_property
    def event_surfaces(self) -> frozenset[str]:
        return frozenset(
            self.surface_name(namespace, name)
            for namespace, spec in self.definitions.items()
            for name in spec.events
        )

    @cached_property
    def state_surfaces(self) -> frozenset[str]:
        return frozenset(
            self.surface_name(namespace, name)
            for namespace, spec in self.definitions.items()
            for name in spec.singles
        )

    @cached_property
    def text_fields(self) -> Mapping[str, tuple[str, ...]]:
        """Each section's text fields in one order: content fields, then string facts, each as
        declared (``INPUT_TEXTS``)."""

        return MappingProxyType(
            {
                surface: (
                    *declaration.content_names,
                    *(name for name, fact in declaration.facts.items() if fact.type == "string"),
                )
                for surface, declaration in self.declarations.items()
            }
        )

    def resolve(self, namespace: str, name: str) -> str | None:
        spec = self.definitions.get(namespace)
        if spec is None or (name not in spec.events and name not in spec.singles):
            return None
        return self.surface_name(namespace, name)

    def configured(self) -> str:
        """The configured sections, for a load error that names a misspelt one."""
        return ", ".join(sorted(self.declarations)) or "none"

    def rewrite_rule(self, rule: Any) -> None:
        """Resolve dotted extension names to scanner-local compiler surfaces.

        This is an AST transformation, not a text replacement: unrelated fields and
        quoted strings cannot accidentally acquire extension meaning.
        """

        # The names the rule calls as imports, such as ``pi(w)``, whose argument may be an
        # input item or a single section.
        called = {group.name for group in getattr(rule, "import_groups", ())}

        def rewrite(root: ConditionNode) -> ConditionNode:
            # The parser can produce expressions close to its depth budget. Rewrite
            # bottom-up without using Python's much smaller call-stack budget.
            rewritten: dict[int, ConditionNode] = {}
            stack: list[tuple[ConditionNode, Mapping[str, _ScopedName], bool]] = [(root, {}, False)]
            while stack:
                node, scope, ready = stack.pop()
                if not ready:
                    stack.append((node, scope, True))
                    stack.extend(
                        (child, child_scope, False)
                        for child, child_scope in _scoped_children(node, scope)
                    )
                    continue

                if (
                    isinstance(node, FieldAccess)
                    and isinstance(node.base, Identifier)
                    and node.base.name in self.definitions
                    and node.base.name not in scope
                    and node.fields
                ):
                    namespace, name = node.base.name, node.fields[0]
                    surface = self.resolve(namespace, name)
                    if surface is None:
                        raise ConfigurationError(
                            f"Unknown input {namespace}.{name}; configured inputs: "
                            f"{self.configured()}"
                        )
                    if name not in self.definitions[namespace].singles:
                        if len(node.fields) != 1:
                            raise ConfigurationError(
                                f"Use {namespace}.{name} as a collection, then read its bound item"
                            )
                        rewritten[id(node)] = Identifier(surface)
                        continue
                    rewritten[id(node)] = (
                        FieldAccess(Identifier(surface), node.fields[1:])
                        if len(node.fields) > 1
                        else Identifier(surface)
                    )
                    continue

                if (
                    isinstance(node, FunctionCall)
                    and getattr(node, "receiver", None) is None
                    and node.name in called
                    and node.name not in scope
                    and len(node.args) == 1
                ):
                    argument = rewritten[id(node.args[0])]
                    projected = self._projection(argument, scope)
                    if projected is not None:
                        rewritten[id(node)] = replace(node, args=[projected])
                        continue
                    indexed = self._indexed_call(
                        node, self._aliased_index(argument, scope) or argument
                    )
                    if indexed is not None:
                        rewritten[id(node)] = indexed
                        continue

                updates: dict[str, Any] = {}
                for field_name in condition_child_fields(node):
                    value = getattr(node, field_name)
                    if isinstance(value, ConditionNode):
                        child = rewritten[id(value)]
                        if child is not value:
                            updates[field_name] = child
                    elif isinstance(value, list | tuple):
                        children = type(value)(
                            rewritten[id(child)] if isinstance(child, ConditionNode) else child
                            for child in value
                        )
                        if any(a is not b for a, b in zip(children, value, strict=True)):
                            updates[field_name] = children
                # Parsed nodes are dataclasses, although the base marker does not
                # carry that type information for the static checker.
                rewritten[id(node)] = (
                    cast(ConditionNode, replace(cast(Any, node), **updates)) if updates else node
                )
            return rewritten[id(root)]

        for name in ("condition", "collect", "conversation"):
            node = getattr(rule, name, None)
            if node is not None:
                self._reject_namespace_names(node)
        for name in ("condition", "collect"):
            node = getattr(rule, name)
            if node is not None:
                rewritten = rewrite(node)
                self._validate_rule_fields(rewritten)
                self._diagnose_unregistered_surfaces(rewritten)
                setattr(rule, name, rewritten)

    def _reject_namespace_names(self, root: ConditionNode) -> None:
        """A comprehension item or an ``as`` name may not take an input namespace's name: a
        field of it, such as ``memory.response``, would read as the input section."""

        binders = (
            AnyComprehension,
            AllComprehension,
            FilterComprehension,
            MapComprehension,
            Assignment,
        )
        for node in walk_condition_nodes(root):
            if isinstance(node, binders) and node.variable in self.definitions:
                raise ConfigurationError(
                    f"`{node.variable}` names an input namespace; give the item or the "
                    "value another name"
                )

    def _projection(
        self, argument: ConditionNode, scope: Mapping[str, _ScopedName]
    ) -> ConditionNode | None:
        """``argument`` as the messages it projects to when it is a whole input item (the item
        of a comprehension over a list section) or a single section: one message of the
        section's ``role`` whose text parts are its content fields, in declared order. None
        for any other argument."""

        if not isinstance(argument, Identifier):
            return None
        # A name `as` gave the item or the section is that item or section.
        while (
            (named := scope.get(argument.name)) is not None
            and named.kind == "alias"
            and isinstance(named.expression, Identifier | FieldAccess)
        ):
            expression = named.expression
            if isinstance(expression, FieldAccess):
                single = self._single_section(expression)
                return None if single is None else self._projected(single, single)
            argument, scope = expression, named.scope
        surface: str | None = None
        if argument.name in self.state_surfaces and argument.name not in scope:
            surface = argument.name
        elif (bound := scope.get(argument.name)) is not None and bound.kind == "element":
            # A slice of a section ranges over that section's items.
            surface = self._list_section(bound.expression, bound.scope, slices=True)
        if surface is None:
            return None
        return self._projected(surface, argument.name)

    def _single_section(self, node: FieldAccess) -> str | None:
        """The single section ``node``, ``namespace.name`` as written, names, or None."""

        if not (isinstance(node.base, Identifier) and len(node.fields) == 1):
            return None
        surface = self.resolve(node.base.name, node.fields[0])
        return surface if surface in self.state_surfaces else None

    def _projected(self, surface: str, name: str) -> ConditionNode:
        """The item ``name`` of ``surface``, or the single section ``surface`` itself, as one
        message of the section's role whose text parts are its content fields."""

        declaration = self.declarations[surface]
        if not declaration.content_names:
            raise ConfigurationError(
                f"{surface} has no content fields, so an item of it is no messages"
            )
        parts: list[ConditionNode] = [
            FieldAccess(Identifier(name), [field]) for field in declaration.content_names
        ]
        return FunctionCall("_input_message", [Literal(surface), Literal(declaration.role), *parts])

    def _aliased_index(
        self, argument: ConditionNode, scope: Mapping[str, _ScopedName]
    ) -> ConditionNode | None:
        """The indexed item of a list section a name ``as`` gave, such as ``note`` after
        ``(memory.notes[0] as note)``, in the form a direct call names it; None otherwise."""

        while (
            isinstance(argument, Identifier)
            and (named := scope.get(argument.name)) is not None
            and named.kind == "alias"
        ):
            argument, scope = named.expression, named.scope
        if not (
            isinstance(argument, FunctionCall)
            and argument.name == "_index"
            and len(argument.args) == 2
        ):
            return None
        # An index counts from the start of the section, so not of a slice of it.
        surface = self._list_section(argument.args[0], scope, slices=False)
        if surface is None:
            return None
        return replace(argument, args=[Identifier(surface), argument.args[1]])

    def _list_section(
        self, collection: ConditionNode, scope: Mapping[str, _ScopedName], *, slices: bool
    ) -> str | None:
        """The list section ``collection`` names: ``namespace.name`` as written or as the
        rewrite names it, a name ``as`` gave one, or, with ``slices``, a slice of one. None for
        any other collection."""

        while True:
            if slices and isinstance(collection, FunctionCall) and collection.name == "_slice":
                collection = collection.args[0]
            elif (
                isinstance(collection, Identifier)
                and (alias := scope.get(collection.name)) is not None
                and alias.kind == "alias"
            ):
                collection, scope = alias.expression, alias.scope
            else:
                break
        if (
            isinstance(collection, FieldAccess)
            and isinstance(collection.base, Identifier)
            and collection.base.name in self.definitions
            and collection.base.name not in scope
            and len(collection.fields) == 1
        ):
            surface = self.resolve(collection.base.name, collection.fields[0])
            return surface if surface in self.event_surfaces else None
        if isinstance(collection, Identifier) and collection.name in self.event_surfaces:
            return collection.name
        return None

    def _indexed_call(self, call: FunctionCall, argument: ConditionNode) -> ConditionNode | None:
        """``call`` of one item of a list section by index, such as ``pi(memory.notes[-1])``, as
        the call of that item over the one-item slice that holds it, or None for any other
        argument. A section without that item has no messages, so the call does not hold."""

        if not (
            isinstance(argument, FunctionCall)
            and argument.name == "_index"
            and len(argument.args) == 2
            and isinstance(argument.args[0], Identifier)
            and argument.args[0].name in self.event_surfaces
            and isinstance(argument.args[1], Literal)
            and type(argument.args[1].value) is int
        ):
            return None
        surface, index = argument.args[0].name, cast(int, argument.args[1].value)
        variable = f"__item_of_{call.name}"
        one = FunctionCall(
            "_slice",
            [Identifier(surface), Literal(index), Literal(None if index == -1 else index + 1)],
        )
        return AnyComprehension(
            variable, one, replace(call, args=[self._projected(surface, variable)])
        )

    def _diagnose_unregistered_surfaces(self, root: ConditionNode) -> None:
        """Give a useful load error for a dotted collection with no schema.

        A bound ``x.field`` is an ordinary object read. An unbound dotted
        collection used by a quantifier or collection function is an attempted
        extension surface; silently treating it as an unknown value hides a
        missing host registration.
        """

        from quarry.language.runtime import surface_names

        builtins = surface_names() | _RESERVED
        for node, scope, ancestors in _scoped_visits(root):
            if (
                not isinstance(node, FieldAccess)
                or not isinstance(node.base, Identifier)
                or not node.fields
                or node.base.name in scope
                or node.base.name in self.definitions
                or node.base.name in self.event_surfaces
                or node.base.name in self.state_surfaces
                or node.base.name in builtins
            ):
                continue
            parent = ancestors[-1] if ancestors else None
            collection_position = (
                isinstance(
                    parent,
                    (AnyComprehension, AllComprehension, FilterComprehension, MapComprehension),
                )
                and parent.collection is node
            ) or (
                isinstance(parent, FunctionCall)
                and parent.name in {"count", "_index", "_slice"}
                and bool(parent.args)
                and parent.args[0] is node
            )
            if not collection_position:
                continue
            namespace = node.base.name
            raise ConfigurationError(
                f"Unknown input {namespace}.{node.fields[0]}; configured inputs: "
                f"{self.configured()}"
            )

    @staticmethod
    def _check_schema_path(
        schema: type[BaseModel], fields: Sequence[str | int], label: str
    ) -> tuple[Any, ...]:
        def variants(annotation: Any) -> tuple[Any, ...]:
            origin = get_origin(annotation)
            if origin in (Union, UnionType):
                return tuple(
                    option
                    for argument in get_args(annotation)
                    if argument is not type(None)
                    for option in variants(argument)
                )
            if origin is Annotated:
                return variants(get_args(annotation)[0])
            return (annotation,)

        # A declaration builds objects, lists of one scalar type, and scalars, some of them
        # optional (a union with None) or checked against allowed values (Annotated); each
        # field is read by its declared name, the schema field's alias.
        current: tuple[Any, ...] = (schema,)
        for field_name in fields:
            next_types: list[Any] = []
            for annotation in current:
                for option in variants(annotation):
                    if isinstance(option, type) and issubclass(option, BaseModel):
                        by_name = {
                            field.alias or name: field
                            for name, field in option.model_fields.items()
                        }
                        field = by_name.get(field_name) if isinstance(field_name, str) else None
                        if field is None:
                            raise ConfigurationError(
                                f"Unknown extension field {label}.{field_name}"
                            )
                        next_types.append(field.annotation)
                    elif get_origin(option) is list:
                        if not isinstance(field_name, int):
                            raise ConfigurationError(
                                f"Cannot read {field_name!r} from list {label}"
                            )
                        next_types.append(get_args(option)[0])
                    else:
                        raise ConfigurationError(f"Cannot read {field_name!r} from scalar {label}")
            current = tuple(next_types)
            label += f"[{field_name}]" if isinstance(field_name, int) else f".{field_name}"
        return current

    def _validate_rule_fields(self, root: ConditionNode) -> None:
        event_types = {
            self.surface_name(namespace, name): schema
            for namespace, spec in self.definitions.items()
            for name, schema in spec.events.items()
        }
        state_types = {
            self.surface_name(namespace, name): schema
            for namespace, spec in self.definitions.items()
            for name, schema in spec.singles.items()
        }

        def event_schema(
            collection: ConditionNode, scope: Mapping[str, _ScopedName]
        ) -> type[BaseModel] | None:
            seen: set[int] = set()
            while True:
                if isinstance(collection, Identifier):
                    binding = scope.get(collection.name)
                    if binding is not None:
                        if binding.kind != "alias" or id(binding) in seen:
                            return None
                        seen.add(id(binding))
                        collection, scope = binding.expression, binding.scope
                        continue
                    if collection.name in event_types:
                        return event_types[collection.name]
                if isinstance(collection, FunctionCall) and collection.name == "_slice":
                    collection = collection.args[0]
                    continue
                return None

        def access_root_and_path(node: FieldAccess) -> tuple[ConditionNode | None, list[str | int]]:
            path: list[str | int] = list(node.fields)
            base = node.base
            while True:
                if isinstance(base, FieldAccess):
                    path = [*base.fields, *path]
                    base = base.base
                elif isinstance(base, FunctionCall) and base.name == "_index":
                    index = base.args[1]
                    literal_index = (
                        int(index.value)
                        if isinstance(index, Literal)
                        and isinstance(index.value, (int, str))
                        and str(index.value).lstrip("-").isdigit()
                        else 0
                    )
                    path = [literal_index, *path]
                    base = base.args[0]
                elif isinstance(base, FunctionCall) and base.name == "_slice":
                    base = base.args[0]
                else:
                    return base, path

        def expand_alias(
            base: ConditionNode | None,
            path: list[str | int],
            scope: Mapping[str, _ScopedName],
        ) -> tuple[ConditionNode | None, list[str | int], Mapping[str, _ScopedName]]:
            seen: set[int] = set()
            while (
                isinstance(base, Identifier)
                and (binding := scope.get(base.name)) is not None
                and binding.kind == "alias"
                and id(binding) not in seen
            ):
                seen.add(id(binding))
                expression = binding.expression
                scope = binding.scope
                if isinstance(expression, Identifier):
                    base = expression
                    continue
                if isinstance(expression, FieldAccess):
                    base, inner_path = access_root_and_path(expression)
                elif isinstance(expression, FunctionCall) and expression.name in {
                    "_index",
                    "_slice",
                }:
                    base, inner_path = access_root_and_path(FieldAccess(expression, []))
                else:
                    break
                path = [*inner_path, *path]
            return base, path, scope

        def extension_types(
            expression: ConditionNode, scope: Mapping[str, _ScopedName]
        ) -> tuple[Any, ...] | None:
            if isinstance(expression, FieldAccess):
                base, path = access_root_and_path(expression)
            elif isinstance(expression, FunctionCall) and expression.name in {
                "_index",
                "_slice",
            }:
                base, path = access_root_and_path(FieldAccess(expression, []))
            elif isinstance(expression, Identifier):
                base, path = expression, []
            else:
                return None
            base, path, base_scope = expand_alias(base, path, scope)
            if not isinstance(base, Identifier):
                return None
            name = base.name
            if name in state_types and name not in base_scope:
                return self._check_schema_path(state_types[name], path, name)
            if path and isinstance(path[0], int) and (schema := event_schema(base, base_scope)):
                return self._check_event_fields(schema, path[1:], name)
            binding = base_scope.get(name)
            if (
                binding is not None
                and binding.kind == "element"
                and (schema := event_schema(binding.expression, binding.scope))
            ):
                return self._check_event_fields(schema, path, name)
            return None

        def is_container(annotation: Any) -> bool:
            origin = get_origin(annotation)
            if origin in (Union, UnionType):
                return any(
                    is_container(arg) for arg in get_args(annotation) if arg is not type(None)
                )
            if origin is Annotated:
                return is_container(get_args(annotation)[0])
            kind = origin or annotation
            return isinstance(kind, type) and (
                issubclass(kind, BaseModel)
                or (issubclass(kind, Iterable) and not issubclass(kind, (str, bytes)))
            )

        def check_container_use(
            expression: ConditionNode, scope: Mapping[str, _ScopedName]
        ) -> None:
            types = extension_types(expression, scope)
            if types is not None and any(is_container(annotation) for annotation in types):
                raise ConfigurationError(
                    "Extension JSON collection/object operations are unsupported; "
                    "expose the needed scalar as a schema field"
                )

        def check_container_comparison(
            types: tuple[Any, ...] | None, other: ConditionNode, operator: str
        ) -> None:
            if types is None or not any(is_container(annotation) for annotation in types):
                return
            # A nullable object may still be checked for absence. Other text
            # comparisons would use Python's container repr, not JSON values.
            if operator in {"=", "!="} and isinstance(other, Literal) and other.value is None:
                return
            raise ConfigurationError(
                "Comparing an extension collection or object is unsupported; "
                "expose the needed scalar as a schema field"
            )

        for node, scope, ancestors in _scoped_visits(root):
            if isinstance(
                node, (AnyComprehension, AllComprehension, FilterComprehension, MapComprehension)
            ):
                check_container_use(node.collection, scope)
            elif isinstance(node, FunctionCall) and node.name not in {"_index", "_slice"}:
                for argument in node.args:
                    check_container_use(argument, scope)
            elif isinstance(node, ContainsAnyOp):
                check_container_use(node.left, scope)
            if (
                isinstance(node, BinaryOp)
                and node.operator
                in {
                    "=",
                    "!=",
                    "contains",
                    "in",
                    "~",
                    "matches",
                    ">",
                    "<",
                    ">=",
                    "<=",
                }
                and not (node.operator == "matches" and isinstance(node.right, PatternMatch))
            ):
                check_container_comparison(
                    extension_types(node.left, scope), node.right, node.operator
                )
                check_container_comparison(
                    extension_types(node.right, scope), node.left, node.operator
                )
            if isinstance(node, PatternMatch):
                parent = ancestors[-1] if ancestors else None
                if not (
                    isinstance(parent, BinaryOp)
                    and parent.operator == "matches"
                    and parent.right is node
                ):
                    continue
                subject = parent.left
                schema = None
                if isinstance(subject, FunctionCall) and subject.name == "_index":
                    schema = event_schema(subject.args[0], scope)
                elif isinstance(subject, Identifier):
                    base, path, base_scope = expand_alias(subject, [], scope)
                    if isinstance(base, Identifier) and path and isinstance(path[0], int):
                        schema = event_schema(base, base_scope)
                    elif isinstance(base, Identifier):
                        binding = base_scope.get(base.name)
                        if binding is not None and binding.kind == "element":
                            schema = event_schema(binding.expression, binding.scope)
                if schema is not None:
                    for predicate in node.patterns:
                        if not isinstance(predicate, BinaryOp):
                            continue
                        field = predicate.left
                        if isinstance(field, Identifier):
                            types = self._check_event_fields(schema, [field.name], "pattern")
                        elif isinstance(field, FieldAccess) and field.base is None:
                            types = self._check_event_fields(schema, field.fields, "pattern")
                        else:
                            types = None
                        if predicate.operator in {
                            "=",
                            "!=",
                            "contains",
                            "in",
                            "~",
                            "matches",
                            ">",
                            "<",
                            ">=",
                            "<=",
                        }:
                            check_container_comparison(types, predicate.right, predicate.operator)
                continue
            if isinstance(node, FieldAccess):
                extension_types(node, scope)

    @classmethod
    def _check_event_fields(
        cls, schema: type[BaseModel], fields: Sequence[str | int], label: str
    ) -> tuple[Any, ...]:
        if not fields:
            return (schema,)
        if fields[0] == ITEM_ID:
            if len(fields) > 1:
                raise ConfigurationError(f"Cannot read a child of {label}.{ITEM_ID}")
            return (str,)
        return cls._check_schema_path(schema, fields, label)

    def prepare(
        self,
        sections: Mapping[str, Any],
        limits: SafetyLimits,
        faults: Mapping[str, str] | None = None,
    ) -> PreparedInputs:
        """Validate the sections a selected rule reads, then publish read-only views.

        ``sections`` maps each section the conversation carries, and some selected rule
        reads, to its raw value. A section it does not hold is not supplied: its list reads
        as empty and is reported missing, and a single section reads as missing. A section
        that fails its declaration or a limit, each on its own, is a fault of that section
        only (``faults``, with any the caller found first): the rules that read it fail with
        ``invalid_input``, and every other rule keeps its verdict.
        """

        from quarry.safety.validator import _validate_json_tree as validate_json_tree

        found: dict[str, str] = dict(faults or {})
        lists: dict[str, tuple[InputObject, ...]] = {}
        singles: dict[str, Any] = {}
        for namespace, spec in self.definitions.items():
            for name, schema in (*spec.events.items(), *spec.singles.items()):
                surface = self.surface_name(namespace, name)
                single = name in spec.singles
                empty: Any = MissingInputState(surface) if single else ()
                if surface not in sections or surface in found:
                    (singles if single else lists)[surface] = empty
                    continue
                try:
                    raw = sections[surface]
                    if (
                        limits.enforce_limits
                        and json_size(raw, limits.max_total_size) > limits.max_total_size
                    ):
                        raise ConversationValidationError(f"{surface} exceeds max_total_size")
                    validate_json_tree(raw, surface, limits.max_nesting_depth)
                    if single:
                        if not isinstance(raw, dict):
                            raise ConversationValidationError(f"{surface} must be an object")
                        singles[surface] = InputObject(self._validated(schema, raw, surface))
                    else:
                        lists[surface] = self._items(schema, raw, surface)
                except ConversationValidationError as exc:
                    found[surface] = str(exc)
                    (singles if single else lists)[surface] = empty
        return PreparedInputs(
            MappingProxyType(lists), MappingProxyType(singles), MappingProxyType(found)
        )

    def _items(self, schema: type[BaseModel], raw: Any, surface: str) -> tuple[InputObject, ...]:
        if not isinstance(raw, list):
            raise ConversationValidationError(f"{surface} must be a list of items")
        items: list[InputObject] = []
        ids: set[str] = set()
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ConversationValidationError(f"{surface}[{index}] must be an object")
            item_id = item.get(ITEM_ID)
            if not isinstance(item_id, str) or not item_id or item_id in ids:
                raise ConversationValidationError(
                    f"{surface}[{index}] needs a unique non-empty string 'id'"
                )
            ids.add(item_id)
            fields = {key: value for key, value in item.items() if key != ITEM_ID}
            validated = self._validated(schema, fields, f"{surface}[{index}]")
            items.append(InputObject({ITEM_ID: item_id, **validated}))
        return tuple(items)

    @staticmethod
    def _validated(schema: type[BaseModel], fields: Mapping[str, Any], label: str) -> dict:
        try:
            # JSON-mode strict validation: no string becomes a number, no number a string.
            model = schema.model_validate_json(orjson.dumps(dict(fields)), strict=True)
        except PydanticValidationError as exc:
            raise ConversationValidationError(
                f"Invalid {label}: {exc.errors(include_url=False, include_input=False)}"
            ) from exc
        except orjson.JSONEncodeError as exc:
            raise ConversationValidationError(f"{label} cannot be encoded as JSON") from exc
        return model.model_dump(mode="json", by_alias=True)

    def timeline_read(
        self, entries: Any, read: frozenset[str], conversation: Mapping[str, Any], chat_read: bool
    ) -> Any:
        """The entries of a timeline that name what the selected rules read: those naming a
        section no rule reads (``read``) that is declared or that ``conversation`` carries, or
        naming the chat when no rule reads it (``chat_read``), are left out, as no flow counts
        them as steps. Each entry's section is looked up in the conversation, which is never
        listed. A timeline that is not a list is left as it is, for its check to refuse."""

        if not isinstance(entries, list):
            return entries

        def unread(entry: dict[str, Any]) -> bool:
            section = entry.get("input")
            if isinstance(section, str):
                if section in read:
                    return False
                namespace, _, name = section.partition(".")
                holder = conversation.get(namespace)
                return section in self.event_surfaces or (
                    namespace in self.definitions and isinstance(holder, dict) and name in holder
                )
            # The chat no rule reads: every entry that names a chat item, whatever else it holds.
            return not chat_read and ("message_index" in entry or "tool_call_id" in entry)

        return [entry for entry in entries if not (isinstance(entry, dict) and unread(entry))]

    def order_timeline(
        self,
        conversation_timeline: Sequence[tuple[int, Any]],
        prepared: PreparedInputs,
        order: list[dict[str, Any]],
        limits: SafetyLimits,
    ) -> list[tuple[int, Any]]:
        """Resolve the conversation's ``timeline``: an explicit interleaving of its messages,
        tool calls and input items.

        Conversation entries keep their order. The timeline lists every conversation item and
        every item of a supplied list section once, so an omitted item cannot silently change
        a temporal result; it may reorder a section's items.
        """
        if not isinstance(order, list):
            raise ConversationValidationError("The timeline must be a list")
        if limits.enforce_limits and (
            json_size(order, limits.max_total_size) > limits.max_total_size
        ):
            raise ConversationValidationError("The timeline exceeds max_total_size")
        from quarry.safety.validator import _validate_json_tree as validate_json_tree

        validate_json_tree(order, "timeline", limits.max_nesting_depth)

        canonical: list[tuple[Any, ...]] = []
        items: dict[tuple[Any, ...], Any] = {}
        message_index = 0
        for _, item in conversation_timeline:
            if isinstance(item, ToolCall):
                key: tuple[Any, ...] = ("tool_call_id", item.id)
            else:
                key = ("message_index", message_index)
                message_index += 1
            canonical.append(key)
            items[key] = item
        for surface, section_items in prepared.lists.items():
            for section_item in section_items:
                items[("input", surface, getattr(section_item, ITEM_ID))] = section_item

        if len(order) != len(items):
            raise ConversationValidationError(
                "The timeline must include every conversation item and input item exactly once"
            )

        seen: set[tuple[Any, ...]] = set()
        conversation_order: list[tuple[Any, ...]] = []
        result: list[tuple[int, Any]] = []
        for position, entry in enumerate(order):
            if not isinstance(entry, dict):
                raise ConversationValidationError(f"Timeline entry {position} must be an object")
            if set(entry) == {"input", ITEM_ID}:
                surface, item_id = entry["input"], entry[ITEM_ID]
                if not isinstance(surface, str) or surface not in self.event_surfaces:
                    raise ConversationValidationError(
                        f"Timeline entry {position} names {surface!r}, not a list input"
                    )
                if not isinstance(item_id, str) or not item_id:
                    raise ConversationValidationError(f"Timeline entry {position} has invalid id")
                key = ("input", surface, item_id)
            elif len(entry) == 1 and "message_index" in entry:
                value = entry["message_index"]
                if type(value) is not int or value < 0:
                    raise ConversationValidationError(
                        f"Timeline entry {position} has invalid message_index"
                    )
                key = ("message_index", value)
            elif len(entry) == 1 and "tool_call_id" in entry:
                value = entry["tool_call_id"]
                if not isinstance(value, str) or not value:
                    raise ConversationValidationError(
                        f"Timeline entry {position} has invalid tool_call_id"
                    )
                key = ("tool_call_id", value)
            else:
                raise ConversationValidationError(
                    f"Timeline entry {position} must be {{message_index}}, {{tool_call_id}} "
                    "or {input, id}"
                )
            if key not in items or key in seen:
                raise ConversationValidationError(
                    f"Timeline entry {position} is unknown or repeated"
                )
            seen.add(key)
            if key[0] != "input":
                conversation_order.append(key)
            result.append((position, items[key]))

        if conversation_order != canonical:
            raise ConversationValidationError("The timeline must keep the conversation's order")
        return result


@dataclass(frozen=True)
class PreparedInputs:
    """The validated sections of one scan: each list section's items, each single section's
    object or ``MissingInputState``, and each faulty section's reason."""

    lists: Mapping[str, tuple[InputObject, ...]]
    singles: Mapping[str, Any]
    # Each read section that failed its declaration or a limit, with why.
    faults: Mapping[str, str]
