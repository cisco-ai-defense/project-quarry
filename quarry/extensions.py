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

"""Explicit, scanner-local schemas for facts outside a conversation.

Extension packages define schemas; the host registers trusted definitions and
passes JSON data to each scan. Rule text never loads Python modules.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType, UnionType
from typing import Annotated, Any, Union, cast, get_args, get_origin

import orjson
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from quarry.safety.limits import SafetyLimits
from quarry.safety.validator import _validate_json_tree
from quarry.types.context import _bounded_deep_size
from quarry.types.dsl import (
    AllComprehension,
    AnyComprehension,
    Assignment,
    BinaryOp,
    ConditionNode,
    FieldAccess,
    FilterComprehension,
    FlowOp,
    FunctionCall,
    Identifier,
    Literal,
    MapComprehension,
    PatternMatch,
    condition_child_fields,
)
from quarry.types.exceptions import ConfigurationError, ConversationValidationError
from quarry.types.messages import ToolCall

_NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*\Z")
_RESERVED = frozenset(
    {"conversation", "true", "false", "null", "params", "extension_events", "extension_state"}
)


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
class ExtensionSpec:
    """One trusted namespace and its event/state Pydantic schemas.

    Event names are local to the namespace. For example, ``memory`` with a
    ``write_committed`` event is available as ``memory.write_committed``.
    Schemas must forbid extra fields; otherwise misspelled evidence is silent.
    """

    namespace: str
    events: Mapping[str, type[BaseModel]]
    state: type[BaseModel] | None = None


class ExtensionObject:
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

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Extension objects are immutable")


@dataclass(frozen=True)
class MissingExtensionState:
    """Marks state that must not silently behave like a complete no-match."""

    namespace: str


def _freeze(value: Any) -> Any:
    if isinstance(value, BaseModel):
        # Serialize declared fields to JSON primitives before publishing them.
        # This snapshots nested mutable values and omits computed model fields.
        return ExtensionObject(value.model_dump(mode="json", include=set(type(value).model_fields)))
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


class ExtensionRegistry:
    """Validated and frozen registry, owned by one scanner generation."""

    def __init__(self, extensions: Sequence[ExtensionSpec] | None = None):
        from quarry.language.runtime import surface_names

        reserved = _RESERVED | surface_names()
        definitions: dict[str, ExtensionSpec] = {}
        for spec in extensions or ():
            if not isinstance(spec, ExtensionSpec):
                raise ConfigurationError("extensions must contain ExtensionSpec values")
            namespace = spec.namespace
            if not isinstance(namespace, str) or not _NAME.fullmatch(namespace):
                raise ConfigurationError(f"Invalid extension namespace {namespace!r}")
            if namespace in reserved or namespace in definitions:
                raise ConfigurationError(f"Duplicate or reserved extension namespace {namespace!r}")
            if not isinstance(spec.events, Mapping):
                raise ConfigurationError(f"Extension {namespace!r} events must be a mapping")
            event_types: dict[str, type[BaseModel]] = {}
            for name, schema in spec.events.items():
                if not isinstance(name, str) or not _NAME.fullmatch(name) or name == "state":
                    raise ConfigurationError(f"Invalid event name {name!r} in {namespace!r}")
                self._check_schema(schema, f"{namespace}.{name}")
                event_types[name] = schema
            if spec.state is not None:
                self._check_schema(spec.state, f"{namespace}.state")
            definitions[namespace] = ExtensionSpec(
                namespace, MappingProxyType(event_types), spec.state
            )
        self.definitions: Mapping[str, ExtensionSpec] = MappingProxyType(definitions)

    @staticmethod
    def surface_name(namespace: str, name: str) -> str:
        """Qualified name of a registered event or state surface."""
        return f"{namespace}.{name}"

    @property
    def event_surfaces(self) -> frozenset[str]:
        return frozenset(
            self.surface_name(namespace, name)
            for namespace, spec in self.definitions.items()
            for name in spec.events
        )

    @property
    def state_surfaces(self) -> frozenset[str]:
        return frozenset(
            self.surface_name(namespace, "state")
            for namespace, spec in self.definitions.items()
            if spec.state is not None
        )

    def resolve(self, namespace: str, name: str) -> str | None:
        spec = self.definitions.get(namespace)
        if spec is None or (name not in spec.events and not (name == "state" and spec.state)):
            return None
        return self.surface_name(namespace, name)

    def rewrite_rule(self, rule: Any) -> None:
        """Resolve dotted extension names to scanner-local compiler surfaces.

        This is an AST transformation, not a text replacement: unrelated fields and
        quoted strings cannot accidentally acquire extension meaning.
        """

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
                        raise ConfigurationError(f"Unknown extension surface {namespace}.{name}")
                    if name != "state":
                        if len(node.fields) != 1:
                            raise ConfigurationError(
                                f"Use {namespace}.{name} as a collection, then read its bound event"
                            )
                        rewritten[id(node)] = Identifier(surface)
                        continue
                    rewritten[id(node)] = (
                        FieldAccess(Identifier(surface), node.fields[1:])
                        if len(node.fields) > 1
                        else Identifier(surface)
                    )
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

        for name in ("condition", "collect"):
            node = getattr(rule, name)
            if node is not None:
                rewritten = rewrite(node)
                self._validate_rule_fields(rewritten)
                self._diagnose_unregistered_surfaces(rewritten)
                setattr(rule, name, rewritten)

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
            if not collection_position and node.fields[0] != "state":
                continue
            namespace = node.base.name
            raise ConfigurationError(
                f"Unregistered extension namespace {namespace!r} in "
                f"{namespace}.{node.fields[0]}; register its ExtensionSpec on the scanner"
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

        current: tuple[Any, ...] = (schema,)
        for field_name in fields:
            next_types: list[Any] = []
            for annotation in current:
                for option in variants(annotation):
                    if option is Any:
                        next_types.append(Any)
                    elif isinstance(option, type) and issubclass(option, BaseModel):
                        field = (
                            option.model_fields.get(field_name)
                            if isinstance(field_name, str)
                            else None
                        )
                        if field is None:
                            raise ConfigurationError(
                                f"Unknown extension field {label}.{field_name}"
                            )
                        next_types.append(field.annotation)
                    elif get_origin(option) in (list, tuple, Sequence):
                        if not isinstance(field_name, int):
                            raise ConfigurationError(
                                f"Cannot read {field_name!r} from list {label}"
                            )
                        args = get_args(option)
                        if (
                            get_origin(option) is tuple
                            and len(args) > 1
                            and args[-1] is not Ellipsis
                        ):
                            if not -len(args) <= field_name < len(args):
                                raise ConfigurationError(
                                    f"Tuple index {field_name} is outside {label}"
                                )
                            next_types.append(args[field_name])
                        else:
                            next_types.append(args[0] if args else Any)
                    elif get_origin(option) in (dict, Mapping):
                        args = get_args(option)
                        next_types.append(args[1] if len(args) == 2 else Any)
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
            self.surface_name(namespace, "state"): spec.state
            for namespace, spec in self.definitions.items()
            if spec.state is not None
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
            # `Any` may be an object or array at runtime; text containment on
            # it would silently search a Python representation instead.
            if annotation is Any:
                return True
            origin = get_origin(annotation)
            if origin in (Union, UnionType):
                return any(
                    is_container(arg) for arg in get_args(annotation) if arg is not type(None)
                )
            if origin is Annotated:
                return is_container(get_args(annotation)[0])
            if origin in (list, tuple, dict, set, frozenset, Mapping, Sequence):
                return True
            return isinstance(annotation, type) and issubclass(annotation, BaseModel)

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
                "read a scalar field or index an item"
            )

        for node, scope, ancestors in _scoped_visits(root):
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
        if fields[0] not in {"id", "type", "run_id", "payload"}:
            raise ConfigurationError(
                f"Unknown extension event field {label}.{'.'.join(map(str, fields))}"
            )
        if fields[0] == "payload":
            return cls._check_schema_path(schema, fields[1:], label + ".payload")
        elif len(fields) > 1:
            raise ConfigurationError(f"Cannot read a child of {label}.{fields[0]}")
        return (str,)

    @staticmethod
    def _check_schema(schema: type[BaseModel], label: str) -> None:
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise ConfigurationError(f"{label} must use a Pydantic BaseModel schema")
        seen: set[type[BaseModel]] = set()

        def visit(annotation: Any) -> None:
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                if annotation in seen:
                    return
                seen.add(annotation)
                if annotation.model_config.get("extra") != "forbid":
                    raise ConfigurationError(
                        f"{label} nested schema {annotation.__name__} must set extra='forbid'"
                    )
                for field in annotation.model_fields.values():
                    visit(field.annotation)
            else:
                for argument in get_args(annotation):
                    visit(argument)

        visit(schema)

    def prepare(
        self,
        events: list[dict[str, Any]] | None,
        state: Mapping[str, Any] | None,
        limits: SafetyLimits,
    ) -> Mapping[str, ExtensionObject]:
        """Validate JSON input, then publish read-only namespace views."""
        supplied = events is not None or state is not None
        events = [] if events is None else events
        state = {} if state is None else state
        if not isinstance(events, list) or not isinstance(state, dict):
            raise ConversationValidationError("Extension events must be a list and state an object")
        if not self.definitions and supplied:
            raise ConversationValidationError("Extension input requires a registered extension")
        if limits.enforce_limits and (events or state):
            try:
                size = _bounded_deep_size({"events": events, "state": state}, limits.max_total_size)
            except RecursionError as exc:
                raise ConversationValidationError(
                    "Extension input exceeds max_nesting_depth"
                ) from exc
            if size > limits.max_total_size:
                raise ConversationValidationError("Extension input exceeds max_total_size")
        # The bounded preflight runs first so a wide input cannot force an
        # unbounded validation walk. Reject cycles/depth before Pydantic runs.
        _validate_json_tree(
            {"events": events, "state": state}, "extension input", limits.max_nesting_depth
        )
        if set(state) - set(self.definitions):
            raise ConversationValidationError("State contains an unregistered extension namespace")

        buckets: dict[str, dict[str, list[ExtensionObject]]] = {
            namespace: {name: [] for name in spec.events}
            for namespace, spec in self.definitions.items()
        }
        ids: set[str] = set()
        for index, event in enumerate(events):
            if not isinstance(event, dict) or set(event) - {"id", "type", "run_id", "payload"}:
                raise ConversationValidationError(
                    f"Extension event {index} has invalid envelope fields"
                )
            event_id, event_type = event.get("id"), event.get("type")
            if not isinstance(event_id, str) or not event_id or event_id in ids:
                raise ConversationValidationError(
                    f"Extension event {index} needs a unique nonempty id"
                )
            ids.add(event_id)
            if not isinstance(event_type, str) or event_type.count(".") != 1:
                raise ConversationValidationError(f"Extension event {index} has invalid type")
            namespace, name = event_type.split(".")
            spec = self.definitions.get(namespace)
            if spec is None or name not in spec.events:
                raise ConversationValidationError(
                    f"Unregistered extension event type {event_type!r}"
                )
            run_id = event.get("run_id")
            if run_id is not None and (not isinstance(run_id, str) or not run_id):
                raise ConversationValidationError(f"Extension event {index} has invalid run_id")
            if not isinstance(event.get("payload"), dict):
                raise ConversationValidationError(
                    f"Extension event {index} payload must be an object"
                )
            try:
                # Input has already passed JSON shape, depth, and size checks.
                # JSON-mode strict validation accepts wire formats such as ISO
                # datetimes and UUID strings without coercing a string to int.
                payload = spec.events[name].model_validate_json(
                    orjson.dumps(event["payload"]), strict=True
                )
            except PydanticValidationError as exc:
                raise ConversationValidationError(
                    f"Invalid payload for {event_type!r}: {exc.errors(include_url=False, include_input=False)}"
                ) from exc
            except orjson.JSONEncodeError as exc:
                raise ConversationValidationError(
                    f"Extension event {index} payload cannot be encoded as JSON"
                ) from exc
            buckets[namespace][name].append(
                ExtensionObject(
                    {"id": event_id, "type": event_type, "run_id": run_id, "payload": payload}
                )
            )

        views: dict[str, ExtensionObject] = {}
        for namespace, spec in self.definitions.items():
            fields: dict[str, Any] = {
                name: tuple(items) for name, items in buckets[namespace].items()
            }
            if spec.state is not None:
                if namespace in state:
                    raw_state = state[namespace]
                    if not isinstance(raw_state, dict):
                        raise ConversationValidationError(f"{namespace}.state must be an object")
                    try:
                        fields["state"] = spec.state.model_validate_json(
                            orjson.dumps(raw_state), strict=True
                        )
                    except PydanticValidationError as exc:
                        raise ConversationValidationError(
                            f"Invalid state for {namespace!r}: {exc.errors(include_url=False, include_input=False)}"
                        ) from exc
                    except orjson.JSONEncodeError as exc:
                        raise ConversationValidationError(
                            f"{namespace}.state cannot be encoded as JSON"
                        ) from exc
                else:
                    fields["state"] = MissingExtensionState(namespace)
            elif namespace in state:
                raise ConversationValidationError(f"{namespace!r} does not register a state schema")
            views[namespace] = ExtensionObject(fields)
        return MappingProxyType(views)

    def event_collections(
        self, views: Mapping[str, ExtensionObject]
    ) -> dict[str, tuple[ExtensionObject, ...]]:
        return {
            self.surface_name(namespace, name): getattr(views[namespace], name)
            for namespace, spec in self.definitions.items()
            for name in spec.events
        }

    def state_values(self, views: Mapping[str, ExtensionObject]) -> dict[str, Any]:
        return {
            self.surface_name(namespace, "state"): views[namespace].state
            for namespace, spec in self.definitions.items()
            if spec.state is not None
        }

    def order_timeline(
        self,
        conversation_timeline: Sequence[tuple[int, Any]],
        views: Mapping[str, ExtensionObject],
        order: list[dict[str, Any]],
        limits: SafetyLimits,
    ) -> list[tuple[int, Any]]:
        """Resolve a complete, explicit interleaving of conversation and extension facts.

        Conversation entries must retain their canonical order. A manifest must
        include every conversation item and every supplied extension event once,
        so omitted facts cannot silently change a temporal result.
        """
        if not isinstance(order, list):
            raise ConversationValidationError("Extension timeline must be a list")
        if limits.enforce_limits:
            try:
                size = _bounded_deep_size(order, limits.max_total_size)
            except RecursionError as exc:
                raise ConversationValidationError(
                    "Extension timeline exceeds max_nesting_depth"
                ) from exc
            if size > limits.max_total_size:
                raise ConversationValidationError("Extension timeline exceeds max_total_size")
        _validate_json_tree(order, "extension timeline", limits.max_nesting_depth)

        canonical: list[tuple[str, int | str]] = []
        items: dict[tuple[str, int | str], Any] = {}
        message_index = 0
        for _, item in conversation_timeline:
            if isinstance(item, ToolCall):
                key: tuple[str, int | str] = ("tool_call_id", item.id)
            else:
                key = ("message_index", message_index)
                message_index += 1
            canonical.append(key)
            items[key] = item

        for namespace, spec in self.definitions.items():
            view = views[namespace]
            for event_name in spec.events:
                for event in getattr(view, event_name):
                    items[("event_id", event.id)] = event

        if len(order) != len(items):
            raise ConversationValidationError(
                "Extension timeline must include every conversation item and event exactly once"
            )

        seen: set[tuple[str, int | str]] = set()
        conversation_order: list[tuple[str, int | str]] = []
        result: list[tuple[int, Any]] = []
        for position, entry in enumerate(order):
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ConversationValidationError(
                    f"Extension timeline entry {position} must have one reference"
                )
            kind, value = next(iter(entry.items()))
            if kind == "message_index":
                if type(value) is not int or value < 0:
                    raise ConversationValidationError(
                        f"Extension timeline entry {position} has invalid message_index"
                    )
            elif kind in {"tool_call_id", "event_id"}:
                if not isinstance(value, str) or not value:
                    raise ConversationValidationError(
                        f"Extension timeline entry {position} has invalid {kind}"
                    )
            else:
                raise ConversationValidationError(
                    f"Extension timeline entry {position} has unknown reference {kind!r}"
                )
            key = (kind, value)
            if key not in items or key in seen:
                raise ConversationValidationError(
                    f"Extension timeline entry {position} is unknown or repeated"
                )
            seen.add(key)
            if kind != "event_id":
                conversation_order.append(key)
            result.append((position, items[key]))

        if conversation_order != canonical:
            raise ConversationValidationError(
                "Extension timeline must preserve conversation item order"
            )
        return result
