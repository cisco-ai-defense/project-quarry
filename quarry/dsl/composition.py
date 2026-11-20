# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Link calls to public rules before validation and compilation.

A rule may import another public rule under a local name and call it on one messages
value. Linking keeps the called rule's strings and parameters distinct, and embeds its
condition as a check over the supplied messages. The compiler evaluates that check as a
rule of its own, using the same scanner inputs and limits as its caller.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import NoReturn

from quarry.engine.contracts import CompiledRuleArtifact, DecisionTree
from quarry.engine.desugar import COMPOSED_CHECK
from quarry.language.runtime import surface_names
from quarry.rule_versioning import InvalidQuarryRequirement, parse_quarry_requirement
from quarry.types.dsl import (
    AnyComprehension,
    ConditionNode,
    FunctionCall,
    Identifier,
    StringCount,
    StringPosition,
    StringSetQuantifier,
    condition_child_fields,
    walk_condition_visits,
)
from quarry.types.dsl.traversal import ConditionVisit, binding_comprehensions
from quarry.types.exceptions import RuleValidationError
from quarry.types.parameters import ParameterSpec
from quarry.types.rules import DSLRule

MAX_IMPORTS_PER_RULE = 128
MAX_IMPORT_NAME_LENGTH = 256
_RESERVED_STRING_PREFIX = "$__quarry_"
_CALL_STRING_PREFIX = "$__quarry_call_"
_CALL_STRING_REFERENCE = re.compile(r"([$@#])__quarry_call_(\d{3})_([a-zA-Z0-9_]+)")


def _fail(rule: DSLRule, sources: Mapping[str, str], message: str, reason: str) -> NoReturn:
    source = sources.get(rule.name, "<unknown>")
    raise RuleValidationError(
        f"Rule '{rule.name}' in {source} has invalid signature composition: {message}",
        {
            "rule_name": rule.name,
            "source": source,
            "error_type": "signature_composition",
            "reason": reason,
        },
    )


def _check_text(rule: DSLRule, sources: Mapping[str, str], value: str) -> None:
    if not value.strip() or len(value) > MAX_IMPORT_NAME_LENGTH:
        _fail(rule, sources, "import names must contain 1..256 characters", "invalid_name")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _fail(rule, sources, "import names must be valid UTF-8 text", "invalid_name")


def _bound_names(visit: ConditionVisit) -> set[str]:
    return {node.variable for node in binding_comprehensions(visit)}


def _call_name(visit: ConditionVisit, names: set[str]) -> str | None:
    node = visit.node
    if (
        isinstance(node, FunctionCall)
        and node.name in names
        and node.name not in _bound_names(visit)
    ):
        return node.name
    return None


def _validate_calls(rule: DSLRule, sources: Mapping[str, str]) -> None:
    """A public rule import is one call in the condition, never a bare value."""
    names = {group.name for group in rule.import_groups}
    for visit in walk_condition_visits(rule.collect):
        if (callee := _call_name(visit, names)) is not None:
            _fail(rule, sources, f"`{callee}(...)` must be called in condition:", "call_placement")
    for visit in walk_condition_visits(rule.condition):
        node = visit.node
        if (
            isinstance(node, Identifier)
            and node.name in names
            and node.name not in _bound_names(visit)
        ):
            _fail(
                rule,
                sources,
                f"import '{node.name}' is a rule; call it as `{node.name}(messages)`",
                "call_reference",
            )
        if isinstance(node, StringSetQuantifier) and names & set(node.patterns):
            _fail(rule, sources, "rule imports cannot be used in `any of (...)`", "call_reference")
        if (
            (callee := _call_name(visit, names)) is not None
            and isinstance(node, FunctionCall)
            and len(node.args) != 1
        ):
            _fail(
                rule,
                sources,
                f"`{callee}(...)` takes one messages value",
                "call_arguments",
            )


def _validate_imports(rule: DSLRule, sources: Mapping[str, str]) -> None:
    groups = rule.import_groups
    if len(groups) > MAX_IMPORTS_PER_RULE:
        _fail(rule, sources, f"imports exceeds {MAX_IMPORTS_PER_RULE} rules", "import_limit")
    names = [group.name for group in groups]
    if len(names) != len(set(names)):
        _fail(rule, sources, "imports names a group twice", "duplicate_import")
    reserved = set(names) & (surface_names() | {"params", "true", "false", "null"})
    if reserved:
        _fail(
            rule,
            sources,
            "import groups cannot shadow rule surfaces: " + ", ".join(sorted(reserved)),
            "reserved_group_name",
        )
    for group in groups:
        _check_text(rule, sources, group.name)
        _check_text(rule, sources, group.rule)
    for name in rule.strings or {}:
        if name.startswith(_RESERVED_STRING_PREFIX):
            _fail(
                rule,
                sources,
                f"string names beginning with {_RESERVED_STRING_PREFIX!r} are reserved",
                "reserved_string_prefix",
            )
    _validate_calls(rule, sources)


def _call_order(rules: Mapping[str, DSLRule], sources: Mapping[str, str]) -> tuple[str, ...]:
    """Each caller follows the rules it calls; reject direct and indirect cycles."""
    order: list[str] = []
    state: dict[str, str] = {}

    def visit(name: str, path: tuple[str, ...]) -> None:
        if state.get(name) == "done":
            return
        if state.get(name) == "open":
            cycle = " -> ".join((*path[path.index(name) :], name))
            _fail(rules[path[-1]], sources, f"rule calls form a cycle: {cycle}", "cycle")
        if len(path) > MAX_IMPORTS_PER_RULE:
            _fail(rules[name], sources, "rule call chain exceeds the import limit", "import_limit")
        state[name] = "open"
        for group in rules[name].import_groups:
            visit(group.rule, (*path, name))
        state[name] = "done"
        order.append(name)

    for name, rule in rules.items():
        if rule.import_groups:
            visit(name, ())
    return tuple(order)


def _rename_strings(condition: ConditionNode, rename) -> None:
    """Carry the called rule's string names into its caller without collisions."""
    seen: set[int] = set()
    for visit in walk_condition_visits(condition):
        node = visit.node
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Identifier) and node.name.startswith("$"):
            node.name = rename(node.name, True)
        elif isinstance(node, (StringPosition, StringCount)):
            node.variable = rename(node.variable, False)
        elif isinstance(node, StringSetQuantifier):
            node.patterns = [rename(p, True) if p.startswith("$") else p for p in node.patterns]


def _namespace_spelling(spelling: str, prefix: str, *, dollar: bool) -> str:
    if dollar:
        return prefix + spelling[1:] if spelling.startswith("$") else spelling
    return prefix[1:] + spelling


def _linked_parameters(
    owner: DSLRule, children: Sequence[DSLRule], sources: Mapping[str, str]
) -> tuple[ParameterSpec, ...]:
    declared: dict[str, tuple[ParameterSpec, str]] = {}
    for child in children:
        for spec in child.parameters:
            found = declared.setdefault(spec.name, (spec, child.name))
            if found[0] != spec:
                _fail(
                    owner,
                    sources,
                    f"parameter '{spec.name}' differs between '{found[1]}' and '{child.name}'",
                    "parameter_conflict",
                )
    return tuple(spec for spec, _ in declared.values())


def _combined_requirement(
    owner: DSLRule, children: Sequence[DSLRule], sources: Mapping[str, str]
) -> str | None:
    declarations = [
        rule.metadata["requires_quarry"]
        for rule in (owner, *children)
        if "requires_quarry" in rule.metadata
    ]
    if not declarations:
        return None
    try:
        requirements = [parse_quarry_requirement(item) for item in declarations]
    except InvalidQuarryRequirement as exc:
        _fail(owner, sources, f"invalid requires_quarry metadata: {exc}", "invalid_requirement")
    lower = max(requirement.lower for requirement in requirements)
    uppers = [requirement.upper for requirement in requirements if requirement.upper is not None]
    upper = min(uppers) if uppers else None
    if upper is not None and lower >= upper:
        _fail(
            owner,
            sources,
            "called rules require incompatible Quarry versions",
            "incompatible_requirements",
        )
    return f">={lower}" + (f",<{upper}" if upper is not None else "")


def composed_sha256(source_sha256: str, imports: Sequence[tuple[str, str | None]]) -> str:
    canonical = json.dumps(
        {"rule": source_sha256, "imports": [list(item) for item in imports]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _check(variable: str, conversation: ConditionNode, body: ConditionNode) -> AnyComprehension:
    return AnyComprehension(variable, copy.deepcopy(conversation), body)


def _link_calls(
    rule: DSLRule,
    linked: Mapping[str, DSLRule],
    sources: Mapping[str, str],
) -> DSLRule:
    groups = rule.import_groups
    calls = {group.name: linked[group.rule] for group in groups}
    order = [group.name for group in groups]
    strings = dict(rule.strings or {})
    condition = copy.deepcopy(rule.condition)
    assert condition is not None
    targets = {
        id(visit.node): name
        for visit in walk_condition_visits(condition)
        if (name := _call_name(visit, set(calls))) is not None
    }

    def callee_check(name: str, argument: ConditionNode) -> ConditionNode:
        callee, index = calls[name], order.index(name)
        prefix = f"{_CALL_STRING_PREFIX}{index:03d}_"
        body = copy.deepcopy(callee.condition)
        assert body is not None
        for key, value in copy.deepcopy(callee.strings or {}).items():
            strings[_namespace_spelling(key, prefix, dollar=True)] = value
        _rename_strings(
            body, lambda spelling, dollar: _namespace_spelling(spelling, prefix, dollar=dollar)
        )
        return _check(f"{COMPOSED_CHECK}call{index}", argument, body)

    def rewrite(node: ConditionNode) -> ConditionNode:
        if id(node) in targets:
            assert isinstance(node, FunctionCall)
            return callee_check(targets[id(node)], rewrite(node.args[0]))
        for field_name in condition_child_fields(node):
            value = getattr(node, field_name)
            if isinstance(value, ConditionNode):
                setattr(node, field_name, rewrite(value))
            elif isinstance(value, list | tuple):
                setattr(node, field_name, type(value)(rewrite(child) for child in value))
        return node

    result = copy.copy(rule)
    result.condition = rewrite(condition)
    children = [calls[name] for name in order]
    result.strings = strings
    result.metadata = dict(rule.metadata)
    result.parameters = _linked_parameters(rule, [rule, *children], sources)
    requirement = _combined_requirement(rule, children, sources)
    if requirement is not None:
        result.metadata["requires_quarry"] = requirement
    result.import_groups = ()
    result.called_rules = tuple(child.name for child in children)
    assert rule.source_sha256 is not None
    result.source_sha256 = composed_sha256(
        rule.source_sha256, [(child.name, child.source_sha256) for child in children]
    )
    return result


def link_public_rule_calls(
    rules: Mapping[str, DSLRule], sources: Mapping[str, str]
) -> dict[str, DSLRule]:
    """Validate and link public-rule calls in an unpublished rule generation."""
    for rule in rules.values():
        if not rule.import_groups:
            continue
        _validate_imports(rule, sources)
        for group in rule.import_groups:
            if group.rule not in rules:
                _fail(rule, sources, f"import '{group.rule}' is not defined", "invalid_import")
    order = _call_order(rules, sources)
    linked = dict(rules)
    for name in order:
        if linked[name].import_groups:
            linked[name] = _link_calls(linked[name], linked, sources)
    return linked


def label_called_rule_artifact(
    artifact: CompiledRuleArtifact, rule: DSLRule
) -> CompiledRuleArtifact:
    """Show the called rule's name in human-readable string references."""

    if not rule.called_rules or artifact.decision_tree is None:
        return artifact

    def display(match: re.Match[str]) -> str:
        index = int(match[2])
        return (
            f"{rule.called_rules[index]}::{match[1]}{match[3]}"
            if index < len(rule.called_rules)
            else match[0]
        )

    nodes = tuple(
        replace(node, label=_CALL_STRING_REFERENCE.sub(display, node.label))
        for node in artifact.decision_tree.nodes
    )
    return replace(artifact, decision_tree=DecisionTree(nodes))
