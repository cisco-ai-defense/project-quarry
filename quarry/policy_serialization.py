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

"""Versioned, data-only serialization for :class:`quarry.policy.CompiledPolicy`.

The host authenticates stored artifacts before passing them to this module. The loader
therefore restores compiler-owned data and checks runtime compatibility; it does not
recompile source or attempt to prove that arbitrary input could have been compiler output.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import orjson

from quarry.config import QuarryConfig
from quarry.engine import temporal as T
from quarry.engine.atoms import AtomDomain, AtomSpec, Cost, ModelExpression, StringSemantics
from quarry.engine.captures import CaptureBind, CapturePlan, CaptureTest, ElementPlan
from quarry.engine.collect import CollectPlan
from quarry.engine.contracts import (
    COMPILER_VERSION,
    ClosedPlan,
    CompiledRuleArtifact,
    DecisionTree,
    ToolQuantifierPlan,
    TreeNode,
)
from quarry.engine.decision import boot_diagram_is_decided
from quarry.engine.symbolic_dfa import DFA, DFAState, Guard, GuardLeaf, GuardNode
from quarry.integrations.resources import requirements_from_artifact
from quarry.language.runtime import surface_names
from quarry.policy import CompiledPolicy, _canonical_sha256
from quarry.rule_versioning import check_rule_compatibility
from quarry.scanner import DSLScanner
from quarry.types.exceptions import RuleCompatibilityError
from quarry.types.parameters import ParameterSpec, validate_parameter_values
from quarry.types.rules import DSLRule

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

FORMAT_KIND = "quarry.compiled-policy"
FORMAT_VERSION = 6
_WIRE_SCHEMAS: list[tuple[type[DataclassInstance], set[str]]] = []
_RESOURCE_REQUIREMENT_KEYS = frozenset(
    {
        "llm_models",
        "llm_score_models",
        "llm_complete_models",
        "llm_token_models",
        "model_classification_models",
        "model_span_models",
        "text_resources",
        "model_operations",
    }
)


def _wire_fields(
    dataclass_type: type[DataclassInstance], names: str, *, unserialized: str = ""
) -> set[str]:
    """Pin the internal dataclass shape represented by the current format version.

    ``unserialized`` names fields deliberately left off the wire; they load as defaults.
    """

    expected = set(names.split())
    _WIRE_SCHEMAS.append((dataclass_type, expected | set(unserialized.split())))
    return expected


def _check_wire_schemas() -> None:
    """Stop serialization work when internal shapes outgrow the current format version."""

    for dataclass_type, expected in _WIRE_SCHEMAS:
        if {item.name for item in fields(dataclass_type)} != expected:
            raise RuntimeError(
                f"{dataclass_type.__name__} changed without updating compiled-policy "
                f"format v{FORMAT_VERSION}"
            )


_TOP_LEVEL_KEYS = {
    "kind",
    "format_version",
    "config_version",
    "source_sha256",
    "policy_digest",
    "policy_parameters",
    "text_resources",
    "models",
    "rule",
}
_RULE_KEYS = {"metadata", "artifact"}
_ARTIFACT_KEYS = _wire_fields(
    CompiledRuleArtifact,
    "rule_name dfa dfa_hash compiler_version atom_specs temporal notes "
    "capture_plan parameter_schema collect_plan tool_quantifiers closed_conditions "
    "full_trace_proof decision_tree",
    # Source provenance serves compile-time visualization of the authored AST, which a
    # loaded policy does not carry.
    unserialized="node_atom_ids",
)
_TOOL_QUANTIFIER_FIELDS = _wire_fields(ToolQuantifierPlan, "body kind body_source_nodes")
_CLOSED_PLAN_FIELDS = _wire_fields(ClosedPlan, "dfa")
_DECISION_TREE_FIELDS = _wire_fields(DecisionTree, "nodes")
_TREE_NODE_FIELDS = _wire_fields(TreeNode, "atom_id label source_node")
_DFA_KEYS = {
    "formula_name",
    "atoms",
    "initial_state",
    "states",
    "guard_nodes",
    "normal_roots",
    "final_roots",
}
_ATOM_FIELDS = _wire_fields(
    AtomSpec,
    "domain selector value_expr_key predicate patterns string_semantics op number cost "
    "detector detector_arg class_index threshold detector_model constitution "
    "constitution_is_ref chunk_max_tokens chunk_combine chunk_scalar chunk_inner_op "
    "chunk_inner_number evidence_policy parameter_names ordinal no_pair_reachability "
    "slice_start slice_stop is_slice input_pipeline constructed_input "
    "constructed_field_paths model_expression atom_id",
)
_STRING_SEMANTICS_FIELDS = _wire_fields(
    StringSemantics,
    "case_insensitive fullword base64 is_regex decode normalize list_mode",
)
_MODEL_EXPRESSION_FIELDS = _wire_fields(ModelExpression, "kind value children")
_CAPTURE_BIND_FIELDS = _wire_fields(
    CaptureBind,
    "surface field patterns is_regex case_insensitive fullword base64 window_start window_stop name",
)
_CAPTURE_TEST_FIELDS = _wire_fields(CaptureTest, "surface field case_insensitive")
_CAPTURE_PLAN_FIELDS = _wire_fields(
    CapturePlan,
    "var bind test bind_atom_id ref_atom_id bind_atom_ids ref_atom_ids",
)
_ELEMENT_PLAN_FIELDS = _wire_fields(
    ElementPlan,
    "var surface at_atom_id ref_atom_ids window_start window_stop filter_atom_id filter_name "
    "own_step",
)
_COLLECT_PLAN_FIELDS = _wire_fields(
    CollectPlan,
    "variable item_type source local_predicate model_predicate has_external_work",
)
_PARAMETER_FIELDS = _wire_fields(ParameterSpec, "name type binding")
_wire_fields(DFA, "formula_name atoms states initial_state normal_guards final_guards")
_wire_fields(DFAState, "id key is_accepting is_rejecting")
_wire_fields(GuardNode, "node_id atom hi lo")
_wire_fields(GuardLeaf, "target")


def _fail(message: str) -> ValueError:
    return ValueError(f"invalid compiled policy: {message}")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _fail(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise _fail(f"{name} must be an array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise _fail(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(f"{name} must be an integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(f"{name} must be a Boolean")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise _fail(
            f"{name} fields do not match format_version {FORMAT_VERSION}: " + "; ".join(details)
        )


def _sha256(value: object, name: str) -> str:
    text = _string(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise _fail(f"{name} must be a lowercase SHA-256 digest")
    return text


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_value(item) for item in sorted(value)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"compiled policy contains unsupported value {type(value).__name__}")


def _guard_table(dfa: DFA) -> tuple[list[list[Any] | None], list[int], list[int]]:
    nodes: list[list[Any] | None] = []
    references: dict[int, int] = {}

    def encode(guard: Guard) -> int:
        if isinstance(guard, GuardLeaf):
            return -(guard.target + 1)
        identity = id(guard)
        if identity in references:
            return references[identity]
        reference = len(nodes)
        references[identity] = reference
        nodes.append(None)
        nodes[reference] = [guard.atom, encode(guard.hi), encode(guard.lo)]
        return reference

    state_ids = [state.id for state in dfa.states]
    normal = [encode(dfa.normal_guards[state_id]) for state_id in state_ids]
    final = [encode(dfa.final_guards[state_id]) for state_id in state_ids]
    return nodes, normal, final


def _dfa_to_wire(dfa: DFA) -> dict[str, Any]:
    nodes, normal, final = _guard_table(dfa)
    return {
        "formula_name": dfa.formula_name,
        "atoms": list(dfa.atoms),
        "initial_state": dfa.initial_state,
        "states": [
            [state.id, state.key, state.is_accepting, state.is_rejecting] for state in dfa.states
        ],
        "guard_nodes": nodes,
        "normal_roots": normal,
        "final_roots": final,
    }


def _artifact_to_wire(artifact: CompiledRuleArtifact) -> dict[str, Any]:
    return {
        "rule_name": artifact.rule_name,
        "dfa_hash": artifact.dfa_hash,
        "compiler_version": artifact.compiler_version,
        "temporal": artifact.temporal,
        "notes": list(artifact.notes),
        "atom_specs": [
            _json_value(artifact.atom_specs[atom_id]) for atom_id in sorted(artifact.atom_specs)
        ],
        "capture_plan": _capture_to_wire(artifact.capture_plan),
        "parameter_schema": [_json_value(spec) for spec in artifact.parameter_schema],
        "collect_plan": _json_value(artifact.collect_plan),
        "tool_quantifiers": [
            [atom_id, plan.kind, _boolean_formula_to_wire(plan.body), list(plan.body_source_nodes)]
            for atom_id, plan in sorted(artifact.tool_quantifiers.items())
        ],
        "closed_conditions": [
            [atom_id, _dfa_to_wire(plan.dfa)]
            for atom_id, plan in sorted(artifact.closed_conditions.items())
        ],
        "full_trace_proof": artifact.full_trace_proof,
        "decision_tree": _decision_tree_to_wire(artifact.decision_tree),
        "dfa": _dfa_to_wire(artifact.dfa),
    }


def _decision_tree_to_wire(tree: DecisionTree | None) -> dict[str, Any] | None:
    if tree is None:
        return None
    return {"nodes": [[node.atom_id, node.label, node.source_node] for node in tree.nodes]}


def _decision_tree_from_wire(raw: object) -> DecisionTree:
    if raw is None:
        # The compiler gives every rule a tree; a policy without one is not its output.
        raise _fail("rule.artifact.decision_tree is required")
    value = _mapping(raw, "rule.artifact.decision_tree")
    _exact_keys(value, {"nodes"}, "rule.artifact.decision_tree")
    nodes = []
    for index, row in enumerate(_list(value["nodes"], "rule.artifact.decision_tree.nodes")):
        name = f"rule.artifact.decision_tree.nodes[{index}]"
        items = _list(row, name)
        if len(items) != 3:
            raise _fail(f"{name} must hold an atom, a label and a source node")
        atom_id, label, source_node = items
        nodes.append(
            TreeNode(
                atom_id=None if atom_id is None else _string(atom_id, f"{name}.atom_id"),
                label=_string(label, f"{name}.label"),
                source_node=(
                    None if source_node is None else _integer(source_node, f"{name}.source_node")
                ),
            )
        )
    if not nodes:
        raise _fail("rule.artifact.decision_tree.nodes must not be empty")
    return DecisionTree(tuple(nodes))


def _check_decision_tree(artifact: CompiledRuleArtifact) -> None:
    """Reject a tree whose checks are not the automaton's, so a walk cannot misread it.

    A decomposed tree's nodes are exactly the root automaton's atoms, each static and read
    at the boot step; any other tree is one node for the whole rule. What a node scores,
    measures and calls is read from its atom, so a file cannot state it differently."""

    tree = artifact.decision_tree
    assert tree is not None
    if not tree.decomposed:
        if len(tree.nodes) != 1:
            raise _fail("an undecomposed decision tree has one whole-rule node")
        return
    atoms = [node.atom_id for node in tree.nodes if node.atom_id is not None]
    if len(atoms) != len(tree.nodes):
        raise _fail("a decomposed decision tree's nodes must each name an atom")
    if artifact.capture_plan is not None or sorted(atoms) != sorted(artifact.dfa.atoms):
        raise _fail("a decomposed decision tree's nodes must be the root automaton's atoms")
    if any(artifact.atom_specs[atom_id].domain is not AtomDomain.STATIC for atom_id in atoms):
        raise _fail("a decomposed decision tree's checks must be static atoms")
    if not boot_diagram_is_decided(artifact.dfa):
        raise _fail("a decomposed decision tree's root must decide the verdict at boot")


def _boolean_formula_to_wire(formula: T.Formula) -> dict[str, Any]:
    match formula:
        case T.TrueConst():
            return {"kind": "true"}
        case T.FalseConst():
            return {"kind": "false"}
        case T.Atom(name):
            return {"kind": "atom", "atom_id": name}
        case T.Not(child):
            return {"kind": "not", "child": _boolean_formula_to_wire(child)}
        case T.And(left, right) | T.Or(left, right) | T.Implies(left, right):
            kind = type(formula).__name__.lower()
            return {
                "kind": kind,
                "left": _boolean_formula_to_wire(left),
                "right": _boolean_formula_to_wire(right),
            }
        case _:
            raise RuntimeError("compiled policy contains a non-Boolean formula")


def serialize_compiled_policy(policy: CompiledPolicy) -> bytes:
    """Serialize compiler output and policy-bound values without runtime state."""

    if any(spec.type == "text" for spec in policy.parameter_schema) or policy._text_resource_hashes:
        raise ValueError(
            "Text parameters and text_ref resources require a private evidence key; "
            "this compiled-policy format cannot serialize them safely"
        )
    _check_wire_schemas()
    scanner = policy._scanner
    artifact = scanner.compiled_rules[policy._rule_name]
    derived_requirements = requirements_from_artifact(artifact)
    mismatched_requirements = sorted(
        name
        for name in _RESOURCE_REQUIREMENT_KEYS
        if derived_requirements[name] != scanner.resource_requirements.get(name, set())
    )
    if mismatched_requirements:
        raise RuntimeError(
            "compiled artifact does not preserve runtime requirements for: "
            + ", ".join(mismatched_requirements)
        )
    rule = scanner.loaded_rules[policy._rule_name]
    payload = {
        "kind": FORMAT_KIND,
        "format_version": FORMAT_VERSION,
        "config_version": scanner.config.config_version,
        "source_sha256": policy._source_sha256,
        "policy_digest": policy.policy_digest,
        "policy_parameters": dict(policy.policy_parameters),
        "text_resources": dict(policy._text_resource_hashes),
        "models": dict(policy._model_hashes),
        "rule": {
            "metadata": dict(rule.metadata),
            "artifact": _artifact_to_wire(artifact),
        },
    }
    return orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)


def _model_expression_from_wire(raw: object, name: str) -> ModelExpression | None:
    if raw is None:
        return None
    value = _mapping(raw, name)
    _exact_keys(value, _MODEL_EXPRESSION_FIELDS, name)
    children = tuple(
        _required_model_expression(child, f"{name}.children[{index}]")
        for index, child in enumerate(_list(value["children"], f"{name}.children"))
    )
    return ModelExpression(
        kind=_string(value["kind"], f"{name}.kind"),
        value=value["value"],
        children=children,
    )


def _required_model_expression(raw: object, name: str) -> ModelExpression:
    expression = _model_expression_from_wire(raw, name)
    if expression is None:
        raise _fail(f"{name} must be a model expression")
    return expression


def _atom_from_wire(raw: object, index: int) -> AtomSpec:
    name = f"rule.artifact.atom_specs[{index}]"
    value = _mapping(raw, name)
    _exact_keys(value, _ATOM_FIELDS, name)
    semantics = _mapping(value["string_semantics"], f"{name}.string_semantics")
    _exact_keys(semantics, _STRING_SEMANTICS_FIELDS, f"{name}.string_semantics")

    kwargs = dict(value)
    try:
        kwargs["domain"] = AtomDomain(value["domain"])
        kwargs["cost"] = Cost(value["cost"])
        kwargs["string_semantics"] = StringSemantics(**semantics)
        for field_name in ("patterns", "parameter_names", "constructed_field_paths"):
            kwargs[field_name] = tuple(_list(value[field_name], f"{name}.{field_name}"))
        if value["input_pipeline"] is not None:
            kwargs["input_pipeline"] = tuple(
                tuple(_list(step, f"{name}.input_pipeline[]"))
                for step in _list(value["input_pipeline"], f"{name}.input_pipeline")
            )
        kwargs["constructed_input"] = tuple(
            tuple(_list(turn, f"{name}.constructed_input[]"))
            for turn in _list(value["constructed_input"], f"{name}.constructed_input")
        )
        kwargs["model_expression"] = _model_expression_from_wire(
            value["model_expression"], f"{name}.model_expression"
        )
        return AtomSpec(**kwargs)
    except (TypeError, ValueError) as exc:
        raise _fail(f"{name} is incompatible with format_version {FORMAT_VERSION}") from exc


def _capture_to_wire(plan: object | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    if isinstance(plan, ElementPlan):
        return {
            "kind": "element",
            **_json_value(plan),
            "own_step": _boolean_formula_to_wire(plan.own_step),
        }
    return {"kind": "value", **_json_value(plan)}


def _optional_integer(value: object, name: str) -> int | None:
    return None if value is None else _integer(value, name)


def _capture_from_wire(raw: object) -> CapturePlan | ElementPlan | None:
    if raw is None:
        return None
    value = _mapping(raw, "rule.artifact.capture_plan")
    kind = value.get("kind")
    if kind == "element":
        _exact_keys(value, _ELEMENT_PLAN_FIELDS | {"kind"}, "rule.artifact.capture_plan")
        return ElementPlan(
            var=_string(value["var"], "rule.artifact.capture_plan.var"),
            surface=_string(value["surface"], "rule.artifact.capture_plan.surface"),
            at_atom_id=_string(value["at_atom_id"], "rule.artifact.capture_plan.at_atom_id"),
            ref_atom_ids=frozenset(
                _string(item, "rule.artifact.capture_plan.ref_atom_ids[]")
                for item in _list(value["ref_atom_ids"], "rule.artifact.capture_plan.ref_atom_ids")
            ),
            window_start=_optional_integer(
                value["window_start"], "rule.artifact.capture_plan.window_start"
            ),
            window_stop=_optional_integer(
                value["window_stop"], "rule.artifact.capture_plan.window_stop"
            ),
            filter_atom_id=_string(
                value["filter_atom_id"], "rule.artifact.capture_plan.filter_atom_id"
            ),
            filter_name=_string(value["filter_name"], "rule.artifact.capture_plan.filter_name"),
            own_step=_boolean_formula_from_wire(
                value["own_step"], "rule.artifact.capture_plan.own_step"
            ),
        )
    if kind != "value":
        raise _fail("rule.artifact.capture_plan.kind must be 'value' or 'element'")
    _exact_keys(value, _CAPTURE_PLAN_FIELDS | {"kind"}, "rule.artifact.capture_plan")
    bind = _mapping(value["bind"], "rule.artifact.capture_plan.bind")
    test = _mapping(value["test"], "rule.artifact.capture_plan.test")
    _exact_keys(bind, _CAPTURE_BIND_FIELDS, "rule.artifact.capture_plan.bind")
    _exact_keys(test, _CAPTURE_TEST_FIELDS, "rule.artifact.capture_plan.test")
    return CapturePlan(
        var=_string(value["var"], "rule.artifact.capture_plan.var"),
        bind=CaptureBind(
            surface=_string(bind["surface"], "rule.artifact.capture_plan.bind.surface"),
            field=_string(bind["field"], "rule.artifact.capture_plan.bind.field"),
            patterns=tuple(_list(bind["patterns"], "rule.artifact.capture_plan.bind.patterns")),
            is_regex=_boolean(bind["is_regex"], "rule.artifact.capture_plan.bind.is_regex"),
            case_insensitive=_boolean(
                bind["case_insensitive"], "rule.artifact.capture_plan.bind.case_insensitive"
            ),
            fullword=_boolean(bind["fullword"], "rule.artifact.capture_plan.bind.fullword"),
            base64=_boolean(bind["base64"], "rule.artifact.capture_plan.bind.base64"),
            window_start=_optional_integer(
                bind["window_start"], "rule.artifact.capture_plan.bind.window_start"
            ),
            window_stop=_optional_integer(
                bind["window_stop"], "rule.artifact.capture_plan.bind.window_stop"
            ),
            name=_string(bind["name"], "rule.artifact.capture_plan.bind.name"),
        ),
        test=CaptureTest(
            surface=_string(test["surface"], "rule.artifact.capture_plan.test.surface"),
            field=_string(test["field"], "rule.artifact.capture_plan.test.field"),
            case_insensitive=_boolean(
                test["case_insensitive"], "rule.artifact.capture_plan.test.case_insensitive"
            ),
        ),
        bind_atom_id=_string(value["bind_atom_id"], "rule.artifact.capture_plan.bind_atom_id"),
        ref_atom_id=_string(value["ref_atom_id"], "rule.artifact.capture_plan.ref_atom_id"),
        bind_atom_ids=frozenset(
            _list(value["bind_atom_ids"], "rule.artifact.capture_plan.bind_atom_ids")
        ),
        ref_atom_ids=frozenset(
            _list(value["ref_atom_ids"], "rule.artifact.capture_plan.ref_atom_ids")
        ),
    )


def _collect_from_wire(raw: object) -> CollectPlan | None:
    if raw is None:
        return None
    value = _mapping(raw, "rule.artifact.collect_plan")
    _exact_keys(value, _COLLECT_PLAN_FIELDS, "rule.artifact.collect_plan")
    return CollectPlan(
        variable=_string(value["variable"], "rule.artifact.collect_plan.variable"),
        item_type=_string(value["item_type"], "rule.artifact.collect_plan.item_type"),
        source=_required_model_expression(value["source"], "rule.artifact.collect_plan.source"),
        local_predicate=_required_model_expression(
            value["local_predicate"], "rule.artifact.collect_plan.local_predicate"
        ),
        model_predicate=_model_expression_from_wire(
            value["model_predicate"], "rule.artifact.collect_plan.model_predicate"
        ),
        has_external_work=_boolean(
            value["has_external_work"], "rule.artifact.collect_plan.has_external_work"
        ),
    )


def _dfa_from_wire(raw: object, where: str = "rule.artifact.dfa") -> DFA:
    value = _mapping(raw, where)
    _exact_keys(value, _DFA_KEYS, where)
    atoms = [
        _string(atom, "rule.artifact.dfa.atoms[]")
        for atom in _list(value["atoms"], "rule.artifact.dfa.atoms")
    ]
    declared_atoms = set(atoms)
    if len(declared_atoms) != len(atoms):
        raise _fail("rule.artifact.dfa.atoms contains duplicate atom ids")
    states = []
    for index, raw_state in enumerate(_list(value["states"], "rule.artifact.dfa.states")):
        row = _list(raw_state, f"rule.artifact.dfa.states[{index}]")
        if len(row) != 4:
            raise _fail(f"rule.artifact.dfa.states[{index}] must have four values")
        state_id = _integer(row[0], f"rule.artifact.dfa.states[{index}].id")
        if state_id != index:
            raise _fail("DFA state ids must be contiguous and ordered from zero")
        states.append(
            DFAState(
                id=state_id,
                key=_string(row[1], f"rule.artifact.dfa.states[{index}].key"),
                is_accepting=_boolean(row[2], f"rule.artifact.dfa.states[{index}].is_accepting"),
                is_rejecting=_boolean(row[3], f"rule.artifact.dfa.states[{index}].is_rejecting"),
            )
        )
    if not states:
        raise _fail("rule.artifact.dfa.states must not be empty")

    initial_state = _integer(value["initial_state"], "rule.artifact.dfa.initial_state")
    if not 0 <= initial_state < len(states):
        raise _fail("rule.artifact.dfa.initial_state references a missing state")
    node_rows = _list(value["guard_nodes"], "rule.artifact.dfa.guard_nodes")
    normal_roots = _list(value["normal_roots"], "rule.artifact.dfa.normal_roots")
    final_roots = _list(value["final_roots"], "rule.artifact.dfa.final_roots")
    if len(normal_roots) != len(states) or len(final_roots) != len(states):
        raise _fail("DFA guard-root counts must match the state count")

    leaves = [GuardLeaf(target=state.id) for state in states]
    nodes: dict[int, GuardNode] = {}
    visiting: set[int] = set()

    def decode(reference_value: object, name: str) -> Guard:
        reference = _integer(reference_value, name)
        if reference < 0:
            target = -reference - 1
            if not 0 <= target < len(leaves):
                raise _fail(f"{name} targets missing state {target}")
            return leaves[target]
        if not 0 <= reference < len(node_rows):
            raise _fail(f"{name} references missing guard node {reference}")
        if reference in nodes:
            return nodes[reference]
        if reference in visiting:
            raise _fail("DFA guard graph contains a cycle")
        visiting.add(reference)
        row = _list(node_rows[reference], f"rule.artifact.dfa.guard_nodes[{reference}]")
        if len(row) != 3:
            raise _fail(f"rule.artifact.dfa.guard_nodes[{reference}] must have three values")
        atom = _string(row[0], f"rule.artifact.dfa.guard_nodes[{reference}].atom")
        if atom not in declared_atoms:
            raise _fail(
                f"rule.artifact.dfa.guard_nodes[{reference}].atom "
                "must be declared in rule.artifact.dfa.atoms"
            )
        node = GuardNode(
            node_id=reference,
            atom=atom,
            hi=decode(row[1], f"rule.artifact.dfa.guard_nodes[{reference}].hi"),
            lo=decode(row[2], f"rule.artifact.dfa.guard_nodes[{reference}].lo"),
        )
        visiting.remove(reference)
        nodes[reference] = node
        return node

    try:
        normal = {
            index: decode(reference, f"rule.artifact.dfa.normal_roots[{index}]")
            for index, reference in enumerate(normal_roots)
        }
        final = {
            index: decode(reference, f"rule.artifact.dfa.final_roots[{index}]")
            for index, reference in enumerate(final_roots)
        }
    except RecursionError as exc:
        raise _fail("DFA guard graph exceeds the supported nesting depth") from exc
    if len(nodes) != len(node_rows):
        raise _fail("DFA guard table contains unreachable nodes")
    return DFA(
        formula_name=_string(value["formula_name"], "rule.artifact.dfa.formula_name"),
        atoms=atoms,
        states=states,
        initial_state=initial_state,
        normal_guards=normal,
        final_guards=final,
    )


def _parameter_schema_from_wire(raw: object) -> tuple[ParameterSpec, ...]:
    result = []
    for index, item in enumerate(_list(raw, "rule.artifact.parameter_schema")):
        name = f"rule.artifact.parameter_schema[{index}]"
        value = _mapping(item, name)
        _exact_keys(value, _PARAMETER_FIELDS, name)
        parameter_type = _string(value["type"], f"{name}.type")
        binding = _string(value["binding"], f"{name}.binding")
        if parameter_type not in {"number", "text"} or binding not in {"policy", "scan"}:
            raise _fail(f"{name} has an unsupported type or binding")
        result.append(
            ParameterSpec(
                name=_string(value["name"], f"{name}.name"),
                type=parameter_type,  # type: ignore[arg-type]
                binding=binding,  # type: ignore[arg-type]
            )
        )
    if len({spec.name for spec in result}) != len(result):
        raise _fail("parameter schema contains duplicate names")
    return tuple(result)


def _boolean_formula_from_wire(raw: object, name: str) -> T.Formula:
    value = _mapping(raw, name)
    kind = _string(value.get("kind"), f"{name}.kind")
    if kind == "true":
        _exact_keys(value, {"kind"}, name)
        return T.TRUE
    if kind == "false":
        _exact_keys(value, {"kind"}, name)
        return T.FALSE
    if kind == "atom":
        _exact_keys(value, {"kind", "atom_id"}, name)
        return T.Atom(_string(value["atom_id"], f"{name}.atom_id"))
    if kind == "not":
        _exact_keys(value, {"kind", "child"}, name)
        return T.Not(_boolean_formula_from_wire(value["child"], f"{name}.child"))
    if kind in {"and", "or", "implies"}:
        _exact_keys(value, {"kind", "left", "right"}, name)
        left = _boolean_formula_from_wire(value["left"], f"{name}.left")
        right = _boolean_formula_from_wire(value["right"], f"{name}.right")
        return {"and": T.And, "or": T.Or, "implies": T.Implies}[kind](left, right)
    raise _fail(f"{name}.kind {kind!r} is unsupported")


def _tool_quantifiers_from_wire(raw: object) -> dict[str, ToolQuantifierPlan]:
    result: dict[str, ToolQuantifierPlan] = {}
    for index, entry in enumerate(_list(raw, "rule.artifact.tool_quantifiers")):
        name = f"rule.artifact.tool_quantifiers[{index}]"
        row = _list(entry, name)
        if len(row) != 4:
            raise _fail(f"{name} must contain an atom id, kind, Boolean body, and source nodes")
        atom_id = _string(row[0], f"{name}.atom_id")
        if atom_id in result:
            raise _fail(f"{name} duplicates atom id {atom_id!r}")
        kind = _string(row[1], f"{name}.kind")
        if kind not in {"any", "all"}:
            raise _fail(f"{name}.kind must be 'any' or 'all'")
        source_nodes = tuple(
            _integer(source, f"{name}.source_nodes[{source_index}]") if source is not None else None
            for source_index, source in enumerate(_list(row[3], f"{name}.source_nodes"))
        )
        result[atom_id] = ToolQuantifierPlan(
            _boolean_formula_from_wire(row[2], f"{name}.body"),
            "any" if kind == "any" else "all",
            source_nodes,
        )
    return result


def _closed_conditions_from_wire(raw: object) -> dict[str, ClosedPlan]:
    result: dict[str, ClosedPlan] = {}
    for index, entry in enumerate(_list(raw, "rule.artifact.closed_conditions")):
        name = f"rule.artifact.closed_conditions[{index}]"
        row = _list(entry, name)
        if len(row) != 2:
            raise _fail(f"{name} must contain an atom id and an automaton")
        atom_id = _string(row[0], f"{name}.atom_id")
        if atom_id in result:
            raise _fail(f"{name} duplicates atom id {atom_id!r}")
        result[atom_id] = ClosedPlan(_dfa_from_wire(row[1], f"{name}.dfa"))
    return result


def _check_closed_conditions(artifact: CompiledRuleArtifact) -> None:
    """Every closed atom a run reads, in an automaton or a tools body, has the automaton
    that decides it, and no closed part reads its own atom."""
    closed = artifact.closed_conditions
    specs = artifact.atom_specs

    def reads(atoms) -> set[str]:
        """``atoms`` and the atoms of the tools bodies among them."""
        plans = artifact.tool_quantifiers
        return set(atoms).union(*(T.atoms_of(plans[a].body) for a in atoms if a in plans))

    read = reads(artifact.automaton_atoms)
    if set(closed) - read:
        raise _fail("closed condition ids must be read by an automaton")
    if any(specs[atom_id].predicate != "closed" for atom_id in closed):
        raise _fail("closed condition ids must have closed specifications")
    if any(specs[atom_id].predicate == "closed" for atom_id in read - set(closed)):
        raise _fail("closed atoms must have a closed condition")
    done: set[str] = set()

    def visit(atom_id: str, path: tuple[str, ...]) -> None:
        if atom_id in path:
            raise _fail("closed conditions must not read themselves")
        if atom_id in done:
            return
        for inner in reads(closed[atom_id].dfa.atoms) & set(closed):
            visit(inner, (*path, atom_id))
        done.add(atom_id)

    for atom_id in closed:
        visit(atom_id, ())


def _artifact_from_wire(raw: object) -> CompiledRuleArtifact:
    value = _mapping(raw, "rule.artifact")
    _exact_keys(value, _ARTIFACT_KEYS, "rule.artifact")
    compiler_version = _string(value["compiler_version"], "rule.artifact.compiler_version")
    if compiler_version != COMPILER_VERSION:
        raise _fail(
            f"compiler_version {compiler_version!r} is unsupported; expected {COMPILER_VERSION!r}"
        )
    specs = [
        _atom_from_wire(item, index)
        for index, item in enumerate(_list(value["atom_specs"], "rule.artifact.atom_specs"))
    ]
    if len({spec.atom_id for spec in specs}) != len(specs):
        raise _fail("rule.artifact.atom_specs contains duplicate atom ids")
    artifact = CompiledRuleArtifact(
        rule_name=_string(value["rule_name"], "rule.artifact.rule_name"),
        dfa=_dfa_from_wire(value["dfa"]),
        dfa_hash=_sha256(value["dfa_hash"], "rule.artifact.dfa_hash"),
        compiler_version=compiler_version,
        atom_specs={spec.atom_id: spec for spec in specs},
        temporal=_boolean(value["temporal"], "rule.artifact.temporal"),
        notes=tuple(_list(value["notes"], "rule.artifact.notes")),
        capture_plan=_capture_from_wire(value["capture_plan"]),
        parameter_schema=_parameter_schema_from_wire(value["parameter_schema"]),
        collect_plan=_collect_from_wire(value["collect_plan"]),
        tool_quantifiers=_tool_quantifiers_from_wire(value["tool_quantifiers"]),
        closed_conditions=_closed_conditions_from_wire(value["closed_conditions"]),
        full_trace_proof=_boolean(value["full_trace_proof"], "rule.artifact.full_trace_proof"),
        decision_tree=_decision_tree_from_wire(value["decision_tree"]),
    )
    missing_specs = sorted(artifact.automaton_atoms - set(artifact.atom_specs))
    if missing_specs:
        raise _fail("DFA atoms lack specifications: " + ", ".join(missing_specs))
    _check_capture_plan(artifact)
    tool_body_atoms = {
        atom_id for plan in artifact.tool_quantifiers.values() for atom_id in T.atoms_of(plan.body)
    }
    if set(artifact.tool_quantifiers) - artifact.automaton_atoms:
        raise _fail("tool quantifier ids must be DFA atoms")
    if tool_body_atoms - set(artifact.atom_specs):
        raise _fail("tool quantifier body atoms lack specifications")
    if any(
        artifact.atom_specs[atom_id].predicate != "tool_quantifier"
        for atom_id in artifact.tool_quantifiers
    ):
        raise _fail("tool quantifier ids must have tool_quantifier specifications")
    if any(
        artifact.atom_specs[atom_id].domain is not AtomDomain.STATIC for atom_id in tool_body_atoms
    ):
        raise _fail("tool quantifier body atoms must be static")
    _check_closed_conditions(artifact)
    _check_decision_tree(artifact)
    return artifact


def _check_capture_plan(artifact: CompiledRuleArtifact) -> None:
    """Reject a capture plan that does not describe the artifact's parameterized atoms.

    The executor binds candidates only into the plan's atoms, so a plan naming a missing atom,
    or omitting one, would leave an atom permanently False and a match unreported."""
    plan = artifact.capture_plan
    specs = artifact.atom_specs
    parameterized = {atom_id for atom_id, spec in specs.items() if spec.parameter_names}
    if plan is None:
        if parameterized:
            raise _fail("rule.artifact has parameterized atoms but no capture_plan")
        return
    if not isinstance(plan, (CapturePlan, ElementPlan)):
        raise _fail("rule.artifact.capture_plan has an unknown kind")
    named = plan.all_capture_atom_ids()
    if named != parameterized:
        raise _fail("rule.artifact.capture_plan does not name exactly the parameterized atoms")
    name = "rule.artifact.capture_plan"
    if isinstance(plan, ElementPlan):
        if plan.surface not in surface_names(domain="event"):
            raise _fail(f"{name}.surface must be an event surface")
        at = specs[plan.at_atom_id]
        if (at.predicate, at.selector, at.parameter_names) != (
            "at_candidate",
            plan.surface,
            (plan.var,),
        ):
            raise _fail(f"{name}.at_atom_id must name the candidate atom of {plan.var!r}")
        for atom_id in plan.ref_atom_ids:
            ref = specs[atom_id]
            joined = ref.predicate in ("contains_any", "contained_in")
            if not joined or not ref.parameter_names[0].startswith(f"{plan.var}."):
                raise _fail(f"{name}.ref_atom_ids must name joins to fields of {plan.var!r}")
        if plan.filter_atom_id or plan.filter_name:
            accessor = specs.get(plan.filter_atom_id)
            if accessor is None or (
                accessor.domain,
                accessor.selector,
                accessor.value_expr_key,
                accessor.predicate,
                accessor.patterns,
                accessor.parameter_names,
            ) != (AtomDomain.EVENT, plan.surface, "field:name", "equals", (plan.filter_name,), ()):
                raise _fail(
                    f"{name}.filter_atom_id must name the test name = {plan.filter_name!r} on "
                    f"{plan.surface!r} events"
                )
        for atom_id in T.atoms_of(plan.own_step):
            spec = specs.get(atom_id)
            if (
                spec is None
                or spec.domain is not AtomDomain.EVENT
                or spec.selector != plan.surface
                or spec.parameter_names
                or spec.cost.rank >= Cost.MODEL.rank
            ):
                raise _fail(
                    f"{name}.own_step must read unparameterized {plan.surface!r} event atoms "
                    "below model cost"
                )
        return
    if not plan.bind_atom_ids or not plan.ref_atom_ids:
        raise _fail(f"{name} must name its bind and reference atoms")
    if any(specs[atom_id].parameter_names != (plan.var,) for atom_id in named):
        raise _fail(f"{name} atoms must be parameterized by {plan.var!r}")


def _string_mapping(raw: object, name: str) -> dict[str, str]:
    value = _mapping(raw, name)
    if not all(isinstance(item, str) for item in value.values()):
        raise _fail(f"{name} values must be strings")
    return value


def _runtime_models(
    required: set[str],
    config: QuarryConfig,
    model_inference_backend,
) -> dict[str, str]:
    configured = (
        model_inference_backend.models if model_inference_backend is not None else config.models
    )
    missing = sorted(required - set(configured))
    if missing:
        raise _fail("runtime configuration is missing models: " + ", ".join(missing))
    return {name: _canonical_sha256(asdict(configured[name])) for name in sorted(required)}


def load_compiled_policy(
    data: bytes | bytearray | memoryview,
    *,
    runtime_config: QuarryConfig | None = None,
    model_inference_backend=None,
    llm_rate_limiter=None,
) -> CompiledPolicy:
    """Hydrate one authenticated compiled policy without parsing or compiling source."""

    _check_wire_schemas()
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("compiled policy data must be bytes, bytearray, or memoryview")
    try:
        decoded = orjson.loads(data)
    except orjson.JSONDecodeError as exc:
        raise _fail(f"malformed JSON: {exc}") from exc
    value = _mapping(decoded, "document")
    _exact_keys(value, _TOP_LEVEL_KEYS, "document")
    if value["kind"] != FORMAT_KIND:
        raise _fail(f"kind must be {FORMAT_KIND!r}")
    format_version = _integer(value["format_version"], "format_version")
    if format_version != FORMAT_VERSION:
        raise _fail(f"format_version {format_version} is unsupported; expected {FORMAT_VERSION}")
    config_version = _integer(value["config_version"], "config_version")
    if config_version not in {1, 2}:
        raise _fail("config_version must be 1 or 2")

    rule_value = _mapping(value["rule"], "rule")
    _exact_keys(rule_value, _RULE_KEYS, "rule")
    metadata = _string_mapping(rule_value["metadata"], "rule.metadata")
    try:
        artifact = _artifact_from_wire(rule_value["artifact"])
        requirements = requirements_from_artifact(artifact)
    except RecursionError as exc:
        raise _fail("compiled artifact exceeds the supported nesting depth") from exc
    if (
        any(spec.type == "text" for spec in artifact.parameter_schema)
        or requirements["text_resources"]
    ):
        raise _fail(
            "serialized text parameters and text_ref resources are unsupported; "
            "compile from source with a private evidence key"
        )
    rule = DSLRule(
        name=artifact.rule_name,
        metadata=metadata,
        condition=None,
        strings={},
        parameters=tuple(artifact.parameter_schema),
    )
    compatibility_issue = check_rule_compatibility(rule, source="compiled-policy")
    if compatibility_issue is not None:
        raise RuleCompatibilityError(
            compatibility_issue.running_version,
            [compatibility_issue],
        )

    source_sha256 = _sha256(value["source_sha256"], "source_sha256")
    policy_digest = _sha256(value["policy_digest"], "policy_digest")
    text_resources = {
        name: _sha256(digest, f"text_resources.{name}")
        for name, digest in _mapping(value["text_resources"], "text_resources").items()
    }
    models = {
        name: _sha256(digest, f"models.{name}")
        for name, digest in _mapping(value["models"], "models").items()
    }
    if set(text_resources) != requirements["text_resources"]:
        raise _fail("text resource fingerprints do not match the compiled artifact")
    if set(models) != requirements["llm_models"]:
        raise _fail("model fingerprints do not match the compiled artifact")

    policy_parameters = validate_parameter_values(
        artifact.parameter_schema,
        _mapping(value["policy_parameters"], "policy_parameters"),
        binding="policy",
    )
    config = (
        QuarryConfig(config_version=config_version)
        if runtime_config is None
        else replace(runtime_config, quarry_rules_paths=(), quarry_rule_texts=())
    )
    if config.config_version != config_version:
        raise _fail(
            f"runtime config_version {config.config_version} does not match {config_version}"
        )
    missing_texts = sorted(set(text_resources) - set(config.judge_prompts))
    if missing_texts:
        raise _fail("runtime configuration is missing text resources: " + ", ".join(missing_texts))
    runtime_models = _runtime_models(requirements["llm_models"], config, model_inference_backend)
    mismatched_models = sorted(name for name in models if runtime_models[name] != models[name])
    if mismatched_models:
        raise _fail(
            "runtime model configuration differs from the compiled policy for: "
            + ", ".join(mismatched_models)
        )

    scanner = DSLScanner._from_compiled_policy(
        config,
        rule=rule,
        artifact=artifact,
        resource_requirements=requirements,
        model_inference_backend=model_inference_backend,
        llm_rate_limiter=llm_rate_limiter,
    )
    return CompiledPolicy(
        _rule_name=artifact.rule_name,
        _parameter_schema=tuple(artifact.parameter_schema),
        policy_parameters=MappingProxyType(policy_parameters),
        policy_digest=policy_digest,
        _scanner=scanner,
        _source_sha256=source_sha256,
        _text_resource_hashes=MappingProxyType(text_resources),
        _model_hashes=MappingProxyType(models),
    )
