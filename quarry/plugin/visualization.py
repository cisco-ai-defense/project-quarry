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

"""Versioned, JSON-ready rule visualization contracts.

The plugin exposes the authored rule shape rather than the DFA as its primary
visualization. The compiler adds node-to-atom provenance, the executor supplies one canonical
witness trace, and the plugin only joins that trace back to authored syntax.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, cast

from quarry.engine.contracts import AtomEvaluation, ExecutionAttempt, WitnessTrace
from quarry.types.dsl import (
    AllComprehension,
    AnyComprehension,
    ConditionNode,
    FilterComprehension,
    MapComprehension,
    condition_child_fields,
)
from quarry.types.dsl.render import json_value, render_condition

VISUALIZATION_SCHEMA_VERSION = 1


def _children(node: ConditionNode):
    """Yield executable child fields while preserving dataclass declaration order."""

    for field_name in condition_child_fields(node):
        value = getattr(node, field_name)
        if isinstance(value, ConditionNode):
            yield field_name, value
        elif isinstance(value, list | tuple):
            for index, child in enumerate(value):
                if isinstance(child, ConditionNode):
                    yield f"{field_name}[{index}]", child


def _attributes(node: ConditionNode) -> dict[str, Any]:
    """Return non-child fields as extensible node metadata."""

    child_names = set(condition_child_fields(node))
    result: dict[str, Any] = {}
    for field in fields(cast(Any, node)):
        if field.name in child_names:
            continue
        value = getattr(node, field.name)
        if value is None or isinstance(value, ConditionNode):
            continue
        if isinstance(value, list | tuple) and any(
            isinstance(item, ConditionNode) for item in value
        ):
            continue
        result[field.name] = json_value(value)
    return result


@dataclass(frozen=True, slots=True)
class VisualizationNode:
    """One occurrence in the authored condition AST."""

    id: str
    node_type: str
    label: str
    attributes: dict[str, Any]
    # Atoms owned by this authored occurrence, excluding atoms attributed only through a
    # descendant. Composite nodes normally have no direct atoms.
    atom_ids: tuple[str, ...] = ()
    # All atoms attributable to this occurrence or its descendants. This is useful for
    # summarising a branch without pretending the composite itself was executed as an atom.
    subtree_atom_ids: tuple[str, ...] = ()
    # The one atom whose value is this occurrence's own truth value. Composites, surface
    # gates, and collection filters have none, so their runtime state stays unknown.
    state_atom_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready visualization node shape."""
        return {
            "id": self.id,
            "node_type": self.node_type,
            "label": self.label,
            "attributes": dict(self.attributes),
            "atom_ids": list(self.atom_ids),
            "subtree_atom_ids": list(self.subtree_atom_ids),
            "state_atom_id": self.state_atom_id,
        }


@dataclass(frozen=True, slots=True)
class VisualizationEdge:
    """A typed parent-to-child relationship in the visualization graph."""

    source: str
    target: str
    role: str

    def to_dict(self) -> dict[str, str]:
        """Return the JSON-ready visualization edge shape."""
        return {"source": self.source, "target": self.target, "role": self.role}


@dataclass(frozen=True, slots=True)
class RuleVisualization:
    """Stable source-level graph plus optional compiled summary metadata."""

    rule_name: str
    graph_id: str
    root: str
    nodes: tuple[VisualizationNode, ...]
    edges: tuple[VisualizationEdge, ...]
    compiled: dict[str, Any] | None = None
    representation: str = "source_ast"
    schema_version: int = VISUALIZATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready rule visualization shape."""
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "representation": self.representation,
            "graph_id": self.graph_id,
            "rule_name": self.rule_name,
            "root": self.root,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }
        if self.compiled is not None:
            result["compiled"] = dict(self.compiled)
        return result


@dataclass(frozen=True, slots=True)
class VisualizationBundle:
    """Response-level container supporting multiple loaded rules."""

    graphs: tuple[RuleVisualization, ...]
    schema_version: int = VISUALIZATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready visualization bundle shape."""
        return {
            "schema_version": self.schema_version,
            "graphs": [graph.to_dict() for graph in self.graphs],
        }


_COMPREHENSIONS = (AnyComprehension, AllComprehension, FilterComprehension, MapComprehension)


def _state_atom_id(node: ConditionNode, direct_atoms: set[str], artifact) -> str | None:
    """Name the single atom whose value is this occurrence's own truth value, if any."""

    if not direct_atoms:
        return None
    candidates = direct_atoms
    if isinstance(node, _COMPREHENSIONS):
        # Surface gates and collection filters only restrict the domain. A tools quantifier
        # atom is the one comprehension result the engine evaluates directly.
        candidates = direct_atoms & set(artifact.tool_quantifiers)
    return next(iter(candidates)) if len(candidates) == 1 else None


def build_rule_visualization(rule, artifact=None) -> RuleVisualization:
    """Build a deterministic source-AST graph for one loaded rule."""

    if rule.condition is None:
        raise ValueError(f"Rule {rule.name!r} has no condition")

    nodes: list[VisualizationNode | None] = []
    edges: list[VisualizationEdge] = []
    node_atom_ids = getattr(artifact, "node_atom_ids", {}) if artifact is not None else {}

    def visit(node: ConditionNode) -> tuple[str, set[str]]:
        node_id = f"n{len(nodes)}"
        node_index = len(nodes)
        # Reserve the preorder id before visiting children, then fill the immutable record once
        # descendant provenance is known.
        nodes.append(None)
        child_atoms: set[str] = set()
        for role, child in _children(node):
            child_id, descendant_atoms = visit(child)
            child_atoms.update(descendant_atoms)
            edges.append(VisualizationEdge(source=node_id, target=child_id, role=role))
        subtree_atoms = set(node_atom_ids.get(node_index, ()))
        direct_atoms = subtree_atoms - child_atoms
        nodes[node_index] = VisualizationNode(
            id=node_id,
            node_type=type(node).__name__,
            label=render_condition(node),
            attributes=_attributes(node),
            atom_ids=tuple(sorted(direct_atoms)),
            subtree_atom_ids=tuple(sorted(subtree_atoms)),
            state_atom_id=_state_atom_id(node, direct_atoms, artifact),
        )
        return node_id, subtree_atoms

    root, _ = visit(rule.condition)
    compiled = None
    if artifact is not None:
        compiled = {
            "dfa_hash": artifact.dfa_hash,
            "compiler_version": artifact.compiler_version,
            "num_states": artifact.num_states,
            "atom_ids": sorted(
                {atom_id for atom_ids in artifact.node_atom_ids.values() for atom_id in atom_ids}
            ),
            "temporal": artifact.temporal,
            "notes": list(artifact.notes),
        }
    return RuleVisualization(
        rule_name=rule.name,
        graph_id=f"{rule.name}:{artifact.dfa_hash}" if artifact is not None else rule.name,
        root=root,
        nodes=tuple(node for node in nodes if node is not None),
        edges=tuple(edges),
        compiled=compiled,
    )


def build_visualization_bundle(rules, artifacts=None) -> VisualizationBundle:
    """Build graphs for a scanner's loaded rules in stable rule-name order."""

    artifacts = artifacts or {}
    graphs = tuple(
        build_rule_visualization(rules[name], artifacts.get(name)) for name in sorted(rules)
    )
    return VisualizationBundle(graphs=graphs)


_STATE_PRIORITY = {
    "unknown": 0,
    "skipped": 1,
    "evaluated_false": 2,
    "evaluated_true": 3,
}


def _evaluation_state(evaluations) -> str:
    """Summarize observations without turning a true observation into causal evidence."""

    if any(evaluation.status == "evaluated_true" for evaluation in evaluations):
        return "evaluated_true"
    if any(evaluation.status == "evaluated_false" for evaluation in evaluations):
        return "evaluated_false"
    if any(evaluation.status == "skipped" for evaluation in evaluations):
        return "skipped"
    return "unknown"


def _witness_state(evaluations) -> str:
    """Report the proof's value only when every witness observation agrees."""

    statuses = {evaluation.status for evaluation in evaluations if evaluation.witness}
    return statuses.pop() if len(statuses) == 1 else "unknown"


def _source_targets(graph: dict[str, Any], evaluation: AtomEvaluation) -> tuple[str, ...]:
    """Choose only source occurrences identified by the canonical evaluation."""

    owners = [
        node["id"]
        for node in graph.get("nodes", ())
        if evaluation.atom_id in node.get("atom_ids", ())
    ]
    if evaluation.source_node_index is not None:
        source = f"n{evaluation.source_node_index}"
        if source in owners:
            return (source,)
    if len(owners) <= 1:
        return tuple(owners)
    # A DFA observation of an interned atom cannot distinguish identical authored
    # occurrences. Attribute it only to their nearest shared expression.
    parents = {edge["target"]: edge["source"] for edge in graph.get("edges", ())}

    def lineage(node_id: str) -> list[str]:
        result = [node_id]
        while node_id in parents:
            node_id = parents[node_id]
            result.append(node_id)
        return result

    common = set(lineage(owners[0]))
    for owner in owners[1:]:
        common.intersection_update(lineage(owner))
    return tuple(node for node in lineage(owners[0]) if node in common)[:1]


def _attempt_projection(graph: dict[str, Any], attempt: ExecutionAttempt) -> dict[str, Any]:
    """Project one coherent execution attempt onto the static graph.

    Each attempt is projected independently. In particular, true atoms from different tool
    bindings or capture candidates are never unioned into one apparent path.
    """

    parents = {edge["target"]: edge["source"] for edge in graph.get("edges", ())}
    state_atoms = {node["id"]: node.get("state_atom_id") for node in graph.get("nodes", ())}
    # Values come only from a node's own state atom. Ancestors receive the proof path and
    # positions, because the plugin does not re-evaluate Boolean or temporal composition.
    own: dict[str, list[AtomEvaluation]] = {node_id: [] for node_id in state_atoms}
    subtree: dict[str, list[AtomEvaluation]] = {node_id: [] for node_id in state_atoms}
    for evaluation in attempt.evaluations:
        for target in _source_targets(graph, evaluation):
            if state_atoms.get(target) == evaluation.atom_id:
                own[target].append(evaluation)
            while target in subtree:
                subtree[target].append(evaluation)
                if target not in parents:
                    break
                target = parents[target]
    node_records = []
    for node in graph.get("nodes", ()):
        evaluations = subtree[node["id"]]
        node_records.append(
            {
                "node_id": node["id"],
                "state": _evaluation_state(own[node["id"]]),
                "witness_state": _witness_state(own[node["id"]]),
                "causal": any(evaluation.witness for evaluation in evaluations),
                "positions": sorted({evaluation.position for evaluation in evaluations}),
                "witness_positions": sorted(
                    {evaluation.position for evaluation in evaluations if evaluation.witness}
                ),
            }
        )

    return {
        "attempt_id": attempt.attempt_id,
        "kind": attempt.kind,
        "binding_index": attempt.binding_index,
        "capture_binding_index": attempt.capture_binding_index,
        "tool_binding_index": attempt.tool_binding_index,
        "matched": attempt.matched,
        "selected": attempt.selected,
        "nodes": node_records,
        "tool_attempts": [_attempt_projection(graph, check) for check in attempt.tool_attempts],
    }


def _node_evidence(graph: dict[str, Any], trace: WitnessTrace) -> list[dict[str, Any]]:
    """Return only references joining canonical evaluations to authored syntax nodes."""

    records = []
    phased_attempts = [("original", attempt) for attempt in trace.attempts]
    if trace.proof_attempt is not None:
        phased_attempts.append(("proof", trace.proof_attempt))
    for phase, attempt in phased_attempts:
        for observed_attempt in (attempt, *attempt.tool_attempts):
            for evaluation in observed_attempt.evaluations:
                for node_id in _source_targets(graph, evaluation):
                    records.append(
                        {
                            "node_id": node_id,
                            "phase": phase,
                            "attempt_id": observed_attempt.attempt_id,
                            "atom_id": evaluation.atom_id,
                            "position": evaluation.position,
                        }
                    )
    return records


def build_runtime_attribution(
    bundle: dict[str, Any],
    rule_traces: dict[str, WitnessTrace],
    *,
    matched_rules: set[str] | None = None,
    completion_state: str = "complete",
) -> dict[str, Any]:
    """Join canonical per-rule traces to an existing source graph bundle.

    This function does not independently evaluate Boolean, flow, temporal, or quantifier
    semantics. Verdicts come from the scanner, evidence comes from ``WitnessTrace``, and this
    projection only resolves compiled atom ids to authored node ids.
    """

    matched = matched_rules or set()
    graphs = []
    for graph in bundle.get("graphs", ()):
        rule_name = graph.get("rule_name")
        trace = rule_traces.get(rule_name, WitnessTrace())
        graph_matched = rule_name in matched
        attempts = [_attempt_projection(graph, attempt) for attempt in trace.attempts]
        proof_attempt = (
            _attempt_projection(graph, trace.proof_attempt) if trace.proof_attempt else None
        )
        chosen = trace.selected_attempt
        selected = (
            proof_attempt
            if chosen is not None and chosen is trace.proof_attempt
            else next(
                (
                    projection
                    for attempt, projection in zip(trace.attempts, attempts, strict=True)
                    if attempt is chosen
                ),
                None,
            )
        )
        graphs.append(
            {
                "graph_id": graph.get("graph_id", rule_name),
                "rule_name": rule_name,
                "runtime_attribution": {
                    "status": completion_state,
                    "matched": graph_matched,
                    "trace": trace.to_dict(),
                    "node_evidence": _node_evidence(graph, trace),
                    "attempts": attempts,
                    "proof_attempt": proof_attempt,
                    # Convenience view for the one selected/unambiguous attempt. An ambiguous
                    # multi-binding non-match deliberately has no combined node path.
                    "nodes": selected["nodes"] if selected is not None else [],
                },
            }
        )
    return {
        "schema_version": bundle["schema_version"],
        "graphs": graphs,
    }


def build_batch_runtime_attribution(
    runtime_bundles: list[dict[str, Any]],
    *,
    static_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect ordered runtime overlays and count executions per node.

    Every execution retains its own trace and attempt boundaries. Within one execution a node
    counts once, at the highest state observed across its attempts, and as a witness if any
    attempt marks it causal. Counts are never fed back into an individual execution path.
    """

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    # Seed from compilation so a batch that times out before producing any result still has
    # the same per-rule ``executions``/``aggregate`` envelope as a successful batch.
    for graph in (static_bundle or {}).get("graphs", ()):
        key = (graph.get("graph_id", graph["rule_name"]), graph["rule_name"])
        grouped.setdefault(key, [])
    for execution_index, bundle in enumerate(runtime_bundles):
        for graph in bundle.get("graphs", ()):
            key = (graph.get("graph_id", graph["rule_name"]), graph["rule_name"])
            grouped.setdefault(key, []).append(
                {
                    "execution_index": execution_index,
                    **graph["runtime_attribution"],
                }
            )

    graphs = []
    for (graph_id, rule_name), executions in grouped.items():
        node_counts: dict[str, dict[str, int]] = {}
        for execution in executions:
            states_by_node: dict[str, str] = {}
            causal_by_node: set[str] = set()
            observed_attempts = [*execution["attempts"]]
            if execution.get("proof_attempt") is not None:
                observed_attempts.append(execution["proof_attempt"])
            observed_attempts = [
                check
                for attempt in observed_attempts
                for check in (attempt, *attempt.get("tool_attempts", ()))
            ]
            for attempt in observed_attempts:
                for node in attempt["nodes"]:
                    previous = states_by_node.get(node["node_id"], "unknown")
                    if _STATE_PRIORITY[node["state"]] > _STATE_PRIORITY[previous]:
                        states_by_node[node["node_id"]] = node["state"]
                    if node["causal"]:
                        causal_by_node.add(node["node_id"])
            for node_id in sorted(set(states_by_node) | causal_by_node):
                state = states_by_node.get(node_id, "unknown")
                counts = node_counts.setdefault(
                    node_id,
                    {
                        "evaluated_count": 0,
                        "true_count": 0,
                        "witness_count": 0,
                        "skipped_count": 0,
                    },
                )
                if state in {"evaluated_false", "evaluated_true"}:
                    counts["evaluated_count"] += 1
                if state == "evaluated_true":
                    counts["true_count"] += 1
                if node_id in causal_by_node:
                    counts["witness_count"] += 1
                if state == "skipped":
                    counts["skipped_count"] += 1

        graphs.append(
            {
                "graph_id": graph_id,
                "rule_name": rule_name,
                "executions": executions,
                "aggregate": {
                    "execution_count": len(executions),
                    "matched_count": sum(bool(item["matched"]) for item in executions),
                    "nodes": [
                        {"node_id": node_id, **counts}
                        for node_id, counts in sorted(node_counts.items())
                    ],
                },
            }
        )
    schema_version = (static_bundle or {}).get("schema_version", VISUALIZATION_SCHEMA_VERSION)
    return {"schema_version": schema_version, "graphs": graphs}
