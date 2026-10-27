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

"""JSON reports of a compiled rule's decision tree, of a scan's path through it, and of
many scans' flows through it.

The compiler puts the tree's checks in the compiled artifact and the executor records the
checks a scan grounds (``quarry.engine.decision``). This module reads what each check scores,
measures and calls from its atom, names the steps, and turns a scan's recorded checks into
its path, so a report needs the compiled rule, not its text. A compiled policy and the
plugin build the same report, keyed by the same node IDs. A path names its tree by
``tree_sha256``, and ``decision_flows`` counts paths of one tree.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

from quarry.engine.atoms import AtomDomain, AtomSpec, ModelExpression
from quarry.engine.decision import (
    COLLECT,
    MATCH,
    NO_MATCH,
    artifact_parts,
    boot_leaf,
    reads,
    reads_all,
)
from quarry.engine.matchers import compare
from quarry.engine.symbolic_dfa import GuardNode
from quarry.integrations.resources import requirements_from_specs

if TYPE_CHECKING:
    from quarry.config import InferenceModelConfig
    from quarry.engine.contracts import CompiledRuleArtifact, ScanRecord
    from quarry.scanner import DSLScanner

DECISION_TREE_SCHEMA_VERSION = 1
# Where a flow ends when it does not reach match or no_match: a moved threshold sent it to a
# check its scan never answered, or its scan failed at a check.
UNKNOWN = "unknown"
ERROR = "error"
# Where a model is served does not change what it answers, so it is not part of a node's ID.
_MODEL_LOCATION = {"endpoint", "aws_profile", "region"}
_MIRRORED = {">": "<", ">=": "<=", "<": ">", "<=": ">=", "=": "=", "!=": "!="}
_ORDERED = {">", ">=", "<", "<="}


@dataclass(frozen=True)
class ScorePlan:
    """How a score node's one model-score comparison decides its outcome.

    ``operator`` is oriented with the score on the left (``0.5 < score`` is ``score > 0.5``),
    and ``threshold`` is a number or the name of the parameter that supplies it.
    """

    quantifier: Literal["none", "any"]
    operator: str
    threshold: float | str
    # Whether the score is the comparison's left operand as authored.
    score_on_left: bool


@dataclass(frozen=True)
class NumberPlan:
    """How a number node's size comparison (``len(S) > 5``) decides its outcome, with the
    size on the left."""

    operator: str
    threshold: float


def _bound(expression: ModelExpression) -> float | str | None:
    if expression.kind == "literal":
        value = expression.value
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return float(value)
    if expression.kind == "parameter":
        return str(expression.value)
    return None


def _is_score(expression: ModelExpression) -> bool:
    return expression.kind == "model_call" and str(expression.value).endswith("\x1fscore")


def _comparison(expression: ModelExpression) -> tuple[str, float | str, bool] | None:
    """``(operator, threshold, score_on_left)`` of ``score op bound``, score-left oriented."""

    if expression.kind != "compare" or expression.value not in _MIRRORED:
        return None
    left, right = expression.children
    for score, bound, score_on_left in ((left, right, True), (right, left, False)):
        threshold = _bound(bound)
        if _is_score(score) and threshold is not None:
            operator = str(expression.value)
            return (operator if score_on_left else _MIRRORED[operator]), threshold, score_on_left
    return None


def score_plan(spec: AtomSpec) -> ScorePlan | None:
    """The score plan of a static model-score comparison atom, or None for any other atom.

    An unquantified comparison may use any operator, since its one score decides it. Over
    ``split`` chunks only an ordered operator has one decision score (the highest or
    lowest chunk), so ``any`` with ``=`` or ``!=`` is a check."""

    expression = spec.model_expression
    if expression is None or spec.domain is not AtomDomain.STATIC:
        return None
    comparison = _comparison(expression)
    if comparison is not None:
        return ScorePlan("none", *comparison)
    if expression.kind == "split_any":
        comparison = _comparison(expression.children[1])
        if comparison is not None and comparison[0] in _ORDERED:
            return ScorePlan("any", *comparison)
    return None


def number_plan(spec: AtomSpec) -> NumberPlan | None:
    """The size comparison of a collection-size atom, or None for any other atom."""

    if spec.predicate != "surface_count":
        return None
    assert spec.number is not None  # the compiler gives every size comparison its bound
    return NumberPlan(spec.op, spec.number)


@dataclass(frozen=True)
class _Node:
    """A tree node with what its check scores, measures and calls, read from its atom."""

    atom_id: str | None
    label: str
    source_node: int | None
    models: tuple[str, ...]
    texts: tuple[str, ...]
    score: ScorePlan | None = None
    number: NumberPlan | None = None


@dataclass(frozen=True)
class Step:
    """One step of the tree: a check at one place in the diagram, and where each outcome
    leads. A check reached with two continuations is two steps."""

    key: str
    node: _Node
    on_true: str
    on_false: str


def model_identity(model: InferenceModelConfig) -> str:
    """A model alias's identity for node IDs: its configuration without where it or its
    tokenizer is served."""

    canonical = json.dumps(
        _without_location(asdict(model)), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _without_location(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_location(v) for k, v in value.items() if k not in _MODEL_LOCATION}
    return value


def _served_config(model: InferenceModelConfig) -> str:
    canonical = json.dumps(asdict(model), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class Digests:
    """What a scanner's configuration makes of a rule's models and texts.

    ``model`` is a model alias's identity for node IDs, without where it is served;
    ``served`` is its whole configuration, where it is served included; ``text`` is a text
    resource's content hash."""

    model: Callable[[str], str]
    served: Callable[[str], str]
    text: Callable[[str], str]


def scanner_digests(scanner: DSLScanner, evidence_key: bytes) -> Digests:
    """The model and text-resource digests of a scanner's configuration."""

    backend = scanner.resource_manager.model_inference_backend
    models = backend.models if backend is not None else scanner.config.models
    return Digests(
        model=lambda alias: model_identity(models[alias]),
        served=lambda alias: _served_config(models[alias]),
        text=lambda name: hmac.new(
            evidence_key, scanner.config.judge_prompts[name].encode("utf-8"), hashlib.sha256
        ).hexdigest(),
    )


def _nodes(artifact: CompiledRuleArtifact) -> list[_Node]:
    """The tree's nodes with the plans, models and texts their atoms imply."""

    assert artifact.decision_tree is not None
    parts = artifact_parts(artifact)
    nodes = []
    for node in artifact.decision_tree.nodes:
        read = (
            reads_all(artifact.dfa, parts) if node.atom_id is None else reads(node.atom_id, parts)
        )
        found = requirements_from_specs(
            artifact.atom_specs[atom_id] for atom_id in read if atom_id in artifact.atom_specs
        )
        spec = None if node.atom_id is None else artifact.atom_specs[node.atom_id]
        nodes.append(
            _Node(
                node.atom_id,
                node.label,
                node.source_node,
                tuple(sorted(found["llm_models"])),
                tuple(sorted(found["text_resources"])),
                None if spec is None else score_plan(spec),
                None if spec is None else number_plan(spec),
            )
        )
    return nodes


def _resources_part(node: _Node, digests: Digests) -> list[str]:
    return [
        *(f"model:{alias}={digests.model(alias)}" for alias in node.models),
        *(f"text:{name}={digests.text(name)}" for name in node.texts),
    ]


def node_id(node: _Node, digests: Digests, whole_rule: str) -> str:
    """A node's stable identity: its check's content hash and what its models and texts are.

    ``whole_rule`` is the rule automaton's hash, the identity of a whole-rule node."""

    parts = [node.atom_id or f"rule:{whole_rule}", *_resources_part(node, digests)]
    return "t:" + hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def score_key(node: _Node, specs: Mapping[str, AtomSpec], digests: Digests) -> str | None:
    """A score node's identity without its threshold, which keys its stored scores."""

    if node.score is None or node.atom_id is None:
        return None
    expression = specs[node.atom_id].model_expression
    assert expression is not None
    compare_node = expression.children[1] if node.score.quantifier == "any" else expression
    score = compare_node.children[0 if node.score.score_on_left else 1]
    scored = (
        (expression.children[0].content_key(), score.content_key())
        if node.score.quantifier == "any"
        else (score.content_key(),)
    )
    operator = node.score.operator
    direction = "up" if operator in {">", ">="} else "down" if operator in _ORDERED else "equal"
    parts = [repr(scored), node.score.quantifier, direction, *_resources_part(node, digests)]
    return "s:" + hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def _steps(
    artifact: CompiledRuleArtifact, nodes: list[_Node], ids: Mapping[str | None, str]
) -> tuple[str, dict[str, Step]]:
    """The tree's root step key and its steps, from the root automaton's boot guard.

    A step's key is its node's ID, or ``<id>~<n>`` for the node's later steps in
    breadth-first order. With a ``collect:`` block, a match passes through a ``collect``
    step first.
    """

    decomposed = nodes[0].atom_id is not None
    # A whole-rule node includes its collect gate; a decomposed match passes through it.
    match = COLLECT if artifact.collect_plan is not None and decomposed else MATCH
    steps: dict[str, Step] = {}
    if not decomposed:
        (node,) = nodes
        steps[ids[None]] = Step(ids[None], node, match, NO_MATCH)
    else:
        dfa = artifact.dfa
        by_atom = {node.atom_id: node for node in nodes}
        by_guard: dict[int, str] = {}
        uses: dict[str, int] = {}

        def key(guard) -> str:
            if not isinstance(guard, GuardNode):
                return match if boot_leaf(dfa, guard.target) == MATCH else NO_MATCH
            if guard.node_id not in by_guard:
                count = uses[guard.atom] = uses.get(guard.atom, -1) + 1
                base = ids[guard.atom]
                by_guard[guard.node_id] = base if count == 0 else f"{base}~{count}"
            return by_guard[guard.node_id]

        root = dfa.normal_guards[dfa.initial_state]
        if not isinstance(root, GuardNode):
            raise ValueError("a decomposed decision tree's root guard tests no check")
        queue = [root]
        while queue:
            guard = queue.pop(0)
            step_key = key(guard)
            if step_key in steps:
                continue
            steps[step_key] = Step(step_key, by_atom[guard.atom], key(guard.hi), key(guard.lo))
            queue.extend(branch for branch in (guard.hi, guard.lo) if isinstance(branch, GuardNode))
    if match == COLLECT:
        steps[COLLECT] = Step(COLLECT, _Node(None, "collect:", None, (), ()), MATCH, NO_MATCH)
    return next(iter(steps)), steps


@dataclass(frozen=True)
class ReportTree:
    """A compiled rule's tree under one configuration: its steps keyed by node ID, its
    report, and the configuration hash its scans' paths carry. Build it once per compiled
    rule and configuration, not per scan."""

    root: str
    steps: dict[str, Step]
    ids: dict[str | None, str]
    report: dict[str, Any]
    config_sha256: str


def report_tree(artifact: CompiledRuleArtifact, digests: Digests) -> ReportTree:
    """The rule's tree with node IDs from its checks and what their models and texts are."""

    nodes = _nodes(artifact)
    ids = {node.atom_id: node_id(node, digests, artifact.dfa_hash) for node in nodes}
    root, steps = _steps(artifact, nodes, ids)
    entries: dict[str, Any] = {}
    for key, step in steps.items():
        node = step.node
        entry: dict[str, Any] = {
            "id": key.split("~", 1)[0],
            "kind": (
                "collect"
                if key == COLLECT
                else "score"
                if node.score
                else "number"
                if node.number
                else "check"
            ),
            "label": node.label,
            "source_nodes": [] if node.source_node is None else [f"n{node.source_node}"],
        }
        if node.score is not None:
            threshold = node.score.threshold
            entry["score_form"] = {
                "quantifier": node.score.quantifier,
                "operator": node.score.operator,
                "threshold": {"parameter": threshold} if isinstance(threshold, str) else threshold,
            }
            entry["score_key"] = score_key(node, artifact.atom_specs, digests)
        if node.number is not None:
            entry["number_form"] = {
                "operator": node.number.operator,
                "threshold": node.number.threshold,
            }
        entry["true"] = step.on_true
        entry["false"] = step.on_false
        entries[key] = entry
    report = {"schema_version": DECISION_TREE_SCHEMA_VERSION, "root": root, "nodes": entries}
    served = sorted(
        {f"model:{alias}={digests.served(alias)}" for node in nodes for alias in node.models}
        | {f"text:{name}={digests.text(name)}" for node in nodes for name in node.texts}
    )
    return ReportTree(
        root,
        steps,
        ids,
        {**report, "tree_sha256": _tree_sha256(report)},
        hashlib.sha256("\x00".join(served).encode()).hexdigest(),
    )


def tree_report(artifact: CompiledRuleArtifact, digests: Digests) -> dict[str, Any]:
    """One rule's decision tree: its steps, each step's check and edges, and its root.

    ``tree_sha256`` hashes the rest of the report; every path of a scan of this tree
    carries it, so a path maps onto its tree by one comparison."""

    return report_tree(artifact, digests).report


def _tree_sha256(report: Mapping[str, Any]) -> str:
    """The hash of a tree report's content, without its own ``tree_sha256``."""

    content = {key: value for key, value in report.items() if key != "tree_sha256"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def conversation_sha256(conversation: Mapping[str, object]) -> str:
    """Hash a scanned conversation as JSON with sorted keys and no whitespace."""

    canonical = json.dumps(conversation, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def sample_value(sha256_hex: str) -> float:
    """Map a conversation hash to a uniform value in ``[0, 1)``: exploration selects a
    conversation when this is below its rate, so a larger rate keeps every one selected."""

    return int(sha256_hex[:16], 16) / 2**64


def explores(conversation_digest: str, rate: float) -> bool:
    """Whether a scan at exploration ``rate`` explores the conversation with this hash."""

    return sample_value(conversation_digest) < rate


def _chunk(chunk: Mapping[str, object], *, with_value: bool = True) -> dict[str, object]:
    entry: dict[str, object] = {"chunk": chunk["ordinal"]}
    if with_value:
        entry["value"] = chunk["value"]
    entry["span"] = chunk["span"]
    entry["context"] = chunk["context"]
    return entry


def score_entry(plan: ScorePlan, evidence, *, chunks: bool) -> dict[str, object]:
    """A score node's decision score from the evidence of its model comparison.

    The decision score is the number whose comparison with any threshold gives the node's
    outcome: the score, or for ``any`` over chunks the highest (``>``, ``>=``) or lowest
    (``<``, ``<=``) chunk score. It is null, from no evaluations, when the input was missing
    or a split had no chunks. ``chunks`` lists every chunk, not only the deciding one.
    """

    if plan.quantifier == "any":
        summary = evidence.score_summary
        count = summary["count"] if summary else 0
        entry: dict[str, object] = {
            "value": (
                None
                if not count
                else summary["maximum"]
                if plan.operator in {">", ">="}
                else summary["minimum"]
            ),
            "operator": plan.operator,
            "threshold": float(evidence.threshold),
            "evaluations": count,
        }
        if count and "chunks" in summary:
            entry["matching"] = summary["matching_count"]
            entry["deciding"] = _chunk(summary["deciding"], with_value=False)
            if chunks:
                entry["chunks"] = [_chunk(chunk) for chunk in summary["chunks"]]
        return entry
    # A comparison records the score as the value and the bound as the threshold.
    value = evidence.expression_value
    entry = {
        "value": None if value is None else float(value),
        "operator": plan.operator,
        "threshold": float(evidence.threshold),
        "evaluations": 0 if value is None else 1,
    }
    if evidence.input_span is not None:
        entry["span"] = evidence.input_span
    return entry


def _entry(key: str, step: Step, value: bool | None, evidence, *, chunks: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {"node": key, "outcome": value}
    node = step.node
    if node.score is not None and evidence is not None:
        entry["score"] = score_entry(node.score, evidence, chunks=chunks)
    elif node.number is not None and evidence is not None:
        # The size the check compared, and the bound it compared it with.
        entry["number"] = {
            "value": evidence,
            "operator": node.number.operator,
            "threshold": node.number.threshold,
        }
    return entry


def path_report(
    tree: ReportTree,
    recorder: ScanRecord | None,
    *,
    matched: bool | None,
    conversation: dict[str, Any],
    parameter_digest: str,
    explain: bool,
    explore_rate: float,
) -> dict[str, Any]:
    """A scan result's ``decision_path``: the steps it reached and the leaf it ended at.

    Args:
        tree: The scanned rule's tree under the scan's configuration.
        recorder: The scan's observed checks; None for a rule with one whole-rule node.
        matched: The verdict, or None when the scan did not complete.
        conversation: The scanned conversation, whose hash selects exploration.
        parameter_digest: The digest of the parameter values the scan used.
        explain: Whether score entries list every chunk.
        explore_rate: The exploration rate the scan used.
    """

    path: list[dict[str, Any]] = []
    off_path: list[dict[str, Any]] = []
    # Checks exploration could not ground because their model failed.
    failed: list[str] = []
    if recorder is None:
        if matched is not None:
            path.append({"node": tree.root, "outcome": matched})
    else:
        key = tree.root
        by_atom = {step.node.atom_id: step for step in tree.steps.values()}
        for check in recorder.checks:
            if check.origin == "explored":
                if check.value is None:
                    failed.append(tree.ids[check.atom_id])
                    continue
                step = by_atom[check.atom_id]
                off_path.append(
                    _entry(tree.ids[check.atom_id], step, check.value, check.evidence, chunks=True)
                )
                continue
            step = tree.steps[key]
            path.append(_entry(key, step, check.value, check.evidence, chunks=explain))
            key = step.on_true if check.value else step.on_false
        if key == COLLECT and matched is not None:
            # The condition held; the collect block decides whether anything matched.
            path.append({"node": COLLECT, "outcome": matched})
    digest = conversation_sha256(conversation)
    report: dict[str, Any] = {
        "tree_sha256": tree.report["tree_sha256"],
        "config_sha256": tree.config_sha256,
        "parameter_digest": parameter_digest,
        "conversation_sha256": digest,
        "leaf": None if matched is None else MATCH if matched else NO_MATCH,
        "path": path,
    }
    if explore_rate > 0:
        selected = explores(digest, explore_rate)
        report["exploration"] = {
            "rate": explore_rate,
            "sample_value": sample_value(digest),
            "selected": selected,
            **({"off_path": off_path, "failed": failed} if selected else {}),
        }
    return report


@dataclass(frozen=True)
class _Answer:
    outcome: bool | None
    # A score or number node's decision value, which a moved threshold compares again.
    value: float | None


def _answers(path: Mapping[str, Any]) -> dict[str, _Answer]:
    """The checks a path answered, by node ID: its steps and the checks exploration scored."""

    entries = [*path["path"], *path.get("exploration", {}).get("off_path", ())]
    answers: dict[str, _Answer] = {}
    for entry in entries:
        measured = entry.get("score") or entry.get("number")
        value = None if measured is None else measured["value"]
        answers[entry["node"].split("~", 1)[0]] = _Answer(entry["outcome"], value)
    return answers


def _applied_thresholds(
    tree: Mapping[str, Any], thresholds: Mapping[str, float]
) -> dict[str, float]:
    """Each score or number node's moved threshold, by node ID. A key names a node ID, or
    a parameter, which moves every node whose threshold is that parameter."""

    applied: dict[str, float] = {}
    used: set[str] = set()
    for node in tree["nodes"].values():
        form = node.get("score_form") or node.get("number_form")
        if form is None:
            continue
        bound = form["threshold"]
        parameter = bound["parameter"] if isinstance(bound, dict) else None
        for key in (node["id"], parameter):
            if key is not None and key in thresholds:
                applied[node["id"]] = float(thresholds[key])
                used.add(key)
                break
    unknown = sorted(set(thresholds) - used)
    if unknown:
        raise ValueError(
            f"thresholds name no score or number node or parameter of the tree: {unknown}"
        )
    return applied


def decision_flows(
    tree: Mapping[str, Any],
    paths: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Count scans through a tree's steps and leaves, optionally at moved thresholds.

    Each path is walked from the tree's root over the checks its scan answered. A score or
    number node with a moved threshold compares the path's recorded value again, so a
    threshold can move without scanning again. A walk that reaches a check its scan never
    answered ends at ``unknown`` when a moved threshold sent it there, and at ``error``
    otherwise, since only a failed check leaves a scan's own route unanswered.

    Args:
        tree: A ``tree_report``.
        paths: The ``decision_path`` of each scan of that tree.
        labels: One label per path, such as its ground truth; without them every path
            counts under ``all``.
        thresholds: New thresholds by node ID, or by parameter name for every node whose
            threshold is that parameter.

    Returns:
        ``tree_sha256``; the ``thresholds`` applied, by node ID; for each step reached, the
        scans by outcome (``true``, ``false``) and label; and the scans at each leaf by
        label.

    Raises:
        ValueError: A tree whose ``tree_sha256`` is not its content's hash or whose edges
            loop, a path of another tree, a label count that is not the path count, or a
            threshold that names no score or number node or parameter.
    """

    if _tree_sha256(tree) != tree["tree_sha256"]:
        raise ValueError("the tree's tree_sha256 is not the hash of its content")
    if labels is not None and len(labels) != len(paths):
        raise ValueError(f"{len(labels)} labels for {len(paths)} paths")
    applied = _applied_thresholds(tree, thresholds or {})
    nodes = tree["nodes"]
    steps: dict[str, dict[str, Counter[str]]] = {}
    leaves: dict[str, Counter[str]] = {
        leaf: Counter() for leaf in (MATCH, NO_MATCH, UNKNOWN, ERROR)
    }
    for index, path in enumerate(paths):
        if path["tree_sha256"] != tree["tree_sha256"]:
            raise ValueError(
                f"path {index} walked tree {path['tree_sha256']}, not {tree['tree_sha256']}"
            )
        label = "all" if labels is None else labels[index]
        answers = _answers(path)
        key, rerouted, walked = tree["root"], False, 0
        while key not in (MATCH, NO_MATCH):
            # A tree is a diagram without cycles, so a walk visits each step at most once.
            walked += 1
            if walked > len(nodes):
                raise ValueError(f"the tree's edges loop at step {key}")
            node = nodes[key]
            answer = answers.get(node["id"])
            outcome = None if answer is None else answer.outcome
            if answer is not None and node["id"] in applied and answer.value is not None:
                form = node.get("score_form") or node["number_form"]
                moved = compare(answer.value, form["operator"], applied[node["id"]])
                rerouted = rerouted or moved != outcome
                outcome = moved
            if outcome is None:
                key = UNKNOWN if answer is None and rerouted else ERROR
                break
            counts = steps.setdefault(key, {"true": Counter(), "false": Counter()})
            counts["true" if outcome else "false"][label] += 1
            key = node["true" if outcome else "false"]
        leaves[key][label] += 1
    return {
        "tree_sha256": tree["tree_sha256"],
        "thresholds": applied,
        "steps": {
            key: {outcome: dict(counter) for outcome, counter in counts.items()}
            for key, counts in steps.items()
        },
        "leaves": {leaf: dict(counter) for leaf, counter in leaves.items()},
    }


def _telemetry_step(entry: Mapping[str, Any], position: Mapping[str, int]) -> dict[str, Any]:
    step: dict[str, Any] = {"node": position[entry["node"]], "outcome": entry["outcome"]}
    measured = entry.get("score") or entry.get("number")
    if measured is not None:
        step["value"] = measured["value"]
        step["threshold"] = measured["threshold"]
        # The span the score came from: a split's deciding chunk, or the scored input.
        span = measured["deciding"]["span"] if "deciding" in measured else measured.get("span")
        if span is not None:
            step["span"] = span
    return step


def telemetry_record(tree: Mapping[str, Any], path: Mapping[str, Any]) -> dict[str, Any]:
    """A scan's decision path cut to what telemetry keeps.

    The record keeps the path's hashes and leaf, and for each step the node's position
    among the tree report's step keys in sorted order (the tree's ``tree_sha256`` pins which
    tree, and sorting makes the position the same in every copy of it), its outcome,
    and a score or number node's value and threshold. A score step keeps one span, the
    deciding chunk of a ``split`` or the scored input, and no chunk list, so the record's
    size depends on the tree, not on the conversation. A sampled scan adds the checks
    exploration ran (``explored``) and those whose model failed (``failed``).

    Raises:
        ValueError: A path of another tree.
    """

    if path["tree_sha256"] != tree["tree_sha256"]:
        raise ValueError(f"the path walked tree {path['tree_sha256']}, not {tree['tree_sha256']}")
    # The hash is over sorted keys, so a copy of the tree may list its nodes in any order;
    # positions in sorted key order mean the same step in every copy with this hash.
    position = {key: index for index, key in enumerate(sorted(tree["nodes"]))}
    record: dict[str, Any] = {
        key: path[key]
        for key in (
            "tree_sha256",
            "config_sha256",
            "parameter_digest",
            "conversation_sha256",
            "leaf",
        )
    }
    record["steps"] = [_telemetry_step(entry, position) for entry in path["path"]]
    exploration = path.get("exploration")
    if exploration is not None and exploration["selected"]:
        record["explored"] = [_telemetry_step(entry, position) for entry in exploration["off_path"]]
        record["failed"] = [position[node] for node in exploration["failed"]]
    return record
