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

"""Witness / evidence extraction (witness-native explain).

When a rule matches, the executor records selected timeline evidence. For existential
paths, ``minimize`` can greedily remove positions while preserving a replayed match.
Universal and absence claims keep the original domain: deleting clean elements would
change what the claim is about. Atom proof membership describes the selected execution
path and does not assert that no alternative proof exists.

This module owns two layers:

  * **pure** — ``minimize`` / ``redact`` — positions + an opaque replay
    callback, no Quarry data imports.
  * **evidence assembly** — ``build_evidence`` renders one rule's scan record as
    :class:`RuleEvidence`, for a match and a non-match alike: what each atom came to
    (``AtomEvidence``) and, for a match, its support, each witness position resolved back
    to its :class:`~quarry.engine.events.ConversationEvent` (surface path, kind, redacted
    content excerpt), with the winning ``... as v`` capture value.

Adapter tier: it may read Quarry *data* (event content for excerpts) but never the AST,
scanner, or DSL layer.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from quarry.engine.atoms import AtomDomain, is_per_tool_static
from quarry.engine.decision import artifact_parts, reads, reads_all
from quarry.engine.matchers import field_text

if TYPE_CHECKING:  # only for typing; no runtime import cycle
    from quarry.engine.contracts import ScanRecord, WitnessTrace
    from quarry.engine.events import ConversationEvent

# Reusable secret patterns for redaction in formatted evidence (not exhaustive; the
# capture layer marks sensitive spans, this is a formatting safety net).
_SECRET_PATTERNS = (
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),  # AWS key id
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),  # JWT
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b[0-9a-fA-F]{32,}\b"),  # long hex secret/hash
)


@dataclass(frozen=True, slots=True)
class Sha256Evidence:
    """A digest produced by Quarry that may be emitted without generic hex redaction."""

    digest: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.digest) is None:
            raise ValueError("SHA-256 evidence must be 64 lowercase hexadecimal characters")


def minimize(
    positions: list[int],
    reproduces: Callable[[list[int]], bool],
) -> list[int]:
    """Greedily reduce ``positions`` to a minimal subset that still ``reproduces`` the
    match: drop any position whose removal keeps the match, until none is removable.

    ``reproduces(subset) -> bool`` replays the rule restricted to those timeline
    positions and returns whether it still matches. Sound (the result still matches) and
    minimal (every remaining position is load-bearing).

    Each pass tries every position once and keeps going after a removal; passes repeat
    until one removes nothing, so a position kept earlier is re-checked against the smaller
    set (removal is not monotone for bounded flows).
    """
    cur = list(positions)
    changed = True
    while changed and len(cur) > 1:
        changed = False
        for pos in list(cur):
            if len(cur) == 1:
                break
            trial = [p for p in cur if p != pos]
            if reproduces(trial):
                cur = trial
                changed = True
    return cur


def redact(text: str) -> str:
    """Mask likely secrets in ``text`` for user-facing evidence, preserving structure."""
    out = text
    for pat in _SECRET_PATTERNS:
        out = pat.sub("«redacted»", out)
    return out


_EXCERPT_LIMIT = 160
_MODEL_RESULTS_LIMIT = 16
_WITNESS_STEPS_LIMIT = 16
_MODEL_ALIASES_LIMIT = 8
_MODEL_ARGUMENTS_LIMIT = 8
# Explanation bounds: atoms reported per rule, positions and excerpts per atom,
# and patterns rendered in one atom's description.
_MAX_OBSERVED_ATOMS = 24
_MAX_POSITIONS_PER_ATOM = 8
_MAX_EXCERPTS_PER_ATOM = 2
_MAX_DESCRIBED_PATTERNS = 8


def _excerpt(text: str | None, limit: int = _EXCERPT_LIMIT) -> str | None:
    """One-line, secret-redacted, length-capped excerpt of an event's content."""
    if text is None:
        return None
    flat = redact(str(text)).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _model_scalar(value: Any) -> str | int | float | bool | None:
    """Return one redacted scalar suitable for bounded model evidence."""

    if isinstance(value, Sha256Evidence):
        return f"sha256:{value.digest}"
    if value is None or isinstance(value, bool | float):
        return value
    if isinstance(value, int):
        # orjson accepts the full 64-bit JSON integer range. Preserve ordinary
        # numbers, but render larger DSL literals as bounded text so explain
        # output cannot make an otherwise valid scan unserializable.
        if -(2**63) <= value <= 2**64 - 1:
            return value
        return _excerpt(str(value))
    return _excerpt(str(value))


@dataclass(frozen=True)
class ModelResultEvidence:
    """One bounded model-derived boolean atom retained for explain output."""

    operation: str
    models: tuple[str, ...]
    arguments: tuple[tuple[str, str | int | float | bool | None], ...]
    expression_value: str | int | float | bool | None
    operator: str | None
    threshold: str | int | float | bool | None
    matched: bool
    source_path: str | None = None
    score_summary: dict[str, Any] | None = None
    omitted_models: int = 0
    omitted_arguments: int = 0
    # The span of the transcript the score read, or ``{"constructed": True}`` for a
    # constructed input (see ``model_expression._span_of``) — with a ``"pieces"`` list when
    # it read conversation fields, each only a position and a field, never text: this
    # record is copied into a path's score entry and into telemetry, both text-free.
    input_span: dict[str, Any] | None = None
    # Each refused call the atom read, as (model alias, operation, reason); never model text.
    refusals: tuple[tuple[str, str, str], ...] = ()

    @classmethod
    def bounded(
        cls,
        *,
        operation: str,
        models,
        arguments,
        expression_value,
        operator,
        threshold,
        matched: bool,
        source_path: str | None,
        score_summary: dict[str, Any] | None = None,
        input_span: dict[str, Any] | None = None,
        refusals=(),
    ) -> ModelResultEvidence:
        """Redact and cap caller-controlled strings, aliases, and arguments."""

        aliases = tuple(dict.fromkeys(_excerpt(str(model)) or "" for model in models))
        projected = tuple(
            (_excerpt(str(name)) or "", _model_scalar(value)) for name, value in arguments
        )
        return cls(
            operation=_excerpt(operation) or "",
            models=aliases[:_MODEL_ALIASES_LIMIT],
            omitted_models=max(0, len(aliases) - _MODEL_ALIASES_LIMIT),
            arguments=projected[:_MODEL_ARGUMENTS_LIMIT],
            omitted_arguments=max(0, len(projected) - _MODEL_ARGUMENTS_LIMIT),
            expression_value=_model_scalar(expression_value),
            score_summary=score_summary,
            operator=_excerpt(operator) if operator is not None else None,
            threshold=_model_scalar(threshold),
            matched=matched,
            source_path=_excerpt(source_path) if source_path is not None else None,
            input_span=input_span,
            refusals=tuple(
                (_excerpt(str(model)) or "", operation_name, reason)
                for model, operation_name, reason in refusals
            )[:_MODEL_ALIASES_LIMIT],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-ready model evidence shape."""

        return {
            "operation": self.operation,
            "models": list(self.models),
            "omitted_models": self.omitted_models,
            "arguments": [{"name": name, "value": value} for name, value in self.arguments],
            "omitted_arguments": self.omitted_arguments,
            "expression_value": self.expression_value,
            "score_summary": self.score_summary,
            "operator": self.operator,
            "threshold": self.threshold,
            "matched": self.matched,
            "source_path": self.source_path,
            "input_span": self.input_span,
            "refusals": [
                {"model": model, "operation": operation, "reason": reason}
                for model, operation, reason in self.refusals
            ],
        }


@dataclass(frozen=True)
class EvidenceStep:
    """One load-bearing timeline turn in a match's witness.

    A witness position resolved back to its conversation event: where it is
    (``position`` on the shared timeline, ``source_path`` like ``tool_responses[0]``),
    what kind of turn it is, and a redacted excerpt of its content.
    """

    position: int
    source_path: str
    kind: str  # "message" | "tool_call" | "tool_response"
    role: str | None = None
    tool_name: str | None = None
    excerpt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-ready evidence step shape."""
        d: dict[str, Any] = {
            "position": self.position,
            "path": self.source_path,
            "kind": self.kind,
        }
        if self.role is not None:
            d["role"] = self.role
        if self.tool_name is not None:
            d["tool_name"] = self.tool_name
        if self.excerpt is not None:
            d["excerpt"] = self.excerpt
        return d


# A rule is explained atom by atom: the question each atom asks, and what its record says
# became of it.


def _atom_subject(spec) -> str:
    """The thing an atom asks about, e.g. ``tool_calls.arguments`` or ``system_prompt``.

    ``value_expr_key`` is a canonical key, not source text: ``field:content`` for a field
    access, ``field:`` or ``""`` when the atom tests the surface's own text.
    """
    key = spec.value_expr_key or ""
    field = key.split(":", 1)[1] if key.startswith("field:") else key
    surface = _windowed_surface(spec)
    return f"{surface}.{field}" if field else surface


def _windowed_surface(spec) -> str:
    """The atom's surface with its index or slice window, e.g. ``user_messages[5:]``."""
    if getattr(spec, "is_slice", False):
        start = "" if spec.slice_start is None else spec.slice_start
        stop = "" if spec.slice_stop is None else spec.slice_stop
        return f"{spec.selector}[{start}:{stop}]"
    if getattr(spec, "ordinal", None) is not None:
        return f"{spec.selector}[{spec.ordinal}]"
    return spec.selector


def _describe_operand(spec) -> str:
    """Render an atom's operands: a capture binding, or a capped pattern list."""
    if spec.parameter_names:
        return f"<captured {', '.join(spec.parameter_names)}>"
    shown = [_excerpt(p, 60) or "" for p in spec.patterns[:_MAX_DESCRIBED_PATTERNS]]
    omitted = len(spec.patterns) - len(shown)
    rendered = ", ".join(f'"{p}"' for p in shown)
    return f"[{rendered}, +{omitted} more]" if omitted > 0 else f"[{rendered}]"


def _describe_semantics(spec) -> str:
    """Trailing note for string modifiers that are on, e.g. ``(nocase, base64)``.

    Only non-defaults are shown, so their absence is itself information: an author who sees
    no ``nocase`` on a failed ``contains_any`` is looking at a case-sensitive test.
    """
    semantics = spec.string_semantics
    flags = []
    if semantics.case_insensitive:
        flags.append("nocase")
    if semantics.fullword:
        flags.append("fullword")
    if semantics.base64:
        flags.append("base64")
    if semantics.decode:
        flags.append(f"decode={semantics.decode}")
    if semantics.normalize:
        flags.append(f"normalize={semantics.normalize}")
    if semantics.list_mode == "all":
        flags.append("all patterns must match")
    return f" ({', '.join(flags)})" if flags else ""


def _describe_number(value: float | None) -> str:
    """Render a folded numeric operand without a spurious ``.0`` on whole numbers."""
    if value is None:
        return ""
    return str(int(value)) if float(value).is_integer() else str(value)


def describe_atom(spec) -> str:
    """Render an :class:`~quarry.engine.atoms.AtomSpec` as a readable question.

    Not the author's source text: atom ids are content-derived and binder-agnostic, so a
    rule's ``any resp in tool_responses`` binder is not in the compiled artifact. The surface
    name stands in for it, and ``AtomEvidence.source_nodes`` points at the authored node.
    """
    subject = _atom_subject(spec)
    predicate = spec.predicate
    semantics = _describe_semantics(spec)
    number = _describe_number(spec.number)

    if spec.model_expression is not None or predicate in ("external", "model_scan", "model_score"):
        detector = spec.detector or "external"
        argument = f'"{spec.detector_arg}", ' if spec.detector_arg else ""
        call = f"{detector}({argument}{subject})"
        return f"{call} >= {spec.threshold}" if spec.threshold is not None else call
    if predicate == "on_surface":
        return f"event in {_windowed_surface(spec)}"
    if predicate == "surface_count":
        return f"count({spec.selector}) {spec.op} {number}"
    if predicate == "is_null":
        return f"{subject} is null"
    if predicate == "tool_names_all":
        return f"every tool name in {_describe_operand(spec)}"
    if predicate == "regex":
        return f"{subject} ~ {_describe_operand(spec)}{semantics}"
    if predicate == "equals":
        # An equality carries one operand in practice (an alternation compiles to separate
        # atoms), and ``= ["user"]`` reads as a list comparison the author never wrote.
        if len(spec.patterns) == 1 and not spec.parameter_names:
            return f'{subject} = "{_excerpt(spec.patterns[0], 60) or ""}"{semantics}'
        return f"{subject} in {_describe_operand(spec)}{semantics}"
    if predicate == "length":
        return f"length({subject}) {spec.op} {number}"
    if predicate == "count":
        return f"count of {_describe_operand(spec)} in {subject} {spec.op} {number}{semantics}"
    if predicate in ("position", "position_abs", "position_cross"):
        operands = " vs ".join(
            f'"{_excerpt(p, 40)}"' for p in spec.patterns[:_MAX_DESCRIBED_PATTERNS]
        )
        bound = f" {spec.op} {number}" if number else f" {spec.op}"
        return f"match position in {subject}: {operands}{bound}{semantics}"
    return f"{subject} {predicate} {_describe_operand(spec)}{semantics}"


def _case_hint(spec, texts: list[str]) -> str | None:
    """A hint when this atom's only problem is that it is case-sensitive.

    Restricted to plain ``contains_any`` with no regex, fullword, decode or normalize
    transform: the subset where ``pattern.lower() in text.lower()`` is exactly ``nocase``,
    not an approximation of it. Outside that subset no hint is offered rather than a guessed
    one; the excerpt still carries the evidence.
    """
    semantics = spec.string_semantics
    if spec.predicate != "contains_any" or not spec.patterns or spec.parameter_names:
        return None
    if (
        semantics.case_insensitive
        or semantics.is_regex
        or semantics.fullword
        or semantics.base64
        or semantics.decode
        or semantics.normalize
    ):
        return None
    lowered = [text.lower() for text in texts if text]
    hits = [
        pattern
        for pattern in spec.patterns
        if pattern and any(pattern.lower() in text for text in lowered)
    ]
    if not hits:
        return None
    quoted = ", ".join(f'"{_excerpt(p, 60)}"' for p in hits[:_MAX_DESCRIBED_PATTERNS])
    return (
        f"would match if case-insensitive: {quoted} is present but differs in case. "
        "Quarry string matching is case-sensitive by default -- add nocase"
    )


@dataclass(frozen=True)
class AtomEvidence:
    """What one of a rule's atoms came to over the scan, read from its record.

    Four statuses, and the three without a true position call for different fixes:

      * ``satisfied``     — true somewhere.
      * ``false``         — computed on every candidate and false on each. Fix the predicate.
      * ``no_candidates`` — nothing on the atom's surface, so the predicate never ran. Fix
        the surface or the input, not the predicate.
      * ``unknown``       — not true anywhere, and not computed on some candidate; ``reason``
        says why: ``budget_exhausted``, ``model_failed``, ``captured_value`` (it tests a
        captured value, which has none outside a binding) or ``not_computed``.

    ``explored`` marks an atom with a value the verdict did not need, computed only to report
    it. ``candidate_events`` is how many events (or tools) the atom's surface holds;
    positions are reported only for those. Excerpts quote the field the atom tests.
    ``source_nodes`` are the authored condition's preorder indexes, and ``check`` is the
    decision-tree check the atom belongs to (None for a whole-rule tree).
    """

    description: str
    status: str  # "satisfied" | "false" | "no_candidates" | "unknown"
    cost: str  # AtomSpec.cost value: pure | decode | model | llm | yara
    true_positions: tuple[int, ...] = ()
    false_positions: tuple[int, ...] = ()
    excerpts: tuple[tuple[int, str], ...] = ()
    candidate_events: int = 0
    explored: bool = False
    reason: str | None = None
    note: str | None = None
    omitted_positions: int = 0
    source_nodes: tuple[int, ...] = ()
    check: str | None = None
    model_results: tuple[ModelResultEvidence, ...] = ()

    def to_dict(self, node_ids: Mapping[str | None, str] | None = None) -> dict[str, Any]:
        """The JSON shape. ``node_ids`` maps a check to its decision-tree node ID; with it,
        ``node`` names the tree node the atom belongs to."""
        d: dict[str, Any] = {
            "atom": self.description,
            "status": self.status,
            "reason": self.reason,
            "cost": self.cost,
            "true_positions": list(self.true_positions),
            "false_positions": list(self.false_positions),
            "omitted_positions": self.omitted_positions,
            "excerpts": [{"position": pos, "excerpt": text} for pos, text in self.excerpts],
            "candidate_events": self.candidate_events,
            "explored": self.explored,
            "note": self.note,
            "source_nodes": [f"n{index}" for index in self.source_nodes],
            "model_results": [result.to_dict() for result in self.model_results],
        }
        if node_ids is not None:
            d["node"] = node_ids.get(self.check)
        return d

    def format_explain(self) -> list[str]:
        """Render as indented lines for the CLI, two spaces in from the rule heading."""
        lines = [f"  {self.description}"]
        if self.status == "unknown":
            lines.append(f"    unknown ({self.reason})")
        elif self.status == "no_candidates":
            lines.append("    never applied -- nothing on this surface")
        elif self.true_positions:
            lines.append(f"    true at {_format_positions(self.true_positions)}")
        elif self.false_positions:
            lines.append(f"    false at {_format_positions(self.false_positions)}")
        else:
            # A static atom is evaluated once, at the boot step, which has no position.
            lines.append(f"    {self.status}, with no timeline position to cite")
        if self.status == "unknown" and self.false_positions:
            lines.append(f"    false at {_format_positions(self.false_positions)}")
        if self.omitted_positions:
            lines.append(f"    (+{self.omitted_positions} more positions)")
        plural = "" if self.candidate_events == 1 else "s"
        lines.append(f"    {self.candidate_events} candidate event{plural} on this surface")
        for position, text in self.excerpts:
            lines.append(f'    [{position}] "{text}"')
        for result in self.model_results:
            lines.append(f"    {result.operation}: {result.expression_value}")
        if self.note:
            lines.append(f"    note: {self.note}")
        return lines


def _format_positions(positions: tuple[int, ...]) -> str:
    return ", ".join(str(p) for p in positions)


@dataclass(frozen=True)
class RuleEvidence:
    """Why a rule matched or did not: one view of its scan record.

    ``steps`` are a match's support, the load-bearing conversation turns (the minimal
    witness when minimization finished), in timeline order, capped at the first 16;
    ``omitted_steps`` counts the rest. ``atoms`` are what each of the rule's atoms came to,
    the actionable ones first, capped at 24; ``omitted_atoms`` counts the rest.
    ``captured_value`` is the winning ``... as v`` substring for a capture rule (already
    redacted), else ``None``. ``formula_note`` is a short human hint about the rule's shape
    (e.g. ``"bounded flow within:1"``), drawn from the artifact notes. ``reason`` says why
    a non-match did not match when its atoms cannot: ``"empty_collection"`` when the
    condition held but ``collect:`` kept nothing.
    """

    rule_name: str
    matched: bool
    reason: str | None = None
    steps: tuple[EvidenceStep, ...] = ()
    captured_value: str | None = None
    formula_note: str | None = None
    model_results: tuple[ModelResultEvidence, ...] = ()
    omitted_model_results: int = 0
    omitted_steps: int = 0
    atoms: tuple[AtomEvidence, ...] = ()
    omitted_atoms: int = 0

    def to_dict(self, node_ids: Mapping[str | None, str] | None = None) -> dict[str, Any]:
        """Return the stable JSON-ready rule evidence shape."""
        return {
            "rule_name": self.rule_name,
            "matched": self.matched,
            "reason": self.reason,
            "witness": [s.to_dict() for s in self.steps],
            "omitted_witness_steps": self.omitted_steps,
            "captured_value": self.captured_value,
            "formula_note": self.formula_note,
            "model_results": [result.to_dict() for result in self.model_results],
            "omitted_model_results": self.omitted_model_results,
            "atoms": [atom.to_dict(node_ids) for atom in self.atoms],
            "omitted_atoms": self.omitted_atoms,
        }


def _event_content(event: ConversationEvent) -> str | None:
    """Best-effort textual content of an event's source (message/tool call/response)."""
    src = event.source
    content = getattr(src, "content", None)
    if content is not None:
        return str(content)
    # Tool calls have no ``content``; surface their arguments as the excerpt.
    args = getattr(src, "arguments", None)
    if args is not None:
        return str(args)
    return None


def _tested_text(spec, event: ConversationEvent) -> str | None:
    """The text ``spec`` tests on ``event``: the field its ``value_expr_key`` names (for
    ``c.name``, the tool call's name), else the event's content."""
    key = spec.value_expr_key or ""
    path = key.split(":", 1)[1] if key.startswith("field:") else ""
    if path:
        text = field_text(event.source, path)
        if text is not None:
            return text
    return _event_content(event)


def _support_steps(trace: WitnessTrace | None, by_position) -> tuple[list[EvidenceStep], int]:
    """A match's support: each witness position resolved to its event."""
    if trace is None:
        return [], 0
    resolved = [pos for pos in trace.positions if pos in by_position]
    steps = []
    for pos in resolved[:_WITNESS_STEPS_LIMIT]:
        ev = by_position[pos]
        steps.append(
            EvidenceStep(
                position=pos,
                source_path=ev.source_path,
                kind=ev.kind.value if hasattr(ev.kind, "value") else str(ev.kind),
                role=ev.role,
                tool_name=ev.tool_name,
                excerpt=_excerpt(_event_content(ev)),
            )
        )
    return steps, len(resolved) - len(steps)


def leaf_source_nodes(artifact) -> dict[str, tuple[int, ...]]:
    """Each atom's authored node: the outermost source node that lowers to it alone."""

    nodes: dict[str, int] = {}
    for index, atom_ids in artifact.node_atom_ids.items():
        if len(atom_ids) == 1 and index < nodes.get(atom_ids[0], index + 1):
            nodes[atom_ids[0]] = index
    return {atom_id: (index,) for atom_id, index in nodes.items()}


STRUCTURAL_PREDICATES = frozenset({"closed", "on_surface", "tool_quantifier"})


def _is_window(spec) -> bool:
    """A slice's window atom. A sliced condition compiles to it and an unsliced content
    atom, which another quantifier can share, so the window is reported as its own
    conjunct. An indexed atom carries its own ordinal and needs none."""
    return spec.predicate == "on_surface" and spec.is_slice


def explained_atoms(artifact, parts) -> list[str]:
    """The leaf atoms an explanation reports, in authored order, so two scans of one rule
    render identically."""
    source_nodes = leaf_source_nodes(artifact)
    return sorted(
        (
            atom_id
            for atom_id in reads_all(artifact.dfa, parts)
            if (spec := artifact.atom_specs.get(atom_id)) is not None
            # A window (``user_messages[5:]``) is reported: it is the conjunct that limits
            # which events the rest of the condition can hold at.
            and (spec.predicate not in STRUCTURAL_PREDICATES or _is_window(spec))
            and atom_id not in artifact.tool_quantifiers
        ),
        key=lambda atom_id: (source_nodes.get(atom_id, (1 << 30,)), atom_id),
    )


def _atom_checks(artifact, parts) -> dict[str, str]:
    """Each atom's decision-tree check, for a decomposed tree."""

    tree = artifact.decision_tree
    checks: dict[str, str] = {}
    if tree is None or not tree.decomposed:
        return checks
    for node in tree.nodes:
        for atom_id in sorted(reads(node.atom_id, parts)):
            checks.setdefault(atom_id, node.atom_id)
    return checks


def _atom_evidence(
    spec,
    entries: list[tuple],
    *,
    candidates: list[int | None],
    tools: int | None,
    by_position: Mapping[int, ConversationEvent],
    source_nodes: tuple[int, ...],
    check: str | None,
) -> AtomEvidence:
    """One atom's evidence from its record ``entries``. ``candidates`` are the positions it
    can be computed at (``None`` for the boot step); ``tools`` is how many tools a per-tool
    atom ranges over."""

    # One value per candidate: a position true under any entry is where the atom held, and
    # one whose every entry is unknown keeps the first reason. The atom is explored when only
    # the second pass computed it.
    values: dict[int | None, bool | None] = {}
    reasons: dict[int | None, str] = {}
    origins: set[str] = set()
    results: list[ModelResultEvidence] = []
    wanted = set(candidates)
    for _atom_id, position, value, origin, reason, _check, evidence in entries:
        if tools is None and position not in wanted:
            # Off its surface the oracle answers False without testing anything.
            continue
        key = None if tools is not None else position
        if value is None:
            reasons.setdefault(key, reason or "not_computed")
            values.setdefault(key, None)
            continue
        if isinstance(evidence, ModelResultEvidence) and evidence not in results:
            results.append(evidence)
        origins.add(origin)
        values[key] = bool(values.get(key)) or value

    explored = origins == {"explored"}
    true_positions = tuple(sorted(p for p, v in values.items() if v and p is not None))
    false_positions = tuple(sorted(p for p, v in values.items() if v is False and p is not None))
    held = any(values.values())
    candidate_events = tools if tools is not None else len(candidates)
    if held:
        status, reason = "satisfied", None
    elif tools == 0 or not candidates:
        status, reason = "no_candidates", None
    elif tools is None and any(values.get(p) is None for p in candidates):
        status = "unknown"
        missing = [p for p in candidates if values.get(p) is None]
        reason = next((reasons[p] for p in missing if p in reasons), "not_computed")
    elif tools is not None and (values.get(None) is None or None in reasons):
        # Every tool shares one key, so a tool that went unknown leaves the atom unknown
        # even when another tool's probe was false.
        status, reason = "unknown", reasons.get(None, "not_computed")
    else:
        status, reason = "false", None
    if spec.parameter_names and status in ("false", "unknown"):
        # A captured value's atom is computed under the bindings the scan tried, at the
        # positions each reached, so false at every position is not false for every value.
        status, reason = "unknown", "captured_value"

    shown = true_positions if status == "satisfied" else false_positions
    texts = {p: _tested_text(spec, by_position[p]) for p in shown if p in by_position}
    excerpts = tuple(
        (p, text)
        for p in shown[:_MAX_EXCERPTS_PER_ATOM]
        if (text := _excerpt(texts.get(p))) is not None
    )

    notes: list[str] = []
    if status in ("false", "unknown"):
        hint = _case_hint(spec, [text for text in texts.values() if text])
        if hint:
            notes.append(hint)
    if status == "no_candidates":
        notes.append(
            f"the source collection is empty -- no events on {_windowed_surface(spec)}, so the "
            "predicate never ran against anything"
        )
    if explored:
        notes.append("computed for explanation; the verdict did not need it")

    shown_true = true_positions[:_MAX_POSITIONS_PER_ATOM]
    shown_false = false_positions[:_MAX_POSITIONS_PER_ATOM]
    omitted = (len(true_positions) - len(shown_true)) + (len(false_positions) - len(shown_false))
    return AtomEvidence(
        description=describe_atom(spec),
        status=status,
        cost=spec.cost.value,
        true_positions=shown_true,
        false_positions=shown_false,
        excerpts=excerpts,
        candidate_events=candidate_events,
        explored=explored,
        reason=reason,
        note="; ".join(notes) if notes else None,
        omitted_positions=omitted,
        source_nodes=source_nodes,
        check=check,
        model_results=tuple(results[:_MAX_EXCERPTS_PER_ATOM]),
    )


def build_evidence(
    rule_name: str,
    matched: bool,
    record: ScanRecord,
    artifact,
    evaluation_input,
    *,
    trace: WitnessTrace | None = None,
    captured_value: str | None = None,
    model_results: tuple[ModelResultEvidence, ...] = (),
    formula_note: str | None = None,
    with_atoms: bool = True,
) -> RuleEvidence:
    """The explanation of one rule's scan, for a match and a non-match alike.

    Each of the rule's atoms is read from ``record``: its status, true and false positions,
    excerpts of the field it tests, and its model score. A match also carries its support,
    the positions of ``trace``, and the model evidence on them. Without ``with_atoms`` it is
    the support alone. ``captured_value`` is redacted here so evidence never leaks a secret
    it just found.
    """
    events = evaluation_input.events
    by_position = {event.position: event for event in events}
    steps, omitted_steps = _support_steps(trace if matched else None, by_position)

    parts = artifact_parts(artifact)
    tool_body_atoms = {atom_id for body in parts.tool_bodies.values() for atom_id in body}
    tools = len(evaluation_input.static.tools or {})
    source_nodes = leaf_source_nodes(artifact)
    checks = _atom_checks(artifact, parts)
    by_atom: dict[str, list[tuple]] = {}
    for entry in record.entries:
        by_atom.setdefault(entry[0], []).append(entry)

    atoms = []
    for atom_id in explained_atoms(artifact, parts) if with_atoms else ():
        spec = artifact.atom_specs[atom_id]
        per_tool = atom_id in tool_body_atoms or is_per_tool_static(spec)
        if spec.domain is AtomDomain.EVENT:
            candidates: list[int | None] = [
                event.position for event in evaluation_input.in_domain(spec)
            ]
        else:
            candidates = [None]
        atoms.append(
            _atom_evidence(
                spec,
                by_atom.get(atom_id, []),
                candidates=candidates,
                tools=tools if per_tool else None,
                by_position=by_position,
                source_nodes=source_nodes.get(atom_id, ()),
                check=checks.get(atom_id),
            )
        )
    # The atoms an author acts on first: false, then unknown, then no candidates, then
    # satisfied. Within a status authored order is kept.
    rank = {"false": 0, "unknown": 1, "no_candidates": 2, "satisfied": 3}
    ordered = sorted(atoms, key=lambda atom: rank[atom.status])
    retained = tuple(ordered[:_MAX_OBSERVED_ATOMS])
    retained_results = model_results[:_MODEL_RESULTS_LIMIT]
    return RuleEvidence(
        rule_name=rule_name,
        matched=matched,
        reason="empty_collection" if record.empty_collection else None,
        steps=tuple(steps),
        captured_value=redact(captured_value) if captured_value is not None else None,
        formula_note=formula_note,
        model_results=retained_results,
        omitted_model_results=max(0, len(model_results) - len(retained_results)),
        omitted_steps=omitted_steps,
        atoms=retained,
        omitted_atoms=len(atoms) - len(retained),
    )
