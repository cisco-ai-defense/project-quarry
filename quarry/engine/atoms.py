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

"""Atom model.

An :class:`AtomSpec` immutably describes a single boolean question about one event or
about the static environment — "does this event's field contain X?", "is this tool
defined?". Atoms are the leaves of the DFA: a rule's LTLf formula is built from atom
ids, and grounding (the oracle) answers each atom against a conversation.

Two properties matter here:

- **Content-derived id.** An atom's id is a structural hash of *what it asks*, not its
  object identity or position in a rule. So two rules that both ask "does a tool call's
  ``arguments.command`` contain ``sudo``?" mint the *same* id and share one spec — the
  basis for scan-local result deduplication across rules.
- **Interning.** :class:`AtomInterner` returns the same :class:`AtomSpec` object for
  equal content, so downstream passes can compare atoms by identity and cache by id.

This is a pure core module (no Quarry imports): an atom is a description, not a
grounding. How each ``predicate`` is actually evaluated (regex libraries, decode /
normalize transforms, detector routing) is the matcher layer's job; here we only carry
enough structure to intern and to type-check.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import StrEnum

# Separator for a match-position *operand group*: a single ``position``/``position_abs``
# operand can be a set of patterns (e.g. the wildcard ``@a*`` expands to every ``$a*``
# string), whose match position is the EARLIEST across the group. The group is encoded as
# a NUL-joined string in one ``patterns`` slot so it stays a plain hashable str (NUL never
# occurs in a real pattern); a single-pattern operand is just the pattern itself, keeping
# the encoding — and thus the atom id — identical to before for the common case.
POSITION_GROUP_SEP = "\x00"

# Separator for a CROSS-FIELD match-position operand (predicate ``position_cross``):
# ``@a in fieldA <op> @b in fieldB`` compares match positions taken from two DIFFERENT
# fields of the SAME event. Each of the two ``patterns`` slots is ``"<field>\x1f<group>"``
# — the operand's field path, then a NUL-joined pattern group (see POSITION_GROUP_SEP).
# (The single-field ``position`` predicate keeps its plain ``(groupA, groupB)`` encoding
# and its one ``value_expr_key`` field, so its atom ids are unchanged.)
POSITION_FIELD_SEP = "\x1f"


class AtomDomain(StrEnum):
    """Where an atom's question is asked."""

    EVENT = "event"
    STATIC = "static"


class Cost(StrEnum):
    """Relative grounding cost tier, cheapest first. Drives the executor's cheap-first
    ordering and skipping (a costly atom is grounded last, or skipped when it cannot
    change the verdict)."""

    PURE = "pure"
    DECODE = "decode"
    MODEL = "model"
    LLM = "llm"
    YARA = "yara"

    @property
    def rank(self) -> int:
        """Return the rank of the cost."""
        return _COST_RANK[self]


_COST_RANK = {Cost.PURE: 0, Cost.DECODE: 1, Cost.MODEL: 2, Cost.LLM: 3, Cost.YARA: 4}


@dataclass(frozen=True, slots=True)
class ModelExpression:
    """Validated model predicate evaluated before the DFA consumes its boolean result.

    The payload deliberately contains no DSL AST nodes. ``kind`` identifies a closed
    operation, ``value`` carries one scalar, and ``children`` contains its operands.
    Keeping the representation frozen and hashable makes the complete predicate part
    of an atom's structural identity without adding non-boolean values to the DFA.
    """

    kind: str
    value: str | int | float | bool | None = None
    children: tuple[ModelExpression, ...] = ()

    def content_key(self) -> tuple:
        """Return a stable recursive identity for atom hashing."""

        return self.kind, self.value, tuple(child.content_key() for child in self.children)


@dataclass(frozen=True)
class StringSemantics:
    """How string matching behaves for an atom's predicate (the DSL string modifiers).

    Carries the Quarry ``StringModifiers`` surface plus decode / normalize
    pre-transforms. The matcher layer consumes these; here they are part of the atom's content identity so two atoms that match the
    same pattern with *different* semantics do not collide.
    """

    case_insensitive: bool = False
    fullword: bool = False
    base64: bool = False
    is_regex: bool = False
    decode: str = ""
    normalize: str = ""
    # "any" (match if any pattern hits) vs "all" (all must hit) for pattern lists.
    list_mode: str = "any"

    def content_key(self) -> tuple:
        """Return the content key of the string semantics."""
        return (
            self.case_insensitive,
            self.fullword,
            self.base64,
            self.is_regex,
            self.decode,
            self.normalize,
            self.list_mode,
        )


@dataclass(frozen=True)
class AtomSpec:
    """An immutable boolean question about one event or the static environment.

    Fields:
        atom_id: content-derived structural hash (set by :func:`make_atom` /
            :class:`AtomInterner`; identical content -> identical id).
        domain: EVENT or STATIC.
        selector: for EVENT, the surface/kind the event must belong to (e.g.
            ``tool_calls``); for STATIC, the static collection name (e.g. ``tools``).
        value_expr_key: a canonical string key for the value being tested (the field
            path / function); the atom stores only this canonical key so it is a pure,
            hashable description.
        predicate: the test kind — contains / regex / equals / count / position /
            length / entropy / ratio / near / order / fuzzy / capability / classifier.
        patterns: the operands of the predicate (patterns, a capability tag, etc.).
        string_semantics: string-matching modifiers + transforms.
        op / number: comparator + numeric operand for scalar-gate predicates.
        cost: grounding cost tier.
        detector / detector_arg / class_index: external-backend routing (classifier).
        cache_policy: scan-local dedup key (defaults to the atom id).
        evidence_policy: what provenance to retain ("span" | "event" | "none").
        parameter_names: capture-binding parameter names, if this atom is
            value-parameterized (empty for ordinary atoms).
    """

    domain: AtomDomain
    selector: str
    value_expr_key: str
    predicate: str
    patterns: tuple[str, ...] = ()
    string_semantics: StringSemantics = field(default_factory=StringSemantics)
    op: str = ""
    number: float | None = None
    cost: Cost = Cost.PURE
    detector: str = ""
    detector_arg: str = ""
    class_index: int = 1
    # External-classifier decision threshold folded from a literal DSL arg (e.g.
    # ``model_scan(model, field, 0.72, 3)`` → 0.72). None = the function's own default
    # applies. Distinct from ``number`` (a folded DSL *comparator* operand) so the two
    # never conflate. Part of the content key so distinct thresholds are distinct atoms.
    threshold: float | None = None
    # Vestigial second detector literal (an explicit per-atom runtime model id). No DSL
    # function folds it, so it stays "". Retained in the content key (never
    # dropped) so every existing atom keeps its exact historical identity and dfa_hash.
    detector_model: str = ""
    # A second detector literal for ``llm_score`` — the constitution/policy text (or a
    # ``text_ref`` name) its rule scores the conversation against (e.g.
    # ``llm_score("model", "policy", conversation())``). ``detector_arg`` holds the model id,
    # so the constitution needs its own slot. Part of the content key so rules scoring against
    # different constitutions are distinct atoms.
    constitution: str = ""
    # True when ``constitution`` is a text_ref NAME (resolved from config.texts at scan time),
    # False when it is literal policy text. Part of the content key so a named ref and a
    # literal of the same string stay distinct atoms.
    constitution_is_ref: bool = False
    # Chunked scoring (``llm_chunk_reduce``): the per-chunk input-token budget folded from
    # ``llm_split_conversation(model, conversation(), N)``'s third argument. 0 = not a chunked
    # atom. Part of the content key: two rules scoring the same constitution under DIFFERENT
    # token budgets get different chunk boundaries, hence different verdicts, hence must not
    # intern to one atom.
    chunk_max_tokens: int = 0
    # How the per-chunk values reduce to the single number the comparator sees: "max" / "min"
    # (the quantifier forms — which one is correct depends on the comparator's direction, not
    # just on any/all) or "count" (the cardinality form, `count([c for c in … if …]) >= N`).
    # Empty = not chunked. In the content key for the same reason as ``chunk_max_tokens``.
    chunk_combine: str = ""
    # WHICH per-chunk scalar is reduced: "llm_score" (a constitution score) or
    # "llm_input_tokens" (a token count). Empty = not chunked. In the content key because two
    # atoms can otherwise agree on every other folded field — an ``llm_score`` atom with an
    # empty constitution and an ``llm_input_tokens`` atom — and must not intern together.
    chunk_scalar: str = ""
    # For ``chunk_combine="count"`` only: the PER-CHUNK threshold test whose satisfying chunks
    # are counted. The atom's own ``op``/``number`` are then the OUTER comparison against the
    # count (`>= 2`), so a cardinality atom carries two comparisons, not one. Empty/0.0 for the
    # max/min forms. In the content key: counting chunks above 0.9 is a different question from
    # counting chunks above 0.5 under an otherwise identical rule.
    chunk_inner_op: str = ""
    chunk_inner_number: float = 0.0
    evidence_policy: str = "span"
    parameter_names: tuple[str, ...] = ()
    # Per-surface ordinal constraint (from indexed access like ``user_messages[0]``):
    # the atom fires only at the event whose ordinal within ``selector`` equals this.
    # None = no positional constraint (any event on the surface).
    ordinal: int | None = None
    # Vestigial. A tool result is ONE canonical event on both the ``messages`` and
    # ``tool_responses`` surfaces, so there is no dual-surface reachability to disable and the
    # compiler never sets this — a ``tool_responses``↔``tool_responses`` flow already requires
    # two distinct response events. Retained (always False) in the content key so every existing
    # atom keeps its exact historical identity/dfa_hash rather than churning the whole corpus.
    no_pair_reachability: bool = False
    # Windowed-surface constraint (from a slice like ``messages[-4:]``): the atom fires only
    # at events whose ordinal within ``selector`` falls in the Python-style window
    # [slice_start:slice_stop), resolved against the surface size at grounding (negative
    # bounds count from the end). ``is_slice`` distinguishes an explicit window from "no
    # constraint"; it gates the window into the content key (below) so a sliced atom interns
    # distinctly WITHOUT perturbing the identity of every non-sliced atom.
    slice_start: int | None = None
    slice_stop: int | None = None
    is_slice: bool = False
    # Typed model-input selection for an inference op (llm_score / llm_input_tokens /
    # llm_complete): the conversation-valued rule argument.
    #
    # ``input_pipeline is None`` is the DEFAULT — ``conversation()``, the whole transcript WITH
    # the request's offered tools — so every existing atom keeps its exact historical
    # identity/dfa_hash. Any other value selects a Messages value (NO tools) and is an ORDERED
    # pipeline of steps applied left-to-right to the root ``Messages`` at grounding time:
    #   ``()``                        → the whole transcript (messages /
    #                                   conversation().messages), no tools
    #   ``(("role", "user_messages"),)`` → a role filter (user_messages / messages.user_messages)
    #   ``(("slice", a, b),)``        → a window (messages[a:b])
    #   chained, order-preserving     → e.g. slice-then-filter vs filter-then-slice differ, and
    #                                   cross-role chains (user→assistant) resolve to empty
    # ``message(...).add_message(...)`` constructed input uses ``constructed_input`` (below).
    # ``input_pipeline`` is gated into the content key whenever it is not None so a distinct
    # selection interns distinctly without perturbing the identity of a plain conversation() atom.
    input_pipeline: tuple[tuple, ...] | None = None
    # Constructed model input for ``llm_score``: a literal synthetic conversation built from a
    # ``message(role, content)[.add_message(...)]*`` chain in the rule. Empty = not constructed
    # (whole transcript or a window). Each turn is a ``(role, content)`` pair. Gated into the
    # content key below so a constructed-input atom interns distinctly WITHOUT perturbing the
    # identity of every non-constructed atom.
    constructed_input: tuple[tuple[str, str], ...] = ()
    # Dynamic constructed-content field paths, one per ``None`` (dynamic) turn in
    # ``constructed_input``, in slot order. Empty = no dynamic slot, OR exactly one dynamic slot
    # (which keeps its historical single-field representation via ``value_expr_key`` so its atom
    # identity is unchanged). Set only for TWO OR MORE dynamic slots, all of which must reference
    # the SAME event source; the oracle resolves each field off that source and the scanner fills
    # the slots in order. Gated into the content key below only when present.
    constructed_field_paths: tuple[str, ...] = ()
    # Closed model-expression payload. It is present only on the version 2 model-object
    # lowering and is evaluated by the scanner before the DFA sees one boolean value.
    model_expression: ModelExpression | None = None
    # Content-derived id; filled by make_atom(). Excluded from the content key so it
    # cannot affect its own hash.
    atom_id: str = ""

    def content_key(self) -> tuple:
        """The tuple that defines this atom's identity — everything *except* the id and
        the cache/evidence *policies* (which are annotations, not part of the question).
        Two atoms with the same content key ask the same boolean question."""
        key = (
            self.domain.value,
            self.selector,
            self.value_expr_key,
            self.predicate,
            self.patterns,
            self.string_semantics.content_key(),
            self.op,
            self.number,
            self.cost.value,
            self.detector,
            self.detector_arg,
            self.class_index,
            self.threshold,
            self.detector_model,
            self.constitution,
            self.constitution_is_ref,
            self.chunk_max_tokens,
            self.chunk_combine,
            self.chunk_scalar,
            self.chunk_inner_op,
            self.chunk_inner_number,
            self.parameter_names,
            self.ordinal,
            self.no_pair_reachability,
        )
        # Append the window ONLY for sliced atoms so non-sliced atoms keep their exact
        # historical identity (and every existing rule keeps its dfa_hash).
        if self.is_slice:
            key = (*key, "slice", self.slice_start, self.slice_stop)
        # A Messages selection (no tools) — any ordered role-filter/slice pipeline — is a
        # distinct model input from the default conversation(); fold it in only when present, so
        # a plain conversation() atom (input_pipeline is None) keeps its exact historical identity.
        if self.input_pipeline is not None:
            key = (*key, "input_pipeline", self.input_pipeline)
        if self.constructed_input:
            key = (*key, "constructed", self.constructed_input)
        # Multiple dynamic content field paths distinguish otherwise-identical constructed atoms;
        # gated only when present so a single-slot (or all-literal) atom keeps its identity.
        if self.constructed_field_paths:
            key = (*key, "constructed_fields", self.constructed_field_paths)
        if self.model_expression is not None:
            key = (*key, "model_expression", self.model_expression.content_key())
        return key


def _encode_content_key(key: tuple) -> str:
    """Canonical identity shared by atom IDs and the interner.

    Tuple equality alone conflates values such as ``True`` and ``1`` even though
    their JSON encodings (and therefore their atom IDs) differ.
    """
    return json.dumps(key, sort_keys=True, separators=(",", ":"), default=_json_default)


def _hash_content_key(key: tuple) -> str:
    """Deterministic short structural hash of an atom content key."""
    return _hash_encoded_content_key(_encode_content_key(key))


def _hash_encoded_content_key(encoded: str) -> str:
    """Hash a canonical content key without encoding it a second time."""
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    return f"atom_{digest[:16]}"


def _json_default(o):
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def is_per_tool_static(spec: AtomSpec) -> bool:
    """True for a genuine PER-TOOL existential atom (``any tool in tools where φ(tool)``).

    ``surface_count`` and ``tool_names_all`` atoms are whole-collection aggregates grounded
    once at the boot step. Routing either through the per-tool loop is wrong: it would
    ground a collection-wide predicate against one tool object. A conversation with
    ZERO tools would run the loop zero times and return no-match even when ``len(tools) < 1``
    should hold. Excluding the aggregate keeps per-tool grounding for real existentials only.
    """
    return (
        spec.domain is AtomDomain.STATIC
        and spec.selector == "tools"
        and spec.predicate not in {"surface_count", "tool_names_all", "tool_quantifier"}
    )


# Selector of an atom that tests a captured value itself (``"x" in v``): a static atom whose
# text is the value the capture runner binds.
CAPTURE_VALUE = "\x00capture_value"


def make_atom(**kwargs) -> AtomSpec:
    """Construct an :class:`AtomSpec` with a content-derived id.

    Any ``atom_id`` passed in is ignored; the id is always derived from the content so
    it is deterministic and collision-resistant across rules.
    """
    kwargs.pop("atom_id", None)
    spec = AtomSpec(**kwargs)
    atom_id = _hash_content_key(spec.content_key())
    # make_atom's callers pass explicit, correctly-typed fields; ty can't verify the merged
    # **dict unpack into the dataclass, so suppress the (false-positive) argument-type
    # diagnostics.
    return AtomSpec(**{**kwargs, "atom_id": atom_id})  # type: ignore


class AtomInterner:
    """Returns one canonical :class:`AtomSpec` per distinct content, across all rules.

    Interning is the mechanism behind cross-rule sharing: two rules asking the same
    question get the same object (and id), so the executor grounds it once per scan.
    """

    def __init__(self) -> None:
        self._by_key: dict[str, AtomSpec] = {}

    def intern(self, spec: AtomSpec) -> AtomSpec:
        """Intern the given atom spec."""
        content_key = spec.content_key()
        key = _encode_content_key(content_key)
        existing = self._by_key.get(key)
        if existing is not None:
            return existing
        canonical = spec if spec.atom_id else replace(spec, atom_id=_hash_encoded_content_key(key))
        self._by_key[key] = canonical
        return canonical

    def make(self, **kwargs) -> AtomSpec:
        """Build (with content id) and intern in one step."""
        return self.intern(make_atom(**kwargs))

    def __len__(self) -> int:
        return len(self._by_key)

    @property
    def specs(self) -> tuple[AtomSpec, ...]:
        """Return the specs of the atom interner."""
        return tuple(self._by_key.values())
