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

"""Value capture / register layer.

``... as v`` binds the substring matched at one timeline point and reuses it at another
("the file executed is the file just written"; "the command run contains the text the
user sent"). A captured value is *unbounded* state, so a single finite DFA cannot hold
it. But it is decidable by **instantiation**: a trace yields only finitely many
candidate values (the substrings the bind pattern matches), and the reuse test is
equality — so the executor runs the rule's *parameterized* DFA **once per candidate
value**, grounding the ``capture_bind``/``capture_ref`` atoms as ``contains <value>``.
The DFA structure is identical across candidates; only the atom parameters vary. Capture
rules therefore stay on the ordinary DFA / lazy / witness path.

This module owns the compile-time **recipe** (:class:`CapturePlan`, built by the compiler)
and the candidate enumeration (:func:`iter_capture_values`). The executor consumes both
to run the parameterized DFA once per candidate value. It is an adapter (grounds against
events), so it imports the engine's matchers/events but never the AST, scanner, or DSL
layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field

from quarry.engine import matchers as _M
from quarry.engine import temporal as T
from quarry.engine.pacing import Pacer

# Cost caps per trace. At most ``MAX_CAPTURE_VALUES`` candidates are instantiated (one DFA
# run each), drawn from at most ``MAX_CAPTURE_CANDIDATES`` distinct enumerated values. When a
# cap cuts candidates off and no instantiated candidate matched, the executor reports the
# result incomplete; a match decides a positive existential and stays complete.
MAX_CAPTURE_VALUES = 256
MAX_CAPTURE_CANDIDATES = 4096


@dataclass(frozen=True)
class CaptureBind:
    """Where/how the captured value is extracted (the ``$pat in field as v`` side).

    Matches of ``patterns`` in ``field`` (on ``surface``) become the candidate values.
    ``is_regex`` selects regex vs. substring extraction. ``fullword``/``base64`` mirror the
    DSL string modifiers; without them a ``$pat`` set declared ``fullword``/``base64`` would
    match with plain-substring semantics during candidate enumeration."""

    surface: str
    field: str
    patterns: tuple[str, ...]
    is_regex: bool = False
    case_insensitive: bool = False
    fullword: bool = False
    base64: bool = False
    # The bind's collection within ``surface``: a slice window, and an accessor's tool name.
    window_start: int | None = None
    window_stop: int | None = None
    name: str = ""


@dataclass(frozen=True)
class CaptureTest:
    """Where the captured value is looked for (the ``v in field`` reference side)."""

    surface: str
    field: str
    case_insensitive: bool = False


@dataclass(frozen=True)
class CapturePlan:
    """A compiled value-capture rule: bind a value here, test it there.

    The plan only enumerates candidate values. Where the bind and test steps sit relative
    to each other (``before/after(within:N)``) is decided by the rule's DFA, which runs
    once per candidate. ``bind_atom_id`` / ``ref_atom_id`` are the FIRST parameterized
    bind/ref atoms (evidence + back-compat).

    ``bind_atom_ids`` / ``ref_atom_ids`` are the FULL sets of parameterized atoms the
    executor substitutes the candidate value into per instantiation. A captured value may be
    referenced at several sites, so every ref atom must receive the binding — not
    just the first. Default to the singletons of the ``*_atom_id`` fields when unset so older
    callers/plans keep working.
    """

    var: str
    bind: CaptureBind
    test: CaptureTest
    bind_atom_id: str = ""
    ref_atom_id: str = ""
    bind_atom_ids: frozenset[str] = frozenset()
    ref_atom_ids: frozenset[str] = frozenset()

    def all_capture_atom_ids(self) -> set[str]:
        """Every parameterized capture atom id (bind + ref) the executor must bind the
        candidate value into. Falls back to the singleton ``*_atom_id`` fields when the sets
        are empty (plans built by external callers that set only the singletons)."""
        binds = set(self.bind_atom_ids) or ({self.bind_atom_id} if self.bind_atom_id else set())
        refs = set(self.ref_atom_ids) or ({self.ref_atom_id} if self.ref_atom_id else set())
        return binds | refs


@dataclass(frozen=True)
class ElementPlan:
    """A comprehension element instantiated per candidate event (the pinned-element fallback).

    ``any var in surface[window_start:window_stop] where ...`` runs the parameterized DFA once
    per event of ``surface`` inside the window that satisfies ``filter_atom_id``, the name
    test ``name = filter_name`` of an accessor such as ``get_tool_calls("x")``. ``at_atom_id``
    holds exactly at the candidate's event;
    each atom in ``ref_atom_ids`` tests another field against a field of the candidate, named
    in its ``parameter_names`` as ``var.field``. ``own_step`` is the conjunction of the tests
    the rule makes on the candidate's own event alone, over event atoms of ``surface``: a
    candidate whose event fails it runs as no candidate bound, so it needs no run of its own."""

    var: str
    surface: str
    at_atom_id: str
    ref_atom_ids: frozenset[str] = frozenset()
    window_start: int | None = None
    window_stop: int | None = None
    filter_atom_id: str = ""
    filter_name: str = ""
    own_step: T.Formula = T.TRUE

    def all_capture_atom_ids(self) -> set[str]:
        """Return all capture atom ids."""
        return {self.at_atom_id, *self.ref_atom_ids}


@dataclass(frozen=True)
class ElementBinding:
    """One candidate of an :class:`ElementPlan`, identified (and cached) by its position.

    A reference atom reads the candidate's field when it is grounded. The scanner validates
    tool arguments before binding candidates, so malformed JSON invalidates the whole scan."""

    position: int
    label: str = field(compare=False)
    source: object = field(compare=False, repr=False)

    def text(self, parameter: str) -> str:
        """The candidate's text for a ``var.field`` parameter ("" when the field is absent)."""
        return _M.field_text(self.source, parameter.split(".", 1)[1]) or ""


# Yielded by :func:`iter_capture_values` after ``MAX_CAPTURE_CANDIDATES`` values when more exist.
MORE = object()


def iter_capture_values(
    plan: CapturePlan,
    events,
    *,
    regex_backend=None,
    pause: object | None = None,
    surface_sizes: Mapping[str, int] | None = None,
) -> Iterator[object]:
    """Distinct candidate values the bind pattern matches on the bind surface, lazily.

    The finite set the executor instantiates the rule's formula over: for each value
    ``v`` the ``capture_bind``/``capture_ref`` atoms ground as ``contains v`` and the
    ordinary DFA runs. Order is deterministic (timeline order, first occurrence wins).
    Matches are streamed; after ``MAX_CAPTURE_CANDIDATES`` distinct values, ``MORE`` is
    yielded once if another distinct value exists. With ``pause``, that object is also
    yielded whenever a pause is due (``pacing``), counting events visited and raw matches.

    ``regex_backend`` (a ``matchers.RegexBackend``) grounds the regex enumeration path:
    the scanner injects the RE2 backend, which is linear in the input, so a
    catastrophic-backtracking capture pattern cannot stall the scan. Defaults to the pure
    stdlib backend for offline callers.

    ``surface_sizes`` gives each surface's size in the whole conversation, which a bind
    window's signed bounds resolve against, so ``events`` may be any subset of it, such as a
    minimization replay's. Without it the size is counted from ``events``.

    The candidate value is the **matched substring**, honoring the bind's string modifiers
    (nocase/fullword/base64/regex). A literal ``$pat`` match yields the actual text
    matched in the field (respecting fullword/base64 pattern transforms), not the raw pattern.
    """
    seen: set[str] = set()
    pacer = Pacer(every=64)
    for value in _iter_bind_values(plan, events, regex_backend or _M.PURE_REGEX, surface_sizes):
        if pause is not None and pacer.due():
            # Many events or repeats of one value can run long without a new candidate;
            # ``pause`` hands control back so an async caller can yield and a deadline fire.
            yield pause
        if value is None or value in seen:
            continue
        if len(seen) == MAX_CAPTURE_CANDIDATES:
            yield MORE
            return
        seen.add(value)
        yield value


def _iter_bind_values(
    plan: CapturePlan, events, regex_backend, surface_sizes: Mapping[str, int] | None = None
) -> Iterator[str | None]:
    """Every non-empty value the bind side yields, in timeline order, with repeats, and
    ``None`` for each event visited (a unit of work that yields no value)."""
    b = plan.bind
    mods = _M.Modifiers(
        nocase=b.case_insensitive, fullword=b.fullword, base64=b.base64, regex=b.is_regex
    )
    # Only events of the bind's collection: inside its window, with its accessor's name. An
    # event's ordinal counts the whole conversation, so the window resolves against the
    # surface's whole size.
    window = None
    if b.window_start is not None or b.window_stop is not None:
        if surface_sizes is not None:
            size = surface_sizes.get(b.surface, 0)
        else:
            events = list(events)
            size = sum(1 for ev in events if b.surface in ev.surface_tags)
        window = slice(b.window_start, b.window_stop).indices(size)[:2]
    for ev in events:
        yield None
        if b.surface not in ev.surface_tags:
            continue
        if window is not None and not window[0] <= ev.ordinals.get(b.surface, -1) < window[1]:
            continue
        if b.name and _M.field_text(ev.source, "name") != b.name:
            continue
        text = _M.field_text(ev.source, b.field) if b.field else None
        if not text:
            continue
        if not b.patterns:
            # Cross-field join (no bind pattern): the candidate value is the WHOLE bind
            # field text (e.g. `command contains msg.content` captures the entire
            # msg.content). One candidate per bind event.
            yield text
            continue
        for pat in b.patterns:
            for value in matches_for(pat, text, mods, regex_backend):
                if value:
                    yield value


def matches_for(pattern: str, text: str, mods, regex_backend) -> Iterable[str]:
    """The matched substrings of ``pattern`` in ``text`` under ``mods`` — the candidate
    values for one bind pattern. Honors the DSL string modifiers and routes all regex work
    through ``regex_backend`` so the injected engine applies.

    A regex/fullword/base64 pattern enumerates EVERY match (``iter_match_texts``, lazily); a
    plain literal yields the single matched substring (case/positionally faithful) when present.
    """
    effective, use_regex = _M.apply_modifiers(pattern, mods)
    flags = _M.re.IGNORECASE if mods.nocase else 0
    if use_regex:
        try:
            return regex_backend.iter_match_texts(effective, text, flags)
        except _M.re.error:
            # Invalid regex: fall back to a literal presence check on the raw pattern
            # (mirrors the matcher layer's tolerance), yielding the matched substring.
            got = _M.extract_match(text, pattern, mods, regex=regex_backend)
            return [got] if got is not None else []
    got = _M.extract_match(text, pattern, mods, regex=regex_backend)
    return [got] if got is not None else []
