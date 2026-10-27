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

"""Predicate grounding oracle.

The single authority that answers "is atom X true at event E (under capture bindings
B)?". It keeps the per-predicate grounding logic in one place, backed by the shared
matchers, with a scan-local cache so identical atoms are grounded once per scan.

Cost tiers (``AtomSpec.cost``):
  - PURE / DECODE — local: substring/regex/equality/count/length/position, grounded
    inline via ``quarry.engine.matchers``.
  - MODEL / LLM / YARA — external detectors. Grounded only on the async path, through the
    scanner's injected ``async_detector``; grounding one synchronously raises
    ``DetectorUnavailable``, so an ungrounded detector is never silently False.

Boot step: STATIC atoms are grounded against the :class:`StaticEnvironment`, not
an event; the executor evaluates them at a synthetic boot step. Capture: a
value-parameterized atom (empty ``patterns`` + ``parameter_names``) is grounded against
a per-candidate binding the executor supplies.

Adapter module: imports engine matchers/events/atoms only (never the AST, scanner, or DSL
layer).
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from quarry.engine import matchers as _M
from quarry.engine.atoms import CAPTURE_VALUE, AtomDomain, AtomSpec
from quarry.engine.atoms import POSITION_FIELD_SEP as _POSITION_FIELD_SEP
from quarry.engine.atoms import POSITION_GROUP_SEP as _POSITION_GROUP_SEP
from quarry.engine.captures import ElementBinding, matches_for
from quarry.engine.events import in_window

# Predicates whose grounding requires an external detector backend (MODEL/LLM/YARA). The
# compiler emits all of them with predicate "external"; the older names are accepted too.
_EXTERNAL_PREDS = ("model_scan", "model_score", "external")
ExternalText = str | tuple[str, ...]


class DetectorUnavailable(Exception):
    """Raised when an external (MODEL/LLM/YARA) atom is grounded but no detector backend
    was provided. Surfaced (not silenced) per the failure-policy contract."""


class ExternalInputRejected(Exception):
    """Raised when an external atom's resolved input is invalid — e.g. a dynamic ``message()``
    content field that resolves to a non-string. Surfaced (not silenced) so the scan fails
    closed: the rule is recorded as incomplete rather than scored on coerced/fabricated input,
    and no provider/adapter call is ever made. Engine-local (the oracle must not import
    ``quarry.types.exceptions``); the scanner's broad per-rule handler turns it into the same
    incomplete verdict as any grounding failure."""


class _InputAbsent(Exception):
    """A dynamic ``message()`` content field is missing from this event, so there is no model
    input and the atom is false here, as any test of a missing field is."""


async def _default_async_detector(spec: AtomSpec, text: ExternalText, event=None) -> bool:
    raise DetectorUnavailable(
        f"external atom {spec.atom_id} (detector={spec.detector!r}) has no async backend; "
        "inject one via Oracle(async_detector=...)"
    )


def _detector_accepts_event(detector: Callable[..., Awaitable[object]]) -> bool:
    """Return whether an async detector accepts the optional event argument."""

    try:
        parameters = inspect.signature(detector).parameters.values()
    except (TypeError, ValueError):
        return True
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    return len(positional) >= 3 or any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    )


# What the executor passes as "where to ground": an event view, or the boot marker.
BOOT = object()


@dataclass(frozen=True)
class Grounded:
    """Result of grounding one atom at one location."""

    value: bool
    evidence: object | None = None  # Field span or object reference.


# An async detector takes ``(spec, text)`` or ``(spec, text, event)``; ``Oracle`` checks
# which at construction (``_detector_accepts_event``) and calls it accordingly.
AsyncDetector = (
    Callable[[AtomSpec, ExternalText], Awaitable[bool | Grounded]]
    | Callable[[AtomSpec, ExternalText, object | None], Awaitable[bool | Grounded]]
)


class Oracle:
    """Scan-local atom grounding with a shared cache.

    ``async_detector`` grounds external atoms; it defaults to raising, so an ungrounded
    detector is never silently False. ``static_env`` supplies the boot-step field values for
    STATIC atoms.
    """

    def __init__(
        self,
        static_env=None,
        async_detector: AsyncDetector | None = None,
        regex_backend: _M.RegexBackend | None = None,
    ):
        self._static = static_env
        # Regex backend for local text matchers. Defaults to the pure stdlib backend; the
        # scanner injects the RE2 backend, which is linear in the input, so a ReDoS
        self._regex = regex_backend or _M.PURE_REGEX
        # Async external backend (used by ground_async / value_async — the scanner path).
        # Defaults to raising so an ungrounded async external is never silently False.
        # Typed by its result only: the call site picks the arity from the flag below.
        self._async_detector: Callable[..., Awaitable[bool | Grounded]] = (
            async_detector or _default_async_detector
        )
        self._async_detector_accepts_event = _detector_accepts_event(self._async_detector)
        self._cache: dict[tuple, Grounded] = {}
        self.external_calls = 0

    @staticmethod
    def _is_external(spec: AtomSpec) -> bool:
        return spec.predicate in _EXTERNAL_PREDS

    @staticmethod
    def _key(spec: AtomSpec, where, binding: str | None) -> tuple:
        loc = ("boot",) if where is BOOT else ("event", where.position)
        # For a value-parameterized atom the binding is part of the question.
        return (spec.atom_id, loc, binding if spec.parameter_names else None)

    def ground(self, spec: AtomSpec, where, binding: str | None = None) -> Grounded:
        """Ground ``spec`` at ``where`` (an event view or ``BOOT``) under an optional
        capture ``binding`` value. Cached per (atom, location, binding)."""
        if isinstance(binding, ElementBinding):
            # A pinned candidate's atoms (its position test and its joins) differ per candidate
            # and event, and each is read about once per run, so they are recomputed instead of
            # retained.
            return self._ground_uncached(spec, where, binding)
        key = self._key(spec, where, binding)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        result = self._ground_uncached(spec, where, binding)
        self._cache[key] = result
        return result

    def value(self, spec: AtomSpec, where, binding: str | None = None) -> bool:
        """Return the value of one grounded atom and binding without evaluating it again."""
        return self.ground(spec, where, binding).value

    def evidence_for(self, spec: AtomSpec, where, binding: str | None = None) -> object | None:
        """Return evidence for one grounded atom and binding without evaluating it again."""

        grounded = self._cache.get(self._key(spec, where, binding))
        return grounded.evidence if grounded is not None else None

    def surface_size(self, spec: AtomSpec) -> int:
        """The collection size a ``surface_count`` atom compares: ``len(S)``/``count(S)``.

        A name filter (patterns[0], e.g. count_tool_calls("x") / len(get_tool_responses("x")))
        narrows to matching events on the relevant surface. The aggregate must match the
        surface — responses for tool_responses, calls otherwise — else a name-filtered
        response count would read call counts.
        """

        assert self._static is not None
        if spec.patterns:
            if spec.selector == "tool_responses":
                size = self._static.tool_response_name_counts.get(spec.patterns[0], 0)
            else:
                size = self._static.tool_call_name_counts.get(spec.patterns[0], 0)
        else:
            size = self._static.surface_sizes.get(spec.selector, 0)
        if spec.is_slice:
            size = len(range(*slice(spec.slice_start, spec.slice_stop).indices(size)))
        return size

    @staticmethod
    def _tool_key(spec: AtomSpec, tool, binding: str | None) -> tuple:
        return (spec.atom_id, ("tool", id(tool)), binding if spec.parameter_names else None)

    def ground_tool(self, spec: AtomSpec, tool, binding: str | None = None) -> Grounded:
        """Ground one tools-static atom against one pinned tool, with replay caching."""

        key = self._tool_key(spec, tool, binding)
        hit = self._cache.get(key)
        if hit is None:
            hit = self._apply_predicate(spec, tool, binding, None)
            self._cache[key] = hit
        return hit

    async def ground_tool_async(self, spec: AtomSpec, tool, binding: str | None = None) -> Grounded:
        """Async pinned-tool grounding uses the same detector path as other async atoms."""

        if not self._is_external(spec):
            return self.ground_tool(spec, tool, binding)
        key = self._tool_key(spec, tool, binding)
        hit = self._cache.get(key)
        if hit is None:
            hit = await self._apply_predicate_async(spec, tool, binding, None)
            self._cache[key] = hit
        return hit

    def evidence_for_tool(self, spec: AtomSpec, tool, binding: str | None = None) -> object | None:
        """Return evidence for one grounded tool atom and binding without evaluating it again."""
        grounded = self._cache.get(self._tool_key(spec, tool, binding))
        return grounded.evidence if grounded is not None else None

    # Only EXTERNAL atoms (MODEL/LLM/YARA) need to await; every local (PURE/DECODE)
    # atom is grounded by the identical sync logic. So the async methods delegate any
    # non-external atom straight to the sync path and differ ONLY at the detector leaf —
    # there is no second copy of the gating/predicate logic to drift.
    async def ground_async(self, spec: AtomSpec, where, binding: str | None = None) -> Grounded:
        """Async counterpart of :meth:`ground`; awaits the injected async detector for
        external atoms. Shares the same scan-local cache as the sync path."""
        if not self._is_external(spec):
            return self.ground(spec, where, binding)
        key = self._key(spec, where, binding)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        result = await self._ground_uncached_async(spec, where, binding)
        self._cache[key] = result
        return result

    async def value_async(self, spec: AtomSpec, where, binding: str | None = None) -> bool:
        return (await self.ground_async(spec, where, binding)).value

    async def _ground_uncached_async(self, spec: AtomSpec, where, binding: str | None) -> Grounded:
        if spec.domain is AtomDomain.STATIC:
            return await self._ground_static_async(spec, where, binding)
        if where is BOOT:
            return Grounded(False)
        return await self._ground_event_async(spec, where, binding)

    async def _ground_event_async(self, spec: AtomSpec, event, binding: str | None) -> Grounded:
        if not self._is_external(spec):
            return self._ground_event(spec, event, binding)
        if spec.selector not in event.surface_tags:
            return Grounded(False)
        if not self._position_ok(spec, event):
            return Grounded(False)
        return await self._apply_predicate_async(spec, event.source, binding, event)

    async def _ground_static_async(self, spec: AtomSpec, where, binding: str | None) -> Grounded:
        if not self._is_external(spec):
            return self._ground_static(spec, where, binding)
        if self._static is None:
            return Grounded(False)
        if spec.selector == "system_prompt":
            return await self._apply_text_predicate_async(
                spec, self._static.system_prompt or "", binding, None
            )
        if spec.selector == "tools":
            for tool in self._static.tools.values():
                if (await self._apply_predicate_async(spec, tool, binding, None)).value:
                    return Grounded(True)
            return Grounded(False)
        if spec.selector == "messages":
            return await self._apply_text_predicate_async(spec, "", binding, None)
        return Grounded(False)

    async def _apply_predicate_async(self, spec: AtomSpec, source, binding, event) -> Grounded:
        # Only external predicates reach here in the async path; extract the scan text
        # exactly as the sync _apply_predicate does, then await the detector.
        fld = (
            spec.value_expr_key.split(":", 1)[1] if spec.value_expr_key.startswith("field:") else ""
        )
        try:
            text = self._external_scan_text(spec, source, fld)
        except _InputAbsent:
            return Grounded(False)
        return await self._apply_text_predicate_async(spec, text, binding, event)

    def _external_scan_text(self, spec: AtomSpec, source, fld: str) -> ExternalText | None:
        """The text an external atom scores off ``source``.

        A dynamic constructed-content atom (``message("user", call.arguments.command)``) fills a
        ``message()`` turn with this value VERBATIM, so it must ALREADY be a string — a
        number / bool / array / object / present-null field is rejected here (before any
        provider dispatch), never ``str()``-coerced into fabricated model input. A missing field
        leaves no input (``_InputAbsent``: the atom is false at this event). An empty string is
        a valid (if empty) string and is preserved, distinct from a missing field. Every other
        external atom keeps the coercing field text."""

        def _require_str(path: str):
            raw = _M.field_access(source, path)
            if raw is _M.MISSING:
                raise _InputAbsent(path)
            if not isinstance(raw, str):
                raise ExternalInputRejected(
                    f"dynamic message() content field {path!r} must resolve to a string, "
                    f"got {type(raw).__name__}; a non-string value is never coerced into model "
                    "input"
                )
            return raw

        # Two or more dynamic slots: resolve each field off THIS event source (all slots
        # reference the same source, guaranteed by the compiler) and return the values in slot
        # order for the scanner to fill.
        paths = getattr(spec, "constructed_field_paths", ()) or ()
        if paths:
            return tuple(_require_str(path) for path in paths)
        constructed = getattr(spec, "constructed_input", None) or ()
        if fld and any(content is None for _, content in constructed):
            return _require_str(fld)
        return _M.field_text(source, fld) if fld else self._text_for(source, spec)

    async def _apply_text_predicate_async(
        self,
        spec: AtomSpec,
        text: ExternalText | None,
        binding: str | None,
        event,
    ) -> Grounded:
        self.external_calls += 1
        if self._async_detector_accepts_event:
            result = await self._async_detector(spec, text or "", event)
        else:
            result = await self._async_detector(spec, text or "")
        return result if isinstance(result, Grounded) else Grounded(bool(result))

    def _ground_uncached(self, spec: AtomSpec, where, binding: str | None) -> Grounded:
        if spec.domain is AtomDomain.STATIC:
            return self._ground_static(spec, where, binding)
        if where is BOOT:
            return Grounded(False)
        return self._ground_event(spec, where, binding)

    def _text_for(self, source, spec: AtomSpec) -> str | None:
        fld = (
            spec.value_expr_key.split(":", 1)[1] if spec.value_expr_key.startswith("field:") else ""
        )
        if fld:
            return _M.field_text(source, fld)
        return _M.field_text(source, "content") if source is not None else None

    def _position_ok(self, spec: AtomSpec, event) -> bool:
        """Enforce indexed/sliced positional constraints against the event's per-surface
        ordinal; see :func:`~quarry.engine.events.in_window`."""
        return in_window(spec, event, self._static)

    def _ground_event(self, spec: AtomSpec, event, binding: str | None) -> Grounded:
        if spec.selector not in event.surface_tags:
            return Grounded(False)
        # Positional gate (indexed access `user_messages[0]` / `[-1]`, or a slice window
        # `messages[-4:]`): the atom fires only at events whose per-surface ordinal matches.
        if not self._position_ok(spec, event):
            return Grounded(False)
        return self._apply_predicate(spec, event.source, binding, event)

    def _ground_static(self, spec: AtomSpec, where, binding: str | None) -> Grounded:
        # Static atoms are grounded against the StaticEnvironment (boot step). An EVENT
        # ground call for a static atom (shouldn't happen) is False.
        if self._static is None:
            return Grounded(False)
        if spec.selector == CAPTURE_VALUE:
            if not isinstance(binding, str):
                return Grounded(False)
            if spec.predicate == "equals":
                return Grounded(binding in spec.patterns)
            return self._apply_text_predicate(replace(spec, parameter_names=()), binding, None)
        if spec.predicate == "tool_names_all":
            return Grounded(all(name in self._static.tools for name in spec.patterns))
        if spec.predicate == "surface_count":
            # Zero-width collection-size comparison: len(S)/count(S) <op> n against the
            # per-surface event count captured in the static environment.
            return Grounded(_M.compare(self.surface_size(spec), spec.op, spec.number))
        if spec.selector == "system_prompt":
            text = self._static.system_prompt or ""
            if spec.predicate == "equals":
                want = spec.patterns[0] if spec.patterns else ""
                return Grounded(text == want)
            return self._apply_text_predicate(spec, text, binding)
        if spec.selector == "tools":
            # Existential over tool definitions: true if ANY tool satisfies the predicate.
            # (The executor also supports per-tool grounding; this any-fold is the
            # single-atom convenience used when the whole rule is one tools-atom.)
            for tool in self._static.tools.values():
                if self._apply_predicate(spec, tool, binding, None).value:
                    return Grounded(True)
            return Grounded(False)
        if spec.selector == "messages" and self._is_external(spec):
            return self._apply_text_predicate(spec, "", binding)
        return Grounded(False)

    def _apply_predicate(self, spec: AtomSpec, source, binding, event) -> Grounded:
        """Apply a per-event predicate against ``source`` (a Message/ToolCall/dict)."""
        pred = spec.predicate
        if pred == "on_surface":
            # Surface-membership atom (count macro): true iff the event is on this
            # surface. The event-level surface gate already confirmed membership.
            return Grounded(True)
        if pred == "at_candidate":
            return Grounded(
                isinstance(binding, ElementBinding)
                and event is not None
                and event.position == binding.position
            )
        if pred == "position_cross":
            # `@a in fieldA <op> @b in fieldB` — positions from two DIFFERENT fields of this
            # same event. Each operand is "field\x1fgroup"; ground each field's text off the
            # source and take the operand's earliest match position, then compare.
            mods = _M.Modifiers(
                nocase=spec.string_semantics.case_insensitive,
                fullword=spec.string_semantics.fullword,
                base64=spec.string_semantics.base64,
                regex=spec.string_semantics.is_regex,
            )
            fa, ga = spec.patterns[0].split(_POSITION_FIELD_SEP, 1)
            fb, gb = spec.patterns[1].split(_POSITION_FIELD_SEP, 1)
            ta = _M.field_text(source, fa) if fa else _M.field_text(source, "content")
            tb = _M.field_text(source, fb) if fb else _M.field_text(source, "content")
            pa = self._group_position(ta, ga, mods) if ta is not None else None
            pb = self._group_position(tb, gb, mods) if tb is not None else None
            if pa is None or pb is None:
                return Grounded(False)
            if spec.number is None:
                return Grounded(_M.compare(pa, spec.op, pb))
            return Grounded(_M.compare(pa - pb, spec.op, spec.number))
        fld = (
            spec.value_expr_key.split(":", 1)[1] if spec.value_expr_key.startswith("field:") else ""
        )
        if pred == "is_null":
            v = _M.field_access(source, fld) if fld else None
            return Grounded(v is _M.MISSING or v is None)
        if pred == "equals":
            if spec.parameter_names and binding is not None:
                # A join candidate is the field's text (see captures.iter_capture_values),
                # so compare the same representation.
                text = _M.field_text(source, fld) if fld else None
                return Grounded(text is not None and text == binding)
            v = _M.field_access(source, fld) if fld else None
            want = spec.patterns[0] if spec.patterns else ""
            return Grounded(v not in (_M.MISSING, None) and str(v) == want)
        text = _M.field_text(source, fld) if fld else self._text_for(source, spec)
        return self._apply_text_predicate(spec, text, binding)

    def _apply_text_predicate(
        self, spec: AtomSpec, text: str | None, binding: str | None
    ) -> Grounded:
        pred = spec.predicate
        if pred in ("model_scan", "model_score", "external"):
            raise DetectorUnavailable(
                f"external atom {spec.atom_id} (detector={spec.detector!r}) is grounded "
                "only on the async path"
            )
        if text is None:
            return Grounded(False)
        mods = _M.Modifiers(
            nocase=spec.string_semantics.case_insensitive,
            fullword=spec.string_semantics.fullword,
            base64=spec.string_semantics.base64,
            regex=spec.string_semantics.is_regex,
        )
        if pred == "capture_match":
            # An explicit bind site: the value is one of the pattern's matches here.
            return Grounded(
                isinstance(binding, str)
                and any(binding in matches_for(p, text, mods, self._regex) for p in spec.patterns)
            )
        if pred == "contains_any":
            # Value-parameterized capture atom: the candidate binding is the pattern; for a
            # pinned element it is the candidate's field named by the parameter.
            if spec.parameter_names and isinstance(binding, ElementBinding):
                patterns = (binding.text(spec.parameter_names[0]),)
            elif spec.parameter_names and binding is not None:
                patterns = (binding,)
            else:
                patterns = spec.patterns
            if not patterns:
                return Grounded(False)
            return Grounded(_M.match_contains_any(text, patterns, mods, regex=self._regex))
        if pred == "contained_in":
            # A join to a pinned element that contains this event's field: the candidate's
            # field named by the parameter contains ``text``.
            return Grounded(
                isinstance(binding, ElementBinding)
                and _M.match_contains(
                    binding.text(spec.parameter_names[0]), text, mods, regex=self._regex
                )
            )
        if pred == "regex":
            return Grounded(
                any(
                    _M.match_regex(text, p, nocase=mods.nocase, regex=self._regex)
                    for p in spec.patterns
                )
            )
        if pred == "count":
            # A comparison with ``n`` is decided once more than ``n`` occur, so counting stops
            # there: a long field of matches costs at most that many.
            enough = max(0, math.floor(spec.number) + 1) if spec.number is not None else None
            total = 0
            for p in spec.patterns:
                left = None if enough is None else enough - total
                total += _M.count_occurrences(text, p, mods, regex=self._regex, limit=left)
                if enough is not None and total >= enough:
                    break
            return Grounded(_M.compare(total, spec.op, spec.number))
        if pred == "length":
            # A length atom always carries a numeric bound (from `len(field) <op> n`); the
            # Optional type on spec.number is never None here.
            return Grounded(_M.match_length(text, spec.op, spec.number))  # type: ignore
        if pred == "position":
            return Grounded(self._position(text, spec, mods))
        if pred == "position_abs":
            # Single match position vs an offset: `@a <op> N` → pos(a) <op> N. A missing
            # match is a non-match (no position → False).
            pa = self._group_position(text, spec.patterns[0], mods)
            return Grounded(pa is not None and _M.compare(pa, spec.op, spec.number))
        raise ValueError(f"oracle: unknown predicate {pred!r}")

    def _group_position(self, text: str, operand: str, mods) -> int | None:
        """Earliest match position of a match-position *operand group*.

        An operand is a NUL-joined pattern group (``atoms.POSITION_GROUP_SEP``): a plain
        ``@a`` is a 1-pattern group; a wildcard ``@a*`` expands to every ``$a*`` pattern.
        Returns the EARLIEST (min) position at which any pattern in the group matches, or
        ``None`` if none matches. For a single pattern this is exactly ``find_position``."""
        best: int | None = None
        for pat in operand.split(_POSITION_GROUP_SEP):
            pos = _M.find_position(text, pat, mods, regex=self._regex)
            if pos is not None and (best is None or pos < best):
                best = pos
        return best

    def _position(self, text: str, spec: AtomSpec, mods) -> bool:
        """Match-position ordering (`@a <op> @b`) / arithmetic (`(@a-@b) <op> n`).

        ``patterns == (a, b)`` (each a NUL-joined operand group; see
        :meth:`_group_position`); ``number is None`` → ordering ``pos(a) <op> pos(b)``;
        else arithmetic ``(pos(a) - pos(b)) <op> number``. A missing operand position is
        a non-match (mirrors Quarry returning None → False)."""
        pa = self._group_position(text, spec.patterns[0], mods)
        pb = self._group_position(text, spec.patterns[1], mods)
        if pa is None or pb is None:
            return False
        if spec.number is None:
            return _M.compare(pa, spec.op, pb)
        return _M.compare(pa - pb, spec.op, spec.number)
