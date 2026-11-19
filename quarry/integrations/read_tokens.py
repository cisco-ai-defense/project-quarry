# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Validated host-supplied reference text tokens; no tokenizer or provider execution.

Offsets address Unicode characters in a canonical source text, independent of provider
templates. Positive-width token spans deliberately cannot include artificial boundary
tokens. Only keyed fingerprints, numeric offsets and bounded configuration survive.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from quarry.integrations.read_coverage import ReadInterval


@dataclass(frozen=True, slots=True)
class TextTokenBasis:
    """A host's stable reference vocabulary/configuration and semantic version.

    The host must supply genuine tokenizer results and version changes to vocabulary,
    normalization or offset conventions. Quarry checks binding and offset integrity;
    it does not independently execute or verify the declared tokenizer. Character
    positions are Unicode code points; zero-width template/BOS/EOS tokens are excluded.
    """

    name: str
    version: str

    def __post_init__(self):
        for value in (self.name, self.version):
            if type(value) is not str or not 1 <= len(value) <= 128:
                raise ValueError("invalid logical text-token basis")
            try:
                value.encode("utf-8")
            except UnicodeError:
                raise ValueError("invalid logical text-token basis") from None
            if any(ord(character) < 32 or ord(character) == 127 for character in value):
                raise ValueError("invalid logical text-token basis")


@dataclass(frozen=True, slots=True)
class TokenProjection:
    state: str
    intervals: tuple[ReadInterval, ...] = ()
    work: int = 0
    limited: bool = False


@dataclass(frozen=True, slots=True)
class _TokenMap:
    state: str
    length: int
    starts: tuple[int, ...] = ()
    ends: tuple[int, ...] = ()


class TextTokenAtlas:
    """Request-owned numeric maps for one explicitly declared basis per model alias."""

    def __init__(self, bases, *, max_axes, max_maps, max_spans, max_work):
        for value, ceiling in (
            (max_axes, 262_144),
            (max_maps, 65_536),
            (max_spans, 262_144),
            (max_work, 10_000_000),
        ):
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError("invalid logical text-token limit")
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._key = secrets.token_bytes(32)
        self.max_axes, self.max_maps = max_axes, max_maps
        self.max_spans, self.max_work = max_spans, max_work
        self._bases: dict[str, TextTokenBasis] = {}
        self._basis_ids: dict[str, bytes] = {}
        if type(bases) is not dict or len(bases) > 4096:
            raise ValueError("invalid logical text-token bases")
        for model, basis in bases.items():
            if (
                type(model) is not str
                or not 1 <= len(model) <= 512
                or type(basis) is not TextTokenBasis
            ):
                raise ValueError("invalid logical text-token bases")
            try:
                model.encode("utf-8")
            except UnicodeError:
                raise ValueError("invalid logical text-token bases") from None
            owned = TextTokenBasis(basis.name, basis.version)
            self._bases[model] = owned
            payload = owned.name.encode() + b"\0" + owned.version.encode()
            self._basis_ids[model] = hashlib.sha256(b"logical-text-basis-v1\0" + payload).digest()
        self._axes: dict[bytes, tuple[bytes, int]] = {}
        self._maps: dict[tuple[bytes, bytes], _TokenMap] = {}
        self._failed_models: set[str] = set()
        self._work = self._spans = 0
        self._limited = self._closed = False

    def _charge(self, amount):
        if amount > self.max_work - self._work:
            self._work = self.max_work
            self._limited = True
            return False
        self._work += amount
        return True

    def reserve_text(self, length: int) -> bool:
        if self._pid != os.getpid() or type(length) is not int or length < 0:
            return False
        # Called under the native identity lock. Token-map allocation can trigger
        # weak cleanup of an identity owner, so this edge must never wait in reverse.
        if not self._lock.acquire(blocking=False):
            return False
        try:
            return not self._closed and self._charge(length)
        finally:
            self._lock.release()

    def observe_axis(self, source: bytes, fingerprint: bytes, length: int):
        if self._pid != os.getpid():
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self._closed or not self._charge(1):
                return
            value = (fingerprint, length)
            existing = self._axes.get(source)
            if existing is not None:
                if existing != value:
                    self._limited = True
                return
            if len(self._axes) >= self.max_axes:
                self._limited = True
                return
            self._axes[source] = value
        finally:
            self._lock.release()

    def provide(self, model, fingerprint, length, offsets, *, state="measured"):
        if self._pid != os.getpid():
            return False
        with self._lock:
            if self._closed or type(model) is not str or model not in self._bases:
                return False
            if not self._charge(1):
                return False
            if (
                type(fingerprint) is not bytes
                or len(fingerprint) != 32
                or type(length) is not int
                or length < 0
            ):
                self._failed_models.add(model)
                return False
            key = (self._basis_ids[model], fingerprint)
            if key not in self._maps and len(self._maps) >= self.max_maps:
                self._limited = True
                return False
            if type(state) is not str:
                self._maps[key] = _TokenMap("failed", length)
                return False
            if state in {"failed", "cancelled", "unavailable"}:
                self._maps[key] = _TokenMap(state, length)
                return True
            if state != "measured" or type(offsets) is not tuple:
                self._maps[key] = _TokenMap("failed", length)
                return False
            count = len(offsets)
            if (
                count > self.max_spans
                or (key not in self._maps and count > self.max_spans - self._spans)
                or not self._charge(1 + 4 * count)
            ):
                self._limited = True
                return False
            starts, ends = [], []
            low = high = 0
            for span in offsets:
                if type(span) is not tuple or len(span) != 2:
                    self._maps[key] = _TokenMap("failed", length)
                    return False
                start, end = span
                if (
                    type(start) is not int
                    or type(end) is not int
                    or not 0 <= low <= start < end <= length
                    or end < high
                ):
                    self._maps[key] = _TokenMap("failed", length)
                    return False
                starts.append(start)
                ends.append(end)
                low, high = start, end
            proposed = _TokenMap("measured", length, tuple(starts), tuple(ends))
            previous = self._maps.get(key)
            if previous is not None:
                if previous != proposed:
                    # Conflicting registrations invalidate earlier readers too; never
                    # silently change their basis or preserve an obsolete measured zero.
                    self._maps[key] = _TokenMap("failed", length)
                    return False
                return True
            self._spans += count
            self._maps[key] = proposed
            return True

    def fail_model(self, model):
        if self._pid != os.getpid() or type(model) is not str:
            return
        with self._lock:
            if not self._closed and model in self._bases:
                self._failed_models.add(model)

    def project(self, model, state, ranges, *, max_work):
        if self._pid != os.getpid():
            return TokenProjection("unavailable")
        if (
            type(model) is not str
            or type(state) is not str
            or type(ranges) is not tuple
            or type(max_work) is not int
            or max_work < 0
        ):
            return TokenProjection("failed")
        with self._lock:
            if self._closed or model not in self._bases:
                return TokenProjection("unavailable")
            if model in self._failed_models:
                return TokenProjection("failed")
            if state not in {"measured", "partial"}:
                return TokenProjection(state)
            cost = 1 + len(ranges) * 4
            if cost > max_work:
                return TokenProjection("limit_exceeded", work=max_work)
            output = []
            missing = set() if state == "measured" else {"partial"}
            basis = self._basis_ids[model]
            for interval in ranges:
                axis = self._axes.get(interval.source)
                if axis is None:
                    missing.add("limit_exceeded" if self._limited else "unavailable")
                    continue
                fingerprint, length = axis
                mapping = self._maps.get((basis, fingerprint))
                if mapping is not None and (
                    mapping.state != "measured" or mapping.length != length
                ):
                    missing.add(mapping.state if mapping.state != "measured" else "failed")
                    continue
                if interval.start == interval.end:
                    continue
                if mapping is None:
                    if length:
                        missing.add("limit_exceeded" if self._limited else "unavailable")
                    continue
                if mapping.length != length or interval.end > length:
                    missing.add("failed")
                    continue
                charge = 2 * max(1, len(mapping.starts).bit_length())
                if charge > max_work - cost:
                    return TokenProjection(
                        "partial" if output else "limit_exceeded", tuple(output), max_work, True
                    )
                cost += charge
                start = bisect_right(mapping.ends, interval.start)
                end = bisect_left(mapping.starts, interval.end)
                if start < end:
                    source = hmac.digest(
                        self._key, b"token-axis-v1\0" + interval.source + basis, "sha256"
                    )
                    output.append(ReadInterval(source, "text_tokens", start, end))
            complete = not missing
            if complete:
                result = "measured"
            elif output:
                result = "partial"
            else:
                result = next(
                    (
                        item
                        for item in (
                            "limit_exceeded",
                            "failed",
                            "cancelled",
                            "unavailable",
                            "partial",
                        )
                        if item in missing
                    ),
                    "unavailable",
                )
            return TokenProjection(result, tuple(output), cost)

    def bases(self):
        if self._pid != os.getpid():
            return []
        with self._lock:
            return [
                {"model": model, "name": basis.name, "version": basis.version}
                for model, basis in sorted(self._bases.items())
            ]

    def basis_key(self, model):
        if self._pid != os.getpid():
            return None
        # Configuration is copied at construction and never mutated or returned.
        basis = self._bases.get(model)
        return None if basis is None else ("reference", basis.name, basis.version)

    def close(self):
        if self._pid != os.getpid():
            return
        with self._lock:
            self._closed = True
            self._axes.clear()
            self._maps.clear()
            self._failed_models.clear()
