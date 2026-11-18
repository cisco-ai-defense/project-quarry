# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Bounded weak text lineage, independent of input storage and model execution.

One occurrence may have different text variants. Preserving copies share a variant;
slices keep coordinates within it, including nested slices. A transformed full text
starts its own axis. Only keyed digests, weak references and integer ranges survive.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import weakref
from dataclasses import dataclass, field
from typing import Protocol

from quarry.integrations.read_coverage import ReadInterval, _validate_interval


class ReadIdentityError(ValueError):
    """Accounting cannot establish validated input identity."""


class ReadIdentityUnavailable(ReadIdentityError):
    """The input cannot be owned weakly, or its lineage is unknown."""


class ReadIdentityLimitError(ReadIdentityError):
    """Independent identity storage or text inspection work is exhausted."""


class ReadIdentityOwnerError(ReadIdentityError):
    """The identity lifetime is closed or belongs to another process."""


class TokenAxisObserver(Protocol):
    def observe_axis(self, source: bytes, fingerprint: bytes, length: int) -> None: ...

    def reserve_text(self, length: int) -> bool: ...


@dataclass(frozen=True, slots=True)
class IdentityTotals:
    owners: int
    reservations: int
    work: int
    limited: bool
    closed: bool
    range_reservations: int


@dataclass(slots=True)
class _Identity:
    owner: weakref.ReferenceType = field(repr=False)
    occurrence: bytes = field(repr=False)
    field_origin: bytes = field(repr=False)
    interval: ReadInterval | None = None
    fingerprint: bytes | None = field(default=None, repr=False)
    unknown: bool = False
    axis_origin: bytes | None = field(default=None, repr=False)
    axis_length: int | None = None
    projection: tuple[ReadInterval, ...] | None = field(default=None, repr=False)


class ReadTextRegistry:
    """Own no input, and never silently recycle or evict an admitted occurrence.

    Calls receive text already read by their owner. Hashing is bounded before traversal,
    uses fixed-size chunks and never stores the text. Work and owner reservations are
    cumulative. Exhaustion disables later identity admission; losing lineage cannot turn
    a later child into an apparently validated independent input.
    """

    def __init__(
        self,
        *,
        max_owners: int = 8192,
        max_work: int = 1_000_000,
        max_ranges: int = 16_384,
        token_axes: TokenAxisObserver | None = None,
    ) -> None:
        for value, ceiling in [
            (max_owners, 262_144),
            (max_work, 10_000_000),
            (max_ranges, 262_144),
        ]:
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError("invalid logical input identity limit")
        self.max_owners = max_owners
        self.max_work = max_work
        self.max_ranges = max_ranges
        self._pid = os.getpid()
        # Allocating weak metadata may trigger cyclic GC and an existing owner's weak
        # callback on this same thread. Cleanup must be able to re-enter the table lock.
        self._lock = threading.RLock()
        self._key = secrets.token_bytes(32)
        self._entries: dict[int, _Identity] = {}
        self._reservations = self._work = 0
        self._range_reservations = 0
        self._limited = self._closed = False
        self._token_axes = None if token_axes is None else weakref.ref(token_axes)

    def _axis(self, source, fingerprint, length):
        target = None if self._token_axes is None else self._token_axes()
        if target is not None:
            target.observe_axis(source, fingerprint, length)

    def token_fingerprint(self, text: str) -> bytes | None:
        """Hash an explicit host registration against its independent token-work budget."""
        self._owner()
        if type(text) is not str:
            return None
        with self._lock:
            target = None if self._token_axes is None else self._token_axes()
            if self._closed or target is None or not target.reserve_text(len(text) + 1):
                return None
            digest = hmac.new(self._key, b"text-v1\0", hashlib.sha256)
            for index in range(0, len(text), 4096):
                digest.update(text[index : index + 4096].encode("utf-8", "surrogatepass"))
            return digest.digest()

    def _owner(self) -> None:
        if self._pid != os.getpid():
            raise ReadIdentityOwnerError("logical input identity belongs to another process")

    def _available(self) -> None:
        if self._closed:
            raise ReadIdentityOwnerError("logical input identity is closed")
        if self._limited:
            raise ReadIdentityLimitError("logical input identity limit exceeded")

    def _charge(self, amount: int) -> None:
        if amount > self.max_work - self._work:
            self._work = self.max_work
            self._limited = True
            raise ReadIdentityLimitError("logical input identity limit exceeded")
        self._work += amount

    def _forget(self, identity: int, reference: weakref.ReferenceType) -> None:
        if self._pid != os.getpid():
            return
        with self._lock:
            entry = self._entries.get(identity)
            if entry is not None and entry.owner is reference:
                del self._entries[identity]

    def _entry(self, owner: object) -> _Identity:
        self._charge(1)
        identity = id(owner)
        entry = self._entries.get(identity)
        if entry is not None and entry.owner() is owner:
            return entry
        if self._reservations >= self.max_owners:
            self._limited = True
            raise ReadIdentityLimitError("logical input identity limit exceeded")
        registry = weakref.ref(self)

        def forget(reference):
            current = registry()
            if current is not None:
                current._forget(identity, reference)

        try:
            reference = weakref.ref(owner, forget)
        except TypeError:
            raise ReadIdentityUnavailable("logical input identity needs a weak owner") from None
        occurrence = secrets.token_bytes(32)
        entry = _Identity(reference, occurrence, occurrence)
        self._entries[identity] = entry
        self._reservations += 1
        return entry

    def _fingerprint(self, text: str) -> bytes:
        return self._fingerprint_parts((text,))[0]

    def _fingerprint_parts(self, texts: tuple[str, ...]) -> tuple[bytes, int]:
        if type(texts) is not tuple:
            raise ValueError("logical input identity needs exact text parts")
        self._charge(len(texts) + 1)
        digest = hmac.new(self._key, b"text-v1\0", hashlib.sha256)
        length = 0
        for text in texts:
            if type(text) is not str:
                raise ValueError("logical input identity needs exact text")
            self._charge(len(text))
            length += len(text)
            # No full-size encoded or joined copy, including astral/lone-surrogate text.
            for index in range(0, len(text), 4096):
                digest.update(text[index : index + 4096].encode("utf-8", "surrogatepass"))
        return digest.digest(), length

    def _observe(self, entry: _Identity, text: str) -> ReadInterval:
        return self._observe_parts(entry, (text,))

    def _observe_parts(self, entry: _Identity, texts: tuple[str, ...]) -> ReadInterval:
        if entry.unknown:
            raise ReadIdentityUnavailable("logical input identity lineage is unavailable")
        fingerprint, length = self._fingerprint_parts(texts)
        if entry.interval is not None and entry.fingerprint == fingerprint:
            return entry.interval
        source = hmac.digest(self._key, b"variant-v1\0" + entry.occurrence + fingerprint, "sha256")
        interval = ReadInterval(source, "characters", 0, length)
        entry.interval, entry.fingerprint = interval, fingerprint
        entry.axis_origin, entry.axis_length = entry.occurrence, length
        entry.projection = None
        self._axis(source, fingerprint, length)
        return interval

    def observe(self, owner: object, text: str) -> ReadInterval:
        """Name this visible full text, or preserve an already established slice axis."""
        self._owner()
        with self._lock:
            self._available()
            return self._export(self._observe(self._entry(owner), text))

    def observe_parts(self, owner: object, texts: tuple[str, ...]) -> ReadInterval:
        """Observe ordered text parts without populating or joining native content caches."""
        self._owner()
        with self._lock:
            self._available()
            return self._export(self._observe_parts(self._entry(owner), texts))

    def observe_ranges(self, owner: object, texts: tuple[str, ...]) -> tuple[ReadInterval, ...]:
        """Read exact original ranges, preserving disjoint projected fields and cuts."""
        self._owner()
        with self._lock:
            self._available()
            entry = self._entry(owner)
            interval = self._observe_parts(entry, texts)
            ranges = entry.projection if entry.projection is not None else (interval,)
            self._charge(len(ranges))
            return tuple(self._export(item) for item in ranges)

    def window_ranges(
        self, owner: object, texts: tuple[str, ...], start: int, end: int
    ) -> tuple[ReadInterval, ...]:
        """Read a selected native text window through its exact original projection."""
        self._owner()
        with self._lock:
            self._available()
            entry = self._entry(owner)
            interval = self._observe_parts(entry, texts)
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start <= end <= interval.end - interval.start
            ):
                raise ValueError("invalid logical input text window")
            ranges = entry.projection
            if ranges is None:
                return (
                    ReadInterval(
                        interval.source, "characters", interval.start + start, interval.start + end
                    ),
                )
            self._charge(len(ranges) + 1)
            selected = self._cut_projection(ranges, start, end)
            return tuple(self._export(item) for item in selected)

    def known(self, owner: object) -> bool:
        """Test weak admission without admitting a formerly untracked native variant."""
        self._owner()
        with self._lock:
            self._available()
            self._charge(1)
            entry = self._entries.get(id(owner))
            return entry is not None and entry.owner() is owner

    def fingerprint_window(self, texts: tuple[str, ...], start: int, end: int) -> bytes:
        """Hash a window without allocating a joined or whole selected text copy."""
        self._owner()
        with self._lock:
            self._available()
            if type(texts) is not tuple or type(start) is not int or type(end) is not int:
                raise ValueError("invalid logical input text window")
            self._charge(len(texts) + 1)
            if any(type(text) is not str for text in texts):
                raise ValueError("invalid logical input text window")
            if not 0 <= start <= end <= sum(map(len, texts)):
                raise ValueError("invalid logical input text window")
            self._charge(end - start)
            digest = hmac.new(self._key, b"text-v1\0", hashlib.sha256)
            position = 0
            for text in texts:
                low, high = max(0, start - position), min(len(text), end - position)
                for index in range(low, high, 4096):
                    digest.update(
                        text[index : min(index + 4096, high)].encode("utf-8", "surrogatepass")
                    )
                position += len(text)
                if position >= end:
                    break
            return digest.digest()

    def field_interval(
        self, owner: object, labels: tuple[str | None, ...], text: str
    ) -> ReadInterval:
        """Name an immutable canonical field without retaining its path, ID or value."""
        self._owner()
        with self._lock:
            self._available()
            if type(labels) is not tuple or len(labels) > 3:
                raise ValueError("invalid logical input field")
            self._charge(len(labels) + 1)
            entry = self._entry(owner)
            if entry.unknown:
                raise ReadIdentityUnavailable("logical input identity lineage is unavailable")
            digest = hmac.new(self._key, b"field-v1\0" + entry.field_origin, hashlib.sha256)
            for label in labels:
                if label is None:
                    digest.update(b"n")
                elif type(label) is str:
                    digest.update(b"s" + self._fingerprint(label))
                else:
                    raise ValueError("invalid logical input field")
            fingerprint = self._fingerprint(text)
            digest.update(fingerprint)
            source = digest.digest()
            self._axis(source, fingerprint, len(text))
            return ReadInterval(source, "characters", 0, len(text))

    def origin_identity(self, text: str | None, values: tuple[bytes | int, ...]) -> bytes:
        """Digest a bounded, validated origin layout using opaque child identities."""
        self._owner()
        with self._lock:
            self._available()
            if type(values) is not tuple:
                raise ValueError("invalid logical input origin")
            self._charge(len(values) + 1)
            digest = hmac.new(self._key, b"origin-v1\0", hashlib.sha256)
            fingerprint = None if text is None else self._fingerprint(text)
            digest.update(b"p" if fingerprint is None else b"d" + fingerprint)
            for value in values:
                if type(value) is bytes and len(value) == 32:
                    digest.update(b"s" + value)
                elif type(value) is int and 0 <= value < 1 << 64:
                    digest.update(b"i" + value.to_bytes(8, "big"))
                else:
                    raise ValueError("invalid logical input origin")
            source = digest.digest()
            if text is not None:
                self._axis(source, fingerprint, len(text))
            return source

    def value_interval(self, scope: int, kind: str, text: str) -> ReadInterval:
        """Name an immutable authored scalar value within one request input scope."""
        self._owner()
        with self._lock:
            self._available()
            if type(scope) is not int or not 0 <= scope < 1 << 64 or type(kind) is not str:
                raise ValueError("invalid logical input value scope")
            fingerprint = self._fingerprint(text)
            source = hmac.digest(
                self._key,
                b"value-input-v1\0"
                + scope.to_bytes(8, "big")
                + self._fingerprint(kind)
                + fingerprint,
                "sha256",
            )
            self._axis(source, fingerprint, len(text))
            return ReadInterval(source, "characters", 0, len(text))

    def _reserve_ranges(self, count: int) -> None:
        self._charge(count + 1)
        if count > self.max_ranges - self._range_reservations:
            self._limited = True
            raise ReadIdentityLimitError("logical input identity range limit exceeded")
        self._range_reservations += count

    @staticmethod
    def _cut_projection(
        ranges: tuple[ReadInterval, ...], start: int, end: int
    ) -> tuple[ReadInterval, ...]:
        result = []
        position = 0
        for interval in ranges:
            length = interval.end - interval.start
            low, high = max(start - position, 0), min(end - position, length)
            if low < high:
                result.append(
                    ReadInterval(
                        interval.source, "characters", interval.start + low, interval.start + high
                    )
                )
            position += length
            if position >= end:
                break
        return tuple(result)

    def projected(
        self,
        target: object,
        texts: tuple[str, ...],
        ranges: tuple[ReadInterval, ...],
        *,
        empty_identity: bytes | None = None,
    ) -> None:
        """Install ordered exact ranges covering every character of a projected value."""
        self._owner()
        with self._lock:
            self._available()
            child = self._entry(target)
            child.unknown = True
            if type(ranges) is not tuple:
                raise ValueError("invalid logical input projection")
            self._reserve_ranges(len(ranges))
            fingerprint, length = self._fingerprint_parts(texts)
            owned: list[ReadInterval] = []
            covered = 0
            for interval in ranges:
                _validate_interval(interval)
                if interval.unit != "characters" or interval.start == interval.end:
                    raise ValueError("invalid logical input projection")
                covered += interval.end - interval.start
                if (
                    owned
                    and owned[-1].source == interval.source
                    and owned[-1].end == interval.start
                ):
                    previous = owned.pop()
                    owned.append(
                        ReadInterval(previous.source, "characters", previous.start, interval.end)
                    )
                else:
                    owned.append(
                        ReadInterval(interval.source, "characters", interval.start, interval.end)
                    )
            if covered != length:
                raise ValueError("invalid logical input projection")
            values = tuple(
                value
                for interval in owned
                for value in (interval.source, interval.start, interval.end)
            )
            if not values and empty_identity is not None:
                if type(empty_identity) is not bytes or len(empty_identity) != 32:
                    raise ValueError("invalid logical input projection")
                values = (empty_identity,)
            child.occurrence = self.origin_identity(None, values)
            source = hmac.digest(
                self._key, b"variant-v1\0" + child.occurrence + fingerprint, "sha256"
            )
            child.interval, child.fingerprint = (
                ReadInterval(source, "characters", 0, length),
                fingerprint,
            )
            child.axis_origin, child.axis_length = child.occurrence, length
            child.projection, child.unknown = tuple(owned), False

    def _export(self, interval: ReadInterval) -> ReadInterval:
        # A host can bypass frozen dataclasses with object.__setattr__. Never expose
        # the internal record shared by a parent and its preserving copies.
        self._charge(1)
        return ReadInterval(interval.source, "characters", interval.start, interval.end)

    def inspect(self, amount: int) -> None:
        """Charge a native adapter's bounded metadata walk before it inspects references."""
        self._owner()
        if type(amount) is not int or amount < 0:
            raise ValueError("invalid logical input identity work")
        with self._lock:
            self._available()
            self._charge(amount)

    def copied(self, source: object, target: object) -> None:
        """Register a preserving copy without reading either owner's fields or text."""
        self._owner()
        with self._lock:
            self._available()
            parent, child = self._entry(source), self._entry(target)
            child.occurrence = parent.occurrence
            child.field_origin = parent.field_origin
            child.interval = parent.interval
            child.fingerprint = parent.fingerprint
            child.unknown = parent.unknown
            child.axis_origin, child.axis_length = parent.axis_origin, parent.axis_length
            child.projection = parent.projection

    def unavailable(self, target: object) -> None:
        """Prevent an untracked text rewrite from masquerading as a new exact input."""
        self._owner()
        with self._lock:
            self._available()
            self._entry(target).unknown = True

    def sliced(self, source: object, target: object, start: int, end: int, text: str) -> None:
        """Register ``text[start:end]`` in the source's established coordinate axis."""
        self._owner()
        if (
            type(text) is not str
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= len(text)
        ):
            raise ValueError("invalid logical input identity slice")
        with self._lock:
            self._available()
            parent, child = self._entry(source), self._entry(target)
            child.field_origin = parent.field_origin
            try:
                interval = self._observe(parent, text)
            except ReadIdentityUnavailable:
                child.unknown = True
                raise
            child.unknown = True
            # Charge before creating even the temporary selected text.
            self._charge(end - start + 1)
            selected = self._fingerprint(text[start:end])
            low, high = interval.start + start, interval.start + end
            assert parent.axis_origin is not None and parent.axis_length is not None
            child.occurrence = (
                parent.axis_origin
                if low == 0 and high == parent.axis_length
                else hmac.digest(
                    self._key,
                    b"slice-region-v1\0"
                    + interval.source
                    + low.to_bytes(8, "big")
                    + high.to_bytes(8, "big"),
                    "sha256",
                )
            )
            child.interval = ReadInterval(interval.source, "characters", low, high)
            child.fingerprint, child.unknown = selected, False
            child.axis_origin, child.axis_length = parent.axis_origin, parent.axis_length
            if parent.projection is not None:
                self._reserve_ranges(len(parent.projection))
                child.projection = self._cut_projection(parent.projection, start, end)
            else:
                child.projection = None

    def derived(self, source: object, target: object, text: str) -> None:
        """Register a full transformed variant belonging to the same occurrence."""
        self.derived_parts(source, target, (text,))

    def derived_parts(self, source: object, target: object, texts: tuple[str, ...]) -> None:
        """Register concatenated transformed text without allocating a joined input."""
        self._owner()
        with self._lock:
            self._available()
            parent, child = self._entry(source), self._entry(target)
            child.occurrence, child.unknown = parent.occurrence, parent.unknown
            child.field_origin = parent.field_origin
            child.interval = child.fingerprint = None
            child.projection = None
            self._observe_parts(child, texts)

    def constructed(
        self, frame: object, constructor: object, scope: tuple, target: object, text: str
    ) -> None:
        """Name an authored turn by weak evaluation frame, expression and binding scope.

        A temporary computed-source counter is unique only inside its frame. Mapping
        expressions in the scope are weak owners too; neither raw pointer reuse nor
        retaining compiler literals may establish an accounting identity.
        """
        self._owner()
        if type(scope) is not tuple or len(scope) > 256:
            raise ValueError("invalid logical input construction scope")
        with self._lock:
            self._available()
            self._charge(len(scope) + 1)
            child = self._entry(target)
            child.unknown = True
            frame_entry, site = self._entry(frame), self._entry(constructor)
            digest = hmac.new(
                self._key,
                b"construction-v1\0" + frame_entry.occurrence + site.occurrence,
                hashlib.sha256,
            )
            for value in scope:
                if type(value) is int:
                    if not 0 <= value < 1 << 64:
                        raise ValueError("invalid logical input construction scope")
                    digest.update(b"i" + value.to_bytes(8, "big"))
                elif type(value) is str:
                    if value != "computed-source":
                        raise ValueError("invalid logical input construction scope")
                    digest.update(b"computed-source\0")
                else:
                    digest.update(b"n" + self._entry(value).occurrence)
            child.occurrence, child.unknown = digest.digest(), False
            child.field_origin = child.occurrence
            child.interval = child.fingerprint = None
            child.projection = None
            self._observe(child, text)

    def snapshot(self) -> IdentityTotals:
        self._owner()
        with self._lock:
            return IdentityTotals(
                len(self._entries),
                self._reservations,
                self._work,
                self._limited,
                self._closed,
                self._range_reservations,
            )

    def close(self) -> None:
        if self._pid != os.getpid():
            return
        with self._lock:
            self._closed = True
            self._entries.clear()
            self._key = b""
