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

"""Keys that name a computation by its inputs."""

import dataclasses
import hashlib
import math
import threading
import uuid
import weakref
from collections.abc import Callable, Iterable, Mapping
from typing import Any

_new_hash = hashlib.sha256


def input_key(*parts: Any) -> bytes:
    """Return the digest of ``parts``, the inputs a result depends on.

    A cache keyed this way holds no input, only its digest. Parts are text, numbers,
    booleans, None, sequences, mappings and dataclasses; a digest (``bytes``) or an object
    that carries its own (``.digest``, such as ``Messages``) stands for its input, so a long
    conversation is never serialized again to name it. Any other value is refused, so an
    input that cannot be named exactly never shares a result.
    """

    digest = _new_hash()
    _feed(digest.update, parts)
    return digest.digest()


def combine_digests(kind: str, digests: Iterable[bytes]) -> bytes:
    """Return the digest of a ``kind`` of composite from its members' digests, in order, so a
    composite is named without reading its members again."""

    digest = _new_hash()
    _feed(digest.update, kind)
    for member in digests:
        _feed(digest.update, member)
    return digest.digest()


def _feed(update: Callable[[bytes], object], value: Any) -> None:
    """Feed ``value`` to a hash in one pass: each value is tagged with its kind and, for text
    and collections, its length, so no two inputs share an encoding, and text is hashed as
    it is rather than escaped first."""

    if value is None:
        update(b"N")
    elif value is True:
        update(b"T")
    elif value is False:
        update(b"F")
    elif isinstance(value, str):
        # Text may hold a lone surrogate, which a model or a pattern can still be given.
        encoded = value.encode("utf-8", "surrogatepass")
        update(b"s%d:" % len(encoded))
        update(encoded)
    elif isinstance(value, int):
        update(b"i%d;" % value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{value!r} is not a cache input")
        update(b"f%r;" % value)
    elif isinstance(value, bytes):
        update(b"d%d:" % len(value))
        update(value)
    elif isinstance(value, (list, tuple)):
        update(b"l%d:" % len(value))
        for item in value:
            _feed(update, item)
    elif isinstance(value, Mapping):
        update(b"m%d:" % len(value))
        for name in sorted(value):
            _feed(update, name)
            _feed(update, value[name])
    else:
        digest = getattr(value, "digest", None)
        if isinstance(digest, bytes):
            # A digest names its own kind, so it stands for its value as it is.
            _feed(update, digest)
        elif callable(getattr(value, "to_openai_dict", None)):
            _feed(update, value.to_openai_dict())
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            # The class is part of the value, so two classes with equal fields differ.
            kind = type(value)
            update(b"c")
            _feed(update, f"{kind.__module__}.{kind.__qualname__}")
            fields = dataclasses.fields(value)
            _feed(update, {field.name: getattr(value, field.name) for field in fields})
        else:
            raise TypeError(f"{type(value).__name__} is not a cache input")


_identities: dict[int, str] = {}
# Objects that cannot be weakly referenced (None, built-in values) are kept, so their id is
# never reused for another object.
_kept: dict[int, object] = {}
_identities_lock = threading.Lock()


def object_identity(value: object) -> str:
    """Name ``value``, such as a configured adapter or prober, as a cache input by a token no
    other live object shares, since its configuration is not visible in its class."""

    key = id(value)
    with _identities_lock:
        token = _identities.get(key)
        if token is None:
            token = _identities[key] = uuid.uuid4().hex
            try:
                # The id can be reused once the object is gone, so its token goes with it.
                weakref.finalize(value, _identities.pop, key, None)
            except TypeError:
                _kept[key] = value
        return token
