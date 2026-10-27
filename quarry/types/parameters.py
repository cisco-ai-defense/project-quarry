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

"""Data contracts for values bound after Quarry source parsing."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

ParameterType = Literal["number", "text"]
ParameterBinding = Literal["policy", "scan"]
ParameterValue = int | float | str


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """One typed value that remains a slot in a compiled rule."""

    name: str
    type: ParameterType
    binding: ParameterBinding


class ParameterValueError(ValueError):
    """A supplied parameter map does not match its compiled schema."""


def validate_parameter_values(
    specs: Iterable[ParameterSpec],
    values: Mapping[str, object] | None,
    *,
    binding: ParameterBinding | None = None,
) -> dict[str, ParameterValue]:
    """Validate exact names and scalar types for one schema or binding subset."""

    selected = {spec.name: spec for spec in specs if binding is None or spec.binding == binding}
    supplied = dict(values or {})
    missing = sorted(set(selected) - set(supplied))
    extra = sorted(set(supplied) - set(selected))
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing: {', '.join(missing)}")
        if extra:
            parts.append(f"unknown: {', '.join(extra)}")
        raise ParameterValueError(
            "parameter names do not match the rule (" + "; ".join(parts) + ")"
        )

    validated: dict[str, ParameterValue] = {}
    for name, spec in selected.items():
        value = supplied[name]
        if spec.type == "number":
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ParameterValueError(f"parameter {name!r} must be a number")
            if isinstance(value, float) and not math.isfinite(value):
                raise ParameterValueError(f"parameter {name!r} must be finite")
        elif not isinstance(value, str):
            raise ParameterValueError(f"parameter {name!r} must be text")
        validated[name] = value
    return validated


def text_sha256(value: str) -> str:
    """Return the SHA-256 hash of the value."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parameter_evidence(
    specs: Iterable[ParameterSpec], values: Mapping[str, ParameterValue]
) -> dict[str, dict[str, object]]:
    """Return auditable values without exposing text or guessable text hashes."""

    evidence: dict[str, dict[str, object]] = {}
    for spec in specs:
        value = values[spec.name]
        item: dict[str, object] = {"type": spec.type, "binding": spec.binding}
        if spec.type == "text":
            item["redacted"] = True
        else:
            item["value"] = value
        evidence[spec.name] = item
    return evidence


def parameter_digest(
    specs: Iterable[ParameterSpec], values: Mapping[str, ParameterValue], *, key: bytes
) -> str:
    """Digest effective values using keyed HMAC for text parameters."""

    if not isinstance(key, bytes) or len(key) < 16:
        raise ValueError("evidence_key must contain at least 16 bytes")
    identity: dict[str, dict[str, object]] = {}
    has_text = False
    for spec in specs:
        value = values[spec.name]
        if spec.type == "text":
            has_text = True
            identity[spec.name] = {
                "type": "text",
                "binding": spec.binding,
                "hmac_sha256": hmac.new(
                    key, str(value).encode("utf-8"), hashlib.sha256
                ).hexdigest(),
            }
        else:
            identity[spec.name] = {
                "type": "number",
                "binding": spec.binding,
                "value": value,
            }
    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return (
        hmac.new(key, canonical, hashlib.sha256).hexdigest()
        if has_text
        else hashlib.sha256(canonical).hexdigest()
    )
