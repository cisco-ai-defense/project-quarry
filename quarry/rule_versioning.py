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

"""Rule-to-runtime compatibility requirements.

This module deliberately depends only on the public Quarry version and
``packaging.version`` so scanner admission and direct compiler callers share one
strict interpretation of ``requires_quarry`` metadata.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from packaging.version import InvalidVersion, Version

from quarry._version import __version__

REQUIREMENT_METADATA_KEY = "requires_quarry"

CompatibilityIssueReason = Literal[
    "invalid_requirement",
    "runtime_below_minimum",
    "runtime_at_or_above_upper_bound",
]

_REQUIREMENT_PATTERN = re.compile(r">=(?P<lower>[^,\s]+)(?:,<(?P<upper>[^,\s]+))?")


class InvalidQuarryRequirement(ValueError):
    """A declared requirement is outside Quarry's supported requirement syntax."""


@dataclass(frozen=True, slots=True)
class QuarryRequirement:
    """A parsed inclusive lower bound and optional exclusive upper bound."""

    lower: Version
    upper: Version | None = None


def _json_safe(value: object) -> Any:
    """Return a JSON-safe representation without losing ordinary metadata values."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return repr(value)


@dataclass(frozen=True, slots=True)
class RuleCompatibilityIssue:
    """One explicit rule requirement that the running Quarry cannot admit."""

    source: str
    rule_name: str
    declaration: object
    running_version: str
    reason: CompatibilityIssueReason

    @property
    def message(self) -> str:
        """Render a concise diagnostic for scanner and compiler callers."""

        if self.reason == "invalid_requirement":
            explanation = (
                f"invalid {REQUIREMENT_METADATA_KEY} declaration {self.declaration!r}; "
                "expected '>=VERSION' or '>=VERSION,<VERSION' with lower < upper"
            )
        elif self.reason == "runtime_below_minimum":
            explanation = (
                f"{REQUIREMENT_METADATA_KEY} {self.declaration!r} does not admit running "
                f"Quarry {self.running_version} (below minimum)"
            )
        else:
            explanation = (
                f"{REQUIREMENT_METADATA_KEY} {self.declaration!r} does not admit running "
                f"Quarry {self.running_version} (at or above upper bound)"
            )
        return f"{self.source}: rule {self.rule_name!r}: {explanation}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the stable public issue contract."""

        return {
            "source": self.source,
            "rule_name": self.rule_name,
            "declaration": _json_safe(self.declaration),
            "running_version": self.running_version,
            "reason": self.reason,
        }


def parse_quarry_requirement(declaration: object) -> QuarryRequirement:
    """Parse exactly ``>=VERSION`` or ``>=VERSION,<VERSION``.

    The surrounding syntax is intentionally narrower than a PEP 440 specifier;
    only each endpoint uses :class:`packaging.version.Version` semantics.
    """

    if not isinstance(declaration, str):
        raise InvalidQuarryRequirement(
            f"{REQUIREMENT_METADATA_KEY} must be a string, got {type(declaration).__name__}"
        )

    match = _REQUIREMENT_PATTERN.fullmatch(declaration)
    if match is None:
        raise InvalidQuarryRequirement(
            f"expected '>=VERSION' or '>=VERSION,<VERSION', got {declaration!r}"
        )

    try:
        lower = Version(match.group("lower"))
        upper_text = match.group("upper")
        upper = Version(upper_text) if upper_text is not None else None
    except InvalidVersion as exc:
        raise InvalidQuarryRequirement(str(exc)) from exc

    if upper is not None and lower >= upper:
        raise InvalidQuarryRequirement(f"lower bound {lower} must be less than upper bound {upper}")
    return QuarryRequirement(lower=lower, upper=upper)


def check_rule_compatibility(
    rule: object,
    *,
    source: str | None = None,
    running_version: str | Version = __version__,
) -> RuleCompatibilityIssue | None:
    """Return an issue for one explicit incompatible declaration, if present."""

    metadata = getattr(rule, "metadata", None)
    if not isinstance(metadata, Mapping) or REQUIREMENT_METADATA_KEY not in metadata:
        return None

    declaration = metadata[REQUIREMENT_METADATA_KEY]
    running_text = str(running_version)
    running = Version(running_text)
    rule_name = str(getattr(rule, "name", ""))
    if source is None:
        source_file = getattr(rule, "source_file", None)
        source = str(source_file) if source_file is not None else "<programmatic>"

    try:
        requirement = parse_quarry_requirement(declaration)
    except InvalidQuarryRequirement:
        return RuleCompatibilityIssue(
            source=source,
            rule_name=rule_name,
            declaration=declaration,
            running_version=running_text,
            reason="invalid_requirement",
        )

    if running < requirement.lower:
        reason: CompatibilityIssueReason = "runtime_below_minimum"
    elif requirement.upper is not None and running >= requirement.upper:
        reason = "runtime_at_or_above_upper_bound"
    else:
        return None

    return RuleCompatibilityIssue(
        source=source,
        rule_name=rule_name,
        declaration=declaration,
        running_version=running_text,
        reason=reason,
    )


__all__ = [
    "QuarryRequirement",
    "CompatibilityIssueReason",
    "InvalidQuarryRequirement",
    "REQUIREMENT_METADATA_KEY",
    "RuleCompatibilityIssue",
    "check_rule_compatibility",
    "parse_quarry_requirement",
]
