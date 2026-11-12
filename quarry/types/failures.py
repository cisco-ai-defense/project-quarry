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

"""Stable public failure contracts, independent of scan execution."""

from dataclasses import dataclass
from enum import StrEnum, auto, unique


@unique
class ScanFailureCategory(StrEnum):
    """Stable public scan-failure categories; diagnostic text is not a classifier."""

    INVALID_INPUT = auto()
    RULE_SELECTION = auto()
    INPUT_READ = auto()
    PREPARATION = auto()
    MODEL = auto()
    RESOURCE = auto()
    EVALUATION = auto()
    TIMEOUT = auto()
    SKIPPED = auto()
    TRUNCATED_INPUT = auto()
    EVALUATION_LIMIT = auto()
    UNKNOWN = auto()


def _category_from_metadata(value: object) -> ScanFailureCategory:
    """Ignore invalid attribution without coercing arbitrary caller-owned objects."""
    if type(value) is ScanFailureCategory:
        return value
    if type(value) is str:
        try:
            return ScanFailureCategory(value)
        except ValueError:
            pass
    return ScanFailureCategory.UNKNOWN


@dataclass(frozen=True, slots=True)
class ScanFailure:
    """One machine-readable failure, optionally attributed to a selected rule.

    The record contains no exception text or provider details. Batch rule names use the
    same ``conv_N:rule`` provenance as matches and evaluation errors.
    """

    category: ScanFailureCategory
    rule_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", ScanFailureCategory(self.category))
        if self.rule_name is not None and not isinstance(self.rule_name, str):
            raise TypeError("failure rule_name must be a string or None")

    def to_dict(self) -> dict[str, str]:
        """Return stable JSON fields without copying arbitrary exception metadata."""
        result = {"category": self.category.value}
        if self.rule_name is not None:
            result["rule_name"] = self.rule_name
        return result
