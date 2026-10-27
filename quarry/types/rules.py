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

"""Rule related types."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from quarry.types.dsl import ConditionNode
from quarry.types.exceptions import ConversationValidationError, IncompleteScanError
from quarry.types.parameters import ParameterSpec
from quarry.types.results import CollectionResult

# Whether a scan's verdict is authoritative, rejected as invalid input, or incomplete
# (some selected rule did not evaluate to a trustworthy result).
ScanCompletionState = Literal["complete", "invalid", "incomplete"]
# False disables evidence, "support" captures match support, and True explores every atom.
ExplainLevel = bool | Literal["support"]


@dataclass
class DSLRule:
    """DSL rule."""

    name: str
    metadata: dict[str, str]
    condition: ConditionNode | None
    strings: dict[str, Any] | None = None
    source_file: Path | None = None
    parameters: tuple[ParameterSpec, ...] = ()
    collect: ConditionNode | None = None

    def __post_init__(self):
        if self.strings is None:
            self.strings = {}

    @property
    def description(self) -> str:
        """Get metadata description."""
        return self.metadata.get("description", "")

    @property
    def severity(self) -> str:
        """Get metadata severity."""
        return self.metadata.get("severity", "MEDIUM")

    def executable_nodes(self) -> tuple[ConditionNode, ...]:
        """Return rule expressions in evaluation order."""
        return tuple(node for node in (self.condition, self.collect) if node is not None)


@dataclass
class ScanResult:
    """Scan result for the DSL."""

    file_path: Path | None
    rule_matches: list[str]
    scan_time: float
    error: str | None = None
    # Whole-conversation input validation failure (distinct from per-rule errors): the scan
    # never ran. Drives the "invalid" completion state.
    validation_error: str | None = None
    was_truncated: bool = False
    truncation_info: dict[str, str] | None = None
    timed_out_rules: list[str] | None = None
    skipped_rules: list[str] | None = None
    # Per-rule evaluation failures {rule_name: error message} — rules that raised during
    # evaluation and so produced no trustworthy verdict. Recording them lets the guard fail
    # closed rather than treat the rule as a non-match.
    evaluation_errors: dict[str, str] | None = None
    rule_timings: dict[str, float] | None = None
    evaluation_warnings: list[str] | None = None
    # Evidence for matched and non-matched rules when explain is enabled.
    # Kept loose so this data module stays free of an engine import.
    explanations: list[Any] | None = None
    policy_digest: str | None = None
    parameter_digest: str | None = None
    parameter_evidence: dict[str, dict[str, object]] | None = None
    outputs: dict[str, CollectionResult[Any]] = field(default_factory=dict)
    # Canonical per-rule runtime evidence. Concrete values are
    # ``quarry.engine.contracts.WitnessTrace``; kept loose here so this data module remains
    # independent of the execution engine. Explanations and visualization both consume these
    # same immutable objects.
    rule_traces: dict[str, Any] | None = None
    # Per decomposed rule, the decision-tree checks the scan grounded, even when the rule
    # timed out or raised. Concrete values are ``quarry.engine.contracts.ScanRecord``.
    decision_paths: dict[str, Any] | None = None

    @property
    def detailed_matches(self) -> list[Any] | None:
        """The explanations of the rules that matched."""
        matched = [evidence for evidence in self.explanations or () if evidence.matched]
        return matched or None

    @property
    def completion_state(self) -> ScanCompletionState:
        """Whether this result is authoritative, invalid, or incomplete."""
        if self.validation_error is not None:
            return "invalid"
        if (
            self.error is not None
            or self.was_truncated
            or bool(self.timed_out_rules)
            or bool(self.skipped_rules)
            or bool(self.evaluation_errors)
            or bool(self.evaluation_warnings)
        ):
            return "incomplete"
        return "complete"

    @property
    def is_complete(self) -> bool:
        """True only when every selected rule evaluated authoritatively."""
        return self.completion_state == "complete"

    @property
    def completion_issues(self) -> list[str]:
        """Deterministic, human-readable reasons this scan is not complete."""
        issues: list[str] = []
        if self.validation_error is not None:
            issues.append(f"invalid conversation: {self.validation_error}")
        elif self.evaluation_errors:
            issues.append("rule evaluation failed: " + ", ".join(self.evaluation_errors.keys()))
        elif self.error is not None:
            issues.append(self.error)

        if self.timed_out_rules:
            issues.append(f"rules timed out: {', '.join(self.timed_out_rules)}")
        if self.skipped_rules:
            issues.append(f"rules skipped: {', '.join(self.skipped_rules)}")
        if self.was_truncated:
            issues.append("conversation was truncated")
        if self.evaluation_warnings:
            issues.append("evaluation was limited")

        return issues

    def raise_for_status(self) -> None:
        """Raise a typed error unless this result is authoritative (fail closed).

        ``complete`` → returns. ``invalid`` → :class:`ConversationValidationError`.
        ``incomplete`` → :class:`IncompleteScanError`. Public guards call this so an
        incomplete scan is never mistaken for a clean pass."""
        if self.is_complete:
            return
        if self.completion_state == "invalid":
            raise ConversationValidationError(
                self.validation_error or self.error or "Invalid conversation",
                scan_result=self,
            )
        raise IncompleteScanError(self)
