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

"""Exception classes."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from quarry.rule_versioning import RuleCompatibilityIssue

if TYPE_CHECKING:  # avoid an import cycle (rules.py imports this module)
    from quarry.types.rules import ScanResult


class QuarryError(Exception):
    """Base exception for all Quarry errors."""

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ConfigurationError(QuarryError):
    """Raised when configuration is invalid."""


class RuleError(QuarryError):
    """Base class for rule-related errors."""


class RuleParseError(RuleError):
    """Raised when rule parsing fails."""

    def __init__(
        self,
        rule_name: str,
        line: int | None,
        message: str,
        source: str | None = None,
    ):
        self.rule_name = rule_name
        self.line = line
        self.reason = message
        self.source = source
        source_text = f" from '{source}'" if source is not None else ""
        details = {"rule_name": rule_name, "line": line}
        if source is not None:
            details["source"] = source
        super().__init__(
            f"Failed to parse rule '{rule_name}'{source_text}: {message}",
            details,
        )


class RuleLoadError(RuleError):
    """Raised when a configured rule source cannot be loaded — a missing rule path, a
    duplicate rule name across files, or a per-source load/parse failure. Fail closed at
    load rather than silently skipping a path or letting one rule shadow another by
    filesystem iteration order."""

    def __init__(self, source: str, phase: str, message: str):
        self.source = source
        self.phase = phase
        super().__init__(
            f"Failed to {phase} rule source '{source}': {message}",
            {"source": source, "phase": phase},
        )


class RuleValidationError(RuleError):
    """Raised when rule validation fails."""


class RuleCompatibilityError(RuleValidationError):
    """Raised when explicit rule requirements reject the running Quarry version."""

    def __init__(
        self,
        running_version: str,
        issues: Sequence[RuleCompatibilityIssue],
    ):
        self.running_version = running_version
        self.issues = tuple(issues)
        count = len(self.issues)
        noun = "issue" if count == 1 else "issues"
        summary = "; ".join(issue.message for issue in self.issues)
        super().__init__(
            f"{count} rule compatibility {noun} for running Quarry {running_version}: {summary}",
            {
                "error_type": "rule_compatibility",
                "running_version": running_version,
                "issues": [issue.to_dict() for issue in self.issues],
            },
        )


class RuleDependencyError(RuleError):
    """Raised when rule dependencies are missing."""

    def __init__(self, rule_name: str, missing_resource: str):
        self.rule_name = rule_name
        self.missing_resource = missing_resource
        super().__init__(
            f"Rule '{rule_name}' requires {missing_resource} which is not available",
            {"rule_name": rule_name, "missing_resource": missing_resource},
        )


class EvaluationError(QuarryError):
    """Raised when rule evaluation fails."""


class IncompleteScanError(EvaluationError):
    """Raised when a public guard cannot return an authoritative scan decision.

    Public guards **fail closed** on an incomplete scan (a rule that errored, timed out,
    was skipped, or ran against truncated input) rather than returning a partial match list
    that is indistinguishable from a clean pass. Carries the originating ``ScanResult`` so a
    caller can inspect what was and wasn't evaluated."""

    def __init__(self, scan_result: "ScanResult", message: str | None = None):
        self.scan_result = scan_result
        # ``result`` is a concise alias for callers handling several result-bearing errors.
        self.result = scan_result
        issues = scan_result.completion_issues
        details: dict[str, Any] = {
            "completion_state": scan_result.completion_state,
            "error": scan_result.error,
            "partial_matches": list(scan_result.rule_matches),
            "timed_out_rules": list(scan_result.timed_out_rules or []),
            "skipped_rules": list(scan_result.skipped_rules or []),
            "evaluation_errors": dict(scan_result.evaluation_errors or {}),
            "evaluation_warnings": list(scan_result.evaluation_warnings or []),
            "was_truncated": scan_result.was_truncated,
            "truncation_info": dict(scan_result.truncation_info or {}),
        }
        if scan_result.file_path is not None:
            details["file_path"] = str(scan_result.file_path)

        summary = message or (
            f"Scan incomplete: {'; '.join(issues)}"
            if issues
            else "Scan incomplete for an unknown reason"
        )
        super().__init__(summary, details)


class EvaluationTimeoutError(EvaluationError):
    """Raised when evaluation times out."""

    def __init__(self, rule_name: str, timeout: float):
        self.rule_name = rule_name
        self.timeout = timeout
        super().__init__(
            f"Rule '{rule_name}' exceeded timeout of {timeout}s",
            {"rule_name": rule_name, "timeout": timeout},
        )


class ResourceError(QuarryError):
    """Base class for resource-related errors."""


class ModelError(ResourceError):
    """Raised when ML model operations fail."""


class LLMError(ResourceError):
    """Raised when LLM operations fail."""


# The ``LLMError`` code of a refused model call: the model gave no usable answer.
MODEL_REFUSED = "model_refused"


class ValidationError(QuarryError):
    """Raised when input validation fails."""


class ConversationValidationError(ValidationError):
    """Raised when conversation format is invalid.

    Optionally carries the originating ``ScanResult`` (when raised from a fail-closed guard
    on an invalid conversation) so callers get the same result-bearing shape as
    :class:`IncompleteScanError`."""

    def __init__(
        self,
        message: str,
        details: dict | None = None,
        *,
        scan_result: "ScanResult | None" = None,
    ):
        self.scan_result = scan_result
        self.result = scan_result
        error_details: dict[str, Any] = dict(details or {})
        if scan_result is not None:
            error_details.update(
                {
                    "completion_state": scan_result.completion_state,
                    "validation_error": scan_result.validation_error,
                }
            )
            if scan_result.file_path is not None:
                error_details["file_path"] = str(scan_result.file_path)
        super().__init__(message, error_details)
