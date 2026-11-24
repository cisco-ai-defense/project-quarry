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

"""Published, extensible response contracts for the Quarry MCP tools.

The service continues to build ordinary JSON-ready dictionaries. These types make the
stable portion of each response discoverable through MCP ``outputSchema`` and validate it
at the transport boundary without dropping additive fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, TypeAdapter, with_config
from typing_extensions import TypedDict

from quarry.decision_schema import (
    EXTENSIBLE_SCHEMA,
    DecisionPathPayload,
    DecisionTreePayload,
    JsonNumber,
    Sha256Digest,
)
from quarry.telemetry import (
    MAX_MODEL_ALIAS_CHARS,
    MAX_MODEL_OPERATION_CHARS,
    MAX_MODEL_USAGE_ALIASES,
    MAX_MODEL_USAGE_CALLS,
    MAX_MODEL_USAGE_RECORDS,
    MAX_SCAN_SECONDS,
)
from quarry.types.parameters import ParameterBinding

ReferenceAction = Literal["index", "search", "get", "rejected"]
ResponseVerbosity = Literal["compact", "full"]
CompletionState = Literal["complete", "invalid", "incomplete"]
FailureCategory = Literal[
    "invalid_input",
    "rule_selection",
    "input_read",
    "preparation",
    "model",
    "resource",
    "evaluation",
    "timeout",
    "skipped",
    "truncated_input",
    "evaluation_limit",
    "unknown",
]
DurationSeconds = Annotated[
    StrictFloat,
    Field(ge=0.0, le=MAX_SCAN_SECONDS, allow_inf_nan=False),
]
UsageCount = Annotated[StrictInt, Field(ge=0, le=MAX_MODEL_USAGE_CALLS)]
TelemetryModelAlias = Annotated[
    StrictStr,
    Field(min_length=1, max_length=MAX_MODEL_ALIAS_CHARS),
]
TelemetryOperation = Annotated[
    Literal["score", "classify", "complete", "input_tokens"],
    Field(max_length=MAX_MODEL_OPERATION_CHARS),
]


@with_config(EXTENSIBLE_SCHEMA)
class VersionEnvelope(TypedDict):
    """Versions shared by reference, compilation, and scan responses."""

    plugin_version: str
    quarry_version: str
    catalog_format_version: str
    language_version: str
    compiler_version: str


@with_config(EXTENSIBLE_SCHEMA)
class CatalogEntry(TypedDict):
    """Stable identity fields shared by catalog records and examples."""

    id: str
    kind: str
    name: NotRequired[str]
    summary: NotRequired[str]


@with_config(EXTENSIBLE_SCHEMA)
class RejectedCatalogEntry(TypedDict):
    """One documented spelling the language rejects."""

    source_form: str
    rejected_by: str
    replacement: NotRequired[str]


@with_config(EXTENSIBLE_SCHEMA)
class ModelAliasSummary(TypedDict):
    """Operations exposed by one trusted runtime model alias."""

    operations: list[str]
    display_name: NotRequired[Annotated[StrictStr, Field(min_length=1, max_length=100)]]
    description: NotRequired[str]


@with_config(EXTENSIBLE_SCHEMA)
class ReferenceResponse(VersionEnvelope):
    """Response from ``quarry_reference`` for any supported action."""

    action: ReferenceAction
    runtime_capabilities: dict[str, str]
    results: list[CatalogEntry | RejectedCatalogEntry]
    runtime_config_version: NotRequired[int]
    model_aliases: NotRequired[dict[str, ModelAliasSummary]]
    operator_precedence: NotRequired[list[dict[str, Any]]]
    query: NotRequired[str]
    total: NotRequired[int]
    offset: NotRequired[int]
    next_offset: NotRequired[int | None]
    verbosity: NotRequired[ResponseVerbosity]
    missing_ids: NotRequired[list[str]]
    filtered_ids: NotRequired[list[str]]


@with_config(EXTENSIBLE_SCHEMA)
class Diagnostic(TypedDict):
    """A bounded machine-readable diagnostic."""

    code: str
    message: NotRequired[str]
    severity: NotRequired[str]
    stage: NotRequired[str]
    rule_name: NotRequired[str]
    line: NotRequired[int]
    details: NotRequired[dict[str, Any]]


@with_config(EXTENSIBLE_SCHEMA)
class Advisory(TypedDict):
    """A non-fatal compilation or language warning."""

    code: str
    severity: str
    message: str
    suggestion: NotRequired[str]
    item_id: NotRequired[str]
    replacement: NotRequired[str | None]


@with_config(EXTENSIBLE_SCHEMA)
class ModelOperation(TypedDict):
    """One model alias and operation required by a rule."""

    model: str
    operation: str


@with_config(EXTENSIBLE_SCHEMA)
class CapabilityRequirements(TypedDict):
    """Runtime resources required by a compiled rule."""

    resource_groups: list[str]
    functions: list[str]
    model_operations: list[ModelOperation]
    models: list[str]
    text_resources: list[str]


@with_config(EXTENSIBLE_SCHEMA)
class ParameterSlot(TypedDict):
    """One declared rule parameter."""

    name: str
    type: str
    binding: str


@with_config(EXTENSIBLE_SCHEMA)
class CompileResponse(VersionEnvelope):
    """Response from ``quarry_compile`` and the scan compilation envelope."""

    rule_hash: Sha256Digest
    warnings: list[Advisory]
    valid: bool
    stage: str
    diagnostics: list[Diagnostic]
    required_capabilities: CapabilityRequirements
    rule_names: NotRequired[list[str]]
    rule_metadata: NotRequired[dict[str, dict[str, Any]]]
    estimated_complexity: NotRequired[str]
    rule_complexities: NotRequired[dict[str, str]]
    parameter_schema: NotRequired[list[ParameterSlot]]
    visualization: NotRequired[dict[str, Any]]


@with_config(EXTENSIBLE_SCHEMA)
class DecisionTreeResponse(TypedDict):
    """Response from ``quarry_decision_tree``."""

    plugin_version: str
    quarry_version: str
    rule_hash: Sha256Digest
    valid: bool
    diagnostics: list[Diagnostic]
    decision_tree: NotRequired[dict[str, DecisionTreePayload]]


@with_config(EXTENSIBLE_SCHEMA)
class OutcomeCounts(TypedDict):
    """Label counts for both outcomes of one decision step."""

    true: dict[str, int]
    false: dict[str, int]


@with_config(EXTENSIBLE_SCHEMA)
class LeafCounts(TypedDict):
    """Label counts for every terminal decision-flow state."""

    match: dict[str, int]
    no_match: dict[str, int]
    unknown: dict[str, int]
    error: dict[str, int]


@with_config(EXTENSIBLE_SCHEMA)
class DecisionFlowResponse(TypedDict):
    """Response from the existing ``quarry_decision_flows`` operation."""

    tree_sha256: Sha256Digest
    thresholds: dict[str, JsonNumber]
    steps: dict[str, OutcomeCounts]
    leaves: LeafCounts
    unmapped_paths: dict[str, int]


@with_config(EXTENSIBLE_SCHEMA)
class FailureRecord(TypedDict):
    """Stable failure category with optional rule attribution."""

    category: FailureCategory
    rule_name: NotRequired[str]


@with_config(EXTENSIBLE_SCHEMA)
class TimingSummary(TypedDict):
    """Bounded scan and per-rule timing values."""

    total_seconds: NotRequired[DurationSeconds]
    rules: NotRequired[dict[str, DurationSeconds]]


@with_config(EXTENSIBLE_SCHEMA)
class ModelUsageRecord(TypedDict):
    """Attempted physical calls for one public model alias and operation."""

    model: TelemetryModelAlias
    operation: TelemetryOperation
    calls: UsageCount


@with_config(EXTENSIBLE_SCHEMA)
class ModelUsageSummary(TypedDict):
    """Bounded provider-neutral model usage for one scan or ordered batch."""

    calls: UsageCount
    records: Annotated[list[ModelUsageRecord], Field(max_length=MAX_MODEL_USAGE_RECORDS)]
    truncated: StrictBool


ModelsCalled = Annotated[
    list[TelemetryModelAlias],
    Field(max_length=MAX_MODEL_USAGE_ALIASES),
]


@with_config(EXTENSIBLE_SCHEMA)
class OutputProducer(TypedDict):
    """Identity of a typed result producer within one scan."""

    kind: str
    name: str
    run_id: str


@with_config(EXTENSIBLE_SCHEMA)
class OutputItem(TypedDict):
    """One typed collection item."""

    item_id: str
    value: dict[str, Any]


@with_config(EXTENSIBLE_SCHEMA)
class RuleOutput(TypedDict):
    """A rule's typed collection output."""

    type: str
    category: str | None
    source: OutputProducer
    items: list[OutputItem]


@with_config(EXTENSIBLE_SCHEMA)
class MatchRecord(TypedDict):
    """One authoritative rule match."""

    rule_name: str
    metadata: dict[str, Any]
    output: NotRequired[RuleOutput]


@with_config(EXTENSIBLE_SCHEMA)
class NumberParameterEvidence(TypedDict):
    """Visible evidence for one effective numeric parameter value."""

    type: Literal["number"]
    binding: ParameterBinding
    value: JsonNumber


@with_config(EXTENSIBLE_SCHEMA)
class TextParameterEvidence(TypedDict):
    """Redacted evidence for one effective text parameter value."""

    type: Literal["text"]
    binding: ParameterBinding
    redacted: Literal[True]


ParameterEvidence = NumberParameterEvidence | TextParameterEvidence


@with_config(EXTENSIBLE_SCHEMA)
class ScanEntry(TypedDict):
    """One conversation's result inside a batch."""

    completion_state: CompletionState
    failures: list[FailureRecord]
    timings: TimingSummary
    model_usage: ModelUsageSummary
    models_called: ModelsCalled
    diagnostics: list[Diagnostic]
    decision_scores: dict[str, JsonNumber | None]
    matched: NotRequired[bool]
    matches: NotRequired[list[MatchRecord]]
    partial_matches: NotRequired[list[str]]
    explanations: NotRequired[list[dict[str, Any]]]
    decision_path: NotRequired[DecisionPathPayload]
    visualization: NotRequired[dict[str, Any]]


@with_config(EXTENSIBLE_SCHEMA)
class ScanResponse(VersionEnvelope):
    """Response from ``quarry_scan`` in single or ordered-batch mode."""

    rule_hash: Sha256Digest
    verbosity: ResponseVerbosity
    compilation: CompileResponse
    completion_state: CompletionState
    model_usage: ModelUsageSummary
    models_called: ModelsCalled
    diagnostics: NotRequired[list[Diagnostic]]
    timings: NotRequired[TimingSummary]
    failures: NotRequired[list[FailureRecord]]
    matched: NotRequired[bool]
    matches: NotRequired[list[MatchRecord]]
    partial_matches: NotRequired[list[str]]
    explanations: NotRequired[list[dict[str, Any]]]
    decision_path: NotRequired[DecisionPathPayload]
    decision_scores: NotRequired[dict[str, JsonNumber | None]]
    policy_digest: NotRequired[Sha256Digest]
    parameter_digest: NotRequired[Sha256Digest]
    parameter_evidence: NotRequired[dict[str, ParameterEvidence]]
    results: NotRequired[list[ScanEntry]]
    count: NotRequired[int]
    visualization: NotRequired[dict[str, Any]]


def published_output_schemas() -> dict[str, dict[str, Any]]:
    """Return each MCP tool's deterministic JSON output schema."""

    response_types = (
        ("quarry_reference", ReferenceResponse),
        ("quarry_compile", CompileResponse),
        ("quarry_decision_tree", DecisionTreeResponse),
        ("quarry_decision_flows", DecisionFlowResponse),
        ("quarry_scan", ScanResponse),
    )
    return {
        name: TypeAdapter(response_type).json_schema() for name, response_type in response_types
    }


def write_tool_schemas(repo_root: Path) -> Path:
    """Write the checked-in output schemas and return their path."""

    path = repo_root / "quarry" / "plugin" / "tool_schemas.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(published_output_schemas(), indent=2, sort_keys=True) + "\n"
    path.write_text(serialized, encoding="utf-8")
    return path


__all__ = [
    "CompileResponse",
    "DecisionFlowResponse",
    "DecisionTreeResponse",
    "ModelUsageRecord",
    "ModelUsageSummary",
    "ReferenceResponse",
    "ScanEntry",
    "ScanResponse",
    "published_output_schemas",
    "write_tool_schemas",
]
