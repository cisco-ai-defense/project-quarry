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

"""Typed wire schemas for decision trees and the paths scans take through them.

These types describe JSON data, not engine objects. They are intentionally extensible:
published fields are checked, while an additive field from a newer Quarry build remains
available to older clients.
"""

from typing import Annotated, Literal, NotRequired

from pydantic import ConfigDict, Field, StrictBool, StrictFloat, StrictInt, with_config
from typing_extensions import TypedDict

EXTENSIBLE_SCHEMA = ConfigDict(extra="allow")

FiniteFloat = Annotated[StrictFloat, Field(allow_inf_nan=False)]
JsonNumber = StrictInt | FiniteFloat
Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ComparisonOperator = Literal["=", "!=", ">", ">=", "<", "<="]


@with_config(EXTENSIBLE_SCHEMA)
class ParameterThreshold(TypedDict):
    """A comparison threshold supplied by a named rule parameter."""

    parameter: str


@with_config(EXTENSIBLE_SCHEMA)
class ScoreNodeForm(TypedDict):
    """The normalized comparison performed by a score node."""

    quantifier: Literal["none", "any"]
    operator: ComparisonOperator
    threshold: JsonNumber | ParameterThreshold


@with_config(EXTENSIBLE_SCHEMA)
class NumberNodeForm(TypedDict):
    """The normalized comparison performed by a collection-size node."""

    operator: ComparisonOperator
    threshold: JsonNumber


@with_config(EXTENSIBLE_SCHEMA)
class DecisionNode(TypedDict):
    """One uniquely keyed decision-tree step and both of its outgoing edges."""

    id: str
    kind: Literal["check", "score", "number", "collect"]
    label: str
    source_nodes: list[str]
    score_form: NotRequired[ScoreNodeForm]
    score_key: NotRequired[str | None]
    number_form: NotRequired[NumberNodeForm]
    true: str
    false: str


@with_config(EXTENSIBLE_SCHEMA)
class DecisionTreePayload(TypedDict):
    """A versioned decision tree whose content is named by ``tree_sha256``."""

    schema_version: Literal[1]
    root: str
    nodes: dict[str, DecisionNode]
    tree_sha256: Sha256Digest


@with_config(EXTENSIBLE_SCHEMA)
class ScoreValue(TypedDict):
    """A normalized score attached to a span."""

    value: FiniteFloat


@with_config(EXTENSIBLE_SCHEMA)
class ConversationSpan(TypedDict):
    """A half-open character range within one message or across adjacent messages."""

    message_index: StrictInt
    start: StrictInt
    end: StrictInt
    end_message_index: NotRequired[StrictInt]
    score: NotRequired[ScoreValue]


@with_config(EXTENSIBLE_SCHEMA)
class ConstructedInputPiece(TypedDict):
    """One conversation field used to build a model input, without its text."""

    position: StrictInt
    field: str | None


@with_config(EXTENSIBLE_SCHEMA)
class ConstructedSpan(TypedDict):
    """An input with its own root and optional references to conversation fields."""

    constructed: Literal[True]
    pieces: NotRequired[list[ConstructedInputPiece]]


ScoreLocation = ConversationSpan | ConstructedSpan


@with_config(EXTENSIBLE_SCHEMA)
class ScoredChunk(TypedDict):
    """One split chunk, its optional repeated context, and its score."""

    chunk: StrictInt
    span: ScoreLocation
    context: ScoreLocation | None
    value: NotRequired[JsonNumber]


@with_config(EXTENSIBLE_SCHEMA)
class ScoreMeasurement(TypedDict):
    """The score value a decision step compared."""

    value: JsonNumber | None
    operator: ComparisonOperator
    threshold: JsonNumber
    evaluations: StrictInt
    matching: NotRequired[StrictInt]
    deciding: NotRequired[ScoredChunk]
    chunks: NotRequired[list[ScoredChunk]]
    span: NotRequired[ScoreLocation]


@with_config(EXTENSIBLE_SCHEMA)
class NumberMeasurement(TypedDict):
    """The collection size a decision step compared."""

    value: JsonNumber
    operator: ComparisonOperator
    threshold: JsonNumber


@with_config(EXTENSIBLE_SCHEMA)
class DecisionStep(TypedDict):
    """One reached decision step, including its unique step key."""

    node: str
    outcome: StrictBool | None
    score: NotRequired[ScoreMeasurement]
    number: NotRequired[NumberMeasurement]


@with_config(EXTENSIBLE_SCHEMA)
class ExplorationDetails(TypedDict):
    """Off-path checks evaluated for a deterministically sampled scan."""

    rate: FiniteFloat
    sample_value: FiniteFloat
    selected: StrictBool
    off_path: NotRequired[list[DecisionStep]]
    failed: NotRequired[list[str]]


@with_config(EXTENSIBLE_SCHEMA)
class DecisionPathPayload(TypedDict):
    """The ordered steps one scan reached and its authoritative leaf, if any."""

    tree_sha256: Sha256Digest
    config_sha256: Sha256Digest
    parameter_digest: Sha256Digest
    conversation_sha256: Sha256Digest | None
    leaf: Literal["match", "no_match"] | None
    path: list[DecisionStep]
    exploration: NotRequired[ExplorationDetails]


__all__ = [
    "ComparisonOperator",
    "ConstructedInputPiece",
    "ConstructedSpan",
    "ConversationSpan",
    "DecisionNode",
    "DecisionPathPayload",
    "DecisionStep",
    "DecisionTreePayload",
    "ExplorationDetails",
    "FiniteFloat",
    "JsonNumber",
    "NumberMeasurement",
    "NumberNodeForm",
    "ParameterThreshold",
    "ScoreMeasurement",
    "ScoreLocation",
    "ScoreNodeForm",
    "ScoredChunk",
    "Sha256Digest",
]
