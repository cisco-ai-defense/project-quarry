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

"""Quarry SDK."""

from quarry._version import __version__
from quarry.core.conversation import Conversation
from quarry.inputs import InputAdapter
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.integrations.read_tokens import TextTokenBasis
from quarry.integrations.read_usage import ReadUsageLimits, ReadUsageSession, model_read_usage
from quarry.integrations.rule_scope import current_rule_name
from quarry.integrations.window_observations import (
    ModelWindowObservations,
    WindowObservationLimits,
    observe_model_windows,
)
from quarry.policy import CompiledPolicy, compile_policy
from quarry.policy_serialization import load_compiled_policy
from quarry.prepared import PreparedRuleCache
from quarry.safety.limits import SafetyLimits
from quarry.scanner import DSLScanner
from quarry.sdk import QuarryGuard
from quarry.types.exceptions import (
    ConfigurationError,
    ConversationValidationError,
    EvaluationError,
    EvaluationTimeoutError,
    IncompleteScanError,
    InvalidInputError,
    LLMError,
    ModelError,
    QuarryError,
    RuleCompatibilityError,
    RuleLoadError,
    RuleParseError,
    RuleValidationError,
)
from quarry.types.failures import ScanFailure, ScanFailureCategory
from quarry.types.messages import (
    AssistantMessage,
    AudioPart,
    ContentPart,
    DeveloperMessage,
    ImagePart,
    Message,
    Messages,
    SystemMessage,
    TextPart,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from quarry.types.read_reports import logical_read_report_schema
from quarry.types.results import CollectionResult, JudgeResult, ProducerRef, ResultItem, Score, Span
from quarry.types.rules import ScanResult
from quarry.types.window_reports import window_observation_report_schema

__all__ = [
    "__version__",
    "current_rule_name",
    "ModelWindowObservations",
    "WindowObservationLimits",
    "observe_model_windows",
    "QuarryError",
    "QuarryGuard",
    "AssistantMessage",
    "AudioPart",
    "ContentPart",
    "Conversation",
    "DeveloperMessage",
    "ImagePart",
    "Message",
    "Messages",
    "SystemMessage",
    "TextPart",
    "Tool",
    "ToolCall",
    "ToolMessage",
    "UserMessage",
    "CollectionResult",
    "ConfigurationError",
    "CompiledPolicy",
    "ConversationValidationError",
    "DSLScanner",
    "EvaluationError",
    "InputAdapter",
    "EvaluationTimeoutError",
    "IncompleteScanError",
    "InvalidInputError",
    "LLMError",
    "JudgeResult",
    "ModelError",
    "ModelInferenceBackend",
    "ProducerRef",
    "PreparedRuleCache",
    "ReadUsageLimits",
    "ReadUsageSession",
    "RuleCompatibilityError",
    "RuleLoadError",
    "RuleParseError",
    "RuleValidationError",
    "Score",
    "ResultItem",
    "SafetyLimits",
    "ScanFailure",
    "ScanFailureCategory",
    "ScanResult",
    "Span",
    "TextTokenBasis",
    "compile_policy",
    "load_compiled_policy",
    "model_read_usage",
    "logical_read_report_schema",
    "window_observation_report_schema",
]
