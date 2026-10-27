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
from quarry.integrations.model_inference import ModelInferenceBackend
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
    LLMError,
    ModelError,
    QuarryError,
    RuleCompatibilityError,
    RuleLoadError,
    RuleParseError,
    RuleValidationError,
)
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
from quarry.types.results import CollectionResult, ProducerRef, ResultItem, Score, Span
from quarry.types.rules import ScanResult

__all__ = [
    "__version__",
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
    "EvaluationTimeoutError",
    "IncompleteScanError",
    "LLMError",
    "ModelError",
    "ModelInferenceBackend",
    "ProducerRef",
    "PreparedRuleCache",
    "RuleCompatibilityError",
    "RuleLoadError",
    "RuleParseError",
    "RuleValidationError",
    "Score",
    "ResultItem",
    "SafetyLimits",
    "ScanResult",
    "Span",
    "compile_policy",
    "load_compiled_policy",
]
