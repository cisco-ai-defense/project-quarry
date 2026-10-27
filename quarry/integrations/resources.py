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

"""Resource management for pre-loading external dependencies."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.integrations.rate_limiter import LLMRateLimiter
from quarry.language.runtime import functions_in_resource_group, inference_functions
from quarry.probers import CapabilityClassifier, TopicProber
from quarry.types.dsl import FunctionCall, Literal, ModelCall, walk_condition_nodes
from quarry.types.rules import DSLRule
from quarry.types.text_resources import TEXT_REF_FUNCTION, TextRef


def _model_functions() -> frozenset[str]:
    return functions_in_resource_group("model")


def _inference_functions() -> frozenset[str]:
    return inference_functions()


def _score_functions() -> frozenset[str]:
    return functions_in_resource_group("score")


def _completion_functions() -> frozenset[str]:
    return functions_in_resource_group("completion")


def _token_functions() -> frozenset[str]:
    return functions_in_resource_group("token")


def _record_model_requirement(
    requirements: dict[str, set[Any]], alias: str, operation: str
) -> None:
    operation_keys = {
        "score": ("llm_score_models",),
        "complete": ("llm_complete_models",),
        "input_tokens": ("llm_token_models",),
        "split": ("llm_token_models",),
        "classify": ("model_classification_models",),
        "spans": ("model_classification_models", "model_span_models"),
    }.get(operation)
    if not alias or operation_keys is None:
        raise ValueError(f"unsupported model operation {operation!r}")
    requirements["llm_models"].add(alias)
    for key in operation_keys:
        requirements[key].add(alias)


# llm_score never accepts this legacy llm_judge reference syntax. Rejecting it
# avoids scoring against the literal resource name when text_ref() was intended.
JUDGE_PROMPT_REF_PREFIX = "prompt:"


def requirements_from_artifact(artifact) -> dict[str, set[Any]]:
    """Derive runtime resources directly from one compiler-owned artifact."""

    requirements = requirements_from_specs(artifact.atom_specs.values())
    if artifact.collect_plan is not None:
        collect = _expression_collector(requirements)
        collect(artifact.collect_plan.source)
        collect(artifact.collect_plan.local_predicate)
        if artifact.collect_plan.model_predicate is not None:
            collect(artifact.collect_plan.model_predicate)
    return requirements


def _expression_collector(requirements: dict[str, set[Any]]):
    def collect_expression(expression) -> None:
        if expression.kind == "text_ref":
            if not isinstance(expression.value, str) or not expression.value:
                raise ValueError("compiled artifact contains an invalid text_ref")
            requirements["text_resources"].add(expression.value)
        elif expression.kind == "model_call":
            value = expression.value
            if not isinstance(value, str) or value.count("\x1f") != 1:
                raise ValueError("compiled artifact contains an invalid model call")
            alias, operation = value.split("\x1f", 1)
            _record_model_requirement(requirements, alias, operation)
            requirements["model_operations"].add((alias, operation))
        for child in expression.children:
            collect_expression(child)

    return collect_expression


def requirements_from_specs(specs) -> dict[str, set[Any]]:
    """Derive the runtime resources that compiled atom specifications call and read.

    ``llm_models`` names every model alias they call and ``text_resources`` every text
    resource they read, which is also what keys a decision-tree node's identity."""

    requirements: dict[str, set[Any]] = {
        "llm_models": set(),
        "llm_score_models": set(),
        "llm_complete_models": set(),
        "llm_token_models": set(),
        "model_classification_models": set(),
        "model_span_models": set(),
        "text_resources": set(),
        "functions": set(),
        "model_operations": set(),
    }

    collect_expression = _expression_collector(requirements)
    model_functions = _model_functions()
    score_functions = _score_functions()
    completion_functions = _completion_functions()
    token_functions = _token_functions()
    resource_functions = (
        model_functions
        | score_functions
        | completion_functions
        | token_functions
        | functions_in_resource_group("topic_prober")
        | functions_in_resource_group("capability_classifier")
    )
    for spec in specs:
        detector = spec.detector
        if detector in resource_functions:
            requirements["functions"].add(detector)
        if detector in model_functions:
            _record_model_requirement(requirements, spec.detector_arg, "classify")
        elif detector in score_functions:
            _record_model_requirement(requirements, spec.detector_arg, "score")
        elif detector in completion_functions:
            _record_model_requirement(requirements, spec.detector_arg, "complete")
        elif detector in token_functions:
            _record_model_requirement(requirements, spec.detector_arg, "input_tokens")
        elif detector == "llm_chunk_reduce":
            _record_model_requirement(requirements, spec.detector_arg, "split")
            if spec.chunk_scalar == "llm_score":
                _record_model_requirement(requirements, spec.detector_arg, "score")
            elif spec.chunk_scalar != "llm_input_tokens":
                raise ValueError(
                    f"compiled artifact contains unsupported chunk scalar {spec.chunk_scalar!r}"
                )
            requirements["functions"].update({"llm_split_conversation", spec.chunk_scalar})
        if spec.constitution_is_ref:
            if not spec.constitution:
                raise ValueError("compiled artifact contains an empty text resource name")
            requirements["text_resources"].add(spec.constitution)
        if spec.model_expression is not None:
            collect_expression(spec.model_expression)
    return requirements


class ResourcePreparationError(RuntimeError):
    """Required resources were unavailable for an analyzed rule set."""

    def __init__(self, message: str, requirements: dict[str, set[Any]]):
        self.requirements = requirements
        super().__init__(message)


def judge_prompt_ref_name(prompt: str) -> str | None:
    """Return the referenced prompt name, or None if this is inline prompt text.

    Args:
        prompt: The prompt argument as written in the rule

    Returns:
        The name in a "prompt:<name>" reference, or None when the argument is prompt text
    """
    if not prompt.startswith(JUDGE_PROMPT_REF_PREFIX):
        return None
    return prompt[len(JUDGE_PROMPT_REF_PREFIX) :].strip()


class ResourceManager:
    """Manages pre-loading of external resources based on rule requirements."""

    def __init__(
        self,
        config: QuarryConfig,
        *,
        model_inference_backend: ModelInferenceBackend | None = None,
        llm_rate_limiter: LLMRateLimiter | None = None,
    ):
        """Initialize resources for the configured model backend and rate limiter."""
        self.config = config
        self.llm_rate_limiter = llm_rate_limiter
        self.model_inference_backend = model_inference_backend

        # Prober backends for semantic agent-drift predicates. These default to None so a
        # rule using a prober FAILS CLOSED at load unless a real backend is registered (see
        # _validate_required_resources). The dependency-free keyword/lexical stubs are
        # installed ONLY when the config explicitly opts in (allow_stub_probers) — otherwise
        # a crude heuristic could silently become an authoritative security decision. Swap
        # for a real embedding/classifier backend by reassigning these attributes.
        self.topic_prober = None
        self.capability_classifier = None
        if getattr(config, "allow_stub_probers", False):
            self.topic_prober = TopicProber()
            self.capability_classifier = CapabilityClassifier()

        self.requires_model_scanner: bool = False
        self.requires_model_inference: bool = False

    @property
    def inference_backend(self) -> ModelInferenceBackend | None:
        """The configured backend used by DSL inference functions."""
        return self.model_inference_backend

    def analyze_rules(self, rules: dict[str, Any]) -> dict[str, set[Any]]:
        """
        Analyze rules to determine required resources.

        Args:
            rules: Dictionary of loaded rules

        Returns:
            Dictionary of resource requirements
        """
        requirements = {
            "llm_models": set(),
            "llm_score_models": set(),
            "llm_complete_models": set(),
            "llm_token_models": set(),
            "model_classification_models": set(),
            "model_span_models": set(),
            "text_resources": set(),
            # Every function name called by any rule (diagnostics + the prober fail-closed
            # gate in _validate_required_resources).
            "functions": set(),
            "model_operations": set(),
        }

        self.requires_model_inference = False

        for _, rule in rules.items():
            function_calls = tuple(self._iter_function_calls(rule))
            model_calls = tuple(self._iter_model_calls(rule))
            requirements["functions"].update(call.name for call in function_calls)
            requirements["model_operations"].update(
                (call.alias, call.operation) for call in model_calls
            )

            configured_calls = tuple(
                call
                for call in function_calls
                if call.name in _inference_functions() or call.name in _model_functions()
            )
            if configured_calls:
                self.requires_model_inference = True
                requirements["llm_models"].update(
                    self._required_models_from_calls(configured_calls)
                )
                # Track operation-specific capabilities so missing config fails at load time.
                requirements["llm_score_models"].update(
                    self._required_models_from_calls(
                        tuple(call for call in configured_calls if call.name in _score_functions())
                    )
                )
                completion_calls = tuple(
                    call for call in configured_calls if call.name in _completion_functions()
                )
                requirements["llm_complete_models"].update(
                    self._required_models_from_calls(completion_calls)
                )
                classification_calls = tuple(
                    call for call in configured_calls if call.name in _model_functions()
                )
                requirements["llm_token_models"].update(
                    self._required_models_from_calls(
                        tuple(call for call in configured_calls if call.name in _token_functions())
                    )
                )
                requirements["model_classification_models"].update(
                    self._required_models_from_calls(classification_calls)
                )

            if model_calls:
                self.requires_model_inference = True
                for call in model_calls:
                    _record_model_requirement(requirements, call.alias, call.operation)

            text_refs = self._extract_text_ref_names_from_calls(function_calls)
            if text_refs:
                requirements["text_resources"].update(text_refs)

        return requirements

    def _validate_required_resources(self, requirements: dict[str, set[Any]]) -> None:
        """
        Validate that all required resources are available. Fail-fast if not.

        Args:
            requirements: Resource requirements from analyze_rules()

        Raises:
            ResourcePreparationError: If required resources are missing or unavailable
        """
        errors = []
        all_functions = requirements.get("functions", set())
        model_operations = requirements.get("model_operations", set())
        if model_operations and self.config.config_version != 2:
            errors.append("model() is available only with config_version: 2")
        if self.config.config_version == 1 and "model_split" in all_functions:
            errors.append("model_split() is available only with config_version: 2")
        if self.config.config_version == 2 and "llm_split_conversation" in all_functions:
            errors.append(
                "llm_split_conversation() was removed in config_version: 2; use model().split()"
            )

        required_inference_models = requirements.get("llm_models", set())
        if required_inference_models:
            inference_models = (
                self.model_inference_backend.models
                if self.model_inference_backend is not None
                else self.config.models
            )
            available_inference_models = (
                self.model_inference_backend.model_ids
                if self.model_inference_backend is not None
                else set(inference_models)
            )
            missing_inference_models = sorted(
                required_inference_models - available_inference_models
            )
            if missing_inference_models:
                available = ", ".join(sorted(available_inference_models)) or "none"
                errors.append(
                    "Rules require inference models not configured: "
                    f"{', '.join(missing_inference_models)}. Available models: {available}"
                )

            if self.config.config_version == 2 and self.model_inference_backend is not None:
                backend_config = self.model_inference_backend.config
                if backend_config.config_version != 2:
                    errors.append("Version 2 rules require a backend built from config_version: 2")
                mismatched = sorted(
                    model_name
                    for model_name in required_inference_models
                    if self.config.models.get(model_name)
                    != self.model_inference_backend.models.get(model_name)
                )
                if mismatched:
                    errors.append(
                        "Injected backend model configuration differs for: " + ", ".join(mismatched)
                    )

            missing_score = sorted(
                model_name
                for model_name in requirements.get("llm_score_models", set())
                if model_name in inference_models and inference_models[model_name].score is None
            )
            if missing_score:
                errors.append(
                    "Rules require score configuration for inference models: "
                    + ", ".join(missing_score)
                )

            missing_tokens = sorted(
                model_name
                for model_name in requirements.get("llm_token_models", set())
                if model_name in inference_models and inference_models[model_name].tokens is None
            )
            if missing_tokens:
                errors.append(
                    "Rules require input-token configuration for inference models: "
                    + ", ".join(missing_tokens)
                )

            missing_classify = sorted(
                model_name
                for model_name, operation in model_operations
                if operation == "classify"
                if model_name in inference_models and inference_models[model_name].classify is None
            )
            if self.config.config_version == 2 and missing_classify:
                errors.append(
                    "Rules require classify configuration for inference models: "
                    + ", ".join(missing_classify)
                )

            missing_spans = sorted(
                model_name
                for model_name, operation in model_operations
                if operation == "spans"
                if model_name in inference_models and inference_models[model_name].spans is None
            )
            if self.config.config_version == 2 and missing_spans:
                errors.append(
                    "Rules require spans configuration for inference models: "
                    + ", ".join(missing_spans)
                )

        missing_texts = sorted(
            requirements.get("text_resources", set()) - set(self.config.judge_prompts)
        )
        if missing_texts:
            available = ", ".join(sorted(self.config.judge_prompts)) or "none"
            errors.append(
                "Rules reference text resources not configured: "
                f"{', '.join(missing_texts)}. Available text resources: {available}"
            )

        # Fail closed on semantic probers with no backend. Function membership comes from the
        # generated catalog; this table only maps each backend kind to its runtime attribute.
        all_functions = requirements.get("functions", set())
        for resource_group, backend_attr, backend_name in (
            ("topic_prober", "topic_prober", "topic prober"),
            ("capability_classifier", "capability_classifier", "capability classifier"),
        ):
            required = all_functions & functions_in_resource_group(resource_group)
            if required and getattr(self, backend_attr) is None:
                for function_name in sorted(required):
                    errors.append(
                        f"Rule uses '{function_name}' but no {backend_name} backend is registered. "
                        "Register a real backend on the ResourceManager, or set "
                        "'allow_stub_probers: true' in config to opt into the dependency-free "
                        "keyword heuristic (NOT a production model)."
                    )

        if errors:
            error_msg = "Cannot proceed - required resources unavailable:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            raise ResourcePreparationError(error_msg, requirements)

    def load_required_resources(self, requirements: dict[str, set[Any]]) -> None:
        """
        Pre-load all required resources.

        Args:
            requirements: Resource requirements from analyze_rules()

        Raises:
            ResourcePreparationError: If required resources cannot be loaded
            ConfigurationError: If the model backend lacks a required model capability or
                credentials
        """
        self._validate_required_resources(requirements)

        if not self.requires_model_inference:
            return

        if self.model_inference_backend is not None:
            self.model_inference_backend.require_completion_models(
                requirements.get("llm_complete_models", set())
            )
            self.model_inference_backend.require_classification_models(
                requirements.get("model_classification_models", set())
            )
            span_models = requirements.get("model_span_models", set())
            if span_models:
                require_spans = getattr(self.model_inference_backend, "require_span_models", None)
                if not callable(require_spans):
                    raise ResourcePreparationError(
                        "Injected model backend cannot provide spans", requirements
                    )
                require_spans(span_models)
            require_credentials = getattr(self.model_inference_backend, "require_credentials", None)
            if callable(require_credentials):
                require_credentials(requirements.get("llm_models", set()))
            return

        if self.llm_rate_limiter is None:
            self.llm_rate_limiter = LLMRateLimiter(self.config.llm_rate_limits)
        self.model_inference_backend = ModelInferenceBackend(
            self.config,
            models_to_load=list(requirements.get("llm_models", set())),
            completion_models_to_load=requirements.get("llm_complete_models", set()),
            classification_models_to_load=requirements.get("model_classification_models", set()),
            span_models_to_load=requirements.get("model_span_models", set()),
            rate_limiter=self.llm_rate_limiter,
        )
        self.model_inference_backend.require_credentials(requirements.get("llm_models", set()))

    def _extract_function_calls(self, rule) -> set[str]:
        """Extract function names called in a rule."""
        return {call.name for call in self._iter_function_calls(rule)}

    def _iter_function_calls(self, rule: DSLRule) -> Iterator[FunctionCall]:
        """Yield every function call in a rule's executable ASTs."""

        for root in rule.executable_nodes():
            for node in walk_condition_nodes(root):
                if isinstance(node, FunctionCall):
                    yield node

    def _iter_model_calls(self, rule: DSLRule) -> Iterator[ModelCall]:
        """Yield every typed model operation in a rule's executable ASTs.

        The compatibility spelling ``model_split("alias", ...)`` is a plain function call in the
        AST that the compiler lowers to a ``split`` operation, so it is yielded in that form.
        """

        for root in rule.executable_nodes():
            for node in walk_condition_nodes(root):
                if isinstance(node, ModelCall):
                    yield node
                elif (
                    isinstance(node, FunctionCall)
                    and node.name == "model_split"
                    and node.args
                    and isinstance(node.args[0], Literal)
                    and isinstance(node.args[0].value, str)
                ):
                    yield ModelCall(
                        alias=node.args[0].value,
                        operation="split",
                        args=list(node.args[1:]),
                        result_type="Messages[]",
                    )

    def _extract_text_ref_names_from_calls(
        self, calls: Iterator[FunctionCall] | tuple[FunctionCall, ...]
    ) -> set[str]:
        """Extract the names from text_ref() calls anywhere in a rule.

        Keyed on the called function rather than on an argument position of whatever consumes the
        result, so a new function taking policy text is discovered without being registered here.

        A non-literal name is skipped rather than guessed at; the validator rejects one outright, so
        it never reaches a running rule.
        """

        names = set()
        for call in calls:
            if call.name != TEXT_REF_FUNCTION or len(call.args) != 1:
                continue
            argument = call.args[0]
            if isinstance(argument, Literal) and isinstance(argument.value, str):
                name = argument.value.strip()
                if name:
                    names.add(name)
        return names

    def resolve_text(self, value: object) -> str:
        """Resolve a rule's text argument to the text a model should be given.

        A :class:`TextRef` from ``text_ref()`` is looked up in the configured text resources.
        Anything else is already the text and is returned unchanged, so inline policies keep working.

        Args:
            value: The argument as the rule produced it, either a reference or the text

        Returns:
            The resolved text

        Raises:
            ValueError: If a named resource is not configured or legacy syntax is used
        """
        if isinstance(value, TextRef):
            return self._lookup_text(value.name)

        text = str(value)
        name = judge_prompt_ref_name(text)
        if name is not None:
            raise ValueError(
                f"'{text}' is not policy text. Name a configured text resource with "
                f'{TEXT_REF_FUNCTION}("{name}") instead of the "prompt:<name>" form'
            )
        return text

    def _lookup_text(self, name: str) -> str:
        """Return a configured text resource by name.

        Raises:
            ValueError: If the name is not configured, listing what is
        """
        configured = self.config.judge_prompts
        if name not in configured:
            available = ", ".join(sorted(configured)) if configured else "none"
            raise ValueError(
                f"Rule references text resource '{name}' which is not configured under 'texts' "
                f"or 'judge_prompts'. Available prompts: {available}"
            )
        return configured[name]

    def _required_models_from_calls(self, calls: tuple[FunctionCall, ...]) -> set[str]:
        """Return every model that a set of calls can select at runtime."""

        models = set()
        has_dynamic_model = False
        for call in calls:
            if not call.args:
                has_dynamic_model = True
                continue
            first_arg = call.args[0]
            if isinstance(first_arg, Literal) and isinstance(first_arg.value, str):
                models.add(first_arg.value)
            else:
                has_dynamic_model = True
        if has_dynamic_model:
            models.update(
                self.model_inference_backend.model_ids
                if self.model_inference_backend is not None
                else self.config.models
            )
        return models
