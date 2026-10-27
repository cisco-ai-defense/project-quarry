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

"""Scanner engine."""

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple, NoReturn

import orjson

from quarry.cache.async_cache import current_scan, result_cache, scan_scope
from quarry.cache.keys import input_key, object_identity
from quarry.cache.regex_cache import regex_cache
from quarry.config import QuarryConfig
from quarry.core.logging import get_app_logger
from quarry.dsl.functions import (
    capability_tags,
    intent_baseline,
    llm_chunk_reduce,
    llm_complete,
    llm_input_tokens,
    llm_score,
    model_scan,
    model_score,
    off_topic,
    topic_drift,
    unrequested_action,
)
from quarry.dsl.parser import DSLParser
from quarry.engine.compiler import compile_rule
from quarry.engine.contracts import CompiledRuleArtifact, ScanRecord
from quarry.engine.events import EvaluationInput, build_evaluation_input
from quarry.engine.executor import evaluate_async
from quarry.engine.matchers import compare as _compare
from quarry.engine.matchers import truth
from quarry.engine.oracle import Grounded
from quarry.engine.witness import build_evidence
from quarry.integrations.collection import evaluate_collect_plan
from quarry.integrations.model_expression import (
    ModelExpressionTrace,
    build_model_result_evidence,
    evaluate_model_expression,
)
from quarry.integrations.model_usage import model_call_budget
from quarry.integrations.rate_limiter import LLMRateLimiter
from quarry.integrations.resources import ResourceManager
from quarry.language.runtime import (
    rejected_call_guidance,
    suggest_function_names,
    token_spellings,
)
from quarry.rule_versioning import check_rule_compatibility
from quarry.safety.limits import SafetyLimits
from quarry.safety.regex_backend import RE2_REGEX
from quarry.safety.timer import (
    QuarryTimeoutError,
    current_evaluation_deadline,
    evaluation_deadline,
    evidence_deadline,
    remaining_evaluation_time,
    scan_deadline,
)
from quarry.safety.validator import (
    RuleValidator,
    Severity,
    validate_conversation,
    validate_conversation_arguments,
)
from quarry.types.context import ScanContext
from quarry.types.dsl import FunctionCall, walk_condition_nodes
from quarry.types.exceptions import (
    ConfigurationError,
    ConversationValidationError,
    ModelError,
    RuleCompatibilityError,
    RuleLoadError,
    RuleValidationError,
)
from quarry.types.messages import Messages, ToolCallTypeRegistry
from quarry.types.parameters import ParameterValue
from quarry.types.rules import DSLRule, ExplainLevel, ScanResult
from quarry.types.text_resources import TextRef

logger = get_app_logger()


@dataclass(frozen=True, slots=True)
class _InitialGeneration:
    """A fully prepared rule generation awaiting atomic publication .

    Loading builds this in full — parse, validate, link, resource-prepare, and DFA-compile —
    before any of it is published onto the scanner. A failure anywhere raises and the scanner
    keeps NO partial state, so a half-loaded ruleset can never be scanned against."""

    rules: dict[str, DSLRule]
    sources: dict[str, str]
    resource_manager: ResourceManager
    resource_requirements: dict[str, set[Any]]
    compiled_rules: dict[str, Any]


def _formula_note(artifact) -> str | None:
    """A short human hint about the matched shape, drawn from the compiled artifact's
    notes (e.g. ``"bounded flow within:1"``, ``"has static (boot-step) atoms"``). Returns
    the most specific flow/capture note if present, else the first note, else None."""
    notes = getattr(artifact, "notes", ()) or ()
    if not notes:
        return None
    for kw in ("flow", "capture"):
        for n in notes:
            if kw in n:
                return n
    return notes[0]


def _shape_note(artifact) -> str | None:
    """Return a flow or capture note for a rule that did not match."""
    for kw in ("flow", "capture"):
        for note in getattr(artifact, "notes", ()) or ():
            if kw in note:
                return note
    return None


class _UnknownFunctionGuidance(NamedTuple):
    """One answer to an unknown call: the message clause and its structured form."""

    hint: str
    details: dict[str, Any]


def _unknown_function_guidance(name: str) -> _UnknownFunctionGuidance:
    """Answer an unknown call from the catalog, preferring exactness over similarity.

    The catalog records the mistakes callers actually make (``messages()`` for the
    ``messages`` surface, ``first(x)`` for ``x[0]``), so a listed name is answered outright
    rather than guessed at. Returning an empty answer is deliberate: when nothing credible
    is known, a wrong suggestion costs the author more than no suggestion.

    A record is keyed by the rejected call name alone, so it is offered as an example to
    follow and never as the literal text to substitute: ``first(tool_calls)`` is the
    documented form, and an author who wrote ``first(messages)`` must end up with
    ``messages[0]``, not with the ``tool_calls[0]`` the table happens to spell. Handing over
    the pattern keeps the author's own collection and arguments intact, and it needs nothing
    per function, so a record added later is explained on the day it is written down."""
    rejections: list[dict[str, str]] = []
    for record in rejected_call_guidance(name):
        # An absent replacement key reads as "documented as rejected, with nothing to copy".
        rejection = {"documented_form": record["source_form"]}
        if record["replacement"] is not None:
            rejection["documented_replacement"] = record["replacement"]
        rejections.append(rejection)

    if rejections:
        examples = [item for item in rejections if "documented_replacement" in item]
        if examples:
            shown = "; ".join(
                f"'{item['documented_form']}' -> '{item['documented_replacement']}'"
                for item in examples
            )
            hint = (
                f". Quarry rejects that spelling. Documented example: {shown}. Apply that "
                "pattern to what this rule wrote, keeping its own operands; the example is "
                "not literal replacement text"
            )
        else:
            forms = " or ".join(f"'{item['documented_form']}'" for item in rejections)
            hint = (
                f". Quarry rejects {forms} by design and offers no direct replacement; "
                "see the language reference for the supported way to express this"
            )
        return _UnknownFunctionGuidance(hint, {"rejected_spellings": rejections})

    if name in token_spellings():
        return _UnknownFunctionGuidance(
            f". '{name}' is an operator keyword, not a function; use it without parentheses",
            {"operator_keyword": name},
        )

    suggestions = suggest_function_names(name)
    if suggestions:
        return _UnknownFunctionGuidance(
            ". Did you mean " + " or ".join(f"'{item}'" for item in suggestions) + "?",
            {"suggestions": list(suggestions)},
        )
    return _UnknownFunctionGuidance("", {})


class DSLScanner:
    """Main scanner for analyzing LLM conversations."""

    def __init__(
        self,
        config: QuarryConfig,
        recursive: bool = False,
        verbose: bool = False,
        limits: SafetyLimits | None = None,
        batch_mode: bool = False,
        model_inference_backend=None,
        tool_call_types: ToolCallTypeRegistry | None = None,
        llm_rate_limiter: LLMRateLimiter | None = None,
        _allow_parameters: bool = False,
    ):
        """
        Initialize scanner with config.

        Args:
            config: QuarryConfig instance with rule paths and resources
            recursive: Recursively scan directories
            verbose: Enable verbose output
            limits: Safety limits configuration (overrides config if provided)
            batch_mode: Process JSON files as arrays of conversations
            tool_call_types: Optional mapping of tool call types for input normalization.
            model_inference_backend: Optional separately constructed model backend.
            llm_rate_limiter: Optional limiter shared across scanner instances.
        """
        self._initialize_execution(
            config,
            recursive=recursive,
            verbose=verbose,
            limits=limits,
            batch_mode=batch_mode,
            model_inference_backend=model_inference_backend,
            tool_call_types=tool_call_types,
            llm_rate_limiter=llm_rate_limiter,
            allow_parameters=_allow_parameters,
        )
        self.parser = DSLParser()
        self.validator = RuleValidator()

        if not self.config.quarry_rules_paths and not self.config.quarry_rule_texts:
            raise ConfigurationError(
                "Config must specify at least one Quarry rule path or rule text",
                {"paths": [], "rule_texts": []},
            )

        # build the ENTIRE generation (parse → validate → link → resource-prepare →
        # DFA-compile) before publishing any of it. A failure anywhere raises with the scanner
        # holding no partial state — a half-loaded ruleset can never be scanned against.
        self._publish_generation(self._build_initial_generation())

    def _initialize_execution(
        self,
        config: QuarryConfig,
        *,
        recursive: bool,
        verbose: bool,
        limits: SafetyLimits | None,
        batch_mode: bool,
        model_inference_backend,
        tool_call_types: ToolCallTypeRegistry | None,
        llm_rate_limiter: LLMRateLimiter | None,
        allow_parameters: bool,
    ) -> None:
        """Initialize state shared by source compilation and compiled-policy loading."""

        self.recursive = recursive
        self.verbose = verbose
        self.batch_mode = batch_mode
        validated_call_types = Messages.validate_tool_call_types(tool_call_types)
        self.tool_call_types: ToolCallTypeRegistry = MappingProxyType(validated_call_types)
        self.model_inference_backend = model_inference_backend
        self._allow_parameters = allow_parameters
        self.llm_rate_limiter = llm_rate_limiter
        # Version 1 assigns a separate timeout to rules that call external services.
        self._external_rule_cache: dict[str, bool] = {}

        self.config = config
        self.limits = limits or self.config.get_safety_limits()

    def _publish_generation(self, generation: _InitialGeneration) -> None:
        """Publish one fully prepared generation after every preparation step succeeds."""

        self.loaded_rules = generation.rules
        self._rule_sources = generation.sources
        self.resource_manager = generation.resource_manager
        self.resource_requirements = generation.resource_requirements
        self.compiled_rules = generation.compiled_rules

        if self.verbose and self.loaded_rules:
            logger.info(
                f"[STATUS] Loaded {len(self.loaded_rules)} rules: {list[str](self.loaded_rules.keys())}"
            )

    @classmethod
    def _from_compiled_policy(
        cls,
        config: QuarryConfig,
        *,
        rule: DSLRule,
        artifact: CompiledRuleArtifact,
        resource_requirements: dict[str, set[Any]],
        model_inference_backend=None,
        llm_rate_limiter: LLMRateLimiter | None = None,
    ) -> "DSLScanner":
        """Build an execution-only scanner without constructing a parser or compiler."""

        scanner = cls.__new__(cls)
        scanner._initialize_execution(
            config,
            recursive=False,
            verbose=False,
            limits=None,
            batch_mode=False,
            model_inference_backend=model_inference_backend,
            tool_call_types=None,
            llm_rate_limiter=llm_rate_limiter,
            allow_parameters=True,
        )
        resource_manager = ResourceManager(
            config,
            model_inference_backend=model_inference_backend,
            llm_rate_limiter=llm_rate_limiter,
        )
        requirements = {name: set(values) for name, values in resource_requirements.items()}
        resource_manager.requires_model_inference = bool(requirements.get("llm_models"))
        resource_manager.load_required_resources(requirements)
        scanner._publish_generation(
            _InitialGeneration(
                rules={rule.name: rule},
                sources={rule.name: "compiled-policy"},
                resource_manager=resource_manager,
                resource_requirements=requirements,
                compiled_rules={rule.name: artifact},
            )
        )
        return scanner

    def _build_initial_generation(self) -> _InitialGeneration:
        """Build and prepare a complete rule generation before publishing it ."""
        rule_files = self._discover_rule_files() if self.config.quarry_rules_paths else []
        parsed_rules = [(self._parse_rule_file(path), str(path)) for path in rule_files]

        for i, rule_text in enumerate(self.config.quarry_rule_texts):
            source = f"inline:{i}"
            parsed_rules.append((self._parse_rule_text(rule_text, source), source))

        # Reject a duplicate rule name across files up front: silently overwriting would make
        # which rule "wins" depend on load order (fail closed for deterministic loading).
        discovered_sources: dict[str, str] = {}
        for rule, source in parsed_rules:
            first_source = discovered_sources.get(rule.name)
            if first_source is not None:
                raise RuleValidationError(
                    f"Duplicate rule name '{rule.name}' in {source}; "
                    f"already defined in {first_source}",
                    {
                        "rule_name": rule.name,
                        "first_source": first_source,
                        "duplicate_source": source,
                        "error_type": "duplicate_rule_name",
                    },
                )
            discovered_sources[rule.name] = source

        compatibility_issues = [
            issue
            for rule, source in parsed_rules
            if (issue := check_rule_compatibility(rule, source=source)) is not None
        ]
        if compatibility_issues:
            # Every explicit requirement is checked before provider/resource construction
            # and before an execution profile can exclude a rule. One bad declaration makes
            # the candidate generation inadmissible; no compatible subset is published.
            raise RuleCompatibilityError(
                compatibility_issues[0].running_version,
                compatibility_issues,
            )

        resource_manager = ResourceManager(
            self.config,
            model_inference_backend=self.model_inference_backend,
            llm_rate_limiter=self.llm_rate_limiter,
        )

        rules: dict[str, DSLRule] = {}
        active_sources: dict[str, str] = {}
        for rule, source in parsed_rules:
            self._validate_rule(rule, source)
            if rule.parameters and not self._allow_parameters:
                raise RuleValidationError(
                    f"Rule '{rule.name}' in {source} declares parameters; "
                    "use compile_policy() to bind them",
                    {
                        "rule_name": rule.name,
                        "source": source,
                        "error_type": "parameterized_rule_requires_policy",
                    },
                )

            rules[rule.name] = rule
            active_sources[rule.name] = source

        self._link_candidate_functions(rules, active_sources)
        resource_requirements = self._prepare_candidate_resources(rules, resource_manager)
        compiled_rules = self._compile_candidate_rules(rules)

        return _InitialGeneration(
            rules=rules,
            sources=active_sources,
            resource_manager=resource_manager,
            resource_requirements=resource_requirements,
            compiled_rules=compiled_rules,
        )

    def _discover_rule_files(self) -> list[Path]:
        """Discover configured rule files in stable configured-path order .

        Configured-path order is preserved; within a directory, globs are sorted by POSIX
        path so load order is stable across platforms/filesystems. A configured path that
        does not exist is a hard error (fail closed), not a silent warning that would yield
        a partial ruleset."""
        discovered: list[Path] = []
        for configured_path in self.config.quarry_rules_paths:
            if configured_path.is_file():
                discovered.append(configured_path)
                continue

            if configured_path.is_dir():
                glob_pattern = "**/*.rule" if self.recursive else "*.rule"
                discovered.extend(
                    sorted(
                        (
                            rule_path
                            for rule_path in configured_path.glob(glob_pattern)
                            if rule_path.is_file()
                        ),
                        key=lambda rule_path: rule_path.as_posix(),
                    )
                )
                continue

            raise ConfigurationError(
                f"Configured rule path does not exist: {configured_path}",
                {"path": str(configured_path)},
            )

        if not discovered and not self.config.quarry_rule_texts:
            raise ConfigurationError(
                "No rule files found in configured rule paths",
                {"paths": [str(path) for path in self.config.quarry_rules_paths]},
            )

        return discovered

    def _parse_rule_file(self, path: Path) -> DSLRule:
        """Read and parse one rule source without mutating scanner state."""
        try:
            rule_text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise RuleLoadError(str(path), "read", str(exc)) from exc

        rule = self.parser.parse(rule_text, source=str(path))
        if not isinstance(rule, DSLRule):
            raise RuleLoadError(str(path), "parse", "Parser did not return a DSLRule")
        rule.source_file = path
        return rule

    def _compile_candidate_rules(self, rules: dict[str, DSLRule]) -> dict[str, Any]:
        """Compile every candidate rule to a DFA artifact .

        Runs on the unpublished generation. A rule that cannot be compiled fails
        at load time with located diagnostics; no partial ruleset is published.
        """
        if self.verbose:
            logger.info("[STATUS] Compiling rules to DFA artifacts...")
        compiled: dict[str, Any] = {}
        for name, rule in rules.items():
            result = compile_rule(
                rule,
                response_schemas={
                    alias: model.complete.response_schema
                    for alias, model in self.config.models.items()
                    if model.complete is not None
                },
            )
            if not result.ok:
                msgs = "; ".join(d.message for d in result.diagnostics.errors) or "unknown error"
                raise RuleValidationError(
                    f"Rule '{name}' could not be compiled to a DFA artifact: {msgs}"
                )
            compiled[name] = result.artifact
        if self.verbose:
            logger.info(f"[STATUS] Compiled {len(compiled)} rules")
        return compiled

    def _parse_rule_text(self, rule_text: str, source: str) -> DSLRule:
        """Parse a rule from an inline text string without filesystem access."""

        rule = self.parser.parse(rule_text, source=source)
        if not isinstance(rule, DSLRule):
            raise RuleLoadError(source, "parse", "Parser did not return a DSLRule")
        return rule

    def _prepare_candidate_resources(
        self,
        rules: dict[str, DSLRule],
        resource_manager: ResourceManager,
    ) -> dict[str, set[Any]]:
        """Analyze and pre-load resources for an unpublished generation."""
        if self.verbose:
            logger.info("[STATUS] Analyzing rule requirements...")

        requirements = resource_manager.analyze_rules(rules)
        try:
            resource_manager.load_required_resources(requirements)
        except Exception as exc:
            # Scanner construction is atomic, so callers cannot inspect the unpublished
            # resource manager after a load failure. Preserve its completed analysis on
            # the original exception without changing the exception type.
            # Exceptions accept ad hoc attributes at runtime; ty models no such attribute.
            exc.requirements = requirements  # ty: ignore[unresolved-attribute]
            raise

        if self.verbose:
            logger.info("[STATUS] Resources loaded and ready")
            rule_timeout = (
                f"{self.limits.rule_timeout}s"
                if self.limits.rule_timeout is not None
                else "disabled"
            )
            logger.info(f"[CONFIG] Rule timeout: {rule_timeout}")
        return requirements

    def _link_candidate_functions(
        self,
        rules: dict[str, DSLRule],
        sources: dict[str, str],
    ) -> None:
        """Link every function in an unpublished generation before resource work.

        An unknown function name surfaces as a fail-closed ``RuleValidationError`` (structured
        details) rather than a raw ``ValueError`` escaping from ``FunctionCall.link()``."""
        if self.verbose:
            logger.info("[STATUS] Linking function calls...")

        for rule_name, rule in rules.items():
            for root in rule.executable_nodes():
                for node in walk_condition_nodes(root):
                    if not isinstance(node, FunctionCall):
                        continue
                    try:
                        node.link()
                    except ValueError as exc:
                        source = sources[rule_name]
                        guidance = _unknown_function_guidance(node.name)
                        raise RuleValidationError(
                            f"Rule '{rule_name}' in {source} uses unknown function "
                            f"'{node.name}'{guidance.hint}",
                            {
                                "rule_name": rule_name,
                                "source": source,
                                "function": node.name,
                                "error_type": "unknown_function",
                                **guidance.details,
                            },
                        ) from exc

    def load_rules(self) -> NoReturn:
        """Reject live mutation; callers must replace the scanner atomically .

        Loading is a one-shot, atomic operation performed in ``__init__``. Reloading in place
        would risk publishing a partially-rebuilt generation, so it is refused — construct a
        new ``DSLScanner`` and swap the host reference instead."""
        raise RuntimeError(
            "Live rule reload is unsupported; construct a new DSLScanner "
            "and atomically replace the host reference"
        )

    def _validate_rule(self, rule: DSLRule, source: str) -> None:
        """Validate one candidate rule without registering it (fail closed on fatal errors)."""
        validation_result = self.validator.validate(rule)

        if validation_result.regex_errors:
            error_msgs = [
                f"  {err.variable}: {err.error} (pattern: {err.pattern})"
                for err in validation_result.regex_errors
            ]
            raise RuleValidationError(
                f"Rule '{rule.name}' in {source} has invalid regex pattern(s):\n"
                + "\n".join(error_msgs),
                {
                    "rule_name": rule.name,
                    "source": source,
                    "error_type": "regex",
                    "errors": error_msgs,
                },
            )

        if validation_result.semantic_errors:
            error_msgs = [f"  {err.message}" for err in validation_result.semantic_errors]
            raise RuleValidationError(
                f"Rule '{rule.name}' in {source} has semantic error(s):\n" + "\n".join(error_msgs),
                {
                    "rule_name": rule.name,
                    "source": source,
                    "error_type": "semantic",
                    "errors": error_msgs,
                },
            )

        for warning in validation_result.warnings:
            if warning.severity in (Severity.HIGH, Severity.EXTREME, Severity.MEDIUM):
                logger.warning(
                    f"Rule '{rule.name}' has {warning.severity.value} severity warning: "
                    f"{warning.message}"
                )

    def _make_async_detector(
        self, rule_ctx: ScanContext, *, explain: bool = False, regex_backend=None
    ):
        """Build the async external-atom grounding backend for one rule evaluation.

        The engine's oracle grounds MODEL/LLM/YARA atoms by awaiting
        ``async_detector(spec, text, event) -> bool | Grounded``. This bridge routes each
        external atom to
        the live DSL function (``model_scan`` / ``model_score`` / ``llm_score`` / the
        semantic probers from ``quarry.dsl.functions``), through the rule's ``ScanContext`` so
        the ResourceManager / function cache are reused. It lives here (not in the engine) so
        the engine stays free of any DSL/detector imports.

        ``spec.detector`` selects the backend and ``spec.detector_arg`` carries its first
        argument (model name / judge prompt / YARA signature). For a ``model_score``
        comparison atom the compiler folds the comparator into ``spec.op``/``spec.number``,
        so the score is reduced to the boolean the rule's ``score <op> n`` comparison means.
        """

        async def _probe(func, *args):
            # A prober's result depends on the prober, its arguments and the conversation's
            # intent baseline, so scans that probe the same inputs share it. Model functions
            # need no cache here: the backend caches each provider request by what it sends.
            manager = rule_ctx.resource_manager
            # Each prober instance is its own configuration, so it is named by itself.
            probers = [
                object_identity(prober)
                for prober in (
                    getattr(manager, "topic_prober", None),
                    getattr(manager, "capability_classifier", None),
                )
            ]
            key = input_key("prober", probers, func.__name__, args, intent_baseline(rule_ctx))
            return await result_cache.call(key, func, rule_ctx, *args)

        def _model_input_selector(spec, text: str | tuple[str, ...] | None = None) -> dict:
            """The atom's folded typed model-input selection, for _select_model_input.

            When the constructed input has dynamic slots (``None`` turns, from
            message("user", call.arguments.command)[.add_message(...)]), fill them with the
            values the oracle grounded for this EVENT atom: a single string for one slot, or a
            tuple of strings — one per slot, in order — for several. The oracle already resolved
            each RAW field value and rejected any non-string (number/bool/array/object/missing)
            before dispatch, so each value is guaranteed a string; an empty string is a valid
            (if empty) content and is preserved. A ``None`` value would mean the oracle never
            ran, so it fails closed."""
            constructed = spec.constructed_input
            none_slots = [
                index for index, (_, content) in enumerate(constructed) if content is None
            ]
            if none_slots:
                values = list(text) if isinstance(text, tuple) else [text] * len(none_slots)
                if len(values) != len(none_slots) or any(value is None for value in values):
                    raise ModelError("dynamic message() content resolved to a missing value")
                filled = dict(zip(none_slots, values, strict=True))
                constructed = tuple(
                    (role, filled[index] if content is None else content)
                    for index, (role, content) in enumerate(constructed)
                )
            return {
                "pipeline": spec.input_pipeline,
                "constructed_input": constructed,
            }

        async def detector(spec, text: str | tuple[str, ...], event=None) -> bool | Grounded:
            name = spec.detector
            arg = spec.detector_arg
            if name == "model_expression":
                trace = ModelExpressionTrace() if explain else None
                value = truth(
                    await evaluate_model_expression(
                        spec.model_expression,
                        rule_ctx,
                        event=event,
                        trace=trace,
                        regex_backend=regex_backend,
                    )
                )
                if trace is None:
                    return value
                return Grounded(
                    value=value,
                    evidence=build_model_result_evidence(trace, matched=value, event=event),
                )
            if name == "model_scan":
                # Forward the compiler-folded threshold + class_index (output index). None
                # threshold lets model_scan apply its own default; class_index defaults to 1.
                return bool(await model_scan(rule_ctx, arg, text, spec.threshold, spec.class_index))
            if name == "model_score":
                # class_index selects which output the score is read from; the comparator
                # (op/number) is the threshold folded in by the compiler.
                score = await model_score(rule_ctx, arg, text, spec.class_index)
                # A bare model_score atom (no comparator) is truthy on a positive score;
                # a comparison atom carries op/number folded in by the compiler.
                if spec.op and spec.number is not None:
                    return bool(_compare(score, spec.op, spec.number))
                return bool(score)
            if name == "llm_score":
                # Constitution score (0..1). The STATIC anchor `text` is ignored; the function
                # reads the selected conversation from ctx. arg = model id; spec.constitution =
                # the compiler-folded policy (literal text, or a text_ref NAME when
                # constitution_is_ref → wrapped in TextRef and resolved at scan time). The
                # folded typed selector (spec.input_*) chooses the whole conversation (default),
                # a messages/role-view selection or slice, or a constructed conversation.
                # Comparator folded in.
                constitution = (
                    TextRef(spec.constitution) if spec.constitution_is_ref else spec.constitution
                )
                score = await llm_score(
                    rule_ctx, arg, constitution, _model_input_selector(spec, text)
                )
                if spec.op and spec.number is not None:
                    return bool(_compare(score, spec.op, spec.number))
                return bool(score)
            if name == "llm_chunk_reduce":
                # Chunked measurement: split the conversation into token-bounded chunks, measure
                # spec.chunk_scalar on each CONCURRENTLY, reduce with spec.chunk_combine ("max"/
                # "min" for the quantifier forms, "count" for the cardinality form — which
                # additionally applies the folded per-chunk test), then apply the folded outer
                # comparator. The loop lives entirely here, in front of the DFA — the automaton
                # still sees one boolean leaf, exactly as for a plain llm_score atom. Reads
                # ctx.conversation, so the STATIC anchor `text` is ignored.
                constitution = (
                    TextRef(spec.constitution) if spec.constitution_is_ref else spec.constitution
                )
                value = await llm_chunk_reduce(
                    rule_ctx,
                    arg,
                    spec.chunk_scalar,
                    constitution,
                    spec.chunk_max_tokens,
                    spec.chunk_combine,
                    spec.chunk_inner_op,
                    spec.chunk_inner_number,
                    _model_input_selector(spec),
                )
                if spec.op and spec.number is not None:
                    return bool(_compare(value, spec.op, spec.number))
                return bool(value)
            if name == "llm_input_tokens":
                # Input-token count over the folded typed selection (default: whole
                # conversation + tools). arg = model id; comparator folded in by the compiler.
                count = await llm_input_tokens(rule_ctx, arg, _model_input_selector(spec))
                if spec.op and spec.number is not None:
                    return bool(_compare(count, spec.op, spec.number))
                return bool(count)
            if name == "llm_complete":
                # Generate a completion over the folded typed selection, then apply the folded
                # comparison → bool. arg = model id; spec.patterns[0] = the RHS literal;
                # spec.op is "~"/"matches" (regex search on the completion) or "=" (exact
                # equality). Negation (!= / not-match) was folded into the DFA at compile, so
                # only the positive test reaches here.
                text_out = await llm_complete(rule_ctx, arg, _model_input_selector(spec))
                literal = spec.patterns[0] if spec.patterns else ""
                if spec.op in ("~", "matches"):
                    return bool(regex_cache.search(str(literal), str(text_out)))
                return str(text_out) == literal
            if name == "topic_drift":
                # Field-first score prober: arg is unused (intent baseline is read from
                # the conversation inside the function); text is the action's content.
                score = await _probe(topic_drift, text)
                if spec.op and spec.number is not None:
                    return bool(_compare(score, spec.op, spec.number))
                return bool(score)
            if name == "off_topic":
                return bool(await _probe(off_topic, text))
            if name == "capability_tags":
                return bool(await _probe(capability_tags, arg, text))
            if name == "unrequested_action":
                # Field-first boolean: text is the action subject; intent baseline read
                # from the conversation inside the function.
                return bool(await _probe(unrequested_action, text))
            raise ValueError(f"unknown external detector {name!r}")

        return detector

    def _make_async_collector(self, rule_ctx: ScanContext, *, regex_backend=None):
        """Build the exhaustive typed-collection evaluator for one rule."""

        max_items = self.limits.max_comprehension_size if self.limits.enforce_limits else None

        async def collector(plan):
            return await evaluate_collect_plan(
                plan,
                rule_ctx,
                max_items=max_items,
                regex_backend=regex_backend,
            )

        return collector

    @staticmethod
    def _reject_out_of_range_dynamic_indexes(artifact, conversation) -> None:
        """Fail closed if a model input built from an event references an out-of-range index.

        ``message("user", messages[99].content)`` folds ordinal 99 onto an EVENT atom. If no
        event has that ordinal the atom simply never fires — a silent no-match. That is wrong for
        a model-input reference: the requested content does not exist, so raise a typed error
        (→ incomplete rule) BEFORE any provider dispatch rather than scoring it as false. Bound
        comprehension elements (no fixed ordinal) are unaffected.

        Only atoms RETAINED by the executable automata (``artifact.automaton_atoms``: the rule's
        and its closed parts') are checked. Constant folding (``false and MODEL(messages[99])``,
        ``true or MODEL(...)``) removes the model atom from the DFA while it lingers in
        ``atom_specs``; such an eliminated atom never runs, so it must not raise — the rule
        completes with the folded constant result."""

        retained = artifact.automaton_atoms
        specs = getattr(artifact, "atom_specs", {}) or {}
        for atom_id, spec in specs.items():
            if atom_id not in retained:
                continue
            constructed = getattr(spec, "constructed_input", None) or ()
            dynamic = any(content is None for _, content in constructed)
            if not dynamic and getattr(spec, "model_expression", None) is None:
                continue
            ordinal = getattr(spec, "ordinal", None)
            if ordinal is None:
                continue
            surface = spec.selector
            view = getattr(conversation, surface, None)
            size = len(view) if view is not None else 0
            target = ordinal if ordinal >= 0 else size + ordinal
            if not 0 <= target < size:
                raise ModelError(
                    f"dynamic message() content references {surface}[{ordinal}], out of range "
                    f"for a {surface} surface of size {size}; the model input cannot be built"
                )

    def _rule_grounds_external(self, rule_name: str) -> bool:
        """Return whether a compiled rule invokes an external detector."""

        cached = self._external_rule_cache.get(rule_name)
        if cached is not None:
            return cached
        artifact = self.compiled_rules.get(rule_name)
        specs = getattr(artifact, "atom_specs", {}) or {}
        collect_plan = getattr(artifact, "collect_plan", None)
        result = any(
            getattr(spec, "predicate", "") == "external" for spec in specs.values()
        ) or bool(collect_plan is not None and getattr(collect_plan, "has_external_work", False))
        self._external_rule_cache[rule_name] = result
        return result

    def scan_data(
        self,
        data: dict[str, Any],
        file_path: Path | None = None,
        enable_explain: ExplainLevel = False,
        rules_subset: set[str] | None = None,
        deadline_seconds: float | None = None,
    ) -> ScanResult:
        """Scan a single JSON file with loaded rules (sync wrapper)."""
        return asyncio.run(
            self.scan_data_async(
                data,
                file_path,
                enable_explain,
                rules_subset,
                deadline_seconds,
            )
        )

    async def scan_data_async(
        self,
        data: dict[str, Any],
        file_path: Path | None = None,
        enable_explain: ExplainLevel = False,
        rules_subset: set[str] | None = None,
        deadline_seconds: float | None = None,
    ) -> ScanResult:
        """Scan a single JSON file with loaded rules (async).

        ``deadline_seconds`` is an overall wall-clock budget for the whole inspection
        (e.g. derived from an incoming gRPC ``grpc-timeout``). Each rule's effective timeout
        is ``min(per_rule_timeout, remaining_deadline)``; once the deadline is exhausted the
        remaining rules time out immediately rather than starting doomed work.
        """
        with scan_deadline(deadline_seconds), scan_scope():
            return await self._scan_data_async(
                data,
                file_path=file_path,
                enable_explain=enable_explain,
                record_traces=False,
                rules_subset=rules_subset,
                parameters=None,
            )

    async def _scan_data_with_parameters_async(
        self,
        data: dict[str, Any],
        parameters: dict[str, ParameterValue],
        *,
        enable_explain: ExplainLevel = False,
        record_traces: bool = False,
        rules_subset: set[str] | None = None,
        decision_paths: bool = False,
        explore: bool = False,
        deferred_evidence: list | None = None,
    ) -> ScanResult:
        """Execute parameters already validated by ``CompiledPolicy``."""

        with scan_scope():
            return await self._scan_data_async(
                data,
                file_path=None,
                enable_explain=enable_explain,
                record_traces=record_traces,
                rules_subset=rules_subset,
                parameters=parameters,
                decision_paths=decision_paths,
                explore=explore,
                deferred_evidence=deferred_evidence,
            )

    async def _scan_data_async(
        self,
        data: dict[str, Any],
        *,
        file_path: Path | None,
        enable_explain: ExplainLevel,
        record_traces: bool,
        rules_subset: set[str] | None,
        parameters: dict[str, ParameterValue] | None,
        decision_paths: bool = False,
        explore: bool = False,
        deferred_evidence: list | None = None,
    ) -> ScanResult:
        """Execute one scan inside the caller's installed inspection deadline.

        ``record_traces`` returns each rule's runtime trace in ``ScanResult.rule_traces``.
        ``decision_paths`` returns each decomposed rule's grounded checks, with their model
        evidence, in ``ScanResult.decision_paths``; ``explore`` also grounds the checks each
        path skips.

        The verdicts run first, under the inspection and rule deadlines. The evidence they
        leave for later (support minimization, exploration, explanation) then runs under
        ``SafetyLimits.explore_timeout`` and a model-call budget of its own, so it can never
        change or delay a verdict. With ``deferred_evidence``, a list, the scan returns its
        verdicts and appends ``finish(result) -> result``, which the caller awaits once every
        verdict of its request is final, under its own ``evidence_deadline``.
        """

        start_time = datetime.now()
        timed_out_rules = []

        selected_names = rules_subset if rules_subset is not None else set(self.compiled_rules)
        parameter_schema = tuple(
            spec
            for name in sorted(selected_names)
            if name in self.compiled_rules
            for spec in self.compiled_rules[name].parameter_schema
        )
        if parameter_schema and parameters is None:
            message = "parameterized rules require CompiledPolicy"
            return ScanResult(
                file_path=file_path,
                rule_matches=[],
                scan_time=0.0,
                error=f"Parameter validation error: {message}",
                validation_error=message,
            )
        bound_parameters = parameters or {}

        try:
            validate_conversation(
                data,
                parse_arguments=False,
                max_nesting_depth=self.limits.max_nesting_depth,
            )
        except ConversationValidationError as e:
            message = str(e)
            return ScanResult(
                file_path=file_path,
                rule_matches=[],
                scan_time=0.0,
                error=f"Conversation validation error: {message}",
                validation_error=message,
            )
        except Exception as e:
            # Whole-conversation validation failure → the scan is INVALID (it never ran).
            # Surfaced via ``validation_error`` so ``completion_state`` is "invalid" and a
            # guard's ``raise_for_status`` fails closed distinctly from a per-rule error.
            return ScanResult(
                file_path=file_path,
                rule_matches=[],
                scan_time=0.0,
                validation_error=f"Conversation validation error: {e}",
            )

        if rules_subset is not None:
            unknown_rules = sorted(rules_subset - self.loaded_rules.keys())
            if not rules_subset or unknown_rules:
                reason = (
                    "explicit rule subset is empty"
                    if not rules_subset
                    else f"unknown rule(s) selected: {', '.join(unknown_rules)}"
                )
                return ScanResult(
                    file_path=file_path,
                    rule_matches=[],
                    scan_time=(datetime.now() - start_time).total_seconds(),
                    error=f"Rule selection error: {reason}",
                )

        messages = data.get("messages", [])
        tools = data.get("tools", data.get("functions", []))
        try:
            context = ScanContext(
                messages,
                tools,
                self.limits,
                self.resource_manager,
                tool_call_types=self.tool_call_types,
                parameters=bound_parameters,
                parameter_specs={spec.name: spec for spec in parameter_schema},
            )
        except ConversationValidationError as e:
            message = str(e)
            return ScanResult(
                file_path=file_path,
                rule_matches=[],
                scan_time=(datetime.now() - start_time).total_seconds(),
                error=f"Conversation validation error: {message}",
                validation_error=message,
            )
        except Exception as e:
            diagnostic = f"{type(e).__name__}: {e}"
            return ScanResult(
                file_path=file_path,
                rule_matches=[],
                scan_time=(datetime.now() - start_time).total_seconds(),
                error=f"Conversation preparation failed: {diagnostic}",
                evaluation_errors={"conversation_preparation": diagnostic},
            )

        # Size limits run before decoding argument JSON. Validate every call before
        # evaluating any rule, including rules that only inspect tool names, so bad
        # arguments cannot contribute to an authoritative verdict.
        try:
            validate_conversation_arguments(context.conversation.messages)
        except QuarryTimeoutError as e:
            if self.limits.fail_on_timeout:
                raise
            return ScanResult(
                file_path=file_path,
                rule_matches=[],
                scan_time=(datetime.now() - start_time).total_seconds(),
                error=f"Tool argument validation timed out: {e}",
            )
        except ConversationValidationError as e:
            message = str(e)
            return ScanResult(
                file_path=file_path,
                rule_matches=[],
                scan_time=(datetime.now() - start_time).total_seconds(),
                error=f"Conversation validation error: {message}",
                validation_error=message,
            )

        # No post-construction limit handling. ``ScanContext`` raises on any breach and the
        # handler above turns that into an INVALID result naming the limit, so there is no
        # over-limit conversation left here to report as merely truncated.

        # Explain/witness is produced by the DFA path (minimize_witness + build_rule_evidence
        # below), not by a ScanContext capture hook — no context.enable_capturing() on the
        # compiled engine.

        shared_input: list[EvaluationInput] = []

        def scan_input() -> EvaluationInput:
            # The input depends only on the conversation, which every rule context shares, so
            # one input serves all rules; a failed build is retried by the next rule.
            if not shared_input:
                shared_input.append(build_evaluation_input(context))
            return shared_input[0]

        async def evaluate_rule(rule_name: str, rule) -> dict[str, Any]:
            """Evaluate a single rule with isolated context."""
            rule_ctx = context.copy()
            rule_timeout = self.limits.rule_timeout
            if self.config.config_version == 1 and self._rule_grounds_external(rule_name):
                rule_timeout = self.limits.inference_timeout

            if self.verbose:
                logger.debug(f"[EVAL] Evaluating rule: {rule_name}")

            artifact = self.compiled_rules[rule_name]
            tree = artifact.decision_tree
            record = ScanRecord()
            reports_path = decision_paths and tree is not None and tree.decomposed
            path_record = record if reports_path else None
            try:
                rule_start = datetime.now()
                evaluation_input = scan_input()
                # Fail closed BEFORE any provider dispatch if a version 1 model-input reference
                # points at a message index that does not exist (e.g. messages[99].content on a
                # short conversation): the requested content cannot be built, so the rule is
                # incomplete rather than a silent no-match.
                self._reject_out_of_range_dynamic_indexes(artifact, rule_ctx.conversation)
                warnings: list[str] = []
                regex_backend = RE2_REGEX
                detector = self._make_async_detector(
                    rule_ctx,
                    explain=bool(enable_explain) or reports_path,
                    regex_backend=regex_backend,
                )
                collector = self._make_async_collector(
                    rule_ctx,
                    regex_backend=regex_backend,
                )
                # Version 1 preserves its separate external-inference budget. Version 2 uses
                # one optional rule-wide budget for local work, limiter waits, retries, and
                # provider calls. The inspection deadline remains an independent upper bound.
                with evaluation_deadline(rule_timeout, f"Rule {rule_name} timeout"):
                    active_deadline = current_evaluation_deadline()
                    effective_timeout = remaining_evaluation_time()
                    try:
                        async with asyncio.timeout(effective_timeout):
                            exec_result = await evaluate_async(
                                artifact,
                                evaluation_input,
                                async_detector=detector,
                                async_collector=collector,
                                warnings=warnings,
                                regex_backend=regex_backend,
                                # Witness minimization is O(witness²) DFA replays. Only pay it
                                # when explain evidence will actually be read; otherwise the match
                                # verdict is all we need and minimization is pure wasted work.
                                minimize_witness=bool(enable_explain),
                                time_left=remaining_evaluation_time,
                                # The verdict is decided inside the rule deadline; the
                                # evidence polish runs after it (below) and can only shorten
                                # the witness, never turn the verdict into a timeout.
                                defer_minimization=True,
                                record_trace=record_traces,
                                record=record,
                            )
                    except TimeoutError as e:
                        message = (
                            active_deadline.error_message
                            if active_deadline is not None
                            else f"Rule {rule_name} timeout"
                        )
                        raise QuarryTimeoutError(message) from e
                rule_elapsed = (datetime.now() - rule_start).total_seconds()
                rule_ctx.evaluation_warnings.extend(warnings)

                return {
                    "rule_name": rule_name,
                    "matched": exec_result.matched,
                    "exec_result": exec_result,
                    "evaluation_input": evaluation_input,
                    "record": record,
                    "timing": rule_elapsed,
                    "warnings": rule_ctx.evaluation_warnings,
                    "timeout": False,
                    "error": None,
                    "metadata": rule.metadata,
                    "output": exec_result.output,
                    "decision_path": path_record,
                }

            except QuarryTimeoutError:
                logger.warning(
                    f"[TIMEOUT] Rule timed out: name={rule_name!r} timeout={rule_timeout}s"
                )
                if self.limits.fail_on_timeout:
                    logger.error(f"[ERROR] Re-raising rule timeout: name={rule_name!r}")
                    raise
                return {
                    "rule_name": rule_name,
                    "timeout": True,
                    "error": None,
                    "matched": False,
                    "decision_path": path_record,
                }

            except Exception as e:
                logger.error(
                    f"[ERROR] Rule evaluation failed: name={rule_name!r} "
                    f"error_type={type(e).__name__}"
                )
                return {
                    "rule_name": rule_name,
                    "error": e,
                    "timeout": False,
                    "matched": False,
                    "decision_path": path_record,
                }

        # Run rules in parallel (filter by subset if provided). Invalid empty subsets
        # were rejected above; keep the explicit None check so selection is exact.
        rules_to_run = (
            {name: rule for name, rule in self.loaded_rules.items() if name in rules_subset}
            if rules_subset is not None
            else self.loaded_rules
        )
        call_limit = self.limits.max_model_calls if self.limits.enforce_limits else None
        with model_call_budget(call_limit):
            tasks = [
                asyncio.create_task(evaluate_rule(name, rule))
                for name, rule in rules_to_run.items()
            ]
            try:
                results = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        matches = []
        outputs = {}
        rule_timings = {}
        evaluation_errors: dict[str, str] = {}
        paths = {}
        verdicts = []

        for res in results:
            rule_name = res["rule_name"]
            if res["decision_path"] is not None:
                paths[rule_name] = res["decision_path"]

            if res.get("timeout"):
                timed_out_rules.append(rule_name)
                continue

            if res.get("error"):
                # Record the failure instead of silently dropping it (was fail-open): a rule
                # that raised produced no trustworthy verdict, so the scan is incomplete and
                # a guard must be able to fail closed on it.
                evaluation_errors[rule_name] = str(res["error"])
                continue

            verdicts.append(res)
            rule_timings[rule_name] = res["timing"]

            if res.get("warnings"):
                context.evaluation_warnings.extend(res["warnings"])

            if res["matched"]:
                if self.verbose:
                    logger.info(f'[MATCH] Rule matched: name="{rule_name}"')
                matches.append(rule_name)
                if res.get("output") is not None:
                    outputs[rule_name] = res["output"]

        if self.verbose:
            logger.debug(
                f"[EVAL] Completed evaluation of {len(rules_to_run)} rules. Matches: {len(matches)}, Timeouts: {len(timed_out_rules)}"
            )

        async def gather_evidence() -> None:
            """The evidence the verdicts above leave for later, now that they are final: a
            match's minimal support, the values exploration and explanation read. Runs under
            the caller's evidence budget, and a failure leaves the evidence unminimized or
            absent, never a verdict changed."""

            # The full explanation needs every value; support alone needs only the verdict's.
            full = enable_explain is True

            def explores(res) -> bool:
                return res["exec_result"].explore is not None and (
                    full or (explore and res["decision_path"] is not None)
                )

            async def one(res) -> None:
                exec_result = res["exec_result"]
                try:
                    if exec_result.refine is not None:
                        exec_result = await exec_result.refine(remaining_evaluation_time)
                        res["exec_result"] = exec_result
                    if explores(res):
                        await exec_result.explore(full, remaining_evaluation_time)
                except Exception as error:
                    logger.debug(f'[EVIDENCE] stopped: rule="{res["rule_name"]}" error="{error}"')

            pending = [
                res for res in verdicts if res["exec_result"].refine is not None or explores(res)
            ]
            if pending:
                await asyncio.gather(*(one(res) for res in pending))

        def evidence() -> dict[str, Any]:
            """The outputs read from the evidence: each explained rule's evidence and each
            rule's runtime trace."""
            explanations = []
            rule_traces = {}
            for res in verdicts:
                rule_name = res["rule_name"]
                exec_result = res["exec_result"]
                artifact = self.compiled_rules[rule_name]
                # Non-matches keep their trace too: visualization needs false and skipped checks.
                if record_traces:
                    rule_traces[rule_name] = exec_result.witness
                if not enable_explain or (enable_explain == "support" and not exec_result.matched):
                    continue
                explanations.append(
                    build_evidence(
                        rule_name,
                        exec_result.matched,
                        res["record"],
                        artifact,
                        res["evaluation_input"],
                        trace=exec_result.witness,
                        captured_value=exec_result.captured_value,
                        model_results=exec_result.model_results,
                        formula_note=(
                            _formula_note(artifact)
                            if exec_result.matched
                            else _shape_note(artifact)
                        ),
                        with_atoms=enable_explain is True,
                    )
                )
            return {
                "explanations": explanations or None,
                "rule_traces": rule_traces or None,
            }

        def scan_result(scan_time: float) -> ScanResult:
            return ScanResult(
                file_path=file_path,
                rule_matches=matches,
                scan_time=scan_time,
                error=None,
                was_truncated=context.was_truncated,
                truncation_info=context.truncation_info if context.was_truncated else None,
                timed_out_rules=timed_out_rules if timed_out_rules else None,
                skipped_rules=None,
                evaluation_errors=evaluation_errors if evaluation_errors else None,
                rule_timings=rule_timings if rule_timings else None,
                evaluation_warnings=context.evaluation_warnings
                if context.evaluation_warnings
                else None,
                outputs=outputs,
                decision_paths=paths if paths else None,
                **evidence(),
            )

        if deferred_evidence is not None:
            # The caller runs the evidence once every verdict of its request is final, as
            # part of this scan.
            scan = current_scan()

            async def finish(result: ScanResult) -> ScanResult:
                with scan_scope(scan):
                    await gather_evidence()
                return replace(result, **evidence())

            deferred_evidence.append(finish)
            return scan_result((datetime.now() - start_time).total_seconds())

        with evidence_deadline(self.limits.explore_timeout), model_call_budget(call_limit):
            await gather_evidence()
        return scan_result((datetime.now() - start_time).total_seconds())

    def scan_batch_data(
        self,
        conversations: list[dict[str, Any]],
        file_path: Path,
        enable_explain: ExplainLevel = False,
    ) -> ScanResult:
        """Scan multiple conversations from a batch file."""
        start_time = datetime.now()
        all_matches = []
        all_outputs = {}
        all_evidence = []
        total_scan_time = 0.0
        errors = []
        aggregated_timings = {}
        # Aggregate every child's completion signals so the batch's completion_state reflects
        # the WORST child: an invalid, truncated, skipped, or erroring conversation must block a
        # "complete" batch, or the batch fails open. Signals are conv-prefixed for provenance.
        agg_evaluation_errors: dict[str, str] = {}
        agg_skipped_rules: list[str] = []
        agg_timed_out_rules: list[str] = []
        agg_evaluation_warnings: list[str] = []
        agg_truncation_info: dict[str, str] = {}
        batch_was_truncated = False

        # Fail closed on an empty batch: there is nothing to authoritatively scan, so a "clean"
        # (exit 0) result would be indistinguishable from a batch that ran and matched nothing.
        # Surface it as INVALID (completion_state == "invalid") instead.
        if not conversations:
            return ScanResult(
                file_path=file_path,
                rule_matches=[],
                scan_time=0.0,
                validation_error="Batch must contain at least one conversation",
            )

        if self.verbose:
            logger.info(
                f"[STATUS] Processing {len(conversations)} conversations from batch file: {file_path}"
            )

        for i, conversation in enumerate(conversations):
            label = f"conv_{i + 1}"
            try:
                # A subclass may override scan_data with only (data, file_path); pass explain
                # only when it is requested so that override keeps working.
                result = (
                    self.scan_data(conversation, file_path, enable_explain)
                    if enable_explain
                    else self.scan_data(conversation, file_path)
                )
                if result.rule_matches:
                    prefixed_matches = [f"{label}:{rule}" for rule in result.rule_matches]
                    all_matches.extend(prefixed_matches)
                all_evidence.extend(
                    replace(evidence, rule_name=f"{label}:{evidence.rule_name}")
                    for evidence in result.explanations or ()
                )
                for rule_name, output in result.outputs.items():
                    all_outputs[f"{label}:{rule_name}"] = output
                total_scan_time += result.scan_time

                if result.rule_timings:
                    for rule_name, timing in result.rule_timings.items():
                        if rule_name not in aggregated_timings:
                            aggregated_timings[rule_name] = []
                        aggregated_timings[rule_name].append(timing)

                # A child that failed whole-conversation validation never ran: record it so the
                # batch cannot report complete. (The batch itself ran, so it is INCOMPLETE, not
                # INVALID — that state is reserved for a batch that could not run at all.)
                if result.validation_error:
                    agg_evaluation_errors[label] = (
                        f"invalid conversation: {result.validation_error}"
                    )
                if result.evaluation_errors:
                    for rule_name, message in result.evaluation_errors.items():
                        agg_evaluation_errors[f"{label}:{rule_name}"] = message
                if result.was_truncated:
                    batch_was_truncated = True
                    if result.truncation_info:
                        for key, message in result.truncation_info.items():
                            agg_truncation_info[f"{label}:{key}"] = message
                if result.skipped_rules:
                    agg_skipped_rules.extend(f"{label}:{rule}" for rule in result.skipped_rules)
                if result.evaluation_warnings:
                    agg_evaluation_warnings.extend(
                        f"{label}: {warning}" for warning in result.evaluation_warnings
                    )

                if result.error:
                    errors.append(f"Conversation {i + 1}: {result.error}")

                if result.timed_out_rules:
                    agg_timed_out_rules.extend(f"{label}:{rule}" for rule in result.timed_out_rules)
                    for rule_name in result.timed_out_rules:
                        errors.append(f"Conversation {i + 1}: Rule {rule_name} timeout")
            except Exception as e:
                errors.append(f"Conversation {i + 1}: {str(e)}")
                # An unexpected per-conversation failure is a non-authoritative result; make sure
                # it also blocks a complete batch even if nothing else set a completion signal.
                agg_evaluation_errors[label] = str(e)

        avg_timings = {}
        for rule_name, timings in aggregated_timings.items():
            avg_timings[rule_name] = sum(timings) / len(timings)

        scan_time = (datetime.now() - start_time).total_seconds()
        error_msg = "; ".join(errors) if errors else None

        return ScanResult(
            file_path=file_path,
            rule_matches=all_matches,
            scan_time=scan_time,
            outputs=all_outputs,
            error=error_msg,
            was_truncated=batch_was_truncated,
            truncation_info=agg_truncation_info or None,
            timed_out_rules=agg_timed_out_rules or None,
            skipped_rules=agg_skipped_rules or None,
            evaluation_errors=agg_evaluation_errors or None,
            rule_timings=avg_timings if avg_timings else None,
            evaluation_warnings=agg_evaluation_warnings or None,
            explanations=all_evidence or None,
        )

    def scan_file(self, json_path: Path, enable_explain: ExplainLevel = False) -> ScanResult:
        """Scan a single JSON file with loaded rules."""
        start_time = datetime.now()

        if self.verbose:
            logger.info(f"[STATUS] Scanning file: {json_path}")
        try:
            with open(json_path, "rb") as f:
                data = orjson.loads(f.read())

            if self.batch_mode and isinstance(data, list):
                return self.scan_batch_data(data, json_path, enable_explain)
            else:
                return self.scan_data(data, json_path, enable_explain)
        except Exception as e:
            return ScanResult(
                file_path=json_path,
                rule_matches=[],
                scan_time=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )

    def scan_directory(
        self, directory: Path, enable_explain: ExplainLevel = False
    ) -> list[ScanResult]:
        """Scan all JSON files in a directory."""
        results = []
        pattern = "**/*.json" if self.recursive else "*.json"
        files = sorted(directory.glob(pattern))

        if self.verbose:
            logger.info(f"[STATUS] Scanning {len(files)} files in directory: {directory}")
        else:
            print(f"Scanning {len(files)} files in {directory}", flush=True)

        for i, json_file in enumerate(files, 1):
            if not self.verbose:
                print(f"[{i}/{len(files)}] {json_file.name}", flush=True)
            results.append(self.scan_file(json_file, enable_explain))
        return results
