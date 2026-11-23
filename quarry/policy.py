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

"""Compile one parameterized Quarry rule once and execute it many times."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from types import MappingProxyType
from typing import Any

from quarry.async_runtime import run_engine, run_sync
from quarry.config import QuarryConfig
from quarry.decision_report import conversation_sha256, explores, scanner_digests, tree_report
from quarry.engine.contracts import CompiledRuleArtifact
from quarry.engine.pattern_checks import PatternCheck
from quarry.input_registry import InputRegistry
from quarry.inputs import InputAdapter, adapter_fingerprints
from quarry.integrations.model_usage import model_call_budget
from quarry.integrations.read_results import collect_read_reports, transfer_result_scope
from quarry.language.runtime import surface_names
from quarry.safety.limits import SafetyLimits, json_size
from quarry.safety.local_work_reporting import aggregate_local_match_work
from quarry.safety.timer import (
    EvaluationDeadline,
    QuarryTimeoutError,
    absolute_scan_deadline,
    evidence_deadline,
    raise_if_evaluation_expired,
    remaining_evaluation_time,
    scan_deadline,
)
from quarry.safety.validator import _validate_json_tree, _validate_raw_json_container_types
from quarry.scanner import DSLScanner, _DeferredEvidence
from quarry.telemetry import add_durations, empty_model_usage, merge_model_usage
from quarry.types.exceptions import ConversationValidationError
from quarry.types.parameters import (
    ParameterSpec,
    ParameterValue,
    parameter_digest,
    parameter_evidence,
    validate_parameter_values,
)
from quarry.types.rules import ExplainLevel, ScanResult


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _referenced_model_configuration(scanner: DSLScanner, required: set[str]) -> dict[str, object]:
    """Return the configured identity of each model required by the compiled rule."""

    backend = scanner.resource_manager.model_inference_backend
    models = backend.models if backend is not None else scanner.config.models
    return {name: models[name].identity_dict() for name in sorted(required)}


def _hmac_sha256(key: bytes, value: str) -> str:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()


def _referenced_text_hashes(scanner: DSLScanner, required: set[str], key: bytes) -> dict[str, str]:
    """Key configured text identities so public reports do not expose guessable hashes."""

    return {
        name: _hmac_sha256(key, scanner.config.judge_prompts[name]) for name in sorted(required)
    }


def _referenced_input_hashes(
    scanner: DSLScanner, artifact: CompiledRuleArtifact
) -> tuple[dict[str, str], dict[str, str]]:
    read = artifact.read_surfaces
    declarations = scanner.input_registry.declarations
    inputs = {
        section: _canonical_sha256(declarations[section].describe())
        for section in sorted(read)
        if section in declarations
    }
    return inputs, adapter_fingerprints(scanner.input_adapters, read)


def _policy_identity(
    *,
    source_sha256: str,
    rule_name: str,
    artifact: CompiledRuleArtifact,
    parameter_schema: tuple[ParameterSpec, ...],
    policy_parameters: Mapping[str, ParameterValue],
    text_resources: Mapping[str, str],
    models: Mapping[str, object],
    evidence_key: bytes,
    inputs: Mapping[str, str] | None = None,
    adapters: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the stable identity assigned once when source is compiled."""

    identity = {
        "source_sha256": source_sha256,
        "rule_name": rule_name,
        "dfa_hash": artifact.dfa_hash,
        "compiler_version": artifact.compiler_version,
        "parameter_schema": [
            {"name": spec.name, "type": spec.type, "binding": spec.binding}
            for spec in parameter_schema
        ],
        "policy_parameters": {
            spec.name: (
                {
                    "type": "text",
                    "hmac_sha256": _hmac_sha256(evidence_key, str(policy_parameters[spec.name])),
                }
                if spec.type == "text"
                else {"type": "number", "value": policy_parameters[spec.name]}
            )
            for spec in parameter_schema
            if spec.binding == "policy"
        },
        "text_resources": dict(text_resources),
        "models": dict(models),
    }
    if inputs:
        identity["inputs"] = dict(inputs)
    if adapters:
        identity["adapters"] = dict(adapters)
    return identity


def _sampled_values(
    conversation: Any,
    read: frozenset[str],
    adapter_keys: Iterable[str],
    ordered: bool,
    registry: InputRegistry,
) -> dict[str, Any] | None:
    if type(conversation) is not dict:
        return None
    if any(type(key) is not str for key in dict.__iter__(conversation)):
        return None
    chat_read = bool(read & (surface_names() - {"tools"}))
    keys = (
        *(("messages",) if chat_read else ()),
        *(("tools", "functions") if "tools" in read else ()),
        *sorted(adapter_keys),
    )
    sampled = {key: conversation[key] for key in keys if key in conversation}
    if ordered and "timeline" in conversation:
        sampled["timeline"] = registry.timeline_read(
            conversation["timeline"], read, conversation, chat_read
        )
    for surface in sorted(read):
        namespace, _, name = surface.partition(".")
        holder = conversation.get(namespace)
        if name and namespace in conversation:
            if type(holder) is not dict:
                return None
            if any(type(key) is not str for key in dict.__iter__(holder)):
                return None
            if name in holder:
                sampled[surface] = holder[name]
    return sampled


def _validated_sample(
    conversation: Any,
    read: frozenset[str],
    adapter_keys: Iterable[str],
    ordered: bool,
    limits: SafetyLimits,
    registry: InputRegistry,
) -> dict[str, Any] | None:
    """Return only JSON-safe rule inputs for best-effort hashing and exploration."""

    try:
        sampled = _sampled_values(conversation, read, adapter_keys, ordered, registry)
    except Exception:
        return None
    if sampled is None:
        return None
    most = limits.max_total_size if limits.enforce_limits else None
    for key, value in sampled.items():
        try:
            _validate_raw_json_container_types(value)
            _validate_json_tree(value, key, limits.max_nesting_depth)
        except ConversationValidationError:
            return None
        if most is not None and json_size(value, most) > most:
            return None
    return sampled


def _sampled(
    conversation: Any,
    read: frozenset[str],
    adapter_keys: Iterable[str],
    ordered: bool,
    rate: float,
    limits: SafetyLimits,
    registry: InputRegistry,
) -> bool:
    sampled = _validated_sample(conversation, read, adapter_keys, ordered, limits, registry)
    return sampled is not None and explores(conversation_sha256(sampled), rate)


@dataclass(frozen=True, slots=True)
class CompiledPolicy:
    """One compiled rule with its policy parameters bound."""

    _rule_name: str = field(repr=False)
    _parameter_schema: tuple[ParameterSpec, ...] = field(repr=False)
    policy_parameters: Mapping[str, ParameterValue] = field(repr=False)
    policy_digest: str
    _scanner: DSLScanner = field(repr=False, compare=False)
    _source_sha256: str = field(repr=False, compare=False)
    _text_resource_hashes: Mapping[str, str] = field(repr=False, compare=False)
    _model_hashes: Mapping[str, str] = field(repr=False, compare=False)
    _input_hashes: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )
    _adapter_hashes: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )
    _evidence_key: bytes = field(
        default_factory=lambda: secrets.token_bytes(32), repr=False, compare=False
    )

    @property
    def parameter_schema(self) -> tuple[ParameterSpec, ...]:
        """The immutable parameter signature stored on the compiled rule."""

        return self._parameter_schema

    @classmethod
    def _from_scanner(
        cls,
        scanner: DSLScanner,
        source: str,
        policy_parameters: Mapping[str, object] | None = None,
        evidence_key: bytes | None = None,
    ) -> CompiledPolicy:
        """Bind policy parameters to the single rule already compiled by ``scanner``."""

        if len(scanner.compiled_rules) != 1:
            raise ValueError("CompiledPolicy requires exactly one compiled rule")
        rule_name, artifact = next(iter(scanner.compiled_rules.items()))
        schema = tuple(artifact.parameter_schema)
        validated_policy_parameters = validate_parameter_values(
            schema,
            policy_parameters,
            binding="policy",
        )
        frozen_policy_parameters = MappingProxyType(dict(validated_policy_parameters))
        key = secrets.token_bytes(32) if evidence_key is None else evidence_key
        if not isinstance(key, bytes) or len(key) < 16:
            raise ValueError("evidence_key must contain at least 16 bytes")
        source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        text_resources = _referenced_text_hashes(
            scanner, scanner.resource_requirements.get("text_resources", set()), key
        )
        model_configuration = _referenced_model_configuration(
            scanner, scanner.resource_requirements.get("llm_models", set())
        )
        model_hashes = {
            name: _canonical_sha256(configuration)
            for name, configuration in model_configuration.items()
        }
        input_hashes, adapter_hashes = _referenced_input_hashes(scanner, artifact)
        digest = _canonical_sha256(
            _policy_identity(
                source_sha256=source_sha256,
                rule_name=rule_name,
                artifact=artifact,
                parameter_schema=schema,
                policy_parameters=frozen_policy_parameters,
                text_resources=text_resources,
                models=model_configuration,
                evidence_key=key,
                inputs=input_hashes,
                adapters=adapter_hashes,
            )
        )
        return cls(
            _rule_name=rule_name,
            _parameter_schema=schema,
            policy_parameters=frozen_policy_parameters,
            policy_digest=digest,
            _scanner=scanner,
            _source_sha256=source_sha256,
            _text_resource_hashes=MappingProxyType(text_resources),
            _model_hashes=MappingProxyType(model_hashes),
            _input_hashes=MappingProxyType(input_hashes),
            _adapter_hashes=MappingProxyType(adapter_hashes),
            _evidence_key=key,
        )

    def decision_tree(self) -> dict[str, Any]:
        """The rule's decision tree: its checks, each check's true and false edges, and
        its root, as ``quarry_compile`` reports it.

        Node IDs hash each compiled check with the identity of the models and texts it
        reads, so a policy loaded from bytes reports the same tree as the one compiled.
        """

        artifact = self._scanner.compiled_rules[self._rule_name]
        return tree_report(artifact, scanner_digests(self._scanner, self._evidence_key))

    def to_bytes(self) -> bytes:
        """Serialize this policy into the versioned, data-only compiled format."""

        from quarry.policy_serialization import serialize_compiled_policy

        return serialize_compiled_policy(self)

    def bind_scan_parameters(
        self, parameters: Mapping[str, object] | None
    ) -> tuple[dict[str, ParameterValue], str, dict[str, dict[str, object]]]:
        """Validate one scan map and return effective values plus redacted provenance."""

        scan_values = validate_parameter_values(
            self.parameter_schema,
            parameters,
            binding="scan",
        )
        effective = {**self.policy_parameters, **scan_values}
        return (
            effective,
            parameter_digest(self.parameter_schema, effective, key=self._evidence_key),
            parameter_evidence(self.parameter_schema, effective),
        )

    def scan(
        self,
        conversation: dict[str, Any],
        parameters: Mapping[str, object] | None = None,
        *,
        explain: ExplainLevel = False,
        deadline_seconds: float | None = None,
        logical_reads: bool = False,
        window_observations: bool = False,
    ) -> ScanResult:
        """Execute one conversation without parsing or compiling the rule again."""
        if type(logical_reads) is not bool:
            raise ValueError("logical_reads must be a boolean")
        if type(window_observations) is not bool:
            raise ValueError("window_observations must be a boolean")
        return run_sync(
            self.scan_async(
                conversation,
                parameters,
                explain=explain,
                deadline_seconds=deadline_seconds,
                **({"logical_reads": True} if logical_reads else {}),
                **({"window_observations": True} if window_observations else {}),
            )
        )

    async def scan_async(
        self,
        conversation: dict[str, Any],
        parameters: Mapping[str, object] | None = None,
        *,
        explain: ExplainLevel = False,
        deadline_seconds: float | None = None,
        logical_reads: bool = False,
        window_observations: bool = False,
    ) -> ScanResult:
        """Asynchronous form of :meth:`scan`."""
        if type(logical_reads) is not bool:
            raise ValueError("logical_reads must be a boolean")
        if type(window_observations) is not bool:
            raise ValueError("window_observations must be a boolean")
        return (
            await self.scan_many_async(
                (conversation,),
                parameters,
                explain=explain,
                deadline_seconds=deadline_seconds,
                **({"logical_reads": True} if logical_reads else {}),
                **({"window_observations": True} if window_observations else {}),
            )
        )[0]

    def scan_many(
        self,
        conversations: Sequence[dict[str, Any]],
        parameters: Mapping[str, object] | None = None,
        *,
        explain: ExplainLevel = False,
        deadline_seconds: float | None = None,
        logical_reads: bool = False,
        window_observations: bool = False,
    ) -> list[ScanResult]:
        """Run an ordered batch through this already compiled policy."""
        if type(logical_reads) is not bool:
            raise ValueError("logical_reads must be a boolean")
        if type(window_observations) is not bool:
            raise ValueError("window_observations must be a boolean")
        return run_sync(
            self.scan_many_async(
                conversations,
                parameters,
                explain=explain,
                deadline_seconds=deadline_seconds,
                **({"logical_reads": True} if logical_reads else {}),
                **({"window_observations": True} if window_observations else {}),
            )
        )

    async def scan_many_async(
        self,
        conversations: Sequence[dict[str, Any]],
        parameters: Mapping[str, object] | None = None,
        *,
        explain: ExplainLevel = False,
        record_traces: bool = False,
        deadline_seconds: float | None = None,
        decision_paths: bool = False,
        explore_rate: float = 0.0,
        logical_reads: bool = False,
        window_observations: bool = False,
    ) -> list[ScanResult]:
        """Run an ordered batch on one event loop under one optional wall-clock deadline.

        ``explain=True`` explains every rule from its scan record; ``"support"`` returns
        each match's minimal support from the verdict pass alone.
        ``record_traces`` returns each rule's runtime trace in ``ScanResult.rule_traces``.
        ``decision_paths`` returns the rule's grounded decision-tree checks in
        ``ScanResult.decision_paths``. ``explore_rate`` selects that fraction of
        conversations, by content hash, whose path also grounds checks it skipped.

        ``logical_reads=True`` attaches each conversation's bounded scalar read snapshot
        after native evidence cleanup. This adds no model/tokenizer work. Reference token
        coverage requires explicit host-provided maps in a ``model_read_usage`` session;
        a borrowed session remains live and its snapshots say ``finalized=False``.
        ``window_observations=True`` separately returns bounded window probabilities,
        selected-input coverage, and accepted-reply cache status for each conversation.
        """
        if type(logical_reads) is not bool:
            raise ValueError("logical_reads must be a boolean")
        if type(window_observations) is not bool:
            raise ValueError("window_observations must be a boolean")
        with scan_deadline(deadline_seconds, "Policy batch deadline exceeded") as deadline:
            batch_deadline = deadline
        # Carry the absolute verdict deadline through the queue, then release it before
        # the evidence finishers install their separate per-conversation budgets.
        return await run_engine(
            partial(
                self._scan_many_reported_async,
                conversations,
                parameters,
                explain=explain,
                record_traces=record_traces,
                batch_deadline=batch_deadline,
                decision_paths=decision_paths,
                explore_rate=explore_rate,
                logical_reads=logical_reads,
                window_observations=window_observations,
            )
        )

    async def _scan_many_reported_async(
        self,
        conversations: Sequence[dict[str, Any]],
        parameters: Mapping[str, object] | None = None,
        *,
        explain: ExplainLevel = False,
        record_traces: bool = False,
        batch_deadline: EvaluationDeadline | None = None,
        decision_paths: bool = False,
        explore_rate: float = 0.0,
        logical_reads: bool = False,
        window_observations: bool = False,
    ) -> list[ScanResult]:
        if not (logical_reads or window_observations):
            return await self._scan_many_async(
                conversations,
                parameters,
                explain=explain,
                record_traces=record_traces,
                batch_deadline=batch_deadline,
                decision_paths=decision_paths,
                explore_rate=explore_rate,
            )
        with collect_read_reports(
            logical_reads=logical_reads, window_observations=window_observations
        ) as reports:
            results = await self._scan_many_async(
                conversations,
                parameters,
                explain=explain,
                record_traces=record_traces,
                batch_deadline=batch_deadline,
                decision_paths=decision_paths,
                explore_rate=explore_rate,
            )
        return reports.attach(results)

    async def _scan_many_async(
        self,
        conversations: Sequence[dict[str, Any]],
        parameters: Mapping[str, object] | None = None,
        *,
        explain: ExplainLevel = False,
        record_traces: bool = False,
        batch_deadline: EvaluationDeadline | None = None,
        decision_paths: bool = False,
        explore_rate: float = 0.0,
    ) -> list[ScanResult]:

        if not conversations:
            raise ValueError("conversations must not be empty")
        effective, digest, evidence = self.bind_scan_parameters(parameters)
        results: list[ScanResult] = []
        # The shared batch deadline covers verdicts. Evidence finishers run afterwards with
        # one bounded time and model-call budget per conversation.
        finishers: list[tuple[int, _DeferredEvidence]] = []
        deferred: list[_DeferredEvidence] = []
        try:
            with absolute_scan_deadline(batch_deadline):
                for index, conversation in enumerate(conversations):
                    remaining = remaining_evaluation_time()
                    if remaining is not None and remaining <= 0:
                        if self._scanner.limits.fail_on_timeout:
                            try:
                                raise_if_evaluation_expired()
                            except QuarryTimeoutError as error:
                                error.with_scan_result(
                                    self._aggregate_timeout_result(results, self._timeout_result())
                                )
                                raise
                        results.extend(
                            self._timeout_result() for _ in range(index, len(conversations))
                        )
                        break
                    first_deferred = len(deferred)
                    try:
                        result = await self._scanner._scan_data_with_parameters_async(
                            conversation,
                            effective,
                            enable_explain=explain,
                            record_traces=record_traces,
                            rules_subset={self._rule_name},
                            decision_paths=decision_paths,
                            explore_selector=lambda value: _sampled(
                                value,
                                self._scanner.compiled_rules[self._rule_name].read_surfaces,
                                self._adapter_hashes,
                                self._scanner.compiled_rules[self._rule_name].orders_input_items,
                                explore_rate,
                                self._scanner.limits,
                                self._scanner.input_registry,
                            ),
                            deferred_evidence=deferred,
                        )
                    except QuarryTimeoutError as error:
                        current = error.scan_result or self._timeout_result()
                        error.with_scan_result(self._aggregate_timeout_result(results, current))
                        raise
                    remaining = remaining_evaluation_time()
                    if remaining is not None and remaining <= 0:
                        timeout = self._timeout_result(result)
                        transfer_result_scope(result, timeout)
                        if self._scanner.limits.fail_on_timeout:
                            try:
                                raise_if_evaluation_expired()
                            except QuarryTimeoutError as error:
                                error.with_scan_result(
                                    self._aggregate_timeout_result(results, timeout)
                                )
                                raise
                        results.append(timeout)
                        results.extend(
                            self._timeout_result() for _ in range(index + 1, len(conversations))
                        )
                        break
                    finishers.extend((index, finish) for finish in deferred[first_deferred:])
                    results.append(result)
            limits = self._scanner.limits

            async def finish_one(index: int, finish: _DeferredEvidence) -> None:
                with (
                    evidence_deadline(limits.explore_timeout),
                    model_call_budget(limits.max_model_calls if limits.enforce_limits else None),
                ):
                    results[index] = await finish(results[index])

            tasks = [asyncio.create_task(finish_one(index, finish)) for index, finish in finishers]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                # Failure of one finisher must not leave its siblings retaining scan replies.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            return [self._attach_provenance(result, digest, evidence) for result in results]
        finally:
            # Includes owners produced immediately before a post-await timeout or failure.
            for finish in deferred:
                finish.close()
            deferred.clear()
            finishers.clear()

    def _timeout_result(self, result: ScanResult | None = None) -> ScanResult:
        """Represent a timed-out item while retaining its already observed telemetry."""

        return ScanResult(
            file_path=result.file_path if result is not None else None,
            rule_matches=[],
            scan_time=result.scan_time if result is not None else 0.0,
            timed_out_rules=[self._rule_name],
            rule_timings=(
                dict(sorted((result.rule_timings or {}).items())) or None
                if result is not None
                else None
            ),
            model_usage=(
                merge_model_usage([result.model_usage])
                if result is not None
                else empty_model_usage()
            ),
            local_match_work=result.local_match_work if result is not None else None,
        )

    def _aggregate_timeout_result(
        self,
        completed: Sequence[ScanResult],
        current: ScanResult,
    ) -> ScanResult:
        """Aggregate an ordered completed prefix and one current timeout exactly once."""

        observed = (*completed, current)
        timings: dict[str, list[float]] = {}
        for result in observed:
            for rule_name, duration in (result.rule_timings or {}).items():
                timings.setdefault(rule_name, []).append(duration)
        averaged_timings = {
            rule_name: add_durations(*(duration / len(durations) for duration in durations))
            for rule_name, durations in sorted(timings.items())
        }
        return ScanResult(
            file_path=current.file_path,
            rule_matches=[],
            scan_time=add_durations(*(result.scan_time for result in observed)),
            timed_out_rules=[self._rule_name],
            rule_timings=averaged_timings or None,
            model_usage=merge_model_usage([result.model_usage for result in observed]),
            local_match_work=aggregate_local_match_work(
                result.local_match_work for result in observed
            ),
        )

    def _attach_provenance(
        self,
        result: ScanResult,
        digest: str,
        evidence: dict[str, dict[str, object]],
    ) -> ScanResult:
        result.policy_digest = self.policy_digest
        result.parameter_digest = digest
        result.parameter_evidence = {name: dict(item) for name, item in evidence.items()}
        return result


def bind_prepared_policy(
    scanner: DSLScanner,
    source: str,
    parameters: Mapping[str, object] | None = None,
    *,
    evidence_key: bytes | None = None,
) -> tuple[CompiledPolicy, dict[str, ParameterValue]]:
    """Bind one complete parameter map to an already compiled single-rule scanner."""

    if len(scanner.compiled_rules) != 1:
        raise ValueError("CompiledPolicy requires exactly one compiled rule")
    schema = tuple(next(iter(scanner.compiled_rules.values())).parameter_schema)
    validated = validate_parameter_values(schema, parameters)
    policy_parameters = {
        spec.name: validated[spec.name] for spec in schema if spec.binding == "policy"
    }
    scan_parameters = {spec.name: validated[spec.name] for spec in schema if spec.binding == "scan"}
    return (
        CompiledPolicy._from_scanner(scanner, source, policy_parameters, evidence_key),
        scan_parameters,
    )


def compile_policy(
    source: str,
    *,
    policy_parameters: Mapping[str, object] | None = None,
    runtime_config: QuarryConfig | None = None,
    model_inference_backend=None,
    llm_rate_limiter=None,
    evidence_key: bytes | None = None,
    input_adapters: Mapping[str, InputAdapter] | None = None,
    pattern_checks: Mapping[str, PatternCheck] | None = None,
) -> CompiledPolicy:
    """Compile one rule and bind its policy parameters into a reusable policy."""

    if not isinstance(source, str) or not source.strip():
        raise ValueError("rule source must be a non-empty string")

    if runtime_config is None:
        config = QuarryConfig(quarry_rule_texts=(source,), config_version=2)
    else:
        # QuarryConfig is deeply immutable, so the request variant can share its models,
        # prompts, and limits by reference — only the rule source differs.
        config = replace(
            runtime_config,
            quarry_rules_paths=(),
            quarry_rule_texts=(source,),
        )
    pattern_options: dict[str, Any] = (
        {"pattern_checks": pattern_checks} if pattern_checks is not None else {}
    )
    scanner = DSLScanner(
        config,
        model_inference_backend=model_inference_backend,
        llm_rate_limiter=llm_rate_limiter,
        input_adapters=input_adapters,
        _allow_parameters=True,
        **pattern_options,
    )
    return CompiledPolicy._from_scanner(
        scanner,
        source,
        policy_parameters,
        evidence_key,
    )
