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
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from typing import Any

from quarry.config import QuarryConfig
from quarry.decision_report import conversation_sha256, explores, scanner_digests, tree_report
from quarry.engine.contracts import CompiledRuleArtifact
from quarry.integrations.model_usage import model_call_budget
from quarry.safety.timer import (
    evidence_deadline,
    raise_if_evaluation_expired,
    remaining_evaluation_time,
    scan_deadline,
)
from quarry.scanner import DSLScanner
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
    return {name: asdict(models[name]) for name in sorted(required)}


def _hmac_sha256(key: bytes, value: str) -> str:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()


def _referenced_text_hashes(scanner: DSLScanner, required: set[str], key: bytes) -> dict[str, str]:
    """Key configured text identities so public reports do not expose guessable hashes."""

    return {
        name: _hmac_sha256(key, scanner.config.judge_prompts[name]) for name in sorted(required)
    }


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
) -> dict[str, object]:
    """Build the stable identity assigned once when source is compiled."""

    return {
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
    ) -> ScanResult:
        """Execute one conversation without parsing or compiling the rule again."""

        return asyncio.run(
            self.scan_async(
                conversation,
                parameters,
                explain=explain,
                deadline_seconds=deadline_seconds,
            )
        )

    async def scan_async(
        self,
        conversation: dict[str, Any],
        parameters: Mapping[str, object] | None = None,
        *,
        explain: ExplainLevel = False,
        deadline_seconds: float | None = None,
    ) -> ScanResult:
        """Asynchronous form of :meth:`scan`."""

        return (
            await self.scan_many_async(
                (conversation,),
                parameters,
                explain=explain,
                deadline_seconds=deadline_seconds,
            )
        )[0]

    def scan_many(
        self,
        conversations: Sequence[dict[str, Any]],
        parameters: Mapping[str, object] | None = None,
        *,
        explain: ExplainLevel = False,
        deadline_seconds: float | None = None,
    ) -> list[ScanResult]:
        """Run an ordered batch through this already compiled policy."""

        return asyncio.run(
            self.scan_many_async(
                conversations,
                parameters,
                explain=explain,
                deadline_seconds=deadline_seconds,
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
    ) -> list[ScanResult]:
        """Run an ordered batch on one event loop under one optional wall-clock deadline.

        ``explain=True`` explains every rule from its scan record; ``"support"`` returns
        each match's minimal support from the verdict pass alone.
        ``record_traces`` returns each rule's runtime trace in ``ScanResult.rule_traces``.
        ``decision_paths`` returns the rule's grounded decision-tree checks in
        ``ScanResult.decision_paths``. ``explore_rate`` selects that fraction of
        conversations, by content hash, whose path also grounds checks it skipped.
        """

        if not conversations:
            raise ValueError("conversations must not be empty")
        effective, digest, evidence = self.bind_scan_parameters(parameters)
        results: list[ScanResult] = []
        # The shared batch deadline covers verdicts. Evidence finishers run afterwards with
        # one bounded time and model-call budget per conversation.
        finishers: list[tuple[int, Any]] = []
        with scan_deadline(deadline_seconds, "Policy batch deadline exceeded"):
            for index, conversation in enumerate(conversations):
                remaining = remaining_evaluation_time()
                if remaining is not None and remaining <= 0:
                    if self._scanner.limits.fail_on_timeout:
                        raise_if_evaluation_expired()
                    results.extend(self._timeout_result() for _ in range(index, len(conversations)))
                    break
                deferred: list = []
                result = await self._scanner._scan_data_with_parameters_async(
                    conversation,
                    effective,
                    enable_explain=explain,
                    record_traces=record_traces,
                    rules_subset={self._rule_name},
                    decision_paths=decision_paths,
                    explore=explores(conversation_sha256(conversation), explore_rate),
                    deferred_evidence=deferred,
                )
                remaining = remaining_evaluation_time()
                if remaining is not None and remaining <= 0:
                    if self._scanner.limits.fail_on_timeout:
                        raise_if_evaluation_expired()
                    results.append(self._timeout_result(scan_time=result.scan_time))
                    results.extend(
                        self._timeout_result() for _ in range(index + 1, len(conversations))
                    )
                    break
                finishers.extend((index, finish) for finish in deferred)
                results.append(result)
        limits = self._scanner.limits

        async def finish_one(index: int, finish) -> None:
            with (
                evidence_deadline(limits.explore_timeout),
                model_call_budget(limits.max_model_calls if limits.enforce_limits else None),
            ):
                results[index] = await finish(results[index])

        await asyncio.gather(*(finish_one(index, finish) for index, finish in finishers))
        return [self._attach_provenance(result, digest, evidence) for result in results]

    def _timeout_result(self, *, scan_time: float = 0.0) -> ScanResult:
        """Represent a batch item that could not finish within the shared deadline."""

        return ScanResult(
            file_path=None,
            rule_matches=[],
            scan_time=scan_time,
            timed_out_rules=[self._rule_name],
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
    scanner = DSLScanner(
        config,
        model_inference_backend=model_inference_backend,
        llm_rate_limiter=llm_rate_limiter,
        _allow_parameters=True,
    )
    return CompiledPolicy._from_scanner(
        scanner,
        source,
        policy_parameters,
        evidence_key,
    )
