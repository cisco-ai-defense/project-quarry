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

"""Manifest preflight using the same compiled predicates as a flow verdict."""

from __future__ import annotations

from collections.abc import Callable

from quarry.engine.atoms import Cost, is_per_tool_static
from quarry.engine.contracts import CompiledRuleArtifact, FlowCandidate
from quarry.engine.events import EvaluationInput
from quarry.engine.executor import _step_value
from quarry.engine.matchers import RegexBackend
from quarry.engine.oracle import BOOT, Oracle


def _static_values(
    artifact: CompiledRuleArtifact,
    evaluation_input: EvaluationInput,
    regex_backend: RegexBackend,
    check_deadline: Callable[[], None],
):
    """The rule's boot-step value of a compiled formula over the static environment alone:
    True, False, or None when it depends on an event, a model or a parameter."""

    oracle = Oracle(static_env=evaluation_input.static, regex_backend=regex_backend)
    specs = artifact.atom_specs
    static_unknown = {
        atom_id: None
        for atom_id, spec in specs.items()
        if spec.parameter_names
        or spec.cost not in {Cost.PURE, Cost.DECODE}
        or spec.predicate == "closed"
    }

    class StaticOracle:
        def __init__(self, tool=None):
            self.tool = tool

        def value(self, spec, _where):
            if spec.atom_id in static_unknown:
                return None
            if plan := artifact.tool_quantifiers.get(spec.atom_id):
                uncertain = False
                for tool in evaluation_input.static.read_tools().values():
                    check_deadline()
                    result = _step_value(
                        StaticOracle(tool), specs, plan.body, BOOT, static_unknown.copy()
                    )
                    if (plan.kind == "any" and result is True) or (
                        plan.kind == "all" and result is False
                    ):
                        return result
                    uncertain |= result is None
                return None if uncertain else plan.kind == "all"
            if spec.predicate == "tool_quantifier":
                return None
            if self.tool is not None and is_per_tool_static(spec):
                return oracle.ground_tool(spec, self.tool).value
            return oracle.value(spec, BOOT)

    def value(formula):
        return _step_value(StaticOracle(), specs, formula, BOOT, static_unknown.copy())

    return oracle, value


def _settled(requirement, static_value) -> bool:
    """Whether the static environment alone decides ``requirement``'s flow."""

    return (
        requirement.static_formula is not None and static_value(requirement.static_formula) is False
    ) or any(static_value(formula) is True for formula in requirement.static_sustained)


def clock_feeds_settled(
    artifact: CompiledRuleArtifact,
    missing: frozenset[str],
    evaluation_input: EvaluationInput,
    regex_backend: RegexBackend,
    check_deadline: Callable[[], None],
) -> bool:
    """Whether every flow that counts a ``missing`` section's items as clock steps is already
    decided by the static environment, so the rule needs none of those sections."""

    check_deadline()
    _, static_value = _static_values(artifact, evaluation_input, regex_backend, check_deadline)
    # A check it runs, such as a called rule's, needs its own clock flows settled.
    return all(
        _settled(requirement, static_value)
        for requirement in artifact.flow_requirements
        if missing.intersection(requirement.clock_surfaces)
    )


def cross_source_flow_has_candidates(
    artifact: CompiledRuleArtifact,
    evaluation_input: EvaluationInput,
    regex_backend: RegexBackend,
    check_deadline: Callable[[], None],
) -> bool:
    """Whether a live flow can depend on conversation/input interleaving.

    Compiled Boolean operand formulas retain conjunction, negation, accessor filters,
    windows and indexes. Unknown detector or capture values count as possible; the
    preflight must never turn an uncertain flow into an authoritative verdict.
    """

    check_deadline()
    oracle, static_value = _static_values(artifact, evaluation_input, regex_backend, check_deadline)
    specs = artifact.atom_specs
    unknown = {
        atom_id: None
        for atom_id, spec in specs.items()
        if spec.parameter_names
        or spec.cost not in {Cost.PURE, Cost.DECODE}
        or spec.predicate in {"closed", "tool_quantifier"}
    }

    def possible(candidate: FlowCandidate) -> bool:
        for event in evaluation_input.events:
            check_deadline()
            if candidate.surface not in event.surface_tags:
                continue
            if candidate.formula is None:
                return True
            value = _step_value(oracle, specs, candidate.formula, event, unknown.copy())
            if value is not False:
                return True
        return False

    def has_ambient_event(surfaces: tuple[str, ...]) -> bool:
        for event in evaluation_input.events:
            check_deadline()
            if event.surface_tags.intersection(surfaces):
                return True
        return False

    for requirement in artifact.flow_requirements:
        check_deadline()
        if requirement.ambient_surfaces and not has_ambient_event(requirement.ambient_surfaces):
            continue
        if _settled(requirement, static_value):
            continue
        if all(possible(candidate) for candidate in requirement.candidates):
            return True
    return False
