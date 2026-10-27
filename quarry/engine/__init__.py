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

"""Quarry DFA backend — self-contained temporal-logic + automaton core.

Core layers (no imports from ``quarry.types.dsl``):

- ``temporal``     — immutable LTLf formula AST, simplify, compile-time progression,
                     branch-atom analysis, canonicalization.
- ``symbolic_dfa`` — reduced guard-decision-graph DFA, deterministic stepping, guard
                     hooks (relevant_atoms / partial_step).
- ``builders``     — optimized shape builders (bounded window, neg-F chain, N-of) and
                     a direct fast-chain DFA.
- ``contracts``    — CompiledRuleArtifact / ExecutionResult data contracts.
- ``artifacts``    — deterministic hashing, structural validation, serialization.

The DFA is the rule decision engine: runtime stepping walks guard graphs and never
invokes formula progression (which is used only at compile time).
"""
