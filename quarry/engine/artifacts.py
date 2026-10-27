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

"""Deterministic hashing, structural validation, and serialization for DFAs.

A compiled artifact must have a *stable identity*: two DFAs that are structurally the
same automaton — regardless of the order states were discovered or guard nodes
interned — must hash identically, and a change in structure must change the hash. The
hash folds in ``COMPILER_VERSION`` so a strategy change invalidates old artifacts.

The scheme:
- states are keyed by their canonical formula key (already construction-order
  independent — see ``temporal.canonical``), then assigned a *stable index* by sorting
  those keys;
- each state's two guard graphs are described by remapping state-id leaves through
  the stable index and digesting guard nodes structurally (atom + child digests), once
  per shared subgraph, so interning ids never leak into the hash and its size is linear
  in the guard graph;
- the whole thing is hashed with SHA-256 over a canonical JSON encoding.
"""

from __future__ import annotations

import hashlib
import json

from quarry.engine.symbolic_dfa import DFA, Guard, GuardLeaf, GuardNode


def _stable_index(dfa: DFA) -> dict[int, int]:
    """Map each state id to a stable index derived from sorted canonical keys.

    State keys are unique per DFA (they are the interning identity), so sorting them
    yields a construction-order-independent numbering.
    """
    order = sorted(range(dfa.num_states), key=lambda sid: dfa.states[sid].key)
    return {sid: idx for idx, sid in enumerate(order)}


def _guard_digest(guard: Guard, remap: dict[int, int], nodes: dict[str, list], seen: dict) -> str:
    """The structural digest of a guard graph, recording each node once in ``nodes``.

    A leaf is ``["leaf", stable_target]`` and a node ``["node", atom, hi, lo]`` with its
    children's digests, so the digest depends on structure alone, not on interning ids, and
    a subgraph shared by several paths is described once.
    """
    known = seen.get(id(guard))
    if known is not None:
        return known
    if isinstance(guard, GuardLeaf):
        shape = ["leaf", remap[guard.target]]
    elif isinstance(guard, GuardNode):
        shape = [
            "node",
            guard.atom,
            _guard_digest(guard.hi, remap, nodes, seen),
            _guard_digest(guard.lo, remap, nodes, seen),
        ]
    else:
        raise TypeError(f"unknown guard type: {type(guard)}")
    digest = hashlib.sha256(json.dumps(shape, separators=(",", ":")).encode()).hexdigest()
    nodes[digest] = shape
    seen[id(guard)] = digest
    return digest


def canonical_structure(dfa: DFA) -> dict:
    """A canonical, construction-order-independent structure for the DFA: each state's
    guards name graphs in one table of guard nodes keyed by structural digest."""
    remap = _stable_index(dfa)
    inv = sorted(range(dfa.num_states), key=lambda sid: remap[sid])
    nodes: dict[str, list] = {}
    seen: dict = {}
    states = []
    for sid in inv:
        st = dfa.states[sid]
        states.append(
            {
                "index": remap[sid],
                "accepting": st.is_accepting,
                "rejecting": st.is_rejecting,
                "normal": _guard_digest(dfa.normal_guards[sid], remap, nodes, seen),
                "final": _guard_digest(dfa.final_guards[sid], remap, nodes, seen),
            }
        )
    return {
        "atoms": sorted(dfa.atoms),
        "initial": remap[dfa.initial_state],
        "states": states,
        "guards": nodes,
    }


def dfa_hash(dfa: DFA, compiler_version: str = "") -> str:
    """Deterministic SHA-256 identity hash of a DFA's structure.

    Independent of state-discovery order and guard-node interning order; sensitive to
    any change in accepting/rejecting status, transitions, atoms, or initial state.
    ``compiler_version`` is folded in so a strategy change invalidates the hash.
    """
    payload = {"compiler_version": compiler_version, "dfa": canonical_structure(dfa)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def to_dict(dfa: DFA, compiler_version: str = "") -> dict:
    """Serialize a DFA to a JSON-friendly dict (canonical structure + hash)."""
    return {
        "formula_name": dfa.formula_name,
        "hash": dfa_hash(dfa, compiler_version),
        "structure": canonical_structure(dfa),
    }
