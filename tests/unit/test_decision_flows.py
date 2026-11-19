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

"""Bounded flow counting across current and prior decision-tree paths."""

from __future__ import annotations

import hashlib
import json

import pytest

from quarry import decision_report
from quarry.decision_report import decision_flows

_CONFIG_DIGEST = "1" * 64
_PARAMETER_DIGEST = "2" * 64
_CONVERSATION_DIGEST = "3" * 64
_PRIOR_TREE_DIGEST = "4" * 64


def _hashed(payload: dict) -> dict:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "tree_sha256": hashlib.sha256(canonical).hexdigest()}


def _tree(*, true: str = "match", false: str = "no_match") -> dict:
    payload = {
        "schema_version": 1,
        "root": "t:volume",
        "nodes": {
            "t:volume": {
                "id": "t:volume",
                "kind": "number",
                "label": "message count exceeds one",
                "source_nodes": ["n0"],
                "number_form": {"operator": ">", "threshold": 1},
                "true": true,
                "false": false,
            }
        },
    }
    return _hashed(payload)


def _path(tree_sha256: str, *, outcome: bool = True, leaf: str | None = "match") -> dict:
    return {
        "tree_sha256": tree_sha256,
        "config_sha256": _CONFIG_DIGEST,
        "parameter_digest": _PARAMETER_DIGEST,
        "conversation_sha256": _CONVERSATION_DIGEST,
        "leaf": leaf,
        "path": [
            {
                "node": "t:volume",
                "outcome": outcome,
                "number": {"value": 2, "operator": ">", "threshold": 1},
            }
        ],
    }


def _repeated_tree() -> dict:
    repeated = {
        "id": "t:shared",
        "kind": "check",
        "label": "shared synthetic check",
        "source_nodes": ["n1"],
        "true": "match",
        "false": "no_match",
    }
    return _hashed(
        {
            "schema_version": 1,
            "root": "t:gate",
            "nodes": {
                "t:gate": {
                    "id": "t:gate",
                    "kind": "number",
                    "label": "synthetic gate",
                    "source_nodes": ["n0"],
                    "number_form": {"operator": ">", "threshold": 5},
                    "true": "t:shared",
                    "false": "t:shared~1",
                },
                "t:shared": repeated,
                "t:shared~1": repeated,
            },
        }
    )


def _repeated_path(tree: dict, *, gate_outcome: bool, repeated_key: str) -> dict:
    gate_value = 6 if gate_outcome else 4
    return {
        "tree_sha256": tree["tree_sha256"],
        "config_sha256": _CONFIG_DIGEST,
        "parameter_digest": _PARAMETER_DIGEST,
        "conversation_sha256": _CONVERSATION_DIGEST,
        "leaf": "match",
        "path": [
            {
                "node": "t:gate",
                "outcome": gate_outcome,
                "number": {"value": gate_value, "operator": ">", "threshold": 5},
            },
            {"node": repeated_key, "outcome": True},
        ],
    }


def _occurrence_tree(suffix: str) -> dict:
    node_id = "t:shared"
    repeated_key = f"{node_id}~{suffix}"
    return _hashed(
        {
            "schema_version": 1,
            "root": node_id,
            "nodes": {
                node_id: {
                    "id": node_id,
                    "kind": "check",
                    "label": "first occurrence",
                    "source_nodes": ["n0"],
                    "true": repeated_key,
                    "false": "no_match",
                },
                repeated_key: {
                    "id": node_id,
                    "kind": "check",
                    "label": "later occurrence",
                    "source_nodes": ["n0"],
                    "true": "match",
                    "false": "no_match",
                },
            },
        }
    )


def test_paths_from_other_trees_are_counted_without_walking_the_target_tree() -> None:
    tree = _tree()
    current = _path(tree["tree_sha256"])
    prior = _path(_PRIOR_TREE_DIGEST, outcome=False, leaf="no_match")

    counted = decision_flows(
        tree,
        [prior, current, prior, current],
        labels=["zeta", "zeta", "alpha", "alpha"],
    )

    assert counted["unmapped_paths"] == {"alpha": 1, "zeta": 1}
    assert counted["steps"] == {"t:volume": {"true": {"alpha": 1, "zeta": 1}, "false": {}}}
    assert counted["leaves"] == {
        "match": {"alpha": 1, "zeta": 1},
        "no_match": {},
        "unknown": {},
        "error": {},
    }
    assert list(counted["unmapped_paths"]) == ["alpha", "zeta"]
    assert list(counted["steps"]["t:volume"]["true"]) == ["alpha", "zeta"]


def test_a_single_tree_keeps_its_existing_flow_counts() -> None:
    tree = _tree()
    yes = _path(tree["tree_sha256"])
    no = _path(tree["tree_sha256"], outcome=False, leaf="no_match")

    counted = decision_flows(tree, [yes, no])

    assert counted == {
        "tree_sha256": tree["tree_sha256"],
        "thresholds": {},
        "steps": {"t:volume": {"true": {"all": 1}, "false": {"all": 1}}},
        "leaves": {
            "match": {"all": 1},
            "no_match": {"all": 1},
            "unknown": {},
            "error": {},
        },
        "unmapped_paths": {},
    }


@pytest.mark.parametrize(
    ("gate_outcome", "repeated_key", "moved_threshold", "rerouted_key"),
    [
        pytest.param(False, "t:shared~1", 3, "t:shared", id="later-to-first"),
        pytest.param(True, "t:shared", 7, "t:shared~1", id="first-to-later"),
    ],
)
def test_rerouting_uses_a_consistent_stable_answer_in_both_directions(
    gate_outcome, repeated_key, moved_threshold, rerouted_key
) -> None:
    tree = _repeated_tree()
    path = _repeated_path(tree, gate_outcome=gate_outcome, repeated_key=repeated_key)

    counted = decision_flows(tree, [path], thresholds={"t:gate": moved_threshold})

    assert counted["steps"][rerouted_key]["true"] == {"all": 1}
    assert counted["leaves"]["match"] == {"all": 1}
    assert counted["leaves"]["unknown"] == {}


def test_exact_and_stable_repeated_answers_must_agree() -> None:
    tree = _repeated_tree()
    consistent = _repeated_path(tree, gate_outcome=True, repeated_key="t:shared")
    consistent["exploration"] = {
        "rate": 1.0,
        "sample_value": 0.0,
        "selected": True,
        "off_path": [{"node": "t:shared~1", "outcome": True}],
        "failed": [],
    }
    assert decision_flows(tree, [consistent])["leaves"]["match"] == {"all": 1}

    conflicting = {
        **consistent,
        "exploration": {
            **consistent["exploration"],
            "off_path": [{"node": "t:shared~1", "outcome": False}],
        },
    }
    with pytest.raises(ValueError, match="conflicting answers for one stable node"):
        decision_flows(tree, [conflicting])

    exact_conflict = {
        **consistent,
        "exploration": {
            **consistent["exploration"],
            "off_path": [{"node": "t:shared", "outcome": False}],
        },
    }
    with pytest.raises(ValueError, match="conflicting answers for one exact step"):
        decision_flows(tree, [exact_conflict])


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda path: path.pop("config_sha256"), id="missing-required-digest"),
        pytest.param(
            lambda path: path["path"][0].update({"outcome": 1}),
            id="non-boolean-outcome",
        ),
        pytest.param(
            lambda path: path["path"][0].update({"node": "t:unknown"}),
            id="step-outside-route",
        ),
        pytest.param(
            lambda path: path.update({"leaf": "no_match"}),
            id="leaf-disagrees-with-route",
        ),
    ],
)
def test_malformed_or_impossible_paths_are_rejected(change) -> None:
    tree = _tree()
    path = _path(tree["tree_sha256"])
    change(path)

    with pytest.raises(ValueError, match=r"path 0"):
        decision_flows(tree, [path])


def test_a_foreign_tree_digest_does_not_bypass_path_validation() -> None:
    tree = _tree()
    malformed = _path(_PRIOR_TREE_DIGEST)
    del malformed["parameter_digest"]

    with pytest.raises(ValueError, match=r"path 0 is not a valid decision path"):
        decision_flows(tree, [malformed])


def test_a_cyclic_tree_is_rejected_before_foreign_paths_are_counted() -> None:
    tree = _tree(true="t:volume")
    prior = _path(_PRIOR_TREE_DIGEST, outcome=False, leaf="no_match")

    with pytest.raises(ValueError, match="edges loop"):
        decision_flows(tree, [prior])


@pytest.mark.parametrize("reserved", ["match", "no_match"])
@pytest.mark.parametrize("location", ["root-key", "non-root-key", "stable-id"])
def test_reserved_leaf_names_cannot_be_decision_nodes(reserved: str, location: str) -> None:
    if location == "root-key":
        payload = {
            "schema_version": 1,
            "root": reserved,
            "nodes": {
                reserved: {
                    "id": reserved,
                    "kind": "check",
                    "label": "reserved root",
                    "source_nodes": ["n0"],
                    "true": "match",
                    "false": "no_match",
                }
            },
        }
    elif location == "non-root-key":
        payload = {
            "schema_version": 1,
            "root": "t:root",
            "nodes": {
                "t:root": {
                    "id": "t:root",
                    "kind": "check",
                    "label": "ordinary root",
                    "source_nodes": ["n0"],
                    "true": reserved,
                    "false": "no_match",
                },
                reserved: {
                    "id": reserved,
                    "kind": "check",
                    "label": "reserved child",
                    "source_nodes": ["n1"],
                    "true": "match",
                    "false": "no_match",
                },
            },
        }
    else:
        payload = {
            "schema_version": 1,
            "root": "t:root",
            "nodes": {
                "t:root": {
                    "id": reserved,
                    "kind": "check",
                    "label": "reserved stable id",
                    "source_nodes": ["n0"],
                    "true": "match",
                    "false": "no_match",
                }
            },
        }

    with pytest.raises(ValueError, match="reserved leaf names"):
        decision_flows(_hashed(payload), [_path(_PRIOR_TREE_DIGEST)])


def test_one_is_the_only_canonical_first_repeated_suffix() -> None:
    tree = _occurrence_tree("1")

    assert decision_flows(tree, [_path(_PRIOR_TREE_DIGEST)])["unmapped_paths"] == {"all": 1}


@pytest.mark.parametrize(
    "suffix",
    ["", "0", "01", "+1", "-1", "one", "١"],
    ids=["empty", "zero", "leading-zero", "plus", "minus", "letters", "non-ascii-decimal"],
)
def test_noncanonical_repeated_suffixes_are_rejected(suffix: str) -> None:
    with pytest.raises(ValueError, match="canonical occurrence suffixes"):
        decision_flows(_occurrence_tree(suffix), [_path(_PRIOR_TREE_DIGEST)])


def test_repeated_occurrence_indexes_must_be_contiguous() -> None:
    with pytest.raises(ValueError, match="non-contiguous repeated occurrence indexes"):
        decision_flows(_occurrence_tree("2"), [_path(_PRIOR_TREE_DIGEST)])


def test_occurrence_aliases_and_separators_in_stable_ids_are_rejected() -> None:
    aliased = _hashed(
        {
            "schema_version": 1,
            "root": "t:gate",
            "nodes": {
                "t:gate": {
                    "id": "t:gate",
                    "kind": "check",
                    "label": "alias gate",
                    "source_nodes": ["n0"],
                    "true": "t:shared~1",
                    "false": "t:shared~01",
                },
                "t:shared~1": {
                    "id": "t:shared",
                    "kind": "check",
                    "label": "canonical alias",
                    "source_nodes": ["n1"],
                    "true": "match",
                    "false": "no_match",
                },
                "t:shared~01": {
                    "id": "t:shared",
                    "kind": "check",
                    "label": "duplicate alias",
                    "source_nodes": ["n1"],
                    "true": "match",
                    "false": "no_match",
                },
            },
        }
    )
    with pytest.raises(ValueError, match="canonical occurrence suffixes"):
        decision_flows(aliased, [_path(_PRIOR_TREE_DIGEST)])

    separated_id = _hashed(
        {
            "schema_version": 1,
            "root": "t:shared~1",
            "nodes": {
                "t:shared~1": {
                    "id": "t:shared~1",
                    "kind": "check",
                    "label": "ambiguous stable id",
                    "source_nodes": ["n0"],
                    "true": "match",
                    "false": "no_match",
                }
            },
        }
    )
    with pytest.raises(ValueError, match="stable node ids"):
        decision_flows(separated_id, [_path(_PRIOR_TREE_DIGEST)])


def test_request_wide_work_budget_accepts_under_and_at_but_rejects_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _tree()
    path = _path(tree["tree_sha256"])
    cost = decision_report._flow_request_cost(
        tree,
        [path],
        None,
        None,
        work_limit=10_000,
        byte_limit=10_000,
    )

    monkeypatch.setattr(decision_report, "_MAX_FLOW_WORK_UNITS", cost.work_units + 1)
    assert decision_flows(tree, [path])["leaves"]["match"] == {"all": 1}
    monkeypatch.setattr(decision_report, "_MAX_FLOW_WORK_UNITS", cost.work_units)
    assert decision_flows(tree, [path])["leaves"]["match"] == {"all": 1}
    monkeypatch.setattr(decision_report, "_MAX_FLOW_WORK_UNITS", cost.work_units - 1)
    with pytest.raises(ValueError, match="structural work limit"):
        decision_flows(tree, [path])


def test_request_wide_budget_counts_repeated_paths_multiplicatively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _tree()
    path = _path(tree["tree_sha256"])
    one_path_cost = decision_report._flow_request_cost(
        tree,
        [path],
        None,
        None,
        work_limit=10_000,
        byte_limit=10_000,
    )
    two_path_cost = decision_report._flow_request_cost(
        tree,
        [path, path],
        None,
        None,
        work_limit=10_000,
        byte_limit=10_000,
    )
    assert two_path_cost.work_units > one_path_cost.work_units
    monkeypatch.setattr(decision_report, "_MAX_FLOW_WORK_UNITS", two_path_cost.work_units - 1)

    with pytest.raises(ValueError, match="structural work limit"):
        decision_flows(tree, [path, path])


def test_request_wide_budget_accounts_for_nested_score_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _tree()
    plain = _path(_PRIOR_TREE_DIGEST)
    plain["path"][0] = {
        "node": "t:score",
        "outcome": True,
        "score": {
            "value": 0.8,
            "operator": ">",
            "threshold": 0.5,
            "evaluations": 1,
        },
    }
    chunked = json.loads(json.dumps(plain))
    chunked["path"][0]["score"]["chunks"] = [
        {
            "chunk": index,
            "span": {"message_index": 0, "start": index, "end": index + 1},
            "context": None,
            "value": 0.8,
        }
        for index in range(4)
    ]
    chunked["path"][0]["score"]["evaluations"] = 4
    plain_cost = decision_report._flow_request_cost(
        tree,
        [plain],
        None,
        None,
        work_limit=10_000,
        byte_limit=10_000,
    )
    chunked_cost = decision_report._flow_request_cost(
        tree,
        [chunked],
        None,
        None,
        work_limit=10_000,
        byte_limit=10_000,
    )
    assert chunked_cost.work_units > plain_cost.work_units
    monkeypatch.setattr(decision_report, "_MAX_FLOW_WORK_UNITS", plain_cost.work_units)

    assert decision_flows(tree, [plain])["unmapped_paths"] == {"all": 1}
    with pytest.raises(ValueError, match="structural work limit"):
        decision_flows(tree, [chunked])


def test_request_byte_budget_accepts_under_and_at_but_rejects_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _tree()
    path = _path(_PRIOR_TREE_DIGEST)
    path["future_text"] = "é" * 10
    cost = decision_report._flow_request_cost(
        tree,
        [path],
        None,
        None,
        work_limit=10_000,
        byte_limit=10_000,
    )

    monkeypatch.setattr(decision_report, "_MAX_FLOW_SCALAR_BYTES", cost.scalar_bytes + 1)
    assert decision_flows(tree, [path])["unmapped_paths"] == {"all": 1}
    monkeypatch.setattr(decision_report, "_MAX_FLOW_SCALAR_BYTES", cost.scalar_bytes)
    assert decision_flows(tree, [path])["unmapped_paths"] == {"all": 1}
    monkeypatch.setattr(decision_report, "_MAX_FLOW_SCALAR_BYTES", cost.scalar_bytes - 1)
    with pytest.raises(ValueError, match="aggregate scalar-byte limit"):
        decision_flows(tree, [path])


def test_integer_bytes_include_zero_sign_and_magnitude_without_changing_bools() -> None:
    def cost(value: object):
        return decision_report._request_cost(
            (value,),
            work_limit=10_000,
            byte_limit=10_000,
        )

    assert cost(0).scalar_bytes == 1
    assert cost(1).scalar_bytes == 1
    assert cost(-1).scalar_bytes == 2
    assert cost(10**100).scalar_bytes == 101
    assert cost(True).scalar_bytes == cost(False).scalar_bytes == 0
    assert cost(True).work_units == 1
    assert cost(1).work_units == 2


def test_single_integer_byte_budget_accepts_under_and_at_but_rejects_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _tree()
    path = _path(_PRIOR_TREE_DIGEST)
    path["future_integer"] = 1 << 4_096
    cost = decision_report._flow_request_cost(
        tree,
        [path],
        None,
        None,
        work_limit=10_000,
        byte_limit=10_000,
    )

    monkeypatch.setattr(decision_report, "_MAX_FLOW_SCALAR_BYTES", cost.scalar_bytes + 1)
    assert decision_flows(tree, [path])["unmapped_paths"] == {"all": 1}
    monkeypatch.setattr(decision_report, "_MAX_FLOW_SCALAR_BYTES", cost.scalar_bytes)
    assert decision_flows(tree, [path])["unmapped_paths"] == {"all": 1}
    monkeypatch.setattr(decision_report, "_MAX_FLOW_SCALAR_BYTES", cost.scalar_bytes - 1)
    with pytest.raises(ValueError, match="aggregate scalar-byte limit"):
        decision_flows(tree, [path])


def test_many_integers_share_one_aggregate_byte_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _tree()
    path = _path(_PRIOR_TREE_DIGEST)
    path["future_integers"] = [1 << 512] * 50
    cost = decision_report._flow_request_cost(
        tree,
        [path],
        None,
        None,
        work_limit=10_000,
        byte_limit=100_000,
    )

    monkeypatch.setattr(decision_report, "_MAX_FLOW_SCALAR_BYTES", cost.scalar_bytes + 1)
    assert decision_flows(tree, [path])["unmapped_paths"] == {"all": 1}
    monkeypatch.setattr(decision_report, "_MAX_FLOW_SCALAR_BYTES", cost.scalar_bytes)
    assert decision_flows(tree, [path])["unmapped_paths"] == {"all": 1}
    monkeypatch.setattr(decision_report, "_MAX_FLOW_SCALAR_BYTES", cost.scalar_bytes - 1)
    with pytest.raises(ValueError, match="aggregate scalar-byte limit"):
        decision_flows(tree, [path])


def test_bounded_integer_in_a_tree_reaches_hash_validation() -> None:
    tree = _tree()
    payload = {key: value for key, value in tree.items() if key != "tree_sha256"}
    payload["future_integer"] = 1 << 1_024
    tree = _hashed(payload)

    assert decision_flows(tree, [_path(_PRIOR_TREE_DIGEST)])["unmapped_paths"] == {"all": 1}


def test_huge_integer_fails_before_tree_hash_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _tree()
    tree["future_integer"] = 1 << 100_000
    monkeypatch.setattr(decision_report, "_MAX_FLOW_SCALAR_BYTES", 1_000)

    def unexpected_hash(_tree):
        raise AssertionError("tree serialization ran before integer preflight")

    monkeypatch.setattr(decision_report, "_tree_sha256", unexpected_hash)
    with pytest.raises(ValueError, match="aggregate scalar-byte limit"):
        decision_flows(tree, [_path(_PRIOR_TREE_DIGEST)])


def test_mapping_keys_and_bytes_count_toward_the_aggregate_byte_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _tree()
    path = _path(_PRIOR_TREE_DIGEST)
    path["future"] = {"wide-key": b"payload"}
    baseline = _path(_PRIOR_TREE_DIGEST)
    baseline_cost = decision_report._flow_request_cost(
        tree,
        [baseline],
        None,
        None,
        work_limit=10_000,
        byte_limit=10_000,
    )
    full_cost = decision_report._flow_request_cost(
        tree,
        [path],
        None,
        None,
        work_limit=10_000,
        byte_limit=10_000,
    )
    assert full_cost.scalar_bytes - baseline_cost.scalar_bytes == len("futurewide-keypayload")
    monkeypatch.setattr(decision_report, "_MAX_FLOW_SCALAR_BYTES", full_cost.scalar_bytes - 1)

    with pytest.raises(ValueError, match="aggregate scalar-byte limit"):
        decision_flows(tree, [path])


def test_huge_string_keys_and_values_fail_the_byte_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _path(_PRIOR_TREE_DIGEST)
    path["future"] = {"k" * 2_000: "v" * 2_000}
    monkeypatch.setattr(decision_report, "_MAX_FLOW_SCALAR_BYTES", 1_000)

    with pytest.raises(ValueError, match="aggregate scalar-byte limit"):
        decision_flows(_tree(), [path])


def test_wide_and_cyclic_nested_values_fail_under_stable_preflight_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _tree()
    wide = _path(_PRIOR_TREE_DIGEST)
    wide["future"] = [0] * 1_000
    monkeypatch.setattr(decision_report, "_MAX_FLOW_WORK_UNITS", 100)
    with pytest.raises(ValueError, match="structural work limit"):
        decision_flows(tree, [wide])

    monkeypatch.setattr(decision_report, "_MAX_FLOW_WORK_UNITS", 10_000)
    cyclic: list[object] = []
    cyclic.append(cyclic)
    path = _path(_PRIOR_TREE_DIGEST)
    path["future"] = cyclic
    with pytest.raises(ValueError, match="cyclic value"):
        decision_flows(tree, [path])


def test_custom_container_hooks_are_never_invoked() -> None:
    class Trap(list):
        called = False

        def __len__(self):
            type(self).called = True
            raise AssertionError("custom hook invoked")

        def __iter__(self):
            type(self).called = True
            raise AssertionError("custom hook invoked")

    class TrapDict(dict):
        called = False

        def items(self):
            type(self).called = True
            raise AssertionError("custom hook invoked")

        def values(self):
            type(self).called = True
            raise AssertionError("custom hook invoked")

    class TrapInt(int):
        called = False

        def bit_length(self):
            type(self).called = True
            raise AssertionError("custom hook invoked")

    with pytest.raises(ValueError, match="flow paths must be a sequence"):
        decision_flows(_tree(), Trap([_path(_PRIOR_TREE_DIGEST)]))
    assert Trap.called is False

    for custom in (Trap(), TrapDict(), TrapInt(1)):
        path = _path(_PRIOR_TREE_DIGEST)
        path["future"] = custom
        with pytest.raises(ValueError, match="unsupported value type"):
            decision_flows(_tree(), [path])
    assert Trap.called is False
    assert TrapDict.called is False
    assert TrapInt.called is False


def test_flow_count_and_label_boundaries_are_enforced() -> None:
    tree = _tree()
    path = _path(tree["tree_sha256"])

    with pytest.raises(ValueError, match="between 1 and 100000 paths"):
        decision_flows(tree, [])
    with pytest.raises(ValueError, match="between 1 and 100000 paths"):
        decision_flows(tree, [path] * 100_001)
    with pytest.raises(ValueError, match="1 labels for 2 paths"):
        decision_flows(tree, [path, path], labels=["only-one"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("location", ["tree", "path", "thresholds"])
def test_non_finite_values_are_rejected_before_flow_validation_or_hashing(value, location):
    tree = _tree()
    path = _path(tree["tree_sha256"])
    thresholds = None
    if location == "tree":
        tree = _hashed(
            {**{k: v for k, v in tree.items() if k != "tree_sha256"}, "future_value": value}
        )
        path["tree_sha256"] = tree["tree_sha256"]
    elif location == "path":
        path["future_value"] = value
    else:
        thresholds = {"t:volume": value}
    with pytest.raises(ValueError, match="non-finite"):
        decision_flows(tree, [path], thresholds=thresholds)


@pytest.mark.parametrize("value", [0.0, -0.5, 1e200])
def test_finite_additive_numeric_fields_still_allow_flow_counting(value):
    tree = _tree()
    path = _path(tree["tree_sha256"])
    path["future_value"] = value
    assert decision_flows(tree, [path])["leaves"]["match"] == {"all": 1}


@pytest.mark.parametrize(
    "value",
    ["nan", "inf", "-inf", "1e10000", pytest.param(1 << 100_000, id="large-native-integer")],
)
def test_threshold_conversion_cannot_introduce_a_non_finite_value(value):
    tree = _tree()
    with pytest.raises(ValueError, match="finite"):
        decision_flows(tree, [_path(tree["tree_sha256"])], thresholds={"t:volume": value})


@pytest.mark.parametrize("value", ["0.0", "0.5", "1"])
def test_existing_finite_threshold_conversion_remains_compatible(value):
    tree = _tree()
    result = decision_flows(tree, [_path(tree["tree_sha256"])], thresholds={"t:volume": value})
    assert result["leaves"]["match"] == {"all": 1}
