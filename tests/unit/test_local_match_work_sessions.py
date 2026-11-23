# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Rule groups have explicit owners and never borrow another scan's allowance."""

import pytest

from quarry.safety.local_match_work import (
    LocalMatchWorkConfig,
    LocalMatchWorkError,
    LocalMatchWorkGroup,
    LocalMatchWorkSession,
)


def group(label="group", rules=("A",)):
    return LocalMatchWorkGroup(label, rules, 1_000_000_000)


@pytest.mark.parametrize(
    "groups",
    [(), (group(),) * 65, (object(),), (group(), group()), (group("first"), group("second"))],
)
def test_group_tables_reject_unbounded_ambiguous_or_invalid_owners(groups):
    with pytest.raises(ValueError):
        LocalMatchWorkConfig(groups)


def test_total_selected_rule_count_is_bounded():
    groups = tuple(group(str(i), tuple(f"{i}:{n}" for n in range(256))) for i in range(5))
    with pytest.raises(ValueError, match="1024"):
        LocalMatchWorkConfig(groups)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"extra": []},
        {"groups": [], "extra": True},
        {"groups": []},
        {"groups": [{"label": "group", "rules": ["A"], "extra": 1}]},
        {"groups": [{"label": "group", "rules": ["A"], "allowance_cpu_ns": 1, "extra": True}]},
    ],
)
def test_strict_mapping_rejects_missing_unknown_or_empty_configuration(raw):
    with pytest.raises(ValueError):
        LocalMatchWorkConfig.from_mapping(raw)


def test_configuration_mapping_detaches_mutable_inputs_and_defaults_to_shadow():
    rules = ["A"]
    raw = {"groups": [{"label": "group", "rules": rules, "allowance_cpu_ns": 100}]}
    config = LocalMatchWorkConfig.from_mapping(raw)
    rules.clear()
    raw["groups"].clear()
    assert config.groups[0].rules == ("A",)
    assert config.groups[0].mode == "shadow"


def test_session_revalidates_a_forged_configuration_before_allocating_owners():
    config = LocalMatchWorkConfig((group(),))
    object.__setattr__(config, "groups", ())
    with pytest.raises(ValueError):
        LocalMatchWorkSession(config, known_rules=("A",))


def test_unknown_selected_rules_fail_before_work_starts():
    with pytest.raises(ValueError, match="unknown rule"):
        LocalMatchWorkSession(LocalMatchWorkConfig((group(),)), known_rules=("B",))


def test_rules_share_only_their_selected_group_within_the_same_scan():
    config = LocalMatchWorkConfig((group("first", ("A", "B")), group("second", ("C",))))
    first = LocalMatchWorkSession(config, known_rules=("A", "B", "C", "Other"))
    second = LocalMatchWorkSession(config, known_rules=("A", "B", "C", "Other"))
    assert first.owner_for("A") is first.owner_for("B")
    assert first.owner_for("A") is not first.owner_for("C")
    assert first.owner_for("A") is not second.owner_for("A")
    assert first.owner_for("Other") is None
    assert not first.report()["complete"]
    first.close()
    assert first.report()["complete"]
    assert first.report()["groups"][0]["value_cpu_ns"] == 0
    assert not second.report()["complete"]
    with pytest.raises(LocalMatchWorkError):
        first.owner_for("A")
