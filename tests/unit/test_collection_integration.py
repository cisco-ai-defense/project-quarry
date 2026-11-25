# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Span list reloads, source coordinates and aliases local to called rules."""

import pytest

from quarry import DSLScanner, compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.types.exceptions import RuleValidationError


@pytest.mark.parametrize(
    ("source", "count"),
    [
        ("union($short.spans(skill.instructions.body), $long.spans(skill.instructions.body))", 2),
        (
            "outermost(union($short.spans(skill.instructions.body), "
            "$long.spans(skill.instructions.body)))",
            1,
        ),
        ("union($short.spans(skill.instructions.body), $short.spans(skill.instructions.body))", 1),
    ],
)
def test_span_list_policy_reloads_with_the_original_input_coordinates(source, count):
    config = QuarryConfig(
        config_version=2, inputs={"skill.instructions": {"single": True, "content": ["body"]}}
    )
    rule = (
        'rule "Read"\nstrings:\n    $short = "ignore"\n    $long = "ignore previous"\n'
        f"condition: len({source}) == {count}\ncollect: [s for s in {source} if true]"
    )
    policy = compile_policy(rule, runtime_config=config)
    loaded = load_compiled_policy(policy.to_bytes(), runtime_config=config)
    assert loaded.to_bytes() == policy.to_bytes()
    for prepared in (policy, loaded):
        result = prepared.scan(
            {"skill": {"instructions": {"body": "prefix ignore previous suffix"}}}
        )
        assert result.is_complete and result.rule_matches == ["Read"]
        spans = [item.value for item in result.outputs["Read"].items]
        assert len(spans) == count
        assert all(s.to_dict()["source"] == {"input": "skill.instructions"} for s in spans)
        assert all(s.to_dict()["field"] == "body" and s.start == 7 for s in spans)
        if source.startswith("outermost"):
            assert spans[0].end == 22


@pytest.mark.parametrize("outer", ["ns", "outside"])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize(("body", "matched"), [("ignore", True), ("ordinary", False)])
@pytest.mark.parametrize("direct", [False, True])
def test_called_rule_aliases_remain_local_to_the_called_messages(
    tmp_path, outer, depth, body, matched, direct
):
    source = (
        'rule "PI" condition: (user_messages as ns) and '
        'any m in ns where m.content contains "ignore"\n'
    )
    previous = "PI"
    if depth == 2:
        source += (
            'rule "Middle" imports: pi = "PI" condition: (user_messages as ns) and pi(ns[0])\n'
        )
        previous = "Middle"
    call = f"pi({outer}[0])" if direct else f"any n in {outer} where pi(n)"
    source += (
        f'rule "Read" imports: pi = "{previous}" condition: (memory.notes as {outer}) and {call}'
    )
    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            quarry_rule_texts=(source,),
            inputs={"memory.notes": {"content": ["body"]}},
        )
    )
    result = scanner.scan_data(
        {"memory": {"notes": [{"id": "n", "body": body}]}}, rules_subset={"Read"}
    )
    assert result.is_complete and result.rule_matches == (["Read"] if matched else [])


def test_duplicate_aliases_in_the_same_and_chain_remain_rejected():
    with pytest.raises(RuleValidationError, match="names two values in one"):
        DSLScanner(
            QuarryConfig(
                config_version=2,
                quarry_rule_texts=(
                    'rule "Read" condition: (messages as ns) and (messages[1:] as ns) '
                    "and len(ns) > 0",
                ),
            )
        )
