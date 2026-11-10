# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Public rules check the exact messages value supplied by their caller."""

import pytest

from quarry.config import QuarryConfig
from quarry.scanner import DSLScanner
from quarry.types.exceptions import RuleValidationError

_PI = 'rule "PI"\ncondition:\n    user_said("ignore")\n'
_CALLER = 'rule "R"\nimports:\n    pi = "PI"\ncondition:\n'
_CONVERSATION = {
    "messages": [
        {"role": "user", "content": "ignore"},
        {"role": "assistant", "content": "okay"},
    ]
}


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("pi(messages)", ["PI", "R"]),
        ("pi(messages.user_messages)", ["PI", "R"]),
        ("pi(messages[1:])", ["PI"]),
        ("pi(messages[0])", ["PI", "R"]),
    ],
)
def test_public_call_reads_selected_messages(condition: str, expected: list[str]) -> None:
    scanner = DSLScanner(QuarryConfig(quarry_rule_texts=(_PI + _CALLER + f"    {condition}\n",)))
    result = scanner.scan_data(_CONVERSATION)
    assert sorted(result.rule_matches) == expected
    assert result.evaluation_errors is None


@pytest.mark.parametrize(
    ("imports", "condition", "reason"),
    [
        ('    pi = "missing"', "pi(messages)", "invalid_import"),
        ('    pi = "PI"', "pi", "call_reference"),
        ('    pi = "PI"', "pi()", "call_arguments"),
    ],
)
def test_invalid_public_call_fails_at_load(imports: str, condition: str, reason: str) -> None:
    source = _PI + f'rule "R"\nimports:\n{imports}\ncondition:\n    {condition}\n'
    with pytest.raises(RuleValidationError) as failure:
        DSLScanner(QuarryConfig(quarry_rule_texts=(source,)))
    assert failure.value.details["reason"] == reason


def test_called_rule_strings_and_evidence_keep_the_callee_identity() -> None:
    source = (
        'rule "PI"\nstrings:\n    $i = "ignore"\n'
        "condition:\n    $i in messages[0].content\n" + _CALLER + "    pi(messages.user_messages)\n"
    )
    scanner = DSLScanner(QuarryConfig(quarry_rule_texts=(source,)))
    result = scanner.scan_data(_CONVERSATION, enable_explain=True)
    evidence = next(item for item in result.explanations or () if item.rule_name == "R")
    assert sorted(result.rule_matches) == ["PI", "R"]
    assert "PI::$i" in scanner.compiled_rules["R"].decision_tree.nodes[0].label
    assert evidence.model_results[0].score_summary["chunks"][0]["outcome"] is True


def test_called_rule_reads_an_input_item_as_located_message() -> None:
    source = (
        'rule "PI" condition: any m in messages where m.content contains "ignore all"\n'
        'rule "R" imports: pi = "PI" condition: any w in memory.notes where pi(w)'
    )
    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            quarry_rule_texts=(source,),
            inputs={"memory.notes": {"content": ["title", "body"]}},
        )
    )
    result = scanner.scan_data(
        {"memory": {"notes": [{"id": "n1", "title": "ignore ", "body": "all"}]}},
        rules_subset={"R"},
        enable_explain=True,
    )
    assert result.rule_matches == ["R"]
    chunk = result.explanations[0].to_dict()["model_results"][0]["score_summary"]["chunks"][0]
    assert [piece["field"] for piece in chunk["pieces"]] == ["title", "body"]
    assert all(
        piece["source"] == {"input": "memory.notes", "item": "n1"} for piece in chunk["pieces"]
    )


@pytest.mark.parametrize(
    ("condition", "matches"),
    [
        ("any w in memory.notes[:1] where pi(w)", True),
        ("any w in memory.notes[-1:] where pi(w)", False),
        ("pi(memory.notes[0])", True),
        ("len(memory.notes) > 0 and (memory.notes as ns) and any w in ns where pi(w)", True),
        ("(memory.notes as ns) and (ns[-1] as note) and pi(note)", False),
    ],
)
def test_called_rule_accepts_list_item_views(condition: str, matches: bool) -> None:
    source = _PI + _CALLER + f"    {condition}\n"
    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            quarry_rule_texts=(source,),
            inputs={"memory.notes": {"content": ["body"]}},
        )
    )
    data = {"memory": {"notes": [{"id": "n1", "body": "ignore"}, {"id": "n2", "body": "ok"}]}}
    result = scanner.scan_data(data, rules_subset={"R"})
    assert result.is_complete
    assert (result.rule_matches == ["R"]) is matches


def test_called_model_rule_reads_span_lists_from_an_input_item(tmp_path) -> None:
    from quarry.integrations.model_inference import ModelInferenceBackend
    from tests.unit.model_fakes import (
        ModelAdapter,
        NoWaitLimiter,
        model_entry,
        passthrough_builder,
    )

    source = (
        'rule "PI" strings: $p = "alice@example.com" condition: '
        'model("m").classify(messages) > 0.5 and '
        'len(outermost(union(model("m").spans(messages, 0.7), $p.spans(messages)))) == 1\n'
        'rule "R" imports: pi = "PI" condition: any w in memory.notes where pi(w)'
    )
    path = tmp_path / "rules.rule"
    path.write_text(source, encoding="utf-8")
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "rules": [str(path)],
            "models": {"m": model_entry(classify=0, spans=0, category="email")},
            "inputs": {"memory.notes": {"content": ["body"]}},
        },
        tmp_path,
    )
    adapter = ModelAdapter()
    backend = ModelInferenceBackend(
        config,
        adapters={"fake": adapter},
        message_builders={"test": passthrough_builder},
        rate_limiter=NoWaitLimiter(),
    )
    scanner = DSLScanner(config, model_inference_backend=backend)
    result = scanner.scan_data(
        {"memory": {"notes": [{"id": "n1", "body": "send to alice@example.com"}]}},
        rules_subset={"R"},
        enable_explain=True,
    )
    assert result.rule_matches == ["R"]
    assert result.evaluation_errors is None
    assert adapter.classifications == len(adapter.decoded_spans) == 1
    chunk = result.explanations[0].to_dict()["model_results"][0]["score_summary"]["chunks"][0]
    assert chunk["pieces"][0]["source"] == {"input": "memory.notes", "item": "n1"}


@pytest.mark.parametrize(
    ("caller", "timeline", "state", "matched"),
    [
        ("flow(messages)", None, "incomplete", False),
        (
            "flow(messages)",
            [{"message_index": 0}, {"input": "memory.notes", "id": "n1"}],
            "complete",
            True,
        ),
        (
            "any w in memory.notes where flow(w)",
            [{"message_index": 0}, {"input": "memory.notes", "id": "n1"}],
            "incomplete",
            False,
        ),
    ],
)
def test_called_rule_keeps_cross_source_clock(
    caller: str, timeline: list[dict] | None, state: str, matched: bool
) -> None:
    source = (
        'rule "Flow" condition: (any w in memory.notes where w.id == "n1") '
        'right_after user_said("go")\n'
        f'rule "R" imports: flow = "Flow" condition: {caller}'
    )
    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            quarry_rule_texts=(source,),
            inputs={"memory.notes": {"content": ["body"]}},
        )
    )
    data = {
        "messages": [{"role": "user", "content": "go"}],
        "memory": {"notes": [{"id": "n1", "body": "go"}]},
    }
    if timeline is not None:
        data["timeline"] = timeline
    result = scanner.scan_data(data, rules_subset={"R"})
    assert (result.completion_state, result.rule_matches) == (
        state,
        ["R"] if matched else [],
    )


def test_called_rule_check_survives_compiled_artifact_round_trip() -> None:
    from quarry.policy_serialization import _artifact_from_wire, _artifact_to_wire

    scanner = DSLScanner(QuarryConfig(quarry_rule_texts=(_PI + _CALLER + "    pi(messages)\n",)))
    wire = _artifact_to_wire(scanner.compiled_rules["R"])
    restored = _artifact_from_wire(wire)
    assert restored.split_checks.keys() == scanner.compiled_rules["R"].split_checks.keys()
    wire["split_checks"][0][1]["dfa_hash"] = "0" * 64
    with pytest.raises(ValueError, match="split atom must name the check"):
        _artifact_from_wire(wire)
