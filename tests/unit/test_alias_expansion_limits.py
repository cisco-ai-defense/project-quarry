# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Forward aliases cannot expand a short rule into unbounded executable syntax."""

from dataclasses import replace

import pytest

from quarry import DSLScanner
from quarry.config import QuarryConfig
from quarry.engine import desugar
from quarry.engine.budget import DFAConstructionLimit, WorkBudget, spending
from quarry.engine.diagnostics import UnsupportedConstruct
from quarry.types.dsl import BinaryOp, Literal
from quarry.types.exceptions import RuleValidationError


def source(depth):
    parts = ["(true as a0)"]
    parts += [f"((a{i - 1} or a{i - 1}) as a{i})" for i in range(1, depth + 1)]
    return 'rule "Aliases" condition: ' + " and ".join([*parts, f"a{depth}"])


def test_expanded_alias_occurrences_are_bounded_before_further_rewriting(monkeypatch):
    monkeypatch.setattr(desugar, "_MAX_ALIAS_EXPANSION_NODES", 100, raising=False)
    with pytest.raises(RuleValidationError, match="alias expansion"):
        DSLScanner(QuarryConfig(config_version=2, quarry_rule_texts=(source(6),)))


def test_small_forward_aliases_keep_their_verdict(monkeypatch):
    monkeypatch.setattr(desugar, "_MAX_ALIAS_EXPANSION_NODES", 100, raising=False)
    scanner = DSLScanner(QuarryConfig(config_version=2, quarry_rule_texts=(source(3),)))
    result = scanner.scan_data({"messages": [{"role": "user", "content": "unrelated"}]})
    assert result.is_complete and result.rule_matches == ["Aliases"]


def test_a_shared_alias_dag_is_counted_without_expanding_its_paths(monkeypatch):
    monkeypatch.setattr(desugar, "_MAX_ALIAS_EXPANSION_NODES", 100)
    root = Literal(True)
    for _ in range(50):
        root = BinaryOp(left=root, operator="or", right=root)
    calls = 0
    original = desugar.iter_condition_children

    def counted(node):
        nonlocal calls
        calls += 1
        return original(node)

    monkeypatch.setattr(desugar, "iter_condition_children", counted)
    with pytest.raises(UnsupportedConstruct, match="alias expansion"):
        desugar._check_alias_expansion([root])
    assert calls <= 102


def test_alias_occurrence_work_is_charged_to_the_enclosing_compile():
    root = Literal(True)
    for _ in range(4):
        root = BinaryOp(left=root, operator="or", right=root)
    budget = WorkBudget(10)
    with spending(budget), pytest.raises(DFAConstructionLimit, match="work budget"):
        desugar._check_alias_expansion([root])
    assert budget.used > budget.allowed


@pytest.mark.parametrize("with_alias", [False, True])
@pytest.mark.parametrize(
    "clause",
    [
        'any m in messages.transform("strip") where true',
        'all m in messages.transform("strip") where true',
        "any x in system_prompt where true",
        "any x in unknown where true",
    ],
)
def test_false_prefix_does_not_bypass_collection_admission(tmp_path, with_alias, clause):
    cfg = QuarryConfig.from_mapping(
        {"config_version": 2, "transforms": {"strip": {"operations": [{"op": "strip"}]}}},
        tmp_path,
        require_rules=False,
    )
    prefix = "false and (user_messages as ns) and " if with_alias else "false and "
    with pytest.raises(RuleValidationError, match="non-surface collection"):
        DSLScanner(
            replace(
                cfg, quarry_rule_texts=('rule "Guarded" condition: ' + prefix + "(" + clause + ")",)
            )
        )


def test_valid_folded_model_input_keeps_runtime_short_circuiting(tmp_path):
    from quarry.integrations.model_inference import ModelInferenceBackend
    from tests.unit.model_fakes import ModelAdapter, NoWaitLimiter, model_entry, passthrough_builder

    cfg = QuarryConfig.from_mapping(
        {"config_version": 2, "models": {"judge": model_entry(score=True)}},
        tmp_path,
        require_rules=False,
    )
    adapter = ModelAdapter()
    backend = ModelInferenceBackend(
        cfg,
        adapters={"fake": adapter},
        message_builders={"test": passthrough_builder},
        rate_limiter=NoWaitLimiter(),
    )
    source = 'rule "Guarded" condition: false and (messages as ms) and model("judge").score(message("user",ms[99].content),"criterion") > 0.5'
    result = DSLScanner(
        replace(cfg, quarry_rule_texts=(source,)), model_inference_backend=backend
    ).scan_data({"messages": [{"role": "user", "content": "x"}]})
    assert result.is_complete and not result.rule_matches
    assert adapter.score_calls == 0
