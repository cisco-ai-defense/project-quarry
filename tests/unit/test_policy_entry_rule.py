# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""A public policy entry point retains its reachable helper generation."""

import json
from unittest.mock import patch

import pytest

from quarry import InputAdapter, compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.safety.local_match_work import LocalMatchWorkConfig
from quarry.types.exceptions import RuleValidationError
from tests.unit.test_policy_parameters import _runtime

LEAF = 'rule "Leaf" condition: user_said("amber")'
ENTRY = 'rule "Entry" imports: leaf = "Leaf" condition: leaf(user_messages)'
DATA = {"messages": [{"role": "system", "content": "blue"}, {"role": "user", "content": "amber"}]}


def reload_without_compilation(policy, **kwargs):
    wire = policy.to_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Loading a selected entry parsed or compiled its source")

    with (
        patch("quarry.dsl.parser.DSLParser.parse", forbidden),
        patch("quarry.dsl.parser.DSLParser.parse_many", forbidden),
        patch("quarry.scanner.compile_rule", forbidden),
    ):
        restored = load_compiled_policy(wire, **kwargs)
    assert restored.to_bytes() == wire
    assert restored.policy_digest == policy.policy_digest
    return restored


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("entry", ["Leaf", "Entry"])
@pytest.mark.parametrize("entry_first", [False, True])
def test_selecting_an_entry_is_independent_of_definition_order(restored, entry, entry_first):
    source = "\n".join([ENTRY, LEAF] if entry_first else [LEAF, ENTRY])
    policy = compile_policy(source, entry_rule=entry)
    if restored:
        policy = reload_without_compilation(policy)
    result = policy.scan(DATA, explain=True)
    assert result.is_complete and result.rule_matches == [entry]


@pytest.mark.parametrize("helper_source", ["inline", "file"])
def test_runtime_configuration_supplies_helper_rules(tmp_path, helper_source):
    if helper_source == "file":
        helper = tmp_path / "helper.rule"
        helper.write_text(LEAF)
        config = QuarryConfig(config_version=2, quarry_rules_paths=(helper,))
    else:
        config = QuarryConfig(config_version=2, quarry_rule_texts=(LEAF,))
    policy = compile_policy(ENTRY, entry_rule="Entry", runtime_config=config)
    if helper_source == "file":
        helper.unlink()
    restored = reload_without_compilation(policy, runtime_config=config)
    for item in [policy, restored]:
        result = item.scan(DATA)
        assert result.is_complete and result.rule_matches == ["Entry"]


def test_selected_entry_serializes_only_its_required_model_resources(tmp_path):
    config, backend, adapter, _ = _runtime(tmp_path)
    unused = 'rule "Unused" condition: model("judge").score(messages, "unselected") > 0.5'
    policy = compile_policy(
        "\n".join([LEAF, unused, ENTRY]),
        entry_rule="Entry",
        runtime_config=config,
        model_inference_backend=backend,
    )
    assert json.loads(policy.to_bytes())["models"] == {}
    restored = reload_without_compilation(policy)
    for item in [policy, restored]:
        result = item.scan(DATA)
        assert result.is_complete and result.rule_matches == ["Entry"]
    assert adapter.calls == 0


@pytest.mark.parametrize("restored", [False, True])
def test_a_selected_entry_calls_a_helper_on_declared_items(restored):
    config = QuarryConfig(config_version=2, inputs={"notes.entries": {"content": ["body"]}})
    source = (
        LEAF
        + '\nrule "Entry" imports: leaf = "Leaf" condition: any n in notes.entries where leaf(n)'
    )
    policy = compile_policy(source, entry_rule="Entry", runtime_config=config)
    if restored:
        policy = reload_without_compilation(policy, runtime_config=config)
    result = policy.scan({"notes": {"entries": [{"id": "n", "body": "amber"}]}}, explain=True)
    assert result.is_complete and result.rule_matches == ["Entry"]


def test_selected_entry_keeps_native_adapter_identity():
    class Native(InputAdapter):
        name, version = "entry-test", "1"
        sections = ("notes.entries",)

        def convert(self, native, wanted):
            assert wanted == frozenset({"notes.entries"})
            return {"notes.entries": native}

    config = QuarryConfig(
        config_version=2,
        inputs={"notes.entries": {"content": ["body"]}},
        adapters={"native": {"adapter": "entry-test"}},
    )
    adapters = {"entry-test": Native()}
    source = (
        LEAF
        + '\nrule "Entry" imports: leaf = "Leaf" condition: any n in notes.entries where leaf(n)'
    )
    policy = compile_policy(
        source, entry_rule="Entry", runtime_config=config, input_adapters=adapters
    )
    restored = reload_without_compilation(policy, runtime_config=config, input_adapters=adapters)
    for item in [policy, restored]:
        result = item.scan({"native": [{"id": "n", "body": "amber"}]})
        assert result.is_complete and result.rule_matches == ["Entry"]
    changed = Native()
    changed.version = "2"
    with pytest.raises(ValueError, match="adapters differ"):
        load_compiled_policy(
            policy.to_bytes(), runtime_config=config, input_adapters={"entry-test": changed}
        )


@pytest.mark.parametrize("entry", ["", " ", 1, True, [], "x" * 257])
def test_invalid_entry_names_fail_before_loading_rules(entry):
    with (
        patch("quarry.policy.DSLScanner", side_effect=AssertionError("Must not load rules")),
        pytest.raises(ValueError, match="entry_rule"),
    ):
        compile_policy('rule "Entry" condition: true', entry_rule=entry)


def test_an_unknown_entry_is_rejected():
    with pytest.raises(ValueError, match="entry_rule"):
        compile_policy(LEAF + "\n" + ENTRY, entry_rule="Missing")


def test_a_module_without_an_explicit_entry_keeps_the_single_rule_requirement():
    with pytest.raises(ValueError, match="exactly one"):
        compile_policy(LEAF + "\n" + ENTRY)


def test_duplicate_names_in_configured_helpers_and_source_are_rejected():
    config = QuarryConfig(config_version=2, quarry_rule_texts=(LEAF,))
    with pytest.raises(RuleValidationError, match="Duplicate rule name"):
        compile_policy(LEAF + "\n" + ENTRY, entry_rule="Entry", runtime_config=config)


def test_default_compilation_keeps_configured_rule_sources_out_of_its_module(tmp_path):
    config = QuarryConfig(
        config_version=2,
        quarry_rules_paths=(tmp_path / "not-read.rule",),
        quarry_rule_texts=(LEAF,),
    )
    policy = compile_policy('rule "Entry" condition: true', runtime_config=config)
    assert policy.scan(DATA).rule_matches == ["Entry"]


def test_explicit_entry_for_a_single_rule_keeps_the_existing_wire_identity():
    source = 'rule "Entry" condition: user_said("amber")'
    original = compile_policy(source)
    explicit = compile_policy(source, entry_rule="Entry")
    assert original.policy_digest == explicit.policy_digest
    assert original.to_bytes() == explicit.to_bytes()


def test_selected_entry_keeps_its_named_local_work_allowance():
    work = LocalMatchWorkConfig.from_mapping(
        {"groups": [{"label": "entry", "rules": ["Entry"], "allowance_cpu_ns": 1_000_000_000}]}
    )
    config = QuarryConfig(config_version=2, local_match_work=work)
    policy = compile_policy(LEAF + "\n" + ENTRY, entry_rule="Entry", runtime_config=config)
    restored = reload_without_compilation(policy, runtime_config=config)
    for item in [policy, restored]:
        result = item.scan(DATA)
        assert result.is_complete and result.rule_matches == ["Entry"]


def test_work_groups_for_other_entry_points_do_not_create_an_unloadable_policy():
    work = LocalMatchWorkConfig.from_mapping(
        {"groups": [{"label": "helper", "rules": ["Leaf"], "allowance_cpu_ns": 1_000_000_000}]}
    )
    config = QuarryConfig(config_version=2, local_match_work=work)
    with pytest.raises(ValueError, match="unknown rule"):
        compile_policy(LEAF + "\n" + ENTRY, entry_rule="Entry", runtime_config=config)
