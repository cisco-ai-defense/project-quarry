# Copyright 2025 Cisco Systems, Inc. and its affilitates
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

"""Public behavior for CompiledPolicy's durable representation.

A compiled policy serializes to deterministic bytes that carry no rule source and no physical
model name. Loading those bytes reproduces the original policy's verdicts; a document that is
from another version, was compiled against other runtime resources, or was altered so that it
could miss a match fails to load.
"""

from __future__ import annotations

import functools
import json
import sys
from collections.abc import Callable
from types import SimpleNamespace

import orjson
import pytest

from quarry import __version__, compile_policy, load_compiled_policy
from quarry.cache.async_cache import result_cache
from quarry.config import QuarryConfig
from quarry.engine.contracts import COMPILER_VERSION
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.policy_serialization import FORMAT_KIND, FORMAT_VERSION
from quarry.types.exceptions import RuleCompatibilityError
from quarry.types.results import Score, Span
from tests.unit.model_fakes import ModelAdapter, passthrough_builder


class _ModelAdapter:
    async def score(self, _config, _messages, *, timeout=None):
        return 0.8

    async def classify_messages(self, _config, _messages, _tools, *, timeout=None):
        return SimpleNamespace(chunks=[SimpleNamespace(probabilities=[0.8])])

    async def decode_spans(self, _config, _raw_result, messages, _threshold):
        content = messages[0].content
        start = content.index("alice@example.com")
        return [Span(0, start, start + len("alice@example.com"), Score(0.91))]

    async def tokenize(self, config, _messages, _tools, *, timeout=None):
        return {config.tokens.count_field: 1}


def _runtime(
    tmp_path,
    *,
    config_version: int = 2,
    model: str = "judge-v1",
    policy_text: str | None = None,
):
    mapping = {
        "config_version": config_version,
        "models": {
            "judge": {
                "adapter": "fake",
                "model": model,
                "category": "EMAIL",
                "classify": {"class_index": 0},
                "spans": {"class_index": 0},
                "score": {"message_builder": "test", "boundary_seed": 7},
                "tokens": {
                    "adapter": "fake",
                    "endpoint": None,
                    "extraction": {"count": "count"},
                },
            }
        },
    }
    if policy_text is not None:
        mapping["texts"] = {"policy": policy_text}
    config = QuarryConfig.from_mapping(mapping, tmp_path, require_rules=False)
    backend = ModelInferenceBackend(
        config,
        adapters={"fake": _ModelAdapter()},
        message_builders={"test": lambda _policy, messages, *_args: messages},
    )
    return config, backend


def test_round_trip_is_deterministic_and_carries_no_source_or_model_name(tmp_path) -> None:
    source = """rule "Serialized guardrail"
metadata:
    description = "Round-trip metadata"
parameters:
    threshold: number
condition:
    model("judge").classify(messages) >= params.threshold
"""
    config, backend = _runtime(tmp_path)
    policy = compile_policy(
        source,
        runtime_config=config,
        model_inference_backend=backend,
    )
    encoded = policy.to_bytes()
    document = orjson.loads(encoded)

    assert policy.to_bytes() == encoded
    assert set(document) == {
        "kind",
        "format_version",
        "config_version",
        "source_sha256",
        "policy_digest",
        "policy_parameters",
        "text_resources",
        "models",
        "rule",
    }
    assert (document["kind"], document["format_version"]) == (FORMAT_KIND, FORMAT_VERSION)
    assert document["rule"]["artifact"]["compiler_version"] == COMPILER_VERSION
    assert document["rule"]["metadata"] == {"description": "Round-trip metadata"}
    assert len(document["models"]["judge"]) == 64
    assert "judge-v1" not in encoded.decode()
    assert source not in encoded.decode()

    loaded = load_compiled_policy(
        memoryview(encoded), runtime_config=config, model_inference_backend=backend
    )
    conversation = {"messages": [{"role": "user", "content": "inspect this"}]}
    expected = policy.scan(conversation, {"threshold": 0.7})
    actual = loaded.scan(conversation, {"threshold": 0.7})

    assert loaded.to_bytes() == encoded
    assert actual.rule_matches == expected.rule_matches == ["Serialized guardrail"]
    assert actual.policy_digest == policy.policy_digest == document["policy_digest"]
    assert actual.parameter_evidence == {
        "threshold": {"type": "number", "binding": "scan", "value": 0.7},
    }


def test_serialization_rejects_text_parameters(tmp_path) -> None:
    source = """rule "Private instructions"
parameters:
    instructions: text policy
condition:
    model("judge").score(messages, params.instructions) >= 0.7
"""
    config, backend = _runtime(tmp_path)
    policy = compile_policy(
        source,
        policy_parameters={"instructions": "private instructions"},
        runtime_config=config,
        model_inference_backend=backend,
    )

    with pytest.raises(ValueError, match="Text parameters.*cannot serialize"):
        policy.to_bytes()


@functools.cache
def _compiled(source: str):
    """Compile each runtime-free policy once; a compiled policy is immutable, so rows share it."""
    return compile_policy(source)


def _user(*contents: str) -> list[dict]:
    return [{"role": "user", "content": content} for content in contents]


def _call(name: str, arguments: str = "{}", call_id: str | None = None) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id or name,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _tools(*definitions: tuple[str, str]) -> list[dict]:
    return [
        {"type": "function", "function": {"name": name, "description": description}}
        for name, description in definitions
    ]


_CAPTURE = """rule "Capture"
condition:
    any msg in user_messages where ("TOKEN" in msg.content as found) and msg.content contains found
"""
_PINNED_JOIN = """rule "Joined"
condition:
    any m in user_messages where
        ((any c in tool_calls where c.arguments.cmd contains m.content) after called_tool("y"))
"""
_TWO_FLOWS = """rule "Two flows"
condition:
    any m in user_messages where
        (called_tool("a") after(within: 20) m.content contains "x") and
        (called_tool("b") after(within: 20) m.content contains "x")
"""
_CLOSED = """rule "Closed parts"
condition:
    any m in user_messages where m.content contains "a" and
        (any n in assistant_messages where n.content contains "b" and called_tool("t"))
        and (any d in tools where d.description contains "D" and user_said("u"))
"""
_FILTERED = """rule "Filtered"
condition:
    any c in get_tool_calls("x") where
        ((any d in tool_calls where d.arguments.cmd contains c.arguments.cmd) after called_tool("y"))
        and not (any e in tool_calls where e.name == "q" and e.arguments.cmd contains c.arguments.cmd)
"""


def _closed_conversation(tool: str) -> dict:
    return {
        "messages": [
            *_user("a", "u"),
            {"role": "assistant", "content": "b"},
            _call(tool, call_id="c1"),
        ],
        "tools": _tools(("d", "D")),
    }


def _cmds(*commands: str) -> dict:
    return {
        "messages": [_call("sh", f'{{"cmd": "{cmd}"}}', f"c{i}") for i, cmd in enumerate(commands)]
    }


_HELLO_257 = _user(*["hello"] * 257)


@pytest.mark.parametrize(
    ("source", "cases"),
    [
        pytest.param(
            _CAPTURE,
            [({"messages": _user("prefix TOKEN suffix")}, ["Capture"])],
            id="captured-value",
        ),
        pytest.param(
            'rule "Two flows"\ncondition:\n    any m in user_messages where ((m.content contains "A"'
            ' or user_said("ok")) until\n        called_tool("y")) and m.content contains "B"\n',
            [
                ({"messages": [*_user("AB"), _call("x"), _call("y")]}, ["Two flows"]),
                ({"messages": [*_user("A"), _call("x"), _call("y")]}, []),
            ],
            id="element-pinned-per-candidate",
        ),
        pytest.param(
            'rule "Contained"\ncondition:\n    any outer in tool_calls where\n'
            "        (any later in tool_calls where outer.arguments.cmd contains later.arguments.cmd)"
            " after\n"
            "        (any earlier in tool_calls where outer.arguments.cmd contains earlier.arguments.cmd)\n",
            [(_cmds("a", "ab"), ["Contained"]), (_cmds("a", "b"), [])],
            id="join-to-a-pinned-containing-element",
        ),
        pytest.param(
            'rule "Tool pair"\ncondition:\n    (any reader in tools where reader.name == "read")\n'
            '    and (any writer in tools where writer.name == "write")\n',
            [
                (
                    {"messages": _user("hello"), "tools": _tools(("read", ""), ("write", ""))},
                    ["Tool pair"],
                )
            ],
            id="coherent-tool-quantifiers",
        ),
        pytest.param(
            'rule "Not every tool"\ncondition:\n    not (all t in tools where t.description contains "A")\n',
            [
                (
                    {"messages": _user("hello"), "tools": _tools(("a", "A"), ("b", "B"))},
                    ["Not every tool"],
                )
            ],
            id="all-tools",
        ),
        pytest.param(
            'rule "Some tool lacks A"\ncondition:\n'
            '    any t in tools where not (t.description contains "A")\n',
            [
                (
                    {"messages": _user("hello"), "tools": _tools(("a", "A"), ("b", "B"))},
                    ["Some tool lacks A"],
                ),
                ({"messages": _user("hello"), "tools": _tools(("a", "A"))}, []),
            ],
            id="negated-tool-body",
        ),
        pytest.param(
            _FILTERED,
            [
                (
                    {
                        "messages": [
                            _call("x", '{"cmd": "ls"}', "c1"),
                            _call("y", call_id="c2"),
                            _call("sh", '{"cmd": "ls -la"}', "c3"),
                        ]
                    },
                    ["Filtered"],
                ),
                (
                    {
                        "messages": [
                            _call("x", '{"cmd": "ls"}', "c1"),
                            _call("y", call_id="c2"),
                            _call("sh", '{"cmd": "ls -la"}', "c3"),
                            _call("q", '{"cmd": "ls"}', "c4"),
                        ]
                    },
                    [],
                ),
            ],
            id="element-filtered-by-tool-name",
        ),
        pytest.param(
            _CLOSED,
            [(_closed_conversation("t"), ["Closed parts"]), (_closed_conversation("other"), [])],
            id="closed-parts",
        ),
        pytest.param(
            'rule "Always"\ncondition:\n  true or any m in user_messages where m.content contains "A"\n',
            [({"messages": _user("x")}, ["Always"])],
            id="simplified-to-a-constant",
        ),
        pytest.param(
            _TWO_FLOWS,
            [
                ({"messages": _HELLO_257}, []),
                ({"messages": [*_HELLO_257, *_user("x"), _call("a"), _call("b")]}, ["Two flows"]),
            ],
            id="element-own-step-test-keeps-a-long-scan-complete",
        ),
    ],
)
def test_a_loaded_policy_reproduces_the_compiled_verdicts(source, cases) -> None:
    compiled = _compiled(source)
    loaded = load_compiled_policy(bytes(compiled.to_bytes()))

    loaded_results = [loaded.scan(conversation) for conversation, _ in cases]
    compiled_results = [compiled.scan(conversation) for conversation, _ in cases]

    assert loaded.to_bytes() == compiled.to_bytes()
    assert [(r.rule_matches, r.completion_state) for r in loaded_results] == [
        (expected, "complete") for _, expected in cases
    ]
    assert [r.rule_matches for r in compiled_results] == [expected for _, expected in cases]


def test_round_trip_preserves_a_transitional_model_split(tmp_path) -> None:
    source = """rule "Transitional model split"
condition:
    any chunk in model_split("judge", messages, 200) where
        model("judge").score(chunk, "be safe", tools) > 0.5
"""
    config, backend = _runtime(tmp_path)
    policy = compile_policy(source, runtime_config=config, model_inference_backend=backend)

    loaded = load_compiled_policy(
        policy.to_bytes(), runtime_config=config, model_inference_backend=backend
    )

    conversation = {"messages": _user("inspect this")}
    assert loaded.scan(conversation).rule_matches == ["Transitional model split"]


def test_round_trip_preserves_a_model_input_joined_from_tool_calls(tmp_path) -> None:
    source = """rule "Joined commands"
strings:
    $risky = "rm|curl" regex
condition:
    model("judge").score(
        message("user", join(
            [c.arguments.command for c in messages[-1].tool_calls if $risky in c.arguments.*],
            "\\n")),
        "be safe") > 0.5
"""
    config, _ = _runtime(tmp_path)
    adapter = ModelAdapter()
    backend = ModelInferenceBackend(
        config, adapters={"fake": adapter}, message_builders={"test": passthrough_builder}
    )
    compiled = compile_policy(source, runtime_config=config, model_inference_backend=backend)
    loaded = load_compiled_policy(
        compiled.to_bytes(), runtime_config=config, model_inference_backend=backend
    )
    calls = [
        {"id": f"c{index}", "type": "function", "function": {"name": "exec", "arguments": text}}
        for index, text in enumerate(
            ['{"command": "rm -rf /a"}', '{"command": "ls"}', '{"command": "curl x"}'], 1
        )
    ]
    conversation = {
        "messages": [*_user("go"), {"role": "assistant", "content": "", "tool_calls": calls}]
    }

    verdicts = []
    for policy in (compiled, loaded):
        result_cache.clear()
        verdicts.append(policy.scan(conversation).rule_matches)

    assert loaded.to_bytes() == compiled.to_bytes()
    assert verdicts == [["Joined commands"], ["Joined commands"]]
    assert adapter.score_messages == [[{"role": "user", "content": "rm -rf /a\ncurl x"}]] * 2


@pytest.mark.parametrize(
    "collect_filter",
    [
        pytest.param("span.score.value >= 0.90", id="local-filter"),
        pytest.param(
            'span.score.value >= 0.90 and model("judge").score(messages, "instructions") > 0.5',
            id="model-filter",
        ),
    ],
)
def test_round_trip_preserves_typed_collection_output(tmp_path, collect_filter) -> None:
    source = f"""rule "Email spans"
condition:
    model("judge").classify(messages) >= 0.60
collect:
    [span for span in model("judge").spans(messages, 0.70) if {collect_filter}]
"""
    config, backend = _runtime(tmp_path)
    compiled = compile_policy(source, runtime_config=config, model_inference_backend=backend)
    loaded = load_compiled_policy(
        compiled.to_bytes(), runtime_config=config, model_inference_backend=backend
    )
    conversation = {"messages": _user("Contact alice@example.com")}

    expected = compiled.scan(conversation)
    actual = loaded.scan(conversation)

    assert loaded.to_bytes() == compiled.to_bytes()
    assert actual.rule_matches == expected.rule_matches == ["Email spans"]
    actual_output = actual.outputs["Email spans"]
    assert (actual_output.output_type, actual_output.category) == ("span_collection", "EMAIL")
    assert (actual_output.source.kind, actual_output.source.name) == ("model", "judge")
    assert [item.value for item in actual_output.items] == [
        item.value for item in expected.outputs["Email spans"].items
    ]


def _set_format_version(offset: int) -> Callable[[dict], None]:
    def mutate(document: dict) -> None:
        document["format_version"] = FORMAT_VERSION + offset

    return mutate


def _set_compiler_version(document: dict) -> None:
    document["rule"]["artifact"]["compiler_version"] = "future-compiler"


def _set_compiler_version_before_the_split_rework(document: dict) -> None:
    document["rule"]["artifact"]["compiler_version"] = "compiler-4"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(_set_format_version(1), "format_version .* unsupported", id="newer-format"),
        pytest.param(_set_format_version(-1), "format_version .* unsupported", id="older-format"),
        pytest.param(_set_compiler_version, "compiler_version .* unsupported", id="other-compiler"),
        # compiler-4 lowered split's third argument as tools; its artifacts must be recompiled.
        pytest.param(
            _set_compiler_version_before_the_split_rework,
            "compiler_version .* unsupported",
            id="compiler-before-the-split-rework",
        ),
    ],
)
def test_load_rejects_another_format_or_compiler_version(mutate, message) -> None:
    document = orjson.loads(compile_policy('rule "Versioned"\ncondition: true').to_bytes())
    mutate(document)

    with pytest.raises(ValueError, match=message):
        load_compiled_policy(orjson.dumps(document))


def test_load_rejects_a_policy_compiled_before_its_requirement_was_enforced() -> None:
    document = orjson.loads(compile_policy('rule "Legacy policy"\ncondition: true').to_bytes())
    document["rule"]["metadata"]["requires_quarry"] = ">=9999"

    with pytest.raises(RuleCompatibilityError) as exc_info:
        load_compiled_policy(orjson.dumps(document))

    error = exc_info.value
    assert error.running_version == __version__
    assert [
        (issue.rule_name, issue.source, issue.declaration, issue.reason) for issue in error.issues
    ] == [("Legacy policy", "compiled-policy", ">=9999", "runtime_below_minimum")]


def test_serialization_rejects_text_resources(tmp_path) -> None:
    source = """rule "Text reference"
condition:
    llm_score(
        "judge",
        text_ref("policy"),
        message("system", "policy").add_message("user", "inspect")
    ) >= 0.7 and
    any chunk in llm_split_conversation("judge", user_messages[-1:], 200) where
        llm_score("judge", text_ref("policy"), chunk) >= 0.7
"""
    config, backend = _runtime(tmp_path, config_version=1, policy_text="original policy")
    policy = compile_policy(source, runtime_config=config, model_inference_backend=backend)

    with pytest.raises(ValueError, match="text_ref resources.*cannot serialize"):
        policy.to_bytes()


def test_load_rejects_a_changed_model(tmp_path) -> None:
    source = 'rule "Model identity"\ncondition: model("judge").classify(messages) > 0.5\n'
    config, backend = _runtime(tmp_path)
    policy = compile_policy(source, runtime_config=config, model_inference_backend=backend)
    changed_config, changed_backend = _runtime(tmp_path, model="judge-v2")

    with pytest.raises(ValueError, match="runtime model configuration differs .* for: judge"):
        load_compiled_policy(
            policy.to_bytes(),
            runtime_config=changed_config,
            model_inference_backend=changed_backend,
        )


# ── tampered documents: anything malformed or inconsistent, which could hide a match, fails to load ──


def _artifact(document: dict) -> dict:
    return document["rule"]["artifact"]


def _atom_with_patterns(document: dict, patterns: list[str]) -> str:
    return next(
        spec["atom_id"]
        for spec in _artifact(document)["atom_specs"]
        if spec["patterns"] == patterns
    )


def _drop_a_closed_automaton(document: dict) -> None:
    _artifact(document)["closed_conditions"].pop()


def _closed_part_reads_itself(document: dict) -> None:
    atom_id, dfa = _artifact(document)["closed_conditions"][0]
    dfa["atoms"].append(atom_id)


def _add_value_plan_ordering_fields(document: dict) -> None:
    _artifact(document)["capture_plan"].update(ordered=False, within=None)


def _filter_is_the_candidate_atom(document: dict) -> None:
    plan = _artifact(document)["capture_plan"]
    plan["filter_atom_id"] = plan["at_atom_id"]


def _filter_is_another_name_test(document: dict) -> None:
    _artifact(document)["capture_plan"]["filter_atom_id"] = _atom_with_patterns(document, ["y"])


def _own_step_reads_the_candidate_atom(document: dict) -> None:
    plan = _artifact(document)["capture_plan"]
    plan["own_step"] = {"kind": "atom", "atom_id": plan["at_atom_id"]}


def _own_step_reads_a_tool_call_atom(document: dict) -> None:
    _artifact(document)["capture_plan"]["own_step"] = {
        "kind": "atom",
        "atom_id": _atom_with_patterns(document, ["a"]),
    }


_DFA = ("rule", "artifact", "dfa")
_TOOLS = ("rule", "artifact", "tool_quantifiers")
_PLAN = ("rule", "artifact", "capture_plan")


def _put(*path, value) -> Callable[[dict], None]:
    """Replace the value at ``path`` (keys and indexes from the document root)."""

    def mutate(document: dict) -> None:
        target = document
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _append(*path, value) -> Callable[[dict], None]:
    def mutate(document: dict) -> None:
        rows = document
        for key in path:
            rows = rows[key]
        rows.append(value)

    return mutate


def _drop(*path) -> Callable[[dict], None]:
    def mutate(document: dict) -> None:
        target = document
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return mutate


def _duplicate_first(*path, renamed_to: str | None = None) -> Callable[[dict], None]:
    """Append a copy of the first row at ``path``; a renamed copy renames the first row too."""

    def mutate(document: dict) -> None:
        rows = document
        for key in path:
            rows = rows[key]
        if renamed_to is not None:
            rows[0][0] = renamed_to
        rows.append(json.loads(json.dumps(rows[0])))

    return mutate


def _set_spec(predicate: str, field: str, value) -> Callable[[dict], None]:
    """Set ``field`` on the first atom specification with ``predicate``."""

    def mutate(document: dict) -> None:
        spec = next(s for s in _artifact(document)["atom_specs"] if s["predicate"] == predicate)
        spec[field] = value

    return mutate


def _not_chain(depth: int) -> dict:
    formula: dict = {"kind": "true"}
    for _ in range(depth):
        formula = {"kind": "not", "child": formula}
    return formula


def _guard_chain_deeper_than_the_interpreter_stack(document: dict) -> None:
    dfa = _artifact(document)["dfa"]
    atom = dfa["atoms"][0]
    length = 5000
    dfa["guard_nodes"] = [[atom, index + 1, -1] for index in range(length - 1)] + [[atom, -1, -1]]
    dfa["normal_roots"][0] = 0
    dfa["final_roots"][0] = 0


def _swap_element_candidate_and_reference(document: dict) -> None:
    plan = _artifact(document)["capture_plan"]
    plan["at_atom_id"], plan["ref_atom_ids"] = plan["ref_atom_ids"][0], [plan["at_atom_id"]]


def _name_an_unplanned_bind_atom(document: dict) -> None:
    _artifact(document)["capture_plan"]["bind_atom_ids"].append("ghost")


_USER_SAID = 'rule "Said"\ncondition:\n    user_said("x")'
_TOOL_PAIR = 'rule "Tool"\ncondition:\n    any reader in tools where reader.name == "read"\n'
_DIGEST = "0" * 64


@pytest.mark.parametrize(
    ("source", "mutate", "message"),
    [
        # document envelope
        pytest.param(
            _USER_SAID,
            _drop("models"),
            f"document fields do not match format_version {FORMAT_VERSION}: missing models",
            id="missing-top-level-field",
        ),
        pytest.param(
            _USER_SAID,
            _put("source", value="rule text"),
            f"document fields do not match format_version {FORMAT_VERSION}: unknown source",
            id="unknown-top-level-field",
        ),
        pytest.param(
            _USER_SAID,
            _put("kind", value="other"),
            "kind must be 'quarry.compiled-policy'",
            id="other-kind",
        ),
        pytest.param(
            _USER_SAID,
            _put("format_version", value=True),
            "format_version must be an integer",
            id="boolean-format-version",
        ),
        pytest.param(
            _USER_SAID,
            _put("config_version", value=3),
            "config_version must be 1 or 2",
            id="unknown-config-version",
        ),
        pytest.param(
            _USER_SAID,
            _put("rule", value=[]),
            "rule must be an object with string keys",
            id="rule-is-not-an-object",
        ),
        pytest.param(
            _USER_SAID,
            _put("rule", "metadata", "severity", value=3),
            "rule.metadata values must be strings",
            id="non-string-metadata",
        ),
        pytest.param(
            _USER_SAID,
            _put("source_sha256", value="A" * 64),
            "source_sha256 must be a lowercase SHA-256 digest",
            id="uppercase-source-digest",
        ),
        pytest.param(
            _USER_SAID,
            _put("text_resources", value={"policy": _DIGEST}),
            "text resource fingerprints do not match the compiled artifact",
            id="text-fingerprint-the-artifact-does-not-read",
        ),
        pytest.param(
            _USER_SAID,
            _put("models", value={"judge": _DIGEST}),
            "model fingerprints do not match the compiled artifact",
            id="model-fingerprint-the-artifact-does-not-call",
        ),
        pytest.param(
            _USER_SAID,
            _put("rule", "artifact", "rule_name", value=7),
            "rule.artifact.rule_name must be a string",
            id="non-string-rule-name",
        ),
        # atom specifications
        pytest.param(
            _USER_SAID,
            _put("rule", "artifact", "atom_specs", 0, "domain", value="weather"),
            f"rule.artifact.atom_specs[0] is incompatible with format_version {FORMAT_VERSION}",
            id="atom-of-unknown-domain",
        ),
        pytest.param(
            _USER_SAID,
            _duplicate_first("rule", "artifact", "atom_specs"),
            "rule.artifact.atom_specs contains duplicate atom ids",
            id="atom-specified-twice",
        ),
        pytest.param(
            _USER_SAID,
            _append(*_DFA, "atoms", value="ghost"),
            "DFA atoms lack specifications: ghost",
            id="automaton-reads-an-unspecified-atom",
        ),
        pytest.param(
            _USER_SAID,
            _duplicate_first(*_DFA, "atoms"),
            "rule.artifact.dfa.atoms contains duplicate atom ids",
            id="automaton-declares-an-atom-twice",
        ),
        pytest.param(
            _USER_SAID,
            _put(*_DFA, "guard_nodes", 0, 0, value="ghost"),
            "rule.artifact.dfa.guard_nodes[0].atom must be declared in rule.artifact.dfa.atoms",
            id="guard-reads-an-undeclared-atom",
        ),
        # rule automaton
        pytest.param(
            _USER_SAID,
            _put(*_DFA, "atoms", value="x"),
            "rule.artifact.dfa.atoms must be an array",
            id="atoms-not-an-array",
        ),
        pytest.param(
            _USER_SAID,
            _put(*_DFA, "states", value=[]),
            "rule.artifact.dfa.states must not be empty",
            id="no-states",
        ),
        pytest.param(
            _USER_SAID,
            _put(*_DFA, "states", 0, value=[0, "k", False]),
            "rule.artifact.dfa.states[0] must have four values",
            id="short-state-row",
        ),
        pytest.param(
            _USER_SAID,
            _put(*_DFA, "states", 0, 0, value=5),
            "DFA state ids must be contiguous and ordered from zero",
            id="out-of-order-state-id",
        ),
        pytest.param(
            _USER_SAID,
            _put(*_DFA, "states", 0, 2, value="yes"),
            "rule.artifact.dfa.states[0].is_accepting must be a Boolean",
            id="non-boolean-acceptance",
        ),
        pytest.param(
            _USER_SAID,
            _put(*_DFA, "initial_state", value=3),
            "rule.artifact.dfa.initial_state references a missing state",
            id="initial-state-missing",
        ),
        pytest.param(
            _USER_SAID,
            _drop(*_DFA, "normal_roots", 2),
            "DFA guard-root counts must match the state count",
            id="guard-root-missing",
        ),
        pytest.param(
            _USER_SAID,
            _put(*_DFA, "normal_roots", 1, value=-4),
            "rule.artifact.dfa.normal_roots[1] targets missing state 3",
            id="guard-targets-a-missing-state",
        ),
        pytest.param(
            _USER_SAID,
            _put(*_DFA, "normal_roots", 1, value=2),
            "rule.artifact.dfa.normal_roots[1] references missing guard node 2",
            id="guard-references-a-missing-node",
        ),
        pytest.param(
            _USER_SAID,
            _put(*_DFA, "guard_nodes", 0, 1, value=0),
            "DFA guard graph contains a cycle",
            id="guard-cycle",
        ),
        pytest.param(
            _USER_SAID,
            _put(*_DFA, "guard_nodes", 0, value=["atom", -1]),
            "rule.artifact.dfa.guard_nodes[0] must have three values",
            id="short-guard-row",
        ),
        pytest.param(
            _USER_SAID,
            _append(*_DFA, "guard_nodes", value=["atom", -1, -1]),
            "DFA guard table contains unreachable nodes",
            id="unreachable-guard-node",
        ),
        pytest.param(
            _USER_SAID,
            _guard_chain_deeper_than_the_interpreter_stack,
            "DFA guard graph exceeds the supported nesting depth",
            id="guard-chain-deeper-than-the-stack",
        ),
        # tool quantifiers
        pytest.param(
            _TOOL_PAIR,
            _put(*_TOOLS, 0, value=["q", "any", {"kind": "true"}]),
            "rule.artifact.tool_quantifiers[0] must contain an atom id, kind, Boolean body, "
            "and source nodes",
            id="short-tool-quantifier-row",
        ),
        pytest.param(
            _TOOL_PAIR,
            _duplicate_first(*_TOOLS, renamed_to="q"),
            "rule.artifact.tool_quantifiers[1] duplicates atom id 'q'",
            id="tool-quantifier-duplicated",
        ),
        pytest.param(
            _TOOL_PAIR,
            _put(*_TOOLS, 0, 1, value="some"),
            "rule.artifact.tool_quantifiers[0].kind must be 'any' or 'all'",
            id="tool-quantifier-of-unknown-kind",
        ),
        pytest.param(
            _TOOL_PAIR,
            _put(*_TOOLS, 0, 2, value={"kind": "xor"}),
            "rule.artifact.tool_quantifiers[0].body.kind 'xor' is unsupported",
            id="tool-body-with-unknown-connective",
        ),
        pytest.param(
            _TOOL_PAIR,
            _put(*_TOOLS, 0, 2, value=_not_chain(1000)),
            "compiled artifact exceeds the supported nesting depth",
            id="tool-body-deeper-than-the-stack",
        ),
        pytest.param(
            _TOOL_PAIR,
            _put(*_TOOLS, 0, 0, value="ghost"),
            "tool quantifier ids must be DFA atoms",
            id="tool-quantifier-the-automaton-does-not-read",
        ),
        pytest.param(
            _TOOL_PAIR,
            _put(*_TOOLS, 0, 2, value={"kind": "atom", "atom_id": "ghost"}),
            "tool quantifier body atoms lack specifications",
            id="tool-body-reads-an-unspecified-atom",
        ),
        pytest.param(
            _TOOL_PAIR,
            _set_spec("tool_quantifier", "predicate", "equals"),
            "tool quantifier ids must have tool_quantifier specifications",
            id="tool-quantifier-with-another-predicate",
        ),
        pytest.param(
            _TOOL_PAIR,
            _set_spec("equals", "domain", "event"),
            "tool quantifier body atoms must be static",
            id="tool-body-reads-an-event-atom",
        ),
        # closed parts
        pytest.param(
            _CLOSED,
            _drop_a_closed_automaton,
            "closed atoms must have a closed condition",
            id="closed-atom-without-its-automaton",
        ),
        pytest.param(
            _CLOSED,
            _closed_part_reads_itself,
            "closed conditions must not read themselves",
            id="closed-part-reads-itself",
        ),
        pytest.param(
            _CLOSED,
            _put("rule", "artifact", "closed_conditions", 0, 0, value="ghost"),
            "closed condition ids must be read by an automaton",
            id="closed-automaton-for-an-unread-atom",
        ),
        pytest.param(
            _CLOSED,
            _set_spec("closed", "predicate", "contains_any"),
            "closed condition ids must have closed specifications",
            id="closed-automaton-for-an-open-atom",
        ),
        pytest.param(
            _CLOSED,
            _put("rule", "artifact", "closed_conditions", 0, value=["ghost"]),
            "rule.artifact.closed_conditions[0] must contain an atom id and an automaton",
            id="closed-row-without-automaton",
        ),
        pytest.param(
            _CLOSED,
            _duplicate_first("rule", "artifact", "closed_conditions", renamed_to="c"),
            "rule.artifact.closed_conditions[3] duplicates atom id 'c'",
            id="closed-row-duplicated",
        ),
        # value capture plans
        pytest.param(
            _CAPTURE,
            _put(*_PLAN, value=None),
            "rule.artifact has parameterized atoms but no capture_plan",
            id="parameterized-atoms-without-a-plan",
        ),
        pytest.param(
            _CAPTURE,
            _put(*_PLAN, "kind", value="other"),
            "rule.artifact.capture_plan.kind must be 'value' or 'element'",
            id="plan-of-unknown-kind",
        ),
        pytest.param(
            _CAPTURE,
            _put(*_PLAN, "ref_atom_ids", value=[]),
            "rule.artifact.capture_plan must name its bind and reference atoms",
            id="value-plan-refs",
        ),
        pytest.param(
            _CAPTURE,
            _name_an_unplanned_bind_atom,
            "rule.artifact.capture_plan does not name exactly the parameterized atoms",
            id="value-plan-names-an-extra-atom",
        ),
        pytest.param(
            _CAPTURE,
            _put(*_PLAN, "var", value="other"),
            "rule.artifact.capture_plan atoms must be parameterized by 'other'",
            id="value-plan-for-another-variable",
        ),
        pytest.param(
            _CAPTURE,
            _add_value_plan_ordering_fields,
            f"rule.artifact.capture_plan fields do not match format_version {FORMAT_VERSION}: "
            "unknown ordered, within",
            id="value-plan-with-retired-ordering-fields",
        ),
        # element capture plans
        pytest.param(
            _FILTERED,
            _put(*_PLAN, "surface", value="not_a_surface"),
            "rule.artifact.capture_plan.surface must be an event surface",
            id="element-plan-surface",
        ),
        pytest.param(
            _FILTERED,
            _swap_element_candidate_and_reference,
            "rule.artifact.capture_plan.at_atom_id must name the candidate atom of 'c'",
            id="element-candidate-is-a-reference-atom",
        ),
        pytest.param(
            _FILTERED,
            _put(*_PLAN, "ref_atom_ids", value=[]),
            "rule.artifact.capture_plan does not name exactly the parameterized atoms",
            id="element-plan-without-references",
        ),
        pytest.param(
            _FILTERED,
            _set_spec("contains_any", "predicate", "equals"),
            "rule.artifact.capture_plan.ref_atom_ids must name joins to fields of 'c'",
            id="element-reference-is-not-a-join",
        ),
        pytest.param(
            _FILTERED,
            _filter_is_the_candidate_atom,
            "rule.artifact.capture_plan.filter_atom_id must name the test name = 'x' on "
            "'tool_calls' events",
            id="element-filter-is-the-candidate-atom",
        ),
        pytest.param(
            _FILTERED,
            _filter_is_another_name_test,
            "rule.artifact.capture_plan.filter_atom_id must name the test name = 'x' on "
            "'tool_calls' events",
            id="element-filter-is-another-name-test",
        ),
        pytest.param(
            _TWO_FLOWS,
            _own_step_reads_the_candidate_atom,
            "rule.artifact.capture_plan.own_step must read unparameterized 'user_messages' "
            "event atoms below model cost",
            id="own-step-reads-the-candidate-atom",
        ),
        pytest.param(
            _TWO_FLOWS,
            _own_step_reads_a_tool_call_atom,
            "rule.artifact.capture_plan.own_step must read unparameterized 'user_messages' "
            "event atoms below model cost",
            id="own-step-reads-another-surface",
        ),
    ],
)
def test_load_rejects_a_tampered_or_inconsistent_document(source, mutate, message) -> None:
    document = orjson.loads(_compiled(source).to_bytes())
    mutate(document)

    # The standard encoder supports the deep case that orjson refuses to write,
    # but on Python 3.11 it needs a higher limit while encoding.
    recursion_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(max(recursion_limit, 10_000))
        encoded = json.dumps(document).encode()
    finally:
        sys.setrecursionlimit(recursion_limit)

    with pytest.raises(ValueError) as exc_info:
        load_compiled_policy(encoded)

    assert str(exc_info.value) == f"invalid compiled policy: {message}"


@pytest.mark.parametrize(
    ("data", "error", "message"),
    [
        pytest.param(
            "{}",
            TypeError,
            "compiled policy data must be bytes, bytearray, or memoryview",
            id="text-instead-of-bytes",
        ),
        pytest.param(
            b"{",
            ValueError,
            "invalid compiled policy: malformed JSON: unexpected end of data: line 1 column 2 "
            "(char 1)",
            id="truncated-json",
        ),
        pytest.param(
            b"[]",
            ValueError,
            "invalid compiled policy: document must be an object with string keys",
            id="json-array",
        ),
    ],
)
def test_load_rejects_input_that_is_not_a_compiled_policy_document(data, error, message) -> None:
    with pytest.raises(error) as exc_info:
        load_compiled_policy(data)

    assert str(exc_info.value) == message


_PARAMETERIZED = """rule "Parameterized"
parameters:
    threshold: number
condition:
    model("judge").score(messages, "instructions") >= params.threshold
"""
_COLLECTED = """rule "Collected"
condition:
    model("judge").classify(messages) >= 0.60
collect:
    [span for span in model("judge").spans(messages, 0.70) if span.score.value >= 0.90]
"""
_MODEL_SCORE_LITERAL = """rule "Model score"
condition:
    model("judge").score(messages, "policy") > 0.5
"""

_LEGACY_SPLIT = """rule "Legacy split"
condition:
    any chunk in llm_split_conversation("judge", messages, 200) where
        llm_score("judge", "policy", chunk) > 0.5
"""
_MODEL_CALL = ("rule", "artifact", "atom_specs", 0, "model_expression", "children", 0)


@pytest.fixture(scope="module")
def judge_runtimes(tmp_path_factory) -> dict:
    """Runtimes a model-backed policy is compiled and loaded against, by name."""

    directory = tmp_path_factory.mktemp("judge")
    return {
        "judge": _runtime(directory, policy_text="be safe"),
        "judge-v1": _runtime(directory, config_version=1, policy_text="be safe"),
        "no-models": (None, None),
        "config-version-1": (QuarryConfig(config_version=1), None),
    }


@pytest.mark.parametrize(
    ("source", "runtime", "mutate", "message"),
    [
        pytest.param(
            _PARAMETERIZED,
            "judge",
            _put("rule", "artifact", "parameter_schema", 0, "type", value="date"),
            "invalid compiled policy: rule.artifact.parameter_schema[0] has an unsupported type "
            "or binding",
            id="parameter-of-unknown-type",
        ),
        pytest.param(
            _PARAMETERIZED,
            "judge",
            _duplicate_first("rule", "artifact", "parameter_schema"),
            "invalid compiled policy: parameter schema contains duplicate names",
            id="parameter-declared-twice",
        ),
        pytest.param(
            _COLLECTED,
            "judge",
            _put("rule", "artifact", "collect_plan", "source", value=None),
            "invalid compiled policy: rule.artifact.collect_plan.source must be a model expression",
            id="collection-without-a-source",
        ),
        pytest.param(
            _MODEL_SCORE_LITERAL,
            "judge",
            _put(*_MODEL_CALL, "value", value="judge"),
            "compiled artifact contains an invalid model call",
            id="model-call-without-an-operation",
        ),
        pytest.param(
            _MODEL_SCORE_LITERAL,
            "judge",
            _put(*_MODEL_CALL, "value", value="judge\x1fsing"),
            "unsupported model operation 'sing'",
            id="model-call-with-an-unknown-operation",
        ),
        pytest.param(
            _LEGACY_SPLIT,
            "judge-v1",
            _put("rule", "artifact", "atom_specs", 0, "chunk_scalar", value="llm_complete"),
            "compiled artifact contains unsupported chunk scalar 'llm_complete'",
            id="split-reduced-by-an-unsupported-scalar",
        ),
    ],
)
def test_load_rejects_an_altered_model_backed_document(
    judge_runtimes, source, runtime, mutate, message
) -> None:
    config, backend = judge_runtimes[runtime]
    compiled = compile_policy(source, runtime_config=config, model_inference_backend=backend)
    document = orjson.loads(compiled.to_bytes())
    mutate(document)

    with pytest.raises(ValueError) as exc_info:
        load_compiled_policy(
            orjson.dumps(document), runtime_config=config, model_inference_backend=backend
        )

    assert str(exc_info.value) == message


@pytest.mark.parametrize(
    ("source", "load_runtime", "message"),
    [
        pytest.param(
            _COLLECTED,
            "no-models",
            "runtime configuration is missing models: judge",
            id="model-absent",
        ),
        pytest.param(
            _MODEL_SCORE_LITERAL,
            "config-version-1",
            "runtime config_version 1 does not match 2",
            id="other-config-version",
        ),
    ],
)
def test_load_rejects_a_runtime_that_lacks_the_compiled_resources(
    judge_runtimes, source, load_runtime, message
) -> None:
    compile_config, compile_backend = judge_runtimes["judge"]
    policy = compile_policy(
        source, runtime_config=compile_config, model_inference_backend=compile_backend
    )
    config, backend = judge_runtimes[load_runtime]

    with pytest.raises(ValueError) as exc_info:
        load_compiled_policy(
            policy.to_bytes(), runtime_config=config, model_inference_backend=backend
        )

    assert str(exc_info.value) == f"invalid compiled policy: {message}"


_TREE = """rule "Story then send"
condition:
    (any m in user_messages where m.content contains "story")
    and not (any call in tool_calls where call.name == "send_email")
"""
_STORY = {"messages": [{"role": "user", "content": "tell me a story"}]}


async def _decision_checks(policy) -> tuple[list[tuple[str, bool]], list[str]]:
    (result,) = await policy.scan_many_async([_STORY], decision_paths=True)
    (recorder,) = result.decision_paths.values()
    return [(check.atom_id, check.value) for check in recorder.checks], result.rule_matches


@pytest.mark.asyncio
async def test_a_loaded_policy_walks_the_same_decision_tree_without_the_rule_text() -> None:
    compiled = compile_policy(_TREE)
    loaded = load_compiled_policy(compiled.to_bytes())

    checks, matches = await _decision_checks(loaded)

    assert (checks, matches) == await _decision_checks(compiled)
    assert [value for _, value in checks] == [True, False]
    assert matches == ["Story then send"]
    (without_paths,) = await loaded.scan_many_async([_STORY], explore_rate=0.5)
    assert without_paths.rule_matches == ["Story then send"]
    assert not without_paths.decision_paths


def _second_node_unnamed(tree: dict) -> dict:
    return {**tree, "nodes": [tree["nodes"][0], [None, *tree["nodes"][1][1:]]]}


def _first_node_names_an_unknown_atom(tree: dict) -> dict:
    return {**tree, "nodes": [["atom_not_in_the_dfa", *tree["nodes"][0][1:]], *tree["nodes"][1:]]}


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        pytest.param(lambda tree: None, "decision_tree is required", id="missing-tree"),
        pytest.param(_second_node_unnamed, "must each name an atom", id="nameless-node"),
        pytest.param(
            _first_node_names_an_unknown_atom, "root automaton's atoms", id="node-outside-the-dfa"
        ),
    ],
)
def test_load_rejects_a_tampered_decision_tree(edit, message) -> None:
    document = orjson.loads(compile_policy(_TREE).to_bytes())
    artifact = document["rule"]["artifact"]
    artifact["decision_tree"] = edit(artifact["decision_tree"])

    with pytest.raises(ValueError, match=message):
        load_compiled_policy(orjson.dumps(document))
