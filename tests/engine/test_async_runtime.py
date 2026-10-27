# Copyright 2026 Cisco Systems, Inc. and its affilitates
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

"""How a scan spends external work and time: which model calls it makes, when it stops, how
it shares the event loop, and the engine's import boundary.

Model calls are observed at configured fake adapters, the external boundary of a scan.
"""

from __future__ import annotations

import ast
import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from quarry import SafetyLimits
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.safety.regex_backend import RE2_REGEX
from quarry.scanner import DSLScanner
from quarry.types.results import Score, Span


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _call(name: str, call_id: str, arguments: str = "{}") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


def _rule(condition: str, *strings: str, name: str = "R") -> str:
    block = "strings:\n" + "".join(f"    {s}\n" for s in strings) if strings else ""
    return f'rule "{name}"\n{block}condition:\n    {condition}\n'


class _RecordingClassifier:
    """A configured provider fake recording each ``(model, text)``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def classify_messages(self, config, messages, _tools, *, timeout=None):
        self.calls.append((config.model, messages[0]["content"]))
        return SimpleNamespace(chunks=[SimpleNamespace(probabilities=[0.1, 0.9])])


async def _scan_with_classifier(rules: tuple[str, ...], messages, tools=()):
    classifier = _RecordingClassifier()
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                alias: {
                    "adapter": "fake",
                    "model": f"configured/{alias}",
                    "classify": {"class_index": 1},
                }
                for alias in ("tox", "pi", "m")
            },
        },
        Path(__file__).resolve().parents[2],
        require_rules=False,
    )
    config = replace(config, quarry_rule_texts=rules)
    backend = ModelInferenceBackend(config, adapters={"fake": classifier})
    scanner = DSLScanner(config, model_inference_backend=backend)
    result = await scanner.scan_data_async({"messages": messages, "tools": list(tools)})
    assert result.is_complete, result.completion_issues
    return result, [text for _, text in classifier.calls]


_GATED = 'any m in user_messages where m.content contains "TRIGGER" and model_scan("tox", m.content, 0.5, 1)'
_CLOSED_MODEL = (
    'any m in user_messages where m.content contains "TRIGGER" and '
    '(any n in assistant_messages where model_scan("tox", n.content, 0.5, 1))'
)
_TOOLS_MODEL = 'any t in tools where model_scan("pi", t.description, 0.5, 1) and t.name == "never"'


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": "d"}}


@pytest.mark.parametrize(
    ("condition", "messages", "tools", "matched", "classified"),
    [
        pytest.param(
            _GATED,
            [_user("nothing interesting here")],
            (),
            False,
            [],
            id="cheap-test-false-skips-the-model",
        ),
        pytest.param(
            _GATED,
            [_user("please TRIGGER the thing")],
            (),
            True,
            ["please TRIGGER the thing"],
            id="cheap-test-true-asks-the-model",
        ),
        pytest.param(
            _CLOSED_MODEL,
            [_user("nothing here"), _assistant("reply")],
            (),
            False,
            [],
            id="closed-part-waits-for-the-cheap-test",
        ),
        pytest.param(
            _CLOSED_MODEL,
            [_user("please TRIGGER it"), _assistant("reply")],
            (),
            True,
            ["reply"],
            id="closed-part-read-after-the-cheap-test",
        ),
        pytest.param(
            _TOOLS_MODEL,
            [_user("hi")],
            (_tool("never"),),
            True,
            ["d"],
            id="tool-name-test-passes-then-the-model",
        ),
        pytest.param(
            _TOOLS_MODEL,
            [_user("hi")],
            (_tool("other"),),
            False,
            [],
            id="tool-name-test-fails-and-skips-the-model",
        ),
    ],
)
async def test_a_model_is_asked_only_once_cheaper_tests_leave_the_verdict_open(
    condition, messages, tools, matched, classified
):
    result, texts = await _scan_with_classifier((_rule(condition),), messages, tools)
    assert bool(result.rule_matches) is matched
    assert texts == classified


async def test_identical_model_calls_across_rules_share_one_inference():
    rules = (
        _rule('any msg in user_messages where model_score("m", msg.content, 0) >= 0.05', name="a"),
        _rule('any msg in user_messages where model_score("m", msg.content, 0) >= 0.02', name="b"),
    )
    result, texts = await _scan_with_classifier(rules, [_user("hi")])
    assert set(result.rule_matches) == {"a", "b"}
    assert texts == ["hi"]


async def test_a_model_rule_without_a_backend_is_rejected_at_load():
    # An unanswerable model check cannot become a quiet no-match.
    with pytest.raises(RuntimeError, match="model|resource"):
        DSLScanner(QuarryConfig(quarry_rule_texts=(_rule(_GATED),)))


_PER_CANDIDATE = (
    "(any m in user_messages where ((any c in tool_calls where c.arguments.cmd contains "
    'm.content) after called_tool("y")) and not (any d in tool_calls where d.name == "q" and '
    "d.arguments.cmd contains m.content)) or (all a in assistant_messages where a.content "
    'contains "z")'
)


def test_a_candidate_cap_that_cuts_off_a_match_makes_the_scan_incomplete():
    # The match needs the 257th candidate value; the scan reports the cut instead of a miss.
    messages = [
        *(_user(f"m{i}") for i in range(256)),
        _assistant("a"),
        _call("y", "c0"),
        _user("last"),
        _call("x", "c1", '{"cmd": "run last"}'),
    ]
    # A generous rule timeout, so only the candidate cap can cut the scan short.
    config = QuarryConfig(
        quarry_rule_texts=(_rule(_PER_CANDIDATE),), safety_limits=SafetyLimits(rule_timeout=60.0)
    )
    result = DSLScanner(config).scan_data({"messages": messages})
    assert result.rule_matches == []
    assert result.completion_state == "incomplete"
    assert "evaluation was limited" in result.completion_issues


def test_a_count_over_a_long_field_stops_once_its_comparison_is_decided(monkeypatch):
    # Verify the bounded count operation; runner speed cannot establish short-circuiting.
    limits = []
    original = RE2_REGEX.count_up_to

    def record_limit(pattern, text, limit, flags=0):
        limits.append(limit)
        return original(pattern, text, limit, flags)

    monkeypatch.setattr(RE2_REGEX, "count_up_to", record_limit)
    scanner = DSLScanner(
        QuarryConfig(
            quarry_rule_texts=(
                _rule("any m in user_messages where #a in m.content >= 3", '$a = "a" regex'),
            )
        )
    )
    result = scanner.scan_data({"messages": [_user("a" * 1_000_000)]})
    assert result.rule_matches == ["R"]
    assert len(limits) == 1 and 0 < limits[0] <= 4


def test_a_capture_over_a_long_field_of_one_repeated_value_matches():
    rule = _rule(
        "(any m in user_messages where ($d in m.content as v)) and "
        "(any t in tool_responses where t.content contains v)",
        '$d = "x" regex',
    )
    messages = [
        _user("hello"),
        _user("x" * 100_000),
        _call("t", "c1"),
        {"role": "tool", "tool_call_id": "c1", "content": "x"},
    ]
    result = DSLScanner(QuarryConfig(quarry_rule_texts=(rule,))).scan_data(
        {"messages": messages}, enable_explain=True
    )
    assert result.rule_matches == ["R"]
    assert result.detailed_matches[0].captured_value == "x"


async def _loop_turns_during(rule: str, messages: list[dict]) -> int:
    """How many times a peer task runs while one scan evaluates ``rule`` over ``messages``."""
    scanner = DSLScanner(QuarryConfig(quarry_rule_texts=(rule,)))
    done = False
    turns = 0

    async def peer():
        nonlocal turns
        while not done:
            await asyncio.sleep(0)
            turns += 1

    async def scan():
        nonlocal done
        try:
            await scanner.scan_data_async({"messages": messages})
        finally:
            done = True

    await asyncio.gather(scan(), peer())
    return turns


_CAPTURE_A = (
    "(any m in user_messages where $a in m.content as v) and "
    "(any r in assistant_messages where r.content contains v)"
)


@pytest.mark.parametrize(
    ("condition", "messages"),
    [
        pytest.param(_CAPTURE_A, [_user("a" * 1_000_000)], id="one-value-repeated-a-million-times"),
        pytest.param(
            _CAPTURE_A,
            [_user("a"), *[_assistant("z" * 100)] * 9_000],
            id="one-value-against-many-reference-sites",
        ),
        pytest.param(
            'any m in user_messages where m.content contains "q"',
            [_user("b")] * 10_000,
            id="a-long-conversation",
        ),
        pytest.param(
            'any m in user_messages where m.content ~ "q+z"',
            [_user("b" * 60_000)] * 400,
            id="many-large-fields",
        ),
    ],
)
async def test_a_long_evaluation_shares_the_event_loop(condition, messages):
    # The same scan over one short message is the baseline: whatever the scanner itself awaits.
    rule = _rule(condition, '$a = "a" regex')
    baseline = await _loop_turns_during(rule, [_user("b")])
    assert await _loop_turns_during(rule, messages) > baseline + 10


class _SpanAdapter:
    """A version 2 model adapter: classifies every message as PII and decodes one span at
    the address ``alice@example.com`` when the text holds it."""

    def __init__(self) -> None:
        self.classifications = 0

    async def classify_messages(self, config, messages, tools, *, timeout=None):
        self.classifications += 1
        return SimpleNamespace(chunks=[SimpleNamespace(probabilities=[0.9, 0.9])])

    async def decode_spans(self, config, raw_result, messages, threshold):
        content = messages[0].content
        if "alice@example.com" not in content:
            return []
        start = content.index("alice@example.com")
        return [Span(0, start, start + len("alice@example.com"), Score(0.91))]


def _collecting_scanner(tmp_path: Path, condition: str, adapter: _SpanAdapter) -> DSLScanner:
    rule = tmp_path / "r.rule"
    rule.write_text(
        f'rule "R"\ncondition: {condition}\ncollect:\n'
        '    [span for span in model("privacy").spans(messages, 0.70) if true]\n'
    )
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "rules": [str(rule)],
            "models": {
                "privacy": {
                    "adapter": "fake",
                    "model": "physical-model",
                    "category": "PII",
                    "spans": {"class_index": 1},
                }
            },
        },
        tmp_path,
    )
    backend = ModelInferenceBackend(config, adapters={"fake": adapter})
    return DSLScanner(config, model_inference_backend=backend)


@pytest.mark.parametrize(
    ("condition", "text", "matched", "spans", "classifications"),
    [
        pytest.param(
            "true",
            "mail alice@example.com",
            True,
            [(0, 5, 22)],
            1,
            id="true-gate-returns-the-collection",
        ),
        pytest.param(
            "true", "nothing to collect", False, None, 1, id="empty-collection-is-no-match"
        ),
        pytest.param(
            "false", "mail alice@example.com", False, None, 0, id="false-gate-runs-no-collector"
        ),
    ],
)
async def test_a_collect_rule_matches_only_with_a_true_gate_and_items(
    tmp_path, condition, text, matched, spans, classifications
):
    adapter = _SpanAdapter()
    scanner = _collecting_scanner(tmp_path, condition, adapter)
    result = await scanner.scan_data_async({"messages": [_user(text)]})

    assert result.is_complete, result.completion_issues
    assert (result.rule_matches == ["R"]) is matched
    output = result.outputs.get("R")
    found = (
        [(i.value.message_index, i.value.start, i.value.end) for i in output.items]
        if output
        else None
    )
    assert found == spans
    assert adapter.classifications == classifications


_ENGINE = Path(__file__).resolve().parents[2] / "quarry" / "engine"
# The automaton core imports nothing from quarry outside the engine.
_PURE = {
    "temporal",
    "symbolic_dfa",
    "builders",
    "contracts",
    "artifacts",
    "atoms",
    "diagnostics",
    "matchers",
    "core",
    "pacing",
}
# Only the AST-to-IR bridge reads the rule AST.
_AST_READERS = {"compiler", "desugar", "lower"}
_DATA_TYPES = (
    "quarry.types.messages",
    "quarry.types.context",
    "quarry.core.conversation",
    "quarry.safety.limits",
    "quarry.types.exceptions",
    "quarry.language.runtime",
    # Pure schema-to-type rules with no quarry imports; the compiler types `.json` reads by it.
    "quarry.types.response_schema",
)
_AST_TYPES = ("quarry.types.dsl", "quarry.language.metadata", "quarry.rule_versioning")


def _quarry_imports(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return {module for module in modules if module.split(".")[0] == "quarry"}


def _within(module: str, packages: tuple[str, ...]) -> bool:
    return any(module == package or module.startswith(package + ".") for package in packages)


def test_the_engine_depends_only_on_data_types_and_never_on_the_parser_or_scanner():
    offenders = {}
    for path in sorted(_ENGINE.glob("*.py")):
        allowed = ("quarry.engine",)
        if path.stem not in _PURE:
            allowed += _DATA_TYPES
        if path.stem in _AST_READERS:
            allowed += _AST_TYPES
        outside = {module for module in _quarry_imports(path) if not _within(module, allowed)}
        if outside:
            offenders[path.name] = sorted(outside)
    assert offenders == {}
