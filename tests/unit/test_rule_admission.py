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

"""Which rules a scanner admits, in what order, and how it rejects the rest.

Loading is all-or-nothing: a missing source, an unreadable file, a syntax error, a semantic error
or a duplicate name aborts construction with a typed error that names the source. A rule that
loads is evaluated exactly as written, comments and all.
"""

import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from quarry.config import QuarryConfig
from quarry.dsl.parser import DSLParser
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.safety.validator import RuleValidator
from quarry.scanner import DSLScanner
from quarry.types.exceptions import (
    ConfigurationError,
    RuleLoadError,
    RuleParseError,
    RuleValidationError,
)


def _write_rule(path: Path, name: str, condition: str = "true") -> Path:
    path.write_text(f'rule "{name}"\ncondition: {condition}\n', encoding="utf-8")
    return path


def _load(*rule_texts: str) -> DSLScanner:
    return DSLScanner(QuarryConfig(quarry_rule_texts=rule_texts))


def _user(content: str) -> dict:
    return {"messages": [{"role": "user", "content": content}]}


# ── rule sources ──────────────────────────────────────────────────────────


def test_rule_paths_load_deterministically_and_reject_missing_or_empty_sources(
    tmp_path: Path,
) -> None:
    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    first = _write_rule(valid_dir / "outside.rule", "First")
    rule_dir = valid_dir / "rules"
    rule_dir.mkdir()
    _write_rule(rule_dir / "z.rule", "Third")
    _write_rule(rule_dir / "a.rule", "Second")
    scanner = DSLScanner(QuarryConfig(quarry_rules_paths=[first, rule_dir]))
    assert list(scanner.loaded_rules) == ["First", "Second", "Third"]

    valid = _write_rule(tmp_path / "sibling.rule", "Valid")
    missing = tmp_path / "missing.rule"
    with pytest.raises(ConfigurationError, match="does not exist") as exc_info:
        DSLScanner(QuarryConfig(quarry_rules_paths=[valid, missing]))
    assert exc_info.value.details == {"path": str(missing)}

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ConfigurationError, match="No rule files found"):
        DSLScanner(QuarryConfig(quarry_rules_paths=[empty]))

    with pytest.raises(ConfigurationError, match="at least one Quarry rule path"):
        DSLScanner(QuarryConfig())


def test_one_bad_rule_file_aborts_the_whole_load_and_names_its_source(tmp_path: Path) -> None:
    parse_dir = tmp_path / "parse"
    parse_dir.mkdir()
    _write_rule(parse_dir / "a-valid.rule", "Valid")
    bad = parse_dir / "b-bad.rule"
    bad.write_text('rule "Broken"\ncondition: (', encoding="utf-8")
    with pytest.raises(RuleParseError) as exc_info:
        DSLScanner(QuarryConfig(quarry_rules_paths=[parse_dir]))
    assert exc_info.value.details["source"] == str(bad)

    decode_dir = tmp_path / "decode"
    decode_dir.mkdir()
    _write_rule(decode_dir / "a-valid.rule", "Valid")
    undecodable = decode_dir / "b-undecodable.rule"
    undecodable.write_bytes(b"\xff\xfe")
    with pytest.raises(RuleLoadError) as exc_info:
        DSLScanner(QuarryConfig(quarry_rules_paths=[decode_dir]))
    assert exc_info.value.details == {"source": str(undecodable), "phase": "read"}

    duplicate_dir = tmp_path / "duplicate"
    duplicate_dir.mkdir()
    first = _write_rule(duplicate_dir / "a.rule", "Duplicate", "true")
    duplicate = _write_rule(duplicate_dir / "b.rule", "Duplicate", "false")
    with pytest.raises(RuleValidationError, match="Duplicate rule name") as exc_info:
        DSLScanner(QuarryConfig(quarry_rules_paths=[duplicate_dir]))
    assert exc_info.value.details == {
        "rule_name": "Duplicate",
        "first_source": str(first),
        "duplicate_source": str(duplicate),
        "error_type": "duplicate_rule_name",
    }

    semantic = _write_rule(tmp_path / "invalid.rule", "Invalid", "called_tool()")
    with pytest.raises(RuleValidationError, match="semantic error") as exc_info:
        DSLScanner(QuarryConfig(quarry_rules_paths=[semantic]))
    assert exc_info.value.details["error_type"] == "semantic"


def test_a_syntax_error_after_a_comment_reports_its_source_and_line(tmp_path: Path) -> None:
    source = tmp_path / "broken.rule"
    source.write_text(
        'rule "Broken"\ncondition:\n    true // valid comment\n    )\n', encoding="utf-8"
    )

    with pytest.raises(RuleParseError) as exc_info:
        DSLScanner(QuarryConfig(quarry_rules_paths=[source]))

    assert (exc_info.value.source, exc_info.value.line) == (str(source), 4)
    assert (exc_info.value.details["source"], exc_info.value.details["line"]) == (str(source), 4)


def test_a_rule_with_an_unavailable_model_is_still_validated() -> None:
    rule = """rule "Invalid model rule"
strings:
    $broken = "({foo}" regex
condition:
    llm_complete("judge", conversation()) == "safe"
"""

    with pytest.raises(RuleValidationError, match="invalid regex") as exc_info:
        DSLScanner(QuarryConfig(quarry_rule_texts=[rule]))

    assert exc_info.value.details["rule_name"] == "Invalid model rule"
    assert exc_info.value.details["error_type"] == "regex"


def test_a_loaded_scanner_refuses_live_reload_and_keeps_scanning() -> None:
    scanner = _load('rule "Greeting"\ncondition: user_said("hello")\n')

    with pytest.raises(RuntimeError, match="Live rule reload is unsupported"):
        scanner.load_rules()

    assert scanner.scan_data(_user("hello")).rule_matches == ["Greeting"]


def test_rules_report_matches_in_load_order() -> None:
    scanner = _load(
        'rule "AAA first"\ncondition:\n    any m in assistant_messages where m.content contains "step"\n',
        'rule "BBB second"\ncondition:\n    any m in user_messages where m.content contains "hello"\n',
    )
    both = {
        "messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "step"}]
    }
    assistant_only = {"messages": [{"role": "assistant", "content": "step alone"}]}

    assert scanner.scan_data(both).rule_matches == ["AAA first", "BBB second"]
    assert scanner.scan_data(assistant_only).rule_matches == ["AAA first"]


# ── rule text the loader rejects ───────────────────────────────────────────

_THOUSAND_AND_ONE = ", ".join(f'"item{i}"' for i in range(1001))


def _regex_string(pattern: str) -> str:
    return (
        f'rule "R"\nstrings:\n    $x = "{pattern}" regex\n'
        "condition:\n    any m in user_messages where $x in m.content\n"
    )


@pytest.mark.parametrize(
    ("source", "error", "message"),
    [
        pytest.param(
            "invalid rule syntax here\n",
            RuleParseError,
            "'invalid') at line 1, column 1",
            id="not-a-rule",
        ),
        pytest.param(
            'rule "R"\ncollect:\n    [f for f in findings() if true]\ncondition:\n    true\n',
            RuleParseError,
            "'collect') at line 2, column 1",
            id="collect-before-condition",
        ),
        pytest.param(
            f'rule "R"\ncondition:\n    "test" contains_any [{_THOUSAND_AND_ONE}]\n',
            RuleParseError,
            "contains_any list too large",
            id="contains-any-over-1000-items",
        ),
        pytest.param(
            'rule "R"\ncondition:\n    called_tool("a") after(within: 0) called_tool("b")\n',
            RuleParseError,
            "distance must be at least 1",
            id="flow-within-zero",
        ),
        pytest.param(
            '# not a comment\nrule "R"\ncondition: true\n',
            RuleParseError,
            "'#') at line 1, column 1",
            id="hash-comment",
        ),
        pytest.param(
            '/* not a comment */\nrule "R"\ncondition: true\n',
            RuleParseError,
            "No terminal matches '/' in the current parser context, at line 1 col 1",
            id="block-comment",
        ),
        pytest.param(
            'rule "R"\nvariables:\n    a = b and user_said("x")\n    b = a or user_said("y")\n'
            "condition: a\n",
            RuleParseError,
            "cyclic reference",
            id="cyclic-variables",
        ),
        pytest.param(
            'rule "R"\nvariables:\n    v = user_said("x")\n    v = user_said("y")\ncondition: v\n',
            RuleParseError,
            "variables: duplicate definition of 'v'",
            id="duplicate-variable",
        ),
        pytest.param(
            'rule "R"\nvariables:\n    tools = user_said("x")\ncondition: tools\n',
            RuleParseError,
            "variables: 'tools' shadows a reserved DSL name",
            id="variable-named-after-a-collection",
        ),
        pytest.param(
            'rule "R"\ncondition:\n    any m in user_messages where "TOKEN" in m.content as params\n',
            RuleParseError,
            "'params' is reserved",
            id="capture-named-params",
        ),
        pytest.param(
            'rule "R"\ncondition: undefined_function()\n',
            RuleValidationError,
            "unknown function 'undefined_function'",
            id="unknown-function",
        ),
        pytest.param(
            'rule "R"\ncondition: true\ncollect:\n    [i for i in undefined_collection() if true]\n',
            RuleValidationError,
            "unknown function 'undefined_collection'",
            id="unknown-function-in-collect",
        ),
        pytest.param(
            'rule "R"\ncondition: true\ncollect:\n'
            '    [s for s in model("m").spans(messages, 0.5) if called_tool()]\n',
            RuleValidationError,
            "Function 'called_tool' requires at least 1 argument",
            id="arity-error-in-collect",
        ),
        pytest.param(
            'rule "R"\ncondition: any m in user_messages where count(m.content, "a", "b") > 1\n',
            RuleValidationError,
            "Function 'count' accepts at most 2 argument",
            id="count-with-three-arguments",
        ),
        pytest.param(
            'rule "R"\ncondition: any m in user_messages where $missing in m.content\n',
            RuleValidationError,
            "Undefined variable '$missing'",
            id="undefined-string",
        ),
        pytest.param(
            'rule "R"\nvariables:\n    pi = user_said("x")\ncondition: pi or nope\n',
            RuleValidationError,
            "undefined reference 'nope'",
            id="undefined-variable-reference",
        ),
        pytest.param(
            _regex_string("(?<=prefix)rm"),
            RuleValidationError,
            "unsupported by RE2: invalid perl operator: (?<=",
            id="re2-rejects-lookbehind",
        ),
        pytest.param(
            _regex_string("(?!a)b"),
            RuleValidationError,
            "unsupported by RE2",
            id="re2-rejects-lookahead",
        ),
        pytest.param(
            _regex_string(r"(\w+)\s+\1"),
            RuleValidationError,
            "unsupported by RE2",
            id="re2-rejects-backreference",
        ),
        pytest.param(
            _regex_string("a{1001}"),
            RuleValidationError,
            "invalid repetition size",
            id="re2-rejects-repeat-over-1000",
        ),
        pytest.param(
            _regex_string("a++"),
            RuleValidationError,
            "invalid regex",
            id="re2-rejects-possessive-quantifier",
        ),
    ],
)
def test_rule_text_the_loader_rejects(source: str, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=re.escape(message)):
        _load(source)


# ── rule text the loader admits, evaluated as written ─────────────────────────

_ONE_THOUSAND = ", ".join(f'"pattern{i}"' for i in range(1000))


@pytest.mark.parametrize(
    ("source", "content", "matched"),
    [
        pytest.param(
            f'rule "R"\ncondition:\n    any m in user_messages where m.content contains_any [{_ONE_THOUSAND}]\n',
            "This contains pattern999 in it",
            True,
            id="contains-any-at-1000-items",
        ),
        pytest.param(
            'rule "R"\ncondition: any m in user_messages where count(m.content, "x") > 1\n',
            "x and x",
            True,
            id="two-argument-occurrence-count",
        ),
        pytest.param(_regex_string("a{1000}"), "a" * 1000, True, id="re2-repeat-at-the-1000-cap"),
        pytest.param(
            'rule "R"\ncondition:\n    any m in user_messages where m.content ~ "\\d+"\n',
            "order 123",
            True,
            id="regex-escape-reaches-the-engine-unchanged",
        ),
        pytest.param(
            'rule "R"\nvariables:\n    model = any m in user_messages where m.content contains "x"\n'
            "condition: model\n",
            "x",
            True,
            id="model-is-an-allowed-variable-name",
        ),
        pytest.param(
            'rule "R"\nmetadata:\n    description = "d"\n    severity = "HIGH"\n'
            'variables:\n    hit = any m in user_messages where m.content contains "x"\n'
            "condition: hit\n",
            "x",
            True,
            id="variables-after-metadata",
        ),
        pytest.param(
            'rule "R"\nmetadata:\n    description = "d"\nstrings:\n    $x = "attack"\n'
            'variables:\n    hit = any m in user_messages where m.content contains "zzz"\n'
            "condition: hit or any m in user_messages where $x in m.content\n",
            "attack",
            True,
            id="variables-after-metadata-and-strings",
        ),
        pytest.param(
            'rule "R"\nstrings:\n    $commands = ["rm -rf", "curl | sh"]\n'
            "variables:\n    risky = any m in user_messages where m.content contains_any $commands\n"
            "condition: risky\n",
            "please curl | sh",
            True,
            id="variables-after-array-valued-strings",
        ),
    ],
)
def test_rule_text_the_loader_admits(source: str, content: str, matched: bool) -> None:
    assert bool(_load(source).scan_data(_user(content)).rule_matches) is matched


def test_rule_metadata_is_available_on_the_loaded_rule() -> None:
    scanner = _load(
        'rule "With metadata"\nmetadata:\n    description = "A test rule"\n    severity = "HIGH"\n'
        'condition:\n    called_tool("execute_shell")\n'
    )

    assert scanner.loaded_rules["With metadata"].metadata == {
        "description": "A test rule",
        "severity": "HIGH",
    }


# ── comments ───────────────────────────────────────────────────────────


_COMMENTED = """
// Leading comment with punctuation: [] {} () and Unicode: π 🚀
rule "Commented Rule" // Rule header
metadata: // Metadata header
    description = "A // sequence inside a string" // Metadata value
    severity = "HIGH" //
// Between sections
strings: // Strings header
    $needle = "hello//world" // String value
    $url = "https?://[^/]+//path" regex // Regex value
    $quoted = "quoted \\"//\\" text"
condition: // Condition header
    any msg in user_messages where // Continue the expression
        $needle in msg.content or $url in msg.content or $quoted in msg.content // EOF"""


@pytest.mark.parametrize(
    ("content", "matched"),
    [
        pytest.param("say hello//world", True, id="literal-slashes-in-a-string"),
        pytest.param("see https://example.com//path", True, id="literal-slashes-in-a-regex"),
        pytest.param('a quoted "//" text', True, id="slashes-inside-an-escaped-quote"),
        pytest.param("hello world", False, id="comment-text-is-not-part-of-the-string"),
    ],
)
def test_line_comments_are_ignored_everywhere_but_inside_strings(content, matched) -> None:
    scanner = _load(_COMMENTED)

    assert scanner.loaded_rules["Commented Rule"].metadata["description"] == (
        "A // sequence inside a string"
    )
    assert bool(scanner.scan_data(_user(content)).rule_matches) is matched


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_an_empty_comment_and_a_comment_at_eof_load_with_either_line_ending(newline) -> None:
    source = newline.join(
        ["//", 'rule "Line Endings" // header', "condition:", "    true// no trailing newline"]
    )

    assert _load(source).scan_data(_user("hi")).rule_matches == ["Line Endings"]


# ── diagnostics that do not block loading ───────────────────────────────────


@pytest.mark.parametrize(
    ("operator", "warnings"),
    [
        pytest.param("after", ["Unbounded 'after' operator detected"], id="unbounded"),
        pytest.param("after(within: 3)", [], id="bounded"),
    ],
)
def test_an_unbounded_flow_is_reported_as_a_warning(operator, warnings) -> None:
    rule = DSLParser().parse(
        f'rule "Flow"\ncondition:\n    called_tool("a") {operator} called_tool("b")\n'
    )
    report = RuleValidator().validate(rule)

    assert report.is_valid
    assert [warning.message for warning in report.warnings] == warnings


# ── collect: compile-time shape ───────────────────────────────────────────


class _SpanAdapter:
    async def classify_messages(self, _config, _messages, _tools, *, timeout=None):
        return SimpleNamespace(chunks=[SimpleNamespace(probabilities=[0.9])])

    async def decode_spans(self, _config, _raw_result, _messages, _threshold):
        return []


def test_a_collect_must_return_the_item_it_iterates(tmp_path: Path) -> None:
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "m": {
                    "adapter": "fake",
                    "model": "span-model",
                    "category": "EMAIL",
                    "classify": {"class_index": 0},
                    "spans": {"class_index": 0},
                }
            },
        },
        tmp_path,
        require_rules=False,
    )
    config = replace(
        config,
        quarry_rule_texts=(
            'rule "Projected"\ncondition: true\ncollect:\n'
            '    [projected for s in model("m").spans(messages, 0.5) if true]\n',
        ),
    )

    with pytest.raises(
        RuleValidationError, match="collect: must use the same item name before and after 'for'"
    ):
        DSLScanner(
            config,
            model_inference_backend=ModelInferenceBackend(
                config, adapters={"fake": _SpanAdapter()}
            ),
        )
