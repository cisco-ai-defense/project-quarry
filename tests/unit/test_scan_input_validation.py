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

"""How a scan answers input it cannot evaluate authoritatively.

Malformed or over-limit input is rejected as ``invalid`` before any rule runs -- never clipped to a
prefix that a rule could read as a clean verdict. A batch is only ``complete`` when every
conversation in it scanned authoritatively.
"""

import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from quarry.config import QuarryConfig
from quarry.core.conversation import Conversation
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.safety import timer as timer_module
from quarry.safety import validator as validator_module
from quarry.safety.limits import SafetyLimits
from quarry.safety.regex_backend import RE2_REGEX
from quarry.safety.timer import QuarryTimeoutError
from quarry.safety.validator import validate_conversation, validate_conversation_arguments
from quarry.scanner import DSLScanner
from quarry.types import context as context_module
from quarry.types import messages as canonical_messages
from quarry.types.exceptions import (
    ConversationValidationError,
    EvaluationError,
    IncompleteScanError,
)
from quarry.types.results import Score, Span

ALWAYS = 'rule "Always"\ncondition: true\n'


def _scanner(*rules: str, limits: SafetyLimits | None = None) -> DSLScanner:
    return DSLScanner(
        QuarryConfig(
            quarry_rule_texts=rules or (ALWAYS,),
            safety_limits=limits if limits is not None else SafetyLimits(rule_timeout=None),
        )
    )


def _user(*contents: str) -> dict:
    return {"messages": [{"role": "user", "content": content} for content in contents]}


def _call(name: str, call_id: str) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
        ],
    }


_FLOW_ACROSS_THREE_EVENTS = {
    "messages": [
        _call("start", "start"),
        {"role": "user", "content": "filler"},
        _call("end", "end"),
    ]
}
_NESTED_TOOL_PAYLOAD = {
    "messages": [{"role": "user", "content": "ok"}],
    "tools": [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"level": [{"payload": "x" * 10_000}]}},
        }
    ],
}


def _tool_schema_conversation(parameters: dict) -> dict:
    return {
        "messages": [{"role": "user", "content": "ok"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": parameters},
            }
        ],
    }


def _shared_tool_schema_conversation(parameters: dict, count: int) -> dict:
    return {
        "messages": [{"role": "user", "content": "ok"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": f"lookup_{index}", "parameters": parameters},
            }
            for index in range(count)
        ],
    }


def _nested_schema(depth: int) -> dict:
    schema: dict = {}
    for _ in range(depth - 1):
        schema = {"child": schema}
    return schema


class _ExpandingDict(dict):
    """A container that would expand the payload if any override ran."""

    triggered = False

    def _expand(self, *args, **kwargs):
        self.triggered = True
        dict.__setitem__(self, "expanded", [0] * 100_000)
        raise AssertionError("overridable dict access must not run")

    __iter__ = _expand
    __getitem__ = _expand
    __len__ = _expand
    __bool__ = _expand
    items = _expand
    keys = _expand
    values = _expand
    get = _expand


class _ExpandingList(list):
    """A container that would expand the payload if any override ran."""

    triggered = False

    def _expand(self, *args, **kwargs):
        self.triggered = True
        list.extend(self, [0] * 100_000)
        raise AssertionError("overridable list access must not run")

    __iter__ = _expand
    __getitem__ = _expand
    __len__ = _expand
    __bool__ = _expand


class _HostileLeaf:
    """An unsupported leaf whose sizing hook must never execute."""

    triggered = False

    def __sizeof__(self):
        self.triggered = True
        raise AssertionError("custom leaf sizing must not run")


@pytest.mark.parametrize(
    ("conversation", "limits", "reason"),
    [
        pytest.param("not a dict", None, "Expected dict, got str", id="string"),
        pytest.param(["list"], None, "Expected dict, got list", id="list"),
        pytest.param({"tools": []}, None, "Missing required 'messages' key", id="no-messages"),
        pytest.param({"messages": "x"}, None, "'messages' must be a list", id="messages-not-list"),
        pytest.param(
            {"messages": [{"role": "user", "content": "t"}], "tools": "x"},
            None,
            "'tools' must be a list",
            id="tools-not-list",
        ),
        pytest.param(
            {"messages": [{"content": "test"}]},
            None,
            "Message at index 0 missing 'role' key",
            id="missing-role",
        ),
        pytest.param(
            {"messages": [{"role": "user"}]},
            None,
            "missing both 'content' and 'tool_calls'",
            id="missing-content",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": None}]},
            None,
            "content must be a string or OpenAI content-part list",
            id="null-user-content",
        ),
        pytest.param({"messages": []}, None, "'messages' list is empty", id="empty-messages"),
        pytest.param(
            _user(*(f"msg {i}" for i in range(500))),
            SafetyLimits(max_messages=100),
            "max_messages",
            id="over-message-count",
        ),
        pytest.param(
            _user("x" * 1_000_000),
            SafetyLimits(max_message_size=1000),
            "max_message_size",
            id="over-message-size",
        ),
        pytest.param(
            _NESTED_TOOL_PAYLOAD,
            SafetyLimits(max_total_size=5_000),
            "total_size",
            id="nested-tool-payload-counts-toward-total-size",
        ),
        pytest.param(
            _FLOW_ACROSS_THREE_EVENTS,
            SafetyLimits(max_timeline_items=2),
            "timeline_items",
            id="over-timeline-items",
        ),
    ],
)
def test_input_that_cannot_be_scanned_whole_is_invalid(conversation, limits, reason) -> None:
    result = _scanner(limits=limits).scan_data(conversation)

    assert result.rule_matches == []
    assert result.completion_state == "invalid"
    assert reason in result.validation_error
    assert result.error.startswith("Conversation validation error: ")
    assert result.completion_issues == [f"invalid conversation: {result.validation_error}"]


def test_unsupported_top_level_branch_is_rejected_without_traversal(monkeypatch) -> None:
    monkeypatch.setattr(validator_module, "_MAX_RAW_JSON_VISITS", 1)
    branch = _ExpandingList([0] * 100_000)
    conversation = {
        "messages": [{"role": "user", "content": "ok"}],
        "unsupported": branch,
    }

    result = _scanner().scan_data(conversation)

    assert (result.rule_matches, result.completion_state) == ([], "invalid")
    assert "unsupported top-level keys: ['unsupported']" in result.validation_error
    assert "raw JSON validation work limit" not in result.validation_error
    assert branch.triggered is False


def test_total_size_counts_shared_scalars_once_but_distinct_copies_separately() -> None:
    """Immutable scalars keep live-object accounting while containers are aggregate-charged."""

    scanner = _scanner(limits=SafetyLimits(max_total_size=10_000))
    content = "shared" * 500
    shared = _user(*([content] * 8))
    copies = _user(*[(content + "!")[:-1] for _ in range(8)])

    assert [message["content"] for message in copies["messages"]] == [content] * 8
    shared_result = scanner.scan_data(shared)
    copies_result = scanner.scan_data(copies)

    assert (shared_result.completion_state, shared_result.rule_matches) == ("complete", ["Always"])
    assert (copies_result.completion_state, copies_result.rule_matches) == ("invalid", [])
    assert "max_total_size" in copies_result.validation_error


@pytest.mark.parametrize(
    ("tool_count", "expected_state"),
    [
        pytest.param(2, "complete", id="shared-schema-within-aggregate-budget"),
        pytest.param(4, "invalid", id="shared-schema-over-aggregate-budget"),
    ],
)
def test_total_size_charges_a_shared_schema_container_for_every_tool(
    tool_count: int, expected_state: str
) -> None:
    schema = {"enum": list(range(10_000))}
    conversation = _shared_tool_schema_conversation(schema, tool_count)

    result = _scanner(limits=SafetyLimits(max_total_size=500_000)).scan_data(conversation)

    assert result.completion_state == expected_state
    if expected_state == "complete":
        assert result.rule_matches == ["Always"]
    else:
        assert result.rule_matches == []
        assert "max_total_size" in result.validation_error


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param(_ExpandingDict({"type": "object"}), id="dict-subclass"),
        pytest.param({"enum": _ExpandingList([1, 2, 3])}, id="list-subclass"),
    ],
)
def test_overridable_schema_containers_are_rejected_without_access(parameters: dict) -> None:
    result = _scanner().scan_data(_tool_schema_conversation(parameters))

    assert (result.rule_matches, result.completion_state) == ([], "invalid")
    assert "use exact built-in dict and list containers" in result.validation_error
    nested = parameters if isinstance(parameters, _ExpandingDict) else dict.get(parameters, "enum")
    assert nested.triggered is False


def test_overridable_top_level_container_is_rejected_without_access() -> None:
    conversation = _ExpandingDict({"messages": [{"role": "user", "content": "ok"}]})

    with pytest.raises(
        ConversationValidationError,
        match="use exact built-in dict and list containers",
    ):
        validate_conversation(conversation)

    assert conversation.triggered is False


def test_repeated_unsupported_leaves_never_run_custom_sizing() -> None:
    leaf = _HostileLeaf()
    conversation = _tool_schema_conversation({"enum": [leaf] * 1_000})

    result = _scanner().scan_data(conversation)

    assert (result.rule_matches, result.completion_state) == ([], "invalid")
    assert "contains unsupported JSON value _HostileLeaf" in result.validation_error
    assert leaf.triggered is False


@pytest.mark.parametrize("enforce_limits", [True, False])
@pytest.mark.parametrize("position", ["schema", "arguments", "content"])
def test_unsupported_leaves_never_run_class_identity_hooks(enforce_limits, position) -> None:
    class HostileIdentity:
        triggered = False

        @property
        def __class__(self):
            self.triggered = True
            raise AssertionError("untrusted class identity hook must not run")

        def __sizeof__(self):
            self.triggered = True
            raise AssertionError("untrusted sizing hook must not run")

    leaf = HostileIdentity()
    if position == "schema":
        conversation = _tool_schema_conversation({"value": leaf})
    elif position == "arguments":
        call = _call("lookup", "call-1")
        call["tool_calls"][0]["function"]["arguments"] = leaf
        conversation = {"messages": [call]}
    else:
        conversation = {"messages": [{"role": "user", "content": leaf}]}

    result = _scanner(
        limits=SafetyLimits(rule_timeout=None, enforce_limits=enforce_limits)
    ).scan_data(conversation)

    assert result.completion_state == "invalid"
    assert result.rule_matches == []
    assert leaf.triggered is False


def test_exact_builtin_containers_reuse_the_bounded_preflight(monkeypatch) -> None:
    calls = 0
    deep_size = context_module._bounded_deep_size

    def counted_deep_size(*args, **kwargs):
        nonlocal calls
        calls += 1
        return deep_size(*args, **kwargs)

    monkeypatch.setattr(context_module, "_bounded_deep_size", counted_deep_size)
    schema = {
        "type": "object",
        "properties": {"query": {"type": "array", "items": {"type": "string"}}},
    }

    result = _scanner().scan_data(_tool_schema_conversation(schema))

    assert (result.rule_matches, result.completion_state) == (["Always"], "complete")
    assert calls == 1


@pytest.mark.parametrize("parse_arguments", [False, True])
def test_tool_schema_depth_accepts_the_exact_boundary(parse_arguments: bool) -> None:
    validate_conversation(
        _tool_schema_conversation(_nested_schema(100)),
        parse_arguments=parse_arguments,
    )

    with pytest.raises(
        ConversationValidationError,
        match=r"tools\[0\]\.function\.parameters exceeds max_nesting_depth 100",
    ):
        validate_conversation(
            _tool_schema_conversation(_nested_schema(101)),
            parse_arguments=parse_arguments,
        )


def test_tool_schema_depth_honors_a_custom_lower_limit() -> None:
    validate_conversation(
        _tool_schema_conversation(_nested_schema(3)),
        max_nesting_depth=3,
    )

    with pytest.raises(ConversationValidationError, match="exceeds max_nesting_depth 3"):
        validate_conversation(
            _tool_schema_conversation(_nested_schema(4)),
            max_nesting_depth=3,
        )


@pytest.mark.parametrize("parse_arguments", [False, True])
def test_tool_schema_cycles_fail_closed(parse_arguments: bool) -> None:
    schema: dict = {}
    schema["self"] = schema

    with pytest.raises(ConversationValidationError, match="contains a cyclic reference"):
        validate_conversation(
            _tool_schema_conversation(schema),
            parse_arguments=parse_arguments,
        )


@pytest.mark.parametrize("parse_arguments", [False, True])
@pytest.mark.parametrize(
    ("schema", "reason"),
    [
        pytest.param({"value": object()}, "unsupported JSON value object", id="unsupported-value"),
        pytest.param({1: "value"}, "non-string object key", id="non-string-key"),
        pytest.param({"value": float("inf")}, "non-finite number", id="non-finite-number"),
    ],
)
def test_tool_schema_non_json_values_fail_closed(
    parse_arguments: bool, schema: dict, reason: str
) -> None:
    with pytest.raises(ConversationValidationError, match=reason):
        validate_conversation(
            _tool_schema_conversation(schema),
            parse_arguments=parse_arguments,
        )


def test_direct_tool_schema_validation_has_a_work_budget(monkeypatch) -> None:
    monkeypatch.setattr(validator_module, "_MAX_JSON_TREE_VISITS", 4)

    with pytest.raises(ConversationValidationError, match="JSON validation work limit 4"):
        validate_conversation(_tool_schema_conversation({"enum": [0, 1, 2, 3]}))


def test_tool_schema_work_budget_is_aggregate_across_tools(monkeypatch) -> None:
    monkeypatch.setattr(validator_module, "_MAX_JSON_TREE_VISITS", 6)
    conversation = _shared_tool_schema_conversation({"enum": [0, 1]}, 2)

    with pytest.raises(
        ConversationValidationError,
        match=r"tools\[1\]\.function\.parameters exceeds JSON validation work limit 6",
    ):
        validate_conversation(conversation)


def test_total_size_preflight_precedes_wide_tool_schema_validation(monkeypatch) -> None:
    calls = 0
    deep_size = context_module._bounded_deep_size

    def counted_deep_size(*args, **kwargs):
        nonlocal calls
        calls += 1
        return deep_size(*args, **kwargs)

    monkeypatch.setattr(context_module, "_bounded_deep_size", counted_deep_size)
    # If schema validation ran first, the unsupported leaf would win. The wide container's
    # shallow footprint alone exceeds the budget, so preflight need not visit that leaf.
    conversation = _tool_schema_conversation({"enum": [object()] * 1_000})

    result = _scanner(limits=SafetyLimits(max_total_size=2_000)).scan_data(conversation)

    assert (result.rule_matches, result.completion_state) == ([], "invalid")
    assert "max_total_size" in result.validation_error
    assert "unsupported JSON value" not in result.validation_error
    assert calls == 1


def test_argument_failures_only_invalidate_the_rule_reading_the_bad_value() -> None:
    scanner = _scanner(
        'rule "Names"\ncondition: called_tool("nonobject") and called_tool("nonfinite") and called_tool("good")',
        'rule "Nonobject"\ncondition: any c in get_tool_calls("nonobject") where c.arguments.value == 1',
        'rule "Nonfinite"\ncondition: any c in get_tool_calls("nonfinite") where c.arguments.value == 1',
        'rule "Good"\ncondition: any c in get_tool_calls("good") where c.arguments.value == 1',
    )
    calls = [
        {"id": "c1", "type": "function", "function": {"name": "nonobject", "arguments": "[]"}},
        {
            "id": "c2",
            "type": "function",
            "function": {"name": "nonfinite", "arguments": '{"value":NaN}'},
        },
        {"id": "c3", "type": "function", "function": {"name": "good", "arguments": '{"value":1}'}},
    ]

    result = scanner.scan_data({"messages": [{"role": "assistant", "tool_calls": calls}]})

    assert (result.rule_matches, result.completion_state) == (["Names", "Good"], "incomplete")
    assert result.validation_error is None
    assert set(result.evaluation_errors) == {"Nonfinite"}
    assert "tool_calls[1].function.arguments" in result.evaluation_errors["Nonfinite"]


def test_a_flow_rejected_over_the_timeline_limit_matches_once_the_limit_allows_it() -> None:
    """Cutting the tail would have dropped ``end`` and answered the flow as absent."""

    rule = 'rule "Bounded Flow"\ncondition:\n    called_tool("end") after called_tool("start")\n'

    tight = _scanner(rule, limits=SafetyLimits(max_timeline_items=2))
    loose = _scanner(rule, limits=SafetyLimits(max_timeline_items=100))

    assert tight.scan_data(_FLOW_ACROSS_THREE_EVENTS).completion_state == "invalid"
    allowed = loose.scan_data(_FLOW_ACROSS_THREE_EVENTS)
    assert (allowed.rule_matches, allowed.completion_state) == (["Bounded Flow"], "complete")


@pytest.mark.parametrize(
    ("rule", "expected_operation"),
    [
        pytest.param(
            'rule "Evil Regex"\ncondition:\n    any m in user_messages where m.content ~ "(a+)+$"\n',
            "search",
            id="regex-match",
        ),
        pytest.param(
            'rule "Evil Regex"\nstrings:\n    $evil = "(a+)+$" regex\n'
            "condition:\n    any m in user_messages where #evil in m.content >= 1\n",
            "count",
            id="regex-count",
        ),
    ],
)
def test_a_catastrophic_backtracking_regex_scans_with_re2(
    rule, expected_operation, monkeypatch
) -> None:
    """The scanner must use its RE2 backend on regex search and count paths."""

    operations = []
    search = RE2_REGEX.search_match
    count = RE2_REGEX.count_up_to

    def record_search(*args, **kwargs):
        operations.append("search")
        return search(*args, **kwargs)

    def record_count(*args, **kwargs):
        operations.append("count")
        return count(*args, **kwargs)

    monkeypatch.setattr(RE2_REGEX, "search_match", record_search)
    monkeypatch.setattr(RE2_REGEX, "count_up_to", record_count)
    scanner = _scanner(rule)
    # Keep a finite input here; the backend assertion proves which regex engine ran.
    result = scanner.scan_data(_user("a" * 12 + "b"))

    assert expected_operation in operations
    assert result.rule_matches == []
    assert result.completion_state == "complete"


def test_an_empty_rule_subset_cannot_return_an_authoritative_clean_result() -> None:
    scanner = _scanner()

    clean = scanner.scan_data(_user("hello"), rules_subset=set())
    invalid = scanner.scan_data({"tools": []}, rules_subset=set())

    assert (clean.rule_matches, clean.completion_state) == ([], "incomplete")
    assert clean.error == "Rule selection error: explicit rule subset is empty"
    with pytest.raises(IncompleteScanError):
        clean.raise_for_status()
    assert invalid.completion_state == "invalid"


@pytest.mark.parametrize("subset", [{"Unknown"}, {"Always", "Unknown"}])
def test_unknown_rule_in_explicit_subset_fails_closed(subset: set[str]) -> None:
    result = _scanner().scan_data(_user("hello"), rules_subset=subset)

    assert result.rule_matches == []
    assert result.completion_state == "incomplete"
    assert result.error == "Rule selection error: unknown rule(s) selected: Unknown"
    with pytest.raises(IncompleteScanError):
        result.raise_for_status()


def test_a_known_rule_subset_still_evaluates_normally() -> None:
    result = _scanner().scan_data(_user("hello"), rules_subset={"Always"})

    assert (result.rule_matches, result.completion_state) == (["Always"], "complete")


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        pytest.param("{not json", "malformed JSON", id="malformed-json"),
        pytest.param("[1, 2]", "must decode to a JSON object", id="non-object-json"),
        pytest.param('{"value": NaN}', "non-finite number", id="non-finite-number"),
        pytest.param(
            '{"value": ' + "[" * 101 + "]" * 101 + "}",
            "max_nesting_depth",
            id="over-depth",
        ),
    ],
)
def test_argument_decode_failures_do_not_affect_name_only_rules(
    arguments: str, reason: str
) -> None:
    conversation = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": arguments},
                    }
                ],
            }
        ]
    }
    result = _scanner('rule "Name"\ncondition: called_tool("lookup")\n').scan_data(conversation)

    assert (result.rule_matches, result.completion_state) == (["Name"], "complete")
    reader = _scanner(
        'rule "Read" condition: any c in tool_calls where c.arguments.value == 1'
    ).scan_data(conversation)
    if reason in {"malformed JSON", "must decode to a JSON object"}:
        assert reader.is_complete and reader.rule_matches == []
    else:
        assert reader.completion_state == "incomplete"
        assert reason in reader.evaluation_errors["Read"]


def test_argument_size_limit_runs_before_json_validation() -> None:
    conversation = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{not json"},
                    }
                ],
            }
        ]
    }
    result = _scanner(limits=SafetyLimits(max_tool_argument_size=4)).scan_data(conversation)

    assert result.completion_state == "invalid"
    assert "max_tool_argument_size" in result.validation_error


def test_valid_wide_integer_arguments_remain_accepted() -> None:
    # The rule-local decoder preserves exact integers within the active JSON
    # serializer's decimal-digit limit.
    arguments = '{"value": ' + "1" + "0" * 399 + "}"
    conversation = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": arguments},
                    }
                ],
            }
        ]
    }

    result = _scanner('rule "Name"\ncondition: called_tool("lookup")\n').scan_data(conversation)

    assert (result.rule_matches, result.completion_state) == (["Name"], "complete")
    # The direct validator retains its existing orjson parsing contract.
    with pytest.raises(ConversationValidationError, match="malformed JSON"):
        validate_conversation(conversation, parse_arguments=True)


@pytest.mark.skipif(sys.get_int_max_str_digits() == 0, reason="no active integer digit limit")
@pytest.mark.parametrize("extra_digit", [False, True], ids=["at-limit", "over-limit"])
@pytest.mark.parametrize("negative", [False, True], ids=["positive", "negative"])
def test_tool_integer_digit_limit_is_explicit_not_malformed(
    extra_digit: bool, negative: bool
) -> None:
    limit = sys.get_int_max_str_digits()
    digits = "1" * (limit + int(extra_digit))
    assert len(digits) < SafetyLimits().max_tool_argument_size
    integer_text = ("-" if negative else "") + digits
    conversation = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"value": ' + integer_text + "}",
                        },
                    }
                ],
            }
        ]
    }
    call = Conversation(conversation["messages"]).messages.tool_calls[0]
    scanner = _scanner('rule "Name"\ncondition: called_tool("lookup")\n')

    if extra_digit:
        with pytest.raises(EvaluationError, match="JSON integer digit limit") as error:
            _ = call.arguments
        assert f"active limit is {limit}" in str(error.value)
        result = scanner.scan_data(conversation)
        assert (result.rule_matches, result.completion_state) == (["Name"], "complete")
        reader = _scanner(
            'rule "Read" condition: any c in tool_calls where c.arguments.value != null'
        ).scan_data(conversation)
        assert reader.completion_state == "incomplete"
        assert f"active limit is {limit}" in reader.evaluation_errors["Read"]
        assert "malformed JSON" not in reader.evaluation_errors["Read"]
    else:
        assert call.arguments["value"] == int(integer_text)
        result = scanner.scan_data(conversation)
        assert (result.rule_matches, result.completion_state) == (["Name"], "complete")


def test_identical_tool_arguments_are_validated_once_per_scan(monkeypatch) -> None:
    arguments = '{"command": "read"}'
    conversation = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": arguments},
                    }
                    for index in range(3)
                ],
            }
        ]
    }
    decode = canonical_messages.json.loads
    calls: list[str] = []

    def counting_decode(value, *args, **kwargs):
        if value == arguments:
            calls.append(value)
        return decode(value, *args, **kwargs)

    monkeypatch.setattr(canonical_messages.json, "loads", counting_decode)
    scanner = _scanner(
        'rule "Name"\ncondition: any c in tool_calls where c.arguments.command == "read"\n'
    )

    first = scanner.scan_data(conversation)
    second = scanner.scan_data(conversation)

    assert (first.rule_matches, first.completion_state) == (["Name"], "complete")
    assert (second.rule_matches, second.completion_state) == (["Name"], "complete")
    # The deciding call is decoded once in each scan; names and preparation decode none.
    assert calls == [arguments] * 2


def test_argument_preflight_does_not_retain_decoded_trees(monkeypatch) -> None:
    raw_values = ['{"value": [1, 2]}', '{"value": [3, 4]}', '{"value": [1, 2]}']
    messages = Conversation(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": arguments},
                    }
                    for index, arguments in enumerate(raw_values)
                ],
            }
        ]
    ).messages
    decode = canonical_messages.json.loads
    decoded: list[str] = []

    def counting_decode(value, *args, **kwargs):
        if value in raw_values:
            decoded.append(value)
        return decode(value, *args, **kwargs)

    monkeypatch.setattr(canonical_messages.json, "loads", counting_decode)
    validate_conversation_arguments(messages)

    assert decoded == raw_values[:2]
    assert all(call._arguments_cache is None for call in messages.tool_calls)
    assert messages.tool_calls[0].arguments == {"value": [1, 2]}
    assert decoded == raw_values[:2] + raw_values[:1]


@pytest.mark.parametrize("fail_on_timeout", [False, True])
def test_conversation_preparation_respects_fail_on_timeout(monkeypatch, fail_on_timeout):
    scanner = _scanner(
        'rule "Read"\ncondition: called_tool("lookup")',
        limits=SafetyLimits(fail_on_timeout=fail_on_timeout),
    )

    def expired():
        raise QuarryTimeoutError("Scan deadline exceeded")

    monkeypatch.setattr(canonical_messages, "_raise_if_combine_expired", expired)
    data = {"messages": [_call("lookup", "c")]}
    if fail_on_timeout:
        with pytest.raises(QuarryTimeoutError, match="Scan deadline exceeded"):
            scanner.scan_data(data)
    else:
        result = scanner.scan_data(data)
        assert result.completion_state == "incomplete"
        assert result.error_category == "timeout"


@pytest.mark.parametrize("fail_on_timeout", [False, True])
def test_distinct_argument_reads_stop_at_the_inspection_deadline(
    monkeypatch, fail_on_timeout: bool
) -> None:
    # A rule requiring every decoded value must respect the shared inspection deadline.
    # Advance only the deadline clock during decoding so setup and scheduling cannot
    # consume this test's budget before it reaches the behavior under inspection.
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(timer_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    arguments = [f'{{"nonce": {index}}}' for index in range(100)]
    conversation = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": value},
                    }
                    for index, value in enumerate(arguments)
                ],
            }
        ]
    }
    decode = canonical_messages.json.loads
    decoded: list[str] = []

    def slow_decode(value, *args, **kwargs):
        if isinstance(value, str) and value.startswith('{"nonce":'):
            decoded.append(value)
            clock.now += 0.015
        return decode(value, *args, **kwargs)

    monkeypatch.setattr(canonical_messages.json, "loads", slow_decode)
    scanner = _scanner(
        'rule "Read"\ncondition: all c in tool_calls where c.arguments.nonce != null\n',
        limits=SafetyLimits(fail_on_timeout=fail_on_timeout),
    )

    if fail_on_timeout:
        with pytest.raises(QuarryTimeoutError, match="Scan deadline exceeded"):
            asyncio.run(scanner.scan_data_async(conversation, deadline_seconds=0.05))
    else:
        result = asyncio.run(scanner.scan_data_async(conversation, deadline_seconds=0.05))
    assert 0 < len(decoded) < len(arguments)
    assert decoded == arguments[:4]
    if not fail_on_timeout:
        assert (result.rule_matches, result.completion_state) == ([], "incomplete")
        assert result.validation_error is None
        with pytest.raises(IncompleteScanError):
            result.raise_for_status()


def test_invalid_argument_keeps_its_call_location_across_scans() -> None:
    scanner = _scanner('rule "Read"\ncondition: any c in tool_calls where c.arguments.value == 1\n')
    for bad_index in (1, 2):
        conversation = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"value": NaN}' if index == bad_index else "{}",
                            },
                        }
                        for index in range(3)
                    ],
                }
            ]
        }

        result = scanner.scan_data(conversation)

        assert result.completion_state == "incomplete"
        assert result.validation_error is None
        assert f"tool_calls[{bad_index}].function.arguments" in result.evaluation_errors["Read"]


def test_argument_cache_does_not_require_hashable_string_subclasses() -> None:
    class UnhashableString(str):
        __hash__ = None

    conversation = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": UnhashableString("{}"),
                        },
                    }
                ],
            }
        ]
    }

    result = _scanner('rule "Name"\ncondition: called_tool("lookup")\n').scan_data(conversation)

    assert (result.rule_matches, result.completion_state) == (["Name"], "complete")


def test_argument_string_subclass_never_runs_sizing_or_encoding_overrides() -> None:
    calls: list[str] = []

    class HookedString(str):
        def __sizeof__(self):
            calls.append("sizeof")
            raise AssertionError("string sizing override must not run")

        def encode(self, *args, **kwargs):
            calls.append("encode")
            raise AssertionError("string encoding override must not run")

    conversation = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": HookedString("{}")},
                    }
                ],
            }
        ]
    }

    result = _scanner('rule "Name"\ncondition: called_tool("lookup")\n').scan_data(conversation)

    assert (result.rule_matches, result.completion_state) == (["Name"], "complete")
    assert calls == []


_COUNTS = 'rule "Test"\ncondition: len(messages) > 0\n'


@pytest.mark.parametrize(
    ("batch", "limits", "matches", "state", "failed"),
    [
        pytest.param(
            [_user("hi"), _user("there")],
            None,
            ["conv_1:Test", "conv_2:Test"],
            "complete",
            [],
            id="every-conversation-clean",
        ),
        pytest.param(
            [_user("hi"), {"messages": []}],
            None,
            ["conv_1:Test"],
            "incomplete",
            ["conv_2"],
            id="an-invalid-conversation-blocks-complete",
        ),
        pytest.param(
            [_user("a", "b")],
            SafetyLimits(max_messages=1),
            [],
            "incomplete",
            ["conv_1"],
            id="an-over-limit-conversation-blocks-complete",
        ),
        pytest.param([], None, [], "invalid", [], id="empty-batch"),
    ],
)
def test_a_batch_is_complete_only_when_every_conversation_is(
    tmp_path, batch, limits, matches, state, failed
) -> None:
    result = _scanner(_COUNTS, limits=limits).scan_batch_data(batch, tmp_path / "batch.json")

    assert result.rule_matches == matches
    assert result.completion_state == state
    assert sorted(result.evaluation_errors or {}) == failed


def test_a_batch_scan_calls_a_two_argument_scan_data_override(tmp_path) -> None:
    # DSLScanner is public; a subclass that overrides scan_data(data, file_path) keeps
    # working in batch mode, which passes explain only when it is requested.
    class TwoArguments(DSLScanner):
        def scan_data(self, data, file_path=None):
            return super().scan_data(data, file_path)

    rule = 'rule "R" condition: any m in user_messages where m.content contains "a"'
    scanner = TwoArguments(QuarryConfig(quarry_rule_texts=(rule,)))

    result = scanner.scan_batch_data([_user("a")], tmp_path / "batch.json")

    assert result.rule_matches == ["conv_1:R"]


def test_an_incomplete_batch_fails_closed_on_raise_for_status(tmp_path) -> None:
    result = _scanner(_COUNTS).scan_batch_data(
        [_user("hi"), {"messages": []}], tmp_path / "batch.json"
    )

    with pytest.raises(IncompleteScanError):
        result.raise_for_status()


# ── collect: the producer's output is bounded and attributed per conversation ──────────────


class _SpanAdapter:
    """Fake model adapter: one classification and one span per character of the first message."""

    async def classify_messages(self, _config, _messages, _tools, *, timeout=None):
        return SimpleNamespace(chunks=[SimpleNamespace(probabilities=[0.9])])

    async def decode_spans(self, _config, _raw_result, messages, _threshold):
        content = messages[0].content
        return [Span(0, index, index + 1, Score(0.9)) for index in range(len(content))]


_SPANS = 'rule "Spans"\ncondition: true\ncollect:\n    [s for s in model("m").spans(messages, 0.5) if true]\n'


def _span_scanner(tmp_path: Path, **limits) -> DSLScanner:
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
            **limits,
        },
        tmp_path,
        require_rules=False,
    )
    config = replace(config, quarry_rule_texts=(_SPANS,))
    backend = ModelInferenceBackend(config, adapters={"fake": _SpanAdapter()})
    return DSLScanner(config, model_inference_backend=backend)


def test_a_collect_producer_over_the_comprehension_limit_fails_the_rule(tmp_path) -> None:
    result = _span_scanner(tmp_path, max_comprehension_size=1).scan_data(_user("ab"))

    assert result.rule_matches == []
    assert result.outputs == {}
    assert result.completion_state == "incomplete"
    assert result.evaluation_errors == {
        "Spans": "collect producer returned 2 items, exceeding max_comprehension_size=1"
    }


def test_batch_outputs_are_attributed_to_their_conversation(tmp_path) -> None:
    result = _span_scanner(tmp_path).scan_batch_data(
        [_user("ab"), _user("xyz")], tmp_path / "batch.json"
    )

    assert result.rule_matches == ["conv_1:Spans", "conv_2:Spans"]
    assert result.completion_state == "complete"
    assert {name: len(output.items) for name, output in result.outputs.items()} == {
        "conv_1:Spans": 2,
        "conv_2:Spans": 3,
    }


# ── a batch carries every conversation's completion signal, attributed to that conversation ──


class _SlowClassifier:
    """Fake model adapter whose classification outlasts any rule timeout used here."""

    async def classify_messages(self, _config, _messages, _tools, *, timeout=None):
        await asyncio.sleep(5)


def _slow_scanner(tmp_path: Path, *, fail_on_timeout: bool) -> DSLScanner:
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "rule_timeout": 0.05,
            "fail_on_timeout": fail_on_timeout,
            "models": {"m": {"adapter": "fake", "model": "slow", "classify": {"class_index": 0}}},
        },
        tmp_path,
        require_rules=False,
    )
    config = replace(
        config,
        quarry_rule_texts=('rule "Slow"\ncondition:\n    model("m").classify(messages) >= 0.5\n',),
    )
    backend = ModelInferenceBackend(config, adapters={"fake": _SlowClassifier()})
    return DSLScanner(config, model_inference_backend=backend)


_CAPPED_JOIN = (
    'rule "Capped"\ncondition:\n'
    "    (any m in user_messages where ((any c in tool_calls where c.arguments.cmd contains "
    'm.content) after called_tool("y")) and not (any d in tool_calls where d.name == "q" and '
    "d.arguments.cmd contains m.content)) or (all a in assistant_messages where a.content "
    'contains "z")\n'
)
_PAST_THE_CANDIDATE_CAP = {
    "messages": [
        *_user(*(f"m{i}" for i in range(257)))["messages"],
        {"role": "assistant", "content": "a"},
    ]
}


@pytest.mark.parametrize(
    ("make_scanner", "batch", "signals"),
    [
        pytest.param(
            lambda tmp_path: _slow_scanner(tmp_path, fail_on_timeout=False),
            [_user("hi")],
            (
                "Conversation 1: Rule Slow timeout",
                ["conv_1:Slow"],
                None,
                None,
                ["Conversation 1: Rule Slow timeout", "rules timed out: conv_1:Slow"],
            ),
            id="a-timed-out-rule",
        ),
        pytest.param(
            lambda tmp_path: _slow_scanner(tmp_path, fail_on_timeout=True),
            [_user("hi")],
            (
                "Conversation 1: Rule Slow timeout",
                None,
                {"conv_1": "Rule Slow timeout"},
                None,
                ["rule evaluation failed: conv_1"],
            ),
            id="a-timeout-raised-by-fail-on-timeout",
        ),
        pytest.param(
            lambda tmp_path: _span_scanner(tmp_path, max_comprehension_size=1),
            [_user("ab")],
            (
                None,
                None,
                {
                    "conv_1:Spans": (
                        "collect producer returned 2 items, exceeding max_comprehension_size=1"
                    )
                },
                None,
                ["rule evaluation failed: conv_1:Spans"],
            ),
            id="a-rule-evaluation-error",
        ),
        pytest.param(
            lambda _tmp_path: _scanner(_CAPPED_JOIN, limits=SafetyLimits(rule_timeout=60.0)),
            [_PAST_THE_CANDIDATE_CAP],
            (
                None,
                None,
                None,
                [
                    "conv_1: 'm' hit the candidate cap (256 user_messages items) without a "
                    "match; result may be incomplete"
                ],
                ["evaluation was limited"],
            ),
            id="a-candidate-cap-warning",
        ),
    ],
)
def test_a_batch_reports_each_conversations_incomplete_evaluation(
    tmp_path, make_scanner, batch, signals
) -> None:
    result = make_scanner(tmp_path).scan_batch_data(batch, tmp_path / "batch.json")

    assert (result.rule_matches, result.completion_state) == ([], "incomplete")
    assert (
        result.error,
        result.timed_out_rules,
        result.evaluation_errors,
        result.evaluation_warnings,
        result.completion_issues,
    ) == signals


# ── directories: every JSON file is scanned, and an unreadable one reports its own error ──


_SAID_HI = 'rule "Said hi"\ncondition:\n    user_said("hi")\n'
_UNREADABLE = "unexpected end of data: line 1 column 2 (char 1)"


@pytest.mark.parametrize(
    ("recursive", "verbose", "results", "progress"),
    [
        pytest.param(
            False,
            False,
            [("a.json", ["Said hi"], None), ("broken.json", [], _UNREADABLE)],
            "Scanning 2 files in {directory}\n[1/2] a.json\n[2/2] broken.json\n",
            id="top-level-files",
        ),
        pytest.param(
            True,
            False,
            [
                ("a.json", ["Said hi"], None),
                ("broken.json", [], _UNREADABLE),
                ("nested/b.json", [], None),
            ],
            "Scanning 3 files in {directory}\n[1/3] a.json\n[2/3] broken.json\n[3/3] b.json\n",
            id="recursive",
        ),
        pytest.param(
            False,
            True,
            [("a.json", ["Said hi"], None), ("broken.json", [], _UNREADABLE)],
            "",
            id="verbose-logs-instead-of-printing",
        ),
    ],
)
def test_a_directory_scan_answers_for_each_json_file(
    tmp_path, capsys, recursive, verbose, results, progress
) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.json").write_text('{"messages": [{"role": "user", "content": "hi"}]}')
    (tmp_path / "broken.json").write_text("{")
    (tmp_path / "nested" / "b.json").write_text(
        '{"messages": [{"role": "user", "content": "hello"}]}'
    )
    (tmp_path / "notes.txt").write_text("not a conversation")
    scanner = DSLScanner(
        QuarryConfig(quarry_rule_texts=(_SAID_HI,)), recursive=recursive, verbose=verbose
    )

    scanned = scanner.scan_directory(tmp_path)

    assert [
        (result.file_path.relative_to(tmp_path).as_posix(), result.rule_matches, result.error)
        for result in scanned
    ] == results
    assert capsys.readouterr().out == progress.format(directory=tmp_path)
