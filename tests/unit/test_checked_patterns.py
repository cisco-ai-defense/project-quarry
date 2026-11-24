# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Host capabilities must filter every reader, including saved-policy execution."""

import asyncio
import json

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.engine.captures import matches_for
from quarry.engine.matchers import (
    Modifiers,
    RegexBackend,
    count_occurrences,
    extract_match,
    find_position,
    match_contains,
    match_regex,
    match_spans,
)
from quarry.engine.pattern_checks import (
    PatternCheckError,
    check_names,
    copy_pattern_checks,
    parse_pattern_flags,
    run_pattern_check,
)
from quarry.safety.regex_backend import Re2RegexBackend
from quarry.types.exceptions import RuleValidationError


@pytest.mark.parametrize("backend_type", [RegexBackend, Re2RegexBackend])
@pytest.mark.parametrize(
    "reader", ["presence", "position", "count", "capped", "capture", "extract", "spans", "regex"]
)
def test_readers_share_the_same_accepted_source_matches(backend_type, reader):
    seen = []

    def even(text):
        seen.append(text)
        return len(text) % 2 == 0

    backend = backend_type(pattern_checks={"even_width": even})
    mods = Modifiers(regex=True, checks=("even_width",))
    text, pattern = "a bb ccc dddd", "[a-z]+"
    if reader == "presence":
        assert match_contains(text, pattern, mods, regex=backend)
    elif reader == "position":
        assert find_position(text, pattern, mods, regex=backend) == 2
    elif reader == "count":
        assert count_occurrences(text, pattern, mods, regex=backend) == 2
    elif reader == "capped":
        assert count_occurrences(text, pattern, mods, regex=backend, limit=1) == 1
    elif reader == "capture":
        assert list(matches_for(pattern, text, mods, backend)) == ["bb", "dddd"]
    elif reader == "extract":
        assert extract_match(text, pattern, mods, regex=backend) == "bb"
    elif reader == "spans":
        assert list(match_spans(text, pattern, mods, regex=backend)) == [(2, 4), (9, 13)]
    else:
        assert match_regex(text, pattern, regex=backend, checks=("even_width",))
    assert seen == (
        ["a", "bb"]
        if reader in {"presence", "position", "capped", "extract", "regex"}
        else ["a", "bb", "ccc", "dddd"]
    )


@pytest.mark.parametrize("backend_type", [RegexBackend, Re2RegexBackend])
@pytest.mark.parametrize(
    "text,pattern,mods,expected",
    [
        ("écat cat caté cat_", "cat", Modifiers(fullword=True), [(5, 8)]),
        ("αx x xβ x!", "x", Modifiers(fullword=True), [(3, 4), (8, 9)]),
        ("١x x x٢", "x", Modifiers(fullword=True), [(3, 4)]),
        ("é--force --force x--force --forceé", "--force", Modifiers(fullword=True), [(9, 16)]),
        ("İ İx xİ", "i", Modifiers(fullword=True, nocase=True), [(0, 1)]),
        ("éYQ== YQ== ZYQ==", "a", Modifiers(fullword=True, base64=True), [(6, 10)]),
        ("αxβ", "x", Modifiers(fullword=True, regex=True), [(1, 2)]),
        ("x\u0301", "x", Modifiers(fullword=True), [(0, 1)]),
    ],
)
def test_unicode_edges_preserve_source_coordinates_in_every_reader(
    backend_type, text, pattern, mods, expected
):
    backend = backend_type()
    assert list(match_spans(text, pattern, mods, regex=backend)) == expected
    assert count_occurrences(text, pattern, mods, regex=backend) == len(expected)
    assert find_position(text, pattern, mods, regex=backend) == expected[0][0]
    assert extract_match(text, pattern, mods, regex=backend) == text[slice(*expected[0])]
    assert list(matches_for(pattern, text, mods, backend)) == [
        text[slice(*span)] for span in expected
    ]


@pytest.mark.parametrize("value", [0, 1, None, "true", [], {}, object()])
def test_non_boolean_callback_replies_are_errors_not_truthy_matches(value):
    with pytest.raises(PatternCheckError, match="must return a bool"):
        run_pattern_check(lambda _: value, "synthetic", "PRIVATE_TEXT")


@pytest.mark.parametrize("kind", [ValueError, RuntimeError, SystemExit, KeyboardInterrupt])
def test_callback_failures_never_echo_source_or_exception_text(kind):
    def failure(text):
        raise kind("PRIVATE_EXCEPTION_" + text)

    with pytest.raises(PatternCheckError) as caught:
        run_pattern_check(failure, "synthetic", "PRIVATE_TEXT")
    assert str(caught.value) == "Pattern check 'synthetic' failed"
    assert caught.value.__suppress_context__


def test_cancellation_is_preserved_and_coroutine_reply_is_closed():
    def canceled(_):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        run_pattern_check(canceled, "synthetic", "x")

    async def reply():
        return True

    coroutine = reply()
    with pytest.raises(PatternCheckError):
        run_pattern_check(lambda _: coroutine, "synthetic", "x")
    assert coroutine.cr_frame is None


@pytest.mark.parametrize("name", ["", "a" * 65, "../module", "a\nb", "é", "1a", "_a", None, False])
def test_registry_names_are_bounded_literal_capabilities(name):
    with pytest.raises(ValueError):
        copy_pattern_checks({name: lambda _: True})


@pytest.mark.parametrize(
    "flags",
    [
        "",
        "000",
        "00000",
        "000x",
        "0000:",
        "0000:a,a",
        "0000:../a",
        "0000:" + ",".join("name" + str(i) for i in range(9)),
        "0000:a:b",
        None,
    ],
)
def test_prepared_modifier_names_are_strict_and_bounded(flags):
    with pytest.raises(ValueError):
        parse_pattern_flags(flags)


def test_registry_copy_order_short_circuit_and_zero_cap():
    seen = []
    values = {"reject": lambda text: seen.append(text) is not None}
    backend = RegexBackend(pattern_checks=values)
    values["reject"] = lambda _: True
    mods = Modifiers(regex=True, checks=("reject", "unavailable"))
    assert not match_contains("a bb", "[a-z]+", mods, regex=backend)
    assert seen == ["a", "bb"]
    seen.clear()
    assert count_occurrences("a bb", "[a-z]+", mods, regex=backend, limit=0) == 0
    assert not seen
    with pytest.raises(TypeError):
        backend.pattern_checks["reject"] = lambda _: True
    with pytest.raises(ValueError):
        check_names(("same", "same"))
    with pytest.raises(ValueError):
        copy_pattern_checks({"name" + str(i): lambda _: True for i in range(65)})


CONDITIONS = [
    "any m in messages where $p in m.content",
    "any m in messages where m.content contains_any $p",
    "any m in messages where m.content ~ $p",
    "any m in messages where #p in m.content == 4",
    "any m in messages where #p in m.content >= 1",
    "any m in messages where @p in m.content == 0",
    "len($p.spans(messages)) == 4",
    'len($p.spans(messages)) > 0 and messages[0].content.extract($p.spans(messages[0].content)) contains "bb"',
    "(any m in messages where $p in m.content as v) and (any n in messages where v in n.content)",
    "len([m for m in messages if $p in m.content]) == 1",
]


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("accept", [False, True])
@pytest.mark.parametrize("saved", [False, True])
def test_scans_and_prepared_reload_honor_host_checks_in_every_reader(condition, accept, saved):
    source = f'rule "Named" strings: $p = "[a-z]+" regex check("accept") condition: {condition}'
    registry = {"accept": lambda _: accept}
    policy = compile_policy(source, pattern_checks=registry)
    if saved:
        data = policy.to_bytes()
        assert b'"checks"' in data and b"accept" in data
        with pytest.raises(RuleValidationError, match="unavailable pattern checks"):
            load_compiled_policy(data)
        policy = load_compiled_policy(data, pattern_checks=registry)
        assert policy.to_bytes() == data
    result = policy.scan({"messages": [{"role": "user", "content": "a bb ccc dddd"}]})
    assert result.is_complete, result.error
    assert result.rule_matches == (["Named"] if accept else [])


def test_unknown_names_fail_source_and_prepared_load_without_imports():
    source = 'rule "Named" strings: $p = "x" check("trusted.synthetic") condition: any m in messages where $p in m.content'
    with pytest.raises(RuleValidationError, match="unavailable pattern checks"):
        compile_policy(source)
    policy = compile_policy(source, pattern_checks={"trusted.synthetic": lambda _: True})
    with pytest.raises(RuleValidationError, match="unavailable pattern checks"):
        load_compiled_policy(policy.to_bytes())
    document = json.loads(policy.to_bytes())
    assert document["format_version"] == 12
    assert "lambda" not in json.dumps(document)


@pytest.mark.parametrize("saved", [False, True])
def test_filtered_extraction_preserves_no_match_fallback_and_cuts_only_accepted_spans(saved):
    source = 'rule "Named" strings: $p = "[a-z]+" regex check("tail") condition: messages[0].content.extract($p.spans(messages[0].content)) contains "bb"'
    for predicate, expected in [(lambda _: False, True), (lambda text: text == "dddd", False)]:
        policy = compile_policy(source, pattern_checks={"tail": predicate})
        if saved:
            policy = load_compiled_policy(policy.to_bytes(), pattern_checks={"tail": predicate})
        result = policy.scan({"messages": [{"role": "user", "content": "a bb ccc dddd"}]})
        assert result.is_complete, result.error
        # No accepted matches preserves the existing documented no-cut fallback.
        assert bool(result.rule_matches) is expected


@pytest.mark.parametrize("surface", ["calls", "declared"])
@pytest.mark.parametrize("accept", [False, True])
@pytest.mark.parametrize("saved", [False, True])
def test_tool_arguments_and_declared_inputs_use_the_same_registry(surface, accept, saved):
    if surface == "calls":
        condition = "any c in tool_calls where #p in c.arguments.command == 4"
        config = QuarryConfig(config_version=2)
        data = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "synthetic",
                            "type": "function",
                            "function": {
                                "name": "synthetic",
                                "arguments": '{"command":"a bb ccc dddd"}',
                            },
                        }
                    ],
                }
            ]
        }
    else:
        condition = "any n in memory.notes where #p in n.body == 4"
        config = QuarryConfig(config_version=2, inputs={"memory.notes": {"content": ["body"]}})
        data = {"memory": {"notes": [{"id": "synthetic", "body": "a bb ccc dddd"}]}}
    source = f'rule "Named" strings: $p = "[a-z]+" regex check("accept") condition: {condition}'
    checks = {"accept": lambda _: accept}
    policy = compile_policy(source, runtime_config=config, pattern_checks=checks)
    if saved:
        policy = load_compiled_policy(
            policy.to_bytes(), runtime_config=config, pattern_checks=checks
        )
    result = policy.scan(data)
    assert result.is_complete, result.error
    assert result.rule_matches == (["Named"] if accept else [])


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_callback_cpu_is_owned_by_the_selected_local_work_budget(monkeypatch, mode):
    from quarry.safety import local_match_work as work

    cpu = [0]
    monkeypatch.setattr(work, "thread_time_ns", lambda: cpu[0])
    called = []

    def predicate(text):
        called.append(text)
        cpu[0] += 50
        return True

    source = 'rule "Named" strings: $p = "x" check("accept") condition: any m in messages where $p in m.content'
    runtime = QuarryConfig(
        config_version=2,
        local_match_work=work.LocalMatchWorkConfig(
            (work.LocalMatchWorkGroup("selected", ("Named",), 10, mode),)
        ),
    )
    result = compile_policy(
        source, runtime_config=runtime, pattern_checks={"accept": predicate}
    ).scan({"messages": [{"role": "user", "content": "x"}]})
    assert called
    assert result.is_complete is (mode == "shadow")
    assert result.rule_matches == (["Named"] if mode == "shadow" else [])
    group = result.local_match_work["groups"][0]
    assert group["closed"] and group["active_segments"] == 0 and group["exhausted"]
    assert group["value_cpu_ns"] >= 50
    if mode == "enforce":
        assert result.evaluation_error_categories == {"Named": "evaluation_limit"}


def test_production_callback_checks_the_deadline_after_host_code_returns(monkeypatch):
    from quarry.safety import timer

    now = [0.0]
    monkeypatch.setattr(timer.time, "monotonic", lambda: now[0])

    def predicate(_):
        now[0] = 2.0
        return True

    backend = Re2RegexBackend(pattern_checks={"accept": predicate})
    with (
        pytest.raises(timer.QuarryTimeoutError),
        timer.evaluation_deadline(1, "synthetic deadline"),
    ):
        backend.check_match("accept", "x")


@pytest.mark.parametrize("reader", ["presence", "regex", "count", "position", "capture", "spans"])
def test_reached_regex_backend_errors_never_fall_back_around_a_host_check(reader):
    class ReachedError(RuntimeError):
        pass

    class UnavailableBackend(RegexBackend):
        def iter_match_spans(self, *args):
            raise ReachedError("synthetic reached backend")

    backend = UnavailableBackend(pattern_checks={"accept": lambda _: True})
    mods = Modifiers(regex=True, checks=("accept",))
    with pytest.raises(ReachedError):
        if reader == "presence":
            match_contains("x", "x", mods, regex=backend)
        elif reader == "regex":
            match_regex("x", "x", regex=backend, checks=mods.checks)
        elif reader == "count":
            count_occurrences("x", "x", mods, regex=backend)
        elif reader == "position":
            find_position("x", "x", mods, regex=backend)
        elif reader == "capture":
            list(matches_for("x", "x", mods, backend))
        else:
            list(match_spans("x", "x", mods, regex=backend))


def test_same_name_has_distinct_capabilities_and_no_cross_policy_cached_answers():
    source = 'rule "Named" strings: $p = "x" check("accept") condition: any m in messages where $p in m.content'
    values = {"accept": lambda _: True}
    yes = compile_policy(source, pattern_checks=values)
    values["accept"] = lambda _: False
    no = compile_policy(source, pattern_checks=values)
    data = {"messages": [{"role": "user", "content": "x"}]}
    assert yes.to_bytes() == no.to_bytes()
    for _ in range(3):
        assert yes.scan(data).rule_matches == ["Named"]
        assert no.scan(data).rule_matches == []


async def test_programmatic_mcp_host_binds_registry_but_default_host_rejects_it(monkeypatch):
    from quarry.plugin import server
    from quarry.plugin.service import QuarryPlugin

    source = 'rule "Named" strings: $p = "[a-z]+" regex check("even_width") condition: any m in messages where $p in m.content'
    monkeypatch.setattr(
        server,
        "_service",
        QuarryPlugin(pattern_checks={"even_width": lambda text: len(text) % 2 == 0}),
    )
    data = {"messages": [{"role": "user", "content": "a bb"}]}
    response = await server.mcp.call_tool(
        "quarry_scan", {"rule_text": source, "conversation": data, "explain": False}
    )
    value = response.structured_content
    assert value["completion_state"] == "complete"
    assert value["matches"] == [{"rule_name": "Named", "metadata": {}}]
    monkeypatch.setattr(server, "_service", QuarryPlugin())
    response = await server.mcp.call_tool(
        "quarry_scan", {"rule_text": source, "conversation": data, "explain": False}
    )
    assert "unavailable pattern checks" in str(response)


def test_packaged_host_check_example_executes_with_its_explicit_synthetic_registry():
    from quarry.language.example_records import EXAMPLE_RECORDS

    record = next(item for item in EXAMPLE_RECORDS if item.id == "example:checked-pattern")
    assert record.backend_required
    policy = compile_policy(
        record.rule_text, pattern_checks={"even_width": lambda text: len(text) % 2 == 0}
    )
    for data, expected in [(record.matching, True), (record.non_matching, False)]:
        result = policy.scan(data)
        assert result.is_complete, result.error
        assert bool(result.rule_matches) is expected


@pytest.mark.parametrize("kind", ["deadline", "work"])
def test_callback_control_errors_preserve_category_without_callback_text(kind):
    from quarry.safety.local_match_work import LocalMatchWorkError
    from quarry.safety.timer import QuarryTimeoutError

    error = QuarryTimeoutError if kind == "deadline" else LocalMatchWorkError

    def predicate(text):
        raise error("PRIVATE_CALLBACK_" + text)

    backend = Re2RegexBackend(pattern_checks={"synthetic": predicate})
    with pytest.raises(error) as caught:
        backend.check_match("synthetic", "PRIVATE_INPUT")
    assert "PRIVATE" not in str(caught.value)
    assert "synthetic" in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("replacement", [["a", "a"], ["a\nb"], [True], ["a"] * 9])
def test_prepared_names_cannot_bypass_source_admission(replacement):
    source = 'rule "Named" strings: $p = "x" check("accept") condition: any m in messages where $p in m.content'
    policy = compile_policy(source, pattern_checks={"accept": lambda _: True})
    document = json.loads(policy.to_bytes())
    document["rule"]["artifact"]["atom_specs"][0]["string_semantics"]["checks"] = replacement
    with pytest.raises(ValueError):
        load_compiled_policy(
            json.dumps(document).encode(), pattern_checks={"accept": lambda _: True}
        )


def test_parallel_scans_keep_their_own_registry_and_allow_callback_release():
    import gc
    import weakref
    from concurrent.futures import ThreadPoolExecutor

    class Predicate:
        def __init__(self, accept):
            self.accept = accept

        def __call__(self, _):
            return self.accept

    def scans():
        source = 'rule "Named" strings: $p = "x" check("accept") condition: any m in messages where $p in m.content'
        yes_predicate, no_predicate = Predicate(True), Predicate(False)
        references = [weakref.ref(yes_predicate), weakref.ref(no_predicate)]
        yes = compile_policy(source, pattern_checks={"accept": yes_predicate})
        no = compile_policy(source, pattern_checks={"accept": no_predicate})
        data = {"messages": [{"role": "user", "content": "x"}]}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(policy.scan, data) for _ in range(4) for policy in (yes, no)]
            for index, future in enumerate(futures):
                result = future.result(timeout=30)
                assert result.is_complete
                assert result.rule_matches == (["Named"] if index % 2 == 0 else [])
        return references

    references = scans()
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize(
    "condition,expected",
    [
        ("false", False),
        ("false and (any m in messages where $p in m.content)", False),
        ("true or (any m in messages where $p in m.content)", True),
    ],
)
def test_declared_checks_survive_constant_folding_without_dispatch(condition, expected):
    source = f'rule "Named" strings: $p = "x" check("accept") condition: {condition}'

    def unreachable(_):
        pytest.fail("folded or unused pattern callback was dispatched")

    registry = {"accept": unreachable}
    policy = compile_policy(source, pattern_checks=registry)
    data = policy.to_bytes()
    document = json.loads(data)
    assert document["rule"]["artifact"]["declared_pattern_checks"] == ["accept"]
    with pytest.raises(RuleValidationError, match="unavailable pattern checks"):
        load_compiled_policy(data)
    restored = load_compiled_policy(data, pattern_checks=registry)
    for candidate in (policy, restored):
        result = candidate.scan({"messages": [{"role": "user", "content": "x"}]})
        assert result.is_complete
        assert bool(result.rule_matches) is expected
