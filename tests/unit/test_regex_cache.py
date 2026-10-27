# Copyright 2025 Cisco Systems, Inc. and its affiliates
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

"""Tests for the RE2-backed regex cache."""

import importlib
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import re2

from quarry.cache.regex_cache import RE2_MAX_MEM_BYTES, RegexCache
from quarry.safety.timer import QuarryTimeoutError, evaluation_deadline

REGEX_CACHE_MODULE = importlib.import_module("quarry.cache.regex_cache")

SUPPORTED_PATTERNS = (
    r"\brm\b",
    r"(?:^|[^\w./-])rm(?:$|[^\w.-])",
    r"rm\s+-rf",
    r"^\s*echo\b",
    r"(?:kill|pkill)\s+-(?:9|KILL)\b",
    r"\.env(?:\b|\.)",
    r"sk-[A-Za-z0-9]{16,}",
)

MATCH_INPUTS = (
    "rm -rf /",
    "warm",
    "a.rm",
    "-rm",
    "rm\n",
    "rm",
    "",
    "warm rm",
    "  echo hi",
    "echo hi\n",
    "RM -RF /",
    "prefix\nrm\n",
    "nothing to see",
    "kill -9 123\n",
    ".env",
    "myenv",
    "sk-ABCDEFGHIJKLMNOP",
)

CATASTROPHIC_PATTERN = r"(a+)+$"


def test_lookaround_patterns_are_rejected_at_compile() -> None:
    cache = RegexCache()

    for pattern in (r"(?<=prefix)rm", r"(?<![\w./-])rm(?![\w.-])", r"(?=a)b", r"(?!a)b"):
        with pytest.raises(re.error, match="unsupported by RE2"):
            cache.compile(pattern)


def test_backreference_patterns_are_rejected_at_compile() -> None:
    cache = RegexCache()

    with pytest.raises(re.error, match="unsupported by RE2"):
        cache.compile(r"(\w+)\s+\1")


def test_rejection_message_names_the_construct_re2_refused() -> None:
    with pytest.raises(re.error) as exc_info:
        RegexCache().compile(r"(?<=prefix)rm")

    assert "(?<=" in str(exc_info.value)


def test_invalid_and_unsupported_syntax_raises_re_error() -> None:
    """Expose RE2 syntax failures as ``re.error``."""

    cache = RegexCache()

    for pattern in ("({foo}", "a{3,1}", r"(?V1)a", r"(?>a+)b", r"a++"):
        with pytest.raises(re.error):
            cache.compile(pattern)


def test_unsupported_flags_raise_re_error() -> None:
    """Reject flags that RE2 cannot preserve."""

    cache = RegexCache()

    for flags in (0, re.IGNORECASE, re.DOTALL, re.IGNORECASE | re.DOTALL):
        assert cache.compile(r"a.b", flags) is not None

    for flags in (re.MULTILINE, re.VERBOSE, re.ASCII, re.UNICODE, re.IGNORECASE | re.MULTILINE):
        with pytest.raises(re.error, match="unsupported by RE2"):
            cache.compile(r"a.b", flags)


def test_nocase_is_applied_through_re2_options() -> None:
    cache = RegexCache()
    pattern = r"critical (security|compliance)"

    for text in ("CRITICAL SECURITY", "critical compliance", "Critical Security\n"):
        assert cache.search(pattern, text, re.IGNORECASE) is True

    for text in ("CRITICAL SECURITY", "Critical Security"):
        assert cache.search(pattern, text) is False

    assert cache.search(pattern, "critical security") is True
    assert cache.search(pattern, "CRITICAL SAFETY", re.IGNORECASE) is False


def test_supported_patterns_match_stdlib_on_boundary_inputs() -> None:
    cache = RegexCache()

    for pattern in SUPPORTED_PATTERNS:
        for text in MATCH_INPUTS:
            for flags in (0, re.IGNORECASE):
                expected = bool(re.search(pattern, text, flags))
                assert cache.search(pattern, text, flags) is expected, (pattern, text, flags)


def test_match_and_count_contracts_are_unchanged() -> None:
    cache = RegexCache()
    match = cache.search_match(
        r"critical (security|compliance)", "CRITICAL SECURITY", re.IGNORECASE
    )

    assert match is not None
    assert match.span() == (0, 17)
    assert match.group(0) == "CRITICAL SECURITY"
    # The process cache keeps only spans, so capture groups are not retained.
    with pytest.raises(IndexError):
        match.group(1)
    assert cache.search_match(r"\bmissing\b", "nothing here") is None
    assert cache.count(r"\brm\b", "rm rm rm") == 3
    assert cache.count(r"\brm\b", "nothing here") == 0
    assert cache.search_match(None, "text") is None
    assert cache.count(r"\brm\b", None) == 0


def test_invalid_inputs_fail_at_the_boundary() -> None:
    cache = RegexCache()

    for pattern in (b"bytes", re.compile("compiled")):
        with pytest.raises(TypeError, match="must be strings"):
            cache.compile(pattern)
    for operation in ("search", "count"):
        with pytest.raises(TypeError, match="must be strings"):
            getattr(cache, operation)(b"bytes", "bytes")


def test_catastrophic_backtracking_pattern_uses_re2() -> None:
    """A classic backtracking pattern is compiled on RE2 before matching long input."""

    cache = RegexCache()
    text = "a" * 10_000 + "b"

    assert isinstance(cache.compile(CATASTROPHIC_PATTERN), type(re2.compile("a")))
    assert cache.search(CATASTROPHIC_PATTERN, text) is False
    assert cache.count(CATASTROPHIC_PATTERN, text) == 0


def test_rule_deadline_still_aborts_evaluation() -> None:
    """Honor the rule deadline around RE2 calls."""

    cache = RegexCache()
    results = []

    with pytest.raises(QuarryTimeoutError), evaluation_deadline(0.001, "rule deadline"):
        time.sleep(0.01)
        results.append(cache.search(r"\brm\b", "rm -rf /"))

    assert results == []


def test_count_checks_deadline_while_iterating(monkeypatch) -> None:
    """Large match sets must not defer the cooperative deadline until counting ends."""

    checks = 0

    def expire_during_iteration() -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            raise QuarryTimeoutError("count deadline")

    monkeypatch.setattr(REGEX_CACHE_MODULE, "raise_if_evaluation_expired", expire_during_iteration)

    with pytest.raises(QuarryTimeoutError, match="count deadline"):
        RegexCache().count("a", "a" * 100)

    assert checks == 3


def test_search_is_thread_safe() -> None:
    cache = RegexCache()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: cache.search(r"thread\d+", "thread123"), range(20)))

    assert results == [True] * 20


def test_re2_memory_budget_avoids_false_negatives() -> None:
    """Keep large combined patterns within RE2's memory budget."""

    assert RE2_MAX_MEM_BYTES >= 48 * 1024 * 1024


def test_re2_dollar_is_end_of_text_so_it_does_not_match_before_a_trailing_newline() -> None:
    """RE2 treats ``$`` as the absolute end of text."""

    cache = RegexCache()

    assert re.search(r"foo$", "foo\n") is not None
    assert cache.search(r"foo$", "foo\n") is False
    assert cache.search(r"foo$", "foo") is True
    assert cache.search(r"foo(?:$|[^\w.-])", "foo\n") is True


def test_re2_word_classes_are_ascii_so_boundaries_shift_around_non_ascii() -> None:
    """RE2 word classes use ASCII semantics."""

    cache = RegexCache()

    assert re.search(r"\bcafé\b", "café") is not None
    assert cache.search(r"\bcafé\b", "café") is False
    assert cache.search(r"café", "café") is True
