# Copyright 2026 Cisco Systems, Inc. and its affiliates
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

"""Public behavior of the host-neutral prepared-rule cache."""

from __future__ import annotations

import pytest

from quarry import PreparedRuleCache


def test_cache_reuses_by_rule_and_runtime_evicts_lru_and_retries_failures() -> None:
    cache = PreparedRuleCache[object](max_entries=2)
    calls: list[str] = []

    def prepare(rule_text: str) -> object:
        value = object()
        calls.append(rule_text)
        return value

    first = cache.get_or_prepare("rule one", runtime_identity="runtime-a", prepare=prepare)
    assert cache.get_or_prepare("rule one", runtime_identity="runtime-a", prepare=prepare) is first
    other_runtime = cache.get_or_prepare("rule one", runtime_identity="runtime-b", prepare=prepare)
    assert other_runtime is not first
    assert cache.get_or_prepare("rule one", runtime_identity="runtime-a", prepare=prepare) is first
    cache.get_or_prepare("rule two", runtime_identity="runtime-a", prepare=prepare)
    assert (
        cache.get_or_prepare("rule one", runtime_identity="runtime-b", prepare=prepare)
        is not other_runtime
    )
    assert calls == ["rule one", "rule one", "rule two", "rule one"]

    def fail(_rule_text: str) -> object:
        calls.append("failed")
        raise RuntimeError("temporary failure")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="temporary failure"):
            cache.get_or_prepare("rule three", runtime_identity="runtime-a", prepare=fail)
    assert calls[-2:] == ["failed", "failed"]
    cache.get_or_prepare("rule three", runtime_identity="runtime-a", prepare=prepare)
    assert calls[-1] == "rule three"


@pytest.mark.parametrize("max_entries", [0, -1, True, 1.5])
def test_cache_rejects_invalid_bounds(max_entries) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        PreparedRuleCache(max_entries=max_entries)


@pytest.mark.parametrize(
    ("rule_text", "runtime_identity", "message"),
    [
        ("", "runtime", "rule_text"),
        ("rule", "", "runtime_identity"),
    ],
)
def test_cache_rejects_empty_identities(rule_text, runtime_identity, message) -> None:
    with pytest.raises(ValueError, match=message):
        PreparedRuleCache().get_or_prepare(
            rule_text,
            runtime_identity=runtime_identity,
            prepare=lambda _source: object(),
        )
