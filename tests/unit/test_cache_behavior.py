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

"""Tests for cache behavior and limits."""

from quarry.cache.regex_cache import RegexCache


class TestCacheBehavior:
    def test_cache_basic_hit(self):
        cache = RegexCache()

        pattern1 = cache.compile("test.*", 0)
        pattern2 = cache.compile("test.*", 0)

        assert pattern1 is pattern2

    def test_cache_search_returns_false_on_invalid_pattern(self):
        cache = RegexCache()

        result = cache.search(None, "text")
        assert result is False

        result = cache.search("pattern", None)
        assert result is False
