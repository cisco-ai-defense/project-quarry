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

"""Caching utilities."""

from quarry.cache.async_cache import AsyncFunctionCache
from quarry.cache.constants import (
    MAX_CONTAINS_ANY_SIZE,
    MAX_STRING_ARRAY_SIZE,
    REGEX_CACHE_SIZE,
)
from quarry.cache.regex_cache import RegexCache, regex_cache

__all__ = [
    "AsyncFunctionCache",
    "RegexCache",
    "regex_cache",
    "REGEX_CACHE_SIZE",
    "MAX_CONTAINS_ANY_SIZE",
    "MAX_STRING_ARRAY_SIZE",
]
