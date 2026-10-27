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

"""String pattern modifiers for YARA-like pattern matching.

Match semantics live in the shared engine matchers (``quarry.engine.matchers``); this module
holds only the parsed ``StringModifiers`` the DSL attaches to a pattern.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StringModifiers:
    """Modifiers for string patterns (nocase, fullword, base64, regex)."""

    nocase: bool = False
    fullword: bool = False
    base64: bool = False
    regex: bool = False
