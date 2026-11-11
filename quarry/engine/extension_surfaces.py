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

"""Scanner-local extension surface names used while compiling a rule generation."""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from types import MappingProxyType

EVENT_SURFACES: ContextVar[frozenset[str]] = ContextVar(
    "quarry_extension_event_surfaces", default=frozenset()
)
STATIC_SURFACES: ContextVar[frozenset[str]] = ContextVar(
    "quarry_extension_static_surfaces", default=frozenset()
)
INPUT_TEXTS: ContextVar[Mapping[str, tuple[str, ...]]] = ContextVar(
    "quarry_input_text_fields", default=MappingProxyType({})
)
