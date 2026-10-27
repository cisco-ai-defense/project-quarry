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

"""References to configuration-backed text resources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from quarry.language.metadata import DSLParameter, dsl_compiler_form

# Rules name a text resource by calling this function. Discovery and validation key on the function
# name, so a new function that consumes policy text needs no per-function argument position anywhere.
TEXT_REF_FUNCTION = "text_ref"


@dataclass(frozen=True)
@dsl_compiler_form(
    name=TEXT_REF_FUNCTION,
    return_type="text_reference",
)
class TextRef:
    """Refer to configured policy text by its exact resource name.

    The DSL accepts this only where an inference function expects policy text. A reference remains
    a distinct type until the resource manager resolves it, preventing an unresolved name from
    being mistaken for the policy itself.
    """

    name: Annotated[str, DSLParameter(literal_only=True)]
