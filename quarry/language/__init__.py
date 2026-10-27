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

"""Installed, version-matched Quarry language introspection."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any


class LanguageSpecError(RuntimeError):
    """Base error for an absent or invalid installed language catalog."""


class LanguageSpecMissingError(LanguageSpecError):
    """Raised when development code runs before the generated catalog exists."""


def get_language_spec() -> dict[str, Any]:
    """Return a fresh copy of the packaged language catalog and example index."""

    try:
        catalog = resources.files(__name__).joinpath("spec.json").read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise LanguageSpecMissingError(
            "Quarry's generated language catalog is missing. "
            "Run `python -m scripts.generate_language_spec`."
        ) from exc
    try:
        value = json.loads(catalog)
    except json.JSONDecodeError as exc:
        raise LanguageSpecError(
            f"Quarry's generated language catalog is invalid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise LanguageSpecError("Quarry's generated language catalog has an invalid root object")
    return value


__all__ = ["LanguageSpecError", "LanguageSpecMissingError", "get_language_spec"]
