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

"""Read-only projections of the generated language catalog used by Quarry itself."""

from __future__ import annotations

import difflib
from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from quarry.language import LanguageSpecError, get_language_spec

# These names are synthesized by the transformer and never appear in authored Quarry rules.
INTERNAL_FUNCTION_NAMES = frozenset({"_index", "_slice", "_method_arg", "_method_has_parameter"})


@lru_cache(maxsize=1)
def _catalog() -> Mapping[str, Any]:
    return _freeze(get_language_spec())


def _freeze(value: Any) -> Any:
    """Recursively freeze JSON data before sharing it through cached runtime views."""

    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def clear_language_spec_cache() -> None:
    """Discard the cached catalog so tests or development tools can reload it."""

    _catalog.cache_clear()


def language_items(*, kind: str | None = None) -> tuple[Mapping[str, Any], ...]:
    """Return installed catalog items, optionally restricted to one item kind."""

    items = _catalog()["items"]
    return tuple(item for item in items if kind is None or item["kind"] == kind)


def function_items() -> tuple[Mapping[str, Any], ...]:
    """Return every callable public form, regardless of its Python owner kind."""

    return tuple(item for item in language_items() if item["kind"] in {"function", "compiler_form"})


def function_by_name(name: str) -> Mapping[str, Any] | None:
    """Return the installed callable record for one exact public name."""

    return next((item for item in function_items() if item["name"] == name), None)


def supported_function_names() -> frozenset[str]:
    """Return the exact function names accepted by the linker."""

    return frozenset(item["name"] for item in function_items())


def function_signatures() -> dict[str, tuple[int, int | None]]:
    """Project public arity as ``name -> (minimum, maximum)``."""

    result: dict[str, tuple[int, int | None]] = {}
    for item in function_items():
        parameters = item["parameters"]
        implicit = int(item.get("implicit_receiver_count", 0))
        variadic = next((parameter for parameter in parameters if parameter["variadic"]), None)
        minimum = implicit + sum(
            1 for parameter in parameters if parameter["required"] and not parameter["variadic"]
        )
        if variadic is not None and variadic["required"]:
            minimum += 1
        maximum = None if variadic is not None else implicit + len(parameters)
        result[item["name"]] = (minimum, maximum)
    return result


def surface_names(
    *,
    domain: str | None = None,
    model_input: bool | None = None,
) -> frozenset[str]:
    """Project surface names by their declared domain or model-input capability."""

    return frozenset(
        item["name"]
        for item in language_items(kind="surface")
        if (domain is None or item["domain"] == domain)
        and (model_input is None or item["model_input"] is model_input)
    )


def surface_by_name(name: str) -> Mapping[str, Any] | None:
    """Return the installed record for one exact public surface name."""

    return next(
        (item for item in language_items(kind="surface") if item["name"] == name),
        None,
    )


def role_view_surface_names() -> frozenset[str]:
    """Return model-input message views other than the complete ``messages`` surface."""

    return surface_names(model_input=True) - {"messages"}


def accessor_functions() -> dict[str, tuple[str, str | None]]:
    """Project collection accessors as ``name -> (surface, exact filter field)``."""

    return {
        item["name"]: (item["accessor_surface"], item["accessor_filter"])
        for item in function_items()
        if item.get("accessor_surface") is not None
    }


def functions_with_tag(tag: str) -> frozenset[str]:
    """Return callable names carrying one compiler capability tag."""

    return frozenset(
        item["name"] for item in function_items() if tag in item.get("compiler_tags", ())
    )


def functions_in_resource_group(group: str) -> frozenset[str]:
    """Return callable names requiring one leaf runtime resource group."""

    return frozenset(
        item["name"] for item in function_items() if item.get("resource_group") == group
    )


def inference_functions() -> frozenset[str]:
    """Return functions that require score, completion, or token inference support."""

    return frozenset().union(
        functions_in_resource_group("score"),
        functions_in_resource_group("completion"),
        functions_in_resource_group("token"),
    )


def external_function_costs() -> dict[str, str]:
    """Project every resource-backed callable to its compiler cost tier."""

    llm_groups = {"score", "completion", "token"}
    return {
        item["name"]: "llm" if item["resource_group"] in llm_groups else "model"
        for item in function_items()
        if item.get("resource_group") is not None
    }


def reserved_variable_names() -> frozenset[str]:
    """Return names that may not be shadowed in a ``variables:`` block."""

    return frozenset(_catalog()["reserved_variable_names"])


def rejected_spellings() -> tuple[Mapping[str, Any], ...]:
    """Return every documented spelling Quarry rejects, with its replacement when one exists."""

    return _catalog()["rejected_spellings"]


def rejected_call_guidance(name: str) -> tuple[Mapping[str, Any], ...]:
    """Return the documented rejected spellings whose unknown call name is ``name``."""

    return tuple(record for record in rejected_spellings() if record["rejected_call"] == name)


def token_spellings() -> frozenset[str]:
    """Return every reserved keyword or operator spelling the grammar recognises."""

    return frozenset(item["spelling"] for item in language_items(kind="token"))


def suggest_function_names(name: str, *, limit: int = 3) -> tuple[str, ...]:
    """Return installed callables and surfaces spelled close to ``name``, nearest first."""

    candidates = sorted(supported_function_names() | surface_names())
    return tuple(difflib.get_close_matches(name, candidates, n=limit, cutoff=0.7))


__all__ = [
    "INTERNAL_FUNCTION_NAMES",
    "LanguageSpecError",
    "accessor_functions",
    "clear_language_spec_cache",
    "external_function_costs",
    "function_by_name",
    "function_items",
    "function_signatures",
    "functions_in_resource_group",
    "functions_with_tag",
    "inference_functions",
    "language_items",
    "rejected_call_guidance",
    "rejected_spellings",
    "reserved_variable_names",
    "role_view_surface_names",
    "suggest_function_names",
    "surface_by_name",
    "supported_function_names",
    "surface_names",
    "token_spellings",
]
