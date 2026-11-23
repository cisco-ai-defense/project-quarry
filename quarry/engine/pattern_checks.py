# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Bounded names and explicit host capabilities for local pattern predicates."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from types import CoroutineType, MappingProxyType

PatternCheck = Callable[[str], bool]
MAX_PATTERN_CHECKS = 64
MAX_CHECKS_PER_PATTERN = 8


class PatternCheckError(RuntimeError):
    """A trusted host pattern predicate failed or returned an invalid result."""


def check_name(value: object) -> str:
    """Validate a literal capability name, never an import path to execute."""
    if (
        type(value) is not str
        or not 1 <= len(value) <= 64
        or not value.isascii()
        or not value[0].isalpha()
        or not value.replace("_", "").replace("-", "").replace(".", "").isalnum()
    ):
        raise ValueError("pattern check names require 1..64 ASCII identifier characters")
    return value


def check_names(value: object) -> tuple[str, ...]:
    """Validate an ordered, duplicate-free tuple of local predicates."""
    if type(value) is not tuple or len(value) > MAX_CHECKS_PER_PATTERN:
        raise ValueError("a pattern accepts at most eight named checks")
    names = tuple(check_name(name) for name in value)
    if len(set(names)) != len(names):
        raise ValueError("pattern checks cannot repeat a name")
    return names


def declared_check_names(value: object) -> tuple[str, ...]:
    """A rule's complete capability declaration survives compile-time folding."""
    if type(value) is not tuple or len(value) > MAX_PATTERN_CHECKS:
        raise ValueError("a rule declares at most 64 pattern check names")
    names = tuple(check_name(name) for name in value)
    if tuple(sorted(set(names))) != names:
        raise ValueError("declared pattern checks must be unique and sorted")
    return names


def encode_pattern_flags(bits: str, checks: tuple[str, ...]) -> str:
    """Keep unchanged four-bit identities when a pattern has no checks."""
    names = check_names(checks)
    if type(bits) is not str or len(bits) != 4 or set(bits) - {"0", "1"}:
        raise ValueError("invalid compiled string modifier flags")
    return bits + (":" + ",".join(names) if names else "")


def parse_pattern_flags(value: object) -> tuple[str, tuple[str, ...]]:
    """Read bounded prepared flags without importing any host capability."""
    if type(value) is not str or len(value) > 524:
        raise ValueError("invalid compiled string modifier flags")
    bits, separator, suffix = value.partition(":")
    checks = check_names(tuple(suffix.split(","))) if separator else ()
    if encode_pattern_flags(bits, checks) != value:
        raise ValueError("invalid compiled string modifier flags")
    return bits, checks


def copy_pattern_checks(
    values: Mapping[str, PatternCheck] | None,
) -> Mapping[str, PatternCheck]:
    """Copy each backend's explicit registry; there is no global or built-in registry."""
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping) or len(values) > MAX_PATTERN_CHECKS:
        raise ValueError("pattern_checks requires a mapping of at most 64 host predicates")
    copied: dict[str, PatternCheck] = {}
    for index, (name, predicate) in enumerate(values.items()):
        if index >= MAX_PATTERN_CHECKS:
            raise ValueError("pattern_checks exceeds 64 host predicates")
        name = check_name(name)
        if (
            not callable(predicate)
            or inspect.iscoroutinefunction(predicate)
            or inspect.isasyncgenfunction(predicate)
        ):
            raise TypeError("pattern checks must be synchronous host callables")
        copied[name] = predicate
        if len(copied) > MAX_PATTERN_CHECKS:
            raise ValueError("pattern_checks exceeds 64 host predicates")
    return MappingProxyType(copied)


def run_pattern_check(
    predicate: PatternCheck,
    name: str,
    text: str,
    *,
    control_errors: tuple[type[BaseException], ...] = (),
) -> bool:
    """Never put callback exceptions, input text or returned objects in diagnostics."""
    try:
        result: object = predicate(text)
    except control_errors:
        raise
    except (Exception, SystemExit, KeyboardInterrupt):
        raise PatternCheckError(f"Pattern check '{name}' failed") from None
    if type(result) is not bool:
        if type(result) is CoroutineType:
            result.close()
        raise PatternCheckError(f"Pattern check '{name}' must return a bool")
    return result


def required_pattern_checks(artifact: object) -> frozenset[str]:
    """Find names in every compiled atom, capture and nested expression, without source."""
    from quarry.engine.atoms import ModelExpression, StringSemantics
    from quarry.engine.captures import CaptureBind
    from quarry.engine.contracts import CompiledRuleArtifact

    pending = [artifact]
    seen: set[int] = set()
    names: set[str] = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, (StringSemantics, CaptureBind)):
            names.update(check_names(value.checks))
        if isinstance(value, CompiledRuleArtifact):
            names.update(declared_check_names(value.declared_pattern_checks))
        if isinstance(value, ModelExpression) and value.kind == "string_patterns":
            names.update(parse_pattern_flags(value.value)[1])
        if is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, field.name) for field in fields(value))
        elif isinstance(value, Mapping):
            pending.extend(value.values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
    return frozenset(names)
