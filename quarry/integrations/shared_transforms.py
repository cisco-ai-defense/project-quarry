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

"""Versioned relative transform records, validated again at every shared-cache read."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Coroutine, Iterable
from dataclasses import dataclass
from typing import Any, TypeVar

from quarry.cache.keys import input_key
from quarry.integrations.adapter_transforms import KeptTransform, keep_transform, read_transform
from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.adapter_transforms import TransformCacheIdentity, TransformResult, TransformRun
from quarry.types.exceptions import ConfigurationError, ModelError
from quarry.types.transforms import (
    MAX_TRANSFORM_EXPANSION_RATIO,
    MAX_TRANSFORM_INSERTED_CHARACTERS,
    MAX_TRANSFORM_MAPPING_SEGMENTS,
    MAX_TRANSFORM_OUTPUT_CHARACTERS,
)

T = TypeVar("T")

TRANSFORM_BATCH_SIZE = 16


async def gather_transform_batch(awaitables: Iterable[Coroutine[Any, Any, T]]) -> list[T]:
    """Join every owned child before returning or propagating the first failure.

    Shield the group so cancellation reaches each child once. Later cancellation requests
    must not interrupt a child's async cleanup or let its caller release live value owners.
    """

    tasks: list[asyncio.Task[T]] = []
    pending: list[asyncio.Task[T]] = []
    group = None
    coroutine = task = None
    try:
        for coroutine in awaitables:
            try:
                tasks.append(asyncio.create_task(coroutine))
            except BaseException:
                coroutine.close()
                raise
        group = asyncio.gather(*tasks)
        return await asyncio.shield(group)
    finally:
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        while any(not task.done() for task in pending):
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait(pending)
        for task in tasks:
            if not task.cancelled():
                task.exception()
        if group is not None and group.done() and not group.cancelled():
            group.exception()
        # A propagated exception retains this frame. Drop the tasks/futures whose outcomes
        # may contain that same exception, along with their copied request contexts.
        del tasks, pending, group, coroutine, task, awaitables


def transform_input_size(unit: dict) -> int:
    """Payload characters in one already validated wire snapshot."""
    return sum(
        len(value) if type(value) is str else sum(map(len, value.values()))
        for field, value in unit.items()
        if field != "type"
    )


def registered_cache_identity(adapter: object, name: str) -> TransformCacheIdentity | None:
    value = getattr(adapter, "transform_cache", None)
    if value is not None and type(value) is not TransformCacheIdentity:
        raise ConfigurationError(
            f"Transform {name!r} transform_cache must be TransformCacheIdentity or None"
        )
    return value


class InvalidSharedTransform(ValueError):
    """A cache record is malformed, corrupt or belongs to a different request."""


@dataclass(frozen=True, slots=True)
class SharedTransformRecord:
    format_version: int
    key: bytes
    kept: KeptTransform
    checksum: bytes


def shared_record(key: bytes, kept: KeptTransform) -> SharedTransformRecord:
    return SharedTransformRecord(1, key, kept, input_key("shared-transform-record-v1", key, kept))


def read_shared_transform(record: object, key: bytes, unit: dict, name: str) -> TransformResult:
    """Bound and validate before slicing, hashing or reconstructing a cached record.

    The checksum detects accidental record corruption; it is not authentication for an
    untrusted storage service. This protocol currently uses the engine's process cache.
    """

    def invalid() -> None:
        raise InvalidSharedTransform("invalid shared transform record")

    if type(record) is not SharedTransformRecord:
        invalid()
    assert isinstance(record, SharedTransformRecord)
    if (
        type(record.format_version) is not int
        or record.format_version != 1
        or type(record.key) is not bytes
        or record.key != key
        or type(record.checksum) is not bytes
        or len(record.checksum) != 32
        or type(record.kept) is not KeptTransform
    ):
        invalid()
    kept = record.kept
    if (
        type(kept.inserted) is not str
        or len(kept.inserted) > MAX_TRANSFORM_OUTPUT_CHARACTERS
        or type(kept.runs) is not tuple
        or type(kept.copied) is not tuple
        or len(kept.runs) > MAX_TRANSFORM_MAPPING_SEGMENTS
        or len(kept.runs) != len(kept.copied)
    ):
        invalid()
    source = unit.get("text") if unit["type"] == "text" else None
    cursor = inserted = 0
    for index, (run, copied) in enumerate(zip(kept.runs, kept.copied, strict=True)):
        if index % 256 == 0:
            raise_if_evaluation_expired()
        if type(run) is not TransformRun or type(copied) is not bool:
            invalid()
        if (
            type(run.start) is not int
            or type(run.end) is not int
            or run.start != cursor
            or not cursor < run.end <= MAX_TRANSFORM_OUTPUT_CHARACTERS
            or type(run.kind) is not str
            or run.kind not in {"linear", "whole"}
        ):
            invalid()
        if run.source_start is None and run.source_end is None:
            if run.kind != "whole" or copied:
                invalid()
        elif (
            type(run.source_start) is not int
            or type(run.source_end) is not int
            or source is None
            or not 0 <= run.source_start < run.source_end <= len(source)
            or (run.kind == "linear" and run.end - run.start != run.source_end - run.source_start)
        ):
            invalid()
        if copied:
            if run.kind != "linear":
                invalid()
        else:
            inserted += run.end - run.start
            if inserted > len(kept.inserted):
                invalid()
        cursor = run.end
    if inserted != len(kept.inserted) or (
        source is not None
        and cursor > len(source) * MAX_TRANSFORM_EXPANSION_RATIO + MAX_TRANSFORM_INSERTED_CHARACTERS
    ):
        invalid()
    if input_key("shared-transform-record-v1", key, kept) != record.checksum:
        invalid()
    result = read_transform(kept, unit)
    # Reuse the adapter contract's Unicode, map coverage, expansion and copy checks.
    try:
        canonical = keep_transform(result, unit, name)
    except ModelError as exc:
        raise InvalidSharedTransform("invalid shared transform record") from exc
    if canonical != kept:
        invalid()
    return result
