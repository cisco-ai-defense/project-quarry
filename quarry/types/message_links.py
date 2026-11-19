# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Resolve causal links without rejecting structurally valid sender data."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LinkEvent:
    kind: str
    value: Any
    identifier: str | None
    readable: bool = True


@dataclass(frozen=True)
class LinkResolution:
    call_faults: dict[int, str]
    response_faults: dict[int, str]
    result_faults: dict[int, str]
    responses: dict[int, Any]
    calls: dict[int, Any]


def resolve_links(events: Sequence[LinkEvent], checkpoint: Callable[[], None]) -> LinkResolution:
    """Match replies to waiting calls; preserve uncertainty as bounded diagnostics.

    Ambiguous replies consume one pending position without inventing an exact target.
    A later, fully resolved reuse can therefore begin a new unambiguous exchange.
    """

    pending: dict[str, deque] = defaultdict(deque)
    call_faults: dict[int, str] = {}
    result_faults: dict[int, str] = {}
    responses: dict[int, Any] = {}
    calls: dict[int, Any] = {}
    call_positions = []
    known_ids: set[str] = set()
    for position, event in enumerate(events):
        if position % 256 == 0:
            checkpoint()
        if event.kind == "call" and event.identifier:
            known_ids.add(event.identifier)
    unassigned: dict[str, int] = {}
    unknown_after = -1
    for position, event in enumerate(events):
        if position % 256 == 0:
            checkpoint()
        identity, identifier = id(event.value), event.identifier
        if event.kind == "call":
            call_positions.append((position, event))
            if not identifier:
                call_faults[identity] = "tool call has no usable identifier"
                continue
            waiting = pending[identifier]
            if waiting:
                call_faults[id(waiting[0])] = "tool calls overlap with the same identifier"
                call_faults[identity] = "tool calls overlap with the same identifier"
            waiting.append(event.value)
            continue
        waiting = pending.get(identifier) if identifier else None
        target = None
        if waiting:
            ambiguous = len(waiting) > 1
            target = waiting.popleft()
            ambiguous = ambiguous or id(target) in call_faults
            if ambiguous:
                result_faults[identity] = "tool result has no exact preceding call"
        else:
            result_faults[identity] = "tool result has no exact preceding call"
        if not event.readable:
            result_faults[identity] = "unreadable result cannot establish a call link"
        if target is not None and identity not in result_faults:
            calls[identity] = target
            responses[id(target)] = event.value
        else:
            if identifier and identifier in known_ids:
                unassigned[identifier] = position
            else:
                unknown_after = position
    response_faults = {}
    for index, (position, event) in enumerate(call_positions):
        if index % 256 == 0:
            checkpoint()
        identity = id(event.value)
        if identity in call_faults:
            response_faults[identity] = call_faults[identity]
        elif max(unknown_after, unassigned.get(event.identifier, -1)) > position:
            response_faults[identity] = "an unassigned result may answer this call"
    return LinkResolution(call_faults, response_faults, result_faults, responses, calls)
