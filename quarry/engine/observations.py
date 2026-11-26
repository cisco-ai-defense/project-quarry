# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Optional read observation contracts without model, scanner or input dependencies."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol


class GroundingReadReplay(Protocol):
    """Opaque metadata for attributing one previously computed value to its reader."""

    def replay(self) -> None: ...


class GroundingReadCapture(Protocol):
    """Complete a bounded read capture without retaining the grounded value or inputs."""

    def result(self) -> GroundingReadReplay | None: ...


class GroundingReadObserver(Protocol):
    """A request adapter whose hooks perform no detector work and do not raise.

    Only external-predicate computations are observed. A missing replay means unavailable
    metadata, not a validated empty footprint. Evidence lookups do not request new reads.
    Implementations own independent metadata limits and must report omissions explicitly.
    """

    def capture(self) -> AbstractContextManager[GroundingReadCapture | None]: ...

    def replay(self, reads: GroundingReadReplay | None) -> None: ...
