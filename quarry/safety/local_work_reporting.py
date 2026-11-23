# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Bounded aggregation of native scalar local-work reports, never scan inputs."""

from collections.abc import Iterable
from typing import Any

_MAX = (1 << 63) - 1
_COUNTS = (
    "known_cpu_ns",
    "known_overshoot_cpu_ns",
    "segments",
    "denied_segments",
    "active_segments",
    "unmetered_segments",
    "conversations",
    "exhausted_conversations",
)


def aggregate_local_match_work(
    reports: Iterable[dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Fold compatible per-conversation groups in constant space (at most 64 rows).

    Allowances remain per conversation. CPU/overshoot and operation counters sum
    actual observed work; exhaustion is any child's exhaustion, never a comparison
    between aggregate CPU and one conversation's allowance. Unknowns stay unknown.
    """
    result: dict[str, Any] | None = None
    for report in reports:
        if report is None:
            continue
        if result is None:
            result = {**report, "scope": "batch", "groups": [dict(row) for row in report["groups"]]}
            if len(result["groups"]) > 64:
                result.update(state="owner_unavailable", complete=False, groups=[])
            continue
        total = result["conversations"] + report["conversations"]
        result["conversations"] = min(_MAX, total)
        result["complete"] = result["complete"] and report["complete"] and total <= _MAX
        rows, others = result["groups"], report["groups"]
        if (
            result["state"] != "collected"
            or report["state"] != "collected"
            or len(rows) != len(others)
            or any(
                (a["label"], a["mode"], a["allowance_cpu_ns"])
                != (b["label"], b["mode"], b["allowance_cpu_ns"])
                for a, b in zip(rows, others, strict=True)
            )
        ):
            result.update(state="owner_unavailable", complete=False, groups=[])
            continue
        for row, other in zip(rows, others, strict=True):
            saturated = row["counter_saturated"] or other["counter_saturated"] or total > _MAX
            for name in _COUNTS:
                value = row[name] + other[name]
                saturated |= value > _MAX
                row[name] = min(_MAX, value)
            for name in ("value_cpu_ns", "overshoot_cpu_ns"):
                a, b = row[name], other[name]
                value = None if a is None or b is None else a + b
                saturated |= value is not None and value > _MAX
                row[name] = value if value is not None and value <= _MAX else None
            row["counter_saturated"] = saturated
            row["complete"] = row["complete"] and other["complete"] and not saturated
            row["closed"] = row["closed"] and other["closed"]
            row["exhausted"] = row["exhausted"] or other["exhausted"]
            row["cut_phase"] = row["cut_phase"] or other["cut_phase"]
            if other["state"] == "unmetered":
                row["state"] = "unmetered"
            if saturated:
                row["value_cpu_ns"] = row["overshoot_cpu_ns"] = None
            result["complete"] = result["complete"] and row["complete"]
    return result
