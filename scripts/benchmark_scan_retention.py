# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Synthetic result-lifetime/work measurements; no provider or optimization acceptance claim."""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import platform
import statistics
import time
import tracemalloc
import weakref
from dataclasses import dataclass

from quarry.cache.async_cache import AsyncFunctionCache, scan_scope
from quarry.config import ModelCacheConfig

CATEGORIES = ("model_responses", "token_counts", "local_transforms", "probes")


@dataclass
class Reply:
    category: str
    value: bytes


async def drain():
    for _ in range(4):
        await asyncio.sleep(0)


async def sample(mode, overlap, args):
    results = AsyncFunctionCache.expiring(max_entries=100_000, ttl_seconds=60, max_bytes=64 << 20)
    policy = (
        None
        if mode == "process"
        else ModelCacheConfig(
            retention="scan", reusable_categories=["token_counts"] if mode == "scan-token" else []
        )
    )
    calls = dict.fromkeys(CATEGORIES, 0)
    references = []
    signatures = []
    ready = 0
    all_ready = asyncio.Event()

    async def provider(category, index):
        calls[category] += 1
        await asyncio.sleep(0)
        seed = hashlib.sha256(f"synthetic {category} {index}".encode()).digest()
        size = 8 if category == "token_counts" else args.payload_bytes
        reply = Reply(category, (seed * ((size + 31) // 32))[:size])
        references.append(weakref.ref(reply))
        return reply

    async def scan():
        nonlocal ready
        scope_args = {} if policy is None else {"cache_policy": policy}
        with scan_scope(**scope_args):
            digest = hashlib.sha256()
            for category in CATEGORIES:
                for index in range(args.unique):
                    key = f"{category}:{index}".encode()
                    options = (
                        {} if policy is None else {"retention": policy.retention_for(category)}
                    )
                    # Each request is read twice within a scan.
                    for _ in range(2):
                        value = await results.call(key, provider, category, index, **options)
                        digest.update(value.value)
                    del value
            signatures.append(digest.hexdigest())
            if overlap:
                ready += 1
                if ready == args.scans:
                    all_ready.set()
                await all_ready.wait()

    started = time.perf_counter()
    if overlap:
        await asyncio.gather(*(scan() for _ in range(args.scans)))
    else:
        for _ in range(args.scans):
            await scan()
    elapsed = time.perf_counter() - started
    await drain()
    idle = {
        "entries": len(results.store),
        "estimated_result_bytes": results.store._byte_total,
        "live_provider_results": sum(ref() is not None for ref in references),
        "ownership_entries": len(getattr(results, "_scan_entries", {})),
        "scan_owners": len(getattr(results, "_scan_owners", {})),
        "traced_current_bytes": tracemalloc.get_traced_memory()[0],
    }
    for category in CATEGORIES:
        reusable = mode == "process" or mode == "scan-token" and category == "token_counts"
        assert calls[category] == args.unique * (1 if overlap or reusable else args.scans)
    retained = len(CATEGORIES) if mode == "process" else 1 if mode == "scan-token" else 0
    assert idle["entries"] == idle["live_provider_results"] == retained * args.unique
    assert idle["ownership_entries"] == idle["scan_owners"] == 0
    assert len(set(signatures)) == 1
    results.clear()
    await drain()
    assert not results.store and all(ref() is None for ref in references)
    return {
        "mode": mode,
        "overlap": overlap,
        "seconds": elapsed,
        "calls_by_category": calls,
        "output_digest": signatures[0],
        "idle_after_all_scans_close": idle,
        "after_clear_live_provider_results": sum(ref() is not None for ref in references),
        "traced_peak_bytes": tracemalloc.get_traced_memory()[1],
    }


async def hits(mode, args):
    results = AsyncFunctionCache.expiring(max_entries=8, ttl_seconds=60)
    calls = 0

    async def provider():
        nonlocal calls
        calls += 1
        return 1

    options = {} if mode == "process" else {"retention": "scan"}
    with scan_scope():
        await results.call(b"k", provider, **options)
        started = time.perf_counter()
        for _ in range(args.hit_calls):
            assert await results.call(b"k", provider, **options) == 1
        elapsed = time.perf_counter() - started
    assert calls == 1
    results.clear()
    return {
        "mode": mode,
        "calls": args.hit_calls,
        "seconds": elapsed,
        "ns_per_hit": elapsed * 1e9 / args.hit_calls,
    }


async def main(args):
    rows, hit_rows = [], []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(args.samples):
            for mode in args.modes:
                for overlap in [False, True]:
                    tracemalloc.start()
                    try:
                        rows.append(await sample(mode, overlap, args))
                    finally:
                        tracemalloc.stop()
                hit_rows.append(await hits(mode, args))
    finally:
        if enabled:
            gc.enable()
    assert len({row["output_digest"] for row in rows}) == 1
    return {
        "python": platform.python_version(),
        "parameters": vars(args),
        "gc_disabled_during_work": True,
        "scope": "Generic deterministic result workloads with actual provider-result weak references and cache byte/entry counters. No network, real provider latency, RSS bound, or independent optimization acceptance.",
        "tracemalloc_scope": "Includes workload tasks, weak-reference observations, result values, cache metadata and allocator overhead visible to tracemalloc; separate from the cache admission byte estimate.",
        "workloads": rows,
        "cached_hits": hit_rows,
        "median_ns_per_hit": {
            mode: statistics.median(row["ns_per_hit"] for row in hit_rows if row["mode"] == mode)
            for mode in args.modes
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans", type=int, default=20)
    parser.add_argument("--unique", type=int, default=8)
    parser.add_argument("--payload-bytes", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--hit-calls", type=int, default=10_000)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["process", "scan", "scan-token"],
        default=["process", "scan", "scan-token"],
    )
    args = parser.parse_args()
    if not (
        1 <= args.scans <= 100
        and 1 <= args.unique <= 100
        and 1 <= args.payload_bytes <= 1 << 20
        and 1 <= args.samples <= 20
        and 1 <= args.hit_calls <= 1_000_000
    ):
        parser.error("workload parameters exceed their documented synthetic bounds")
    print(json.dumps(asyncio.run(main(args)), indent=2))
