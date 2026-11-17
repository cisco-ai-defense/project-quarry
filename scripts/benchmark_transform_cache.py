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

"""Reproducible synthetic transform-cache workloads with fresh input coordinates."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import statistics
import sys
import time

from quarry.cache.async_cache import result_cache
from quarry.cache.keys import input_key
from quarry.config import LocalTransformConfig, QuarryConfig
from quarry.integrations.adapter_message_transforms import apply_adapter_transform
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.types.adapter_transforms import TransformCacheIdentity
from quarry.types.messages import Messages
from quarry.types.results import Span


class WorkloadAdapter:
    local = True
    transform_parts = ("text",)

    def __init__(self, shared: bool, mode: str) -> None:
        self.transform_cache = TransformCacheIdentity("benchmark." + mode, "1") if shared else None
        self.calls = 0

    def transform(self, config, part, *, timeout=None):
        self.calls += 1
        mode = config.settings["mode"]
        if mode == "uppercase":
            return part["text"].upper()
        if mode == "json_projection":
            return str(sum(row["number"] for row in json.loads(part["text"])["rows"]))
        value = part["text"].encode()
        for _ in range(config.settings["rounds"]):
            value = hashlib.sha256(value).digest()
        return value.hex()


async def sample(mode: str, shared: bool, args) -> dict:
    result_cache.clear()
    config = QuarryConfig(
        config_version=2,
        transforms={"work": LocalTransformConfig("owned", {"mode": mode, "rounds": args.rounds})},
    )
    calls = 0
    signatures = []
    started = time.perf_counter()
    for instance in range(args.instances):
        adapter = WorkloadAdapter(shared, mode)
        backend = ModelInferenceBackend(config, adapters={"owned": adapter})
        values = []
        for index in range(args.parts):
            ordinal = index % args.unique
            value = (
                json.dumps(
                    {"rows": [{"number": i + ordinal, "label": "entry"} for i in range(128)]}
                )
                if mode == "json_projection"
                else f"owned input {ordinal}"
            )
            values.append({"type": "text", "text": value})
        wire = [{"role": "system", "content": "prefix"}] * instance + [
            {"role": "user", "content": values}
        ]
        source = Messages.from_openai(wire)
        result = await apply_adapter_transform(source[-1:], backend, "work", _message_value=True)
        span = result._bind_span(Span(0, 0, len(result[0].content)))
        signatures.append(
            input_key(
                result.to_openai_messages(),
                span.to_dict(),
                [item.to_dict() for item in span.source_ranges],
                [item.to_dict() for item in span.source_parts],
            ).hex()
        )
        assert source.to_openai_messages() == wire
        assert all(item.message_index == instance for item in span.source_ranges)
        assert all(item.message_index == instance for item in span.source_parts)
        calls += adapter.calls
    elapsed = time.perf_counter() - started
    expected = min(args.parts, args.unique) if shared else args.parts * args.instances
    assert calls == expected, (mode, shared, calls, expected)
    return {
        "seconds": elapsed,
        "adapter_calls": calls,
        "output_and_current_origin_digests": signatures,
        "retained_result_bytes": result_cache.store._byte_total,
    }


async def main(args) -> None:
    workloads = []
    for mode in ["uppercase", "json_projection", "cpu_digest"]:
        measurements = {"unversioned": [], "versioned_shared": []}
        for iteration in range(args.samples):
            # Alternate execution order while giving every sample a cold process table.
            for shared in [False, True] if iteration % 2 == 0 else [True, False]:
                row = await sample(mode, shared, args)
                measurements["versioned_shared" if shared else "unversioned"].append(row)
        for left, right in zip(*measurements.values(), strict=True):
            assert (
                left["output_and_current_origin_digests"]
                == right["output_and_current_origin_digests"]
            )
        medians = {
            name: statistics.median(row["seconds"] for row in rows)
            for name, rows in measurements.items()
        }
        workloads.append(
            {
                "workload": mode,
                "samples": measurements,
                "median_seconds": medians,
                "unversioned_over_shared_ratio": medians["unversioned"]
                / medians["versioned_shared"],
                "output_and_current_origins_equal": True,
            }
        )
    print(
        json.dumps(
            {
                "scope": "Synthetic process-cache workloads; includes construction, transform, validation and provenance. cpu_digest deliberately models expensive deterministic CPU work. No general latency, RSS or provider claim.",
                "python": sys.version,
                "platform": platform.platform(),
                "parameters": vars(args),
                "workloads": workloads,
            },
            indent=2,
        )
    )


def bounded(minimum, maximum):
    def parse(value):
        result = int(value)
        if not minimum <= result <= maximum:
            raise argparse.ArgumentTypeError(f"must be between {minimum} and {maximum}")
        return result

    return parse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=bounded(1, 10), default=3)
    parser.add_argument("--parts", type=bounded(1, 64), default=32)
    parser.add_argument("--instances", type=bounded(1, 16), default=4)
    parser.add_argument("--unique", type=bounded(1, 64), default=8)
    parser.add_argument("--rounds", type=bounded(1, 20_000), default=2_000)
    asyncio.run(main(parser.parse_args()))
