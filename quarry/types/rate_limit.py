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

"""Shared rate limit configuration types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitConfig:
    """Throughput and concurrency limits for one model API.

    Frozen: a rate-limit definition is a configuration value shared by reference across
    parallel requests. The mutable counters that enforce it live in ``LLMRateLimiter``.
    """

    requests_per_interval: int | None = None
    interval_seconds: float = 60.0
    max_concurrent: int | None = None
