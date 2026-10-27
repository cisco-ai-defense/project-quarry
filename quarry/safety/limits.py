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

"""Safety limits for rule evaluation."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class SafetyLimits:
    """Configuration for safety limits.

    Frozen: safety limits are a configuration value shared by reference across parallel
    requests. Derive a variant with ``dataclasses.replace`` rather than mutating in place.
    """

    # Version 1 defaults. QuarryConfig uses a 60-second rule timeout and no separate
    # inference timeout for version 2.
    rule_timeout: float | None = 1.0
    # Version 1 gives rules that call external services a larger budget. Version 2 uses
    # rule_timeout for the entire rule and leaves this unset.
    inference_timeout: float | None = 30.0
    # Evidence work after a request's verdicts are final: minimizing a match's support and
    # exploring the values the verdicts did not need. It never delays or changes a verdict,
    # and it applies even with rule_timeout unset, so a long batch cannot stall on it.
    explore_timeout: float = 5.0

    max_messages: int = 10_000
    # 10MB. Sized so ONE message can hold a 1M-token context window alongside inline media:
    # 1M tokens of text is ~3-5MB serialized (worst case ~1 token/char for CJK, code, and
    # base64-shaped text), and one image can be up to 5MB before the tightest provider
    # ceiling rejects it. The former 1MB clipped a 1M-token document to a quarter of itself
    # and rejected ordinary inline images outright. ``max_total_size`` remains the ceiling on
    # the conversation as a whole.
    max_message_size: int = 10_000_000
    max_tool_argument_size: int = 1_000_000
    max_tool_calls: int = 1_000
    # Maximum JSON nesting depth for a tool-call argument object. Bounds both the
    # scanner's post-size-preflight validation and the lazy rule-local decoder
    # (``ToolCall.arguments``).
    max_nesting_depth: int = 100
    max_total_size: int = 50_000_000  # 50MB
    max_timeline_items: int = 10_000
    max_comprehension_size: int = 10_000
    # Provider attempts allowed for each configured Quarry model alias in one scan. This is
    # the cost ceiling that concurrency and throughput limits do not provide.
    max_model_calls: int = 128

    enforce_limits: bool = True
    # False returns an incomplete low-level result; public guards always raise.
    fail_on_timeout: bool = False
    # There is deliberately no ``fail_on_limit``. Breaching any input limit above ALWAYS fails the
    # scan: ``Conversation``/``Message`` are the canonical record of the original OpenAI input, so
    # an over-limit conversation is reported, never reshaped to fit. Reshaping would hand rules
    # a conversation that differs from what the caller sent; raise the specific limit instead.

    def __post_init__(self) -> None:
        """Reject timeout values that would silently disable deadline enforcement.

        A NaN/inf/non-positive timeout makes ``min(per_rule, remaining)`` and
        ``asyncio.timeout`` never fire (or fire immediately), so a pathological rule could run
        unbounded — fail loudly at config time."""
        for field_name in ("rule_timeout", "inference_timeout", "explore_timeout"):
            if getattr(self, field_name) is None and field_name != "explore_timeout":
                continue
            try:
                value = float(getattr(self, field_name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field_name} must be a positive finite number") from exc
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be a positive finite number")
            object.__setattr__(self, field_name, value)
        if (
            isinstance(self.max_nesting_depth, bool)
            or not isinstance(self.max_nesting_depth, int)
            or not 1 <= self.max_nesting_depth <= 100
        ):
            raise ValueError("max_nesting_depth must be an integer from 1 to 100")
        if (
            isinstance(self.max_model_calls, bool)
            or not isinstance(self.max_model_calls, int)
            or self.max_model_calls < 1
        ):
            raise ValueError("max_model_calls must be a positive integer")
        if (
            isinstance(self.max_comprehension_size, bool)
            or not isinstance(self.max_comprehension_size, int)
            or self.max_comprehension_size < 1
        ):
            raise ValueError("max_comprehension_size must be a positive integer")
