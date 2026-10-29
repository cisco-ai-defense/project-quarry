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

"""Shared execution policy for direct model API calls."""

from __future__ import annotations

import asyncio
from asyncio import sleep as _sleep
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic
from typing import TypeVar

from quarry.integrations.rate_limiter import LLMRateLimiter
from quarry.safety.timer import (
    QuarryTimeoutError,
    clamp_to_evaluation_deadline,
    current_evaluation_deadline,
)
from quarry.types.exceptions import LLMError

T = TypeVar("T")

RETRYABLE_ERROR_CODES = frozenset(
    {
        "connection",
        "provider_unavailable",
        "rate_limit",
        "timeout",
    }
)


class ModelCallBudget:
    """Limit paid provider attempts per Quarry model alias during one scan."""

    def __init__(self, max_calls_per_alias: int):
        if (
            isinstance(max_calls_per_alias, bool)
            or not isinstance(max_calls_per_alias, int)
            or max_calls_per_alias < 1
        ):
            raise ValueError("max_calls_per_alias must be a positive integer")
        self.max_calls_per_alias = max_calls_per_alias
        self.calls_by_alias: dict[str, int] = {}

    def check(self, alias: str) -> None:
        """Fail when ``alias`` has no attempt left, without reserving one."""

        if self.calls_by_alias.get(alias, 0) >= self.max_calls_per_alias:
            raise LLMError(
                f"Model call cost limit reached for alias '{alias}' "
                f"(max_model_calls={self.max_calls_per_alias})",
                {
                    "code": "model_call_limit",
                    "model": alias,
                    "max_model_calls": self.max_calls_per_alias,
                },
            )

    def consume(self, alias: str) -> None:
        """Reserve one provider attempt or fail before the adapter is called."""

        self.check(alias)
        self.calls_by_alias[alias] = self.calls_by_alias.get(alias, 0) + 1


_current_model_call_budget = ContextVar[ModelCallBudget | None](
    "quarry_model_call_budget", default=None
)


@contextmanager
def model_call_budget(
    max_calls_per_alias: int | None,
) -> Generator[ModelCallBudget | None, None, None]:
    """Install one task-local cost budget inherited by every rule in a scan."""

    budget = ModelCallBudget(max_calls_per_alias) if max_calls_per_alias is not None else None
    token = _current_model_call_budget.set(budget)
    try:
        yield budget
    finally:
        _current_model_call_budget.reset(token)


class ModelUsage:
    """Rate-limit and retry one logical model call within its time budget."""

    def __init__(self, rate_limiter: LLMRateLimiter | None = None):
        self.rate_limiter = rate_limiter

    async def run(
        self,
        model: str,
        operation: Callable[[float | None], Awaitable[T]],
        *,
        budget_alias: str | None = None,
        timeout: float | None = None,
    ) -> T:
        """Run sequential attempts until success, a final error, or timeout.

        The caller timeout and active rule deadline form one logical-call
        budget. Limiter queueing, provider attempts, and retry backoff all
        consume that budget. Without either limit, the call runs once.
        """

        effective_timeout = clamp_to_evaluation_deadline(timeout)
        if effective_timeout is None:
            return await self._attempt(
                model,
                operation,
                deadline=None,
                budget_alias=budget_alias,
            )

        active_evaluation = current_evaluation_deadline()
        deadline = monotonic() + effective_timeout
        evaluation_timeout_message = None
        if active_evaluation is not None and active_evaluation.expires_at <= deadline:
            deadline = active_evaluation.expires_at
            evaluation_timeout_message = active_evaluation.error_message
        retry_number = 0

        while True:
            remaining = self._remaining(deadline, model, evaluation_timeout_message)
            attempt_timeout = asyncio.timeout(remaining)
            try:
                async with attempt_timeout:
                    return await self._attempt(
                        model,
                        operation,
                        deadline=deadline,
                        budget_alias=budget_alias,
                        evaluation_timeout_message=evaluation_timeout_message,
                    )
            except TimeoutError as exc:
                # Preserve the scanner's error type when this inner timer is
                # enforcing the same rule deadline.
                if attempt_timeout.expired():
                    raise self._budget_error(model, evaluation_timeout_message) from exc
                raise
            except LLMError as exc:
                if exc.details.get("code") not in RETRYABLE_ERROR_CODES:
                    raise

                retry_number += 1
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise self._budget_error(model, evaluation_timeout_message) from exc
                delay = min(0.25 * (2 ** min(retry_number - 1, 3)), 2.0)
                if delay >= remaining:
                    raise self._budget_error(model, evaluation_timeout_message) from exc
                await _sleep(delay)

    async def _attempt(
        self,
        model: str,
        operation: Callable[[float | None], Awaitable[T]],
        *,
        deadline: float | None,
        budget_alias: str | None,
        evaluation_timeout_message: str | None = None,
    ) -> T:
        """Acquire one limiter ticket and invoke one physical attempt."""

        async def invoke() -> T:
            budget = _current_model_call_budget.get()
            if budget is not None:
                budget.consume(budget_alias or model)
            remaining = None
            if deadline is not None:
                remaining = self._remaining(deadline, model, evaluation_timeout_message)
            return await operation(remaining)

        if self.rate_limiter is None:
            return await invoke()
        budget = _current_model_call_budget.get()
        if budget is not None:
            budget.check(budget_alias or model)
        async with self.rate_limiter.limit(model):
            return await invoke()

    @staticmethod
    def _remaining(
        deadline: float,
        model: str,
        evaluation_timeout_message: str | None,
    ) -> float:
        """Return positive budget remaining or raise the logical timeout."""

        remaining = deadline - monotonic()
        if remaining > 0:
            return remaining
        raise ModelUsage._budget_error(model, evaluation_timeout_message)

    @staticmethod
    def _budget_error(model: str, evaluation_timeout_message: str | None) -> Exception:
        """Preserve rule-deadline ownership or return a model-call timeout."""

        if evaluation_timeout_message is not None:
            return QuarryTimeoutError(evaluation_timeout_message)
        return ModelUsage._timeout_error(model)

    @staticmethod
    def _timeout_error(model: str) -> LLMError:
        """Build the structured timeout shared by direct model integrations."""

        return LLMError(
            f"Model call timed out for '{model}'",
            {"code": "timeout", "model": model},
        )
