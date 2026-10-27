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

"""Tests for task-local deadlines and the signal timer."""

import asyncio
import time

import pytest

from quarry.safety.limits import SafetyLimits
from quarry.safety.timer import (
    QuarryTimeoutError,
    clamp_to_evaluation_deadline,
    current_evaluation_deadline,
    evaluation_deadline,
    raise_if_evaluation_expired,
)


@pytest.mark.asyncio
async def test_task_deadlines_are_nested_isolated_and_validated() -> None:
    with evaluation_deadline(0.2, "outer"):
        outer_remaining = clamp_to_evaluation_deadline(10)
        with evaluation_deadline(5, "inner"):
            inner_remaining = clamp_to_evaluation_deadline(10)
    assert 0 < inner_remaining <= outer_remaining <= 0.2

    with (
        pytest.raises(QuarryTimeoutError, match="expired"),
        evaluation_deadline(0.001, "expired"),
    ):
        time.sleep(0.01)
        raise_if_evaluation_expired()

    with evaluation_deadline(0.2, "outer") as outer:
        with (
            pytest.raises(QuarryTimeoutError, match="inner"),
            evaluation_deadline(0.001, "inner"),
        ):
            time.sleep(0.01)

        assert current_evaluation_deadline() is outer
        assert 0 < clamp_to_evaluation_deadline(1) <= 0.2
    assert current_evaluation_deadline() is None

    entered = asyncio.Event()
    observed = asyncio.Event()

    async def owner() -> None:
        with evaluation_deadline(1, "owner") as deadline:
            entered.set()
            await observed.wait()
            assert current_evaluation_deadline() is deadline

    async def observer() -> None:
        await entered.wait()
        assert current_evaluation_deadline() is None
        observed.set()

    await asyncio.gather(owner(), observer())
    for timeout in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            SafetyLimits(rule_timeout=timeout)
    for timeout in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"), evaluation_deadline(timeout):
            pass
        with pytest.raises(ValueError, match="finite"):
            clamp_to_evaluation_deadline(timeout)
        with pytest.raises(ValueError, match="finite"):
            SafetyLimits(rule_timeout=timeout)
