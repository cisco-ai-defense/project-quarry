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

"""Safety and validation utilities."""

from quarry.safety.limits import SafetyLimits
from quarry.safety.timer import (
    QuarryTimeoutError,
    evaluation_deadline,
    raise_if_evaluation_expired,
    remaining_evaluation_time,
    scan_deadline,
    without_rule_deadline,
)
from quarry.safety.validator import RuleValidator, ValidationResult, ValidationWarning

__all__ = [
    "QuarryTimeoutError",
    "evaluation_deadline",
    "scan_deadline",
    "raise_if_evaluation_expired",
    "remaining_evaluation_time",
    "without_rule_deadline",
    "RuleValidator",
    "SafetyLimits",
    "ValidationResult",
    "ValidationWarning",
]
