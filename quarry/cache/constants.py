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

"""Cache and performance-related constants."""

REGEX_CACHE_SIZE = 1024
# The process-wide result cache: model responses and regex results by their inputs.
RESULT_CACHE_MAX_ENTRIES = 100_000
RESULT_CACHE_TTL_SECONDS = 43_200.0  # 12 hours after an entry's last use
# A refused model call is kept this long after it was stored, however often it is read: a
# scan's reads of the call agree, and a later scan asks the provider again.
REFUSAL_TTL_SECONDS = 600.0

MAX_CONTAINS_ANY_SIZE = 1000
MAX_STRING_ARRAY_SIZE = 1000
