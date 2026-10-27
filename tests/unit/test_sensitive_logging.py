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

"""Rule text and provider diagnostics must not be copied into application logs."""

from __future__ import annotations

import re

import pytest
from loguru import logger

from quarry.cache.regex_cache import RegexCache
from tests.unit.model_fakes import model_entry, v2_scanner


@pytest.fixture
def captured_app_logs():
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), format="{message} {extra}")
    try:
        yield records
    finally:
        logger.remove(sink_id)


def test_provider_exception_text_is_absent_from_scan_logs(tmp_path, captured_app_logs) -> None:
    sentinel = "SYNTHETIC_PRIVATE_PROVIDER_DIAGNOSTIC"

    class FailingAdapter:
        async def classify_messages(self, _config, _messages, _tools, *, timeout=None):
            raise RuntimeError(sentinel)

    scanner = v2_scanner(
        tmp_path,
        {"R": 'model("m").classify(messages) > 0.5'},
        {"m": model_entry(classify=0)},
        FailingAdapter(),
    )
    result = scanner.scan_data({"messages": [{"role": "user", "content": "hello"}]})

    assert result.completion_state == "incomplete"
    assert sentinel not in "\n".join(captured_app_logs)


def test_authored_regex_is_absent_from_rejection_logs(captured_app_logs) -> None:
    sentinel = "SYNTHETIC_PRIVATE_PATTERN"
    pattern = f"(?<={sentinel})x"

    with pytest.raises(re.error):
        RegexCache().compile(pattern)

    assert sentinel not in "\n".join(captured_app_logs)
