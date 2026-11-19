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

"""The new scan verbosity argument fails closed before host callbacks or compilation."""

from typing import Any

import pytest

from quarry.plugin.service import PluginInputError, QuarryPlugin


@pytest.mark.parametrize("value", ["", "brief", "compact\0", [], {}, 1, True, 1.0])
async def test_invalid_scan_verbosity_is_rejected_before_rule_preparation(
    value: Any, monkeypatch
) -> None:
    plugin = QuarryPlugin()

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("an invalid verbosity value reached rule preparation")

    monkeypatch.setattr(plugin, "_prepare_rule", unexpected_preparation)
    with pytest.raises(PluginInputError, match="verbosity must be 'compact' or 'full'"):
        await plugin.scan(
            rule_text='rule "Read" condition: user_said("hello")',
            conversation={"messages": [{"role": "user", "content": "hello"}]},
            verbosity=value,
        )


@pytest.mark.parametrize("text_subclass", [False, True])
async def test_scan_verbosity_does_not_invoke_custom_equality_or_truth_hooks(
    text_subclass: bool, monkeypatch
) -> None:
    class HookedValue:
        def __eq__(self, other):
            pytest.fail("an unsupported verbosity value ran its equality hook")

        def __bool__(self):
            pytest.fail("an unsupported verbosity value ran its truth hook")

    class HookedText(str):
        def __eq__(self, other):
            pytest.fail("a verbosity text subclass ran its equality hook")

        def __bool__(self):
            pytest.fail("a verbosity text subclass ran its truth hook")

    value: Any = HookedText("compact") if text_subclass else HookedValue()
    plugin = QuarryPlugin()

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("an unsupported verbosity value reached rule preparation")

    monkeypatch.setattr(plugin, "_prepare_rule", unexpected_preparation)
    with pytest.raises(PluginInputError, match="verbosity must be 'compact' or 'full'"):
        await plugin.scan(
            rule_text='rule "Read" condition: user_said("hello")',
            conversation={"messages": [{"role": "user", "content": "hello"}]},
            verbosity=value,
        )
