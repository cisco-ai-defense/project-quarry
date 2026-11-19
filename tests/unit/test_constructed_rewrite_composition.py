# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Constructed role rewrites retain their anchors across root coalescing."""

from __future__ import annotations

import pytest

from quarry import compile_policy, load_compiled_policy
from quarry.types.messages import Messages
from quarry.types.results import Span
from tests.unit.test_message_rewrites import service


@pytest.mark.parametrize("source_backed", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_constructed_role_rewrite_composes_without_mutating_its_original(source_backed, reverse):
    original = Messages().add_message("user", "same")
    rewritten = original.rewritten([{"role": "assistant", "content": "same"}])
    partner = (
        Messages.from_openai([{"role": "user", "content": "source"}])
        if source_backed
        else Messages().add_message("user", "literal")
    )
    parts = (partner, rewritten) if reverse else (rewritten, partner)

    combined = Messages.combine(parts)

    assert combined.to_openai_messages() == [m.to_openai_dict() for p in parts for m in p]
    span = combined._bind_span(Span(int(reverse), 0, 4))
    assert combined.extract(span) == "same"
    assert span.source_unmapped
    assert original[0].role == "user"
    assert rewritten[0]._rewrite_source is original[0]


def test_repeated_coalescing_and_chained_role_rewrites_keep_copied_source_ranges():
    source = Messages.from_openai([{"role": "user", "content": "same source"}])
    original = source._empty_constructor_context()._add_synthetic_message(
        "user", source[0].content, identity=object(), provenance=source[0]._content_provenance()
    )
    current = original.rewritten([{"role": "assistant", "content": source[0].content}])
    for _ in range(3):
        current = Messages.combine((current, Messages().add_message("developer", "new")))[:1]
        current = current.rewritten([{"role": "user", "content": source[0].content}])
        current = current.rewritten([{"role": "assistant", "content": source[0].content}])
        span = current._bind_span(Span(0, 0, 4))
        assert current.extract(span) == "same"
        assert [r.to_dict() for r in span.source_ranges] == [
            {"message_index": 0, "start": 0, "end": 4}
        ]
    with pytest.raises(ValueError, match="canonical source"):
        Messages.combine((current, Messages.from_openai([{"role": "user", "content": "foreign"}])))
    assert source[0].role == original[0].role == "user"


def test_coalescing_does_not_promote_a_foreign_rewrite_anchor():
    original = Messages().add_message("user", "same")
    rewritten = original.rewritten([{"role": "assistant", "content": "same"}])
    foreign = Messages().add_message("user", "same")
    object.__setattr__(rewritten[0], "_rewrite_source", foreign[0])
    with pytest.raises(ValueError, match="does not match its source message"):
        Messages.combine((rewritten, Messages().add_message("user", "other")))


class AsAssistant:
    local = True

    def transform_messages(self, config, wire):
        return [dict(message, role="assistant") for message in wire]


EXPRESSIONS = [
    'message("user", "same").transform("rewrite") + messages',
    'messages + message("user", "same").transform("rewrite")',
    'message("user", "same").transform("rewrite") + message("user", "other")',
    '(message("user", "same").transform("rewrite") + messages) + message("user", "other")',
    '(message("user", "same").transform("rewrite") if true else messages) + messages',
    '[message("user", item.content).transform("rewrite") for item in messages] + messages',
    '[[message("user", inner.content).transform("rewrite") for inner in messages] '
    "for outer in messages] + messages",
]


@pytest.mark.parametrize("expression", EXPRESSIONS)
@pytest.mark.asyncio
async def test_prepared_rules_compose_constructed_role_rewrites(tmp_path, expression):
    config, backend, _ = service(tmp_path, adapter=AsAssistant())
    policy = compile_policy(
        'rule "R" condition: (' + expression + ').assistant_messages[0].content == "same"',
        runtime_config=config,
        model_inference_backend=backend,
    )
    raw = policy.to_bytes()
    loaded = load_compiled_policy(raw, runtime_config=config, model_inference_backend=backend)
    assert loaded.to_bytes() == raw
    result = await loaded.scan_async({"messages": [{"role": "user", "content": "same"}]})
    assert result.is_complete
    assert result.rule_matches == ["R"]
