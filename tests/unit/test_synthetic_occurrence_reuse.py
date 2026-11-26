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

"""Constructed identity survives a change of root-local canonical index."""

import pytest

from quarry.types.messages import Messages
from quarry.types.results import Span
from quarry.types.transforms import slice_provenance


@pytest.mark.parametrize("source_count", [1, 3])
@pytest.mark.parametrize("reverse", [False, True])
def test_reusing_a_grafted_literal_keeps_both_occurrences(source_count, reverse):
    source = Messages.from_openai(
        [{"role": "user", "content": f"source {index}"} for index in range(source_count)]
    )
    literal = Messages().add_message("user", "same")
    first = Messages.combine((source, literal))
    parts = (literal, first) if reverse else (first, literal)

    combined = Messages.combine(parts)

    assert combined.to_openai_messages() == [m.to_openai_dict() for part in parts for m in part]
    assert [m.message_index for m in combined if m.content == "same"] == [source_count] * 2
    assert literal[0].message_index == 0 and first[-1].message_index == source_count


def test_nested_constructed_only_roots_preserve_the_reused_value():
    prefix = Messages().add_message("assistant", "before")
    literal = Messages().add_message("user", "same")
    current = Messages.combine((prefix, literal))
    for count in range(2, 5):
        current = Messages.combine((current, literal))
        assert current.to_openai_messages() == [
            {"role": "assistant", "content": "before"},
            *[{"role": "user", "content": "same"}] * count,
        ]
    assert literal[0].message_index == 0


def test_role_rewritten_literal_keeps_its_anchor_when_reused_after_grafting():
    source = Messages.from_openai([{"role": "user", "content": "source"}])
    literal = Messages().add_message("user", "same")
    rewritten = literal.rewritten([{"role": "assistant", "content": "same"}])
    first = Messages.combine((source, rewritten))

    combined = Messages.combine((first, literal))

    assert combined.to_openai_messages() == [
        {"role": "user", "content": "source"},
        {"role": "assistant", "content": "same"},
        {"role": "user", "content": "same"},
    ]
    assert literal[0].role == "user" and literal[0].message_index == 0


@pytest.mark.parametrize("difference", ["text", "role", "provenance"])
def test_index_normalization_does_not_accept_different_canonical_content(difference):
    source = Messages.from_openai([{"role": "user", "content": "source text"}])
    identity = object()
    literal = Messages()._add_synthetic_message("user", "same", identity=identity)
    first = Messages.combine((source, literal))
    provenance = (
        slice_provenance(source[0]._content_provenance(), 0, 4)
        if difference == "provenance"
        else None
    )
    other = Messages()._add_synthetic_message(
        "assistant" if difference == "role" else "user",
        "else" if difference == "text" else "same",
        identity=identity,
        provenance=provenance,
    )

    with pytest.raises(ValueError, match="constructed identity has different content variants"):
        Messages.combine((first, other))


def test_source_message_coordinates_stay_distinct_beside_reused_literals():
    source = Messages.from_openai(
        [{"role": "user", "content": "same"}, {"role": "user", "content": "same"}]
    )
    literal = Messages().add_message("user", "literal")
    first = Messages.combine((source, literal))
    combined = Messages.combine((first, literal))

    for position in range(2):
        span = combined._bind_span(Span(position, 0, 4))
        assert span.message_index == position
        assert combined.extract(span) == "same"
    assert [m.message_index for m in combined] == [0, 1, 2, 2]
