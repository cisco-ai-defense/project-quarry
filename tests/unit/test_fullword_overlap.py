# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""A boundary-rejected literal must leave later overlapping candidates available."""

import pytest

from quarry.engine.captures import matches_for
from quarry.engine.matchers import (
    Modifiers,
    RegexBackend,
    count_occurrences,
    extract_match,
    find_position,
    match_contains,
    match_spans,
)
from quarry.safety.regex_backend import Re2RegexBackend


@pytest.mark.parametrize("backend_type", [RegexBackend, Re2RegexBackend])
@pytest.mark.parametrize(
    "reader", ["presence", "count", "capped", "position", "capture", "extract", "spans"]
)
@pytest.mark.parametrize(
    "text,pattern,nocase,expected",
    [
        ("x---", "--", False, [(2, 4)]),
        ("x-----", "--", False, [(2, 4), (4, 6)]),
        ("xé-é-é", "é-é", False, [(3, 6)]),
        ("İx---", "--", True, [(3, 5)]),
    ],
)
def test_boundary_rejections_preserve_overlaps_but_accepted_matches_do_not_overlap(
    backend_type, reader, text, pattern, nocase, expected
):
    backend = backend_type()
    mods = Modifiers(fullword=True, nocase=nocase)
    if reader == "presence":
        assert match_contains(text, pattern, mods, regex=backend)
    elif reader == "count":
        assert count_occurrences(text, pattern, mods, regex=backend) == len(expected)
    elif reader == "capped":
        assert count_occurrences(text, pattern, mods, regex=backend, limit=1) == 1
    elif reader == "position":
        assert find_position(text, pattern, mods, regex=backend) == expected[0][0]
    elif reader == "capture":
        assert list(matches_for(pattern, text, mods, backend)) == [
            text[slice(*span)] for span in expected
        ]
    elif reader == "extract":
        assert extract_match(text, pattern, mods, regex=backend) == text[slice(*expected[0])]
    else:
        assert list(match_spans(text, pattern, mods, regex=backend)) == expected


def test_rejected_overlapping_candidates_remain_interruptible_by_a_work_owner():
    from quarry.engine.local_work import local_work_scope

    class AllowanceReached(RuntimeError):
        pass

    class BoundedOwner:
        segments = 0

        def call(self, phase, function, *args, **kwargs):
            self.segments += 1
            if self.segments > 64:
                raise AllowanceReached("synthetic local allowance")
            return function(*args, **kwargs)

        def iterate(self, iterator):
            yield from iterator

        def ensure_authoritative(self):
            pass

    # Every candidate has a word neighbor. No accepted yield can serve as a
    # checkpoint, so the rejected search itself must still consult its work owner.
    text, pattern, mods = "x" * 4096, "xx", Modifiers(fullword=True)
    assert list(match_spans(text, pattern, mods)) == []
    with local_work_scope(BoundedOwner()), pytest.raises(AllowanceReached):
        list(match_spans(text, pattern, mods))
