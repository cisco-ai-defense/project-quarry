# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Synthetic detached geometry: every part/channel is validated before admission."""

from dataclasses import replace

import pytest

from quarry.safety.timer import QuarryTimeoutError
from quarry.types.candidates import (
    CandidateBatchIdentity,
    CandidateBatchLimits,
    CandidateBatchReply,
    CandidateBatchRequest,
    CandidateChannelReply,
    CandidateCoverage,
    CandidateInput,
    CandidateLimitError,
    CandidatePart,
    CandidateRange,
    CandidateReplyError,
    validate_candidate_batch_reply,
)


def request(**changes):
    base = CandidateBatchRequest(
        CandidateBatchIdentity("synthetic", "1", "heads"),
        CandidateInput(
            (
                CandidatePart("content", ("messages", 0, "content"), "abc"),
                CandidatePart("content", ("messages", 1, "content"), "XYZ"),
            )
        ),
        ("left", "right"),
        "0" * 64,
    )
    return replace(base, **changes)


def channel(name, **changes):
    return replace(
        CandidateChannelReply(
            name,
            (CandidateRange(0, 0, 1, 0.9),),
            (CandidateCoverage(0, 0, 3), CandidateCoverage(1, 0, 3)),
            True,
        ),
        **changes,
    )


def reply(value, **changes):
    return replace(
        CandidateBatchReply(value.digest, (channel("left"), channel("right"))), **changes
    )


def validate(raw, value=None, **limits):
    return validate_candidate_batch_reply(
        raw,
        value or request(),
        CandidateBatchLimits(**limits),
        check_deadline=lambda: None,
    )


def test_request_and_reply_snapshot_native_containers_and_child_objects():
    part = CandidatePart("content", ["messages", 0], "synthetic")
    parts = [part]
    value = CandidateInput(parts)
    parts.clear()
    object.__setattr__(part, "text", "later mutation")
    assert value.parts[0].text == "synthetic"
    req = request()
    original = reply(req)
    checked = validate(original, req)
    object.__setattr__(original.channels[0].ranges[0], "end", 999)
    assert checked.channels[0].ranges[0].end == 1
    assert checked is not original and checked.channels[0] is not original.channels[0]


def test_channel_order_is_canonical_and_version_one_is_a_valid_literal_identity():
    left = request(channels=("right", "left"))
    assert left == request() and left.provider.version == "1"


@pytest.mark.parametrize(
    "change",
    ["text", "address", "kind", "order", "namespace", "version", "batch", "settings", "channels"],
)
def test_every_part_and_provider_setting_binds_request_identity(change):
    original = request()
    source = list(original.input.parts)
    if change in {"text", "address", "kind"}:
        field, value = {
            "text": ("text", "different"),
            "address": ("path", ("messages", 2, "content")),
            "kind": ("kind", "tool_argument"),
        }[change]
        source[0] = replace(source[0], **{field: value})
        changed = replace(original, input=CandidateInput(source))
    elif change == "order":
        changed = replace(original, input=CandidateInput(tuple(reversed(source))))
    elif change in {"namespace", "version", "batch"}:
        changed = replace(original, provider=replace(original.provider, **{change: "changed"}))
    elif change == "settings":
        changed = replace(original, settings_digest="1" * 64)
    else:
        changed = replace(original, channels=("left",))
    assert changed.digest != original.digest
    with pytest.raises(CandidateReplyError, match="different request"):
        validate(reply(original), changed)


@pytest.mark.parametrize(
    "values",
    [
        (True, 0, 1, 0.9),
        (-1, 0, 1, 0.9),
        (4096, 0, 1, 0.9),
        (0, True, 1, 0.9),
        (0, 0, False, 0.9),
        (0, -1, 1, 0.9),
        (0, 1, 0, 0.9),
        (0, 0, 0, 0.9),
        (0, 0, 1, True),
        (0, 0, 1, -0.1),
        (0, 0, 1, 1.1),
        (0, 0, 1, float("nan")),
        (0, 0, 1, float("inf")),
    ],
)
def test_range_geometry_and_probabilities_reject_non_native_or_nonfinite_values(values):
    with pytest.raises(CandidateReplyError):
        CandidateRange(*values)


@pytest.mark.parametrize("name", ["", "../load", "x\ny", "é", "_bad", None, False, "a" * 65])
def test_capability_names_never_become_import_paths_or_unbounded_identifiers(name):
    with pytest.raises(CandidateReplyError):
        CandidateBatchIdentity(name, "1", "heads")


@pytest.mark.parametrize("version", ["", "\n", "has space", "\ud800", None, False, "a" * 129])
def test_versions_require_bounded_visible_literal_text(version):
    with pytest.raises(CandidateReplyError):
        CandidateBatchIdentity("synthetic", version, "heads")


@pytest.mark.parametrize(
    "bad",
    [
        {"ranges": (CandidateRange(2, 0, 1, 0.0),)},
        {"ranges": (CandidateRange(0, 2, 4, 0.0),)},
        {"coverage": (CandidateCoverage(0, 0, 4),)},
        {"coverage": (CandidateCoverage(2, 0, 3),)},
        {"coverage": (CandidateCoverage(0, 0, 1),)},
    ],
)
def test_even_an_unselected_or_zero_score_channel_is_rejected_atomically(bad):
    req = request()
    raw = reply(req, channels=(channel("left"), channel("right", **bad)))
    with pytest.raises(CandidateReplyError):
        validate(raw, req)


def test_missing_extra_and_repeated_channels_are_not_admitted():
    req = request()
    for channels in [(channel("left"),), (channel("left"), channel("foreign"))]:
        with pytest.raises(CandidateReplyError):
            validate(reply(req, channels=channels), req)
    with pytest.raises(CandidateReplyError):
        reply(req, channels=(channel("left"), channel("left")))


def test_partial_coverage_stays_partial_and_ranges_must_be_inside_known_coverage():
    req = request()
    partial = channel("left", complete=False, coverage=(CandidateCoverage(0, 0, 1),))
    checked = validate(reply(req, channels=(partial, channel("right"))), req)
    assert checked.channels[0].complete is False
    outside = replace(partial, ranges=(CandidateRange(0, 1, 2, 0.9),))
    with pytest.raises(CandidateReplyError, match="declared coverage"):
        validate(reply(req, channels=(outside, channel("right"))), req)


def test_adjacent_coverage_is_joined_only_within_one_requested_part():
    req = request()
    covered = channel(
        "left",
        coverage=(
            CandidateCoverage(0, 0, 1),
            CandidateCoverage(0, 1, 3),
            CandidateCoverage(1, 0, 3),
        ),
    )
    assert validate(reply(req, channels=(covered, channel("right"))), req)
    with pytest.raises(CandidateReplyError):
        validate(
            reply(
                req,
                channels=(
                    replace(covered, ranges=(CandidateRange(0, 2, 4, 0.1),)),
                    channel("right"),
                ),
            ),
            req,
        )


@pytest.mark.parametrize(
    "limits",
    [
        {"max_parts": 1},
        {"max_input_characters": 5},
        {"max_ranges_per_channel": 0},
        {"max_coverage_per_channel": 1},
        {"max_work": 1},
    ],
)
def test_reader_limits_fail_before_candidate_filtering_without_clipping(limits):
    req = request()
    with pytest.raises(CandidateLimitError):
        validate(reply(req), req, **limits)


@pytest.mark.parametrize("limits", [{"max_parts": 1}, {"max_input_characters": 5}, {"max_work": 1}])
def test_reader_limits_refuse_before_reconstructing_the_input_digest(monkeypatch, limits):
    import quarry.types.candidates as candidates

    req = request()
    raw = reply(req)

    def forbidden(*args):
        raise AssertionError("input was hashed before applying the reader's limits")

    monkeypatch.setattr(candidates, "_feed", forbidden)
    with pytest.raises(CandidateLimitError):
        validate(raw, req, **limits)


def test_deadline_category_and_identity_are_preserved():
    expected = QuarryTimeoutError("synthetic owned deadline")

    def expired():
        raise expected

    req = request()
    with pytest.raises(QuarryTimeoutError) as caught:
        validate_candidate_batch_reply(
            reply(req), req, CandidateBatchLimits(), check_deadline=expired
        )
    assert caught.value is expected


@pytest.mark.parametrize("max_work", [16, 70])
def test_part_address_work_is_bounded_before_rehashing(monkeypatch, max_work):
    import quarry.types.candidates as candidates

    part = CandidatePart("content", ("p" * 256,) * 64, "a")
    req = request(input=CandidateInput((part,)))
    raw = reply(
        req,
        channels=tuple(
            CandidateChannelReply(
                name, (CandidateRange(0, 0, 1, 0.9),), (CandidateCoverage(0, 0, 1),), True
            )
            for name in req.channels
        ),
    )

    def forbidden(*args):
        raise AssertionError("source addresses were hashed before checking their work limit")

    monkeypatch.setattr(candidates, "_feed", forbidden)
    with pytest.raises(CandidateLimitError):
        validate(raw, req, max_work=max_work)


def test_forged_non_native_part_address_is_rejected_before_hashing(monkeypatch):
    import quarry.types.candidates as candidates

    req = request()
    raw = reply(req)
    object.__setattr__(req.input.parts[0], "path", (True,))

    def forbidden(*args):
        raise AssertionError("invalid source address reached hashing")

    monkeypatch.setattr(candidates, "_feed", forbidden)
    with pytest.raises(CandidateReplyError):
        validate(raw, req)


def test_repr_and_failures_do_not_echo_input_text_or_part_addresses():
    part = CandidatePart("content", ("SYNTHETIC_PRIVATE_KEY",), "SYNTHETIC_PRIVATE_TEXT")
    req = request(input=CandidateInput((part,)))
    assert "SYNTHETIC_PRIVATE" not in repr(part) + repr(req) + repr(req.input)


def test_mutated_digest_is_rejected_without_calling_hostile_equality():
    class Hostile:
        def __eq__(self, other):
            raise AssertionError("hostile equality was invoked")

    req = request()
    raw = reply(req)
    object.__setattr__(raw, "request_digest", Hostile())
    with pytest.raises(CandidateReplyError):
        validate(raw, req)


def test_empty_input_can_be_completely_inspected_without_fabricating_a_span():
    req = request(input=CandidateInput(()))
    raw = reply(
        req, channels=tuple(CandidateChannelReply(name, (), (), True) for name in req.channels)
    )
    assert validate(raw, req) == raw
