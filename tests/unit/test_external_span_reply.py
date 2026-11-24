# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Fresh synthetic native-protocol controls, independent of provider transports."""

from dataclasses import replace

import pytest

from quarry.types.external_spans import (
    ExternalReadRange,
    ExternalSpanLimitError,
    ExternalSpanLimits,
    ExternalSpanRange,
    ExternalSpanReply,
    ExternalSpanReplyError,
    validate_external_span_reply,
)


def reply(*, complete=True):
    return ExternalSpanReply(
        6,
        (ExternalSpanRange(1, 3, 0.9), ExternalSpanRange(4, 5, 1)),
        (ExternalReadRange(3, 6), ExternalReadRange(0, 2), ExternalReadRange(2, 4)),
        complete,
    )


def test_request_digest_has_a_stable_native_protocol_vector():
    native = ExternalSpanReply.for_input("abc", (), (ExternalReadRange(0, 3),), complete=True)
    assert native.input_digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert validate_external_span_reply(native, 3).input_digest == native.input_digest
    assert (
        ExternalSpanReply.for_input(
            "xyz", (), (ExternalReadRange(0, 3),), complete=True
        ).input_digest
        != native.input_digest
    )


@pytest.mark.parametrize("digest", [False, b"0" * 64, "0" * 63, "g" * 64])
def test_malformed_native_request_digests_are_rejected(digest):
    with pytest.raises(ExternalSpanReplyError):
        ExternalSpanReply(0, (), (), True, digest)


def test_request_digest_handles_unicode_character_offsets_without_byte_coercion():
    text = "🙂\ud800a"
    native = ExternalSpanReply.for_input(text, (), (ExternalReadRange(0, 3),), complete=True)
    assert native.input_characters == 3 and len(native.input_digest) == 64
    assert validate_external_span_reply(native, 3).covered_characters == 3


def test_complete_ranges_are_detached_and_overlapping_coverage_is_a_union():
    raw = reply()
    accepted = validate_external_span_reply(raw, 6)
    assert accepted == raw and accepted is not raw
    assert accepted.spans[0] is not raw.spans[0]
    assert accepted.coverage[0] is not raw.coverage[0]
    assert accepted.covered_characters == 6 and accepted.complete is True
    assert accepted.spans[1].score == 1.0 and type(accepted.spans[1].score) is float
    object.__setattr__(raw.spans[0], "score", 0.1)
    assert accepted.spans[0].score == 0.9


def test_partial_coverage_and_detections_remain_partial():
    raw = ExternalSpanReply(
        6,
        (ExternalSpanRange(0, 2, 0.8),),
        (ExternalReadRange(0, 2), ExternalReadRange(4, 6)),
        False,
    )
    accepted = validate_external_span_reply(raw, 6)
    assert accepted.complete is False and accepted.covered_characters == 4
    assert accepted.spans == (ExternalSpanRange(0, 2, 0.8),)


def test_full_coverage_does_not_upgrade_a_declared_incomplete_reply():
    accepted = validate_external_span_reply(reply(complete=False), 6)
    assert accepted.covered_characters == 6 and accepted.complete is False


@pytest.mark.parametrize("complete", [False, True])
def test_empty_input_has_explicit_completeness_and_zero_coverage(complete):
    accepted = validate_external_span_reply(ExternalSpanReply(0, (), (), complete), 0)
    assert accepted.complete is complete and accepted.spans == ()
    assert accepted.covered_characters == 0


def test_missing_nonempty_input_is_unknown_not_a_complete_negative():
    accepted = validate_external_span_reply(ExternalSpanReply(6, (), (), False), 6)
    assert not accepted.complete and accepted.covered_characters == 0
    with pytest.raises(ExternalSpanReplyError, match="omits input coverage"):
        ExternalSpanReply(6, (), (), True)


@pytest.mark.parametrize(
    "coverage", [(), (ExternalReadRange(0, 2),), (ExternalReadRange(0, 2), ExternalReadRange(3, 6))]
)
def test_complete_declaration_cannot_hide_a_gap(coverage):
    with pytest.raises(ExternalSpanReplyError, match="omits input coverage"):
        ExternalSpanReply(6, (), coverage, True)


@pytest.mark.parametrize("span", [ExternalSpanRange(1, 4, 0.8), ExternalSpanRange(2, 3, 0.8)])
def test_detection_in_an_uninspected_gap_is_rejected(span):
    with pytest.raises(ExternalSpanReplyError, match="outside inspected"):
        ExternalSpanReply(6, (span,), (ExternalReadRange(0, 2), ExternalReadRange(3, 6)), False)


@pytest.mark.parametrize("value", [True, -1, 4_000_001, 1.5, "1"])
def test_native_input_lengths_are_strict(value):
    with pytest.raises(ExternalSpanReplyError):
        ExternalSpanReply(value, (), (), False)
    with pytest.raises(ExternalSpanLimitError):
        validate_external_span_reply(reply(), value)


@pytest.mark.parametrize("score", [True, float("nan"), float("inf"), -0.1, 1.1, "0.5"])
def test_invalid_scores_are_not_coerced(score):
    with pytest.raises(ExternalSpanReplyError, match="score"):
        ExternalSpanRange(0, 1, score)


@pytest.mark.parametrize(
    "start,end", [(True, 2), (0, True), (-1, 2), (2, 1), (0.0, 1), (0, 4_000_001)]
)
def test_invalid_character_coordinates_are_rejected(start, end):
    for kind in (ExternalReadRange, ExternalSpanRange):
        with pytest.raises(ExternalSpanReplyError, match="interval"):
            kind(start, end, *([0.5] if kind is ExternalSpanRange else []))


def test_empty_detections_are_invalid_while_empty_read_intervals_add_no_units():
    with pytest.raises(ExternalSpanReplyError):
        ExternalSpanRange(1, 1, 0.5)
    raw = ExternalSpanReply(6, (), (ExternalReadRange(1, 1),), False)
    assert raw.covered_characters == 0


@pytest.mark.parametrize("value", [False, 1, "true", None])
def test_completeness_requires_a_native_boolean(value):
    if type(value) is bool:
        assert ExternalSpanReply(0, (), (), value).complete is value
    else:
        with pytest.raises(ExternalSpanReplyError, match="boolean"):
            ExternalSpanReply(0, (), (), value)


def test_host_conversion_and_container_hooks_are_never_invoked():
    class Host:
        def __bool__(self):
            pytest.fail("boolean conversion")

        def __float__(self):
            pytest.fail("numeric conversion")

        def __len__(self):
            pytest.fail("container length")

        def __iter__(self):
            pytest.fail("container iteration")

    with pytest.raises(ExternalSpanReplyError):
        ExternalSpanRange(0, 1, Host())
    with pytest.raises(ExternalSpanReplyError):
        ExternalSpanReply(0, (), (), Host())
    with pytest.raises(ExternalSpanReplyError):
        ExternalSpanReply(0, Host(), (), False)


def test_range_admission_precedes_inspection_of_an_over_limit_container():
    with pytest.raises(ExternalSpanLimitError):
        ExternalSpanReply(6, [object()] * 4097, (), False)
    raw = reply()
    with pytest.raises(ExternalSpanLimitError):
        validate_external_span_reply(raw, 6, ExternalSpanLimits(max_spans=1))


@pytest.mark.parametrize(
    "limits",
    [
        ExternalSpanLimits(max_input_characters=5),
        ExternalSpanLimits(max_coverage_ranges=0),
        ExternalSpanLimits(max_work=1),
    ],
)
def test_reader_limits_precede_admission(limits):
    with pytest.raises(ExternalSpanLimitError):
        validate_external_span_reply(reply(), 6, limits)


@pytest.mark.parametrize("field", list(ExternalSpanLimits.__dataclass_fields__))
@pytest.mark.parametrize("value", [True, -1, 1.5, "1", 10**20])
def test_reader_caps_are_strict_bounded_native_integers(field, value):
    with pytest.raises(ExternalSpanLimitError):
        replace(ExternalSpanLimits(), **{field: value})


def test_mutated_frozen_ranges_and_completeness_are_revalidated():
    raw = reply()
    object.__setattr__(raw.spans[0], "end", 7)
    with pytest.raises(ExternalSpanReplyError):
        validate_external_span_reply(raw, 6)
    raw = ExternalSpanReply(6, (), (ExternalReadRange(0, 2),), False)
    object.__setattr__(raw, "complete", True)
    object.__setattr__(raw, "_coverage_union", ((0, 6),))
    with pytest.raises(ExternalSpanReplyError, match="omits input coverage"):
        validate_external_span_reply(raw, 6)


def test_input_binding_is_exact_and_uses_character_counts():
    text = "α😀"
    raw = ExternalSpanReply(2, (ExternalSpanRange(1, 2, 0.5),), (ExternalReadRange(0, 2),), True)
    assert validate_external_span_reply(raw, len(text)).covered_characters == 2
    with pytest.raises(ExternalSpanReplyError, match="does not match"):
        validate_external_span_reply(raw, len(text.encode("utf-8")))


def test_uninitialized_native_reply_is_a_typed_invalid_reply():
    raw = object.__new__(ExternalSpanReply)
    with pytest.raises(ExternalSpanReplyError, match="reply fields"):
        validate_external_span_reply(raw, 0)


def test_uninitialized_native_range_is_a_typed_invalid_reply():
    raw = object.__new__(ExternalSpanRange)
    with pytest.raises(ExternalSpanReplyError, match="detection range"):
        ExternalSpanReply(6, (raw,), (ExternalReadRange(0, 6),), True)


def test_uninitialized_native_limits_are_a_typed_limit_failure():
    raw = object.__new__(ExternalSpanLimits)
    with pytest.raises(ExternalSpanLimitError, match="limits"):
        validate_external_span_reply(reply(), 6, raw)
