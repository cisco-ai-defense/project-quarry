# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Located input projections retain owners through the newer message runtime."""

from array import array

import pytest

from quarry.integrations.content_transforms import apply_content_transform
from quarry.integrations.model_expression import _cover_index, _is_covered, _view_source_texts
from quarry.types.messages import Messages, _synthetic_key
from quarry.types.origins import Derived, Place, SourceText, origin_key
from quarry.types.results import InputText, Span
from quarry.types.transforms import ContentTransform, TransformOperation


def projection(owner=None, field="body", role="user", derived=False):
    if owner is None:
        owner = {"id": "same", "body": "  Straße  ", "title": "  Straße  "}
    origin = Place(0, field, 0, owner, InputText("memory.notes", "same", ("title", "body")))
    if derived:
        origin = Derived(
            owner[field], (origin,), array("q", (0, len(owner[field]), 0, 0, len(owner[field]), 0))
        )
    return Messages.message_parts(role, [owner[field]], origin), origin


@pytest.mark.parametrize("derived", [False, True])
@pytest.mark.parametrize("shared_owner", [False, True])
def test_projection_identity_shares_only_the_same_canonical_input_owner(derived, shared_owner):
    first, origin = projection(derived=derived)
    owner = origin.owner if isinstance(origin, Place) else origin.sources[0].owner
    second, _ = projection(owner if shared_owner else dict(owner), derived=derived)
    assert (_synthetic_key(first[0]) == _synthetic_key(second[0])) is shared_owner
    combined = Messages.combine((first, second, first))
    assert len(combined) == 3
    assert (combined[0].message_index == combined[1].message_index) is shared_owner
    for i in range(3):
        bound = combined._bind_span(Span(i, 2, 4))
        assert bound._source is (
            owner
            if i != 1 or shared_owner
            else second._root._origins[0].owner
            if not derived
            else second._root._origins[0].sources[0].owner
        )


@pytest.mark.parametrize("derived", [False, True])
def test_equal_wire_spans_do_not_cross_foreign_input_owners(derived):
    first, a = projection(derived=derived)
    second, b = projection(derived=derived)
    assert a != b and origin_key(a) != origin_key(b)
    span = SourceText(first[0].content, a, 0).span(2, 4)
    assert SourceText(second[0].content, b, 0).offsets(span) is None
    foreign = SourceText(second[0].content, b, 0).span(2, 3)
    assert not _is_covered(_cover_index([span]), foreign)


def test_batch_constructors_extend_origin_slots_for_every_added_message():
    first, origin = projection()
    combined = first._add_synthetic_messages(
        (
            ("system", "one", object(), None),
            ("assistant", "two", object(), None),
        )
    )
    assert combined._root._origins == (origin, None, None)
    assert combined._bind_span(Span(0, 2, 4))._source is origin.owner
    assert combined._bind_span(Span(2, 0, 3)).source is None


@pytest.mark.asyncio
@pytest.mark.parametrize("grafted", ["none", "chat", "constructed"])
async def test_transformed_input_model_spans_keep_local_text_and_original_field_offsets(grafted):
    first, origin = projection()
    transformed = await apply_content_transform(
        first,
        ContentTransform(
            (
                TransformOperation("strip"),
                TransformOperation("casefold"),
                TransformOperation("prefix", ">>"),
            )
        ),
    )
    if grafted != "none":
        header = (
            Messages.from_openai([{"role": "system", "content": "header"}])
            if grafted == "chat"
            else Messages().add_message("system", "header")
        )
        transformed = Messages.combine((header, transformed))[1:]
    local = transformed[0].content.index("ss")
    span = transformed._bind_span(Span(0, local, local + 2))
    assert span._source is origin.owner and span.source == origin.source
    assert (span.start, span.end) == (
        origin.owner["body"].index("ß"),
        origin.owner["body"].index("ß") + 1,
    )
    assert span._view is transformed[0]
    assert transformed.extract(span) == "ss"
    texts = _view_source_texts(Messages(), transformed)
    extracted = texts[0].span(local, local + 2)
    assert (extracted.start, extracted.end) == (span.start, span.end)
    assert texts[0].offsets(extracted) == (local, local + 2)


@pytest.mark.parametrize("index", [1, 8])
def test_input_projection_models_keep_strict_selected_local_indices(index):
    first, _ = projection()
    with pytest.raises(ValueError, match="index is outside"):
        first._bind_span(Span(index, 0, 1))


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_owner", [False, True])
async def test_shared_physical_replies_bind_spans_to_each_readers_input_owner(
    tmp_path, shared_owner
):
    from quarry.cache.async_cache import scan_scope
    from quarry.config import QuarryConfig
    from quarry.integrations.model_inference import ModelInferenceBackend
    from tests.unit.model_fakes import ModelAdapter, NoWaitLimiter, model_entry

    owner = {"id": "same", "body": "alice@example.com"}
    first, _ = projection(owner)
    other = owner if shared_owner else dict(owner)
    second, _ = projection(other)
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {"reader": model_entry(classify=0, spans=0, category="EMAIL")},
        },
        tmp_path,
        require_rules=False,
    )
    adapter = ModelAdapter()
    backend = ModelInferenceBackend(
        config, adapters={"fake": adapter}, rate_limiter=NoWaitLimiter()
    )
    with scan_scope():
        a = await backend.model_span_messages("reader", first, 0.5)
        b = await backend.model_span_messages("reader", second, 0.5)
    assert adapter.classifications == 1 and len(adapter.decoded_spans) == 2
    assert a.items[0].value._source is owner
    assert b.items[0].value._source is other
    assert a.items[0].value.score.value == b.items[0].value.score.value == 0.91
    assert (
        first.extract(a.items[0].value) == second.extract(b.items[0].value) == "alice@example.com"
    )


def test_union_retains_foreign_owners_with_identical_wire_coordinates():
    from quarry.integrations.model_expression import _union_spans
    from quarry.types.results import CollectionResult, ProducerRef, ResultItem

    first, a = projection()
    second, b = projection()
    spans = [
        SourceText(value[0].content, origin, 0).span(2, 4)
        for value, origin in [(first, a), (second, b)]
    ]
    collection = CollectionResult(
        "span_collection",
        None,
        ProducerRef("pattern", "p", "synthetic"),
        tuple(ResultItem(str(i), value) for i, value in enumerate(spans)),
    )
    result = _union_spans([collection, collection])
    assert len(result.items) == 2
    assert {id(item.value._source) for item in result.items} == {id(a.owner), id(b.owner)}


@pytest.mark.asyncio
async def test_located_source_items_do_not_spend_the_constructed_literal_budget():
    values = [projection({"id": str(index), "body": "source"})[0] for index in range(300)]
    combined = await Messages._combine_async(values, _mapped=True)
    assert len(combined) == 300
    assert len(combined._root._origins) == 300
    for index in (0, 255, 299):
        assert (
            combined._bind_span(Span(index, 0, 6))._source is values[index]._root._origins[0].owner
        )
    appended = combined.add_message("system", "tail")
    assert len(appended) == 301 and appended._root._origins[-1] is None


@pytest.mark.parametrize("role", ["system", "assistant"])
def test_distinct_role_projections_of_one_owner_keep_their_wire_variants(role):
    first, origin = projection()
    other, _ = projection(origin.owner, role=role)
    combined = Messages.combine((first, other))
    assert [message.role for message in combined] == ["user", role]
    assert all(combined._bind_span(Span(index, 2, 4))._source is origin.owner for index in range(2))
