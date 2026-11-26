# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Synthetic adapters distinguish planned heads and physical model revisions."""

import pytest

from quarry.cache.async_cache import result_cache
from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.types.exceptions import ConfigurationError
from quarry.types.messages import Messages
from quarry.types.results import Score, Span
from tests.unit.model_fakes import ModelAdapter, NoWaitLimiter, model_entry, passthrough_builder


class HeadAdapter(ModelAdapter):
    supports_token_head = True

    def __init__(self):
        super().__init__()
        self.head_requests = []
        self.head_decodes = []

    async def classify_messages(self, config, messages, tools, *, timeout=None, token_head=False):
        self.head_requests.append(token_head)
        return await super().classify_messages(config, messages, tools, timeout=timeout)

    async def decode_spans(self, config, raw, messages, threshold, *, token_head=False):
        self.head_decodes.append(token_head)
        return [Span(0, 0, 3, Score(0.9))]


class LegacyAdapter(ModelAdapter):
    async def decode_spans(self, config, raw, messages, threshold):
        return [Span(0, 0, 3, Score(0.9))]


def backend(tmp_path, models, adapter):
    config = QuarryConfig.from_mapping(
        {"config_version": 2, "models": models}, tmp_path, require_rules=False
    )
    return ModelInferenceBackend(config, adapters={"fake": adapter}, rate_limiter=NoWaitLimiter())


@pytest.mark.asyncio
@pytest.mark.parametrize("span_first", [False, True])
async def test_compatible_native_class_and_span_readers_share_a_requested_head(
    tmp_path, span_first
):
    adapter = HeadAdapter()
    runtime = backend(
        tmp_path,
        {
            "classes": model_entry(classify=0),
            "segments": model_entry(spans=0, category="SYNTHETIC"),
        },
        adapter,
    )
    messages = Messages.from_openai([{"role": "user", "content": "synthetic text"}])
    result_cache.clear()
    if span_first:
        spans = await runtime.model_span_messages("segments", messages, 0.5)
        value = await runtime.model_classify_messages("classes", messages)
    else:
        value = await runtime.model_classify_messages("classes", messages)
        spans = await runtime.model_span_messages("segments", messages, 0.5)
    assert value == 0.8
    assert [messages.extract(item.value) for item in spans.items] == ["syn"]
    assert adapter.classifications == 1
    assert adapter.head_requests == [True]
    assert adapter.head_decodes == [True]


@pytest.mark.asyncio
async def test_legacy_adapter_signatures_keep_the_existing_shared_request(tmp_path):
    adapter = LegacyAdapter()
    runtime = backend(
        tmp_path,
        {
            "classes": model_entry(classify=0),
            "segments": model_entry(spans=0, category="SYNTHETIC"),
        },
        adapter,
    )
    messages = Messages.from_openai([{"role": "user", "content": "synthetic text"}])
    result_cache.clear()
    assert await runtime.model_classify_messages("classes", messages) == 0.8
    result = await runtime.model_span_messages("segments", messages, 0.5)
    assert len(result.items) == 1 and adapter.classifications == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "left_revision,right_revision,calls",
    [(None, None, 1), ("r1", "r1", 1), ("r1", "r2", 2), (None, "r1", 2)],
)
async def test_revision_identity_shares_only_the_same_physical_revision(
    tmp_path, left_revision, right_revision, calls
):
    models = {name: model_entry(classify=0) for name in ["left", "right"]}
    for name, revision in [("left", left_revision), ("right", right_revision)]:
        if revision is not None:
            models[name]["revision"] = revision
    adapter = LegacyAdapter()
    runtime = backend(tmp_path, models, adapter)
    messages = Messages.from_openai([{"role": "user", "content": "synthetic text"}])
    result_cache.clear()
    assert await runtime.model_classify_messages("left", messages) == 0.8
    assert await runtime.model_classify_messages("right", messages) == 0.8
    assert adapter.classifications == calls


@pytest.mark.parametrize("revision", [True, 3, "", "   "])
def test_revision_must_be_a_nonempty_string(tmp_path, revision):
    model = model_entry(classify=0)
    model["revision"] = revision
    with pytest.raises(ConfigurationError):
        backend(tmp_path, {"reader": model}, LegacyAdapter())


def test_unset_revision_preserves_public_parent_request_identities(tmp_path):
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            "models": {
                "reader": model_entry(
                    classify=0, score=True, tokens=True, spans=0, category="SYNTHETIC"
                )
            },
        },
        tmp_path,
        require_rules=False,
    )
    runtime = ModelInferenceBackend(
        config,
        adapters={"fake": LegacyAdapter()},
        message_builders={"test": passthrough_builder},
        rate_limiter=NoWaitLimiter(),
    )
    model = config.models["reader"]
    assert "revision" not in model.identity_dict()
    # Captured from the public parent before adding the optional revision field.
    assert {
        method: runtime._settings_digest(method, model).hex()
        for method in ["score", "classify_messages", "tokenize", "transform"]
    } == {
        "score": "bc745d7775a6d83d0b9d67add0c58d4c9681428049570475779a1f35398f9e9a",
        "classify_messages": "55dd7d977edbcf1541c7723d4e8b37fbb82ae3280feff9d96403f6599862d7cf",
        "tokenize": "3bafcae5579465aa58f83f4e6fe6221d75c51105328b9eab70492f11845b9ef1",
        "transform": "6988a4a865cb793ed836ce2876e1613660e547c7aef578c9460f52cd4f0fbc9b",
    }
