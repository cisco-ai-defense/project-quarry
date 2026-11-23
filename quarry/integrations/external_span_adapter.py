# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Per-reader external-span admission; retained responses contain no source graphs."""

import asyncio
import uuid
from dataclasses import replace

from quarry.config import ExternalSpanConfig
from quarry.integrations.message_readability import require_model_messages
from quarry.integrations.model_window_input import join_window_input, json_window_input
from quarry.integrations.read_operations import current_model_read
from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.exceptions import (
    ExternalSpanInvalidOutputError,
    ExternalSpanPartialReadError,
    ExternalSpanProviderError,
    LLMError,
)
from quarry.types.external_spans import (
    ExternalSpanLimitError,
    ExternalSpanReplyError,
    external_span_input_digest,
    validate_external_span_reply,
)
from quarry.types.results import CollectionResult, ProducerRef, ResultItem, Score


class _PartialReply(Exception):
    def __init__(self, reply):
        self.reply = reply
        super().__init__("external span reply is incomplete")


class _InvalidReply(Exception):
    pass


def _error(kind, alias, code, message):
    return kind(f"Inference model '{alias}' {message}", {"code": code, "model": alias})


async def read_external_spans(backend, alias, messages, threshold, *, timeout):
    """Require full inspection before any verdict, including positive/negated reads."""
    model = backend._model(alias)
    config = model.spans
    assert isinstance(config, ExternalSpanConfig)
    config = replace(config)
    raise_if_evaluation_expired()
    await require_model_messages(messages)
    observation = current_model_read()
    if observation is not None:
        observation.unavailable(slot="input")
    render = json_window_input if config.renderer == "json" else join_window_input
    rendering = await render(messages, [])
    source = rendering.source
    if len(source.text) > config.limits.max_input_characters:
        raise ExternalSpanLimitError("external span input character limit exceeded")
    expected_digest = external_span_input_digest(source.text)
    raise_if_evaluation_expired()

    def prepare(raw):
        try:
            reply = validate_external_span_reply(raw, len(source.text), config.limits)
            if reply.input_digest != expected_digest:
                raise ExternalSpanReplyError("external reply belongs to another request")
        except ExternalSpanReplyError:
            raise _InvalidReply("invalid external span reply") from None
        if not reply.complete:
            # Failure occurs before cache admission. Existing active waiters receive
            # bounded numeric coverage; a later reader must issue its own request.
            raise _PartialReply(reply)
        return reply

    async def consume(reply):
        raise_if_evaluation_expired()
        if observation is not None:
            observation.external_input(source, reply.coverage, complete=True)
        items = []
        run = uuid.uuid4()
        for ordinal, detected in enumerate(reply.spans):
            if ordinal % 256 == 0:
                await asyncio.sleep(0)
                raise_if_evaluation_expired()
            if detected.score < threshold:
                continue
            try:
                span = replace(
                    source.span(detected.start, detected.end), score=Score(detected.score)
                )
            except (TypeError, ValueError):
                raise _InvalidReply("external detection has no valid source binding") from None
            items.append(ResultItem(str(uuid.uuid5(run, str(ordinal))), span))
        raise_if_evaluation_expired()
        return CollectionResult(
            "span_collection",
            model.category,
            ProducerRef("model", alias, str(run)),
            tuple(items),
        )

    try:
        return await backend._call(
            alias,
            model.model,
            backend.adapters[model.adapter],
            "external_spans",
            model,
            lambda: (source.text,),
            key=("external-spans-v1", source.text),
            timeout=timeout,
            prepare=prepare,
            read=consume,
        )
    except _PartialReply as exc:
        raise_if_evaluation_expired()
        if observation is not None:
            observation.external_input(source, exc.reply.coverage, complete=False)
        raise _error(
            ExternalSpanPartialReadError, alias, "partial_read", "external span read is incomplete"
        ) from None
    except _InvalidReply:
        raise _error(
            ExternalSpanInvalidOutputError,
            alias,
            "invalid_output",
            "returned invalid external spans",
        ) from None
    except LLMError as exc:
        if exc.details.get("code") in {"model_call_limit", "timeout"}:
            raise
        raise _error(
            ExternalSpanProviderError, alias, "provider_error", "external span provider failed"
        ) from None
