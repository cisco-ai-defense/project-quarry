# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Atomic candidate-group admission and per-reader native source ownership."""

import asyncio
import uuid

from quarry.cache.keys import input_key
from quarry.config import CandidateSpanConfig
from quarry.engine.local_work import measured_local_work
from quarry.integrations.candidate_input import bind_candidate_input
from quarry.integrations.candidate_providers import (
    HostCandidateProvider,
    LocalCandidateProvider,
    require_candidate_provider,
)
from quarry.integrations.read_operations import current_model_read
from quarry.integrations.request_limits import raise_if_owned_evaluation_expired
from quarry.types.candidates import (
    CandidateBatchRequest,
    CandidateLimitError,
    CandidateReplyError,
    validate_candidate_batch_reply,
)
from quarry.types.exceptions import (
    CandidateInvalidInputError,
    CandidateInvalidOutputError,
    CandidatePartialReadError,
    CandidateProviderError,
    LLMError,
)
from quarry.types.results import CollectionResult, ProducerRef, ResultItem


class _PartialBatch(Exception):
    def __init__(self, reply):
        self.reply = reply
        super().__init__("candidate batch is incomplete")


@measured_local_work("prepare")
def _request(provider, source):
    return CandidateBatchRequest(
        provider.identity, source, provider.channels, provider.settings_digest
    )


@measured_local_work("prepare")
def _validated_reply(raw, request, limits):
    return validate_candidate_batch_reply(
        raw, request, limits, check_deadline=raise_if_owned_evaluation_expired
    )


async def read_candidate_spans(backend, alias, messages, threshold, *, timeout):
    model = backend._model(alias)
    config = model.spans
    assert type(config) is CandidateSpanConfig
    provider = require_candidate_provider(model, backend.adapters)
    observation = current_model_read()
    if observation is not None:
        observation.candidate_pending()
    input_failure = None
    failure_kind = None
    try:
        bindings = await bind_candidate_input(messages, config.limits)
    except CandidateLimitError:
        input_failure = "limit"
    except CandidateReplyError:
        input_failure = "input"
    finally:
        del messages
    if input_failure == "limit":
        raise CandidateLimitError("candidate input exceeds its configured finite limits")
    if input_failure is not None:
        raise CandidateInvalidInputError("candidate input requires exact native source parts")
    try:
        request = _request(provider, bindings.input)
        limits = config.limits

        async def prepare(raw):
            invalid = False
            over_limit = False
            try:
                reply = _validated_reply(raw, request, limits)
            except CandidateLimitError:
                over_limit = True
            except CandidateReplyError:
                invalid = True
            if over_limit:
                raise CandidateLimitError("candidate batch exceeds its configured finite limits")
            if invalid:
                raise CandidateInvalidOutputError(
                    "Candidate provider returned an invalid batch",
                    {"code": "invalid_output", "model": alias},
                )
            if not all(channel.complete for channel in reply.channels):
                # Share only validated numeric coverage with active readers; do
                # not admit a partial group or attach this reader's source graph.
                raise _PartialBatch(reply)
            if type(provider) is HostCandidateProvider:
                return await provider.filter_reply(request, reply, timeout=timeout)
            return reply

        async def consume(reply):
            raise_if_owned_evaluation_expired()
            selected = next(
                channel for channel in reply.channels if channel.channel == config.channel
            )
            if observation is not None:
                observation.candidate_input(
                    bindings.source_texts(), selected.coverage, complete=True
                )
            raise_if_owned_evaluation_expired()
            run = uuid.uuid4()
            items = []
            for ordinal, value in enumerate(selected.ranges):
                if ordinal % 256 == 0:
                    await asyncio.sleep(0)
                    raise_if_owned_evaluation_expired()
                if value.score >= threshold:
                    items.append(
                        ResultItem(str(uuid.uuid5(run, str(ordinal))), bindings.bind(value))
                    )
            return CollectionResult(
                "span_collection",
                model.category,
                ProducerRef("model", alias, str(run)),
                tuple(items),
            )

        return await backend._call(
            alias,
            model.model,
            provider,
            "candidate_batch",
            model,
            # A callback receives a fresh deep native snapshot for every attempt.
            # Even deliberate object.__setattr__ must not rewrite our expected identity.
            lambda: (
                CandidateBatchRequest(
                    request.provider, request.input, request.channels, request.settings_digest
                ),
            ),
            key=("candidate-batch-v1", request.digest),
            timeout=timeout,
            prepare=prepare,
            read=consume,
            local=type(provider) is LocalCandidateProvider,
            request_settings_digest=input_key(
                "candidate-settings-v1",
                provider.identity.namespace,
                provider.identity.version,
                provider.identity.batch,
                provider.settings_digest,
                limits,
            ),
        )
    except _PartialBatch as error:
        raise_if_owned_evaluation_expired()
        if observation is not None:
            selected = next(
                channel for channel in error.reply.channels if channel.channel == config.channel
            )
            observation.candidate_input(bindings.source_texts(), selected.coverage, complete=False)
        failure_kind = "partial_read"
    except CandidateLimitError:
        raise
    except CandidateReplyError:
        failure_kind = "invalid_output"
    except LLMError as error:
        code = error.details.get("code")
        if code in {"model_call_limit", "timeout"}:
            raise
        failure_kind = code if code in {"invalid_output", "partial_read"} else "provider_error"
    finally:
        bindings.close()
    kind = {
        "invalid_output": CandidateInvalidOutputError,
        "partial_read": CandidatePartialReadError,
    }.get(failure_kind, CandidateProviderError)
    raise kind("Candidate batch could not be admitted", {"code": failure_kind, "model": alias})
