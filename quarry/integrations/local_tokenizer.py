# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Lazy local artifact encoding and whole-request counting under one worker owner."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import replace
from math import isfinite
from typing import TYPE_CHECKING

from quarry.integrations.local_tokenizer_artifact import (
    _LoadedTokenizer,
    load_local_tokenizer,
    validate_local_text,
)
from quarry.integrations.local_tokenizer_input import snapshot_request
from quarry.integrations.local_tokenizer_template import render_local_request
from quarry.integrations.local_tokenizer_worker import _Job, _LocalWorkerOwner
from quarry.types.exceptions import LocalTokenizerInvalidInputError
from quarry.types.local_tokenizer import (
    LocalTokenizerConfig,
    LocalTokenizerEncoding,
    LocalTokenizerValidationError,
)

if TYPE_CHECKING:
    from quarry.config import InferenceModelConfig


class LocalTokenizerAdapter:
    """At most2 native workers and8 accepted jobs per backend adapter.

    Each worker retains at most2 pinned artifacts; no message, rendered input,
    encoding or count is cached here. Worker-local eviction runs outside the
    owner's mutex. Closing retains accepted work until actual native termination;
    worker exit then releases its artifact cache. Construction imports neither
    tokenizers nor Jinja and opens no artifact. Only an actual call loads bytes.
    """

    def __init__(self):
        self._owner = _LocalWorkerOwner()
        self._artifacts = threading.local()

    @staticmethod
    def _config(config: object, timeout: float | None) -> tuple[LocalTokenizerConfig, float]:
        if type(config) is not LocalTokenizerConfig:
            raise LocalTokenizerInvalidInputError("Local tokenizer requires native configuration")
        # Clone/revalidate even objects forged after construction. Configuration
        # checks import no optional libraries and never load or render anything.
        try:
            config = replace(config)
        except LocalTokenizerValidationError:
            raise LocalTokenizerInvalidInputError(
                "Local tokenizer configuration is invalid"
            ) from None
        seconds = config.limits.max_wall_seconds
        if timeout is not None:
            if (
                type(timeout) not in (int, float)
                or timeout <= 0
                or type(timeout) is float
                and not isfinite(timeout)
            ):
                raise LocalTokenizerInvalidInputError(
                    "Local tokenizer timeout must be finite and positive"
                )
            seconds = min(seconds, timeout)
        return config, seconds

    def _loaded(self, config: LocalTokenizerConfig, job: _Job) -> _LoadedTokenizer:
        job.checkpoint()
        cache = getattr(self._artifacts, "cache", None)
        if cache is None:
            cache = self._artifacts.cache = OrderedDict()
        loaded = cache.get(config)
        if loaded is not None:
            cache.move_to_end(config)
            return loaded
        loaded = load_local_tokenizer(config, job)
        job.checkpoint()
        cache[config] = loaded
        if len(cache) > 2:
            cache.popitem(last=False)
        return loaded

    async def encode_text(
        self, config: LocalTokenizerConfig, text: str, *, timeout: float | None = None
    ) -> LocalTokenizerEncoding:
        """Actual IDs and original code-point offsets, with no implicit framing IDs."""
        config, seconds = self._config(config, timeout)

        def encode(job: _Job):
            validated = validate_local_text(text, config, job)
            return self._loaded(config, job).encode(validated, config, job)

        return await self._owner.run(encode, timeout=seconds)

    async def count_request(
        self,
        config: LocalTokenizerConfig,
        messages: list,
        tools: list,
        *,
        timeout: float | None = None,
    ) -> int:
        """Render the whole supported request off-loop and add framing exactly once."""
        config, seconds = self._config(config, timeout)

        def count(job: _Job):
            detached_messages, detached_tools = snapshot_request(messages, tools, config, job)
            text = render_local_request(config, detached_messages, detached_tools, job)
            encoding = self._loaded(config, job).encode(text, config, job)
            return len(encoding.token_ids) + config.framing.total(len(detached_messages))

        return await self._owner.run(count, timeout=seconds)

    async def tokenize(
        self,
        config: InferenceModelConfig,
        messages: list,
        tools: list,
        *,
        timeout: float | None = None,
    ) -> dict[str, int]:
        tokens = config.tokens
        local = None if tokens is None else getattr(tokens, "local", None)
        if type(local) is not LocalTokenizerConfig or local.model_revision != config.revision:
            raise LocalTokenizerInvalidInputError("Local tokenizer model revision does not match")
        assert tokens is not None
        count = await self.count_request(local, messages, tools, timeout=timeout)
        return {tokens.count_field: count}

    def close(self) -> None:
        """Reject new jobs; accepted jobs and worker artifacts retain actual ownership."""
        self._owner.close()

    def report(self) -> dict:
        return self._owner.report()
