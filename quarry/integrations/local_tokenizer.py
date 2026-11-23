# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Lazy local artifact encoding and whole-request counting under one worker owner."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import replace
from math import isfinite
from typing import TYPE_CHECKING

from quarry.cache.keys import input_key, object_identity
from quarry.integrations.local_tokenizer_artifact import (
    _LoadedTokenizer,
    load_local_tokenizer,
    validate_local_text,
)
from quarry.integrations.local_tokenizer_input import snapshot_request
from quarry.integrations.local_tokenizer_template import render_local_request
from quarry.integrations.local_tokenizer_worker import _Job, _LocalWorkerOwner
from quarry.integrations.tokenizer_width import VerifiedTokenizerWidth, attest_loaded_width
from quarry.types.exceptions import LocalTokenizerInvalidInputError
from quarry.types.local_tokenizer import (
    LocalTokenizerConfig,
    LocalTokenizerEncoding,
    LocalTokenizerValidationError,
)
from quarry.types.tokenizer_width import (
    CappedTokenCount,
    TokenizerWidthClaim,
    TokenizerWidthValidationError,
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

    @staticmethod
    def _claim(claim: TokenizerWidthClaim) -> TokenizerWidthClaim:
        if type(claim) is not TokenizerWidthClaim:
            raise LocalTokenizerInvalidInputError("Local tokenizer width requires a native claim")
        try:
            return replace(claim)
        except TokenizerWidthValidationError:
            raise LocalTokenizerInvalidInputError(
                "Local tokenizer width claim is invalid"
            ) from None

    def _attested(
        self,
        config: LocalTokenizerConfig,
        claim: TokenizerWidthClaim,
        loaded: _LoadedTokenizer,
        job: _Job,
    ) -> VerifiedTokenizerWidth | None:
        job.checkpoint()
        proofs = getattr(self._artifacts, "proofs", None)
        if proofs is None:
            proofs = self._artifacts.proofs = OrderedDict()
        key = (config, claim)
        if key in proofs:
            proofs.move_to_end(key)
            result = proofs[key]
            if result is None or result.is_compatible(config, claim):
                return result
        result = attest_loaded_width(config, claim, loaded, job)
        job.checkpoint()
        # Only bounded configuration metadata; no request, encoding or native
        # artifact reference survives here. Each worker retains at most2 proofs.
        proofs[key] = result
        if len(proofs) > 2:
            proofs.popitem(last=False)
        return result

    async def attest_width(
        self,
        config: LocalTokenizerConfig,
        claim: TokenizerWidthClaim,
        *,
        timeout: float | None = None,
    ) -> VerifiedTokenizerWidth | None:
        """Attest actual pinned native state off-loop, or return no supported proof."""
        config, seconds = self._config(config, timeout)
        claim = self._claim(claim)

        def attest(job: _Job):
            return self._attested(config, claim, self._loaded(config, job), job)

        return await self._owner.run(attest, timeout=seconds)

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

    async def count_request_capped(
        self,
        config: LocalTokenizerConfig,
        claim: TokenizerWidthClaim,
        messages: list,
        tools: list,
        *,
        cap: int,
        timeout: float | None = None,
    ) -> CappedTokenCount:
        """Prove strict overflow, or return the canonical exact whole-request count.

        Every message/tool is snapshotted and rendered before any proof. Input
        and native-work admission still apply. When allocation admission cannot
        also guarantee the configured output-token limit, encode canonically so
        overflow cannot hide that error. Exact counting uses its original path.
        """
        config, seconds = self._config(config, timeout)
        claim = self._claim(claim)
        if type(cap) is not int or not 0 <= cap < 1 << 63:
            raise LocalTokenizerInvalidInputError("Local tokenizer cap must be a bounded integer")

        def count(job: _Job):
            detached_messages, detached_tools = snapshot_request(messages, tools, config, job)
            text = render_local_request(config, detached_messages, detached_tools, job)
            text = validate_local_text(text, config, job)
            loaded = self._loaded(config, job)
            upper = loaded.bounds.admit(text, job)
            framing = config.framing.total(len(detached_messages))
            proof = self._attested(config, claim, loaded, job)
            if proof is not None and upper <= config.limits.max_tokens:
                width = claim.max_source_characters_per_token
                lower = (len(text) + width - 1) // width + framing
                if lower > cap:
                    job.checkpoint()
                    return CappedTokenCount(cap, lower, proof_fingerprint=proof.fingerprint)
            encoding = loaded.encode(text, config, job)
            actual = len(encoding.token_ids) + framing
            return CappedTokenCount(cap, actual, exact_count=actual)

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

    def capped_fingerprint(self, config: InferenceModelConfig, *, alias: str, cap: int) -> str:
        """Bind an actual native reply to adapter lifetime, alias, revisions and cap."""
        self._owner._process()
        tokens = config.tokens
        if (
            tokens is None
            or type(tokens.local) is not LocalTokenizerConfig
            or type(tokens.source_width) is not TokenizerWidthClaim
            or tokens.local.model_revision != config.revision
            or type(alias) is not str
            or not 1 <= len(alias) <= 256
            or not alias.isprintable()
            or type(cap) is not int
            or not 0 <= cap < 1 << 63
        ):
            raise LocalTokenizerInvalidInputError("Local capped tokenizer binding is invalid")
        local, _ = self._config(tokens.local, None)
        claim = self._claim(tokens.source_width)
        return input_key(
            "local-tokenizer-width-request-v1",
            object_identity(self),
            alias,
            config.revision,
            local.count_settings_sha256,
            claim,
            cap,
        ).hex()

    async def tokenize_capped(
        self,
        config: InferenceModelConfig,
        messages: list,
        tools: list,
        *,
        alias: str,
        cap: int,
        timeout: float | None = None,
    ) -> CappedTokenCount:
        fingerprint = self.capped_fingerprint(config, alias=alias, cap=cap)
        assert config.tokens is not None
        local, claim = config.tokens.local, config.tokens.source_width
        assert local is not None and claim is not None
        result = await self.count_request_capped(
            local, claim, messages, tools, cap=cap, timeout=timeout
        )
        if result.is_exact:
            return result
        return replace(result, proof_fingerprint=fingerprint)

    def close(self) -> None:
        """Reject new jobs; accepted jobs and worker artifacts retain actual ownership."""
        self._owner.close()

    def report(self) -> dict:
        return self._owner.report()
