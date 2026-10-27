# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""External DSL functions executed by the scanner's detector bridge.

Compiled builtins are documented beside their implementing conversation method or compiler
handler. Public name and arity checks come from the generated language specification rather
than another registry in this module.
"""

import asyncio
import math
from typing import Annotated, Any

from quarry.language.metadata import DSLParameter, Use, dsl_function
from quarry.types import messages as _messages
from quarry.types.exceptions import ModelError


async def _classify(ctx, model_name: str, text: str, class_index: int) -> float:
    """Return the max probability at ``class_index`` of classifying ``(model, text)``.

    The configured model backend caches the raw response per request, so reads for
    different class indices share one inference. It rejects unconfigured models and
    discards responses rejected by the reader.
    """
    return await _inference_backend(ctx).model_classification(
        model_name,
        text,
        read=lambda result: _extract_max_probability(result, class_index=class_index),
    )


def _extract_max_probability(classification_result, class_index: int = 1) -> float:
    """Max probability at ``class_index`` across all chunks.

    Fails closed (raises ``ModelError``) rather than silently returning 0.0 on a malformed
    result, so a broken/misindexed classification surfaces as an evaluation error instead of
    a false negative: no chunks, a chunk with no probabilities, ``class_index`` out of range
    for any chunk, or a non-finite probability."""
    chunks = getattr(classification_result, "chunks", None)
    if not chunks:
        raise ModelError("model_classification returned no chunks")
    max_prob: float | None = None
    for chunk in chunks:
        probs = getattr(chunk, "probabilities", None)
        if not probs:
            raise ModelError("model_classification chunk has no probabilities")
        if class_index >= len(probs) or class_index < 0:
            raise ModelError(
                f"class_index {class_index} out of range for {len(probs)} output classes"
            )
        value = probs[class_index]
        # Reject bool explicitly (``True``/``False`` are ints, so they'd otherwise pass as
        # 1.0/0.0), non-numeric, non-finite, and out-of-[0,1] — a probability outside the
        # documented contract must fail closed, not silently drive a match decision.
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ModelError(f"model_classification returned a non-numeric probability: {value!r}")
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ModelError(f"model_classification probability out of range [0,1]: {value!r}")
        max_prob = value if max_prob is None else max(max_prob, value)
    return max_prob if max_prob is not None else 0.0


def _require_class_index(function_name: str, class_index: int | None) -> int:
    """Require an explicit, valid output-head index before model classification.

    A model can expose several heads and the
    "positive" one is model-specific, so silently defaulting to head 1 could read the wrong
    output. The rule must name the head. Validated as a non-negative int (``bool`` rejected)."""
    if class_index is None:
        raise ModelError(f"{function_name} requires an explicit class_index")
    if isinstance(class_index, bool) or not isinstance(class_index, int) or class_index < 0:
        raise ModelError(f"Model class_index must be a non-negative integer, got {class_index!r}")
    return class_index


@dsl_function(
    return_type="boolean",
    allowed_uses=(Use.CONDITION, Use.COMPREHENSION_BODY),
    resource_group="model",
    compiler_tags=("score_external",),
)
async def model_scan(
    ctx: Annotated[Any, DSLParameter(hidden=True)],
    model_name: Annotated[str, DSLParameter(name="model", literal_only=True)],
    text: Annotated[str, DSLParameter(field_only=True)],
    threshold: Annotated[
        float | None,
        DSLParameter(accepted_types=("number",), required=True, literal_only=True),
    ] = None,
    class_index: Annotated[
        int | None,
        DSLParameter(
            accepted_types=("integer",),
            required=True,
            literal_only=True,
            minimum=0,
        ),
    ] = None,
) -> bool:
    """Return whether a classifier score is greater than a required threshold.

    ``model`` names the configured classifier. ``text`` is the field being
    classified, and ``class_index`` selects the model output whose probability is tested.
    """
    class_index = _require_class_index("model_scan", class_index)
    score = await _classify(ctx, str(model_name), str(text), class_index)
    return score > (threshold if threshold is not None else 0.5)


@dsl_function(
    return_type="number",
    resource_group="model",
    compiler_tags=("score_external",),
)
async def model_score(
    ctx: Annotated[Any, DSLParameter(hidden=True)],
    model_name: Annotated[str, DSLParameter(name="model", literal_only=True)],
    text: Annotated[str, DSLParameter(field_only=True)],
    class_index: Annotated[
        int | None,
        DSLParameter(
            accepted_types=("integer",),
            required=True,
            literal_only=True,
            minimum=0,
        ),
    ] = None,
) -> float:
    """Return a classifier probability for the selected output class.

    ``model`` names the configured classifier. Compare the returned value with
    a numeric threshold in the rule. ``class_index`` is mandatory in the DSL.
    """
    class_index = _require_class_index("model_score", class_index)
    return await _classify(ctx, str(model_name), str(text), class_index)


def _inference_backend(ctx):
    """The configured ``ModelInferenceBackend`` on the ResourceManager, or fail closed.

    llm_score / llm_input_tokens ground through the configured backend (adapter routing +
    constitution scoring + tokenizer counting); a rule reaching here without one is a
    configuration error, not a silent pass."""
    rm = getattr(ctx, "resource_manager", None)
    backend = getattr(rm, "inference_backend", None) if rm is not None else None
    if backend is None:
        raise ModelError(
            "model inference backend not initialized. Configure an inference model "
            "under 'models' for llm_score / llm_input_tokens."
        )
    return backend


def _select_model_input(ctx, selector: dict | None = None):
    """Resolve a typed model-input ``selector`` to the value passed to the backend.

    The selector is the inference atom's folded fields (see ``_score_model_input`` in the
    compiler). Semantics:

    - default / ``conversation()`` (no selector) → the whole ``ctx.conversation`` (its
      messages AND the request's offered tools).
    - a Messages selection (``messages`` / a role view, optionally sliced) → that canonical
      ``Messages`` view itself, returned by reference (the backend normalizes it to a
      tools-less conversation without reparsing). Carries NO tools.
    - constructed (``message(...)[.add_message(...)]``) → a ``Conversation`` from the folded
      synthetic turns (genuinely new content, no tools).

    The value is passed to the backend AS-IS — the canonical objects are never round-tripped
    through the parser. A Messages selection that is empty, or not a standalone-valid OpenAI
    sequence (e.g. a tool result without the assistant call it answers), fails closed here —
    before any builder/adapter/provider call — rather than being widened or scored as an
    empty request.
    """
    selector = selector or {}
    constructed = selector.get("constructed_input") or ()
    if constructed:
        # A constructed model input is a Messages value (no tools) built from the folded turns —
        # NOT a Conversation. The turns are synthetic literal/dynamic content, so this is genuine
        # new data; the backend wraps it tools-less by reference.
        return _messages.Messages.from_openai(
            [{"role": role, "content": content} for role, content in constructed]
        )
    pipeline = selector.get("pipeline")
    if pipeline is None:
        return ctx.conversation
    # A Messages selection: start at the root Messages and apply each ordered step to the result
    # of the previous — role filters compose (user→assistant = empty), and slice/filter order is
    # preserved. Every step returns a reference-backed view (no copy/reparse); only conversation()
    # carries tools, so a Messages selection never does.
    view = ctx.conversation.messages
    for step in pipeline:
        view = getattr(view, step[1]) if step[0] == "role" else view[step[1] : step[2]]
    try:
        view.assert_self_contained()
    except ValueError as exc:
        raise ModelError(str(exc)) from exc
    return view


@dsl_function(
    return_type="number",
    resource_group="score",
    compiler_tags=("score_external", "whole_conversation_inference", "chunk_scalar"),
)
async def llm_score(
    ctx: Annotated[Any, DSLParameter(hidden=True)],
    model_name: Annotated[str, DSLParameter(name="model", literal_only=True)],
    constitution: Annotated[
        Any,
        DSLParameter(
            accepted_types=("string", "text_reference"),
            name="constitution",
            literal_only=True,
        ),
    ],
    selector: Annotated[
        dict | None,
        DSLParameter(accepted_types=("model_input",), name="input", required=True),
    ] = None,
) -> float:
    """Score model input against policy text and return a number from 0 through 1.

    The policy may be written directly or loaded by name with ``text_ref("name")``.

    Internal:

    The input parameter is the atom's typed model-input selection. The backend wraps the policy
    plus selected conversation and normalizes the positive-token probability. A named text
    reference is resolved through the resource manager before the backend call.
    """
    backend = _inference_backend(ctx)
    resolved = ctx.resource_manager.resolve_text(constitution)
    conversation = _select_model_input(ctx, selector)
    return await backend.model_text_score(
        str(model_name),
        resolved,
        conversation,
    )


def _compare_chunk(value: float, op: str, threshold: float) -> bool:
    """Apply one folded per-chunk threshold test. Only the ordering operators are reachable —
    the compiler rejects ``=``/``!=`` on a per-chunk scalar."""
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    raise ModelError(f"unsupported per-chunk comparison {op!r}")


async def _chunk_scalar(ctx, backend, scalar: str, model_name: str, constitution, chunk):
    """One whole-chunk scalar measurement. Each arm must map a whole chunk to a number; adding
    a new arm here is the runtime half of adding an entry to the compiler's ``_CHUNK_SCALARS``."""
    if scalar == "llm_score":
        return await backend.model_text_score(
            model_name,
            constitution,
            chunk,
        )
    if scalar == "llm_input_tokens":
        return await backend.model_input_tokens(model_name, chunk)
    raise ModelError(f"unsupported chunked scalar {scalar!r}")


async def llm_chunk_reduce(
    ctx,
    model_name: str,
    scalar: str,
    constitution,
    max_input_tokens: int,
    combine: str,
    inner_op: str = "",
    inner_number: float = 0.0,
    selector: dict | None = None,
) -> float:
    """Split the conversation into token-bounded chunks, measure ``scalar`` on each, and reduce
    the per-chunk values to one number (expensive - cached).

    Runtime half of the two chunked rule forms; the compiler folds each into a single atom
    (``_Compiler._desugar_chunked_score`` / ``_desugar_chunked_count``) and applies the outer
    comparator to whatever this returns:

    ===============  ===================================  ====================================
    ``combine``      rule form                            returns
    ===============  ===================================  ====================================
    ``max``/``min``  ``any``/``all`` quantifier           the reduced scalar
    ``count``        ``count([c for c in … if …]) <op> N``  how many chunks pass the inner test
    ===============  ===================================  ====================================

    Which of max/min makes the quantifier form exact depends on the comparator's DIRECTION as
    well as any/all — the compiler decides and folds it, so this function just applies it.

    NOT registered in ``EXTERNAL_FUNCTIONS``: a rule cannot call it by name, because its
    semantics only make sense together with the comparisons the compiler folded in.

    The per-chunk measurements are awaited CONCURRENTLY. Bounding is already handled below this
    layer: every backend call passes through ``ModelUsage.run`` → the ``LLMRateLimiter``
    semaphore (``llm_rate_limits.max_concurrent``) and its rolling-window throughput limit, and
    the remaining deadline is recomputed *after* a permit is acquired, so a chunk that queued for
    a while still gets a correctly shrunk timeout. Exceptions propagate (fail closed): a partial
    reduction would silently under-report a positive chunk.
    """
    backend = _inference_backend(ctx)
    resolved = ctx.resource_manager.resolve_text(constitution)
    # Split the SELECTED input (default conversation() → whole transcript + tools; a Messages
    # selection → that transcript, no tools), not always the whole request.
    chunks = await backend.model_split_conversation(
        str(model_name),
        _select_model_input(ctx, selector),
        int(max_input_tokens),
    )
    if not chunks:
        # Empty domain: return the reduction's IDENTITY element, which makes the folded outer
        # comparator produce the right vacuous answer on its own (∃ over ∅ = False, ∀ = True) for
        # ANY scalar range. Using 0.0/1.0 here would only work for scores in [0,1] and would give
        # the wrong vacuous verdict for a token count.
        if combine == "count":
            return 0.0
        return -math.inf if combine == "max" else math.inf
    values = await asyncio.gather(
        *(_chunk_scalar(ctx, backend, scalar, str(model_name), resolved, chunk) for chunk in chunks)
    )
    if combine == "count":
        return float(sum(1 for value in values if _compare_chunk(value, inner_op, inner_number)))
    return float(min(values) if combine == "min" else max(values))


@dsl_function(
    return_type="integer",
    resource_group="token",
    compiler_tags=("score_external", "whole_conversation_inference", "chunk_scalar"),
)
async def llm_input_tokens(
    ctx: Annotated[Any, DSLParameter(hidden=True)],
    model_name: Annotated[str, DSLParameter(name="model", literal_only=True)],
    selector: Annotated[
        dict | None,
        DSLParameter(accepted_types=("model_input",), name="input", required=True),
    ] = None,
) -> int:
    """Return the model-specific token count for the supplied model input.

    Internal:

    The configured inference backend performs tokenization. ``conversation()`` includes offered
    tools, while a message-surface selection contains only the transcript.
    """
    backend = _inference_backend(ctx)
    return await backend.model_input_tokens(str(model_name), _select_model_input(ctx, selector))


@dsl_function(
    return_type="string",
    resource_group="completion",
    compiler_tags=("complete_comparison",),
)
async def llm_complete(
    ctx: Annotated[Any, DSLParameter(hidden=True)],
    model_name: Annotated[str, DSLParameter(name="model", literal_only=True)],
    selector: Annotated[
        dict | None,
        DSLParameter(accepted_types=("model_input",), name="input", required=True),
    ] = None,
) -> str:
    """Generate text from model input for comparison with a literal string or regex.

    Write this function on the left of ``==`` or ``~``. A bare call is not a true-or-false
    condition.

    Internal:

    The compiler folds the comparison operator and literal into one Boolean external atom; the
    scanner applies that comparison to the generated text.
    """
    backend = _inference_backend(ctx)
    return await backend.model_text_completion(
        str(model_name),
        _select_model_input(ctx, selector),
    )


def intent_baseline(ctx) -> str:
    """The first user message — the original intent a topic prober scores drift against."""
    conv = getattr(ctx, "conversation", None)
    if conv is None:
        return ""
    user_messages = getattr(conv, "user_messages", None) or []
    if not user_messages:
        return ""
    return str(getattr(user_messages[0], "content", "") or "")


def _topic_prober(ctx):
    """The topic prober registered on the ResourceManager. No silent stub fallback: a rule
    reaching here without a backend is a fail-closed error (the loader's prober gate should
    have rejected it, but grounding must not quietly fabricate a keyword heuristic)."""
    rm = getattr(ctx, "resource_manager", None)
    prober = getattr(rm, "topic_prober", None) if rm is not None else None
    if prober is None:
        raise ModelError(
            "topic prober not initialized. Register a backend on the ResourceManager, or "
            "set allow_stub_probers=True to opt into the keyword stub."
        )
    return prober


def _capability_classifier(ctx):
    """The capability classifier registered on the ResourceManager. No silent stub fallback
    (see :func:`_topic_prober`)."""
    rm = getattr(ctx, "resource_manager", None)
    clf = getattr(rm, "capability_classifier", None) if rm is not None else None
    if clf is None:
        raise ModelError(
            "capability classifier not initialized. Register a backend on the ResourceManager, "
            "or set allow_stub_probers=True to opt into the keyword stub."
        )
    return clf


@dsl_function(
    return_type="number",
    resource_group="topic_prober",
    compiler_tags=("field_first_external", "score_external"),
)
async def topic_drift(
    ctx: Annotated[Any, DSLParameter(hidden=True)],
    text: Annotated[str, DSLParameter(field_only=True)],
) -> float:
    """Return how far ``text`` has moved from the first user message, from 0 through 1.

    A result of 0 means on-topic and 1 means fully off-topic. This function requires a
    configured topic-analysis backend.
    """
    return _topic_prober(ctx).drift_score(intent_baseline(ctx), str(text))


@dsl_function(
    return_type="boolean",
    allowed_uses=(Use.CONDITION, Use.COMPREHENSION_BODY),
    resource_group="topic_prober",
    compiler_tags=("field_first_external",),
)
async def off_topic(
    ctx: Annotated[Any, DSLParameter(hidden=True)],
    text: Annotated[str, DSLParameter(field_only=True)],
) -> bool:
    """Return whether ``text`` has moved too far from the first user message.

    This function requires a configured topic-analysis backend.
    """
    return _topic_prober(ctx).off_topic(intent_baseline(ctx), str(text))


@dsl_function(
    return_type="boolean",
    allowed_uses=(Use.CONDITION, Use.COMPREHENSION_BODY),
    resource_group="capability_classifier",
)
async def capability_tags(
    ctx: Annotated[Any, DSLParameter(hidden=True)],
    tag: Annotated[str, DSLParameter(literal_only=True)],
    text: Annotated[str, DSLParameter(field_only=True)],
) -> bool:
    """Return whether ``text`` describes the capability named by ``tag``.

    The built-in classifier recognizes ``external_communication``, ``private_data_access``,
    ``code_execution``, ``state_modification``, and ``untrusted_ingress``. A configured
    classifier may support additional tags. This function requires a configured capability
    classifier.
    """
    return _capability_classifier(ctx).has_tag(str(text), str(tag))


@dsl_function(
    return_type="boolean",
    allowed_uses=(Use.CONDITION, Use.COMPREHENSION_BODY),
    resource_group="capability_classifier",
    compiler_tags=("field_first_external",),
)
async def unrequested_action(
    ctx: Annotated[Any, DSLParameter(hidden=True)],
    text: Annotated[str, DSLParameter(field_only=True)],
) -> bool:
    """Return whether ``text`` describes an action not requested in the first user message.

    Pass a tool name, description, or arguments field as ``text``. This function requires a
    configured capability classifier.
    """
    return _capability_classifier(ctx).unrequested_capability(str(text), intent_baseline(ctx))


EXTERNAL_FUNCTIONS = {
    "llm_score": llm_score,
    "llm_input_tokens": llm_input_tokens,
    "llm_complete": llm_complete,
    "model_scan": model_scan,
    "model_score": model_score,
    "topic_drift": topic_drift,
    "off_topic": off_topic,
    "capability_tags": capability_tags,
    "unrequested_action": unrequested_action,
}
