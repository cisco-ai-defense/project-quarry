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

"""Evaluate the compiler's closed model-expression payload at scan time."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, cast

from quarry.cache.keys import input_key
from quarry.cache.message_values import BindingIdentity
from quarry.engine.atoms import (
    MAX_CONSTRUCTED_MESSAGES,
    MAX_CONSTRUCTED_TEXT_CHARACTERS,
    MAX_MESSAGE_MAP_CHARACTERS,
    MAX_MESSAGE_MAP_ITEMS,
    ModelExpression,
)
from quarry.engine.matchers import (
    MISSING,
    PURE_REGEX,
    Modifiers,
    RegexBackend,
    compare,
    field_access,
    field_text,
    match_contains,
    match_contains_any,
    match_regex,
    truth,
)
from quarry.engine.pacing import Pacer
from quarry.engine.witness import MAX_JUDGE_REASONS, ModelResultEvidence
from quarry.integrations.adapter_message_transforms import apply_adapter_transform
from quarry.integrations.content_transforms import (
    apply_content_transform,
    compiled_content_transform,
)
from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.adapter_transforms import (
    adapter_transform_reference,
    adapter_transform_reference_parts,
)
from quarry.types.exceptions import MODEL_REFUSED, LLMError
from quarry.types.media import require_readable_media
from quarry.types.messages import Message, Messages, _synthetic_key
from quarry.types.text_resources import TextRef

# The operator that keeps a comparison's meaning when its operands swap sides.
_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}

# The operations a model or provider can refuse; any other operation's refusal is an error.
REFUSABLE_OPERATIONS = frozenset({"score", "complete"})

# The binding that holds the field texts one map comprehension item has read.
_ITEM_TEXTS = "\x00item_texts"
_CONSTRUCTION_SCOPE = "\x00construction_scope"
_BINDING_SCOPES = "\x00binding_scopes"


@dataclass
class ModelExpressionTrace:
    """Ephemeral values used to build one bounded atom-level explanation."""

    calls: list[tuple[str, str, tuple[tuple[str, Any], ...], Any]] = field(default_factory=list)
    expression_value: Any = None
    operator: str | None = None
    threshold: Any = None
    split_scores: list[float] | None = None
    parameters: list[tuple[str, Any]] = field(default_factory=list)
    # The span of the transcript each score read: ``{"constructed": True}`` for a constructed
    # input, or, when it read conversation fields, ``{"constructed": True, "pieces": [...]}`` —
    # each piece a message's or call's position and field, never its text — so evidence can
    # point at what was scored without holding text that a path entry or telemetry would copy.
    input_span: dict[str, Any] | None = None
    # The (position, field) pieces the input being built has read so far. Folded into
    # input_span and reset by _record_call, so pieces never leak from one model call's input
    # into the next.
    input_pieces: list[dict[str, Any]] = field(default_factory=list)
    # The ``context`` a ``split`` repeated in every chunk, and per chunk its context span
    # (or ``None``) and its own span.
    split_context: Messages | None = None
    split_spans: list[tuple[dict[str, Any] | None, dict[str, Any]]] | None = None
    # Each refused call, as (model alias, operation, reason).
    refusals: list[tuple[str, str, str]] = field(default_factory=list)
    # Per split chunk, whether its score call was refused, so a fallback value is not
    # presented as a measured score.
    split_refused: list[bool] | None = None
    # Explicit user-facing explanations only; never provider private reasoning.
    judge_reasons: list[tuple[int, str, str]] = field(default_factory=list)
    omitted_judge_reasons: int = 0

    def add_judge_reason(self, ordinal: int, alias: str, reason: str) -> None:
        """Retain reasons in evaluation order with one shared bound for calls and chunks."""

        if len(self.judge_reasons) < MAX_JUDGE_REASONS:
            self.judge_reasons.append((ordinal, alias, reason))
        else:
            self.omitted_judge_reasons += 1


def _finite(value: Any) -> Any:
    """Reject non-finite model and arithmetic results before they reach evidence or JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("model expression produced a non-finite number")
    return value


@dataclass(frozen=True, slots=True)
class _BoundMessage:
    """One scoped message together with the selected view that owns its variant."""

    message: Message
    view: Messages


def _message_value_key(expression: ModelExpression, bindings: dict[str, Any], event) -> tuple:
    """Name a source expression and the live bindings it actually reads, within this rule.

    The digest preserves scalar types (unlike Python's True == 1), while binding wrappers
    compare by identity and keep their objects alive only as long as this rule's memo.
    """

    names = set()
    pending = [(expression, frozenset())]
    while pending:
        node, local_names = pending.pop()
        if node.kind == "binding" and node.value not in local_names:
            names.add(str(node.value))
        for index, child in enumerate(node.children):
            child_names = local_names
            if node.kind in {"map", "message_map", "message_select", "split_any"} and index:
                child_names = local_names | {str(node.value)}
            pending.append((child, child_names))
    identities = []
    for name in sorted(names):
        value = bindings.get(name, event.source if event is not None else None)
        if isinstance(value, _BoundMessage):
            identities.append((name, BindingIdentity(value.message)))
            identities.append((name, BindingIdentity(value.view)))
        else:
            identities.append((name, BindingIdentity(value)))
    scopes = bindings.get(_BINDING_SCOPES, {})
    value_scope = max((scopes.get(name, ()) for name in names), key=len, default=())
    return (
        input_key(expression.content_key()),
        tuple(identities),
        value_scope,
    )


async def _evaluate_owned_value(expression, context, *, event, bindings, trace, regex_backend):
    """Evaluate once, retaining the owning selection for an indexed or bound message."""

    if expression.kind == "index":
        base = await evaluate_model_expression(
            expression.children[0],
            context,
            event=event,
            bindings=bindings,
            trace=trace,
            regex_backend=regex_backend,
        )
        if base is MISSING:
            return MISSING, None
        index = int(expression.value)
        try:
            value = base[index]
        except (IndexError, KeyError, TypeError):
            return MISSING, None
        return value, base._select_positions((index,)) if isinstance(base, Messages) else None
    if expression.kind == "binding":
        bound = bindings.get(str(expression.value))
        if isinstance(bound, _BoundMessage):
            return bound.message, bound.view
    value = await evaluate_model_expression(
        expression,
        context,
        event=event,
        bindings=bindings,
        trace=trace,
        regex_backend=regex_backend,
    )
    if isinstance(value, Message):
        transcript = context.conversation.messages
        ordinal = value.message_index
        if transcript._contains_root_ordinal(ordinal) and transcript._message_at(ordinal) is value:
            return value, transcript._view_ordinals((ordinal,))
    return value, None


async def evaluate_model_expression(
    expression: ModelExpression,
    context,
    *,
    event=None,
    bindings: dict[str, Any] | None = None,
    trace: ModelExpressionTrace | None = None,
    regex_backend: RegexBackend | None = None,
) -> Any:
    """Evaluate one compiler-validated expression against the active scan context."""

    regex_backend = regex_backend or PURE_REGEX
    environment = bindings or {}
    kind = expression.kind

    if kind in {"map", "join", "field_text"}:
        return await _evaluate_model_text(
            expression, context, event, environment, trace, regex_backend
        )
    if kind == "literal":
        return expression.value
    if kind == "absent":
        return None
    if kind == "parameter":
        name = str(expression.value)
        if name not in context.parameters:
            raise ValueError(f"model expression parameter {name!r} is not bound")
        value = context.parameters[name]
        if trace is not None and all(existing != name for existing, _ in trace.parameters):
            trace.parameters.append((name, _parameter_trace_value(context, name, value)))
        return value
    if kind == "messages":
        return context.conversation.messages
    if kind == "message_view":
        return getattr(context.conversation.messages, str(expression.value))
    if kind == "tools":
        return context.conversation.tools
    if kind == "binding":
        name = str(expression.value)
        if name in environment:
            value = environment[name]
            return value.message if isinstance(value, _BoundMessage) else value
        if event is None:
            raise ValueError(f"model expression binding {name!r} has no current event")
        return event.source
    if kind == "text_ref":
        return context.resource_manager.resolve_text(TextRef(str(expression.value)))
    if kind == "string_patterns":
        return tuple(str(child.value) for child in expression.children), str(expression.value)
    if kind == "text_equals":
        # `=` on a field of a map comprehension item, as a rule step reads it: the text of a
        # value that is not missing or null, or for `= null` whether the value is missing or
        # null.
        value = await evaluate_model_expression(
            expression.children[0],
            context,
            event=event,
            bindings=environment,
            trace=trace,
            regex_backend=regex_backend,
        )
        if expression.value is None:
            return value is MISSING or value is None
        return value is not MISSING and value is not None and str(value) == expression.value

    if kind in {"transform", "local_transform", "model_transform"}:
        source = await evaluate_model_expression(
            expression.children[0],
            context,
            event=event,
            bindings=environment,
            trace=trace,
            regex_backend=regex_backend,
        )
        if not isinstance(source, Messages):
            raise TypeError("content transform source must be Messages")
        if not isinstance(expression.value, str):
            raise TypeError("compiled content transform identity must be a string")
        if kind == "transform":
            return await apply_content_transform(
                source, compiled_content_transform(expression.value)
            )
        name, alias = adapter_transform_reference_parts(expression.value, kind == "model_transform")
        backend = context.resource_manager.inference_backend
        if backend is None:
            raise RuntimeError("transform backend is not initialized")
        definition = backend.config.transforms.get(name)
        if adapter_transform_reference(name, definition) != expression.value:
            raise ValueError("transform configuration differs from its compiled reference")
        part_count = 0

        def record_part() -> None:
            nonlocal part_count
            part_count += 1

        async def transform() -> Messages:
            return await apply_adapter_transform(
                source, backend, name, _on_part=record_part, _message_value=True
            )

        memo = getattr(context, "message_value_memo", None)
        if kind == "local_transform" and memo is not None:
            key = _message_value_key(expression, environment, event)
            result = await memo.evaluate(key, transform, scope=key[-1])
        else:
            result = await transform()
        if kind == "model_transform" and trace is not None and part_count:
            # Evidence records logical reads, including cache hits, as other model calls do.
            # Neither the source parts nor generated text leave this evaluation trace.
            trace.calls.append(
                (alias, "transform", (("transform", name), ("parts", part_count)), len(result))
            )
        return result

    if kind == "conditional":
        predicate = await evaluate_model_expression(
            expression.children[0],
            context,
            event=event,
            bindings=environment,
            trace=trace,
            regex_backend=regex_backend,
        )
        if predicate is not MISSING and not isinstance(predicate, bool):
            raise TypeError("conditional value predicate must be boolean")
        branch = expression.children[1] if predicate is True else expression.children[2]
        return await evaluate_model_expression(
            branch,
            context,
            event=event,
            bindings=environment,
            trace=trace,
            regex_backend=regex_backend,
        )

    if kind == "concat":
        parts = []
        for child in expression.children:
            parts.append(
                await evaluate_model_expression(
                    child,
                    context,
                    event=event,
                    bindings=environment,
                    trace=trace,
                    regex_backend=regex_backend,
                )
            )
        if any(not isinstance(part, Messages) for part in parts):
            raise TypeError("message combination operands must be Messages")
        if context.limits.enforce_limits:
            count = sum(len(part) for part in parts)
            limit = context.limits.max_comprehension_size
            if count > limit:
                raise ValueError(
                    f"combined message selection has {count} items, exceeding "
                    f"max_comprehension_size={limit}"
                )
        raise_if_evaluation_expired()
        return await Messages._combine_async(cast(list[Messages], parts))

    if kind in {"message_select", "message_map"}:
        source = await evaluate_model_expression(
            expression.children[0],
            context,
            event=event,
            bindings=environment,
            trace=trace,
            regex_backend=regex_backend,
        )
        projection_trace = ModelExpressionTrace()

        async def project() -> Messages:
            return await _evaluate_message_selection(
                expression, source, context, event, environment, projection_trace, regex_backend
            )

        memo = getattr(context, "message_value_memo", None)
        if memo is None:
            result = await project()
        else:
            key = _message_value_key(expression, environment, event)
            result = await memo.evaluate(
                key, project, reads=projection_trace.parameters, scope=key[-1]
            )
        if trace is not None:
            for name, parameter in projection_trace.parameters:
                if all(existing != name for existing, _ in trace.parameters):
                    trace.parameters.append((name, parameter))
        return result

    if kind == "field":
        base = await evaluate_model_expression(
            expression.children[0],
            context,
            event=event,
            bindings=environment,
            trace=trace,
            regex_backend=regex_backend,
        )
        if base is MISSING:
            return MISSING
        return field_access(base, str(expression.value))
    if kind == "extract_span":
        messages, span, left, right = [
            await evaluate_model_expression(
                child,
                context,
                event=event,
                bindings=environment,
                trace=trace,
                regex_backend=regex_backend,
            )
            for child in expression.children
        ]
        if not isinstance(messages, Messages):
            raise TypeError("messages.extract() receiver must be Messages")
        return messages.extract(span, left, right)
    if kind == "index":
        base = await evaluate_model_expression(
            expression.children[0],
            context,
            event=event,
            bindings=environment,
            trace=trace,
            regex_backend=regex_backend,
        )
        if base is MISSING:
            return MISSING
        try:
            return base[int(cast(int, expression.value))]
        except (IndexError, KeyError, TypeError):
            return MISSING
    if kind == "as_messages":
        message, owner = await _evaluate_owned_value(
            expression.children[0],
            context,
            event=event,
            bindings=environment,
            trace=trace,
            regex_backend=regex_backend,
        )
        if message is MISSING:
            return context.conversation.messages[0:0]
        if not isinstance(message, Message) or owner is None:
            raise ValueError("a promoted message must retain its owning selection")
        return owner
    if kind == "slice":
        base = await evaluate_model_expression(
            expression.children[0],
            context,
            event=event,
            bindings=environment,
            trace=trace,
            regex_backend=regex_backend,
        )
        start = expression.children[1].value
        stop = expression.children[2].value
        return base[slice(start, stop)]
    if kind in {"message", "add_message"}:
        constructors = []
        current = expression
        while current.kind == "add_message":
            constructors.append(current)
            current = current.children[0]
        if current.kind != "message":
            raise ValueError("constructed message chain has an invalid receiver")
        constructors.append(current)
        constructors.reverse()

        specifications = []
        construction_source = None
        for index, constructor in enumerate(constructors):
            if index and index % 16 == 0:
                await asyncio.sleep(0)
            raise_if_evaluation_expired()
            offset = 0 if constructor.kind == "message" else 1
            role = constructor.children[offset].value
            content_expression = constructor.children[offset + 1]
            # A field is read from its owner here, so the explain record can point at the owner.
            field_read = content_expression.kind == "field"
            content, owner_view = await _evaluate_owned_value(
                content_expression.children[0] if field_read else content_expression,
                context,
                event=event,
                bindings=environment,
                trace=trace,
                regex_backend=regex_backend,
            )
            owner = content
            if field_read and owner is not MISSING:
                content = field_access(owner, str(content_expression.value))
            # Missing and null fields omit a turn. An explicitly empty string is still an
            # authored message and remains distinct in model input and cache identity.
            if content is MISSING or content is None:
                continue
            if not isinstance(content, str):
                raise TypeError("model message content must resolve to a string")
            provenance = None
            if field_read:
                _record_pieces(trace, context, [(owner, content)], content_expression)
                if isinstance(owner, Message) and content_expression.value == "content":
                    provenance = owner._content_provenance()
                    if any(segment.sources or segment.parts for segment in provenance):
                        candidate = None if owner_view is None else owner_view._root._source_root
                        if candidate is None:
                            raise ValueError("copied content must retain its owning source")
                        if construction_source is not None and candidate is not construction_source:
                            raise ValueError("copied content must share one canonical source")
                        construction_source = candidate
            specifications.append(
                (
                    str(role),
                    content,
                    (id(constructor), *environment.get(_CONSTRUCTION_SCOPE, (0,))),
                    provenance,
                )
            )
        raise_if_evaluation_expired()
        base = (
            Messages()
            if construction_source is None
            else Messages._from_items(
                (),
                construction_source._tool_call_types,
                construction_source._max_nesting_depth,
                source_root=construction_source,
            )
        )
        return base._add_synthetic_messages(tuple(specifications))

    if kind in {"refused", "if_refused"}:
        try:
            value = await evaluate_model_expression(
                expression.children[0],
                context,
                event=event,
                bindings=bindings,
                trace=trace,
                regex_backend=regex_backend,
            )
        except LLMError as exc:
            if exc.details.get("code") != MODEL_REFUSED:
                raise
            if kind == "refused":
                return True
            return await evaluate_model_expression(
                expression.children[1],
                context,
                event=event,
                bindings=bindings,
                trace=trace,
                regex_backend=regex_backend,
            )
        return False if kind == "refused" else value
    if kind == "model_call":
        return await _evaluate_model_call(
            expression, context, event, environment, trace, regex_backend
        )

    if kind == "split_any":
        chunks = await evaluate_model_expression(
            expression.children[0],
            context,
            event=event,
            bindings=environment,
            trace=trace,
            regex_backend=regex_backend,
        )
        if chunks is MISSING:
            chunks = ()
        variable = str(expression.value)
        child_traces = [ModelExpressionTrace() for _ in chunks]
        if trace is not None and all(isinstance(chunk, Messages) for chunk in chunks):
            trace.split_spans = [
                _chunk_spans(chunk, trace.split_context, context.conversation.messages)
                for chunk in chunks
            ]
        tasks = [
            asyncio.create_task(
                evaluate_model_expression(
                    expression.children[1],
                    context,
                    event=event,
                    bindings={**environment, variable: chunk},
                    trace=child_trace,
                    regex_backend=regex_backend,
                )
            )
            for chunk, child_trace in zip(chunks, child_traces, strict=True)
        ]
        try:
            values = await asyncio.gather(*tasks)
            result = any(truth(value) for value in values)
            if trace is not None:
                for child_trace in child_traces:
                    offset = len(trace.calls)
                    for ordinal, alias, reason in child_trace.judge_reasons:
                        trace.add_judge_reason(offset + ordinal, alias, reason)
                    trace.omitted_judge_reasons += child_trace.omitted_judge_reasons
                    trace.calls.extend(child_trace.calls)
                    trace.refusals.extend(child_trace.refusals)
                    for parameter in child_trace.parameters:
                        if parameter not in trace.parameters:
                            trace.parameters.append(parameter)
                trace.split_scores = [
                    float(child_trace.expression_value) for child_trace in child_traces
                ]
                trace.split_refused = [bool(child_trace.refusals) for child_trace in child_traces]
                trace.expression_value = result
                trace.operator, trace.threshold = _split_comparator(expression.children[1], context)
            return result
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    if kind == "not":
        return not truth(
            await evaluate_model_expression(
                expression.children[0],
                context,
                event=event,
                bindings=environment,
                trace=trace,
                regex_backend=regex_backend,
            )
        )
    if kind in {"and", "or"}:
        left = await evaluate_model_expression(
            expression.children[0],
            context,
            event=event,
            bindings=environment,
            trace=trace,
            regex_backend=regex_backend,
        )
        if left is not MISSING and not isinstance(left, bool):
            raise TypeError(f"{kind} left operand must be boolean")
        left = truth(left)
        if (kind == "and" and not left) or (kind == "or" and left):
            return left
        right = await evaluate_model_expression(
            expression.children[1],
            context,
            event=event,
            bindings=environment,
            trace=trace,
            regex_backend=regex_backend,
        )
        if right is not MISSING and not isinstance(right, bool):
            raise TypeError(f"{kind} right operand must be boolean")
        return truth(right)
    if kind in {"+", "-", "*"}:
        left, right = await _evaluate_children(
            expression, context, event, environment, trace, regex_backend
        )
        if left is MISSING or right is MISSING:
            return MISSING
        result = left + right if kind == "+" else left - right if kind == "-" else left * right
        return _finite(result)
    if kind == "compare":
        left, right = await _evaluate_children(
            expression, context, event, environment, trace, regex_backend
        )
        result = (
            False
            if left is MISSING or right is MISSING
            else compare(left, str(expression.value), right)
        )
        if trace is not None:
            operator = str(expression.value)
            dynamic, static = expression.children
            if _contains_model_call(static) and not _contains_model_call(dynamic):
                dynamic, static, left, right = static, dynamic, right, left
                operator = _FLIP.get(operator, operator)
            trace.expression_value = (
                None if left is MISSING else _trace_operand(dynamic, left, context)
            )
            trace.operator = operator
            trace.threshold = None if right is MISSING else _trace_operand(static, right, context)
        return result
    if kind == "string_compare":
        left, right = await _evaluate_children(
            expression, context, event, environment, trace, regex_backend
        )
        operator = str(expression.value)
        if left is MISSING or right is MISSING or left is None or right is None:
            result = False
        elif operator == "contains":
            result = match_contains(str(left), str(right), regex=regex_backend)
        elif operator == "in":
            if expression.children[0].kind == "string_patterns":
                patterns, flags = left
                result = match_contains_any(
                    str(right),
                    patterns,
                    Modifiers(
                        nocase=flags[0] == "1",
                        fullword=flags[1] == "1",
                        base64=flags[2] == "1",
                        regex=flags[3] == "1",
                    ),
                    regex=regex_backend,
                )
            else:
                result = match_contains(str(right), str(left), regex=regex_backend)
        elif operator in {"~", "matches"}:
            result = match_regex(str(left), str(right), regex=regex_backend)
        else:
            raise ValueError(f"unknown string comparison {operator!r}")
        if trace is not None:
            trace.expression_value = (
                None if left is MISSING else _trace_operand(expression.children[0], left, context)
            )
            trace.operator = operator
            trace.threshold = (
                None if right is MISSING else _trace_operand(expression.children[1], right, context)
            )
        return result

    raise ValueError(f"unknown model expression operation {kind!r}")


async def _evaluate_message_selection(
    expression, source, context, event, environment, trace, regex_backend
):
    """Compute one bounded map value with scoped callbacks and source ownership."""

    kind = expression.kind
    if kind == "message_select" and not isinstance(source, Messages):
        raise TypeError("message selection source must be Messages")
    if source is MISSING or source is None:
        return context.conversation.messages[:0]
    if not isinstance(source, Sequence) or isinstance(source, str | bytes | bytearray):
        raise TypeError("message map source must be a structured collection")
    if kind == "message_map" and len(source) > MAX_MESSAGE_MAP_ITEMS:
        raise ValueError(f"message map source exceeds {MAX_MESSAGE_MAP_ITEMS} items")
    limit = context.limits.max_comprehension_size
    if context.limits.enforce_limits and len(source) > limit:
        raise ValueError(
            f"message selection source has {len(source)} items, exceeding "
            f"max_comprehension_size={limit}"
        )
    variable = str(expression.value)
    positions: list[int] = []
    parts: list[Messages] = []
    output_count = output_characters = 0
    constructed_keys: set[tuple[str, object]] = set()
    constructed_characters = 0
    pacer = Pacer()
    for position, item in enumerate(source):
        raise_if_evaluation_expired()
        if pacer.due():
            await asyncio.sleep(0)
        item_environment = {
            **environment,
            variable: _BoundMessage(item, source[position : position + 1])
            if isinstance(source, Messages)
            else item,
        }
        item_scope = (*environment.get(_CONSTRUCTION_SCOPE, ()), id(expression), position)
        item_environment[_BINDING_SCOPES] = {
            **environment.get(_BINDING_SCOPES, {}),
            variable: item_scope,
        }
        item_environment[_CONSTRUCTION_SCOPE] = item_scope
        memo = getattr(context, "message_value_memo", None)
        try:
            predicate = await evaluate_model_expression(
                expression.children[1],
                context,
                event=event,
                bindings=item_environment,
                trace=trace,
                regex_backend=regex_backend,
            )
            if predicate is not MISSING and not isinstance(predicate, bool):
                raise TypeError("message selection predicate must be boolean")
            if predicate is True:
                if kind == "message_select":
                    positions.append(position)
                    continue
                part = await evaluate_model_expression(
                    expression.children[2],
                    context,
                    event=event,
                    bindings=item_environment,
                    trace=trace,
                    regex_backend=regex_backend,
                )
                if not isinstance(part, Messages):
                    raise TypeError("message map values must be Messages")
                output_count += len(part)
                if output_count > MAX_MESSAGE_MAP_ITEMS:
                    raise ValueError(f"message map output exceeds {MAX_MESSAGE_MAP_ITEMS} items")
                if context.limits.enforce_limits and output_count > limit:
                    raise ValueError(f"message map output exceeds max_comprehension_size={limit}")
                for selected in part:
                    raise_if_evaluation_expired()
                    if pacer.due():
                        await asyncio.sleep(0)
                    output_characters += len(selected.content)
                    if output_characters > MAX_MESSAGE_MAP_CHARACTERS:
                        raise ValueError(
                            f"message map output exceeds {MAX_MESSAGE_MAP_CHARACTERS} characters"
                        )
                    key = _synthetic_key(selected)
                    if key is not None and key not in constructed_keys:
                        constructed_keys.add(key)
                        constructed_characters += len(selected.content)
                        if len(constructed_keys) > MAX_CONSTRUCTED_MESSAGES:
                            raise ValueError(
                                f"message map exceeds {MAX_CONSTRUCTED_MESSAGES} constructed messages"
                            )
                        if constructed_characters > MAX_CONSTRUCTED_TEXT_CHARACTERS:
                            raise ValueError(
                                "message map constructed text exceeds "
                                f"{MAX_CONSTRUCTED_TEXT_CHARACTERS} characters"
                            )
                parts.append(part)
        finally:
            if memo is not None:
                memo.release_scope(item_scope)
    if kind == "message_select":
        return source._select_positions(positions)
    return await Messages._combine_async(parts, _mapped=True)


def _trace_operand(expression: ModelExpression, value: Any, context) -> Any:
    """Keep numeric operands and authored literals, omitting computed text from evidence."""

    if type(value) is int and not -(2**63) <= value < 2**63:
        return str(value)
    if expression.kind == "parameter":
        return _parameter_trace_value(context, str(expression.value), value)
    if isinstance(value, str) and expression.kind not in {"literal", "integer"}:
        return "«redacted»"
    return value


def _parameter_trace_value(context, name: str, value: Any) -> Any:
    """Expose only values proven to be declared numeric parameters."""

    spec = context.parameter_specs.get(name)
    if getattr(spec, "type", None) == "number":
        return value
    return "«redacted»"


def _split_comparator(expression: ModelExpression, context) -> tuple[str | None, Any]:
    """Normalize a split predicate to ``dynamic score <op> static threshold``."""

    if expression.kind != "compare":
        return None, None
    left, right = expression.children
    if right.kind == "literal":
        return str(expression.value), right.value
    if right.kind == "parameter":
        name = str(right.value)
        value = context.parameters[name]
        return str(expression.value), _parameter_trace_value(context, name, value)
    if _contains_model_call(right):
        operator = _FLIP.get(str(expression.value), str(expression.value))
        if left.kind == "literal":
            return operator, left.value
        if left.kind == "parameter":
            name = str(left.value)
            value = context.parameters[name]
            return operator, _parameter_trace_value(context, name, value)
    return None, None


def _span_of(messages: Messages, transcript: Messages) -> dict[str, Any]:
    """Return the span of the scanned transcript an input covers, first message to last.

    A mixed input retains its source envelope and is also marked as containing constructed text.
    """

    if not (messages.shares_root_with(transcript) or messages.shares_source_with(transcript)):
        return {"constructed": True}
    result = messages.span.to_dict()
    if not messages.shares_root_with(transcript):
        result["constructed"] = True
    return result


def _chunk_spans(
    chunk: Messages, split_context: Messages | None, transcript: Messages
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return one chunk's context span, when the split repeated one, and its own span."""

    if not split_context:
        return None, _span_of(chunk, transcript)
    own = chunk[len(split_context) :]
    return _span_of(split_context, transcript), _span_of(own, transcript)


def _contains_model_call(expression: ModelExpression) -> bool:
    return expression.kind in {"model_call", "model_transform"} or any(
        _contains_model_call(child) for child in expression.children
    )


async def _evaluate_children(
    expression, context, event, bindings, trace, regex_backend
) -> tuple[Any, Any]:
    """Evaluate the two operands of an arithmetic, comparison, or string operation."""

    left = await evaluate_model_expression(
        expression.children[0],
        context,
        event=event,
        bindings=bindings,
        trace=trace,
        regex_backend=regex_backend,
    )
    right = await evaluate_model_expression(
        expression.children[1],
        context,
        event=event,
        bindings=bindings,
        trace=trace,
        regex_backend=regex_backend,
    )
    return left, right


async def _evaluate_model_call(expression, context, event, bindings, trace, regex_backend) -> Any:
    """Dispatch one validated ModelRef operation through the configured backend.

    Only a score or completion can be refused; a refused one is recorded, with no result,
    before its refusal propagates.
    """

    alias, operation = str(expression.value).split("\x1f", 1)
    arguments = [
        await evaluate_model_expression(
            child,
            context,
            event=event,
            bindings=bindings,
            trace=trace,
            regex_backend=regex_backend,
        )
        for child in expression.children
    ]
    if any(argument is MISSING for argument in arguments):
        return MISSING
    if isinstance(arguments[0], Messages) and not arguments[0]:
        # An input with no message (a message that does not exist, or a message() chain whose
        # every turn had no content): the same "no input, no call".
        return MISSING
    try:
        return await _dispatch_model_call(expression, alias, operation, arguments, context, trace)
    except LLMError as exc:
        if exc.details.get("code") != MODEL_REFUSED:
            raise
        if operation not in REFUSABLE_OPERATIONS:
            raise LLMError(
                f"model().{operation}() cannot be refused: {exc.message}",
                {**exc.details, "code": "provider_error"},
            ) from exc
        if trace is not None:
            _record_call(trace, context, expression, alias, operation, arguments, None)
            trace.refusals.append((alias, operation, str(exc.details.get("reason"))))
        raise


async def _dispatch_model_call(expression, alias, operation, arguments, context, trace) -> Any:
    """Run one model operation on its evaluated arguments."""

    backend = context.resource_manager.inference_backend
    if backend is None:
        raise RuntimeError("model inference backend is not initialized")

    # Custom backends receive canonical selections too. Guard every selection, including
    # split context, before handing control to host code.
    for argument in arguments:
        if isinstance(argument, Messages):
            await require_readable_media(argument)

    if operation == "score":
        messages, constitution, *optional = arguments
        # Legacy/custom backends keep their numeric contract. The built-in backend can
        # retain the same request's explicitly enabled explanation for explain evidence.
        from quarry.integrations.model_inference import ModelInferenceBackend

        numeric = backend.model_score_messages
        tools = optional[0] if optional else None
        reason = None
        if (
            trace is not None
            and isinstance(backend, ModelInferenceBackend)
            and isinstance(numeric, MethodType)
            and numeric.__func__ is ModelInferenceBackend.model_score_messages
            and isinstance(backend.model_judge_messages, MethodType)
            and backend.model_judge_messages.__func__ is ModelInferenceBackend.model_judge_messages
        ):
            judged = await backend.model_judge_messages(
                alias, constitution, messages, tools, timeout=context.limits.rule_timeout
            )
            result, reason = judged.score, judged.reason
        else:
            result = await numeric(
                alias, constitution, messages, tools, timeout=context.limits.rule_timeout
            )
        result = _finite(result)
        _record_call(trace, context, expression, alias, operation, arguments, result)
        if trace is not None and reason is not None:
            trace.add_judge_reason(len(trace.calls) - 1, alias, reason)
        return result
    if operation == "classify":
        messages, *optional = arguments
        result = _finite(
            await backend.model_classify_messages(
                alias,
                messages,
                optional[0] if optional else None,
                timeout=context.limits.rule_timeout,
            )
        )
        _record_call(trace, context, expression, alias, operation, arguments, result)
        return result
    if operation == "spans":
        messages, threshold = arguments
        if not isinstance(messages, Messages):
            raise TypeError("model().spans() input must be Messages")
        messages.assert_self_contained()
        return await backend.model_span_messages(
            alias,
            messages,
            threshold,
            timeout=context.limits.rule_timeout,
        )
    if operation == "complete":
        messages, *optional = arguments
        # With a response schema the result also carries its JSON content as `.json`.
        result = await backend.model_complete(
            alias,
            messages,
            optional[0] if optional else None,
            limits=context.limits,
            tool_call_types=context.tool_call_types,
            timeout=context.limits.rule_timeout,
        )
        _record_call(trace, context, expression, alias, operation, arguments, result.content)
        return result
    if operation == "input_tokens":
        messages, *optional = arguments
        result = await backend.model_input_tokens_messages(
            alias,
            messages,
            optional[0] if optional else None,
            timeout=context.limits.rule_timeout,
        )
        _record_call(trace, context, expression, alias, operation, arguments, result)
        return result
    if operation == "split":
        messages, max_input_tokens, *optional = arguments
        overlap, split_context = [*optional, None, None][:2]
        result = await backend.model_split_messages(
            alias,
            messages,
            max_input_tokens,
            overlap=overlap or 0,
            context=split_context,
            timeout=context.limits.rule_timeout,
        )
        if trace is not None:
            trace.split_context = split_context
        _record_call(trace, context, expression, alias, operation, arguments, len(result))
        return result
    raise ValueError(f"unsupported model operation {operation!r}")


def _record_call(trace, context, expression, alias, operation, arguments, result) -> None:
    """Record bounded call metadata while omitting conversation collections themselves."""

    if trace is None:
        return
    projected: list[tuple[str, Any]] = []
    if operation == "score":
        if isinstance(arguments[0], Messages):
            span = _span_of(arguments[0], context.conversation.messages)
            if span.get("constructed") and trace.input_pieces:
                span = {"constructed": True, "pieces": trace.input_pieces}
            trace.input_span = span
        constitution_expression = expression.children[1]
        constitution = (
            f"text_ref:{constitution_expression.value}"
            if constitution_expression.kind == "text_ref"
            else "«redacted»"
            if constitution_expression.kind == "parameter"
            else arguments[1]
        )
        projected.append(("constitution", constitution))
    if operation == "split":
        overlap, split_context = [*arguments[2:], None, None][:2]
        projected.append(("max_input_tokens", arguments[1]))
        projected.append(("overlap", overlap or 0))
        projected.append(("context_supplied", split_context is not None))
    else:
        tools_index = 2 if operation == "score" else 1
        projected.append(("tools_supplied", len(arguments) > tools_index))
    trace.calls.append((alias, operation, tuple(projected), result))
    # Pieces belong to the call that just consumed them; a later, unrelated call must not
    # inherit them.
    trace.input_pieces = []


def build_model_result_evidence(
    trace: ModelExpressionTrace,
    *,
    matched: bool,
    event=None,
) -> ModelResultEvidence:
    """Convert an ephemeral evaluation trace to one fixed-size evidence item."""

    models = [alias for alias, _, _, _ in trace.calls]
    arguments = [(f"parameter:{name}", value) for name, value in trace.parameters]
    arguments.extend(
        dict.fromkeys(argument for _, _, projected, _ in trace.calls for argument in projected)
    )
    score_summary = None
    if trace.split_scores is not None:
        matches = (
            [
                (ordinal, value)
                for ordinal, value in enumerate(trace.split_scores)
                if trace.operator is not None and compare(value, trace.operator, trace.threshold)
            ]
            if trace.operator is not None
            else []
        )
        score_summary = {
            "count": len(trace.split_scores),
            "refused_count": sum(trace.split_refused or ()),
            "minimum": min(trace.split_scores) if trace.split_scores else None,
            "maximum": max(trace.split_scores) if trace.split_scores else None,
            "matching_count": len(matches),
            "first_match": (
                {"ordinal": matches[0][0], "value": matches[0][1]} if matches else None
            ),
        }
        scores = trace.split_scores
        if trace.split_spans is not None and scores:
            # The deciding chunk is the one whose score is the decision score: the
            # highest for `>`/`>=`, the lowest for `<`/`<=`, the lowest ordinal on a tie.
            lowest_decides = trace.operator in {"<", "<="}
            deciding = min(
                range(len(scores)),
                key=lambda ordinal: (
                    scores[ordinal] if lowest_decides else -scores[ordinal],
                    ordinal,
                ),
            )
            refused = trace.split_refused or [False] * len(scores)
            chunks = [
                {
                    "ordinal": ordinal,
                    "context": context,
                    "span": own,
                    "value": value,
                    "refused": refused[ordinal],
                }
                for ordinal, ((context, own), value) in enumerate(
                    zip(trace.split_spans, trace.split_scores, strict=True)
                )
            ]
            score_summary["deciding"] = chunks[deciding]
            score_summary["chunks"] = chunks
        operation = "split_score"
    elif len(trace.calls) > 1:
        operation = "numeric_expression"
    else:
        operation = trace.calls[0][1] if trace.calls else "model_expression"

    expression_value = trace.expression_value
    if expression_value is None and trace.calls:
        expression_value = trace.calls[-1][3]
    return ModelResultEvidence.bounded(
        operation=operation,
        models=models,
        arguments=arguments,
        expression_value=expression_value,
        score_summary=score_summary,
        operator=trace.operator,
        threshold=trace.threshold,
        matched=matched,
        source_path=getattr(event, "source_path", None),
        input_span=trace.input_span if trace.split_scores is None else None,
        refusals=trace.refusals,
        judge_reasons=trace.judge_reasons,
        omitted_judge_reasons=trace.omitted_judge_reasons,
    )


async def _evaluate_model_text(expression, context, event, bindings, trace, regex_backend):
    """Evaluate ``join()``, a map comprehension, or the text of a value list.

    ``join()`` keeps only non-empty text. It leaves out a value that is missing or not text.
    It counts the encoded bytes of the text it builds, and fails the rule rather than
    truncating a text over ``max_tool_argument_size``."""

    async def evaluate(child, scope=bindings, child_trace=trace):
        return await evaluate_model_expression(
            child,
            context,
            event=event,
            bindings=scope,
            trace=child_trace,
            regex_backend=regex_backend,
        )

    kind = expression.kind
    limits = context.limits
    if kind == "field_text":
        # The text a string test in a rule step reads from the same value. The tests of one
        # map comprehension item read the same text, so the item keeps it.
        texts = bindings.get(_ITEM_TEXTS, {})
        if expression not in texts:
            text = field_text(await evaluate(expression.children[0]), "")
            texts[expression] = MISSING if text is None else text
        return texts[expression]
    if kind == "join":
        pairs = await evaluate(expression.children[0])
        kept = [(item, value) for item, value in pairs if _is_text(value)]
        if not kept:
            return MISSING
        separator = str(expression.value)
        if not limits.enforce_limits:
            _record_pieces(trace, context, kept, expression.children[0].children[2])
            return separator.join(text for _, text in kept)
        cap = limits.max_tool_argument_size
        sep_size = len(separator.encode("utf-8"))
        size = 0
        parts: list[tuple[Any, str]] = []
        for item, text in kept:
            size += len(text.encode("utf-8")) + (sep_size if parts else 0)
            if size > cap:
                # Stop before building the rest of the text: a benign call must not be able to
                # push a risky call out of the judge's input unnoticed. Counts encoded bytes, as
                # max_tool_argument_size does everywhere else it is enforced.
                raise ValueError(
                    f"join() text has at least {size} bytes, exceeding max_tool_argument_size={cap}"
                )
            parts.append((item, text))
        _record_pieces(trace, context, parts, expression.children[0].children[2])
        return separator.join(text for _, text in parts)

    source = await evaluate(expression.children[0])
    if source is MISSING or source is None:
        return []
    limit = limits.max_comprehension_size
    if limits.enforce_limits and len(source) > limit:
        raise ValueError(
            f"map comprehension source has {len(source)} items, exceeding "
            f"max_comprehension_size={limit}"
        )
    variable = str(expression.value)
    keep, value = expression.children[1], expression.children[2]
    # The filter and the value cannot call a model. They write to a trace of their own, so
    # the text they test does not go into the evidence. Only the parameters they read do.
    item_trace = None if trace is None else ModelExpressionTrace()
    passed = _filter_results(context, keep, variable, regex_backend)
    values = []
    pacer = Pacer()
    for position, item in enumerate(source):
        # The items are synchronous work, so stop at the rule deadline and let other tasks run.
        raise_if_evaluation_expired()
        if pacer.due():
            await asyncio.sleep(0)
        bound = (
            _BoundMessage(item, source[position : position + 1])
            if isinstance(source, Messages)
            else item
        )
        scope = {**bindings, variable: bound, _ITEM_TEXTS: {}}
        # A call whose argument text Quarry cannot decode — not JSON, not a JSON object, or
        # another reason such as more levels than max_nesting_depth — fails the rule, as a rule
        # step that reads the call's arguments does: it cannot be told apart from a call whose
        # risky argument decodes fine but is never reached, so it must not be silently skipped.
        known = passed.get(id(item)) if passed is not None else None
        if known is not None and known[0] is item:
            passes = known[1]
        else:
            passes = await evaluate(keep, scope, item_trace) is True
            if passed is not None:
                passed[id(item)] = (item, passes)
        if passes:
            values.append((item, await evaluate(value, scope, item_trace)))
    if trace is not None and item_trace is not None:
        for name, parameter in item_trace.parameters:
            if all(existing != name for existing, _ in trace.parameters):
                trace.parameters.append((name, parameter))
    return values


def _record_pieces(trace, context, parts, read: ModelExpression) -> None:
    """Append the pieces of a model input's text, each read by ``read`` from one conversation
    message or call, to the trace's explain record: the item's ``context.timeline`` position
    and the field, never the text. input_span is copied into a score path entry and into
    telemetry, both of which must stay text-free. A consumer that wants the text renders it
    from the conversation at the piece's position and field."""

    if trace is None or not parts:
        return
    read = read.children[0] if read.kind == "field_text" else read
    field = str(read.value).rsplit(".", 1)[-1] if read.kind == "field" else None
    positions = {id(obj): position for position, obj in context.timeline}
    for item, _text in parts:
        if id(item) in positions:
            trace.input_pieces.append({"position": positions[id(item)], "field": field})


def _filter_results(context, keep: ModelExpression, variable: str, regex_backend) -> dict | None:
    """The results of a map comprehension filter by item, which the map comprehensions of one
    rule evaluation share. Each per-field ``join()`` of a rule can have the same filter. A
    filter that reads only its item and the conversation gives the same result for that item
    each time, so the rule computes it one time. None for any other filter.

    The key is the text of the filter, because ``true`` and ``1`` compare equal as values."""

    shared = getattr(context, "map_filter_results", None)
    if shared is None or not _reads_only(keep, variable):
        return None
    return shared.setdefault((repr(keep), regex_backend), {})


def _reads_only(expression: ModelExpression, variable: str) -> bool:
    """Return whether ``expression`` reads no binding other than ``variable`` and no
    parameter. The evidence records a parameter each time a filter reads it."""

    if expression.kind == "binding":
        return expression.value == variable
    if expression.kind == "parameter":
        return False
    return all(_reads_only(child, variable) for child in expression.children)


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and value != ""
