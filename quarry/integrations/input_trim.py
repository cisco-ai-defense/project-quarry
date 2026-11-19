# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""One bounded, source-preserving prefix/suffix selection; no detached work."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from quarry.integrations.message_ranges import MessageRangeIndex, RangeBudget
from quarry.types.messages import Messages


@dataclass(frozen=True)
class TrimmedInput:
    messages: Messages
    omitted_messages: int
    omitted_characters: int


async def trim_messages(
    messages: Messages,
    size: int,
    *,
    context: Messages | None,
    direction: str,
    cut_keep: str,
    count: Callable[[Messages], Awaitable[int]],
) -> TrimmedInput:
    if type(size) is not int or size <= 0:
        raise ValueError("trim() size must be a positive integer")
    if direction not in {"start", "end"} or cut_keep not in {"start", "none"}:
        raise ValueError("invalid trim selection settings")
    budget = RangeBudget()
    # Remove only the actual context occurrences; equal text or a different transformed
    # occurrence is not the same input. Context moves to the front with native origins.
    available = {}
    for position in range(len(messages) - 1, -1, -1):
        budget.add("queries")
        available.setdefault(id(messages[position]), []).append(position)
        await budget.checkpoint()
    consumed = set()
    for message in context or ():
        budget.add("queries")
        matches = available.get(id(message), [])
        if matches:
            consumed.add(matches.pop())
        await budget.checkpoint()
    body = [position for position in range(len(messages)) if position not in consumed]
    empty = messages._view_positions(())

    async def view(parts):
        budget.add("queries", len(parts) + 1)
        await budget.checkpoint()
        ordered = [part for _, part in sorted(parts.items())]
        variants = tuple(part[0] for part in ordered)
        selected = messages._view_ordinals(message.message_index for message in variants)
        selected._install_variants(variants)
        # A trim is an ordinary model request. It must not inherit the orphan-result
        # exemption reserved for split chunks that score builders flatten into text.
        if context is not None and context:
            return (
                await Messages._combine_async([context, selected])
                if selected
                else context._view_positions(range(len(context)))
            )
        return selected if selected else empty

    whole = await view({position: messages[position : position + 1] for position in body})
    if not whole or await count(whole) <= size:
        return TrimmedInput(whole, 0, 0)
    if context is not None and context and await count(context) > size:
        raise ValueError("trim() context exceeds the input budget")
    selected = {}
    omitted_characters = 0
    order = body if direction == "start" else reversed(body)
    for position in order:
        unit = messages[position : position + 1]
        candidate = await view({**selected, position: unit})
        if await count(candidate) <= size:
            selected[position] = unit
            continue
        message = messages[position]
        if cut_keep == "start" and message._can_split_text() and message.content:
            prepared = await MessageRangeIndex.prepare(message, budget)

            async def prefix(length, position=position, prepared=prepared):
                part = messages[position : position + 1]
                part._install_variants((prepared.segment(0, length),))
                return part

            low, high, best = 1, len(message.content) - 1, 0
            while low <= high:
                middle = (low + high) // 2
                candidate = await view({**selected, position: await prefix(middle)})
                if await count(candidate) <= size:
                    best, low = middle, middle + 1
                else:
                    high = middle - 1
            if best:
                selected[position] = await prefix(best)
                omitted_characters = len(message.content) - best
        break
    result = await view(selected)
    # Always confirm the complete selected request; summed individual token counts are
    # not a request-fit proof when framing or tools have nonlinear costs.
    if result and await count(result) > size:
        raise ValueError("trim() selected request exceeds the input budget")
    await budget.checkpoint(finish=True)
    return TrimmedInput(result, len(body) - len(selected), omitted_characters)
