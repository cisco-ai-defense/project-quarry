# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Readability checks for canonical messages crossing a model boundary."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from quarry.engine.pacing import Pacer
from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.exceptions import InvalidInputError
from quarry.types.media import require_readable_media

if TYPE_CHECKING:
    from quarry.types.messages import Messages


async def require_model_messages(messages: Messages, *, standalone: bool = True) -> None:
    """Reject unreadable call values and media before cache, budget or host dispatch."""

    seen: set[int] = set()
    pacer = Pacer()
    for message in messages:
        raise_if_evaluation_expired()
        if pacer.due():
            await asyncio.sleep(0)
        if id(message) not in seen:
            seen.add(id(message))
            message._require_model_readable()
    if standalone and messages:
        try:
            messages.assert_self_contained()
        except ValueError as exc:
            raise InvalidInputError("model input does not contain exact call/result links") from exc
    await require_readable_media(messages)
