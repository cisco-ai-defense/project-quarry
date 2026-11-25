# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Media readability checks at the point a rule hands content to an adapter."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any

from quarry.engine.pacing import Pacer
from quarry.safety.timer import raise_if_evaluation_expired
from quarry.types.exceptions import InvalidInputError

if TYPE_CHECKING:
    from quarry.types.messages import Message, Messages


_IMAGE_HEADER = re.compile(r"data:image/[^\s/;,]+;base64,")
_BASE64_CHARACTERS = re.compile(r"[A-Za-z0-9+/]*")
_VALIDATION_BLOCK = 65_536


def _valid_image_url(url: str) -> bool:
    """Check the envelope in place, without decoding or copying the encoded body."""

    raise_if_evaluation_expired()
    header = _IMAGE_HEADER.match(url)
    if header is None:
        return False
    start, end = header.end(), len(url)
    padding = 0
    while end > start and url[end - 1] == "=" and padding < 2:
        end -= 1
        padding += 1
    characters = end - start
    if padding:
        if (characters + padding) % 4 or characters % 4 != 4 - padding:
            return False
    elif characters % 4 == 1:
        return False
    # Fixed-size regex windows keep deadline observation bounded and avoid creating
    # a second copy of a potentially large data URL. Pixel validity is not inspected.
    for position in range(start, end, _VALIDATION_BLOCK):
        raise_if_evaluation_expired()
        if (
            _BASE64_CHARACTERS.fullmatch(url, position, min(position + _VALIDATION_BLOCK, end))
            is None
        ):
            return False
    raise_if_evaluation_expired()
    return True


def media_fault(role: object, part: Mapping[str, Any]) -> str | None:
    """Return a fixed diagnostic for media a model or transform cannot read.

    This leaves text-only reads untouched. External URL policy is enforced separately
    by the provider boundary; this function checks inline image envelopes only.
    """

    kind = part.get("type")
    if kind not in {"image_url", "input_audio"}:
        return None
    if role == "tool":
        return "tool-result media cannot be read by a model or transform"
    if kind == "image_url":
        source = part.get("image_url")
        url = source.get("url") if isinstance(source, Mapping) else None
        if isinstance(url, str) and url.startswith("data:") and not _valid_image_url(url):
            return "inline image must use an image data URL with a base64 payload"
    return None


def require_media_part(role: object, part: Mapping[str, Any], *, owner: str = "media") -> None:
    """Reject unreadable media without including its URL, encoded body or host fields."""

    fault = media_fault(role, part)
    if fault is not None:
        raise InvalidInputError(f"{owner}: {fault}")


def require_message_media(message: Message, selected_parts: Collection[str] | None = None) -> None:
    """Synchronous defensive check for direct canonical-to-wire projections."""

    content = message.source_content
    if isinstance(content, tuple):
        for index, part in enumerate(content):
            raise_if_evaluation_expired()
            if isinstance(part, Mapping) and (
                selected_parts is None or part.get("type") in selected_parts
            ):
                require_media_part(
                    message.role, part, owner=f"messages[{message.message_index}].content[{index}]"
                )


async def require_readable_media(
    messages: Messages,
    selected_parts: Collection[str] | None = None,
    *,
    rewritable_only: bool = False,
) -> None:
    """Check selected media before any adapter dispatch, cache hit or budget charge."""

    seen: set[int] = set()
    pacer = Pacer()
    for message in messages:
        raise_if_evaluation_expired()
        if pacer.due():
            await asyncio.sleep(0)
        if id(message) in seen:
            continue
        seen.add(id(message))
        if rewritable_only and not message._readable_for_rewrite():
            continue
        content = message.source_content
        if not isinstance(content, tuple):
            continue
        for index, part in enumerate(content):
            raise_if_evaluation_expired()
            if pacer.due():
                await asyncio.sleep(0)
            if isinstance(part, Mapping) and (
                selected_parts is None or part.get("type") in selected_parts
            ):
                require_media_part(
                    message.role, part, owner=f"messages[{message.message_index}].content[{index}]"
                )
