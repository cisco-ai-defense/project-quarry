# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Immutable message-valued map entries, before conversion to one model input."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload

from quarry.types.messages import Messages


@dataclass(frozen=True, slots=True)
class MessageGroups(Sequence[Messages]):
    """A map's written entries; length and indexing do not flatten those entries."""

    _items: tuple[Messages, ...]
    _empty: Messages

    def __post_init__(self) -> None:
        if type(self._items) is not tuple or any(
            not isinstance(item, Messages) for item in self._items
        ):
            raise TypeError("message-valued map entries must be an immutable tuple of Messages")
        if not isinstance(self._empty, Messages) or self._empty:
            raise TypeError("a message-valued map requires an empty source view")

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> Messages: ...

    @overload
    def __getitem__(self, index: slice) -> MessageGroups: ...

    def __getitem__(self, index):
        selected = self._items[index]
        return MessageGroups(selected, self._empty) if isinstance(index, slice) else selected
