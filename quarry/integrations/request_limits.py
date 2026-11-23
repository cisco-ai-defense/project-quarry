# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Private authority for a shared request ended by its starter's actual limits."""

from typing import TypeVar

from quarry.safety.timer import QuarryTimeoutError, raise_if_evaluation_expired

_Error = TypeVar("_Error", bound=BaseException)
_OWNED_LIMIT = object()


def owned_request_limit(error: _Error) -> _Error:
    """Mark only errors created while enforcing an actual deadline or attempt limit.

    Error names and provider-controlled diagnostic codes do not establish ownership.
    The marker retains no request, deadline, input, traceback or caller scope.
    """

    error.__dict__["_quarry_owned_request_limit"] = _OWNED_LIMIT
    return error


def is_owned_request_limit(error: BaseException) -> bool:
    """Identify our private marker without interpreting an adapter's error payload."""

    return error.__dict__.get("_quarry_owned_request_limit") is _OWNED_LIMIT


def raise_if_owned_evaluation_expired() -> None:
    """Prove ownership at a native request's own deadline checkpoint only.

    Keep host calls outside this handler: an adapter can report the same public
    exception type without having exhausted Quarry's active evaluation deadline.
    """

    try:
        raise_if_evaluation_expired()
    except QuarryTimeoutError as error:
        owned_request_limit(error)
        raise
