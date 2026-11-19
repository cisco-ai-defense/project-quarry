# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Distinguish native engine methods from custom host backend implementations."""

from functools import partial
from types import MethodType

from quarry.async_runtime import call_host, call_host_sync


async def _await_backend(operation, *args, **kwargs):
    try:
        return await operation(*args, **kwargs)
    finally:
        del operation, args, kwargs


async def call_backend(operation, *args, **kwargs):
    from quarry.integrations.model_inference import ModelInferenceBackend

    try:
        if (
            isinstance(operation, MethodType)
            and isinstance(operation.__self__, ModelInferenceBackend)
            and getattr(ModelInferenceBackend, operation.__name__, None) is operation.__func__
        ):
            return await operation(*args, **kwargs)
        return await call_host(partial(_await_backend, operation, *args, **kwargs))
    finally:
        del operation, args, kwargs


async def call_backend_sync(operation, *args, **kwargs):
    from quarry.integrations.model_inference import ModelInferenceBackend

    try:
        if (
            isinstance(operation, MethodType)
            and isinstance(operation.__self__, ModelInferenceBackend)
            and getattr(ModelInferenceBackend, operation.__name__, None) is operation.__func__
        ):
            return operation(*args, **kwargs)
        return await call_host_sync(partial(operation, *args, **kwargs))
    finally:
        del operation, args, kwargs
