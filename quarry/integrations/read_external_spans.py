# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Project only external coverage actually declared for this reader's source."""

from quarry.integrations.read_identity import ReadIdentityLimitError, ReadIdentityUnavailable
from quarry.integrations.read_inputs import InputMeasurement
from quarry.integrations.read_origins import ReadOriginProjection
from quarry.integrations.read_usage import UnitMeasurement


def collect_external_input(session, source, coverage, *, complete):
    native, registry = session._text_inputs, session._text_inputs.registry
    try:
        scope = session.input_scope()
        if scope is None:
            raise ReadIdentityUnavailable("external input scope is unavailable")
        registry.inspect(len(coverage) + 1)
        projector = ReadOriginProjection(registry, native._parts, native._ensure)
        length = projector.prepare(source.text, source.origin)
        keys = [
            registry.value_interval(scope, "input-shape", "external-spans").source,
            projector.identity(source.origin),
            int(complete),
            len(coverage),
        ]
        intervals = []
        for interval in coverage:
            registry.inspect(1)
            if not 0 <= interval.start <= interval.end <= length:
                raise ReadIdentityUnavailable("external coverage lies outside its source")
            selected = projector.window_ranges(source.origin, interval.start, interval.end)
            if len(intervals) + len(selected) > registry.max_ranges:
                raise ReadIdentityLimitError("external logical input range limit exceeded")
            intervals.extend(selected)
            keys.extend((interval.start, interval.end))
        return InputMeasurement(
            UnitMeasurement("characters", "measured" if complete else "partial", tuple(intervals)),
            registry.origin_identity(None, tuple(keys)) if complete else None,
        )
    except Exception as error:
        return InputMeasurement(UnitMeasurement("characters", native._failed(error)), None)
