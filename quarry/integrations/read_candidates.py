# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Optional coverage from one validated channel, projected only onto native parts."""

from quarry.integrations.read_identity import ReadIdentityLimitError, ReadIdentityUnavailable
from quarry.integrations.read_inputs import InputMeasurement
from quarry.integrations.read_origins import ReadOriginProjection
from quarry.integrations.read_usage import UnitMeasurement


def collect_candidate_input(session, sources, coverage, *, complete):
    native, registry = session._text_inputs, session._text_inputs.registry
    try:
        scope = session.input_scope()
        if scope is None:
            raise ReadIdentityUnavailable("candidate input scope is unavailable")
        registry.inspect(len(sources) + len(coverage) + 1)
        by_part = {}
        for interval in coverage:
            registry.inspect(1)
            if not 0 <= interval.part < len(sources):
                raise ReadIdentityUnavailable("candidate coverage belongs to another source part")
            by_part.setdefault(interval.part, []).append(interval)
        projector = ReadOriginProjection(registry, native._parts, native._ensure)
        keys = [
            registry.value_interval(scope, "input-shape", "candidate-parts-v1").source,
            int(complete),
            len(sources),
        ]
        intervals = []
        for part, source in enumerate(sources):
            registry.inspect(1)
            length = projector.prepare(source.text, source.origin)
            selected = by_part.get(part, ())
            keys.extend((part, projector.identity(source.origin), length, len(selected)))
            for interval in selected:
                registry.inspect(1)
                if not 0 <= interval.start <= interval.end <= length:
                    raise ReadIdentityUnavailable("candidate coverage lies outside its source part")
                projected = projector.window_ranges(source.origin, interval.start, interval.end)
                if len(intervals) + len(projected) > registry.max_ranges:
                    raise ReadIdentityLimitError("candidate logical coverage range limit exceeded")
                intervals.extend(projected)
                keys.extend((interval.start, interval.end))
        return InputMeasurement(
            UnitMeasurement("characters", "measured" if complete else "partial", tuple(intervals)),
            registry.origin_identity(None, tuple(keys)) if complete else None,
        )
    except Exception as error:
        return InputMeasurement(UnitMeasurement("characters", native._failed(error)), None)
