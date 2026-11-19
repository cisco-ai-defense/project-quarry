# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Optional accounting for actual native window inputs, using only retained host offsets."""

from quarry.integrations.model_windows import CharacterWindow, TokenWindow, token_character_ranges
from quarry.integrations.read_identity import ReadIdentityLimitError, ReadIdentityUnavailable
from quarry.integrations.read_inputs import InputMeasurement
from quarry.integrations.read_origins import ReadOriginProjection
from quarry.integrations.read_usage import UnitMeasurement


def collect_window_input(session, source, windows, encoding):
    native, registry = session._text_inputs, session._text_inputs.registry
    try:
        scope = session.input_scope()
        if scope is None:
            raise ReadIdentityUnavailable("window input scope is unavailable")
        if not windows:
            empty = registry.value_interval(scope, "empty-window-input", "")
            return InputMeasurement(UnitMeasurement("characters", "measured"), empty.source)
        registry.inspect(len(windows) + 1)
        keys = [registry.value_interval(scope, "input-shape", "model-windows").source, len(windows)]
        first = windows[0]
        if type(first) is CharacterWindow:
            ranges = ((first.start, windows[-1].end),)
            keys.append(registry.value_interval(scope, "window-unit", "characters").source)
            for window in windows:
                if type(window) is not CharacterWindow or window.source is not source:
                    raise ReadIdentityUnavailable("window input plan is unavailable")
                keys.extend((window.start, window.end))
        elif type(first) is TokenWindow:
            if encoding is None or encoding.character_offsets is None:
                raise ReadIdentityUnavailable("window token character offsets are unavailable")
            start = first.positions[int(encoding.prefix_token)]
            last = windows[-1]
            end = last.positions[len(last.positions) - 1 - int(encoding.suffix_token)] + 1
            ranges = token_character_ranges(
                encoding, range(start, end), lambda: registry.inspect(1)
            )
            keys.append(registry.value_interval(scope, "window-unit", "tokens").source)
            keys.extend((int(encoding.prefix_token), int(encoding.suffix_token)))
            for window in windows:
                if type(window) is not TokenWindow:
                    raise ReadIdentityUnavailable("window input plan is unavailable")
                keys.extend(
                    (
                        window.positions[int(encoding.prefix_token)],
                        window.positions[len(window.positions) - 1 - int(encoding.suffix_token)]
                        + 1,
                    )
                )
        else:
            raise ReadIdentityUnavailable("window input plan is unavailable")
        projector = ReadOriginProjection(registry, native._parts, native._ensure)
        length = projector.prepare(source.text, source.origin)
        keys.append(projector.identity(source.origin))
        intervals = []
        for start, end in ranges:
            registry.inspect(1)
            if not 0 <= start <= end <= length:
                raise ReadIdentityUnavailable("window character interval is outside its source")
            selected = projector.window_ranges(source.origin, start, end)
            if len(intervals) + len(selected) > registry.max_ranges:
                raise ReadIdentityLimitError("window logical input range limit exceeded")
            intervals.extend(selected)
            keys.extend((start, end))
        return InputMeasurement(
            UnitMeasurement("characters", "measured", tuple(intervals)),
            registry.origin_identity(None, tuple(keys)),
        )
    except Exception as error:
        return InputMeasurement(UnitMeasurement("characters", native._failed(error)), None)
