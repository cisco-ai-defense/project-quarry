# External span adapters

An embedded host can register a generic span adapter in `ModelInferenceBackend(adapters=...)`.
Select it explicitly in the model's `spans` configuration:

```yaml
config_version: 2
models:
  reader:
    adapter: my_external_adapter
    model: synthetic-v1
    category: SYNTHETIC
    spans:
      backend: external
      renderer: join
      limits:
        max_input_characters: 1000000
        max_spans: 4096
        max_coverage_ranges: 4096
        max_work: 262144
```

The adapter implements `external_spans(config, text, *, timeout=None)`. It receives an
immutable string and returns a native `ExternalSpanReply`. It needs no classification,
token-head or span-decoder methods unless the policy also selects those operations.
Existing classifier-backed span configurations retain their `class_index` syntax and
identity. External spans cannot be combined with `class_index` or model `windows`.

`renderer: join` sends selected message contents separated by newlines. `renderer: json`
sends the selected messages' JSON representation, including their structured fields.
Choose the renderer according to the information the host should receive. The host
receives no framework source graph. The framework retains the original source mapping
and binds returned ranges independently for each reader, including cached replies.
Rendering also retains the framework's existing character and graph limits.

Here is a synthetic demonstration adapter:

```python
from quarry.types.external_spans import (
    ExternalReadRange,
    ExternalSpanRange,
    ExternalSpanReply,
)


class SyntheticAdapter:
    async def external_spans(self, config, text, *, timeout=None):
        marker = "SYNTHETIC_MARKER"
        start = text.find(marker)
        detections = (ExternalSpanRange(start, start + len(marker), 0.9),) if start >= 0 else ()
        return ExternalSpanReply.for_input(
            text,
            detections,
            (ExternalReadRange(0, len(text)),),
            complete=True,
        )
```

Pass the adapter as `adapters={"my_external_adapter": SyntheticAdapter()}` when
constructing the backend and supply that backend through the existing scanner or
compiled-policy API. Adapter registration is trusted host code; this configuration
does not import arbitrary modules or create a transport. A standalone installation
cannot resolve an arbitrary custom adapter name using only its configuration file.

Ranges use half-open Python string character offsets, including Unicode characters,
and scores must be finite native numbers in `[0, 1]`. Detections must be nonempty and
inside explicitly inspected coverage. Coverage may overlap; its union determines how
many characters were inspected. A complete reply covers the entire actual input.
`for_input` binds the reply to its exact request with SHA-256, detecting stale replies
for different inputs of the same length. This digest does not prove that a provider's
declared inspection occurred, and it is not emitted in scan telemetry or result IDs.

An incomplete reply uses `complete=False` and lists only the ranges actually inspected.
Every selected incomplete read fails with `model_partial_read`, including a reply with
positive detections or a false completeness declaration covering the whole input.
Negation and quantifiers therefore cannot treat missing data as an authoritative
negative or positive. Unselected branches retain their normal short-circuit behavior.
A selected message with empty rendered text is authoritative only after a validated
complete reply; an incomplete empty reply remains incomplete. Selecting no messages
retains the framework's missing-input, no-call behavior. Empty conversations retain
their existing validation gate.

Validation copies bounded native ranges before cache admission. Malformed or stale
replies fail with `model_invalid_output`; provider failures use `model_provider`.
Existing timeout and per-alias call-budget errors keep their original behavior.
Configured input-character capacity failures use `evaluation_limit`. Counts, input characters and validation
work have hard ceilings; zero input/span/coverage capacities are explicit limits, while
the work budget must be positive. `max_input_tokens` can additionally configure the
existing operation-aware `trim()` behavior when token counting is configured.

Eligible concurrent readers share one pending callback and retain separate rule/alias
attribution. Cancelling one reader preserves work still owned by another; cancelling
the last reader waits for owned host cleanup. Partial replies fail before admission,
so a later reader issues its own request. Validated complete numeric replies may be
reused under the configured model-response retention policy. Thresholds are applied
when each reader consumes a reply; the adapter's supplied scores are not recalibrated.

With `logical_reads=True`, coverage projects only the declared inspected intervals
through that reader's native source mapping. Partial coverage has `value: null`, a
known lower bound, and a `partial` measurement state, including unknown empty coverage.
Physical dispatches appear as the `spans` operation in model usage. Non-window external
reads do not create rows in the separate window-observation report. These reports
contain bounded counts and caller-selected labels, not input text or detections.
