# Model window observations

Window observations are optional, bounded metadata for reached windowed classifiers and
span readers. They describe the reply each logical reader accepted and the input cut it
actually consumed. They do not make model or tokenizer requests, change verdicts, log data,
or send observations anywhere. Ordinary non-windowed model operations do not add rows.

## Request a result snapshot

```python
result = await policy.scan_async(conversation, window_observations=True)
report = result.window_observations
```

`window_observations` is a strict boolean on the compiled-policy `scan`, `scan_async`,
`scan_many`, and `scan_many_async` methods and on the scanner's `scan_data`,
`scan_data_async`, `scan_batch_data`, `scan_file`, and `scan_directory` methods. Each
conversation gets its own detached snapshot; a scanner file batch combines its conversations
and lists their scopes. `logical_reads=True` requests the separate logical-read report.
Neither report changes the physical attempts in `model_usage`.

The CLI accepts `quarry scan --window-observations --output-format json ...` and requires
JSON output for this flag. The MCP `quarry_scan` tool accepts `window_observations: true`
and returns the report on each result, including compact batch results. The default JSON
response omits `window_observations`; the Python result field defaults to `None`.

## Collect across calls

```python
from quarry import WindowObservationLimits, observe_model_windows

with observe_model_windows(WindowObservationLimits(max_rows=64)) as observations:
    first = await policy.scan_async(conversation)
    second = await policy.scan_async(conversation)
    live_snapshot = observations.report()

final_snapshot = observations.report()
```

The context owns a logical-read session. Reports are ordinary JSON-shaped defensive copies;
changing a returned list or dictionary cannot rewrite an observation. Await owned work
inside the context. `finalized` remains false while an owner still holds a metadata lease.

When a result option borrows an existing `model_read_usage` session, it enables window
collection in that session from that point until the session closes. It does not reconstruct
earlier replies. Per-result snapshots still select only their own conversation scopes. Their
`finalized=False` flag remains a historical statement even after the borrowed session closes.
Configure custom bounds with `model_read_usage(window_limits=WindowObservationLimits(...))`
or the `observe_model_windows` context's `limits` argument. Its `read_limits` argument
separately bounds underlying logical identities, bindings, captures, and source measurements.

## Version 1 fields

`window_observation_report_schema()` returns a fresh dependency-free JSON Schema for both
context reports and result snapshots. The MCP output schema also describes the report.

| Field | Meaning |
| --- | --- |
| `version` | Report format version, currently `1`. |
| `enabled`, `finalized` | Collection was enabled; all session owners have finished. |
| `rows` | Bounded logical reader observations. |
| `omitted` | Capped count of refused or missing metadata observations. Zero alone does not establish complete measurements. |
| `scopes`, `state` | Result snapshots only: selected conversation scopes and `collected` or `scope_unavailable`. An unavailable scope never borrows another conversation's rows. |

Each row includes `scope`, `rule`, `phase`, `model` (the configured alias), `provider_model`
(when admitted), `operation`, the existing logical `outcome`, `unit`, `window_count`,
`source_characters`, `covered_tokens`, `cached`, `state`, `probabilities`,
`probability_states`, and `reads`. Phases keep decision, evidence, and measurement readers
separate. Replayed identical reads within a rule may share one row with an increased `reads`
count; its input coverage is not multiplied by that count.

`cached=False` means this reader accepted the reply its own request prepared. `True` means
it accepted an existing or shared reply, including a shared in-flight request another reader
started. It is not an attempt count. A cancelled starter can make a physical call while a
surviving waiter reports `cached=True`. No accepted reply means `cached` can be null.

`source_characters` is the union of known native source-character intervals selected by the
windows, not a sum of overlapping window lengths or the whole conversation. It excludes the
separate window context. `covered_tokens` counts distinct selected full-encoding positions,
excluding the declared retained endpoint tokens. It does not estimate characters from token
counts. Both contain `state` and `value`; an unavailable or incomplete measurement has a
null value, not a measured zero. Token encodings without character offsets leave source
character coverage unavailable. Empty input has zero windows and zero coverage without a
provider call; its state and outcome are `not_sent`.

`probabilities` contains detached finite class-probability vectors in window order. A
window whose vector cannot safely be observed has a null vector and an explicit
`probability_states` entry. Optional collection never invokes an unused property or numeric
conversion to obtain a vector. A selected classifier's already-read native values can be
observed without reading its property again. Invalid unused classes, opaque storage,
conflicting mutable replies, and exhausted bounds remain explicit; they do not become new
classification errors. Failed, cancelled, or still pending reads never publish accepted
vectors. Classifier refusals retain the native failed-read outcome; the existing distinct
refusal outcome applies to score and completion operations.

Reports contain labels, counts, and probabilities, but no raw messages, rendered text,
context strings, token IDs, span text, provider replies, exceptions, or host objects. Labels
are caller-provided metadata and can themselves contain sensitive names; handle the report
according to the application's data policy.

## Bounds and live rule ownership

`WindowObservationLimits` separately caps retained replies, event frames, output rows,
windows, probability scalars, label bytes, and traversal work. Limits require native positive
bounded integers. Collection and each report rendering have bounded work; output copies also
share a report-wide probability budget. Resource refusal is visible through row states,
null fields, and `omitted`, and leaves the native scan's verdict and errors authoritative.
These bounds apply to metadata, not to the provider's ordinary request limits or deadlines.

`current_rule_name()` returns the currently evaluating rule name from a callback's task
context, including awaited work on a host loop or worker thread. It returns `None` outside
that rule's live scope and in a forked child. Nested scans restore their caller's owner.
A copied context does not keep a finished rule alive. If a request's starter is cancelled
while another rule continues waiting, the surviving provider callback does not acquire the
waiter's rule name: historical attribution stays in the logical read rows, while the live
accessor returns `None` once its original owner has exited.
