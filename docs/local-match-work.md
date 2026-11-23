# Local matching work allowances

`local_match_work` is an optional runtime configuration for explicitly selected
rule groups. Each conversation creates its own owners; rules in one group share
one allowance. Rules outside those groups keep their ordinary behavior. An absent
block allocates no owner or report and reads no thread CPU clock.

```yaml
config_version: 2
rules: [rules/local.quarry]
local_match_work:
  groups:
    - label: local_checks
      rules: [Selected, Related]
      allowance_cpu_ns: 5000000
      mode: shadow
```

An allowance is a native integer from 1 through 2^63 - 1, in nanoseconds of CPU
time on the executing thread. Booleans, floating point values, empty/unknown
fields, duplicate labels, overlapping rule membership and unknown rule names are
rejected. There are at most 64 groups, 256 rules per group and 1,024 selected
rules overall. Labels contain 1–64 characters; rule names contain 1–256.
The default mode is `shadow`. Explicit null/false/empty blocks are invalid.
The scanner resolves selected rule names before constructing its resource
providers. A compiled policy therefore requires groups that select its rule.

## Ownership and outcomes

The owner measures synchronous matching and its required preparations: copying
the rule context, building canonical evaluation input, field/text preparation,
modifier handling, literal/regex comparisons, positions, counts, captures and
actual lazy iterator steps. Called checks retain their caller's owner. Nested
synchronous operations count once. A copied context executing on a worker uses
that worker's native thread clock, with shared synchronized counters.

Binding an owner across an await does not measure the suspended interval.
Provider callbacks, limiter waits, asynchronous suspension, input adapters and
unrelated nested scans do not spend the selected local matching allowance.
Existing rule/scan deadlines, cancellation, model-call budgets and retention
rules still apply. This allowance does not bound arbitrary host callback CPU.

`shadow` measures exhaustion while preserving the underlying verdicts and errors.
`enforce` rejects work when the allowance is unavailable or already exhausted.
The selected rule then has no authoritative verdict and reports the existing
`evaluation_limit` failure category; the scan is incomplete. A caught work denial
cannot become an authoritative false value or enter an immediate/message-value
cache. Already completed independent rules remain available, including a rule
that finished before another member of its group exhausted the allowance.

A synchronous/native call cannot be interrupted by this accounting mechanism.
Its CPU is charged when it returns; an enforcing owner rejects an over-budget
return value and records the observed overshoot. Completing exactly at the cap
can preserve that completed value; another operation is denied. Original
exceptions, including timeout/cancellation, retain precedence. An unavailable or
reversed clock leaves the work unmetered: shadow preserves outcomes, while
enforcement rejects an authoritative result.

Evidence refinement and exploration use the same group owner until their actual
completion. Evidence remains best effort under its existing evidence deadline:
an exhausted evidence operation stops refinement without changing an already
committed verdict. Closing the owner rejects late work but does not claim that
an active native worker has ended.

## Reports and public surfaces

`ScanResult.local_match_work` is `None` when the configuration is absent. With a
configured allowance, scanner and compiled-policy results carry a version 1
report; CLI JSON and MCP structured/text responses include `local_match_work`.
Use `--output-format json` to inspect it from the CLI. MCP request arguments
cannot change the host's configured budgets. Saved policy bytes and their
identities exclude this runtime-only configuration; pass it again when loading.

Each report has a scope (`conversation` or `batch`), observed conversation count,
measurement completion and at most 64 group rows. Rows contain configured label,
mode and per-conversation allowance; known and exact CPU/overshoot; operation,
denial, active, unmetered and exhausted-conversation counts; saturation state;
first observed limit phase (`prepare`, `match` or `iterate`); and owner closure.
The report stores no conversation text, arguments, results, callbacks or request
digests. Configured labels are deliberately visible.

Report `complete` describes measurement completion. Scan `is_complete` describes
authoritative rule evaluation. An enforcing cut can leave a complete measurement
of an incomplete scan; a shadow overrun can accompany a complete scan.
`cut_phase` also records a shadow overrun, without implying that shadow stopped
the operation. Exact `value_cpu_ns` and `overshoot_cpu_ns` remain null while an
owner is open, work is active, a clock is unavailable or a counter saturated.
Known fields remain lower bounds; unavailable work is never reported as measured
zero. Counters saturate at 2^63 - 1 without wrapping.

Ordered compiled-policy batches retain a separate report per conversation.
Scanner batches and aggregate timeout/MCP responses combine the same configured
groups in constant space. Allowances remain per conversation: observed CPU and
per-conversation overshoot sum, exhaustion means any observed conversation
exhausted its own allowance, and unknown values remain unknown. Batch size does
not multiply the number of report rows. Reports do not make routing or fail-open
decisions; callers continue to use the scan's completion and failure contract.
