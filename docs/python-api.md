# Project Quarry: Python API

## `QuarryGuard`

`QuarryGuard` loads a YAML configuration and returns matching rules. It raises a typed error for invalid or incomplete scans.

```python
from quarry import QuarryGuard, ConversationValidationError, IncompleteScanError

guard = QuarryGuard("config.yaml")
conversation = {"messages": [{"role": "user", "content": "Hello"}]}
try:
    matches = guard.scan(conversation)
except (ConversationValidationError, IncompleteScanError) as error:
    handle_scan_failure(error)
else:
    for match in matches:
        print(match["rule"])
```

Use `await guard.scan_async(conversation)` in an event loop.

## `DSLScanner`

`DSLScanner` loads, validates, and compiles a complete rule generation when constructed. It returns a `ScanResult` with verdicts, timing, completion issues, and optional witness evidence. Check status before using matches:

```python
from pathlib import Path
from quarry import DSLScanner
from quarry.config import QuarryConfig

config = QuarryConfig.from_yaml(Path("config.yaml"))
scanner = DSLScanner(config)
result = scanner.scan_data(
    {"messages": [{"role": "user", "content": "Hello"}]}, enable_explain=True
)
result.raise_for_status()
print(result.rule_matches, result.detailed_matches, result.outputs)
```

`scan_data_async` serves async applications. `ScanResult.completion_state` is `complete`, `invalid`, or `incomplete`; `is_complete` is true only for an authoritative result. Explain evidence and typed `outputs` follow the same status rule.

When constructing `SafetyLimits` in Python, `max_total_size` must be a nonnegative integer;
booleans, floats, strings and `None` raise `ValueError` at construction. Zero is a real budget
that rejects nonempty input. To disable limit enforcement for trusted input, set
`enforce_limits=False` with a valid budget.

Set `logical_reads=True` on scanner or compiled-policy scans to attach optional input-read
snapshots. See [logical reads](logical-reads.md) for exact coverage unions, requesting-reader
attribution, host reference token maps, limits and collection-lifetime semantics.

Request `window_observations=True` for a separate bounded snapshot of reached windowed
classifiers and span readers. Use `observe_model_windows()` to collect across calls and
`current_rule_name()` for live callback ownership. See [window observations](window-observations.md).

Configure `local_match_work` to shadow or enforce per-conversation CPU allowances for
explicit local rule groups. Results then include bounded `local_match_work` evidence;
an enforcing cut leaves the affected rule incomplete. See [local matching work](local-match-work.md).

### Scan timing and model usage

`ScanResult.scan_time` is the bounded, finite, nonnegative monotonic duration for the scan.
`ScanResult.rule_timings` is either `None` or a map from each started rule to its monotonic
evaluation duration; concurrent rule durations can overlap and therefore need not sum to
`scan_time`. `scan_batch_data` reports the monotonic elapsed time of the whole ordered batch and
the average duration of each rule across child results that started it. Validation or selection
failures before rule execution have no rule timings.

`ScanResult.model_usage` is always present. It contains bounded provider-neutral physical
adapter-attempt counts:

```python
{
    "calls": 1,
    "records": [{"model": "judge", "operation": "score", "calls": 1}],
    "truncated": False,
}
```

The `model` value is the configured public alias, never a provider model name. Records use only
the generic operations `score`, `classify`, `complete`, and `input_tokens`. The summary does not
contain prompts, model content, responses, endpoints, credentials, trace data, or pricing.
Telemetry is diagnostic and never changes a verdict or completion state.
Window token encoding and token-count requests use `input_tokens`; a window classifier batch
uses one `classify` attempt per adapter invocation, regardless of the number of windows.
Model-backed transforms use `complete`; local transforms and span decoders add no attempts.
Payload preparation consumes the logical model or rule deadline. If preparation exhausts
that budget, the adapter is not invoked and no physical attempt is charged. Otherwise the
adapter receives only the time still remaining after preparation.
An expired requesting rule retains its `QuarryTimeoutError` when shared work reports a timeout.
Adapter execution and limiter cleanup also consume the logical budget. A successful response
that finishes after the deadline is rejected before cache admission, while its physical attempt
remains counted.

Whole-message rewrites preserve the active rule or scan deadline as the owner of
their evaluation timer. Expiry raises `QuarryTimeoutError`; an adapter's independent
timeout remains a transform error while the evaluation timer is still active.

When `fail_on_timeout` raises `QuarryTimeoutError`, `error.scan_result` (also available as
`error.result`) carries the safe partial timing and model-usage snapshot recorded before the
exception escaped. File and batch wrappers retain that telemetry in their returned incomplete
results while preserving their existing failure categories and exception text.
For a `CompiledPolicy` batch, a timeout on a later item aggregates the completed prefix and the
current timed-out item once: `scan_time` is their saturated sum, each `rule_timings` value is the
average across observed items that started that rule, and `model_usage` is the bounded merge in
input order. An unstarted current item contributes zero.

See [scan telemetry](PLUGIN.md#scan-telemetry) for exact retry, cache, shared-call, cancellation,
batch, saturation, and CLI/plugin serialization semantics.

## Typed scan failures

`result.failures` is an immutable tuple of `ScanFailure` records. Each record has a
`ScanFailureCategory` and an optional `rule_name`; `failure.to_dict()` returns JSON fields
`category` and, when present, `rule_name`. The same records appear as `failures` in CLI JSON
and CSV output, plugin scan payloads, and result-bearing guard exception `details`.

The stable category strings are:

| Category | Meaning |
| --- | --- |
| `invalid_input` | Input or scan parameter validation rejected the scan. |
| `rule_selection` | An explicit selected rule set was empty or named unavailable rules. |
| `input_read` | A scan file could not be read or decoded as JSON. |
| `preparation` | Conversation preparation failed before rule evaluation. |
| `model` | A model or provider operation failed. |
| `resource` | A non-model resource operation failed. |
| `evaluation` | A rule evaluation failed, including legacy results without type metadata. |
| `timeout` | A rule or scan operation exceeded its deadline. |
| `skipped` | A selected rule was skipped. |
| `truncated_input` | A legacy or custom scanner returned a result for truncated input. |
| `evaluation_limit` | Evaluation reported a limit warning. |
| `unknown` | A failure could not be classified from its exception type. |

Use enum members rather than parsing human-readable error messages:

```python
from quarry import ScanFailureCategory

if any(f.category == ScanFailureCategory.MODEL for f in result.failures):
    handle_model_failure()
result.raise_for_status()
```

These categories supplement existing diagnostics and do not change verdicts or completion
states. Partial matches remain non-authoritative. Records omit exception messages, exception
metadata, and provider details; existing diagnostic strings retain their existing behavior.
Unknown exceptions are never treated as a clean result. A directly raised exception, such as
caller cancellation or `fail_on_timeout`, retains its existing behavior and does not become a
returned failure record.

To render a parsed condition back to public DSL syntax, use the bounded renderer:

```python
from quarry.dsl.parser import DSLParser
from quarry.types.dsl.render import ConditionRenderError, render_condition

rule = DSLParser().parse('rule "Read" condition: messages[0].content == "🔥"')
written = render_condition(rule.condition)
assert DSLParser().parse('rule "Read" condition: ' + written).condition == rule.condition
```

`render_condition` preserves the grammar's precedence and literal/regex spelling semantics.
Its keyword limits default to `max_nodes=50_000`, `max_depth=128` and
`max_output=1_000_000`; each must be a positive plain integer. Rendering uses iterative graph
validation and output traversal. Unsupported or malformed nodes, cycles and exceeded limits
raise `ConditionRenderError`. Scalar and container subclasses are rejected rather than
converted through host callbacks. Compiler and visualization entry points convert rendering
failures to bounded diagnostics. Rendering a parsed condition does not require its source text.
Internal diagnostic placeholders retain their established display conventions; for example,
an empty internal transform displays as `messages.transform()`.
Extraction keeps its parsed operation and argument count: selection/alias receivers retain
function-call spelling, and explicit trailing `null` bounds retain positional spelling.

For batches, `rule_name` uses `conv_N:rule` for rule failures and `conv_N` for a whole-child
failure. The batch can also carry an unattributed `unknown` record for its aggregate error
summary. An empty batch has `invalid_input`. Both matched and unmatched complete scans have
an empty `failures` tuple. The projection follows active legacy failure fields even when a
caller constructs or updates a `ScanResult` directly. Optional `error_category` and
`evaluation_error_categories` supply typed attribution for those existing error fields;
metadata without its corresponding active error does not create a failure. Known enum
members and category strings are accepted as attribution. Malformed or unrecognized supplied
metadata projects to `unknown`, so mutable caller metadata cannot interrupt failure reporting.
A missing per-rule category preserves the legacy `evaluation` default. Construct explicit
`ScanFailure` records with a `ScanFailureCategory` member; construction validates its category
and rejects invalid values. `rule_name` accepts actual strings and genuine string subclasses,
which are normalized to plain string values for serialization. Runtime exception categories use actual inheritance, including
genuine exception subclasses, rather than a caller-controlled `__class__` attribute.

Batch aggregation retains both a child's top-level error category under `conv_N` and its
per-rule categories under `conv_N:rule`. File `input_read` failures describe the read/decode
stage; a later scan execution failure retains its exception-based category.

## Prepared policies

`compile_policy(source, policy_parameters=...)` creates a reusable `CompiledPolicy`. Supply scan parameters with `policy.scan(conversation, parameters=...)` or its async and batch methods. A prepared policy can report its decision tree and produce versioned bytes:

```python
from quarry import compile_policy, load_compiled_policy

policy = compile_policy('rule "Greeting"
condition:
    user_said("hello")
')
compiled_bytes = policy.to_bytes()
restored = load_compiled_policy(compiled_bytes)
result = restored.scan({"messages": [{"role": "user", "content": "hello"}]})
result.raise_for_status()
```

Authenticate stored compiled bytes before passing them to the loader. The loader checks the wire version, compiler version, rule requirement, and runtime resource identity; it does not prove arbitrary bytes came from your build. Recompile source after an incompatible version change. Policies with bound text parameters also require the host to manage their evidence key and follow the loader's key requirements.

See [parameters](parameters.md) for binding and digest behavior. Public result, exception, conversation, and message types are exported from `quarry`; configuration classes live in `quarry.config`.

## Judge results

`ModelInferenceBackend.model_judge_messages(alias, constitution, messages, tools=None, timeout=None)` is async and returns the public immutable `JudgeResult`:

```python
judge = await backend.model_judge_messages("judge", "Detect unsafe requests.", messages)
assert 0 <= judge.score <= 1
payload = judge.to_dict()  # {"score": 0.8, "reason": "..."} or reason=None
```

The score is calibrated exactly as in `model_score_messages`; both methods share the same request cache. The reason is present only when the configured alias enables `score.explanation` and its adapter returns a valid reason. Neither absent reasons nor float-only adapters imply an error. Refused, malformed and failed calls raise the existing model errors and never return a successful judge result. See [model-inference.md](model-inference.md#optional-judge-explanations) for limits and safe handling of untrusted generated text.

For operation-aware model input limits, see [input trimming](input-trim.md).
For scoped call collections and their aggregate evidence, see
[message-local tool calls](message-local-tool-calls.md).
