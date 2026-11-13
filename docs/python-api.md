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
