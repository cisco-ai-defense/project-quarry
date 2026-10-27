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
