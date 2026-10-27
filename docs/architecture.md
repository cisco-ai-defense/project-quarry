# Project Quarry: Architecture

Quarry compiles each rule before it scans a conversation. The core path is:

```text
configuration + rule source
          |
  parse -> validate -> link calls -> prepare resources
          |
  lower expressions -> build symbolic DFA -> publish rule generation
          |
conversation -> normalize and bound input -> construct events
          |
    execute DFA with scan context, deadline, and cache
          |
    ScanResult with verdicts, witness evidence, and outputs
```

A scanner publishes a new generation only after every selected rule passes validation, resource admission, and compilation. Unsupported constructs and compiler complexity limits reject the rule at load time. There is no tree-evaluation fallback at scan time.

The compiled automaton reads a finite event trace. Events represent messages, tool calls, and tool results in conversation order. Flow operators such as `after`, `before`, `right_after`, and `until` operate on that trace. The compiler also handles closed, conversation-wide checks and capture bindings. The executor evaluates model-backed checks through the configured resource manager when a rule needs them.

`ScanContext` holds the canonical conversation, limits, deadline, parameters, and a cache scoped to one scan. Model operations use configured aliases and adapters; Quarry checks required aliases and operations while preparing rules. Provider calls still depend on the adapter's timeout, cancellation, and concurrency behavior.

`ScanResult` separates an authoritative `complete` verdict from `invalid` input and `incomplete` execution. Call `raise_for_status()` before acting on matches. Explain mode records witness evidence from the compiled execution. A matching rule may also return a typed collection through `collect:` when its source is a configured span model.

`compile_policy` prepares one parameterized rule for repeated scans. `CompiledPolicy.to_bytes()` writes a versioned, data-only artifact for policies without bound text parameters or `text_ref` resources. Serialization rejects those text-bearing policies because their private evidence key cannot safely travel with the artifact. `load_compiled_policy()` checks compatibility and required runtime resources. The host must authenticate stored bytes before loading them. Quarry does not treat arbitrary bytes as trusted compiler output.

For the precise DSL surface, use the [generated language reference](../quarry/language/reference.md). The [integration guide](integration.md) covers the application trust boundary.
