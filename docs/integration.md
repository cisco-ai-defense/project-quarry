# Project Quarry: Application integration

Treat rules, configuration, text resources, model endpoints, credentials, adapter code, and compiled-policy bytes as trusted application inputs. Treat conversations and model responses as untrusted data. Store and review the two groups under separate permissions.

Prepare a scanner before accepting traffic. Quarry discovers, validates, links, admits resources for, and compiles all selected rules before publishing the generation. A rule that fails this process should block the deployment of that generation.

For each scan, validate the OpenAI-shaped conversation, enforce size and deadline limits, and require an authoritative result before acting. `QuarryGuard` performs the status check before it returns matches. With `DSLScanner` or `CompiledPolicy`, call it yourself:

```python
result = scanner.scan_data(conversation)
result.raise_for_status()
use_matches(result.rule_matches)
```

Keep conversation and text-parameter content out of logs, metric labels, exceptions, and traces. Treat validation errors as sensitive: malformed field names, role values, and tool-call identifiers can appear in diagnostics. Do not forward those diagnostics to public telemetry or clients without redaction. Model adapters define the network boundary. Register them with `ModelInferenceBackend`, verify their endpoints and credentials in the host application, and preserve cancellation and timeout behavior.

If you store a compiled policy, authenticate it before loading. The format is data-only and versioned, but its loader assumes the host has established provenance. Recompile source when the wire or compiler version changes.
