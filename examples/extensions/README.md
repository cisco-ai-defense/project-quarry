# Registered extension examples

Run from the repository root:

```console
uv run --frozen python -m examples.extensions.demo
```

`memory_extension.py` and `sandbox_extension.py` stand in for independently installed adapter packages. Each defines Pydantic schemas, an `ExtensionSpec`, and a function that turns an observed action into a JSON event. The host registers both specs on one scanner, then supplies the events and optional state for each scan.

The memory scan uses a complete timeline to order a confirmed write immediately after a tool call. The sandbox scan uses only a sandbox event and policy state; it needs no synthetic conversation. Each scan selects the rules for the facts it supplies. The rule names and schemas are examples of the general extension interface.

See the [extension guide](../../docs/extensions.md) for the event contract, missing-input behavior, and clock semantics.
