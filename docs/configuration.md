# Project Quarry: Configuration

Quarry reads YAML configuration. Use `config_version: 2` for the current model API and point `rules` at reviewed `.rule` files or directories:

```yaml
config_version: 2
rules:
  - rules/
rule_timeout: 5.0
max_messages: 10000
max_message_size: 1000000
max_tool_argument_size: 1000000
max_tool_calls: 1000
max_total_size: 50000000
max_nesting_depth: 32
max_timeline_items: 10000
max_comprehension_size: 10000
max_model_calls: 20
enforce_limits: true
fail_on_timeout: true
```

Quarry sorts discovered rule files and rejects duplicate names. A directory loads its top-level `.rule` files. Use the CLI's `--recursive` option or `DSLScanner(recursive=True)` for nested directories. Unknown configuration fields fail validation.

The scanner validates and compiles selected rules before it accepts scans. The compiler also applies its own work and state limits to prevent rules with excessive automaton construction cost. A compile failure blocks the candidate rule generation. Input limits and scan deadlines apply when conversations arrive. `max_total_size` measures the in-memory footprint of the input graph, counting a shared Python object once; use the message, tool-call, and per-item limits alongside it. Breaching an input limit always fails the scan. Keep limits enabled for input you do not control, and call `ScanResult.raise_for_status()` before using matches.

## Named text

Use `texts` for longer reviewed instructions. Quarry resolves a file path relative to the configuration file:

```yaml
texts:
  policy:
    file: policies/policy.txt
```

An inline text value also works. A rule reads it with `text_ref("policy")`. Keep policy text and credentials out of logs.

## Model aliases

A model alias names an adapter, model, and the operations a rule may use. Here is a score alias:

```yaml
models:
  guard:
    adapter: litellm
    model: ${MODEL_ID}
    score:
      message_builder: constitution
      boundary_seed: 482913
      tokens:
        negative: "0"
        positive: "1"
```

Supported model fields can use an exact `${ENV_NAME}` reference. Set the value in the application environment, not in a committed configuration file. The [model guide](model-inference.md) covers `classify`, `complete`, `spans`, token counting, and application adapters. The examples use a local adapter and need no provider key.
