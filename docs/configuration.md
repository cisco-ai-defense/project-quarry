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

The scanner validates and compiles selected rules before it accepts scans. The compiler also applies its own work and state limits to prevent rules with excessive automaton construction cost. A compile failure blocks the candidate rule generation. Input limits and scan deadlines apply when conversations arrive. `max_total_size` measures the aggregate in-memory footprint that validation and canonicalization materialize: each repeated container occurrence is charged separately, while shared immutable scalars may be identity-deduplicated. Use the message, tool-call, and per-item limits alongside it. Breaching an input limit always fails the scan. Keep limits enabled for input you do not control, and call `ScanResult.raise_for_status()` before using matches.

With `enforce_limits: true`, a span `any` or `all` checks its complete input against
`max_comprehension_size` before evaluating its predicate. The same bound applies to
direct span sources, filters, `outermost()`, and the distinct result of `union()`;
combining individually permitted sources cannot bypass it. With limits disabled,
pattern-derived quantifiers can stream and short-circuit without materializing
the complete collection. Rule deadlines still apply in either mode.

Raw conversation objects must use exact built-in `dict` and `list` containers and exact string object keys. Mapping wrappers and container subclasses are rejected without invoking their accessors. Tool schemas receive JSON type, cycle, depth and aggregate-work validation before canonicalization. An input rejected before scanning has no verdict or conversation digest and is never hashed for exploration. Declared sections are validated only when read by a selected rule; see [inputs](inputs.md).

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
    description: Policy judge; 1.0 means the conversation violates the supplied policy.
    adapter: litellm
    model: ${MODEL_ID}
    score:
      message_builder: constitution
      boundary_seed: 482913
      tokens:
        negative: "0"
        positive: "1"
```

`description` is optional and limited to 500 characters. Use it to explain what the alias is for
and what its output means. The agent plugin returns it beside the alias's supported operations so
rule authors can select the right model. Descriptions are human-facing metadata: changing one does
not change provider requests, model-result cache keys, or compiled-policy model digests.

Supported model fields can use an exact `${ENV_NAME}` reference. Set the value in the application environment, not in a committed configuration file. The [model guide](model-inference.md) covers `classify`, `complete`, `spans`, token counting, and application adapters. The examples use a local adapter and need no provider key.

## Model templates and inheritance

Use `model_templates` to share reviewed provider and operation settings without creating a runtime
alias. A model or template can use `extends` with one other model or template:

```yaml
model_templates:
  scored_model:
    adapter: application_adapter
    score:
      message_builder: constitution
      boundary_seed: 7
      tokens:
        negative: "0"
        positive: "1"

models:
  policy_score:
    extends: scored_model
    model: synthetic/policy-score
    description: Synthetic policy score used by the example.
    score:
      boundary_seed: 11
```

Declaration order does not matter. Nested mappings merge recursively and the child value wins.
Setting a field or section to `null` removes the inherited value. `description` belongs to one
runtime alias, cannot appear in a template, and is not inherited when one model extends another.
Template and model names share one namespace. Unknown parents, duplicate names, inheritance cycles,
unknown fields, and malformed sections fail configuration loading, including errors in templates
that no model uses.

Resolution does not mutate the parsed mapping. It is bounded to 256 total model/template
definitions, 32 inheritance levels, 16 nested container levels, 20,000 aggregate source values,
20,000 aggregate logical values, and 1,000,000 aggregate scalar/key bytes across all resolved
definitions after environment references expand. Each referenced environment variable is read once
before inheritance fanout is charged. Templates are load-time configuration only; after
resolution, models use the same strict validation and immutable runtime representation as fully
written-out aliases.
