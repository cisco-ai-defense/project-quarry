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

## Content transforms

Version 2 model inputs can apply named, reviewed local transforms to message text:

```yaml
transforms:
  normalized:
    operations:
      - op: strip
      - op: casefold
      - op: replace
        old: "legacy-label"
        new: "category"
```

Use the transform as `messages.transform("normalized")`. The receiver can also be a message
selection, filter, slice, composition, or another transformed value. Transform names are literal
and must exist when the rule compiles.

The operation allowlist is `strip`, `casefold`, literal `replace`, `prefix`, and `suffix`.
`replace.old` must not be empty. Operations never execute custom code, read files, call a model,
or resolve another transform. Configuration is immutable and rejects unknown fields and value
types.

Transforms apply independently to string content and each text part of multimodal content.
Non-text parts, roles, message indexes, and tool-call/response links are preserved. Inserted text
has no source position; a span that touches it is rejected instead of receiving invented
coordinates. Replacements map to the complete source range they replaced, deletions map no output,
and unchanged characters retain exact message and content-part offsets.

The registry supports at most 64 transforms; an operation plan has at most 16 operations. A rule
may apply at most 16 transforms in one expression. Operation-plan work is capped at 1,000,000 input characters,
2,000,000 output characters, 8x expansion plus 4,096 inserted characters, 8,192 provenance
segments, and 4,000,000 character-work units. `strip` uses Unicode whitespace rules; `casefold`
processes each Unicode character deterministically and maps expansions back to that character.

Named transforms can also use adapters registered by the host application:

```yaml
transforms:
  upper:
    adapter: uppercase
    settings: {}
    parts: [text]
  extracted:
    model: extractor
    parts: [image_url]
```

The `adapter` form receives an immutable `LocalTransformConfig` containing bounded JSON
`settings`. The `model` form names a configured model alias and passes its `InferenceModelConfig`
to that model's adapter. These forms cannot be mixed with each other or with `operations`.
The host registers adapter instances with `ModelInferenceBackend`; configuration never imports
adapter code. An adapter declares `transform_parts` as a tuple of supported types (`text`,
`image_url`, `input_audio`) and implements `transform(config, part, *, timeout=None)`:

```python
class Uppercase:
    local = True
    transform_parts = ("text",)

    def transform(self, config, part, *, timeout=None):
        return part["text"].upper()


backend = ModelInferenceBackend(config, adapters={"uppercase": Uppercase()})
```

The method may be synchronous or asynchronous. Synchronous adapters must check deadlines during
long work. A transform adapter must declare a boolean `local`: `True` for an `adapter` entry and
`False` for a `model` entry. A mismatch fails while loading rules. Omitting `parts` selects every
type the adapter declares; other parts pass through. Adapter capabilities may be a tuple or
`frozenset`; configuration `parts` remains a list.
Selected media parts become text parts in their original positions. Local work does not consume
model-call budget. Model-backed work uses the configured alias's limiter, timeout, retry and
model-call budget, sharing identical requests and validating results before cache admission.

An adapter returns a string or `TransformResult(text, runs)`. Each immutable `TransformRun`
contains an output `[start, end)` range, optional `source_start`/`source_end` character offsets,
and `kind="linear"` or `"whole"`. Runs must cover the output in order, without gaps or overlap.
Linear ranges have equal input/output lengths; whole ranges associate their output with the
complete source range. A changed plain string reply maps to the entire input text. Media output
has no input character offsets and instead reports its source message/content part. Results are
bounded and validated before caching; malformed results are errors, not provider refusals.

Prepared policies contain a configuration fingerprint rather than adapter settings. Reload them
with the same transform registry and registered capabilities. Changed or missing definitions fail
at load time. Model-backed transforms cannot run inside local map values, selection predicates,
or conditional predicates.

A local adapter can instead implement `transform_messages(config, wire_messages)`. This method
receives the selected messages once and returns one wire message per input in the same order.
The operation runs locally without caching, model-call charges or rate-limiter tickets. Do not
set `parts` for a whole-message adapter. If both transform methods exist, `transform_messages`
takes precedence.

Each output must be the original wire value or a valid message holding the same flattened text.
A changed message and its source must make no tool calls. For example, a host can reinterpret an
assistant's plain text as a user turn:

```python
class AssistantTextAsUser:
    local = True

    def transform_messages(self, config, messages):
        return [
            {"role": "user", "content": message["content"]}
            if message["role"] == "assistant"
            and not message.get("tool_calls")
            and isinstance(message.get("content"), str)
            else message
            for message in messages
        ]
```

Role views and tool-result associations follow the rewritten value. Message indexes and character
provenance still point to the source conversation. A changed non-text layout retains message-level
context rather than inventing original content-part indexes. The current message-view contract
requires repeated occurrences of one source message to receive the same rewrite; conflicting
variants fail instead of overwriting an earlier occurrence.

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

## Process result cache

Long-lived plugin servers can bound the process-wide cache of completed model, regex, and prober
results by age, result count, and retained-result bytes:

```yaml
model_cache:
  ttl_seconds: 43200
  max_entries: 100000
  max_bytes: 268435456
```

The defaults shown above retain a result for 12 hours after its last use, up to 100,000 completed
results and 256 MiB of estimated result data. When either aggregate bound is exceeded, Quarry
drops completed results from least to most recently used. A result larger than `max_bytes` is
returned to its caller but is not kept in this process table. Calls still in flight remain shared, do not count
toward either aggregate bound, and begin their TTL and byte accounting only when they complete.

`max_bytes` is a deterministic estimate made when a result is inserted, not process RSS or a
guarantee about allocator overhead. It sums runtime-reported shallow sizes across exact built-in
containers and their contents; strings, bytes, bytearrays, numeric `array.array` values, and known
memoryview backing buffers include their payload. Dataclass and `SimpleNamespace` fields are also
followed. Shared references and cycles count once within one result. Cache keys, cache bookkeeping,
external allocations, and arbitrary objects' hidden state are not included. A result graph that
exceeds the byte bound or the fixed accounting-work bound is treated as oversized instead of
being traversed without limit. If retained-size inspection fails, the result is returned but not
cached. This does not detect every concurrent mutation: a change that does not make inspection
fail can go unnoticed. Cached results should not be mutated; growth after insertion is not
remeasured.

## Accepted model replies during a scan

A scan keeps the first accepted raw reply for each identical model request until its work,
including deferred evidence, finishes. Process-cache eviction, expiry, clearing, or reconfiguration
therefore preserves that scan's accepted reply. Aliases that send the same request share its raw
reply; each reader still validates and interprets it using its own settings. A reader that rejects
a reply invalidates it for later reads. Failed or cancelled reads do not turn an unaccepted reply
into an accepted one.

Scan-held references are separate from the process table. They can retain replies that exceed
`max_bytes`, and are outside that byte total. The limit controls process-cache entries, not total
memory retained by active scans or their deferred work. Those references are released when their
scan work finishes, fails, is cancelled, or its deferred owner is closed. Other scans continue to
use the process cache and shared running requests normally.

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

Prepared policies bind registered transforms to their complete reviewed definition. This layer
uses compiler 11 and compiled-policy format 9; policies from earlier formats must be recompiled
from their saved rule source. The format change records transform resource requirements.

Adapter diagnostics omit host exception messages and arbitrary exception details. Known retry
and refusal codes are retained so model accounting and recovery still apply.
