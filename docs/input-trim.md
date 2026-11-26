# Operation-aware model input trimming

Use `trim()` directly as the messages operand of `model().score()`, `classify()`,
`complete()` or `spans()`:

```text
model("judge").classify(trim(messages, 200, from="end", cut_keep="none")) > 0.5
```

`from="end"` keeps the newest contiguous body; `from="start"` keeps the oldest.
The default is `"end"`. `cut_keep="start"` (the default) permits a fitting text
prefix of the boundary message; `"none"` keeps whole messages. A partial cut never
splits non-text content. Selection stops at its first non-fitting boundary and
constructs one chunk. It does not calculate every split chunk and discard the rest.

A written size is a positive integer measured with the
configured split tokenizer over the selected messages. It includes context but
does not separately reserve score policy or offered tools. Closed integer
arithmetic can subtract counts of written text using the same alias:

```text
model("judge").classify(
    trim(messages, 200 - model("judge").input_tokens(message("user", "fixed text")))) > 0.5
```

Such counts accept constructed written text, text parameters and `text_ref()`;
they cannot hide completion/classification requests or count a different alias's
vocabulary. Arithmetic work is bounded before each operation using operand bit
lengths; the final size must be positive. Existing private-text
serialization restrictions still apply.

## Implicit operation budgets

Without a written size, the consuming operation must declare `max_input_tokens`:

```yaml
classify:
  class_index: 0
  max_input_tokens: 2048
```

The same optional setting belongs to `score`, `complete` and `spans`. It supplies
the budget for implicit `trim()`; it does not enable automatic truncation of other
model calls. Unset limits retain the previous model configuration identity. Both
written and implicit trimming require the alias's existing `tokens` configuration.

Implicit scoring counts the actual score-builder messages, including policy and
rendered tools. Other operations count their canonical messages and offered tools
through the configured token adapter. That adapter owns its tokenizer/framing
conventions. A final complete-request count confirms the selected input; adding
individual message counts is insufficient when framing or tools cost nonlinearly.
These are configured tokenizer measurements, not billing or provider-throughput
guarantees. A custom backend supporting this syntax implements `model_trim_messages`.

## Context, provenance and outcomes

`context=messages[:1]` moves those actual source occurrences to the front and
removes them from the body being trimmed. Source positions and partial-character
ranges remain attached, so model spans still point to their original input.
Constructed context, such as `message("system", "fixed context")`, combines under
the existing construction, byte and provenance bounds. Empty optional context is
ignored. Context that alone exceeds the budget fails before the consuming request.

Empty main input or a missing required policy makes no consuming request or trim
tokenizer probes. Empty span inputs return a typed empty collection. Existing
input/link validation stays active. A failed tokenizer, model-call allowance,
work limit, timeout or cancellation never becomes a validated zero or clean
verdict. Physical tokenizer attempts and the consuming request share the ordinary
model-call budget and inspection deadline. There is no detached or post-decision
trimming task.

Explanation arguments contain `trim_omitted_messages` and
`trim_omitted_characters`; the latter counts discarded characters of a partial
boundary message. They contain no omitted input text. Logical-read reports include
the tokenizer probes as `input_tokens` reads and the final consuming operation
separately. Enabling logical accounting adds no trimming or provider work.

Prepared policies bind each trim to its exact consuming alias, operation and
policy/tool/threshold operands. Loaders reject forged owners, context shapes and
non-closed budget expressions. Compiler 29 requires recompiling policies from
earlier compiler versions; the serialized envelope remains format 11.

Trimmed inputs obey normal exact call/result-link validation. A cut that removes a
tool result's originating call fails before the consuming request. Partial text
cuts and moved context do not grant split's flattened-score-evidence exemption.
