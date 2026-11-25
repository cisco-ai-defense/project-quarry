# Project Quarry: Messages and tools

Quarry accepts OpenAI-shaped conversations:

```json
{
  "messages": [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Summarize the report."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search",
        "description": "Search public documents",
        "parameters": {
          "type": "object",
          "properties": {"query": {"type": "string"}}
        }
      }
    }
  ]
}
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`. Assistant messages can
contain tool calls. A tool response must reference an earlier assistant tool call by ID.

Quarry rejects duplicate tool-call IDs, orphan tool responses, unexpected top-level fields,
non-finite numbers, and data that exceeds configured safety limits. Tool-call arguments are
decoded when a rule reads `call.arguments`; malformed JSON then makes that scan incomplete.

Messages, tools, and nested JSON values remain immutable during a scan. Serialization returns new
Python containers. Text, image URL, and input-audio parts retain their OpenAI wire shape.

Rules can inspect documented message and tool fields. A model call receives the selected messages
and the tool definitions passed by the rule.

Message content can be text or a non-empty ordered array of text, image URL and input-audio
parts, including in tool results. `content` and the tool-result `result` alias flatten only the
text parts; wire serialization preserves the original array and media.

Media readability is checked when a rule sends selected messages to a model or adapter transform.
Inline image URLs starting with `data:` must have a `data:image/<type>;base64,` envelope and a
standard Base64 payload, padded or unpadded. Quarry checks this envelope without decoding image
pixels. Image and audio parts in tool results cannot be read by models or transforms. Such a read
fails that rule with `invalid_input`; text-only reads and unrelated rules retain their results.
A part transform checks only its selected part types; a whole-message adapter reads all media in
the messages it receives. These checks precede cache reuse and provider dispatch. External image
URL restrictions remain enforced separately at the provider boundary.

## Immutable message composition

Python integrations can combine views from one canonical source:

```python
from quarry.core.conversation import Conversation
from quarry.types.messages import Messages

conversation = Conversation.from_openai(
    {"messages": [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}]}
)
combined = Messages.combine((conversation.messages[-1:], conversation.messages[:1]))
```

`combined` iterates in written order while each item keeps its immutable value, content,
provenance, `message_index`, and tool identity. Combination normally reuses the source message
objects. When independently evaluated transformed views contribute opposite sides of a tool
call/result pair, Quarry creates immutable relinked variants so the combined graph is
self-contained. Object identity is therefore not guaranteed.

Repeated selections retain each occurrence independently. Two variants of one source message
can have different roles, content layouts or transformed text while keeping the same source
`message_index`. Slicing, role views, composition and token-bounded splitting keep their written
order and preserve each variant's provenance. A model span binds to the particular occurrence
that produced it. If variants map the same character offsets to different original ranges,
`source_ranges(message_index, start, end)` is ambiguous and raises; use the bound model span's
`source_ranges` and `messages.extract(span)` instead.

Text splitting prepares a scan-local range index once for each message variant. Range lookup and
source-map copying are bounded across the entire split operation, including tokenizer probes and
overlap. Exceeding 4,000,000 range work units raises an evaluation error and leaves the scan
incomplete. Preparation yields for cancellation and checks the active evaluation deadline.
A failed or cancelled split cancels and joins its owned token probes before returning, including
when its caller repeats cancellation while probe cleanup is still running.
Compiler identity advances to 13 with compiled-policy format 9 unchanged. Earlier compiler
artifacts are explicitly refused; recompile their saved rule sources before loading them.

Empty parts have no effect. Source-backed parts must share one source; explicitly constructed text
messages can be placed before, after, or between source views. Constructed roles are restricted to
`system`, `developer`, `user`, and `assistant`, and constructors do not accept provider fields or
tool IDs. Composition accepts at most 256 parts, 256 constructed messages, and 10,000 messages
overall. A tool response must still follow the selected assistant call it answers before the value
can be sent to a model. Selections with repeated tool-call IDs or repeated results for one call are
rejected before model dispatch.

Conflicting variants that both retain the same tool-call or tool-result identity are refused
when the conflict is found. A text-preserving copy of a tool result, rewritten to an ordinary
non-tool role, can accompany the original result without duplicating its tool identity.

Constructed messages have stable scan-local identity but no source coordinates. Their text is
explicitly unmapped unless it was copied through a provenance-preserving field operation.
`Messages.source_ranges(...)` fails closed when a requested range touches unmapped text. Model
spans over transformed or constructed inputs expose exact mapped ranges in `source_ranges` and
set `source_unmapped` when any selected character has no original position. Untransformed model
spans keep their existing wire shape.
Adapter-derived text can also report `source_parts`, identifying the original message and optional
content-part index without inventing character offsets.

## Map messages into model input

A message-valued map keeps its selected items in written order and flattens each projected
message sequence into the model input. For example, with a configured `normalized` transform:

```text
model("judge").classify(
    [item.transform("normalized") if item.role == "tool" else item for item in messages])
```

Maps can also construct text messages from selected message fields, tool calls, or offered tools.
The optional `if` filter runs before the projection, so excluded items are not transformed or
constructed. The projection and filter cannot call a model. Nested maps preserve lexical scope;
source-backed items keep their selected content, source coordinates, and tool links.

`max_comprehension_size` limits both source and output items when safety limits are enabled.
Each map also has fixed caps of 10,000 source/output items and 2,000,000 retained text characters,
including when configurable limits are disabled. The existing 256 constructed-message limit
still applies. Exceeding a limit fails the rule before provider dispatch; input is not truncated.

## Transformed model inputs

A reviewed transform configured under `transforms` can be applied with:

```text
model("judge").classify(messages.transform("normalized"))
```

Transforms are lazy model-input expressions. They can be chained and combined with message slices,
role views, identity filters, and composition. Each result is immutable. Source messages keep
their transcript index, role, and tool links. Allowlisted operations preserve non-text parts;
registered adapters replace only the content-part types selected by their configuration.

A local adapter's transformed value is shared by repeated reads of the same source expression
within one rule evaluation. Each new scan computes a fresh value. Separate messages, content
parts, transform aliases, and source expressions remain distinct even when their text is equal.
Model-backed transforms retain provider-request reuse and model-call accounting. The separate
Python `transform_unit()` API retains its raw-part result cache.

Local transformed values have fixed per-rule retention bounds of 256 values and 256 MiB of
estimated retained state. Shared source graphs are conservatively charged per value. Exceeding
a bound fails the rule; values are never evicted and recomputed during that evaluation.

Quarry carries a character map through every operation. `Span.source_ranges` reports original
message and content-part offsets when a model span came from transformed text. Python integrations
can query the same mapping with `messages.source_ranges(message_index, start, end)`. Replacement
text maps to the source range it replaced. Deleted text contributes no output. Prefixes and
suffixes are insertions with no source coordinates, so mapping a range that touches one fails
closed.

For example, text extracted from an image can produce a span with `source_unmapped: true` and
`source_parts: [{"message_index": 0, "part_index": 1}]`. Its `start`/`end` index the derived
text. They are not positions in the image URL or encoded bytes. Part context survives subsequent
local transforms, slices and message composition. `Messages.source_ranges(...)` still requires
exact character provenance and rejects a range that has only part context.

Python hosts can use `messages.rewritten(wire_messages)` for a text-preserving change of role or
other message fields. The result keeps written order, transcript indexes and character provenance.
Unchanged wire values reuse the original selection. A changed message and its source must make
no tool calls; replacing text or adding/removing messages is rejected. A user turn can, for
example, become a tool result with the same text and an explicit `tool_call_id`. Its result link
is resolved in the derived view, while the original call/result graph stays unchanged. Model
dispatch still requires a self-contained sequence with the call before its result.

Copied media-derived text retains its owning source even when it has part context without
character offsets. Combining values from different conversations fails before model dispatch.
