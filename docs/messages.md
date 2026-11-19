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

Empty parts have no effect. Source-backed parts must share one source; explicitly constructed text
messages can be placed before, after, or between source views. Constructed roles are restricted to
`system`, `developer`, `user`, and `assistant`, and constructors do not accept provider fields or
tool IDs. Composition accepts at most 256 parts, 256 constructed messages, and 10,000 messages
overall. A tool response must still follow the selected assistant call it answers before the value
can be sent to a model. Selections with repeated tool-call IDs or repeated results for one call are
rejected before model dispatch.

Constructed messages have stable scan-local identity but no source coordinates. Their text is
explicitly unmapped unless it was copied through a provenance-preserving field operation.
`Messages.source_ranges(...)` fails closed when a requested range touches unmapped text. Model
spans over transformed or constructed inputs expose exact mapped ranges in `source_ranges` and
set `source_unmapped` when any selected character has no original position. Untransformed model
spans keep their existing wire shape.

## Transformed model inputs

A reviewed transform configured under `transforms` can be applied with:

```text
model("judge").classify(messages.transform("normalized"))
```

Transforms are lazy model-input expressions. They can be chained and combined with message slices,
role views, identity filters, and composition. Each result is immutable. Source messages keep
their transcript index, role, multimodal non-text parts, and tool links.

Quarry carries a character map through every operation. `Span.source_ranges` reports original
message and content-part offsets when a model span came from transformed text. Python integrations
can query the same mapping with `messages.source_ranges(message_index, start, end)`. Replacement
text maps to the source range it replaced. Deleted text contributes no output. Prefixes and
suffixes are insertions with no source coordinates, so mapping a range that touches one fails
closed.
