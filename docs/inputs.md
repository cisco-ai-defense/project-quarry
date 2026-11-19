# Declared inputs

A conversation can carry named data beside its messages. The host declares each input section in `inputs:`. Rules read a list of items, such as `memory.writes`, or one object, such as `sandbox.state`, using the same field access and flow syntax as conversation data.

```yaml
inputs:
  memory.writes:
    content: [content]
    facts:
      operation: {type: string, values: [save, delete]}
      source: {type: string, optional: true}
  sandbox.state:
    single: true
    facts:
      mode: {type: string, values: [enforce, monitor]}
```

A host may pass the same mapping as `QuarryConfig(inputs={...})`. Names use lowercase letters, digits, and underscores. A section name has the form `<namespace>.<name>`. Every list item has a nonempty `id` unique within its section. `content` fields hold text; `facts` have types `string`, `integer`, `number`, `boolean`, `list` of a scalar type, or `object` with declared fields. Optional fields may be absent or `null`. Undeclared fields and wrong types are invalid. Declarations have no defaults.

The conversation carries sections under their namespace:

```json
{
  "messages": [{"role": "user", "content": "Summarize this."}],
  "memory": {"writes": [
    {"id": "w1", "content": "Remember the result", "operation": "save"}
  ]},
  "sandbox": {"state": {"mode": "enforce"}}
}
```

A rule can read the items and fields directly:

```text
rule "Memory write in enforce mode"
condition:
    (any w in memory.writes where w.operation == "save")
    and sandbox.state.mode == "enforce"
```

`memory.writes` can be ranged over or sliced; `w.id` and declared fields are available. A misspelled section or field fails when the rule loads. A namespace with no section value leaves that section unobserved, so a rule that needs it is skipped and the scan is incomplete. An explicit `[]` means an observed empty list. An absent `messages` section similarly leaves chat rules unobserved; absent `tools` means no tools.

A public-rule import checks its argument as that rule's messages. Local `as` aliases in a called rule belong to that check and may reuse a caller's alias name. Defining the same alias twice in one `and` chain is rejected.

Only sections read by selected rules are validated. A bad section fails rules that read it while other rules retain their verdicts. An unknown top-level key makes the scan invalid. Each read section and the chat have their own size and nesting limits.

## Text spans

Declared content fields support string spans and bounded extraction windows:

```text
rule "Located memory text"
strings:
    $p = "Remember"
condition:
    any w in memory.writes where
        w.content.extract($p.spans(w.content), right=12) contains "Remember"
```

Input evidence identifies its section, item ID, content field and character offsets.
Selecting or combining input projections retains their canonical owners. Equal text from
different input objects stays distinct. Local text transforms map detections back through
their character provenance while extraction from the selected message retains its current
text. A model's message indices always refer to its selected input order.

Span searches follow each selected occurrence, including different transformed copies of
one source message. Indexed `.content` and `.result` sources retain their owning selection;
reading message content alone does not decode its tool arguments.

Span collections support `union(first, second)` and `outermost(spans)`. Union removes
duplicate spans; outermost removes spans contained by another span. With limits enabled,
each union input and its deduplicated result must fit `max_comprehension_size`. Span
filters, quantifiers and outermost inputs use the same bound. Exceeding it fails the
evaluation without truncating results. These operations also honor the scan deadline;
disabling size limits does not disable that deadline.

Compiler identity is 26 with compiled-policy format 10. Recompile older artifacts
from their saved rule sources; current artifacts reload without recompilation.

For declared-input scans, plugin decision paths hash only the rule's selected, validated data. Unread sections do not change `conversation_sha256`; it is null when the selected data cannot be safely hashed.

## Timeline and clocks

A list section keeps its own item order. To compare its items with chat steps or items from another section, provide a `timeline` listing the observed message, tool call, and input item IDs in order:

```json
"timeline": [
  {"message_index": 0},
  {"tool_call_id": "call-7"},
  {"input": "memory.writes", "id": "w1"}
]
```

The timeline must contain each read item exactly once and preserve chat order. Without a needed timeline, a cross-source flow is incomplete. A flow over chat alone counts chat steps. A flow that reads input items counts timeline entries; `within` and `right_after` therefore include intervening input items. A bounded input flow also needs the sections whose items could occupy steps between its operands, unless a static condition already settles it.

An `until` flow that sustains one list section and releases on another also requires their
timeline when both can affect the verdict. Declaration order never supplies that ordering.
An `until` over a single section keeps its known item order; empty operands and conditions
already settled without ordering retain their verdicts.

## Native data adapters

An `InputAdapter` converts native data under one conversation key into requested sections. Configure its key and registered adapter name in `adapters:`. The adapter declares the sections it can fill and returns only the sections the scanner requests. The host can supply trusted adapter instances to `DSLScanner(input_adapters={...})`; packages can register factories under the `quarry.input_adapters` entry point group. Adapter name and version, and every input declaration read by a compiled policy, are part of that policy's identity.

Native data is checked against `max_total_size` before conversion. This preflight counts compact JSON bytes in UTF-8, including integer digits, boolean spelling, keys and punctuation. Printable Python integers can use the available byte budget up to the interpreter's active integer conversion limit. A value that cannot be printed within that limit fails the size check before the adapter runs.

The plugin's `quarry_reference` response lists configured sections under `inputs` so rule authors can discover their fields.

Declared inputs preserve lazy chat-field reads. Freeform argument text and non-object JSON
expose no named fields. Non-finite numbers and excessive decoded nesting fault only rules
that read those decoded arguments, with `invalid_input`; independent input, tool, and
name-only call rules retain their results. Raw model input preserves the original argument
text without forcing a decoded-field read. Unreadable senders, calls, or causal links still
prevent model dispatch.
