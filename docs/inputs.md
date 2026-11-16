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

Only sections read by selected rules are validated. A bad section fails rules that read it while other rules retain their verdicts. An unknown top-level key makes the scan invalid. Each read section and the chat have their own size and nesting limits.

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

## Native data adapters

An `InputAdapter` converts native data under one conversation key into requested sections. Configure its key and registered adapter name in `adapters:`. The adapter declares the sections it can fill and returns only the sections the scanner requests. The host can supply trusted adapter instances to `DSLScanner(input_adapters={...})`; packages can register factories under the `quarry.input_adapters` entry point group. Adapter name and version, and every input declaration read by a compiled policy, are part of that policy's identity.

The plugin's `quarry_reference` response lists configured sections under `inputs` so rule authors can discover their fields.
