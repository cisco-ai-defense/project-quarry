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
