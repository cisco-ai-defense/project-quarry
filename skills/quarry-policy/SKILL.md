---
name: quarry-policy
description: Use when writing, explaining, compiling, or testing Quarry rules for LLM conversations and tool use.
---

# Quarry policy workflow

Use the plugin's installed language catalog. Do not guess syntax or copy language details from
these instructions.

1. Call `quarry_reference` with `action: search` and a short description of the behavior.
2. Call `quarry_reference` with `action: get` for the relevant item and example IDs. Ask for as
   many as the request's ID limit allows per call (the tool schema gives the maximum), splitting
   a longer list across calls. Check `missing_ids` and `filtered_ids` before searching again: a
   `filtered_ids` entry means the ID was right and your own `kinds` filter excluded it, so widen
   or drop the filter rather than looking for another ID.
3. Write one complete rule, including its `rule` name and `condition` block.
4. Call `quarry_compile`. Fix every error before presenting or scanning the rule.
5. Call `quarry_scan` with one explicit `conversation` or an ordered `conversations` batch when
   behavior must be demonstrated. Pass one shared `parameters` map when the rule declares inputs.

Add parentheses when a condition mixes boolean, flow, comparison, or text operators. The
`operator_precedence` table says how they bind. It arrives with `action: index`; on any other
action, pass `include_precedence: true` to read it, and only once — it never changes.

A reference record lists only the keys that apply to it. A missing key means "not applicable",
so read the keys that are there and do not infer a value for one that is absent. `action: get`
names each tested example and whether each of its checks is meant to match; add
`verbosity: "full"` when the conversation a check scans is what you need to see.

Quarry deliberately rejects several spellings that look plausible, such as `messages()` for the
`messages` surface or `first(tool_calls)` for `tool_calls[0]`. Call `quarry_reference` with
`action: rejected` to read the full table before writing a rule, and read the `rejected_spellings`,
`operator_keyword`, or `suggestions` details on an `unknown_function` diagnostic instead of guessing
another spelling.

Treat a `rejected_spellings` row as an example to follow, never as text to paste. Its
`documented_form` and `documented_replacement` are spelled with whatever arguments the table
documents, so apply the pattern to the rule at hand and keep its own operands: given
`first(tool_calls)` → `tool_calls[0]`, a rule that wrote `first(messages)` becomes
`messages[0]`. Pasting the documented replacement would change what the rule detects.

For example, begin this request with reference search:

```json
{
  "action": "search",
  "query": "file read followed by HTTP request within three events"
}
```

Then retrieve the full records the search identifies:

```json
{
  "action": "get",
  "ids": [
    "function:called_tool",
    "construct:flow-after-within",
    "example:flow-operators"
  ]
}
```

Apply the returned syntax to the application's exact tool names:

```quarry
rule "File data sent"
condition:
    called_tool("http_request") after(within: 3) called_tool("read_file")
```

Treat scan status as part of the result:

- Use `matched` only when `completion_state` is `complete`.
- Do not treat `invalid` or `incomplete` as a clean non-match.
- Read taxonomy labels from `matches[].metadata`. Quarry copies rule metadata; it does not infer a
  taxonomy.
- Use documented fields such as `call.name`, `call.response.result`, and
  `call.arguments.<application_key>`. Compilation rejects unknown documented object fields.
- Report missing model, text, or prober configuration from the returned diagnostics. Do not add
  credentials or backend settings to tool arguments.
- Use only examples returned by `quarry_reference`; those examples were checked with the installed
  Quarry version.
- Treat `parameters:` as the rule signature. Supply every declared value and no undeclared values.
  Unmarked declarations are supplied per scan; add `policy` only for a value fixed on the compiled
  policy. `number` accepts finite JSON numbers but not booleans or strings; `text` accepts strings.
  Text is represented by SHA-256 in evidence, never by its raw value.
- For a batch, the rule compiles once, the shared parameter map applies to every conversation, and
  `results` preserves input order. Read each item's `completion_state` independently.
