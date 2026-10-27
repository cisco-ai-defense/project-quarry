# Project Quarry: String Matching

Quarry supports literal, whole-word, Base64, and RE2 regular-expression matching on text-valued conversation fields. A declaration can hold one string or an array of strings:

```text
rule "Credential terms"
strings:
    $terms = ["api key", "password", "access token"] nocase
condition:
    any msg in user_messages where msg.content contains_any $terms
```

| Form | Use |
| --- | --- |
| `text contains "literal"` | Search text for a literal. |
| `text contains_any ["a", "b"]` | Search for any literal in the list. |
| `$pattern in text` | Search using a declaration and its modifiers. |
| `text ~ "pattern"` | Search with an inline case-insensitive RE2 pattern. |

Declarations accept `nocase`, `fullword`, `base64`, and `regex` modifiers. `nocase` ignores case but preserves source casing in captured text. `fullword` wraps a literal with word boundaries. `base64` encodes the declaration before matching. `regex` interprets the declaration as an RE2 pattern.

```text
strings:
    $word = "admin" nocase fullword
    $encoded = "payload" base64
    $email = "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+" regex
```

Quarry uses RE2 to avoid backtracking blowups. RE2 does not support lookbehind or backreferences. Rules must use regex forms the parser, validator, and compiler accept. Invalid declared patterns fail while rules load. For inline or computed patterns, test the exact rule and input shape; unsupported expressions fail closed rather than switching to another regex engine.

Use `as` to capture a matched substring for a later condition:

```text
rule "Repeated marker"
strings:
    $marker = "TOKEN=[A-Za-z0-9]+" regex
condition:
    any msg in user_messages where
        ($marker in msg.content as found) and
        found in msg.content
```

The [generated language reference](../quarry/language/reference.md) has the current operator and helper signatures. The [rule-writing guide](rule-writing.md) covers how to combine text checks with event flow.
