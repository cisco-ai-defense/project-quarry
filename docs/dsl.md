# Quarry DSL guide

A `.rule` file contains one named rule. Quarry parses, validates, links, and compiles it when the scanner loads. The [generated language reference](../quarry/language/reference.md) lists current functions, fields, operators, and runnable examples from the implementation.

```text
rule "Suspicious request"
metadata:
    severity = "medium"
strings:
    $blocked = ["reveal a secret", "ignore the policy"] nocase
variables:
    has_blocked_text = any msg in user_messages where msg.content contains_any $blocked
condition:
    has_blocked_text
```

Optional blocks appear in this order: `metadata`, `strings`, `variables`, `parameters`. Every rule needs `condition:`. A model span rule can also have a `collect:` block after `condition:`. Use `//` for line comments; `#name` is the occurrence-count operator.

Use `=` to define values in `metadata:`, `strings:`, and `variables:`. Use `==` to compare values in `condition:` expressions, including predicates inside `matches { ... }`. Other comparisons remain `!=`, `<`, `<=`, `>`, and `>=`.

```quarry
rule "Shell call"
strings:
    $needle = "shell"
condition:
    any call in tool_calls where call.name == "shell"
```

## Data and expressions

Rules read canonical `messages`, role-filtered message collections, `tool_calls`, `tool_responses`, and offered `tools`. The scan input is an OpenAI-shaped object with a required `messages` array and optional `tools` array. See [messages.md](messages.md) for its accepted form.

Use `any` and `all` comprehensions to inspect collections. String declarations support literal, whole-word, Base64, and RE2 matching. Arithmetic, numeric comparison, field access, array indexing, and `as` capture are available where the compiler can type and lower them. A condition must produce a Boolean verdict.

An `as` capture name cannot also be declared in the same rule's `variables:` block. Quarry rejects that ambiguity when the rule loads.

```text
rule "Repeated shell requests"
condition:
    count([msg for msg in user_messages if msg.content contains "sudo"]) >= 2
```

## Event flow

Flow operators read the finite conversation trace. `A after B` requires a witness for `B` before a witness for `A`; `right_after` requires adjacent relevant events. `before`, `right_before`, `until`, and bounded `within` forms are also available. Use parentheses when combining flow and Boolean logic.

`until` sustains its left condition until the right condition occurs. A text capture referenced on the sustained side is rejected at load time because it may need a different value at each event. Capture outside the sustained condition or on the release side.

```text
rule "Read then send"
condition:
    called_tool("send") after called_tool("read")
```

The compiler rejects combinations it cannot represent faithfully. It also limits automaton construction work, state growth, and capture complexity. Split a large rule into smaller rules or shorten its flow window when it hits a compile limit. See [rule-writing.md](rule-writing.md) for examples.

## Parameters, models, and outputs

`parameters:` declares typed policy or scan values. `compile_policy` binds policy values once and accepts scan values for each evaluation. [parameters.md](parameters.md) covers the API.

`model("alias")` calls an alias from `config_version: 2` configuration. Supported operations include scoring, classification, completion, token counting, splitting, and spans, subject to adapter capabilities. `collect:` currently accepts a direct `model(...).spans(...)` source with a typed filter and returns a collection on a match. See [model-inference.md](model-inference.md).

A conversation message can stand in for a model input. For example, `user_messages[-1]` supplies only the newest user message. If that message does not exist, the input is empty and the model call is skipped. For a model input built from several values, `join([value for item in collection if test], "\n")` can supply the content of `message(...)`. A joined input with no content also skips the model call. See the [joined tool-command example](../quarry/language/examples/joined-tool-commands.rule).

Rule metadata can contain an optional `requires_quarry = ">=0.1.0"` or `requires_quarry = ">=0.1.0,<0.2.0"` declaration. Quarry rejects an explicit incompatible requirement before compiling the rule. The declaration does not replace behavioral tests for a ruleset.
