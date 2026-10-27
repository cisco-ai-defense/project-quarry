# Project Quarry: Rule writing guide

Write one rule per `.rule` file. Quarry compiles each selected rule before scanning, so a rule that uses an unsupported expression or exceeds a compile budget fails while the scanner loads. Start with the smallest condition that captures the behavior you need, then add metadata and reusable strings.

Use `//` for standalone or trailing line comments. A `//` sequence inside a quoted string stays part of the string. Quarry does not support `#` comments or `/* ... */` block comments; `#name` counts string occurrences.

```text
rule "Privileged shell call"
metadata:
    description = "Detect a shell call that requests elevated privileges"
    severity = "high"
strings:
    $privileged = ["sudo", "doas", "runas"] nocase
condition:
    any call in tool_calls where call.arguments.* contains $privileged
```

The blocks `metadata`, `strings`, `variables`, and `parameters` are optional and must appear in that order. `condition:` is required. Use `//` for comments. See the [generated language reference](../quarry/language/reference.md) for every supported form.

## Match the right surface

Use `user_messages` for user text, `assistant_messages` for assistant text, `tool_calls` for calls, and `tool_responses` for returned data. The `tools` collection contains offered tool definitions, which are distinct from calls that actually happened.

```text
rule "Sensitive file read"
condition:
    any call in get_tool_calls("read_file") where
        call.arguments.path contains ".env"
```

A field path such as `call.arguments.*` searches top-level argument values. A missing dictionary field produces `null`; test the field you need before relying on it.

## Combine local checks and flow

`A after B` means Quarry found `B` before `A` in the finite event trace. `right_after` requires adjacent relevant events. A bounded `within` form limits the distance. Keep the operand order explicit and parenthesize compound conditions.

```text
rule "Read then network send"
condition:
    called_tool("http_request") after(within: 4) called_tool("read_file")
```

Flow proves event order, not data provenance by itself. Add captures or compare fields when a detection depends on the same value moving between events. The [synthetic examples](../examples/README.md) include flow, capture, count, and string rules.

An `as` capture must have a distinct name from every declaration in the same rule's `variables:` block. In an `until` flow, a text capture referenced by the sustained (left) condition is rejected at load time; move it outside that condition or capture on the release (right) side.

## Counts and filters

```text
rule "Repeated shell requests"
condition:
    count([msg for msg in user_messages if msg.content contains "sudo"]) >= 2
```

Use bounded conversations and a rule deadline for large inputs. `collect:` has a narrower purpose than a filter comprehension: it returns spans from a direct configured `model(...).spans(...)` call when the rule condition matches.

## Model-backed checks

A rule may use a model alias that the application configures and admits at load time. Keep the model endpoint and credentials in configuration and the host environment. A score comparison needs an explicit threshold and a tested provider response contract. See [model inference](model-inference.md).

```text
rule "High score"
condition:
    model("guard").score(messages, text_ref("policy"), tools) >= 0.8
```

Test expected matches and clean controls before deployment. Recheck the full result status: an incomplete scan is not a clean verdict.
