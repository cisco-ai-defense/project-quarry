# How Quarry works: rules, models, and witnesses

Quarry scans an agent conversation using rules you choose. Rules test message text, structured tool arguments, tool results, counts, and event order. A scan reports matching rules and, if you request an explanation, the conversation evidence behind each match.

You can use local checks alone or add a configured model for questions that need a semantic judgment. Combining model-derived signals with explicit rules is the **neurosymbolic** approach in Quarry. Local-only rules use the same rule engine without a model call.

## The symbolic and model parts

The **symbolic** part is the rule's explicit logic. For example, a rule might require an agent to call `read_file` on a sensitive path before it calls `http_request` for a particular host. Each local check answers a small question, such as “Does this tool call's `path` contain `.env`?” The rule combines those answers with `and`, `not`, counts, and event-order operators such as `after` and `until`.

Quarry expresses event order using **LTLf** (linear temporal logic on finite traces). The trace is the conversation supplied for this scan, from its first event to its last. `A after B` asks whether a qualifying `B` occurred before a qualifying `A`. A condition about absence or “all messages” applies to this supplied trace; another scan with more events can reach a different verdict.

Quarry compiles each rule into a **deterministic finite automaton** (DFA) when it loads the rule. The DFA has a finite set of states that track which parts of the pattern have happened. For the read-then-send example, one state can mean “waiting for a qualifying read” and another “a qualifying read has occurred.” At each event, the rule's checks determine the next state. Quarry can decide some matches before the trace ends and resolves remaining conditions at the end. Given the same check results, the DFA takes the same path; a configured model may return different results on separate calls.

Quarry uses a *symbolic* DFA: each state tests relevant checks instead of listing every possible combination of answers. During a verdict scan, Quarry skips a check when its answer cannot affect the current transition. That can avoid a configured model call.

The **model** part is optional. A configured model can score or classify selected conversation data. In a rule, you compare the model result with a threshold or category. Quarry turns that comparison into a true or false check, which the DFA combines with the rule's other conditions. Scores may be uncalibrated unless the model configuration includes calibration. The chosen adapter may send selected messages, tools, or policy text to its configured endpoint.

## From input to verdict

1. **Load the rules.** Quarry parses and validates the selected `.rule` files, checks required resources, and compiles their DFAs before scanning conversations. A rule that cannot compile prevents that rule generation from loading.
2. **Validate the conversation.** The input has an OpenAI-shaped `messages` array and may include offered `tools`. Quarry checks its structure, tool-call relationships, and configured size limits. It rejects an over-limit input instead of scanning a shortened transcript.
3. **Build the event trace.** Quarry keeps one event for each message and adds an event for each assistant tool call immediately after its assistant message. A tool result occupies one message event that rules can also inspect through `tool_responses`. Quarry exposes offered tool definitions and conversation-wide facts to rules without adding trace events.
4. **Evaluate the rule.** Quarry evaluates the local or configured model checks needed for each DFA transition. It caches check results for that scan. If you request an explanation, Quarry may run extra checks to collect evidence.
5. **Return the result.** A `complete` scan reports rule matches and any supported outputs for the supplied conversation. An `invalid` input or `incomplete` scan, such as one stopped by a timeout or evaluation error, has no clean verdict. Check the completion status before acting on matches or their absence.

Quarry numbers trace events separately from entries in `messages`. A message that contains a tool call contributes both a message event and a tool-call event, so an event position may differ from a message index.

Quarry decodes tool-call argument JSON when a rule reads those arguments. Malformed arguments can make that scan incomplete.

## A read-then-send example

The repository's [flow example](../examples/rules/flow_after.rule) looks for a call to `read_file` on a sensitive path followed by a call to `http_request` for a specified host:

```text
rule "Flow After"
condition:
    any http in get_tool_calls("http_request") where
        http.arguments.url contains "evil-corp.io"
    after
    any read in get_tool_calls("read_file") where
        read.arguments.path contains_any [".env", "credentials", "secrets"]
```

In the [matching conversation](../examples/data/flow_after.json), the agent calls `read_file` for `/app/.env`, receives a tool result, and later calls `http_request` for `https://backup.evil-corp.io/store`. The rule matches because both tool-call checks hold in the required order. The [reversed-order conversation](../examples/data/flow_after.negative.json) does not match, even though it contains the same kinds of calls.

The match establishes that the calls occurred in order with the specified arguments. To detect whether the request carried file content, write a rule that compares or captures the relevant values.

## What a witness tells you

A **witness** is Quarry's selected supporting evidence for a rule match. For the matching conversation above, the witness points to `tool_calls[0]` (`read_file`) and `tool_calls[1]` (`http_request`). The intervening tool result is part of the scanned trace, but those two calls support this rule's match.

For many rules that ask whether some event occurred, Quarry tries removing events and reruns the rule. It keeps a smaller set that still matches. Time and work limits can leave extra events in the witness, and another set may support the same match. A witness gives you one explanation, not necessarily the shortest possible one or a complete record of everything Quarry examined.

If a rule uses `within` or `right_after`, inspect the original event positions for spacing. Adjacent entries in a shortened witness may have had events between them in the conversation.

A rule about *all* user messages or the *absence* of an event may cite the whole relevant set, because removing an event would change the claim. A static condition or an `all` condition over an empty collection can match with no timeline events in its witness. An empty witness does not, by itself, mean that the rule failed.

Request explanations with `--explain` in the CLI or `enable_explain=True` in `DSLScanner`. The CLI displays a match summary with its witness. The Python explanation also exposes per-check results, including checks for rules that did not match; those rules have no supporting witness.

Quarry truncates excerpts and masks some recognized secret patterns, but you should still treat them as conversation data. Displayed evidence has caps with omitted counts. A witness supports the rule's stated condition; it is not proof of an agent's intent, an actual data transfer, or the correctness of a model judgment.

## Benefits and limits

The DFA tracks event order as Quarry scans, so a rule can distinguish read-then-send from send-then-read. Quarry validates rules before scanning and skips checks that cannot change the current transition. You can add model checks for semantic questions and use witness paths to find the events behind a match.

Quarry sees the conversation you supply, not unrecorded activity. A rule match means its stated pattern held on that input; it does not certify a broader incident. Detection quality depends on your rule, input data, and any model you configure. Test both matching and clean conversations that reflect your application, and require a `complete` result before using a verdict.

Start with [getting started](getting-started.md) to run a scan. See [rule writing](rule-writing.md) for conditions and event flow, [messages and tools](messages.md) for accepted input, [model inference](model-inference.md) for configured model calls, and the [Python API](python-api.md) for result status and explanations.
