# Quarry language reference

This file is generated from the Python functions and grammar that Quarry runs.
Do not edit it by hand. Run `python -m scripts.generate_language_spec` after changing
a public function, surface, or grammar form.

## Functions

| Function | What it does |
|---|---|
| `conversation() -> model_input` | Select the complete conversation, including offered tool definitions. |
| `count(value: collection \| string, pattern: string = null) -> integer` | Return the size of a collection or text, or count a pattern in text. |
| `if_refused(value: boolean \| number \| string, fallback: boolean \| number \| string) -> boolean \| number \| string` | Return `value`, or `fallback` when a model call in `value` was refused. |
| `join(values: collection<string>, separator: string) -> string` | Combine (concatenate) the text values of a map comprehension with a separator. |
| `len(value: collection \| string) -> integer` | Return the size of a supported collection or the length of text. |
| `llm_split_conversation(model: string, input: model_input, max_input_tokens: integer) -> collection<model_input>` | Split model input into chunks no larger than `max_input_tokens`. |
| `message(role: string, content: string) -> model_input` | Build a one-message model input from a role and literal or field-backed content. |
| `model(alias: string) -> model_reference` | Select a configured model by its literal alias for a following method call. |
| `model.classify(messages: model_input, tools: collection<tool_definition> = null) -> number` | Return the configured classifier head's probability for these messages. |
| `model.complete(messages: model_input, tools: collection<tool_definition> = null) -> message` | Generate one assistant message from the supplied messages. |
| `model.input_tokens(messages: model_input, tools: collection<tool_definition> = null) -> integer` | Return the model-specific token count for the supplied messages and optional tools. |
| `model.score(messages: model_input, constitution: string \| text_reference, tools: collection<tool_definition> = null) -> number` | Score messages against policy text and return a number from 0 through 1. |
| `model.spans(messages: model_input, threshold: number) -> collection<span>` | Return original-conversation spans from the alias-configured span operation. |
| `model.split(messages: model_input, max_input_tokens: integer, overlap: integer = null, context: model_input = null) -> collection<model_input>` | Partition messages into model-sized chunks for one non-nested `any` score check. |
| `model_split(model: string, messages: model_input, max_input_tokens: integer) -> collection<model_input>` | Deprecated. Compatibility spelling for `model("alias").split(...)` in version 2 rules. |
| `refused(value: boolean \| number \| string \| message) -> boolean` | Return whether a `model("alias").score(...)` or `.complete(...)` call in `value` was refused. |
| `text_ref(name: string) -> text_reference` | Refer to configured policy text by its exact resource name. |
| `add_message(role: string, content: string) -> model_input` | Append one text message to a model input created with `message(...)`. |
| `assistant_said(text: string, case_sensitive: boolean = false) -> boolean` | Return whether any assistant message contains the supplied text. |
| `call_sequence() -> collection<string>` | Return assistant tool-call names in chronological order. |
| `called_tool(name: string) -> boolean` | Return whether the assistant called a tool with this exact name. |
| `called_tools(*names: string) -> boolean` | Return whether the assistant called every named tool. |
| `capability_tags(tag: string, text: string) -> boolean` | Return whether `text` describes the capability named by `tag`. |
| `count_tool_calls(name: string) -> integer` | Count assistant calls to the tool with this exact name. |
| `extract(span: span, left: integer = 0, right: integer = 0) -> string` | Return one span plus bounded context from its canonical source message. |
| `get_tool_calls(name: string) -> collection<tool_call>` | Return assistant calls to the tool with this exact name. |
| `get_tool_responses(name: string) -> collection<tool_response>` | Return results linked to calls of the tool with this exact name. |
| `has_sequence(*names: string) -> boolean` | Return whether these names occur contiguously in the tool-call order. |
| `has_tool(name: string) -> boolean` | Return whether a tool with this exact name was offered to the model. |
| `has_tools(*names: string) -> boolean` | Return whether every named tool was offered to the model. |
| `llm_complete(model: string, input: model_input) -> string` | Generate text from model input for comparison with a literal string or regex. |
| `llm_input_tokens(model: string, input: model_input) -> integer` | Return the model-specific token count for the supplied model input. |
| `llm_score(model: string, constitution: string \| text_reference, input: model_input) -> number` | Score model input against policy text and return a number from 0 through 1. |
| `model_scan(model: string, text: string, threshold: number, class_index: integer) -> boolean` | Return whether a classifier score is greater than a required threshold. |
| `model_score(model: string, text: string, class_index: integer) -> number` | Return a classifier probability for the selected output class. |
| `off_topic(text: string) -> boolean` | Return whether `text` has moved too far from the first user message. |
| `topic_drift(text: string) -> number` | Return how far `text` has moved from the first user message, from 0 through 1. |
| `unrequested_action(text: string) -> boolean` | Return whether `text` describes an action not requested in the first user message. |
| `user_said(text: string, case_sensitive: boolean = false) -> boolean` | Return whether any user message contains the supplied text. |

### `conversation`

`conversation() -> model_input`

Select the complete conversation, including offered tool definitions.

Examples: [examples/model-inference.rule](examples/model-inference.rule) (requires a configured backend)

### `count`

`count(value: collection | string, pattern: string = null) -> integer`

Return the size of a collection or text, or count a pattern in text.

`count(value)` measures a supported collection or text value. `count(text, pattern)`
counts occurrences of literal pattern text. When `value` is a filtered timeline
collection, compare the result with an integer from 0 through 8.

- `pattern`: must be written directly in the rule

Examples: [examples/indexes-and-counts.rule](examples/indexes-and-counts.rule)

### `if_refused`

`if_refused(value: boolean | number | string, fallback: boolean | number | string) -> boolean | number | string`

Return `value`, or `fallback` when a model call in `value` was refused.

`if_refused(model("guard").score(messages, text_ref("policy")), 1) > 0.5` counts a
refusal as a score of 1; `if_refused(model("guard").complete(messages).content, "")`
reads a refused completion as empty text. `fallback` is a literal or a parameter of
the same kind as `value`. In a `model().split()` score predicate it may wrap only the
chunk's score call.

Quarry does not make a call whose model input has no content, so that call is not
refused. Then a boolean `value` is false, and a `not` around it is true.

Examples: [examples/model-refusals.rule](examples/model-refusals.rule) (requires a configured backend)

### `join`

`join(values: collection<string>, separator: string) -> string`

Combine (concatenate) the text values of a map comprehension with a separator.

`separator` goes between two values. Use `join()` as `message()` content in a version 2
model input. For example, `message("user", join([c.arguments.command for c in
messages[-1].tool_calls if $risky in c.arguments.*], "\n"))` gives the commands of the
risky calls in the newest message to the model as one input.

`join()` combines the values in the order of the collection, each rendered as the same
text the same test in a rule step reads from it: a list is its elements joined with
`"\n"`, and a number or a boolean is its text. It leaves out a value that is missing,
null, or empty, and keeps a value that is only whitespace. One value gives that value
unchanged.

When no value stays, the model input has no content. Quarry does not call the model, and
the comparison that reads the result is false.

When Quarry enforces its limits, `join()` counts the encoded bytes of the text it makes
as it builds it. A text at or under `max_tool_argument_size` bytes does not change. A
longer text fails the rule, rather than truncating it: a benign value must not be able to
push a risky value out of the judge's input unnoticed.

- `separator`: must be written directly in the rule

Examples: [examples/joined-tool-commands.rule](examples/joined-tool-commands.rule) (requires a configured backend)

### `len`

`len(value: collection | string) -> integer`

Return the size of a supported collection or the length of text.

When `value` is a filtered timeline collection, compare the result with an integer
from 0 through 8.

Examples: [examples/indexes-and-counts.rule](examples/indexes-and-counts.rule)

### `llm_split_conversation`

`llm_split_conversation(model: string, input: model_input, max_input_tokens: integer) -> collection<model_input>`

Split model input into chunks no larger than `max_input_tokens`.

Use this only after `in` in a documented `any` or `all` chunk check, or in a
filtered chunk list passed to `count` or `len`.

- Allowed in: comprehension_source
- Requires: `token` backend support
- `model`: must be written directly in the rule
- `max_input_tokens`: must be written directly in the rule; minimum: 1

Examples: [examples/chunked-inference.rule](examples/chunked-inference.rule) (requires a configured backend)

### `message`

`message(role: string, content: string) -> model_input`

Build a one-message model input from a role and literal or field-backed content.

In a version 2 model expression, `content` can also be the text that `join()` gives.

- `role`: must be written directly in the rule; choices: system, developer, user, assistant

Examples: [examples/model-inference.rule](examples/model-inference.rule) (requires a configured backend)

### `model`

`model(alias: string) -> model_reference`

Rule syntax: `model("alias").operation(...)`

Select a configured model by its literal alias for a following method call.

Use this as the start of a model operation, such as
`model("guard").score(messages, "policy")`. `model(...)` is not a condition by itself.

- `alias`: must be written directly in the rule

Examples: [examples/model-interface.rule](examples/model-interface.rule) (requires a configured backend)

### `model.classify`

`model.classify(messages: model_input, tools: collection<tool_definition> = null) -> number`

Rule syntax: `model("alias").classify(messages[, tools])`

Return the configured classifier head's probability for these messages.

The model alias comes from the surrounding `model("alias")` reference. Pass the bare
`tools` value only when the classifier should also see offered tool definitions.

- Requires: `model` backend support

Examples: [examples/model-interface.rule](examples/model-interface.rule) (requires a configured backend)

### `model.complete`

`model.complete(messages: model_input, tools: collection<tool_definition> = null) -> message`

Rule syntax: `model("alias").complete(messages[, tools])`

Generate one assistant message from the supplied messages.

Read fields such as `.content`, `.refusal`, or `.tool_calls` from the result. Pass the
bare `tools` value only when the model may call offered tools. When the alias declares
`complete.response_schema`, read the reply's fields by name with `.json.<field>`; a
reply that does not match the schema is a refused call.

- Requires: `completion` backend support

Examples: [examples/model-interface.rule](examples/model-interface.rule) (requires a configured backend)

### `model.input_tokens`

`model.input_tokens(messages: model_input, tools: collection<tool_definition> = null) -> integer`

Rule syntax: `model("alias").input_tokens(messages[, tools])`

Return the model-specific token count for the supplied messages and optional tools.

- Requires: `token` backend support

Examples: [examples/model-interface.rule](examples/model-interface.rule) (requires a configured backend)

### `model.score`

`model.score(messages: model_input, constitution: string | text_reference, tools: collection<tool_definition> = null) -> number`

Rule syntax: `model("alias").score(messages, constitution[, tools])`

Score messages against policy text and return a number from 0 through 1.

The model alias comes from the surrounding `model("alias")` reference. `constitution`
may be literal policy text or a `text_ref("name")`. Pass the bare `tools` value only
when the model should also see the request's offered tools.

- Requires: `score` backend support

Examples: [examples/model-interface.rule](examples/model-interface.rule) (requires a configured backend)

### `model.spans`

`model.spans(messages: model_input, threshold: number) -> collection<span>`

Rule syntax: `model("alias").spans(messages, threshold)`

Return original-conversation spans from the alias-configured span operation.

`messages` must be the root conversation or one of its views. Quarry reuses the raw
classifier response cached for the same model input. The adapter owns span decoding and
how the required `threshold` is applied.

- Allowed in: comprehension_source
- Requires: `model` backend support
- `threshold`: must be written directly in the rule; minimum: 0.0; maximum: 1.0

Examples: [examples/model-spans.rule](examples/model-spans.rule) (requires a configured backend)

### `model.split`

`model.split(messages: model_input, max_input_tokens: integer, overlap: integer = null, context: model_input = null) -> collection<model_input>`

Rule syntax: `model("alias").split(messages, max_input_tokens[, overlap][, context])`

Partition messages into model-sized chunks for one non-nested `any` score check.

Each chunk is one span of the conversation within `max_input_tokens`, and nothing is
repeated by default; a message longer than the limit is cut into ranges. The optional
arguments keep the order `overlap`, `context` and either may be left out: `overlap` is
the most tokens repeated from the end of the previous chunk, and `context` a view
such as `system_messages` repeated at the start of every chunk. Tools are not
messages: pass them to the chunk's score, outside the limit.

- Allowed in: comprehension_source
- Requires: `token` backend support
- `max_input_tokens`: must be written directly in the rule; minimum: 1
- `overlap`: must be written directly in the rule; minimum: 0

Examples: [examples/model-interface.rule](examples/model-interface.rule) (requires a configured backend)

### `model_split`

`model_split(model: string, messages: model_input, max_input_tokens: integer) -> collection<model_input>`

Rule syntax: `model_split("alias", messages, max_input_tokens)`

Compatibility spelling for `model("alias").split(...)` in version 2 rules.

This spelling remains available during migration. New rules should use the model method.

- Deprecated. Use `model("alias").split(messages, max_input_tokens)` instead
- Allowed in: comprehension_source
- Requires: `token` backend support
- `model`: must be written directly in the rule
- `max_input_tokens`: must be written directly in the rule; minimum: 1

Examples: [examples/compatibility-model-split.rule](examples/compatibility-model-split.rule) (requires a configured backend)

### `refused`

`refused(value: boolean | number | string | message) -> boolean`

Return whether a `model("alias").score(...)` or `.complete(...)` call in `value` was refused.

A call is refused when a guardrail fired, so the provider stopped the reply or
rejected the request, or when the reply is not in the format the operation asked for,
such as a score reply that is not a configured verdict token. A completion that says
no in its own words is an answer, not a refusal; read it from `.refusal` or
`.content`. `refused()` reads the call's own request, so it sends none of its own.

Without `refused()` or `if_refused()`, a refused call fails the rule. A rule that reads
a call inside them must read it there everywhere, so no unguarded read can fail first.

Examples: [examples/model-refusals.rule](examples/model-refusals.rule) (requires a configured backend)

### `text_ref`

`text_ref(name: string) -> text_reference`

Refer to configured policy text by its exact resource name.

The DSL accepts this only where an inference function expects policy text. A reference remains
a distinct type until the resource manager resolves it, preventing an unresolved name from
being mistaken for the policy itself.

- `name`: must be written directly in the rule

Examples: [examples/model-inference.rule](examples/model-inference.rule) (requires a configured backend)

### `add_message`

`add_message(role: string, content: string) -> model_input`

Rule syntax: `.add_message("role", content)`

Append one text message to a model input created with `message(...)`.

Use this only after a model-input value, for example
`message("user", "hello").add_message("assistant", "hi")`.

- `role`: must be written directly in the rule; choices: system, developer, user, assistant

Examples: [examples/model-inference.rule](examples/model-inference.rule) (requires a configured backend)

### `assistant_said`

`assistant_said(text: string, case_sensitive: boolean = false) -> boolean`

Return whether any assistant message contains the supplied text.

Matching is case-insensitive by default. Pass `true` as the second argument to require
the same letter case.

- Allowed in: condition, flow_operand, comprehension_body
- `text`: must be written directly in the rule
- `case_sensitive`: must be written directly in the rule

Examples: [examples/language-basics.rule](examples/language-basics.rule)

### `call_sequence`

`call_sequence() -> collection<string>`

Return assistant tool-call names in chronological order.

- Allowed in: comprehension_source

Examples: [examples/tool-operations.rule](examples/tool-operations.rule)

### `called_tool`

`called_tool(name: string) -> boolean`

Return whether the assistant called a tool with this exact name.

This checks calls that occurred. It does not check tools that were merely offered in the
request.

- Allowed in: condition, flow_operand, comprehension_body
- `name`: must be written directly in the rule

Examples: [examples/exact-tool-name.rule](examples/exact-tool-name.rule)

### `called_tools`

`called_tools(*names: string) -> boolean`

Return whether the assistant called every named tool.

At least one exact tool name is required. This test does not impose an order.

- Allowed in: condition, flow_operand, comprehension_body
- `names`: must be written directly in the rule

Examples: [examples/tool-operations.rule](examples/tool-operations.rule)

### `capability_tags`

`capability_tags(tag: string, text: string) -> boolean`

Return whether `text` describes the capability named by `tag`.

The built-in classifier recognizes `external_communication`, `private_data_access`,
`code_execution`, `state_modification`, and `untrusted_ingress`. A configured
classifier may support additional tags. This function requires a configured capability
classifier.

- Allowed in: condition, comprehension_body
- Requires: `capability_classifier` backend support
- `tag`: must be written directly in the rule
- `text`: must be a conversation field or surface

Examples: [examples/agent-probers.rule](examples/agent-probers.rule) (requires a configured backend)

### `count_tool_calls`

`count_tool_calls(name: string) -> integer`

Count assistant calls to the tool with this exact name.

- `name`: must be written directly in the rule

Examples: [examples/tool-operations.rule](examples/tool-operations.rule)

### `extract`

`extract(span: span, left: integer = 0, right: integer = 0) -> string`

Rule syntax: `messages.extract(span[, left[, right]])`

Return one span plus bounded context from its canonical source message.

`left` and `right` default to zero and clamp at that message's boundaries.

- Allowed in: comprehension_body
- `left`: minimum: 0
- `right`: minimum: 0

Examples: [examples/model-spans.rule](examples/model-spans.rule) (requires a configured backend)

### `get_tool_calls`

`get_tool_calls(name: string) -> collection<tool_call>`

Return assistant calls to the tool with this exact name.

Use the result as a comprehension collection or count it.

- Allowed in: comprehension_source
- `name`: must be written directly in the rule

Examples: [examples/exact-tool-name.rule](examples/exact-tool-name.rule)

### `get_tool_responses`

`get_tool_responses(name: string) -> collection<tool_response>`

Return results linked to calls of the tool with this exact name.

The call/result relationship is resolved through `tool_call_id`; unrelated tool
results are excluded.

- Allowed in: comprehension_source
- `name`: must be written directly in the rule

Examples: [examples/tool-operations.rule](examples/tool-operations.rule)

### `has_sequence`

`has_sequence(*names: string) -> boolean`

Return whether these names occur contiguously in the tool-call order.

Other messages and tool results may appear between the calls, but another tool call
between two requested names breaks the sequence.

- Allowed in: condition, comprehension_body
- `names`: must be written directly in the rule

Examples: [examples/tool-operations.rule](examples/tool-operations.rule)

### `has_tool`

`has_tool(name: string) -> boolean`

Return whether a tool with this exact name was offered to the model.

This checks the request's tool definitions, not the tool calls made by the assistant.

- Allowed in: condition
- `name`: must be written directly in the rule

Examples: [examples/tool-operations.rule](examples/tool-operations.rule)

### `has_tools`

`has_tools(*names: string) -> boolean`

Return whether every named tool was offered to the model.

At least one name is required. This checks availability, not whether the assistant used
the tools.

- Allowed in: condition
- `names`: must be written directly in the rule

Examples: [examples/tool-operations.rule](examples/tool-operations.rule)

### `llm_complete`

`llm_complete(model: string, input: model_input) -> string`

Generate text from model input for comparison with a literal string or regex.

Write this function on the left of `==` or `~`. A bare call is not a true-or-false
condition.

- Requires: `completion` backend support
- `model`: must be written directly in the rule

Examples: [examples/model-inference.rule](examples/model-inference.rule) (requires a configured backend)

### `llm_input_tokens`

`llm_input_tokens(model: string, input: model_input) -> integer`

Return the model-specific token count for the supplied model input.

- Requires: `token` backend support
- `model`: must be written directly in the rule

Examples: [examples/model-inference.rule](examples/model-inference.rule) (requires a configured backend)

### `llm_score`

`llm_score(model: string, constitution: string | text_reference, input: model_input) -> number`

Score model input against policy text and return a number from 0 through 1.

The policy may be written directly or loaded by name with `text_ref("name")`.

- Requires: `score` backend support
- `model`: must be written directly in the rule
- `constitution`: must be literal text or a `text_ref(...)` call

Examples: [examples/model-inference.rule](examples/model-inference.rule) (requires a configured backend)

### `model_scan`

`model_scan(model: string, text: string, threshold: number, class_index: integer) -> boolean`

Return whether a classifier score is greater than a required threshold.

`model` names the configured classifier. `text` is the field being
classified, and `class_index` selects the model output whose probability is tested.

- Allowed in: condition, comprehension_body
- Requires: `model` backend support
- `model`: must be written directly in the rule
- `text`: must be a conversation field or surface
- `threshold`: must be written directly in the rule
- `class_index`: must be written directly in the rule; minimum: 0

Examples: [examples/model-classification.rule](examples/model-classification.rule) (requires a configured backend)

### `model_score`

`model_score(model: string, text: string, class_index: integer) -> number`

Return a classifier probability for the selected output class.

`model` names the configured classifier. Compare the returned value with
a numeric threshold in the rule. `class_index` is mandatory in the DSL.

- Requires: `model` backend support
- `model`: must be written directly in the rule
- `text`: must be a conversation field or surface
- `class_index`: must be written directly in the rule; minimum: 0

Examples: [examples/model-classification.rule](examples/model-classification.rule) (requires a configured backend)

### `off_topic`

`off_topic(text: string) -> boolean`

Return whether `text` has moved too far from the first user message.

This function requires a configured topic-analysis backend.

- Allowed in: condition, comprehension_body
- Requires: `topic_prober` backend support
- `text`: must be a conversation field or surface

Examples: [examples/agent-probers.rule](examples/agent-probers.rule) (requires a configured backend)

### `topic_drift`

`topic_drift(text: string) -> number`

Return how far `text` has moved from the first user message, from 0 through 1.

A result of 0 means on-topic and 1 means fully off-topic. This function requires a
configured topic-analysis backend.

- Requires: `topic_prober` backend support
- `text`: must be a conversation field or surface

Examples: [examples/agent-probers.rule](examples/agent-probers.rule) (requires a configured backend)

### `unrequested_action`

`unrequested_action(text: string) -> boolean`

Return whether `text` describes an action not requested in the first user message.

Pass a tool name, description, or arguments field as `text`. This function requires a
configured capability classifier.

- Allowed in: condition, comprehension_body
- Requires: `capability_classifier` backend support
- `text`: must be a conversation field or surface

Examples: [examples/agent-probers.rule](examples/agent-probers.rule) (requires a configured backend)

### `user_said`

`user_said(text: string, case_sensitive: boolean = false) -> boolean`

Return whether any user message contains the supplied text.

Matching is case-insensitive by default. Pass `true` as the second argument to require
the same letter case.

- Allowed in: condition, flow_operand, comprehension_body
- `text`: must be written directly in the rule
- `case_sensitive`: must be written directly in the rule

Examples: [examples/language-basics.rule](examples/language-basics.rule)

## Type conversions

| From | To | Rule |
|---|---|---|
| `message` | `model_input` | A message of the conversation, such as `user_messages[-1]` or the element `m` of `any m in messages`, is the model input that holds only that message. A message the conversation does not have is an empty model input, which makes no model call. |

## Data surfaces

Surfaces are bare names. Do not add `()`.

| Surface | Type | Item fields | What it contains |
|---|---|---|---|
| `assistant_messages` | `collection<message>` | `content`, `refusal`, `role`, `tool_calls` | Assistant-role messages from the selected conversation, in source order. |
| `developer_messages` | `collection<message>` | `content`, `role` | Developer-role messages from the selected conversation, in source order. |
| `messages` | `collection<message>` | `content`, `role` | All messages in the inspected conversation, in chronological order. |
| `system_messages` | `collection<message>` | `content`, `role` | System-role messages from the selected conversation, in source order. |
| `system_prompt` | `string` | — | Text content of the leading system message. |
| `tool_calls` | `collection<tool_call>` | `arguments`, `id`, `name`, `raw_arguments`, `response`, `type` | Assistant tool calls in chronological order. |
| `tool_responses` | `collection<tool_response>` | `call`, `content`, `id`, `name`, `result`, `role`, `tool_call_id` | Tool-result messages linked to assistant calls by `tool_call_id`. |
| `tools` | `collection<tool_definition>` | `description`, `name`, `parameters`, `type` | Tools offered to the model in the inspected request. |
| `user_messages` | `collection<message>` | `content`, `role` | User-role messages from the selected conversation, in source order. |

### Surface fields

Only the fields listed here are supported for rule authoring.

| Surface | Field | Type | Meaning |
|---|---|---|---|
| `assistant_messages` | `content` | `string` | Flattened message text. For multimodal input, this contains text parts only. |
| `assistant_messages` | `refusal` | `string \| null` | Assistant refusal text when supplied by the provider; otherwise null. |
| `assistant_messages` | `role` | `string` | OpenAI message role. |
| `assistant_messages` | `tool_calls` | `collection<tool_call>` | Calls emitted with this assistant message. Use the `tool_calls` surface to iterate over calls; this field can be measured with `len(...)`. |
| `developer_messages` | `content` | `string` | Flattened message text. For multimodal input, this contains text parts only. |
| `developer_messages` | `role` | `string` | OpenAI message role. |
| `messages` | `content` | `string` | Flattened message text. For multimodal input, this contains text parts only. |
| `messages` | `role` | `string` | OpenAI message role. |
| `system_messages` | `content` | `string` | Flattened message text. For multimodal input, this contains text parts only. |
| `system_messages` | `role` | `string` | OpenAI message role. |
| `tool_calls` | `arguments` | `object` | Decoded JSON arguments. Read a key with a path such as `call.arguments.command`, or all values with `call.arguments.*`. |
| `tool_calls` | `id` | `string` | Provider-assigned tool-call ID. |
| `tool_calls` | `name` | `string` | Exact called function name. |
| `tool_calls` | `raw_arguments` | `string` | Original JSON argument text. |
| `tool_calls` | `response` | `tool_response \| null` | Linked tool response, or null when no response exists. Continue with a field such as `call.response.result`. |
| `tool_calls` | `type` | `string` | Tool-call type; currently `function`. |
| `tool_responses` | `call` | `tool_call` | Originating call. Continue with a field such as `response.call.name`. |
| `tool_responses` | `content` | `string` | Flattened message text. For multimodal input, this contains text parts only. |
| `tool_responses` | `id` | `string` | Alias for `tool_call_id`. |
| `tool_responses` | `name` | `string` | Exact name of the called function. |
| `tool_responses` | `result` | `string` | Tool-result text; alias for `content`. |
| `tool_responses` | `role` | `string` | OpenAI message role. |
| `tool_responses` | `tool_call_id` | `string` | ID of the call this message answers. |
| `tools` | `description` | `string` | Description shown to the model. |
| `tools` | `name` | `string` | Exact offered function name. |
| `tools` | `parameters` | `object` | JSON Schema for the function arguments. |
| `tools` | `type` | `string` | Offered tool type; currently `function`. |
| `user_messages` | `content` | `string` | Flattened message text. For multimodal input, this contains text parts only. |
| `user_messages` | `role` | `string` | OpenAI message role. |

## Operator precedence

Rows are ordered from lowest to highest precedence. Parentheses override this
order. Add parentheses when mixing boolean, flow, comparison, or text operators
so the intended grouping is explicit.

| Precedence | Family | Operators | Associativity | Meaning |
|---|---|---|---|---|
| 1 | boolean or | `or` | left | Evaluated after `and`. |
| 2 | boolean and | `and` | left | Evaluated before `or`. |
| 3 | boolean not | `not` | right | Applies to the expression immediately after it. |
| 4 | flow | `before`, `before(within:N)`, `right_before`, `after`, `after(within:N)`, `right_after`, `until` | none | Relates two event conditions. Each operand is one event: a single condition, an `any` comprehension, `and`/`or`/`not` over the element of an enclosing comprehension, or an `or` of such operands. `all` is allowed only on the sustained side of `until`. Chain flow operators only with parentheses. |
| 5 | comparison | `==`, `!=`, `>`, `<`, `>=`, `<=` | none | Compares two values. Chain comparisons only with parentheses. |
| 6 | text and membership | `contains`, `contains_any`, `~`, `matches`, `in`, `as` | none | Tests text or membership and binds captures. |
| 7 | addition | `+`, `-` | left | Adds or subtracts numeric values. |
| 8 | multiplication | `*` | left | Multiplies numeric values. |
| 9 | postfix access | `.field`, `.method(...)`, `[index]`, `[slice]`, `.*` | left | Reads fields, calls methods, indexes, slices, or expands object values. |

## Language forms

### addition

Add two numeric values.

Grammar: `arith_expr: arith_expr "+" product_expr`

Examples: [examples/patterns-and-aliases.rule](examples/patterns-and-aliases.rule)

### adjacent-alias

Name a value for use in the clause immediately joined by the same `and`.

Grammar: `in_expr: arith_expr "as" IDENTIFIER`

Examples: [examples/patterns-and-aliases.rule](examples/patterns-and-aliases.rule)

### all-comprehension

Match when every item in a collection satisfies a condition.

A part of the condition that does not read the item is checked over the whole
conversation. An empty collection always matches.

Grammar: `all_comp: "all" IDENTIFIER "in" expr "where" expr`

Examples: [examples/message-surfaces.rule](examples/message-surfaces.rule)

### any-comprehension

Match when at least one item in a collection satisfies a condition.

A part of the condition that does not read the item, such as `called_tool("x")` or
a comprehension over another collection, is checked over the whole conversation, not
only after the item. Over `tools`, every test in the condition reads one tool
definition. An empty collection never matches.

One item can take part in several flows, and a flow operand can be another
comprehension joined to the item (`any c in tool_calls where c.arguments.cmd
contains m.content`). Three shapes are checked item by item, for at most 256 items of
the collection per scan, where a scan with more items and no match is incomplete: a
flow operand joined to the item, the item's own condition next to other alternatives
on the left of `until`, and an event before the item whose condition relates later
events.

Grammar: `any_comp: "any" IDENTIFIER "in" expr "where" expr`

Examples: [examples/case-insensitive-match.rule](examples/case-insensitive-match.rule), [examples/language-basics.rule](examples/language-basics.rule), [examples/tool-argument-drilldown.rule](examples/tool-argument-drilldown.rule)

### array-literal

Write an ordered literal list between brackets.

Grammar: `array: "[" [value ("," value)*] "]"`

Examples: [examples/language-basics.rule](examples/language-basics.rule)

### boolean-and

Match when every operand is true.

Grammar: `and_expr: not_expr ("and" not_expr)*`

Examples: [examples/language-basics.rule](examples/language-basics.rule), [examples/tool-argument-drilldown.rule](examples/tool-argument-drilldown.rule)

### boolean-not

Invert one boolean expression.

Grammar: `not_expr: "not" not_expr`

Examples: [examples/language-basics.rule](examples/language-basics.rule)

### boolean-or

Match when at least one operand is true.

Grammar: `or_expr: and_expr ("or" and_expr)*`

Examples: [examples/language-basics.rule](examples/language-basics.rule)

### capture

Capture matched text under a name for comparison elsewhere in the rule.

The rule matches when some captured value satisfies every place that uses it, or when
it holds with no value captured, every place that uses the value then being false. The
capture itself cannot appear under `not` or inside an `all` comprehension; such a
rule fails to load. A place that uses the value may appear there.

Grammar: `in_expr: arith_expr "in" arith_expr "as" IDENTIFIER`

Examples: [examples/capture.rule](examples/capture.rule)

### contains

Test whether text or a collection contains a supplied value.

Text comparison is case-sensitive. To match text regardless of letter case, declare the
value in a `strings:` block with the `nocase` modifier and reference it here.

Grammar: `in_expr: arith_expr "contains" arith_expr`

Examples: [examples/case-insensitive-match.rule](examples/case-insensitive-match.rule), [examples/tool-argument-drilldown.rule](examples/tool-argument-drilldown.rule), [examples/tool-operations.rule](examples/tool-operations.rule)

### contains-any

Match when text contains at least one value from a literal or named array.

Comparison against a literal array is case-sensitive and has no case option. For
case-insensitive matching, declare a `nocase` pattern in a `strings:` block --
use a `regex` alternation such as `"alpha|beta" nocase regex` for several
alternatives -- and match it with `contains`.

Grammar: `in_expr: arith_expr "contains_any" (array | STRING_VAR)`

Examples: [examples/language-basics.rule](examples/language-basics.rule)

### equal-comparison

Match when the two values are equal.

Grammar: `comparison: in_expr "==" in_expr`

Examples: [examples/exact-tool-name.rule](examples/exact-tool-name.rule), [examples/tool-argument-drilldown.rule](examples/tool-argument-drilldown.rule)

### field-access

Read a named field with dot syntax, such as `msg.content`.

Grammar: `postfix: "." IDENTIFIER`

Examples: [examples/case-insensitive-match.rule](examples/case-insensitive-match.rule), [examples/language-basics.rule](examples/language-basics.rule), [examples/tool-argument-drilldown.rule](examples/tool-argument-drilldown.rule)

### filter-comprehension

Keep collection items that satisfy a condition.

In `[msg for msg in user_messages if msg.content contains "sudo"]`, the result
contains only user messages whose content contains `sudo`. Use the same item name
before and after `for`.

Grammar: `filter_comp: "[" IDENTIFIER "for" IDENTIFIER "in" expr "if" expr "]"`

Examples: [examples/filter-count.rule](examples/filter-count.rule)

### flow-after

Match when the left event occurs sometime after the right event.

Grammar: `flow_expr: comparison "after" comparison`

Examples: [examples/flow-operators.rule](examples/flow-operators.rule)

### flow-after-within

Match when the left event occurs after the right event within a bounded distance.

`N` counts timeline steps and must be at least 1. The window compiles to about
`N` automaton states; a single window beyond about 14,000 steps exceeds the compiler's
work budget and fails to load.

Grammar: `flow_expr: comparison "after" "(" "within" ":" NUMBER ")" comparison`

Examples: [examples/flow-operators.rule](examples/flow-operators.rule)

### flow-before

Match when the left event occurs sometime before the right event.

Grammar: `flow_expr: comparison "before" comparison`

Examples: [examples/flow-operators.rule](examples/flow-operators.rule)

### flow-before-within

Match when the left event occurs before the right event within a bounded distance.

`N` counts timeline steps and must be at least 1. The window compiles to about
`N` automaton states; a single window beyond about 14,000 steps exceeds the compiler's
work budget and fails to load.

Grammar: `flow_expr: comparison "before" "(" "within" ":" NUMBER ")" comparison`

Examples: [examples/flow-operators.rule](examples/flow-operators.rule)

### flow-right-after

Match when the left event occurs immediately after the right event.

Grammar: `flow_expr: comparison "right_after" comparison`

Examples: [examples/flow-operators.rule](examples/flow-operators.rule)

### flow-right-before

Match when the left event occurs immediately before the right event.

Grammar: `flow_expr: comparison "right_before" comparison`

Examples: [examples/flow-operators.rule](examples/flow-operators.rule)

### flow-until

Match when the left condition remains true until the right event occurs.

The left condition must hold at every event of its own kind before the right event:
every tool call for `called_tool(...)`, every user message for `user_said(...)`, or
every element of a comprehension's collection. Events of other kinds do not break it.
If the right event is the first one, the left condition need not hold at all.

Grammar: `flow_expr: comparison "until" comparison`

Examples: [examples/flow-operators.rule](examples/flow-operators.rule)

### function-call

Call a public Quarry function with positional arguments.

Grammar: `function_call: IDENTIFIER "(" [arguments] ")"`

Examples: [examples/language-basics.rule](examples/language-basics.rule)

### greater-than-comparison

Match when the left number is greater than the right number.

Grammar: `comparison: in_expr ">" in_expr`

Examples: [examples/indexes-and-counts.rule](examples/indexes-and-counts.rule)

### greater-than-or-equal-comparison

Match when the left number is greater than or equal to the right number.

Grammar: `comparison: in_expr ">=" in_expr`

Examples: [examples/indexes-and-counts.rule](examples/indexes-and-counts.rule)

### less-than-comparison

Match when the left number is less than the right number.

Grammar: `comparison: in_expr "<" in_expr`

Examples: [examples/indexes-and-counts.rule](examples/indexes-and-counts.rule)

### less-than-or-equal-comparison

Match when the left number is less than or equal to the right number.

Grammar: `comparison: in_expr "<=" in_expr`

Examples: [examples/string-operations.rule](examples/string-operations.rule)

### map-comprehension

Project each selected item for use as the first argument of `join()`.

The map is supported in `message()` content of a version 2 model input.
Its value and optional filter cannot call a model.

Grammar: `map_comp: "[" expr "for" IDENTIFIER "in" expr "]" | "[" expr "for" IDENTIFIER "in" expr "if" expr "]"`

Examples: [examples/joined-tool-commands.rule](examples/joined-tool-commands.rule) (requires a configured backend)

### membership

Test whether a value or text pattern occurs in another value.

Grammar: `in_expr: arith_expr "in" arith_expr`

Examples: [examples/string-operations.rule](examples/string-operations.rule)

### metadata

Attach string metadata fields to a rule.

Grammar: `metadata_block: "metadata:" metadata_item+`

Examples: [examples/case-insensitive-match.rule](examples/case-insensitive-match.rule), [examples/language-basics.rule](examples/language-basics.rule)

### method-call

Call a supported Quarry function on the value written before the dot.

For example, `messages.called_tool("read")` checks the complete message collection.
A `message(...)` model input can use `.add_message(...)` to append another message.
Other values cannot use method-call syntax.

Grammar: `postfix: "." IDENTIFIER "(" [arguments] ")"`

Examples: [examples/tool-operations.rule](examples/tool-operations.rule)

### multiplication

Multiply two numeric values.

Grammar: `product_expr: product_expr "*" atom`

Examples: [examples/model-interface.rule](examples/model-interface.rule) (requires a configured backend)

### not-equal-comparison

Match when the two values are not equal.

Grammar: `comparison: in_expr "!=" in_expr`

Examples: [examples/indexes-and-counts.rule](examples/indexes-and-counts.rule)

### parameter-declaration

Declare a per-scan value, or a policy-bound value when marked `policy`.

Grammar: `parameter_def: PARAM_NAME ":" PARAM_TYPE POLICY_BOUND?`

Examples: [examples/model-interface.rule](examples/model-interface.rule) (requires a configured backend)

### parameters

Declare typed policy-bound or scan-bound inputs as the last block before `condition:`.

Grammar: `parameters_block: "parameters:" parameter_def+`

Examples: [examples/model-interface.rule](examples/model-interface.rule) (requires a configured backend)

### pattern-block

Group alternative field checks between braces.

Grammar: `pattern_block: "{" pattern_expr ("or" pattern_expr)* "}"`

Examples: [examples/patterns-and-aliases.rule](examples/patterns-and-aliases.rule)

### pattern-match

Test one value against a block containing alternative field predicates.

Grammar: `in_expr: arith_expr "matches" pattern_block`

Examples: [examples/patterns-and-aliases.rule](examples/patterns-and-aliases.rule)

### pattern-predicate

Check one field inside a pattern block.

Grammar: `pattern_expr: atom PATTERN_OPERATOR value | IDENTIFIER PATTERN_OPERATOR value`

Examples: [examples/patterns-and-aliases.rule](examples/patterns-and-aliases.rule)

### regex-match

Match text with an RE2-compatible regular expression.

Matching is case-sensitive. For case-insensitive matching, declare the pattern in a
`strings:` block with `regex nocase` and match it with `in`.

Grammar: `in_expr: arith_expr "~" arith_expr`

Examples: [examples/string-operations.rule](examples/string-operations.rule)

### rule

Define one named rule with a required condition and optional metadata, strings,
variables, parameters, and typed collection output.

Grammar: `rule: "rule" STRING metadata_block? strings_block? variables_block? parameters_block? "condition:" condition collect_block?`

Examples: [examples/case-insensitive-match.rule](examples/case-insensitive-match.rule), [examples/language-basics.rule](examples/language-basics.rule)

### signed-index

Select one collection item by position, such as `messages[-1]`.

Negative positions count backward from the end of the collection.

Grammar: `postfix: "[" SIGNED_INT "]"`

Examples: [examples/model-classification.rule](examples/model-classification.rule) (requires a configured backend)

### slice-all

`[:]` — the whole surface as a window (identity slice).

Grammar: `postfix: "[" ":" "]"`

Examples: [examples/indexes-and-counts.rule](examples/indexes-and-counts.rule)

### slice-from

`[start:]` — from `start` to the end.

Grammar: `postfix: "[" SIGNED_INT ":" "]"`

Examples: [examples/indexes-and-counts.rule](examples/indexes-and-counts.rule)

### slice-range

`[start:stop]` — a bounded window.

Grammar: `postfix: "[" SIGNED_INT ":" SIGNED_INT "]"`

Examples: [examples/indexes-and-counts.rule](examples/indexes-and-counts.rule)

### slice-to

`[:stop]` — from the beginning to `stop`.

Grammar: `postfix: "[" ":" SIGNED_INT "]"`

Examples: [examples/indexes-and-counts.rule](examples/indexes-and-counts.rule)

### string-count

Count occurrences of a declared string pattern in the inspected text.

Write `#name` for a pattern declared as `$name`. Add `*` to include every
declared pattern whose name starts with the supplied prefix. The pattern's modifiers
apply, so a `nocase` pattern counts matches in any letter case.

Grammar: `string_count: "#" IDENTIFIER WILDCARD?`

Examples: [examples/string-operations.rule](examples/string-operations.rule)

### string-modifiers

Change a declared pattern with `nocase`, `fullword`, `base64`, or `regex`.

`nocase` matches the pattern regardless of letter case. Text matching is otherwise
case-sensitive, so `nocase` is the mechanism for case-insensitive `contains` and
`contains_any` checks. Modifiers apply to a single pattern or an array of patterns,
and to every use of the pattern, including `#name` counts and `@name` positions.

Grammar: `string_modifiers: MODIFIER+`

Examples: [examples/case-insensitive-match.rule](examples/case-insensitive-match.rule), [examples/language-basics.rule](examples/language-basics.rule)

### string-position

Return the position of a declared string pattern in the inspected text.

Write `@name` for a pattern declared as `$name`. Add `*` to include every
declared pattern whose name starts with the supplied prefix. The pattern's modifiers
apply. Positions of patterns with different modifiers cannot be compared; such a rule
fails to load.

Grammar: `string_position: "@" IDENTIFIER WILDCARD?`

Examples: [examples/string-operations.rule](examples/string-operations.rule)

### string-set

Match when any, all, or a requested number of declared patterns occurs in text.

`any of` needs one listed pattern, `all of` needs every listed pattern, and
`N of` needs at least `N`. Each pattern keeps its own modifiers.

Grammar: `string_set_expr: quantifier "of" "(" string_list ")"`

Examples: [examples/string-operations.rule](examples/string-operations.rule)

### string-variable-reference

Refer to a pattern declared in the `strings:` block by its `$name`.

Grammar: `primary: STRING_VAR`

Examples: [examples/case-insensitive-match.rule](examples/case-insensitive-match.rule), [examples/language-basics.rule](examples/language-basics.rule)

### strings

Declare reusable literal or regular-expression patterns and their modifiers.

Grammar: `strings_block: "strings:" string_def+`

Examples: [examples/case-insensitive-match.rule](examples/case-insensitive-match.rule), [examples/language-basics.rule](examples/language-basics.rule)

### subtraction

Subtract the right numeric value from the left value.

Grammar: `arith_expr: arith_expr "-" product_expr`

Examples: [examples/string-operations.rule](examples/string-operations.rule)

### typed-collection

Return every item that survives one exhaustive collection filter.

The rule's Boolean condition runs first. A false condition skips collection work;
a true condition matches only when this block returns at least one item. The filter may
call models and remains subject to the rule timeout and model-call budget.

Grammar: `collect_block: "collect:" filter_comp`

Examples: [examples/model-spans.rule](examples/model-spans.rule) (requires a configured backend)

### variables

Declare reusable expressions that are expanded before validation and compilation.

A binding may use another binding. Inside a comprehension, the loop variable hides a
binding with the same name, so `any y in user_messages where y.content ...` reads
the element even when a binding named `y` exists.

Grammar: `variables_block: "variables:" variable_def+`

Examples: [examples/language-basics.rule](examples/language-basics.rule)

### wildcard-field-access

Select wildcard child values with a final `.*` field step.

Grammar: `wildcard_postfix: "." WILDCARD`

Examples: [examples/indexes-and-counts.rule](examples/indexes-and-counts.rule)

## Fixed words and symbols

| Spelling | Category | Reserved as a variable name |
|---|---|---|
| `after` | flow_word | no |
| `all` | operator | yes |
| `and` | operator | yes |
| `any` | operator | yes |
| `as` | operator | yes |
| `=` | operator | no |
| `base64` | modifier | no |
| `before` | flow_word | no |
| `collect:` | block_keyword | no |
| `:` | punctuation | no |
| `,` | punctuation | no |
| `condition:` | block_keyword | no |
| `contains` | operator | no |
| `contains_any` | operator | no |
| `.` | punctuation | no |
| `==` | operator | no |
| `false` | literal | yes |
| `for` | operator | no |
| `fullword` | modifier | no |
| `>` | operator | no |
| `>=` | operator | no |
| `if` | operator | no |
| `in` | operator | yes |
| `{` | punctuation | no |
| `[` | punctuation | no |
| `(` | punctuation | no |
| `<` | operator | no |
| `<=` | operator | no |
| `//` | punctuation | no |
| `matches` | operator | no |
| `metadata:` | block_keyword | no |
| `-` | operator | no |
| `nocase` | modifier | no |
| `not` | operator | yes |
| `!=` | operator | no |
| `null` | literal | yes |
| `number` | block_keyword | no |
| `of` | flow_word | no |
| `or` | operator | yes |
| `parameters:` | block_keyword | no |
| `params` | block_keyword | yes |
| `+` | operator | no |
| `policy` | block_keyword | no |
| `regex` | modifier | no |
| `~` | operator | no |
| `right_after` | flow_word | no |
| `right_before` | flow_word | no |
| `}` | punctuation | no |
| `]` | punctuation | no |
| `)` | punctuation | no |
| `rule` | block_keyword | no |
| `#` | operator | no |
| `@` | operator | no |
| `$` | punctuation | no |
| `strings:` | block_keyword | no |
| `text` | block_keyword | no |
| `true` | literal | yes |
| `until` | flow_word | no |
| `variables:` | block_keyword | no |
| `where` | operator | yes |
| `*` | operator | no |
| `within` | flow_word | no |

## Rejected spellings

These spellings are not alternate runtime functions. Quarry rejects them at the
listed stage so the error is predictable.

| Spelling | Rejected during | Use instead |
|---|---|---|
| `any call in tool_calls where call.name = "shell"` | parse | `any call in tool_calls where call.name == "shell"` |
| `any msg in user_messages where msg matches { role = "user" }` | parse | `any msg in user_messages where msg matches { role == "user" }` |
| `llm_score("m", "policy", conversation) > 0.5` | compile | `llm_score("m", "policy", conversation()) > 0.5` |
| `messages()` | link | `messages` |
| `tools()` | link | `tools` |
| `tool_calls()` | link | `tool_calls` |
| `tool_responses()` | link | `tool_responses` |
| `user_messages()` | link | `user_messages` |
| `assistant_messages()` | link | `assistant_messages` |
| `system_messages()` | link | `system_messages` |
| `developer_messages()` | link | `developer_messages` |
| `system_prompt()` | link | `system_prompt` |
| `first(tool_calls)` | link | `tool_calls[0]` |
| `last(tool_calls)` | link | `tool_calls[-1]` |
| `any msg in messages where regex_match(msg.content, "pattern")` | link | `any msg in messages where msg.content ~ "pattern"` |
| `regex_match_case("text", "pattern")` | link | No direct replacement |
| `regex_extract("text", "pattern")` | link | No direct replacement |
| `user_said("IGNORE", case_sensitive=True)` | parse | `user_said("IGNORE", true)` |
| `assistant_said("IGNORE", case_sensitive=True)` | parse | `assistant_said("IGNORE", true)` |
| `model_score("m", system_prompt, 1)` | compile | `model_score("m", system_prompt, 1) > 0.5` |
| `topic_drift(system_prompt)` | compile | `topic_drift(system_prompt) > 0.5` |
| `llm_score("m", "policy", conversation())` | compile | `llm_score("m", "policy", conversation()) > 0.5` |
| `llm_input_tokens("m", conversation())` | compile | `llm_input_tokens("m", conversation()) > 100` |
| `model_score("m", get_tool_calls("read")[0].arguments, 1)` | compile | No direct replacement |
| `model_score("m", get_tool_responses("read")[0].content, 1)` | compile | No direct replacement |
| `[msg.content for msg in user_messages]` | compile | No direct replacement |
| `msg.content matches { content contains "sudo" }` | compile | `msg matches { content contains "sudo" }` |
