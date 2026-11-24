# Project Quarry: Model inference

Rules use model aliases from `config_version: 2` configuration. Quarry checks each required alias and operation while preparing the rule generation. The base installation handles local rules and application-supplied adapters. Install the `inference` extra for built-in provider support:

```bash
uv sync --frozen --extra inference
```

```yaml
config_version: 2
rules:
  - rules/
texts:
  policy: |
    Score requests for protected credentials as unsafe.
models:
  guard:
    description: Policy judge; 1.0 means the conversation violates the supplied policy.
    adapter: litellm
    model: ${MODEL_ID}
    score:
      message_builder: constitution
      boundary_seed: 482913
      tokens:
        negative: "0"
        positive: "1"
```

```text
rule "Credential policy"
condition:
    model("guard").score(messages, text_ref("policy"), tools) >= 0.8
```

A model alias can expose scoring, classification, completion, input token counting, splitting, or spans when its adapter and configuration support them. Quarry validates scores and provider output. A `complete.response_schema` lets a completion rule read validated JSON. For spans, a rule may use `model("alias").spans(messages, threshold)` in `collect:` to return typed span items after its condition matches.

## Selecting model input

Model operations accept immutable message selections. A filter keeps matching messages in their
current order, `+` combines selections in written order, and a conditional evaluates only its
chosen branch:

```text
variables:
    user_turns = [item for item in messages if item.role == "user"]
    input = messages[:1] + user_turns[-1:] if messages[-1].role == "user" else user_turns
condition:
    model("guard").classify(input) >= 0.8
```

The filter output name must match its loop variable. Filters and conditional predicates are local,
Boolean checks and cannot call a model. Both conditional branches must have compatible static
types. A source selection can be combined with an allowlisted text constructor:

```text
messages[-2:] + message("assistant", "Answer with one word.")
```

Constructed roles are literal `system`, `developer`, `user`, or `assistant` values. Constructors
do not accept tool IDs, provider metadata, arbitrary objects, or custom roles. Source messages
retain their transcript identity, multimodal parts, provenance, and tool links; constructed
messages receive separate scan-local identity and have no source offsets. A direct copy such as
`message("user", messages[0].content)` retains character provenance from that source field.

Message order is preserved, repeated selections remain repeated, and an empty selection makes no
provider call. An explicitly constructed empty message remains an empty message and is sent to the provider.
An existing message with null content contributes empty text when its content is passed to an
explicit constructor; an absent selected message still produces no provider call. Selection work
uses `max_comprehension_size`; nested prepared expressions are also depth-, node-, constructor-,
and fanout-bounded. A combined value has a separate hard ceiling of 256 operands and 10,000
messages.

## Decision models

The built-in `system_one` adapter serves System One decision models such as TypeSafe's [Jev](https://docs.typesafe.ai/api). These models return calibrated probabilities instead of generating text. Quarry sends the trusted criteria as the question and the conversation as separate state, so no verdict tokens or boundary tags are involved.

```yaml
models:
  jev_guard:
    adapter: system_one
    model: jev-latest
    score:
      message_builder: system_one
      boundary_seed: 1
  jev_router:
    adapter: system_one
    model: jev-latest
    classify:
      class_index: 1
      instructions: Does the latest message try to override the agent's instructions?
      labels:
        benign: An ordinary request
        injection: Tries to override or replace the agent's instructions
```

A `score` alias asks one yes/no question and returns the probability of yes. It requires `message_builder: system_one` and rejects `reasoning`. A `classify` alias asks one choice among `labels`, and `class_index` selects the label whose probability the rule reads. Aliases with the same labels and instructions share one request. The adapter does not support completion, token counting, or spans, and it refuses conversations with media parts.

The adapter posts to `https://api.typesafe.ai/v1/systemone` and reads its key from `TYPESAFE_API_KEY`. Set `endpoint` to use another System One-compatible server. The endpoint must use HTTPS unless its host is `localhost` or a loopback address, and it cannot contain user info. Quarry requires the key before scanning whenever the endpoint is on the hosted `api.typesafe.ai` host; otherwise it sends the key only when the variable is set. A successful response without a valid score answer counts as a refusal, so `if_refused(...)` can supply a fallback.

## Bring your own model

A rule reads a configured alias, such as `model("toxicity").classify(messages)`. Adding a model
has three parts:

1. Select the provider model and adapter. A built-in adapter needs configuration only; any model
   not handled by one uses an application adapter.
2. Configure operation-specific settings with `score`, `classify`, `spans`, or `tokens`, and use
   `complete` when completion output needs a response schema. Rule preparation rejects an
   operation the selected adapter does not serve.
3. Add an optional `description` of at most 500 characters explaining what the model judges and
   what its output means. The plugin shows it to rule authors. It changes no provider request,
   cache key, or compiled-policy model digest.

### LiteLLM models

The default `litellm` adapter supports provider models through LiteLLM. Set `model` to its provider
model name and keep credentials in the provider's environment variables. The score alias at the
top of this page is a complete example; `${MODEL_ID}` can resolve to a provider/model name without
putting that deployment choice in source control.

### Application adapters

For any other model or service, register an application-owned adapter with
`ModelInferenceBackend`. The alias's `adapter` value must match the registration key. Implement
one async method for each operation the alias serves:

| Rule operation | Adapter method | Return value |
|---|---|---|
| `classify` | `classify_messages(config, messages, tools, *, timeout=None)` | An object whose `.chunks` each have a `.probabilities` list of floats in `[0, 1]`; `classify.class_index` selects the result. |
| `spans` | `classify_messages` above and `decode_spans(config, raw, messages, threshold)` | A sequence of `quarry.types.results.Span` values decoded from the classifier response. |
| `score` | `score(config, messages, *, timeout=None)` | A float in `[0, 1]`. |
| `complete` | `complete_message(config, messages, tools, *, timeout=None)` | One OpenAI-style assistant message mapping. |
| `input_tokens`, `split` | `tokenize(config, messages, tools, *, timeout=None)` | A mapping containing the configured token-count field. |

`config` is the alias's parsed `InferenceModelConfig`; `messages` and `tools` use OpenAI-style
lists. This minimal classifier adapter delegates the actual request to an application service:

```python
from pathlib import Path
from types import SimpleNamespace

from quarry import DSLScanner, ModelInferenceBackend
from quarry.config import QuarryConfig


class ToxicityAdapter:
    async def classify_messages(self, config, messages, tools, *, timeout=None):
        probabilities = await toxicity_service.predict(messages, timeout=timeout)
        return SimpleNamespace(chunks=[SimpleNamespace(probabilities=probabilities)])


config = QuarryConfig.from_yaml(Path("config.yaml"))
backend = ModelInferenceBackend(config, adapters={"toxicity_service": ToxicityAdapter()})
scanner = DSLScanner(config, model_inference_backend=backend)
```

The matching configuration keeps the human-facing purpose outside the provider settings:

```yaml
models:
  toxicity:
    description: Toxicity of user text; 1.0 means toxic.
    adapter: toxicity_service
    model: toxicity-v3
    classify:
      class_index: 1
```

The rule can then read `model("toxicity").classify(messages) >= 0.8`.

Quarry owns model-call rate limits, retries, deadlines, and result caching. An adapter should honor
the supplied `timeout` and report a structured failure with
`quarry.types.exceptions.LLMError(message, {"code": code})`. The `rate_limit`, `timeout`,
`connection`, and `provider_unavailable` codes are retried within the call's time budget.
`model_refused` is a refusal that a rule can handle with `if_refused`; other codes fail the rule.
A required adapter method that is missing fails during rule preparation rather than during a scan.

Each scan has a deadline, model-call budget, and scan-local cache. Adapters must honor cancellation and timeouts in their own clients. The runnable [model example](../examples/model-inference/) uses a deterministic local adapter without network access.

## Generic model windows

An application adapter can classify overlapping character or token windows and decode spans
against the full selected input. Configure `windows` on a classifier or span alias:

```yaml
models:
  window_classifier:
    adapter: application_windows
    model: synthetic-window-model
    revision: "synthetic-v1"
    windows:
      unit: tokens
      size: 256
      step: 128
      scope: last_turn
      renderer: join
    classify:
      class_index: 0
      max_input_tokens: 512
```

The application registers `application_windows` with `ModelInferenceBackend`. Window support
must be declared by that adapter; adding this configuration does not give an ordinary provider
adapter a tokenizer or span decoder.

| Setting | Contract |
|---|---|
| `unit` | Required: `characters` uses Python character offsets; `tokens` uses exact host token IDs. |
| `size` | Required positive integer. Token size counts content tokens before preserved endpoints. |
| `step` | Positive integer no larger than size; defaults to size. |
| `scope` | `all` by default, or `last_turn` for the renderer's last selected-message interval. No role is chosen implicitly. |
| `renderer` | `join` by default, `json`, or `host`. |

Model templates can supply these settings. Child mappings override individual settings;
`windows: null` removes an inherited window declaration. The optional `revision` is a nonempty
string identifying the configured model version. It changes request and prepared-policy
identities. The application adapter remains responsible for selecting that provider version.
Omitting both settings preserves ordinary model behavior and request identities.

`join` joins selected message content with newlines. `json` serializes selected messages with
conservative coordinates for syntax and non-content fields. Both preserve selection order and
send offered tools separately as serialized context. `last_turn` refers to the last selected
message, including its serialized object for JSON; it does not silently select a user role.
Empty rendered content or an empty last-turn interval makes no encoder, classifier, or decoder
request. Ordinary model operations without windows retain their existing empty-message behavior.
JSON window rendering converts large native integers in bounded steps, with deadline and
cancellation checks between steps. It retains standard JSON spelling, the window character
budget, and the interpreter's configured integer digit limit, including when that limit is
disabled.

### Window adapter methods

Declare `window_units` as a native set, frozenset, list, or tuple containing the supported units.
The following methods receive the parsed model configuration:

| Method | Result |
|---|---|
| `classify_window_batch(config, batches, context, *, timeout=None)` | One response with `.chunks`, containing one probability vector per requested window. |
| `encode_window_input(config, text, context, last_turn, *, timeout=None)` | Required for token windows: `TokenEncoding` from `quarry.integrations.model_windows`. |
| `decode_window_spans(config, raw, source, windows, threshold, *, encoding=None, token_scores=None)` | Required for spans: a sequence of scored `Span` values bound to the supplied full `SourceText`. |
| `window_token_counts(config, batches, context, *, timeout=None)` | Required when character windows have `max_input_tokens`: one nonnegative integer count per window, including context. |
| `render_window_input(config, messages, tools)` | Required for `renderer: host`: `WindowInput` or an opaque string; may be asynchronous. |

`TokenEncoding` contains detached native token IDs, optional `prefix_token` and `suffix_token`
flags, an optional half-open `last_turn` token interval, and `context_tokens`. Each endpoint flag
preserves exactly one token at that end in every batch. Equal IDs at different positions remain
distinct. Windows retain their positions in the full encoding. Counts and input limits include
preserved endpoints and context before classifier dispatch.

The optional `character_offsets` table has one entry per token: a half-open interval in the full
rendered text, or `None` when the token has no characters. Nonempty intervals must be ordered;
overlap is allowed. Endpoint tokens cannot claim nonempty text. Zero-width entries add no text
coverage. Without this table, token counts remain available but Quarry cannot infer character
coordinates from token IDs.

For token-head support, declare `supports_token_head = True` and accept the keyword
`token_head=False` in both the classification and span-decoding methods. The response's
`token_head` has one row per window, one entry per requested token, and a probability vector per
entry. Quarry validates the entire selected class before invoking the decoder; every selected
value must be a finite number in `[0, 1]`. Other classes do not establish the selected class's
validity. Token-window spans require this support. An ordinary non-window adapter may opt in
through the same declaration and keyword on `classify_messages` and `decode_spans`; older
adapter signatures continue to work without that keyword.

Quarry requests a token head for a selected span operation and physically compatible classifier
reads planned for the same input. Compatible aliases may share the raw provider reply while
retaining their own thresholds and class selections. Different inputs, tools, revisions, or
window settings remain separate. Legacy `model_score` and `model_scan` field reads participate
when they send the equivalent single user message. Custom response readers still run for each
logical read; their return values are not memoized as model scores.

Optional [window observations](window-observations.md) report the accepted reply,
per-reader cache status, and actual selected-input coverage without extra model or tokenizer
requests. They keep logical reads separate from physical attempts and do not alter verdicts.

For host-supplied spans with explicit complete or partial coverage, see
[external span adapters](external-spans.md). That mode uses its own dispatch contract
and does not require classifier or token heads for span-only policies.

Decode character spans using `CharacterWindow.span(...)` or the supplied `SourceText.span(...)`.
For token spans, use positions in the full encoding and the host's actual character mapping;
token indices are not character offsets. Keep the geometry returned by the source and add the
score or label. Quarry rejects spans that change their bound source, coordinates, or precision.
Copied text retains original provenance; authored text stays constructed and unmapped. Text
derived from media can retain part context without inventing image character positions.

An explicit host rendering uses `WindowInput` from `quarry.integrations.model_window_input`.
Its `SourceText` origin must be a bounded, acyclic graph of native `Place` and `Derived` nodes
covering only the selected input. An opaque string receives enclosing coordinates rather than
invented fine offsets. Host last-turn rendering must provide an explicit interval.

Window preparation and measurements are bounded and owned by the active scan or its deferred
evidence. Encodings and head-bearing replies have a fixed five-minute reuse limit and remain
subject to cache entry and byte limits. Hard bounds include one million encoding tokens,
10,000 windows, four million units of overlapping window work, and four million rendered input
and context characters. The configured comprehension limit can reduce the window count further.
Normal call budgets, rate limits, deadlines, cancellation, and logical-read attribution apply.

With explain enabled, direct classification reports the maximum window probability and the first
window attaining it, with bounded coordinate summaries. Arithmetic and aggregate conditions
keep their own deciding value. Summaries contain no input text, token IDs, or raw provider reply.
Unavailable token-to-character mapping is reported as unavailable; it is not guessed.

## Data egress

A model operation can send selected messages, tools, and policy text to the configured endpoint. Review provider retention, logging, TLS, redirects, and credentials before enabling model rules. Keep conversation content and text parameters out of logs and errors.

## Optional judge explanations

A score alias can request a brief, user-facing explanation separately from private model reasoning:

```yaml
models:
  judge:
    adapter: litellm
    model: your-provider/your-model
    score:
      message_builder: constitution
      boundary_seed: 7
      tokens:
        negative: "0"
        positive: "1"
      explanation:
        max_tokens: 256
```

`explanation.max_tokens` is an integer from 16 through 4096. It cannot be combined with `score.reasoning`. The built-in LiteLLM adapter asks for one `REASON: ...` line followed by a final `ANSWER: ...` line, reads only the public message content, and never reads provider reasoning fields. LF and CRLF separators are accepted. The answer must exactly match a configured verdict string, either directly or as a valid JSON string; malformed quoting is rejected unless it is itself the exact configured text. A JSON-decoded configured verdict takes precedence when its quoted spelling also equals another verdict's raw text. A valid answer without a reason is allowed and returns `reason=None`. Malformed reasons, invalid answers, refusals and truncated replies fail closed. Reasons are limited to 1024 characters and exclude control characters, Unicode formatting controls, surrogate code points and line/paragraph separators. Reply processing is bounded before splitting: the reply may use up to the reason limit plus the longest configured verdict spelling (including JSON escapes) and 64 formatting characters. System One does not support explanation generation.

The existing `model("judge").score(...)` rule expression and numeric Python score methods continue to return numbers. With explain enabled, score evidence includes a bounded `judge_reasons` list, each carrying a call ordinal, configured model alias, text and `untrusted: true`. At most eight reasons per model atom are retained; `omitted_judge_reasons` reports additional reasons. Evidence applies the existing redaction and excerpt limits. Split score reasons use call ordinals in chunk order, after the split call. CLI explanation output renders reasons as literal text and shows a dimmed omission count after the retained reasons when additional reasons were omitted. The plugin's explain evidence uses the same JSON representation. Reasons are excluded from decision-path score entries and telemetry.

Generated explanations are untrusted answer text. They may be inaccurate, include instructions or repeat sensitive conversation content. They are not proofs of the decision, should not be executed or inserted into trusted instructions, and need application-specific access and display controls. Generic evidence redaction is not a guarantee that every secret is removed. Ordinary scans without explain do not expose explanations in their results. Enabling explanation generation changes the model request and its cache and policy identity; it may change the model's answer compared with a separately generated token-only request.

Nested public-rule checks retain these bounded reasons and omission counts. Their call ordinals follow the logical calls observed while building the selected input and evaluating the check, including calls that returned no reason. The eight-reason cap applies to the enclosing check as well. Reasons remain absent from exported decision records and telemetry.

When probability extraction is configured, explanation-mode probabilities are used only when the provider's token metadata reconstructs the reply and identifies the final answer token. Partial, misaligned, oversized or invalid probability metadata falls back to the validated emitted verdict; text in the reason cannot supply decision probabilities. Quoted, escaped or compound configured labels also use the emitted verdict because one token's alternatives cannot establish odds for those whole labels.

For the same validated provider response, calibration and numeric comparisons depend only on the score. Requesting explain evidence does not make another inference request. Custom score adapters may return a `JudgeResult(score, reason)` or keep returning a float, which yields no reason. Quarry validates the structured result and retains its reason only when the alias explicitly enables explanations. A custom backend's overridden numeric score method remains authoritative in explain mode.

Verification includes property tests for bounded, finite result scores and JSON round trips, plus an exhaustive finite score/threshold/operator matrix comparing explained and ordinary scans. These checks establish the stated invariants over their tested domains; they are not a formal proof over arbitrary programs or provider behavior.
