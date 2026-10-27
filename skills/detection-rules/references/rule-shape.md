# Rule shape, generation, and shipping

Syntax comes from `quarry_reference` and `quarry_compile`, never from this file. This file says what
the rule must do.

## One rule per profile

- Write one rule per threat profile. It carries the category metadata your runtime routes on.
- The condition ORs one branch per inspected surface. Each branch scopes to the surface's new
  items, requires a prefilter hit (an OR of the RE2 groups over the surface string), then scores the
  same string with the judge and compares the score to the threshold parameter. Put the prefilter
  first, so a branch that does not match makes no model call.
- Score through the version 2 model interface, `model("<entry>").score(...)`, which requires
  `config_version: 2`. Comparing `llm_score(...)` with a rule parameter does not compile.

## New items only

A gateway resends the whole conversation every turn. A branch that inspects any item flags the
same item again on every later turn and pays for the judge each time. Scope each branch to what is
new in this request:

| surface | inspect | string |
|---|---|---|
| `tool_call` | every pending call: each tool call with no linked response | raw JSON arguments, the same bytes as the item's `text` |
| `tool_response` | every tool result since the newest assistant message: the trailing batch, which can answer several parallel calls | result body |
| `message` | the newest message, when it is the user's or the assistant's | content |

Quarry represents each tool call as its own event. A branch over only the last call of the newest
message skips earlier parallel calls in the same assistant turn. Their results can arrive together
in one request, so a branch over only the newest tool response skips earlier results in the batch.
Do not rely on the gateway sending one result per scan.

Where the target version has no form for "since the newest assistant message", the generator can
emit one clause per batch size k, up to a cap: the last k messages are all tool results, and the
k-th from the end matches. Use this fallback only when the host enforces a maximum trailing
tool-result batch size **before** rule evaluation. Set the clause cap to that maximum, and record
the enforced limit and its source in `inspection_scope` and the report. A typical or observed
maximum is not enough. If the host has no such bound, add an ingress check that rejects an
over-cap request or blocks it without allowing uninspected results through; otherwise do not
ship this fallback for tool responses. Never silently skip results beyond the cap.

Prove the scope with `quarry_scan`, using a prefilter-only copy of each branch. For tool calls, use
four conversations:

1. the newest assistant turn holds the matching pending call: it must match;
2. the same call has been answered and a newer message follows: it must not match;
3. the newest turn holds two pending calls with the match in the first: it must match;
4. an older answered call matches and the newest pending call does not: it must not match.

For tool responses, use four:

1. a single tool result is the newest message and matches: it must match;
2. two parallel results, with the match in the first: it must match;
3. a matching batch has been followed by an assistant and a user message: it must not match;
4. an older matching batch is followed by a new assistant turn and a benign batch: it must not
   match.

For a capped fallback, also scan a batch at the enforced limit with the match in its earliest
result; it must match. Submit one result over the limit and verify the host rejects the request or
blocks it before rule evaluation. An ordinary non-match on the over-cap input fails this gate.

For messages, use two: the newest message holds the match, which must match; the same item sits one
turn earlier, which must not.

## Tool definitions

Quarry's compiler rejects a model call inside a comprehension over the static `tools` collection, so
a tool-definition branch cannot call the judge. Check with `quarry_compile` for the version you
target. Where the judge cannot run there, either defer the surface, or ship a prefilter-only
tool-definition branch as a separate deterministic detector. Measure that detector on its own
recall and benign rate, and keep it out of the judge and calibration metrics. Definitions are
resent with every request, so its prefilter must stay cheap.

## Generate, never hand-edit

- Keep one spec per profile: category, title, surfaces, the taxonomy triple verbatim from the
  taxonomy source, and any secondary mapping. Keep the prefilter JSON and a calibration JSON beside
  it. A generator writes the rule, the judge texts, and the inference entries. A pattern fixed in an
  emitted rule and not in its source drifts.
- Declare the threshold as a policy-bound parameter (`threshold: number policy`), compare the
  score with `params.threshold`, and bind the value when the policy is compiled, for example with
  `compile_policy(rule_text, policy_parameters={"threshold": ...})`. The grammar has no parameter
  defaults, so record the value bound in step 8 in `threshold_note` and the calibration JSON. A
  deployment can then run other tiers by binding other values. If the target runtime cannot bind
  rule parameters, have the generator write the bound value into the rule instead.
- Metadata: the routing category, profile, description, author, revision, updated, the taxonomy
  fields, `threshold_note`, `inspection_scope`, and `generated_from`.
- Keep rules free of comments. Rationale belongs in metadata, the report, and the PR.
- Until the prefilter is designed, emit a placeholder group that cannot match.

## Judge entries

- Give each surface its own judge model entry. Each surface has its own policy block and its own
  score distribution, and Quarry attaches a calibration artifact to a model entry.
- Each entry needs the message builder the serving path uses, first-token `"0"`/`"1"`, and one text
  resource per policy block. Replace entries by key, so a rerun never duplicates them.
- The entry must return a consistent, granular score in `[0, 1]`. With the one-token judge, enable
  probability extraction (`score.extraction.probabilities`): a bare `0`/`1` score makes every
  threshold behave the same.
- Attach the calibration artifact through `score.calibration`.

## Compile against the version that runs it

Compile with `quarry_compile` from the Quarry version that will serve the rule, and load the whole
rules package with that version before shipping.

## Engine against evaluator

The offline evaluator is not the engine. Copy each rule with the judge call removed, leaving the
prefilter only. Wrap each test item in the smallest conversation that places it on its surface,
and scan with the real engine. The hit set must equal the offline prefilter evaluator's, surface by
surface. The same harness runs the deterministic baseline; skip its rules that call a model.

## Shipping

Follow your rules repository's own process for layout, categories, CI datasets, and review.
