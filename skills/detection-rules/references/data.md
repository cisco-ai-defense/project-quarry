# Candidates, labels, benign pools, and splits

## One item format for every stage

An item is one thing a rule inspects. Its `text` is the exact string the rule hands the judge on
that surface, so the labeler, the prefilter, and the judge all see the same bytes. Store items as
JSONL, one object per line.

| field | meaning |
|---|---|
| `id` | `<profile>:<source_slug>:<n>`, stable across reruns; the split depends on it |
| `profile` | the profile the candidate was collected for |
| `surface` | `tool_call` \| `tool_response` \| `tool_definition` \| `message` |
| `text` | the string the rule tests on that surface (below) |
| `tool_name` | the native tool name, for tool surfaces |
| `role` | `user` \| `assistant` \| `system`, for messages |
| `source`, `source_row` | dataset or repository, and the row or path inside it, for audit |
| `license` | as verified when the source was collected |
| `source_label` | the source's own label, else `null`; recorded, never trusted alone |
| `label_type` | `human` \| `synthetic_llm` \| `rule_generated` \| `unlabeled` |
| `projection` | `native`, or the name of the wrapper applied (below) |

`text` per surface:

- `tool_call`: the raw JSON argument string of the call, with the tool's own keys.
- `tool_response`: the body the tool returned.
- `tool_definition`: the tool's description.
- `message`: the message content.

## Positive candidates

- Search public datasets per threat: Hugging Face, GitHub corpora, benchmark suites, and the
  deterministic signatures' own fixtures. Record license and label type per source.
- When a source lacks the surface's shape, wrap it the same way for every row and name the wrapper
  in `projection`. For example, a bare shell command becomes `{"command": "<cmd>"}` with tool
  `Bash`. A wrapper must add no signal: the same template for every row, nothing that correlates
  with the label.
- Deduplicate on `(surface, text)` within a profile before splitting.
- Cap any one source, with a fixed seed, so no single source dominates recall; 3,000 per profile
  is a workable cap.
- Write a manifest per profile: counts per source and surface, cap, projection, and sha256.

## Labels

- One labeler call per item. The constitution is a cached system block; the header and the unit
  are the user message.
- Leave a large output budget (about 4,000 tokens). Models may reason before the JSON, and a small
  budget cuts those replies off. Record the stop reason.
- Keep every row. Unparsable replies (`parse_error`) and refusals stay in the denominators with
  their outcome.
- Never count a refusal as benign. Relabel refusals with a second labeler against the same
  constitution, and record `labeler` on every row. First measure the two labelers' agreement on a
  sample. A fallback labeler often agrees only partly and labels more strictly, so report its slice
  separately.

## Benign pools

- Build one pool per surface, shared by every profile that inspects that surface: shell commands,
  file writes, other tool calls, tool responses, tool definitions, and messages. Add
  profile-specific pools where a surface has a special benign population, such as agent
  configuration files in public repositories.
- Draw from benign traffic from your own deployment, with public coding-agent traces as the
  fallback. Draw tool definitions from public tool registries. The pools are unlabeled and assumed
  benign; the labeler verifies only the items the judge flags.
- Deduplicate tool calls on canonical JSON of the arguments, and replace base64 images with a
  placeholder. On large surfaces, give each source an equal share and let smaller sources take all
  they have, so one source does not dominate.
- Size each pool with the planning rule n ≈ 4 / (r²p), counted per calibration fit, for target FPR
  p and relative 95% half-width r. For 0.1% at ±25% that is about 64,000 benign items per surface,
  a rough 95% range of 0.075% to 0.125%; 15,000 items give about ±52%. Resolution is not
  confidence: one flag in 10,000 items resolves 0.01% but says little about it.
- Write a manifest: per-source counts, length percentiles, and sha256 per file and per split.

## Splits

Split on the input the detector sees, not on the item id: the same input can carry different ids
across profiles and pools. Key each item, positive or benign, by its surface and canonical text
(tool-call arguments as canonical JSON with sorted keys; other surfaces as is), and assign it by
`int(sha256(key), 16) mod 3`: 0 to `dev`, 1 to `accept`, 2 to `test`. The same input then falls in
the same split everywhere. Record the split in the manifest before any scoring, and never re-split.

When the same input is a labeler-confirmed positive and a benign-pool item for the same profile,
drop it from that profile's benign items.

- Make every design choice on dev: prefilter atoms, policy variants, calibration fits.
- Make every accept-or-reject decision on accept, against a gate written before the run. Record
  each use of accept, and disclose repeated use in the report.
- Report once on test.
