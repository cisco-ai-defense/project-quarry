# Quarry Agent Plugin

The Quarry plugin gives Codex, Claude Code, and other MCP hosts the same five read-only tools.

| Tool | What it does | What it does not do |
|---|---|---|
| `quarry_reference` | Searches the installed language catalog and returns full records with tested examples. | It does not invent syntax or search the web. |
| `quarry_compile` | Runs a complete rule through Quarry's parse, validation, link, resource, and compile path, and reports validity, diagnostics, metadata, parameters, and required capabilities. `visualize` adds the static source graph. | It does not save the rule or return its decision tree; `quarry_decision_tree` does. |
| `quarry_decision_tree` | Compiles a rule and returns its decision tree: each check, its stable ID, and where its true and false outcomes lead. | It does not scan; a scan's `decision_path` names the checks it reached. |
| `quarry_decision_flows` | Counts scans' `decision_path`s through one rule's tree, by step outcome and leaf, optionally by label and at moved score or number thresholds. | It does not scan again; a moved threshold that reaches a check a scan never answered counts as `unknown`. |
| `quarry_scan` | Compiles a rule and scans explicit conversation JSON. | It does not read files, sessions, or stored records. |

## Install from the repository

The repository is one installable unit:

```text
GitHub repository
    -> local plugin installation
       -> agent instructions in skills/quarry-policy/ and skills/detection-rules/
       -> host file resolver in hooks/
       -> thin MCP entry point: quarry-mcp
          -> installed Quarry catalog, compiler, and scanner
```

Codex installation metadata lives in `plugin.json`, `mcp.json`, and
`.agents/plugins/marketplace.json`, with `.codex-plugin/plugin.json` for compatibility. Claude Code
uses `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`, and `.mcp.json`. Both hosts
start the same `quarry-mcp` entry point and load both skills: `skills/quarry-policy/SKILL.md` for
writing, compiling, and scanning rules, and `skills/detection-rules/SKILL.md` for building
and validating a production threat-category rule.

Codex can install the marketplace directly from GitHub:

```shell
codex plugin marketplace add cisco-ai-defense/project-quarry
codex plugin add quarry@project-quarry
```

The marketplace downloads the repository locally. The MCP code only validates tool calls and
translates results. It calls the installed Quarry catalog, compiler, and scanner directly.

| Part | Runs where | Responsibility |
|---|---|---|
| Agent skills | Plugin installation | Teach the agent how to search, write, compile, and scan rules, and how to build and validate a threat-category rule. |
| MCP adapter | Local process | Exposes five tools and validates their inputs. |
| Quarry package | Same local installation | Owns the language, compiler, configuration, and scanner. |

Splitting these into two downloads would add another version and failure boundary without reducing
what must run locally. The MCP adapter should stay small, but the Quarry engine must still be
installed because it performs the work.

The launch configuration uses `uv` and the repository lock file. The host must therefore have
`uv` available.

## Typical use

1. Search the reference in plain language.
2. Retrieve the full records and tested examples for the selected IDs.
3. Write a complete rule.
4. Compile the rule.
5. Scan an explicit conversation and inspect `completion_state` before using `matched`.

```json
{
  "action": "search",
  "query": "file read followed by HTTP request within three events"
}
```

```quarry
rule "File data sent"
metadata:
    category = "data_exfiltration"
    severity = "high"
condition:
    called_tool("http_request") after(within: 3) called_tool("read_file")
```

Search is deterministic. It normalizes query words, applies a small synonym table, and
ranks matches from identifiers, signatures, syntax, descriptions, parameters, and packaged
examples. Each query word receives the weight of its strongest matching field, so repetition
across fields or linked examples does not multiply its score. A complete multiword name match
gets an additional boost. Tested examples are ranked using their rule text. Callable functions
and compiler forms used in an example rule receive a smaller relevance signal only when their
own catalog fields also match the query, even if the example's coverage list does not name them.
Receiver calls such as `model("alias").score(...)`, `message(...).add_message(...)`,
and `messages.called_tool(...)` count under their catalog names. Quoted strings and
comments do not count as calls.
Functions and examples win score ties over tokens. A worked example close to the tenth result's
score can enter the top ten when none is present; unrelated examples are not promoted just to
fill a fixed quota. Each result lists the fields that matched.

A `get` accounts for every ID it was given in exactly one of three places:

| Key | Meaning | What to do about it |
|---|---|---|
| `results` | The record, with its linked tested examples | Read it |
| `missing_ids` | No record has that ID | Search for the feature by name; the ID was wrong |
| `filtered_ids` | The record exists, but `kinds` excluded it | Drop the `kinds` filter or widen it; the ID was right |

Reporting the third case keeps an agent holding a correct ID from searching again for a record it
already has. `missing_ids` and `filtered_ids` are different mistakes
with different fixes, which is why they are reported apart. Both ID lists are present even when
empty, unlike the record keys described next: this is the response's account of the request, and a
caller reading an absent key would have to guess whether nothing was filtered or nothing was
reported.

`quarry_compile` also reports advisory detection-quality warnings. A high-severity warning
identifies a condition proven always true or false for complete scans, using the compiled
automaton and known facts such as `len(messages) >= 0`. A medium-severity warning identifies
a rule with no specific event or message-content anchor, such as a broad collection-size
check or a test of tool availability. A count of calls or responses for a named tool
identifies a specific event signal. Checks of a tool's description or schema fields also
count as specific signals; name-only availability checks such as `has_tool("name")` remain
broad. The anchor check follows atoms that can affect a compiled automaton transition,
so a content predicate eliminated from every transition does not suppress the medium warning.
If advisory analysis fails, compilation continues without that warning. These warnings
do not change `valid`, compilation, or scan verdicts. The analysis is conservative: a
rule may have a quality issue without receiving a warning. It proves relationships
between comparisons of the same collection size, including
`len(messages) > 5 and len(messages) < 3`, by considering possible nonnegative integer
counts. The analysis considers at most 32 combinations of independent collections and
leaves additional counts unconstrained, so some contradictions can still be missed.
It also treats distinct collections and event predicates independently, and a predicate
that appears in a rule may receive no medium warning even if it has no effect on the
verdict. A rule with a `collect` clause still receives a proven never-match warning,
but skips always-true and no-anchor condition warnings because its collector may provide
the detection signal or yield no output. A per-candidate rule skips an always-true
condition warning because a conversation may contain no candidate. `quarry_scan` carries
these warnings under `compilation.warnings` because it reuses the same compile report.
The CLI also shows them in `quarry scan --validate-only`. High detection-quality warnings
fail that command; medium detection-quality warnings are shown without changing its exit
code. Existing validator warnings keep their prior exit behavior.

### Parameterized scans

`parameters:` is the rule's typed input signature. An unmarked declaration is supplied for each
scan. Add `policy` only when the value is fixed on the compiled policy:

```quarry
rule "Customer guardrail"
parameters:
    instructions: text policy
    threshold: number
condition:
    model("guard").score(messages, params.instructions) >= params.threshold
```

Pass all declared values in the existing `parameters` input to `quarry_scan`. For an ordered
`conversations` batch, Quarry compiles the rule once and applies the same parameter map to every
conversation. Text values are represented by SHA-256 in results; numeric values remain visible.
Production code can use `compile_policy` to retain the resulting `CompiledPolicy` and avoid
compilation on the scan path. The portable plugin accepts at most 100,000 characters across all
text parameter values in one request.

The plugin server reuses the model results its scans have already obtained. A score,
classification, completion, token count or compatibility call (`llm_score`, `llm_complete`,
`llm_input_tokens`) for the same model and input is not requested again, whichever scan and
rule version asked first, so scanning the same conversations again, with a new version of
a rule or another threshold, costs no model call for what was already scored. The threshold is not
part of a score's input. Scans that send identical input share its result, whoever sends it;
nothing else is shared, but a reused result returns faster than a new call. The server shares
one cache across all its rules, as every `DSLScanner` does across its own scans (see
[USAGE.md](USAGE.md#result-caching)).

Results are keyed by a digest of the request each call sends (the operation, the adapter, the
model's provider settings and the payload's inputs), which the key does not retain.
Results expire `model_cache.ttl_seconds` after their last use (12 hours by default). Past
`model_cache.max_entries` (100,000 by default, counting model, regex and prober results alike) the least recently used result is dropped:

```yaml
model_cache:
  ttl_seconds: 43200
  max_entries: 100000
```

The cache lives in the server process, so a host keeps one server running across scans to
benefit from it. A shared call is made under the model-call budget and deadline of the scan
that started it; another scan waiting on it that sees it fail because that scan ran out of time or budget retries under its own, and any other failure reaches every waiter. An
output that fails Quarry's checks after the call, such as a completion outside its response
schema, is dropped, so the next scan asks the model again.

Long-lived hosts with their own model backend can use the same bounded prepared-rule cache as the
plugin without adopting the plugin's response format:

```python
from quarry import CompiledPolicy, PreparedRuleCache, compile_policy

prepared = PreparedRuleCache[CompiledPolicy](max_entries=64)
policy = prepared.get_or_prepare(
    rule_text,
    runtime_identity=runtime_digest,
    prepare=lambda source: compile_policy(
        source,
        runtime_config=config,
        model_inference_backend=backend,
    ),
)
```

`runtime_identity` is a non-secret digest or version label covering every runtime setting that can
change the compiled policy, including model profiles, text resources, configuration version and
policy-bound parameters. The cache is worker-local and opportunistic. Cache only prepared rule
artifacts, never conversations or scan results; model results have their own cache, above. A failed preparation is retried on a later call.

## What a response spends

A reference response is the agent's context, so it carries what applies to the request and
nothing else.

| Choice | Behaviour | Why |
|---|---|---|
| Keys that do not apply are absent | A missing key means "not applicable", never "withheld" | `"replacement": null` on nine records out of ten costs the caller for a fact the absent key states as well. A false flag is dropped only when it names a restriction or a precondition, where absence reads as "unrestricted" and says exactly what `false` said. A flag that asserts a property is spelled out in both polarities: `model_input: false` on `system_prompt` is a positive fact about that surface, and `required` on a parameter and `expected_match` on an example check are the same. An empty *collection* is kept unless its emptiness says nothing: `allowed_uses: []` means the form is allowed in no expression position, which the compiler acts on, so only a redundant empty like a parameter's unrestricted `allowed_values` is dropped. |
| `operator_precedence` accompanies `action: index` | Pass `include_precedence: true` to read it on any other action, or `false` to refuse it on `index` | The table is identical on every call. Repeating it on each search and each get billed for the same paragraph again without ever saying anything new. |
| `get` defaults to `verbosity: "compact"` | It names each tested example, its `rule_text`, each check's `input_path`, and whether the check is meant to match. `verbosity: "full"` is the ground-truth view: it keeps every empty and null field, each example's coverage list, and spells out the conversation each check scans | Compact prunes absences for its budget; the conversations are the bulkiest thing in the catalog and the part an author reads least. When an author needs to tell a field that does not exist from one that was not set, `full` is one parameter away. |
| A linked example omits `covered_item_ids` under `compact` | `get` on the example's own id returns the list, and `verbosity: "full"` keeps it on the linked example too | Reached through a record it covers, the list answers a question the caller did not ask, and one popular example repeats it under every record in the response; `full` trades that brevity for completeness. |

Compiler bookkeeping — `source`, `handler`, `compiler_tags`, `accessor_filter`,
`accessor_surface`, `implicit_receiver_count` — is never returned. A rule author cannot write
it and an agent cannot act on it. `resource_group` is reported under `availability`, which is
where the answer about a capability belongs.

### Local file inputs

Local Codex and Claude Code installations can pass workspace file markers without changing the
three MCP tools. The bundled `PreToolUse` hook validates and reads each file, then replaces the
markers before the MCP call:

```text
agent -> small file markers -> host hook -> resolved values -> unchanged Quarry MCP tool
```

The bundled `SessionStart` hook supplies the local marker instructions. The shared Quarry policy
skill remains host-neutral.

Compile or scan a rule file with `quarry+file://` in `rule_text`. Supply a single conversation file
with `file`:

```json
{
  "rule_text": "quarry+file://rules/check.rule",
  "conversation": {"file": "samples/one.json"}
}
```

For an ordered batch, provide one marker per conversation:

```json
{
  "rule_text": "quarry+file://rules/check.rule",
  "conversations": [
    {"file": "samples/one.json"},
    {"file": "samples/two.json"}
  ]
}
```

One `batch_file` marker can instead name a JSON array of conversations. The hook expands it
into the same `conversations` input. Quarry compiles the rule once, scans up to 1,000 items, and
preserves input order.

The paths must be relative to the host's current workspace. The hook rejects traversal, symlink
escapes, non-files, unexpected extensions, invalid UTF-8, invalid JSON, oversized inputs, and mixed
batch markers. Rule paths must end in `.rule`; conversation and batch paths must end in `.json`.
The hook validates all paths before it reads any requested file. A resolution failure blocks the
tool call. The host runs the hook only after the user or administrator trusts the installed hook
definition. Codex requires an `allow` decision to rewrite MCP arguments, so this installation trust
also authorizes matching local Quarry calls without another prompt. The hook matches only the
bundled Quarry server names and the project-local `quarry` server name. Host debug traces can include
rewritten arguments, so do not enable them for sensitive conversations.

The MCP server never receives paths or file markers. Other applications can resolve authorized
artifacts into the same value-based inputs.

## Spellings Quarry rejects

Some plausible-looking spellings are rejected on purpose, so the error is predictable rather
than a silent reinterpretation. `action: rejected` returns that whole table, or only the rows
matching an optional `query` substring:

```json
{
  "action": "rejected",
  "query": "messages"
}
```

Each row gives the `source_form` Quarry refuses, the stage that refuses it (`validate`, `link`,
or `compile`), and the `replacement` to write instead when one exists. The table is small by
design and is returned whole, so `limit` does not apply to this action; generation fails if the
table ever outgrows the size that makes a single response affordable.

A compile that fails at the link stage carries the same guidance inline. An `unknown_function`
diagnostic adds `operator_keyword` when the name is an operator used with parentheses,
`suggestions` with the nearest installed names, or `rejected_spellings` for a documented
mistake. A `rejected_spellings` row is an **example, not a substitution**: it names the
`documented_form` and its `documented_replacement`, both spelled with the arguments the table
happens to document. Apply the pattern to the rule at hand and keep its own operands — an
author who wrote `first(messages)` is shown `first(tool_calls)` → `tool_calls[0]` and must
write `messages[0]`, not `tool_calls[0]`.

## What a scan returns

Quarry decides whether the supplied rule matched. It does not create a security taxonomy.

If the rule contains taxonomy labels in `metadata`, the scan copies those labels into the match:

```json
{
  "completion_state": "complete",
  "matched": true,
  "matches": [
    {
      "rule_name": "File data sent",
      "metadata": {
        "category": "data_exfiltration",
        "severity": "high"
      }
    }
  ]
}
```

The rule author owns the metadata names and values. Quarry only preserves them.

### Why a rule matched or did not

With `explain` on (the default), a complete scan adds `explanations`: one entry per rule,
matched or not, rendered from the same scan record. A match cites its supporting positions
in `witness`. Every entry lists the rule's atoms, the actionable ones first:

```json
{
  "rule_name": "inj",
  "matched": false,
  "reason": null,
  "witness": [],
  "atoms": [
    {
      "atom": "tool_calls.name = \"send_email\"",
      "status": "false",
      "reason": null,
      "cost": "pure",
      "true_positions": [],
      "false_positions": [2],
      "omitted_positions": 0,
      "excerpts": [{"position": 2, "excerpt": "send_mail"}],
      "candidate_events": 1,
      "explored": true,
      "note": "computed for explanation; the verdict did not need it",
      "source_nodes": ["n7"],
      "model_results": [],
      "node": "t:5a57..."
    }
  ],
  "omitted_atoms": 0
}
```

| Field | Meaning |
|---|---|
| `status` | `satisfied`: true somewhere. `false`: computed everywhere, never true. `no_candidates`: the surface had no events. `unknown`: not computed everywhere; `reason` says why. |
| `reason` | For `unknown` only: `budget_exhausted` (the explanation pass ran out of time or model calls), `model_failed`, or `captured_value` (the atom tests a value bound elsewhere; its positions are those the scan's bindings computed, which is not every value). |
| `true_positions`, `false_positions` | Event positions, not message indices: an assistant turn with a tool call is two events. Only events on the atom's surface are cited. |
| `excerpts` | The field the atom tests at those positions, such as a tool call's name. |
| `explored` | The verdict did not need this atom; the explanation pass computed it. |
| `model_results` | Scores a model atom returned, up to 2. |
| `note` | The diagnosis, such as a case-only mismatch, then where the evidence came from. |
| `node` | The decision-tree check this atom belongs to, the same ID `decision_path` uses. |

An indexed atom such as `user_messages[-1].content contains "x"` is computed only at its
event. A slice such as `user_messages[5:]` is listed as its own atom, `event in
user_messages[5:]`, which is `no_candidates` when the window is empty.

The entry's own `reason` is `empty_collection` when the condition held but `collect:` kept
nothing, so the rule did not match. It is null otherwise.

The explanation pass runs after every verdict of the request, batch included, under its own
`explore_timeout` and model-call budget for each conversation. It may call models, but it
never changes the verdict, `completion_state`, diagnostics, or `decision_path.path`. Output
is bounded: 24 atoms per rule, 8 positions and 2 excerpts per atom.

Set `explain: "support"` for a match's minimal support alone. It is read from the verdict's
own pass, so it runs no explanation pass and calls no model the verdict did not: only
matched rules are listed, each with its `witness` and no `atoms`.

### Visualization response

Compile responses always include a static `visualization` bundle. Scans record and return
runtime visualization only when `quarry_scan` is called with `visualize: true`, because its size
grows with the conversation. With it, the scan returns the static bundle once at
`compilation.visualization`, and each entry of the response-level `visualization.graphs[]`
holds only `graph_id`, `rule_name`, and runtime data that refers to static nodes by `node_id`.
Without it, a scan returns neither.

Version 1 of the static bundle contains one `source_ast` graph per loaded rule. Each graph has a `root`, typed `nodes`, and typed
`edges`; nodes are authored AST occurrences, and `compiled` carries the DFA hash, atom IDs,
state count, and compiler notes. A node's `atom_ids` are lowered directly from that occurrence;
`subtree_atom_ids` also includes its descendants. `state_atom_id` names the one atom whose value
is the occurrence's own truth value: a predicate such as `m.content contains "x"`, or the
quantifier result of a tools comprehension. It is `null` for composites (`and`, `not`, flow,
multi-step functions such as `has_sequence`) and for timeline comprehensions, whose direct atoms
are surface gates or collection filters that restrict a domain rather than decide the node.

A single-conversation scan with `visualize` returns `runtime_attribution` for each graph. Its
`trace` is the engine's canonical `WitnessTrace`, the same trace explanations are built from:
attempts remain separate, lazy atoms are explicitly `skipped`, and unmatched attempts past the
first 16 are counted in `omitted_attempts` instead of kept. An evaluation marked `witness`
has clear support from a selected DFA transition or tool check. The engine uses the recorded
guard and does not infer a proof for every negative or temporal combination. An unmarked
evaluation remains an observation, even when its message is cited in the explanation.
This is deliberately conservative: it does not claim that every relevant atom is highlighted
or that a flagged atom is logically necessary under every possible proof. The original
attempts retain all scan observations; `proof_attempt` records a minimized replay and may select
a different capture binding. When it does, `trace.selected_attempt_id` names `proof_attempt`
rather than an entry in `attempts`, and the explanation's `captured_value` reports the proof's
binding. With `explain` off there is no replay, so `captured_value` is the first binding the
scan matched. Universal and absence claims keep the original input because deleting an
element changes their domain; their positions cite visited in-domain events independently
of atom flags. Simple one-predicate claims can flag decisive checks. Compound claims may
cite the complete visited domain while leaving every atom unflagged. A match with no
in-domain events, such as an `all` over an empty collection, can cite no positions.

A tools comprehension is evaluated as one coherent boot-step question; its
`tool_attempts` retain each checked tool separately inside the enclosing timeline attempt,
without replaying the timeline for every tool. A nested check's `matched` is the result of
its tool body, not a separate full-rule verdict; `selected` identifies one success for a
true `any`, every failure for a false `any`, every success for a true `all`, or one failure
for a false `all`, provided the quantifier belongs to the final rule proof. An empty tool
list makes `any` false and `all` true. `node_evidence` joins
those canonical evaluations to authored node IDs without copying their truth or causal state;
its `phase` distinguishes original observations from the proof replay.
Tool-body evaluations carry `body_occurrence` and `source_node_index`, so identical
authored atoms remain distinct: a skipped second branch stays skipped even when its
atom ID was evaluated in the first branch. When an interned DFA atom has no occurrence
identity, attribution stays at the nearest shared expression instead of claiming that
each identical authored branch ran.
Human explanations include model results only for witness-marked evaluations from the
selected trace, including failed checks selected under a negated tools condition. Each
explanation lists at most the first 16 cited steps in `witness` and counts the rest in
`omitted_witness_steps`; with `visualize`, `runtime_attribution.trace.positions` holds every
cited position.
Each attempt is projected independently onto the static graph's nodes; the top-level `nodes`
convenience projection is present only for the selected or sole unambiguous attempt. An edge's
runtime state is its target node's state, so projections carry nodes only. Node `state` is one of `unknown`, `skipped`, `evaluated_false`, or `evaluated_true`. It summarizes only the
evaluations of the node's `state_atom_id` in that attempt: `evaluated_true` means the node was
observed true at least once, not that it was true in the proof. `witness_state` is the value of
the node's witness-marked evaluations when they all agree, and `unknown` otherwise. A node
without a `state_atom_id` always reports `unknown` for both; the plugin does not re-evaluate
Boolean, temporal, or quantifier composition. The historical `causal` field means "supported
by a selected transition or tool check"; it does not mean strict logical necessity and never
changes the evaluated value. `causal`, `positions`, and `witness_positions` propagate from atoms
to every ancestor, so a composite node on the proof path is `causal` with `state: unknown`.
For example, in `not (all m in user_messages where m.content contains "ok")` over user messages `"ok"` and
`"bad"`, the predicate reports `state: evaluated_true` (message 0), `witness_state:
evaluated_false`, and `witness_positions` naming message 1; the `all` and `not` nodes are causal
with `state: unknown`. Position `-1` denotes the static boot step. Capture indexes belong to
timeline attempts; tool indexes belong to nested checks.

```json
{
  "compilation": {
    "visualization": {
      "schema_version": 1,
      "graphs": [
        {
          "schema_version": 1,
          "representation": "source_ast",
          "graph_id": "Bad request:…",
          "rule_name": "Bad request",
          "root": "n0",
          "nodes": [
            {
              "id": "n0",
              "node_type": "FunctionCall",
              "label": "user_said(\"bad\")",
              "attributes": {"name": "user_said"},
              "atom_ids": ["atom_bad"],
              "subtree_atom_ids": ["atom_bad"],
              "state_atom_id": "atom_bad"
            },
            {
              "id": "n1",
              "node_type": "Literal",
              "label": "\"bad\"",
              "attributes": {"value": "bad", "raw_value": "bad"},
              "atom_ids": [],
              "subtree_atom_ids": [],
              "state_atom_id": null
            }
          ],
          "edges": [
            {"source": "n0", "target": "n1", "role": "args[0]"}
          ],
          "compiled": {
            "dfa_hash": "…",
            "compiler_version": "0.1.0-phase1-tool-occurrences",
            "num_states": 3,
            "atom_ids": ["atom_bad"],
            "temporal": true,
            "notes": []
          }
        }
      ]
    }
  },
  "visualization": {
    "schema_version": 1,
    "graphs": [
      {
        "graph_id": "Bad request:…",
        "rule_name": "Bad request",
        "runtime_attribution": {
          "status": "complete",
          "matched": true,
          "trace": {
            "positions": [0],
            "selected_attempt_id": "default:0",
            "minimized": true,
            "attempts": [
              {
                "attempt_id": "default:0",
                "kind": "default",
                "binding_index": null,
                "capture_binding_index": null,
                "tool_binding_index": null,
                "matched": true,
                "selected": true,
                "evaluations": [
                  {
                    "atom_id": "atom_bad",
                    "position": 0,
                    "status": "evaluated_true",
                    "value": true,
                    "witness": false,
                    "source_node_index": null,
                    "body_occurrence": null
                  }
                ],
                "tool_attempts": []
              }
            ],
            "omitted_attempts": 0,
            "proof_attempt": {
              "attempt_id": "default:0",
              "kind": "default",
              "binding_index": null,
              "capture_binding_index": null,
              "tool_binding_index": null,
              "matched": true,
              "selected": true,
              "evaluations": [
                {
                  "atom_id": "atom_bad",
                  "position": 0,
                  "status": "evaluated_true",
                  "value": true,
                  "witness": true,
                  "source_node_index": null,
                  "body_occurrence": null
                }
              ],
              "tool_attempts": []
            }
          },
          "node_evidence": [
            {
              "node_id": "n0",
              "phase": "original",
              "attempt_id": "default:0",
              "atom_id": "atom_bad",
              "position": 0
            },
            {
              "node_id": "n0",
              "phase": "proof",
              "attempt_id": "default:0",
              "atom_id": "atom_bad",
              "position": 0
            }
          ],
          "attempts": [
            {
              "attempt_id": "default:0",
              "kind": "default",
              "binding_index": null,
              "capture_binding_index": null,
              "tool_binding_index": null,
              "matched": true,
              "selected": true,
              "nodes": [
                {
                  "node_id": "n0",
                  "state": "evaluated_true",
                  "witness_state": "unknown",
                  "causal": false,
                  "positions": [0],
                  "witness_positions": []
                },
                {
                  "node_id": "n1",
                  "state": "unknown",
                  "witness_state": "unknown",
                  "causal": false,
                  "positions": [],
                  "witness_positions": []
                }
              ],
              "tool_attempts": []
            }
          ],
          "proof_attempt": {
            "attempt_id": "default:0",
            "kind": "default",
            "binding_index": null,
            "capture_binding_index": null,
            "tool_binding_index": null,
            "matched": true,
            "selected": true,
            "nodes": [
              {
                "node_id": "n0",
                "state": "evaluated_true",
                "witness_state": "evaluated_true",
                "causal": true,
                "positions": [0],
                "witness_positions": [0]
              },
              {
                "node_id": "n1",
                "state": "unknown",
                "witness_state": "unknown",
                "causal": false,
                "positions": [],
                "witness_positions": []
              }
            ],
            "tool_attempts": []
          },
          "nodes": [
            {
              "node_id": "n0",
              "state": "evaluated_true",
              "witness_state": "evaluated_true",
              "causal": true,
              "positions": [0],
              "witness_positions": [0]
            },
            {
              "node_id": "n1",
              "state": "unknown",
              "witness_state": "unknown",
              "causal": false,
              "positions": [],
              "witness_positions": []
            }
          ]
        }
      }
    ]
  }
}
```

| Output | Meaning |
|---|---|
| `completion_state` | Whether the scan completed, was invalid, or stopped incomplete. |
| `matched` | Whether at least one rule matched. Present only after a complete scan. |
| `matches` | Matched rule names and their authored metadata. |
| `explanations` | Why each rule matched or did not, when `explain` is enabled: up to 16 cited steps and the rule's atoms; see [Why a rule matched or did not](#why-a-rule-matched-or-did-not). |
| `visualization` | Runtime attribution keyed to `compilation.visualization` by node ID. Present only when `visualize` is true. |
| `diagnostics` | Validation, timeout, truncation, or evaluation problems. |
| `policy_digest` | Identity of the compiled policy, its fixed values, and referenced runtime configuration. |
| `parameter_digest` | Identity of all effective parameter values for this scan. |
| `parameter_evidence` | Numeric values and SHA-256 evidence for text, with `scan` or `policy` binding. |

For an ordered batch with `visualize`, the response returns the static graph once at
`compilation.visualization`. Each result's `visualization` is a runtime-only bundle keyed
by rule name, and the response-level `visualization.graphs[]` repeats those records, unchanged,
in `executions`. `aggregate` counts executions per node. Within one execution a node counts
once: its state is the highest observed across that execution's attempts, proof replay, and
tool checks (`evaluated_true`, then `evaluated_false`, then `skipped`), and it adds to
`witness_count` if any of them marks it causal. `evaluated_count`, `true_count`, and
`skipped_count` describe observations rather than proofs and count only nodes with their own
`state`; a composite on the proof path adds only to `witness_count`. A batch timeout uses the
same shape with an empty `executions` list and zero counts.

## Trusted runtime configuration

| Need | Configuration |
|---|---|
| Inspect conversation data | None |
| Use configured models normally | Put the settings in `~/.quarry/plugin.yaml` |
| Use a different file for a project, CI job, or test | Set `QUARRY_PLUGIN_CONFIG` to that file |

When the local MCP server starts, it looks for `~/.quarry/plugin.yaml`. If that file is absent,
model-backed features report that configuration is missing while other rules continue to work.

Use `QUARRY_PLUGIN_CONFIG` only to load a different file, such as a project, CI, or test
configuration. An explicit path takes precedence over `~/.quarry/plugin.yaml` and fails clearly
when the file is missing or invalid. The file may contain models, text resources, safety limits,
and the configuration version. It must not contain `rules`; every tool call supplies its own
`rule_text`.

```yaml
config_version: 2
models:
  guard:
    model: provider/model-id
```

The model name remains explicit because it controls where requests go. The adapter defaults to
LiteLLM, which routes the request using the provider prefix and its normal provider credentials.
Set `adapter` only for a specialized backend. If a rule names a missing model alias, compilation
fails instead of choosing another model.

Model aliases, backend endpoints, named `text_ref` resources, safety limits, and credentials are
server settings. Declared text parameters are tool inputs by design; they are bound into the ad hoc
policy for that scan and represented only by SHA-256 in results.
Configured model rate limits are shared across every compile and scan handled by that server.

## Request limits

| Input | Limit |
|---|---:|
| Reference search query | 4,096 characters |
| IDs in one reference `get` request | 50 |
| One reference ID | 200 characters |
| Rule text | 100,000 characters |
| Conversations in one batch | 1,000 |
| Text across all parameter values | 100,000 characters |

`limit` bounds `index` and `search`. `get` and `rejected` reject `limit`; a `get` returns every
requested ID, with duplicates collapsed in the response after enforcing the 50-ID input bound.

Reference and compilation work run outside the MCP event loop. Scanning awaits Quarry's async
engine directly so transport cancellation reaches the active scan. Conversation size and nesting
are enforced by Quarry's configured safety limits before evaluation.

## Result guarantees

- Reference output comes from the catalog packaged with the running Quarry version.
- The reference index includes the operator-precedence table. Other reference actions include it
  only when `include_precedence` is true.
- Local features report `available`. Configured model features report `requires_compile` because
  the complete rule determines which alias and adapter operation it needs.
- Detailed examples include the packaged rule, input JSON, and declared result.
- Compile output includes a rule hash, versions, failed stage, diagnostics, deprecations, and
  required capabilities.
- A scan compiles its supplied rule once. Single and batch inputs use the same compiled-policy
  execution path, and batch results preserve input order.
- Scan output includes `matched` only when `completion_state` is `complete`.
- Match metadata is copied from the rule. Quarry does not infer or replace the caller's taxonomy.
- Invalid, timed-out, skipped, truncated, or failed evaluation returns a structured non-complete
  result and cannot appear as a clean non-match.
