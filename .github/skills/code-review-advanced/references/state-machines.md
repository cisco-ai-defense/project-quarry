# Review implicit state machines

Support the request: **"find all implicit state machines in this codebase"**.
For that request, search the repository broadly and report the coverage boundary. For a PR
review, model machines affected by the change and follow their interacting lifecycles.

## Discover machines encoded in ordinary data

Search entry points and state mutation sites using repository navigation and text search.
Look beyond named enums:

- Combinations of booleans, nullable fields, counters, timestamps, status strings, and sentinels.
- Initialization, acquisition, handoff, completion, cancellation, cleanup, close, and reuse.
- Tasks and futures, queue membership, cache admission and eviction, retries and circuit state.
- Persisted records, checkpoints, leases, generations, and schema or protocol transitions.

Read writers, readers, callbacks, and cleanup paths. A field name is a lead, not proof of a
machine. Group fields by the lifecycle or protocol they jointly govern. Distinguish explicit
automata already represented in the code from the implicit lifecycle surrounding them.

## Reconstruct observed behavior and intended properties

For each discovered machine, record its owner, purpose, encoding, initial state, and code
locations. Build a transition table with:

| From | Event or operation | Guard | State updates and visible effects | To | Code evidence |
| --- | --- | --- | --- | --- | --- |

Add concrete rows from the reviewed code. Mark inferred states and uncertain transitions.
Separate what the implementation allows from what documentation, callers, and tests require.
Identify terminal states, re-entry and reset rules, and who may initiate each transition.

State the properties before looking for counterexamples:

- **Safety:** what must always hold, including legal field combinations, one owner per resource,
  result immutability, and terminal-state behavior where the contract requires them.
- **Progress:** what must eventually happen, such as releasing a waiter or completing shutdown.
  State assumptions about dependency responses, scheduling, timeouts, and fairness.

## Try to break the model

- Derive combinations from the state encoding. Which are legal, impossible by construction,
  or reachable only through errors? Check guards on every route to a transition.
- Trace failure or cancellation between updates that together represent one logical change.
  Can an observer see an invalid intermediate combination, or can it survive an exception?
- Check repeated, reordered, stale, and late events: finish twice, cancel before start, close
  during initialization, reuse after close, or completion from an earlier generation.
- Follow interactions between machines, such as a scan finishing while shared work continues.
  Which actor still owns progress, cleanup, publication, or recovery?
- Check cycles without progress, states with no reachable exit, and success reported before
  required transitions. Explain why any claimed stall persists under the stated assumptions.

For interleavings, consult [concurrency.md](concurrency.md). For interrupted writes and
retries, read the relevant sections of [review-prompts.md](review-prompts.md).

## Deliver an inventory and evidenced findings

Provide an inventory of discovered machines with their encoding, owner, code locations, and
review status. Expand transition tables for machines analyzed in depth; identify inventoried
machines whose transitions remain unreviewed. Include expected properties and a minimal
reachable counterexample for each defect, with exact locations and user impact. Compare base
and head before labeling PR regressions; keep speculative concerns separate.

List directories, entry points, and mutation patterns searched, plus excluded or inaccessible
areas. Say which properties were reasoned about or executed. A bounded search cannot prove
that every implicit machine was found; an inventory or diagram alone does not prove correctness.
