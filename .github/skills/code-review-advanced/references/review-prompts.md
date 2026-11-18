# Advanced review prompts

Read the sections relevant to the code under review. A category becomes a finding only
with a reachable trigger, a violated contract, and impact.

## Boundaries and interacting conditions

- Where do missing, explicit null, false, zero, empty strings, empty collections, and invalid
  values take different paths? Can truthiness or a default silently merge distinct meanings?
- At each limit, reason about below, equal, and above; include negative indexes, inclusive
  versus exclusive bounds, non-finite numbers, overflow, truncation, and precision where the
  types permit them. Which combinations of two boundaries change control flow?
- Put a malformed record between valid records. Is rejection atomic or per item? Can mutation
  during iteration skip the next item, reorder results, or leave stale derived state?
- Across batching, filtering, splitting, and retries, account for every input and output. Can
  work, results, permits, or charges disappear or be counted twice?

## Failure atomicity and recovery

- Interrupt a multistep mutation after each visible or durable effect. What state remains,
  what result has already escaped, and what will a retry or restart do?
- Locate the commit point and any external effects outside it. Does rollback undo another
  actor's work? Can cleanup fail, mask the original error, or delete recovery information?
- Can recovery itself be interrupted and rerun? Are repeated close, abort, and delete operations
  safe under their documented contracts? Establish persistence guarantees before assuming a
  successful write survives a crash.

## Retries, idempotency, and budgets

- Assume a remote mutation succeeded but its reply was lost. Is another attempt safe? Check
  deduplication scope, concurrent duplicate requests, the atomicity of recording a key and
  performing the effect, retention expiry, late requests, and a reused key with different
  parameters.
- Trace queueing, connection setup, attempts, backoff, and cleanup against the logical operation's
  deadline. Which layers retry independently? Calculate the maximum combined attempt count.
  Can permanent errors, overload, synchronized retries, or reset deadlines multiply work?
- Does a timeout mean the caller stopped waiting, work stopped, or the outcome is unknown?
  Inspect the actual cancellation behavior before claiming an effect was prevented.

## Data contracts and representation

- Trace raw bytes through decoding, parsing, normalization, validation, caching, and output.
  Do duplicate keys, Unicode, number precision, heterogeneous arrays, or alternate encodings
  cause two components to interpret the same input differently? Use a supported input to
  demonstrate the disagreement. Distinguish JSON syntax from application schema.
- Does validation occur before every consumer that needs it, including cached and deserialized
  values? Can a valid outer container hide malformed elements or unexpected nested types?
- Compare single/batch, sync/async, cached/uncached, CLI/SDK/plugin, and source/prepared paths
  only where they promise the same contract. Does an optimization change meaning, ordering,
  diagnostic attribution, or result status?

## Identity, cache invalidation, and resource ownership

- List everything that changes a result: input, configuration, model or schema version,
  parameters, identity, authorization, and mutable dependencies. Which belong in its cache key,
  namespace, or invalidation rule under the intended reuse contract?
- After invalidation, eviction, clear, reconfiguration, or delete-and-recreate, can an older
  callback publish a stale value or remove a replacement? Check generation and object identity.
- Who owns tasks, buffers, handles, locks, and permits after each handoff? Can one waiter cancel
  shared work still needed by another? Can an accepted result change through a mutable alias?
- Check both retention and release: unbounded keys, bytes, queued work, and background tasks;
  early release while still in use; double release; and counter drift after failure.

## Security boundaries and result semantics

- Follow authorization and validation from the checked object to the object used for the
  effect. Can another actor replace a path, resource, credential context, or policy between
  check and use? Identify the mutator and ordering.
- Trace invalid, unavailable, refused, timed out, cancelled, skipped, and partially evaluated
  outcomes to their consumers. Can an unknown result become success, an empty result, or a
  clean non-match? Show the consumer that makes the unsafe interpretation.
- Where untrusted content enters prompts, commands, paths, URLs, or logs, identify the trust
  boundary, the final sink, and any escaping or authorization already enforced there.

## Quarry starting points

Use these as navigation hints and verify current paths and contracts at the reviewed revision:

| Area | Start with | Questions |
| --- | --- | --- |
| Scan lifecycle | `quarry/scanner.py`, `quarry/engine/decision.py`, `quarry/safety/timer.py` | Do completion state and diagnostics survive deadlines, skipped work, and partial results? |
| Shared work and inference | `quarry/cache/async_cache.py`, `quarry/integrations/model_usage.py`, `quarry/integrations/model_inference.py` | Who owns pending work, cancellation, cache identity, and attempt budgets? |
| Language semantics | `quarry/dsl/`, `quarry/engine/`, `quarry/policy_serialization.py` | Do lowering, temporal bounds, missing values, and prepared policies preserve the documented meaning? |
| Public surfaces | `quarry/cli.py`, `quarry/plugin/`, `quarry/types/` | Do input validation, statuses, metadata, and errors agree across supported entry points? |

Infer additional targets from the diff. These hints describe areas to inspect, not known defects.
