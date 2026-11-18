---
name: code-review-concurrency
description: Review race conditions, shared work, async cancellation, reentrancy, deadlocks, and stale publication using concrete execution schedules. Use for concurrency-focused code review.
---

# Review concurrency and races

Read repository instructions and establish the revision, review scope, supported runtimes,
and actual concurrency model. For a PR, trace changed behavior through all access paths and
compare base and head before labeling a defect a regression.

## Map actors, shared state, and ordering

Identify tasks, threads, processes, callbacks, signal handlers, and remote actors that actually
exist in the supported deployment. For each shared resource, record:

| Resource | Readers and writers | Owner and lifetime | Synchronization or version guard | Required invariant |
| --- | --- | --- | --- | --- |

Include mutable aliases, context-inherited objects, queues, caches, files, external records,
and counters. Mark suspension points, blocking operations, callback invocations, lock boundaries,
publication, and cleanup. Check the protection of every reader and writer, including alternate
entry points and failure paths. Establish the operation's intended atomic effect and visible
commit point; use documented weaker guarantees when full atomicity is not promised.

Do not infer safety from the GIL, an async function, a thread-safe container, or the presence
of a lock. Equally, do not invent a task switch between synchronous statements on one event
loop without an actual yield, reentrant callback, or another supported actor.

## Construct adversarial schedules

Choose cases grounded in the mapped resources:

- **Check then act:** change a value or resource identity between validation and use. Does the
  guarded action still refer to the version or object that was checked?
- **Compound updates:** interleave read/modify/write, initialize-if-absent, admission and
  accounting, or completion and removal. Can a limit be exceeded or a replacement be removed?
- **Shared work:** cancel or time out the creator and one waiter independently. Does remaining
  work have an owner? Are all waiters resolved on success, error, cancellation, and eviction?
- **Stale publication:** pause an old operation, invalidate or reconfigure, start a new operation,
  then complete the old one. Can it overwrite new state? Check delete-and-recreate cycles.
- **Reentrancy:** invoke user code, destructors, or callbacks while an invariant is temporarily
  broken or a lock is held. Can they re-enter the same object or acquire locks in another order?
- **Deadlock and starvation:** construct a wait-for graph including locks, futures, queues,
  executors, and shutdown joins. Distinguish a cycle from a slow dependency, and state the
  scheduling or fairness assumptions behind a starvation claim.
- **Shutdown and isolation:** overlap close, clear, retry, or restart with in-flight work.
  Inspect inherited state after fork only where forking is supported. Check process and
  distributed protection separately from thread and event-loop protection.

Use a schedule such as `A reads generation 1; B replaces it with generation 2; A publishes`.
For every step, identify the code site and why another actor can run at that point. Explain
the violated invariant and visible effect; two accesses to one object alone do not prove a race.

## Trace cancellation and ownership

At each relevant await or handoff, determine who cancels, who cleans up, and who waits for
cleanup. Inspect suppressed cancellation, shielded work, executor jobs, child tasks, and remote
effects. A caller timeout does not establish that work stopped. Verify exact API behavior for
the supported runtime; use primary documentation when source is insufficient.

Check that cancellation releases permits and locks exactly as required, preserves other
waiters' ownership, and cannot report a clean result while an effect is still ambiguous.
Use [review-prompts.md](../code-review-advanced/references/review-prompts.md) for retry budgets,
failure atomicity, and the primary sources underlying these prompts.

## Validate and report

When execution is authorized and helpful, use barriers, events, controllable futures, or
injected failures to force the proposed ordering. Avoid timing-only sleeps as evidence of a
race. Account for every spawned task and thread in the reproduction itself.

Report a minimal actor-by-actor schedule, exact locations, synchronization inspected, violated
contract, and user impact. Distinguish demonstrated deadlocks, races, and stalls from hypotheses
requiring missing deployment facts. State whether the schedule was executed or only traced,
which actors and paths were reviewed, and remaining limits. Do not edit or publish unless
requested. Follow `SECURITY.md`; leave merging to the user.
