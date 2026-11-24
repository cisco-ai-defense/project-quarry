# Asynchronous scans and host callbacks

Awaited scanner and compiled-policy scans evaluate on one lazy engine worker per process.
This includes single scans, batches, and policies restored from compiled bytes. Synchronous
scan methods evaluate on their calling thread and retain their existing `asyncio.run`
contract. Use an asynchronous scan method from an application that already has a running
event loop.

The engine copies the submitting task's context variables. Model adapter factories and
their awaitables run on the submitting application's event loop and thread. The same host
ownership applies to custom model backends, registered transformations, window renderers
and decoders, score builders and calibration, declared-input conversion, semantic probers,
and custom rate limiters. Each callback retains its declared sync or async return contract.
A callback can await another
scan; nested engine work shares the managed worker without blocking its host loop.

Cache hits and shared waiters retain the configured request-reuse behavior. An actual
shared request uses its initiating host. Reader validation, model budgets, result retention,
and logical-input accounting remain attached to their existing scan owners.
Delegating custom backends preserve logical-read attribution and transform input provenance.

Each engine pacer yields every 64 steps independently. Ready host requests are handed off
before the next queued local continuation, allowing both remote requests and local work to
make progress. The application's event loop schedules a callback after its handoff.

Inspection deadlines begin before worker queueing. Queueing, rate-limit waits, retries,
and provider work consume their applicable budgets. Cancellation waits for the owned
remote task's cleanup, including repeated cancellation while that cleanup is running.
Extension code still needs to yield cooperatively and to finish its cancellation cleanup.
Callbacks that perform blocking work can block their own application loop.

Worker generations drain before replacement. A forked child starts a fresh worker and
does not acquire inherited worker locks. Ordinary transferred exceptions retain their
identity, cause, and notes. Nonstandard `BaseException` exits are translated into bounded
runtime failures at the loop boundary.

Completed tool-quantifier evaluations and recursive token splitting release their local
callbacks after their owned work joins, including failure and cancellation paths. Message
collections preserve canonical root identity without storing strong references to themselves;
a live derived view still owns its source until the view is released. Shared cache failures
keep the same exception available to every waiter, while finished wait frames release their
references to the futures carrying that failure. These changes do not alter verdicts, request
counts, cache acceptance or prepared-policy versions.

This is not a universal reference-count-only lifetime guarantee. Native tool calls and their
responses retain their bidirectional links, and a host retaining an exception may retain its
traceback. Such graphs can require Python's cyclic collector after their external owners are
released. Completed scan contexts, plain message roots and joined split callbacks do not need
those cycles to remain alive.
