# Registered candidate providers

Candidate span readers use explicitly registered local patterns or a host batch
callback. A provider declares a namespace, revision, batch and fixed channel set.
Quarry supplies no pattern catalog or implicit check registry.

```python
from quarry.integrations.candidate_checks import LocalCandidatePattern
from quarry.integrations.candidate_providers import LocalCandidateProvider
from quarry.types.candidates import CandidateBatchIdentity

provider = LocalCandidateProvider(
    CandidateBatchIdentity("synthetic-candidates", "1", "heads"),
    ("markers",),
    (LocalCandidatePattern("markers", r"SYNTHETIC_[0-9]+"),),
)

reader = {
    "adapter": "registered",
    "model": "synthetic-candidates",
    "revision": "1",
    "category": "SYNTHETIC",
    "spans": {
        "backend": "candidates",
        "batch": "heads",
        "channel": "markers",
        "settings_digest": provider.settings_digest,
    },
}
```

Register the instance under `registered` through `model_adapters` or a
`ModelInferenceBackend`'s `adapters` mapping. Configure a model alias with `reader`,
then read it with `model("reader").spans(messages, 0.5)`. Identity, revision,
batch, channel and effective settings must match the registered capability.
Candidate readers cannot use classifier heads or model windows. Saved policies
also require matching runtime registration. An attested native local candidate
predicate has a cost before service work; unknown capabilities retain their
ordinary service cost. Local matching creates no billable model calls.

`LocalCandidatePattern` accepts a caller-authored RE2 expression, channel, optional
score, ignore-case/dot-all flags and up to eight named checks. Supply corresponding
synchronous predicates in the provider's `checks` mapping. A predicate receives
the exact candidate substring and must return a native bool. Implementation
changes require a new revision. Declared patterns, check names and execution
limits contribute to the effective settings digest.

For a host batch, register `HostCandidateProvider(identity, channels,
declared_settings_digest, callback, ...)`. The callback receives a detached
`CandidateBatchRequest` and keyword `timeout`, and may return a reply directly or
await one. Configure its effective `provider.settings_digest`, which includes
declared channel checks and execution limits. The callback's own settings belong
in its declared digest; implementation changes also require a new revision.
`CandidateChannelChecks` associates named checks with host channels.

Requests contain separate text parts and source addresses. A `CandidateRange`
names one part and a nonempty character interval. A `CandidateCoverage` names an
inspected interval. A `CandidateChannelReply` declares ranges, coverage and
completeness. Return exactly the requested channels in a `CandidateBatchReply`
carrying the request digest. All channel geometry, candidate counts and work
bounds are checked before named checks, score thresholds or cache admission. A
partial channel leaves the whole read incomplete, including under negation.
Over-limit replies are refused without clipping.

Adjacent messages and content parts remain separate. Native linear copied text
keeps original coordinates. Unmapped, interpolated and enclosing origins cannot
supply exact candidate authority. This text capability refuses unsupported media
inputs. Tool arguments remain their original JSON field, with no joining of
decoded scalar values.

`CandidateBatchLimits` bounds input characters, parts, per-channel ranges,
coverage and validation work. `CandidateWorkLimits` bounds execution time, work,
candidate counts and worker capacity. Enumeration counts every candidate before
rejecting checks can hide excess work. Source preparation and worker execution
spend a selected local CPU-work allowance. Checks are trusted synchronous host
functions: cancellation bounds a waiter and requests cooperative termination.
A still-running function retains its slot until it actually returns. Close the
provider when the host has finished using its registration.

A shorter request timeout is owned by its initiating caller. Eligible waiters
can retry a group ended by that actual timeout under their own remaining budget.
The registered provider's work and candidate caps apply to every compatible
reader; reaching those caps remains a shared failure.

Compatible aliases using the same provider instance share one fully validated
group. Different admission limits stay separate. Each reader binds numeric ranges
to its own native sources and drops its bindings on completion, failure or
cancellation. Caller-held result spans continue to own their evidence source.

Optional logical-read reports project the selected channel's validated coverage
onto original native parts. A partial group has no authoritative complete input
cut, even when one channel inspected all of its input. Invalid replies do not
claim inspected coverage. Telemetry exhaustion reports an unavailable or bounded
measurement without changing verdicts or physical-call counts. Scope-owned group
entries close together on the last scan/deferred-work lease. A configured process
cache may keep its detached numeric reply for reuse without retaining native
input owners or provider callbacks.

Register local candidates separately in each process. A forked child cannot run
an inherited native worker; a fresh child registration owns its own capacity.
