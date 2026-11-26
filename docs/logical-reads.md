# Logical model-input reads

Set `logical_reads=True` on `DSLScanner.scan_data`, `scan_data_async`, `scan_file`,
`scan_batch_data`, or `scan_directory`, or on `CompiledPolicy.scan`, `scan_async`,
`scan_many`, or `scan_many_async`. Returned `ScanResult` objects carry a scalar
`logical_reads` snapshot. The default is `False` and the field is `None`.
Existing CLI and plugin responses omit the field unless requested.

```python
result = await policy.scan_async(conversation, logical_reads=True)
result.raise_for_status()
report = result.logical_reads
characters = report["aggregates"]["totals"]["coverage"]["characters"]
print(characters["value"], characters["known_units"], characters["complete"])
```

The report records input that a model operation logically reads. A cached reply can
serve several rules or conversations while each reader still receives attribution.
`observations` counts distinct reader/evaluation bindings; `evaluations` counts the
distinct attributed evaluations within a conversation. Neither is a provider-call
or billing count. Existing physical usage and model-call budgets retain their meaning.

## Coverage and attribution

`rows` separates scope, requesting rule, phase (`decision`, `evidence`, or
`measurement`), model alias, operation and outcome. Nested rule calls and memoized
values retain the requesting entry rule. Replaying a value's read metadata performs
no extra provider work.

`aggregates` provides `by_reader`, `by_model`, `by_scope` and `totals`. A reader group
is one conversation/rule/phase/model combination. Each group unions actual input
ranges across operations and outcomes. Adding detail-row totals can count overlapping
input more than once; use the aggregate for combined coverage. Repeated or overlapping
cuts share canonical source coordinates. Separate equal-text occurrences remain
separate inputs. Distinct conversation scopes add independently, including repeated
batch conversations served from one cached reply. Cut counts use opaque input-cut
identities and are reported separately from interval coverage.

Characters use Unicode character positions, not bytes. Selected and located input
windows retain their original coordinates; disjoint windows count only their actual
ranges. Semantic body, policy, tool schema, refusal, function-name and opaque argument
text is eligible when an operation reads it. Structural message layout and synthetic
model-builder boundary text do not enter text coverage. This is text coverage, not an
image/audio size estimate. No raw input, prompt, provider reply or opaque identity is
included in the report.

## Reference text tokens

Without explicit reference maps, native token coverage is unavailable. A requested
`input_tokens` operation still returns its physical request count; that count is not
substituted for logical text coverage. Collection never calls a tokenizer, provider,
host callback or post-decision auxiliary operation.

A host can supply already computed token spans for the canonical full text:

```python
from quarry import TextTokenBasis, model_read_usage

with model_read_usage(
    token_bases={"judge": TextTokenBasis("host-vocabulary", "revision-1")}
) as reads:
    for text, offsets in host_token_maps:
        accepted = reads.provide_text_tokens("judge", text, offsets)
        if not accepted:
            handle_unavailable_reference_map()
    result = await policy.scan_async(conversation, logical_reads=True)

final_request_report = reads.report()
```

Each `offsets` value is a native tuple of `(start, end)` integer tuples in Unicode
character coordinates over the entire supplied `text`. Spans must have positive width,
remain within the text and have nondecreasing starts and ends. Gaps and overlapping
spans are allowed, including several tokens per character. An empty text uses `()`.
Zero-width BOS/EOS/template tokens are excluded. A token counts if its span intersects
a nonempty selected source range; partial reads are never estimated by a character ratio.

The host is responsible for genuine tokenizer output and for versioning vocabulary,
normalization and offset conventions together. Quarry validates metadata shape, bounds,
source binding and map consistency; it cannot verify the declared vocabulary. Register
the actual canonical text, rather than a formatted full request or an unrelated equal-size
string. Maps can arrive before or after a read while the session is live. A later valid
map refines subsequent reports for memoized readers without repeating evaluation.

Aliases with the same basis name and version share maps. Conflicting maps invalidate
earlier and later measurements for that text/basis. Different bases have separate
`token_bases` breakdowns. An aggregate spanning incompatible bases has
`basis_compatible=False`, `value=None`, and `known_units=None`; summing those vocabularies
would not yield a meaningful scalar. Low-level manually supplied token intervals use an
explicit model-specific `supplied_intervals` label.

Registration accepts `state="failed"`, `"cancelled"` or `"unavailable"` when an external
measurement did not succeed. A failed or missing count stays unavailable, including for
an empty source. Malformed or conflicting registration fails closed. Token-atlas
contention can make metadata unavailable without blocking native input-owner cleanup.

## Completion, limits and ownership

`value` is numeric only when that unit's coverage is complete; otherwise it is `None`.
`known_units` retains a validated lower bound when one coordinate basis permits it.
`measurement_states` distinguishes measured, partial, pending, unavailable, failed,
cancelled, abandoned and limit-exceeded measurements. Operation outcomes separately
distinguish succeeded, refused, failed, cancelled, not-sent, pending and unknown.
A failed request can still have known input coverage. A successful request can have an
unavailable token measurement. A requested-but-unsent operation is not inferred from
an absent row. A genuinely observed empty input can measure zero; missing scope admission
and unrun policy batch items return `state="scope_unavailable"` with unknown coverage.

Native preparation work shares its scan's conversation scope before individual rule
attribution is available. If custom scanner work invokes or replays native reads outside
the registered scan scopes, the result view reports an unavailable scope rather than a
complete zero. Standalone session reports still retain those observations. This detection
uses bounded scope metadata and adds no provider work.

`finalized` describes collection lifetime, not measurement validity or verdict authority.
Owned public calls finalize after their native evidence and value owners close. Calls
inside a caller-owned `model_read_usage` session borrow that session: their result
snapshots say `finalized=False` and remain snapshots after the outer context exits.
Use `reads.report()` or `reads.scope_report(result.logical_reads["scope"])` for a newer
snapshot. Collection does not wait for additional work, extend a deadline, create a task
or change evidence budgets. Cancellation keeps its existing exception and cleanup behavior.

`ScanResult.is_complete`, `failures` and `raise_for_status()` retain their existing
verdict semantics. Optional accounting incompleteness does not change a verdict or
turn an incomplete scan into a clean result.

`ReadUsageLimits` bounds events, reader bindings, scopes, captures, input identities,
interval storage and traversal work. Reference maps have independent token axis/map/span
and work caps. Aggregation has independent defaults of 2,048 groups, 8,192 coverage keys,
32,768 ranges and 2,000,000 work units. Exceeded limits remain explicit; they do not
create a complete zero or trigger a fallback provider call. Result/scope associations
are weak and bounded by the session's binding limit. File-batch reports combine only
their own conversation scopes, even inside an existing session.

## CLI, MCP and schema

```shell
quarry scan --config config.yaml --target conversation.json --output-format json --logical-reads
```

The flag requires JSON output. Each file result includes `logical_reads`; a batch file
combines its conversation scopes. `quarry_scan` accepts the Boolean `logical_reads`
argument. For an MCP batch, each `results` item has its own report. CLI and standalone
MCP calls have no host token maps, so native reference-token coverage stays unavailable.
Plugin compile/parameter rejection before scanning retains its existing diagnostic-only
response without inventing a read report.

`quarry.logical_read_report_schema()` returns a fresh JSON Schema 2020-12 document for
version-1 full reports, including opt-in result snapshots and `ReadUsageSession.report()`.
The lower-level `ReadUsageSession.snapshot()` retains its detail-only contract.
