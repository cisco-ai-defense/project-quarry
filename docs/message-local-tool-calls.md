# Message-local tool calls

An assistant message exposes its own `tool_calls`. Filters, counts, indexes, slices
and nested `any`/`all` keep those calls attached to the original message and results:

```quarry
rule "A selected local call"
condition:
    any m in messages where
        count([c for c in m.tool_calls if c.name == "lookup"]) == 1
```

`assistant_messages[-1].tool_calls[0].arguments.limit == 2` reads a decoded argument.
`c.response.result` and `c.response.content` read the linked result; pattern spans
over these fields retain the result's original text coordinates. Inner predicates
can read their enclosing message. Each comprehension binds its name only in its
predicate, so reusing a name does not capture the source or a sibling predicate.

Readable non-assistant messages have no call collection. Direct `len()`/`count()`
on an absent collection stays missing, while an identity filter has zero survivors.
Cardinality does not read call fields. An opaque call can be counted; reading its
unavailable fields fails only the rule that reads them. Existing input and link
validation still applies.

Filters can call configured models under the ordinary per-scan physical request
allowance, deadline, pacing and cache policies. Cheap membership tests run before
the body reads selected fields. An existential can reject a cheap false body before
paying for membership; a filtered universal can accept a cheap true body without
that request and stops at its first selected counterexample. Model calls inside an
unfiltered `all ... in tool_calls` or `all ... in m.tool_calls` remain unsupported.

Explanations distinguish the aggregate condition from its judges. A message-owned
count or quantifier reports a Boolean outcome without a child score, threshold or
input span. An indexed count comparison keeps its numeric value and comparison.
Refusals and bounded public judge reasons remain attributed. A direct judge,
including one consuming a pure mapped input, retains its own score and provenance;
global temporal counts retain their child evidence. Disabled explanations add no
provider work. Existing called, computed and declared-input frames retain their
ownership and field-isolation rules.

Prepared policies bind the exact compiler and envelope versions used to build them.
When upgrading either version, recompile existing policies from their saved rule
sources.
