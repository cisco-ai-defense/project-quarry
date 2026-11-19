# Formal rule verification

Quarry has an optional, bounded formal check for rules that cannot match because their
message-content predicates conflict. Install the solver extra when you want this check:

```bash
uv sync --frozen --extra verification
```

The check runs during CLI validation:

```bash
uv run --frozen quarry scan \
  --config examples/config.yaml \
  --target examples/data/ \
  --validate-only
```

A proved `NO` adds a high-severity advisory, so `--validate-only` returns nonzero. Normal
compilation still succeeds, and scan verdicts remain unchanged. If the solver is absent, a
construct is unsupported, a limit is reached, a feasible path exists, or analysis fails,
the answer is `UNKNOWN` and no advisory is added.

This initial slice supports bounded DFA searches over nonempty conversations of
string-content messages. Message content may be empty. It models case-sensitive `content`
equality or substring predicates on `messages`, `user_messages`, and
`assistant_messages`. Captures, collection output, parameters, decision-tree decomposition,
tool quantifiers, regular expressions, string modifiers, and positional constraints
currently return `UNKNOWN`.

Literal backslashes, control characters and Unicode text retain their runtime meaning;
text such as `\u{41}` is not interpreted as a solver escape.

The Python API exposes the same fail-safe result:

```python
from quarry.safety.quality import verify_matchability

result = verify_matchability(compiled_artifact)
print(result.answer, result.basis, result.reason)
```

This slice only proves non-matchability. A feasible model path remains `UNKNOWN` until a
future slice can construct, validate, and replay a concrete witness.
