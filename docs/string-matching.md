# Project Quarry: String Matching

Quarry supports literal, whole-word, Base64, and RE2 regular-expression matching on text-valued conversation fields. A declaration can hold one string or an array of strings:

```text
rule "Credential terms"
strings:
    $terms = ["api key", "password", "access token"] nocase
condition:
    any msg in user_messages where msg.content contains_any $terms
```

| Form | Use |
| --- | --- |
| `text contains "literal"` | Search text for a literal. |
| `text contains_any ["a", "b"]` | Search for any literal in the list. |
| `$pattern in text` | Search using a declaration and its modifiers. |
| `text ~ "pattern"` | Search with an inline case-insensitive RE2 pattern. |

Declarations accept `nocase`, `fullword`, `base64`, `regex`, and named `check("name")` modifiers. `nocase` ignores case but preserves source casing in captured text. `fullword` matches a literal only when no Unicode alphanumeric character or underscore touches either end, so `"--force" fullword` matches `git push --force` but not `x--forcey`, and `"café" fullword` does not match inside `caféñ`. Combining marks are not alphanumeric characters. `base64` encodes the declaration before matching. An explicit `regex` retains precedence over literal `fullword` handling and interprets the declaration as an RE2 pattern.

```text
strings:
    $word = "admin" nocase fullword
    $encoded = "payload" base64
    $email = "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+" regex
```

Backslashes work differently in the two kinds of declaration. Quarry decodes escapes in a literal, so `"line\nnext"` matches a line break. A `regex` declaration reaches RE2 as written, so write `"\d+"` for digits. A doubled backslash, as in `"\\d"`, matches a literal backslash followed by `d`, and the rule still loads.

Quarry uses RE2 to avoid backtracking blowups. RE2 does not support lookbehind or backreferences. Rules must use regex forms the parser, validator, and compiler accept. Invalid declared patterns fail while rules load. For inline or computed patterns, test the exact rule and input shape; unsupported expressions fail closed rather than switching to another regex engine.

Use `as` to capture a matched substring for a later condition:

```text
rule "Repeated marker"
strings:
    $marker = "TOKEN=[A-Za-z0-9]+" regex
condition:
    any msg in user_messages where
        ($marker in msg.content as found) and
        found in msg.content
```

The [generated language reference](../quarry/language/reference.md) has the current operator and helper signatures. The [rule-writing guide](rule-writing.md) covers how to combine text checks with event flow.

Trusted Python hosts can filter candidate matches with named predicates:

```python
from quarry import compile_policy, load_compiled_policy

source = """rule "Even marker"
strings:
    $marker = "[a-z]+" regex check("even_width")
condition:
    any msg in messages where $marker in msg.content
"""
checks = {"even_width": lambda text: len(text) % 2 == 0}
policy = compile_policy(source, pattern_checks=checks)
restored = load_compiled_policy(policy.to_bytes(), pattern_checks=checks)
```

`DSLScanner`, `QuarryGuard`, and the programmatic `QuarryPlugin` host accept the same `pattern_checks` keyword. Each backend copies the mapping and exposes it read-only. The default registry is empty; there are no built-in product checkers. Registry names contain 1–64 ASCII characters, start with a letter, and otherwise allow letters, digits, `_`, `-`, and `.`. A registry contains at most 64 names; a declaration can apply at most eight distinct checks, in written order. The first rejection stops that candidate's remaining checks.

Checks receive only the matched substring with its original case and source characters. They apply to presence, regex predicates, counts and capped counts, positions, captures, span collections, extraction windows, and pattern-based selections. They do not change the existing extraction fallback: when no spans are selected, extraction retains the original text. A check must return exactly `True` or `False`. Exceptions or other replies become a typed `PatternCheckError` with the bounded check name; callback messages, returned objects and matched text are omitted. Cancellation remains cancellation.

Host predicates are trusted synchronous code. Keep them deterministic, thread-safe, bounded, and free of I/O; Quarry does not sandbox arbitrary Python or preempt a blocking callback. Existing local-work measurement includes reached predicates, and evaluation deadlines are checked around each production callback. Rules and prepared data select names only: they never load Python modules, carry callables, or register capabilities. Every referenced name must be available at source or prepared load, even when its condition would not be reached. CLI and plugin hosts with no registry reject such rules explicitly.

The Unicode literal-boundary change requires `compiler-31` and prepared format 12. Earlier prepared policies must be recompiled from source, then loaded with their explicit host registry. This prevents previously compiled ASCII boundary decisions from silently acquiring new behavior. Saved policies contain check names, not predicate implementations; the host remains responsible for binding each name to the intended trusted implementation.
