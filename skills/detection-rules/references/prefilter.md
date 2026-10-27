# Prefilter

The prefilter decides which items reach the judge. It exists for cost: a branch whose prefilter does
not match makes no model call. Its recall caps the rule's recall, so measure it against the
positives the judge would catch.

## Goal

Maximize recall on judge-positive dev positives (those the judge scores above the operating
threshold) while passing at most the budgeted share of benign items per surface. A positive the
judge rejects gains nothing from passing the prefilter.

- Budget: at most 20 to 25% of benign items on cheap surfaces. Tool responses are several times
  larger than tool calls and cost more per judge call: about 3% or below. Tool definitions are
  resent with every request, so keep them low as well.
- Target: at least 98% of judge-positive dev positives.
- Use the budget. A prefilter far under its cap loses recall for nothing.

## Constraints

- Patterns compile in RE2: no lookaround, no backreferences, no unescaped double quote. Match
  case-insensitively. Write `\uXXXX` escapes as `\x{XXXX}`.
- Keep the prefilter within what the target Quarry version compiles. Check with `quarry_compile`
  rather than relying on a fixed group count.

## Atom sources

1. Prefilter atoms that already exist for related categories.
2. The deterministic signatures for the same threat (their `strings:` blocks). Reusing their atoms
   keeps the two efforts aligned.
3. Atoms mined from judge-positive dev positives that no current atom covers: program names,
   flags, file names, path prefixes, key prefixes, API hosts, and control-character classes.

## Gates, checked by someone other than the designer

- **Overfitting.** Prefer general patterns that describe the behavior over strings copied from one
  dataset. A drop of more than 3 points from dev to accept in judge-positive recall is overfitting.
- **Source coverage.** Measure per-source recall on accept, with sources anonymized. A source far
  below the rest means the patterns fit the other sources, not the threat. Justify any pattern that
  matches only one source by its held-out recall.
- **Integrity.** Every pattern compiles, no group is empty or duplicated, and the generated rule
  compiles and scans a benign conversation per surface without evaluation errors.
- **Revisions.** A revision passes a gate written before the run, for example: accept
  judge-positive recall at least the previous value, and a dev-to-accept drop of at most 3 points.
  Keep the previous prefilter as the fallback. When the fallback stands, disclose its weakness in
  the report.
- Report the final prefilter once on test. Record each iteration's numbers and atom counts, never
  pattern text next to item text.

## One method that met these gates

A script that rebuilds the prefilter from its inputs:

1. Pool the candidate atoms from the three sources.
2. Add the atom with the best ratio of newly covered judge-positive dev positives to (newly passed
   benign dev items + 1), while pass-through stays under the budget. Repeat until the target, the
   budget, or no gain.
3. Drop atoms whose removal loses no covered positive.
4. Pack the survivors into alternations without changing the match set.
