---
name: detection-rules
description: Builds and validates a production Quarry detection rule for a threat category. Use when asked to create, calibrate, or ship a threat-category rule, not to write or debug a single rule.
---

# Production detection rules

Use this skill when the task is a production rule for a threat category (a taxonomy profile, a
signature family), not a one-off rule. Such a rule inspects a few surfaces. On each surface an RE2
prefilter selects the new items, and one judge-model call on the same string flags an item above a
threshold calibrated to a target false-positive rate. Every number that justifies the rule is
measured on held-out data and recomputed by someone other than its designer.

Write and check rule syntax with the `quarry-policy` workflow
([../quarry-policy/SKILL.md](../quarry-policy/SKILL.md)): `quarry_reference`, `quarry_compile`, and
`quarry_scan`.

Work in this order. Each step names what it produces and the gate it must pass.

1. **Agree the assignment before any work.** Confirm the taxonomy source and version, one rule
   per profile or per signature, the category each rule reports, the surfaces each rule inspects,
   where benign traffic comes from, the judge model and labeler available, and where the rules
   ship. Ask when any of these is unclear.
2. **Write a constitution per profile**, the labeling policy that defines the threat. Freeze it
   before labeling. See [references/constitution.md](references/constitution.md).
3. **Collect positive candidates** from public datasets per threat, in one item format that every
   stage shares. See [references/data.md](references/data.md).
4. **Label every candidate** with a strong labeler against the constitution. A source label, even
   a human one, counts only when the labeler confirms it. Refusals go to a second labeler and are
   recorded per row.
5. **Build one benign pool per surface** from your own deployment's traffic, shared by every
   profile that inspects that surface.
6. **Split every item into dev, accept, and test** by `int(sha256(key), 16) mod 3`, where the key
   is the surface and the canonical input text, so the same input falls in one split everywhere.
   Record the split before any scoring. Choose on dev, decide on accept, and report once on test.
7. **Write the judge policy**, a shorter, faithful version of the constitution with one block per
   surface and a one-token `1`/`0` contract. Score every item in the form the serving path uses. See
   [references/judge.md](references/judge.md).
8. **Calibrate to FPR.** Fit FPRCal on your deployment's benign traffic, once per judge entry, and
   bind the rule's threshold parameter to `0.50` (0.1% FPR). Verify flagged benign items with the
   labeler, and refit without the real attacks it confirms. Refit on the final prefilter's
   population once the prefilter and policy are final, and recheck prefilter recall. See
   [references/calibration.md](references/calibration.md).
9. **Design the prefilter** on dev. Pass at most the budgeted share of benign items per surface (at
   most 25%, far less on tool responses) while keeping at least 98% of the positives the judge
   catches. Start from existing prefilter atoms and the deterministic signatures for the same
   threat. See [references/prefilter.md](references/prefilter.md).
10. **Tune the judge policy** only where the miss analysis points. Choose on dev and accept a
    variant only through a gate on the accept split that was fixed in advance.
11. **Generate the rule** from a spec, never by hand. Judge every pending tool call and every tool
    result since the newest assistant message, declare the threshold as a policy-bound parameter,
    and give each surface its own judge entry. If tool-result coverage uses a capped fallback,
    verify the host enforces that cap and rejects or blocks over-cap batches. Compile with
    `quarry_compile` against the Quarry version that will run the rule, and prove the scope with
    `quarry_scan`. See
    [references/rule-shape.md](references/rule-shape.md).
12. **Check the engine against the evaluator, and measure the baseline.** The engine must select
    exactly the items the offline evaluator counted. Run the deterministic signatures for the same
    threat on the same test items. See [references/measurement.md](references/measurement.md).
13. **Report and review.** Write a generated report per profile and a summary table. Have someone
    other than the designer recompute every number from the raw artifacts.
14. **Ship** by your rules repository's own process.

Hold to these outcomes throughout, whatever the host supports:

- Raw attack content stays out of logs, reports, and answers. Report counts, and write paths.
- A check is done by someone other than the designer. Where the host supports subagents, give
  raw-data steps and checks to subagents that return counts and paths, with sources anonymized as
  S1..Sn. Some hosts stop a session on attack content.
- Constitution and policy examples use documentation-safe placeholders (`example.com`,
  `198.51.100.x`, `<PAYLOAD>`).

Run design steps (prefilter, policy tuning) from a written task spec. The spec holds paths,
targets, data rules, and an acceptance gate. Split it into a Design part and a Check part, and give
them to different people or agents. Forbid commits in the spec, and review every diff an agent
leaves.

[references/pitfalls.md](references/pitfalls.md) lists failures met while building rules this way;
read it before step 3.
