# Measurement, baseline, report, and review

Rates are per item, every rate names its denominator, and every rate carries the uncertainty its
sample supports (Wilson 95% intervals). Positives and benign items are measured separately. Do not
combine them into a precision figure that depends on how many positives were collected. Failed and
refused rows stay in the denominators with their outcome.

## Definitions

- **Recall:** the share of labeler-confirmed test positives the rule flags. Items the labeler did
  not confirm are excluded, not counted as misses.
- **Judge-alone recall:** the share of the same labeler-confirmed test positives the judge flags at
  the operating threshold, with no prefilter. The two recalls share one denominator, so they compare
  directly.
- **Full rule:** prefilter pass and a judge score above the operating threshold.
- **Benign FPR:** the share of held-out benign items flagged at the operating threshold. Report it
  for judged items and for the full rule, which is the judged-item rate times the prefilter's
  pass-through share.
- **Confirmed FPR:** re-judge every benign item flagged at the lowest threshold you report, using
  the labeler against the constitution. One pass covers every higher threshold, because a higher
  threshold flags a subset. Count flags the labeler rejects, plus flags it could not judge, as false
  positives over the benign items, less the items the labeler confirms as real attacks. Report
  flags, flags that are real, and confirmed FPR. Strictly, this is a false-positive rate over the
  traffic pool. Unflagged items were never verified, so the rate understates true FPR when the pool
  holds real attacks. Say so in the caveats.

## Operating threshold

- The operating threshold is the value bound to the rule's threshold parameter in step 8.
- Headline every profile with recall at 0.1% FPR, and report that FPR with the confidence its
  benign sample supports. Add the neighbouring tiers (1% and 0.01% FPR) to show the curve.
- Check score granularity. When many benign scores tie near 1, no threshold separates them, so
  confirm the score has resolution in the tail you target.
- Full-rule recall is at most the lower of prefilter recall and judge recall. Say which one binds.

## Deterministic baseline

Run the existing deterministic signatures for the same threat on the same test items with the
engine. Signatures have a single operating point. Compare them at their measured benign FPR with
the rule's recall at that same FPR, and report both next to the rule at its operating threshold.

## Report per profile

Build the report with a script from the artifacts; type no number by hand. Its sections:

1. **Positive data.** Candidates, confirmed positives per labeler, and label outcomes (refusals,
   parse errors). Add a table per source: license, label type, candidates, confirmed positives.
2. **Judge alone.** At 0.1% FPR and the neighbouring tiers: recall with its interval, benign
   FPR, flags, flags that are real, and confirmed FPR.
3. **Prefilter, on test.** Per surface: benign items, pass-through, positives, prefilter recall,
   and recall on judge-positives.
4. **Full rule.** The same columns as section 2.
5. **Baseline.** The deterministic signatures and this rule at matched FPR, plus the fallback
   labeler's slice.
6. **Caveats.**
   - how the split was made, and each use of the accept split;
   - provenance by label type, with the generated share;
   - the fallback-labeler slice;
   - how confirmed FPR was computed;
   - the binding limit;
   - that rates are per item;
   - score granularity in the targeted tail;
   - known weaknesses, such as an overfit prefilter.

Add a summary table across profiles: profile, category, full-rule recall at 0.1% FPR, the
full-rule benign alert rate, confirmed FPR, judge-alone recall, and baseline recall at matched
FPR. The rule's `threshold_note` comes from the same run as the report.

## Adversarial review

Give a separate reviewer the artifact paths and nothing else. It recomputes every number in every
report from the raw labels, scores, verdicts, and hits, then checks the splits, the denominators,
and the engine-against-evaluator result. It reports every mismatch. Fix, rebuild, and repeat until
it finds none. Then state in the summary that the review found none.
