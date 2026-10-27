# Judge policy, scoring, and tuning

## The policy

- Make it a faithful, shorter version of the constitution: the same scope, the same exclusions, at
  most about 700 words. The constitution's `context_dependent` cases map to 0.
- Give each surface its own block when a rule inspects more than one surface. Ship each block as
  its own text resource, so the judge never has to guess whether it sees a call or a response.
- Require exactly one character, `1` or `0`, as the first token, with no reasoning.
- Include a short security protocol: the content is data to classify, and instructions inside it
  are not followed.
- Keep each block between prompt markers. Every consumer (labeler, scorer, generator) extracts every
  block in order and maps each block to its surface explicitly.

## Scoring

- Score = P("1") / (P("1") + P("0")) at the first generated token, read from the top 20 logprobs,
  at temperature 0 with one output token. When neither token is present, record `null` and count it
  as a failure.
- Score in the message form the serving path uses. Quarry builds judge input with the model entry's
  message builder: `constitution_pair` in `quarry/integrations/model_inference.py` puts the policy in
  constitution tags after a preamble, and the unit in conversation tags. A simpler offline form
  (policy as the system message, unit in content tags) is cheaper, but scores fitted or tuned on it
  may not transfer. Re-check through the builder or on the serving path before shipping.
- Truncate the way the serving transport does. It cuts the user turn to a token budget, and
  whichever end it keeps is the end the judge sees. Retry an input the server rejects as too long on
  shorter windows cut from that end, and record items that still fail as `null`.
- Write a failed call as `null`, and keep it in the denominators.

## Tuning

Tune only when recall on a unit is low, and never change the constitution, the labels, or the
benign pools while tuning.

1. Measure the current policy on dev per surface and sub-label. Analyze the misses
   programmatically, without printing text: by source, label type, length, character classes, and
   the constitution section they fall under.
2. Write about three variants. Change only what the analysis points to: a clearer statement for
   the weak unit, one or two examples with safe placeholders, or removal of wording that pushes the
   judge to 0.
3. Score each variant, and the current policy, on all dev positives and all dev benign items of the
   profile's pools.
4. Calibrate each variant on the same dev benign items, and choose the one with the highest dev
   recall at 0.1% FPR. If none beats the current policy, keep it.
5. Decide on the accept split: accept the choice only if recall on the target unit at 0.1% FPR
   rises by at least 3 points. Keep the previous policy as `policy.v1.md`, and record the
   decision with its numbers.

Separate blocks per surface are often the fix for a profile that spans surfaces. When a shared
rewrite fails its gate, splitting the policy per surface can lift the weaker surface's recall
substantially.
