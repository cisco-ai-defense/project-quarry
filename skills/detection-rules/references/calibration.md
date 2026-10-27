# Calibration to FPR

Calibrate each judge's scores to a fixed false-positive-rate scale with FPRCal
(<https://github.com/cisco-ai-defense/fpr-model-calibration>, arXiv 2607.05481). FPR depends only on
benign data, while positives are unknown unknowns, so a threshold on the calibrated scale keeps its
meaning when the judge model, a policy, or the traffic changes. On that scale 0.30 is 1% FPR, 0.50
is 0.1%, and 0.70 is 0.01%.

## Fit

- Score your deployment's benign traffic per surface with the judge in serving form.
- Fit `fprcal.fit_calibration_pipeline(benign_scores)` with default options, once per judge model
  entry. Positives never enter the fit.
- To evaluate, fit on dev benign items and measure on test.
- Size each fit with the planning rule in the data reference. Thresholds below the smallest
  observed tail FPR are extrapolation.

## Fit order

The prefilter decides which benign items reach the judge, and the calibrated threshold decides
which positives the prefilter must keep. Resolve the loop in this order:

1. **First fit**, before the prefilter exists: fit on all dev benign items of the surface. Its
   threshold defines the judge-positive positives the prefilter is designed for.
2. **Final fit**, once the prefilter and the policy are final: refit on the dev benign items the
   final prefilter passes. Recompute the judge-positive set at the refit threshold, and re-measure
   prefilter recall on it. If recall falls below the target, revise the prefilter and repeat.
3. Repeat the refit and the recall check after any change to a policy block or the judge model.

## Clean the fit

Verify the benign items flagged at the calibrated threshold with the labeler. Remove every item it
confirms as a real attack from the fitting population and from the benign denominators, refit, and
rebuild the artifact. Verify the new fit's flags the same way, until a pass confirms no new attacks.

## Ship

- Ship one artifact per judge model entry through `score.calibration`. The host application
  supplies the `calibrator_loader` that turns the reference into a callable.
- Bind the rule's threshold parameter to `0.50`, which is 0.1% FPR, and record it in
  `threshold_note` and the calibration JSON. A deployment can run another tier, such as alerting at
  `0.30`, by binding another value.
- Refit when the judge model, a policy block, or the traffic changes, in the order above. The
  threshold stays at `0.50`.

## Report

- The final fit describes judged items. The full rule's benign alert rate is 0.1% times the
  prefilter's pass-through share; report both.
