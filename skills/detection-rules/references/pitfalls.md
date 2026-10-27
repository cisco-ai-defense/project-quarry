# Common pitfalls

Each entry gives the failure and then the fix.

**Only the first policy block is used.** A multi-surface policy file read as one block runs every
surface's judge with the first surface's policy, and recall on the other surfaces collapses while
the total can still look healthy. Map each block to its surface explicitly, and measure recall per
surface before trusting any total.

**The labeler's output is cut off.** A tight output-token limit truncates replies that reason before
the JSON, so they do not parse. Allow a generous limit (about 4,000 tokens), record the stop reason,
and keep parse errors and refusals as their own outcomes.

**Trusting source labels.** Public datasets mislabel and over-label. A positive is an item the
labeler confirmed against the constitution. Record the source's label only to compare.

**The fallback labeler is stricter.** Refusal rows relabeled by a second model come out stricter.
Measure agreement first, and report that slice separately.

**Items are over-long.** Very long items, and dense invisible-character text, exceed the served
context and return HTTP 400. Retry on shorter windows cut from the end the serving path keeps.

**Wrappers leak signal.** A wrapper that differs between positive and benign rows teaches the
judge the wrapper. Use one generic template for every row, and name it in `projection`.

**Rules re-flag resent history.** A gateway resends the conversation every turn. Scope each
branch to new items, and prove the scope with the scans in the rule-shape reference.

**Parallel calls are skipped.** A branch over only the last call of the newest message never
judges earlier calls in the same assistant turn, and a branch over only the newest tool response
never judges earlier results in the same batch. Judge every pending call and every tool result
since the newest assistant message.

**Tool definitions run up cost.** They arrive with every request. A narrow prefilter is what keeps
any branch over them cheap.

**Offline and serving input differ.** Tags, preamble, and truncation differ between the offline
scorer and the serving message builder. Re-check scores on the serving path.

**The prefilter overfits to one source.** Atoms mined from one source inflate dev recall. Watch the
dev-minus-accept gap and per-source recall, and require atoms that match at least 2 sources.

**The budget goes unused.** A prefilter far under its pass-through cap leaves recall behind.
Optimize recall under the cap.

**Counting unverified flags as false positives.** Unverified benign pools contain real finds.
Verify the flags, and report confirmed FPR next to the raw benign rate.

**Generated positives dominate.** When most positives come from generated or mixed sources, recall
on native traffic can differ. Report provenance by label type, and state that caveat.

**Delegated work drifts.** Broad task prompts wander and produce changes nobody asked for. Scope each
delegated task to one spec, forbid commits in the spec, and review every diff.

**A wait loop never exits.** `pgrep -f` and `pkill -f` match the command line of the shell that
runs them. Track jobs by PID.

**The session stops on attack content.** Hosts may halt on attack material. Keep raw data
work out of the main session, for example in subagents that report counts and paths only.
