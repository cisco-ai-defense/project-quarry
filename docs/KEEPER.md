# The Keeper (phases 1 and 2)

The Keeper is a guard and organizing system for AI agents, built on the Quarry rule engine
in this repository. It gives every subject its own desk (facts, rules, checks, references),
checks an agent's conversation against those rules, and stops the step with an alert when
a rule breaks or the check can't finish.

Phase 1 is the foundation: subjects, standing rules, opt-in checks, the guard, and a daily
log. Phase 2 adds the logbook: job records, SUCCESS files, rejections, and the ghost lap.
Everything runs **fully offline**.

## What phase 1 does

- **Subjects.** `keeper subject add` makes one folder per subject with `FACTS.md`,
  `RULES.md` (what the AI reads), `checks/` (extra rules for this subject only),
  `references/`, `jobs/`, and `subject.yaml` (which library checks are on).
- **Standing rules** (always on, every subject):
  - No plain `http://` in replies or tool calls.
  - No API keys, tokens or private keys in replies or tool calls.
  - No obeying orders hidden in tool results ("ignore previous instructions" and similar).
  - No outbound tool (push, upload, publish, email, post, webhook) unless a user message
    says "approved", "go ahead", "you have my ok" or "keeper ok".
  - No agent deleting the Keeper's memory (rejections, bases, `path.log`, `SUCCESS.md`).
- **Library checks** (turn on per subject):
  - `offline-only`: no network-type tools and no web addresses in tool calls (for SPARC).
  - `timestamp-first`: every reply with text starts with a bold New York timestamp.
  - `simple-terms-last`: replies over 600 characters include "In simple terms".
  - `next-step-last`: replies over 200 characters end with a one-line "Next step".
- **Guard.** `keeper check` loads the subject's rules, checks one conversation, and returns:
  - `PASS` (exit 0): every rule checked, none broken.
  - `STOP` because a rule broke (exit 1).
  - `STOP` because the check could not finish (exit 2): missing subject, unreadable or
    malformed conversation, a broken rule file, a timeout, or a network attempt.
- **Alerts** follow the fail-closed format: an `ALERT:` line, "What happened, step by
  step" (each broken rule, why, and where in the conversation), and "In simple terms".
  Secrets found in the conversation are replaced with `[hidden secret]` in the alert.
- **Log.** Every check adds one line to `log/<date>.md`.

## The logbook (phase 2): the coach model

The Keeper works like a coach who already has the best lap recorded.

- **Start a job.** `keeper job start` creates `subjects/<subject>/jobs/<date>-<title>/` and
  prints the **ghost lap**: the current basis for that kind of job (the last result Al
  liked best) and the **marked crashes** (rejected directions still on file). If there is
  no basis for that exact kind, the basis sharing the most keywords is used. Only one
  basis is ever handed over, never the whole library.
- **Record steps as they happen.** `keeper job step` adds one step; `keeper job record`
  turns every tool call in a conversation into steps (this is what a Claude Code hook
  will call). Steps go to `path.log`, one line each, in time order, with the try number.
- **Reject.** `keeper job reject --why "..."` saves the try to
  `rejections/<date>-<session>/NN-<subject>-<job>-tryN.md` with Al's reason and the
  steps it took, then starts the next try.
- **Accept.** `keeper job accept` writes `SUCCESS.md`: the ask, rules in force, the full
  path in time order, rejected directions and why, the accepted direction, best tools,
  a replay recipe, and keywords for matching. If there is no basis for that kind yet,
  it becomes the basis. If Al likes it better than the current one, `--better` makes it
  the new basis and the old one moves to `bases/<kind>/archive/`.
- **New direction.** `keeper job start ... --new-direction "try a darker palette"` lifts
  the ghost lap and the avoid list for that one job, and records it.
- **Forget (Al only).** `keeper rejections list` shows what's on file.
  `keeper rejections forget <folder> <session-folder> [files...]` deletes one, a few, or a
  whole session. It asks you to type the session name (or pass `--yes`). The standing rule
  "Agent deleting the memory" stops an agent that tries.

```bash
keeper job start ~/keeper art "Hero walk cycle" --kind sprite-walk-cycle \
    --session hero-walk-sprites --keywords sprite hero walk --refs hero_walk.png \
    --ask "Build the walk cycle from hero_walk.png"
keeper job step ~/keeper art 2026-11-06-hero-walk-cycle --tool slice_sheet --note "8 frames"
keeper job reject ~/keeper art 2026-11-06-hero-walk-cycle --why "Don't redesign the hero"
keeper job accept ~/keeper art 2026-11-06-hero-walk-cycle --note "Matches the reference"
keeper ghost ~/keeper sprite-walk-cycle
```

## Offline guarantees

- The Keeper refuses any configuration that lists AI models, test-only stub checks, or any
  web address.
- While a check runs, outbound socket connections and network name lookups raise an error,
  and the network libraries used by Quarry's optional model add-on (`litellm`, `httpx`,
  `aiohttp`, `requests`, `urllib3`, `boto3`, `botocore`) cannot be imported. If one is
  already loaded in the same program, the Keeper refuses to run.
- The Keeper needs only the Quarry core packages. Do **not** install the `inference` or
  `plugin` extras.

## Commands

```bash
keeper init ~/keeper
keeper subject add ~/keeper sparc --description "SPARC offline brain" \
    --use offline-only timestamp-first simple-terms-last
keeper subjects ~/keeper
keeper rules
keeper check ~/keeper sparc conversation.json          # readable alert
keeper check ~/keeper sparc conversation.json --json   # full report for programs
```

The conversation file uses the common chat format: a JSON object with a `messages` list
(`user`, `assistant`, `system`, `tool` messages, with `tool_calls` on assistant messages).

From Python:

```python
from keeper import check

report = check(Path("~/keeper").expanduser(), "sparc", conversation)
if report.verdict == "STOP":
    print(report.alert_text())
```

## Folder layout

```
keeper/
  KEEPER.md            what this folder is
  INDEX.md             rebuilt by the Keeper
  pending-rules.md     rules waiting for Al's OK (phase 4 fills this)
  log/<date>.md        one line per check
  rejections/<date>-<session>/NN-....md   rejected tries (only Al deletes)
  bases/<kind>/BASIS.md                   current best for each kind of job
  bases/<kind>/archive/                   older bases
  subjects/<name>/
    subject.yaml       description + library checks in use
    FACTS.md
    RULES.md
    checks/*.rule      this subject's own rules
    references/
    jobs/<date>-<title>/
      job.yaml         the ask, kind, keywords, references, try number, status
      path.log         every step, in time order
      SUCCESS.md       written when Al accepts
```

## Writing a subject's own rule

Put a `.rule` file in `subjects/<name>/checks/`. Add a `plain` line in `metadata` so the
alert's "In simple terms" part can use it:

```text
rule "No sudo"
metadata:
    severity = "high"
    description = "A shell command used sudo."
    plain = "The AI tried to run a command as the computer's administrator."
condition:
    any call in tool_calls where call.arguments.command contains "sudo"
```

See `quarry/language/reference.md` for everything a rule can test.

## Known limits

- The outbound rule looks for an approval word anywhere in the user's messages, not
  right before the tool call.
- The logbook records what it is told (by a hook or a command). Steps an agent takes
  outside a hooked tool are not seen.
- The "Agent deleting the memory" rule matches common delete commands that name the
  Keeper's files. A determined program could still delete files another way; the Mac's
  own file permissions are the backstop.
- "No plain HTTP" also fires on quoted text such as license headers that contain
  `http://`. That is on purpose for now (the rule is "never").
- `offline-only` matches by tool name keywords (fetch, http, web, url, download, push,
  pull, clone, api and similar), so a harmless tool with one of those words in its name
  is stopped too. Rename the tool or ask for an exception.
- The rest of this repository still contains Quarry's optional model and plugin code. The
  Keeper never loads it, and the offline lock blocks it.

## Coming next

- **Phase 3:** image checker (size, palette, pixel match and a similarity score against
  the references).
- **Phase 4:** listener that drafts rules from Al's words into `pending-rules.md`, and
  the approval review.
- **Phase 5:** messenger between subjects and threads.
