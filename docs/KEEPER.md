# The Keeper (phase 1)

The Keeper is a guard and organizing system for AI agents, built on the Quarry rule engine
in this repository. It gives every subject its own desk (facts, rules, checks, references),
checks an agent's conversation against those rules, and stops the step with an alert when
a rule breaks or the check can't finish.

Phase 1 is the foundation: subjects, standing rules, opt-in checks, the guard, and a daily
log. It runs **fully offline**.

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
- **Library checks** (turn on per subject):
  - `offline-only`: no network-type tools and no web addresses in tool calls (for SPARC).
  - `timestamp-first`: every reply with text starts with a bold New York timestamp.
  - `simple-terms-last`: replies over 600 characters include "In simple terms".
- **Guard.** `keeper check` loads the subject's rules, checks one conversation, and returns:
  - `PASS` (exit 0): every rule checked, none broken.
  - `STOP` because a rule broke (exit 1).
  - `STOP` because the check could not finish (exit 2): missing subject, unreadable or
    malformed conversation, a broken rule file, a timeout, or a network attempt.
- **Alerts** follow the fail-closed format: an `ALERT:` line, "What happened, step by
  step" (each broken rule, why, and where in the conversation), and "In simple terms".
  Secrets found in the conversation are replaced with `[hidden secret]` in the alert.
- **Log.** Every check adds one line to `log/<date>.md`.

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
  rejections/          phase 2
  subjects/<name>/
    subject.yaml       description + library checks in use
    FACTS.md
    RULES.md
    checks/*.rule      this subject's own rules
    references/
    jobs/
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

## Known limits in phase 1

- The outbound rule looks for an approval word anywhere in the user's messages, not
  right before the tool call. Phase 2 will tie the OK to the step.
- "No plain HTTP" also fires on quoted text such as license headers that contain
  `http://`. That is on purpose for now (the rule is "never").
- `offline-only` matches by tool name keywords (fetch, http, web, url, download, push,
  pull, clone, api and similar), so a harmless tool with one of those words in its name
  is stopped too. Rename the tool or ask for an exception.
- The rest of this repository still contains Quarry's optional model and plugin code. The
  Keeper never loads it, and the offline lock blocks it.

## Coming next

- **Phase 2:** logbook. Automatic path record, `SUCCESS.md` per accepted job, rejection
  folders per session that only Al deletes, and the "ghost lap" handed to the next
  similar job.
- **Phase 3:** image checker (size, palette, pixel match and a similarity score against
  the references).
- **Phase 4:** listener that drafts rules from Al's words into `pending-rules.md`, and
  the approval review.
- **Phase 5:** messenger between subjects and threads.
