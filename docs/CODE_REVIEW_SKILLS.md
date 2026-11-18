# Advanced code review skills

The repository includes three maintainer review skills:

| Skill | Use it for |
| --- | --- |
| [code-review-advanced](../.github/skills/code-review-advanced/SKILL.md) | A broad correctness review: interacting boundaries, failure paths, retries, data contracts, caches, and relevant lifecycle or concurrency analysis. |
| [code-review-state-machines](../.github/skills/code-review-state-machines/SKILL.md) | Finding implicit state machines and reviewing their transitions, invariants, ownership, and progress. |
| [code-review-concurrency](../.github/skills/code-review-concurrency/SKILL.md) | Concrete race schedules, async cancellation, shared work, reentrancy, deadlocks, and stale publication. |

The [prompt reference](../.github/skills/code-review-advanced/references/review-prompts.md)
contains additional review conditions, Quarry navigation hints, and primary research sources.
The prompts are our synthesis of those sources. Skill metadata and static checks do not
establish review accuracy or guarantee that all defects will be found.

## Location and discovery

The canonical files live in `.github/skills/<skill-name>/SKILL.md`. GitHub documents this
location specifically for [Copilot code review](https://docs.github.com/en/copilot/how-tos/use-copilot-agents/use-code-review#mcp-servers-and-agent-skills).
Review-focused names and descriptions help the reviewer select the relevant skill. Adding
skills makes them available to an enabled review surface; it does not configure automatic
review requests or guarantee selection in every review. Check review attributions or session
logs to see which skills a Copilot review used.

Codex discovers repository skills through `.agents/skills/` and
[supports symlinked skill folders](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills).
Each `.agents/skills/<skill-name>` is a relative directory symlink to its canonical
`.github/skills/<skill-name>` folder. Edit the canonical files so both entry points use the
same content. Check out with symlink support; if a platform materializes links as text files,
copy each complete canonical skill folder into `.agents/skills/` locally to enable discovery.

These are repository maintenance workflows. The root `skills/` directory belongs to the
distributed Quarry policy plugin and is configured by its plugin manifests. Keeping maintainer
skills here makes them available in a source checkout without adding them to that plugin.

## Example requests

In Codex, invoke the discovered skill with `$<skill-name>`; other hosts may offer their own
skill selector or select a skill from the request description.

```text
Use $code-review-advanced to review this PR against main. Inspect callers and failure paths.
Report reachable defects and unverified hypotheses separately. Do not change code.
```

```text
Use $code-review-state-machines to find all implicit state machines in this codebase.
Include machines encoded by combinations of flags, nullable fields, counters, caches,
tasks, and resource ownership. Report transitions, invariants, and search coverage.
```

```text
Use $code-review-concurrency to review shared cache work and scanner cancellation.
Give a concrete actor-by-actor schedule for each race or deadlock you report.
```

For a PR, the default scope follows the changed behavior and its dependencies. The explicit
codebase inventory request searches more broadly. Findings need code evidence, a reachable
trigger, a violated contract, and impact. Review output should state its revision, coverage,
and validation performed. Security findings follow [SECURITY.md](../SECURITY.md), and only
the user may merge pull requests.

## Maintaining the skills

Keep the entry points focused and load detailed prompts only when relevant. After changing a
skill, validate its frontmatter, references, and discovery links. For substantial reasoning
changes, try a small synthetic review with known behavior and a correct control case; assess
both missed defects and unsupported findings. Record what was actually checked in the PR.
Avoid tests that merely require the prompt to contain particular wording.
