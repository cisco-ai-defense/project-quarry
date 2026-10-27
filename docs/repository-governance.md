# GitHub repository controls

These settings live on GitHub and are required before making this repository
public. The files in `.github/` do not turn on branch protection by themselves.

## Access

- Create a visible `@cisco-ai-defense/project-quarry-maintainers` team and give it Write or
  Maintain access to this repository. Only trusted maintainers should have
  Write, Maintain, or Admin access. Give community moderators Triage access.
- Keep external contributors on forks. They can open issues and pull requests
  without repository write access.
- Verify that GitHub recognizes every entry in `.github/CODEOWNERS`. A hidden
  team or a team without repository write access does not count as a code owner.

## Protect `main`

Use a branch protection rule for `main` and verify the effective rule if the
organization also has rulesets:

- Require a pull request, at least one approving review, and code owner review.
  Only reviews from eligible maintainers count toward the merge gate; GitHub
  cannot stop other users from submitting advisory reviews.
- Dismiss stale approvals when commits change and require approval of the most
  recent reviewable push by someone other than its author.
- Require conversation resolution and an up-to-date branch before merging.
- Require every CI check, from GitHub Actions as the expected source:
  `quality`, `test (Python 3.11)`, `test (Python 3.12)`,
  `test (Python 3.13)`, `build (Python 3.11)`,
  `build (Python 3.12)`, and `build (Python 3.13)`.
- Restrict who can push to `main` to the maintainer team. People on this list
  must still satisfy the pull request and check requirements.
- Enforce the rule for administrators; disable force pushes and branch deletion.
  Do not configure a bypass for routine merges.

Do not make a required job conditional on paths or changed files: a skipped
job can report success without testing the PR. If enabling merge queue later,
add the `merge_group` event to every required workflow first.

## Actions and security settings

- Under Actions > General > Actions permissions, require actions to be pinned
  to a full-length commit SHA. The current workflows already use full SHAs.
- Under Actions > General > Workflow permissions, use a read-only default
  `GITHUB_TOKEN` and leave "Allow GitHub Actions to create and approve pull
  requests" off. Organization or enterprise policy may control these options.
  The workflows also set their own read-only permission.
- While the repository is private, leave write tokens and secrets unavailable
  to fork pull request workflows. Once it is public, set "Require approval for
  all external contributors" under Actions > General. Review changes to
  `.github/workflows/` as security-sensitive code.
- Enable Dependabot alerts and security updates. The `dependabot.yml` file
  schedules version-update PRs; it does not enable alerts.
- For secret scanning and push protection while private, the organization
  needs GitHub Secret Protection or legacy GitHub Advanced Security, which
  may incur a charge. Public repositories receive secret scanning for free;
  after publication, verify repository alerts and enable push protection.
- After publication, enable private vulnerability reporting under Advanced
  Security. GitHub provides this reporting form for public repositories.
- Review organization and repository collaborators, deploy keys, installed
  apps, webhooks, and environment secrets before publication. Remove access
  that is no longer needed.

## Publishing

There is currently no publishing workflow. A future package release should
run only from trusted refs, use a protected environment with maintainer
approval, and use PyPI trusted publishing through short-lived OIDC credentials.
Never promote artifacts or run scripts supplied by a fork PR in a privileged
release job.

## Verify with the GitHub CLI

Run these with a CLI account that has repository administration access:

```sh
gh api repos/cisco-ai-defense/project-quarry/branches/main/protection
gh api repos/cisco-ai-defense/project-quarry/rulesets
gh api repos/cisco-ai-defense/project-quarry/codeowners/errors
gh api repos/cisco-ai-defense/project-quarry/actions/permissions
gh api repos/cisco-ai-defense/project-quarry/commits/main/check-runs --jq '.check_runs[].name'
```

Confirm the named team exists and has repository write access before relying
on required code owner review. Confirm each required check name against an
actual completed pull request run, then verify that a non-maintainer cannot
merge and that `main` rejects direct updates and force pushes.
