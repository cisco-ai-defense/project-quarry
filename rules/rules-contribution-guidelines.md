<!--
Copyright 2026 Cisco Systems, Inc. and its affiliates

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

SPDX-License-Identifier: Apache-2.0
-->

# Rules contribution guidelines

Core and community rules follow the [pack contract](rule-packs/README.md) and the
[rule pack validation workflow](README.md#validate-rule-packs). Repository maintainers review all
contributions. Anyone may edit any publisher's rules; preserve original author
attribution and add contributor credit. Publisher maintainers are contacts for
their content, and repository maintainers approve changes.

## Prohibited content

Project Quarry does not encourage, support, or recommend creating, acquiring,
using, or sharing prohibited material. Contributors must comply with this
policy, applicable law, and the policies of any platform they use to submit or
reference material. Contributors must not submit, solicit, distribute, or
direct others to any of the following material through the project.

- Active attack payloads, working exploits, malware, and other material that
  can directly trigger harmful or unauthorized actions when run or supplied
  to an agent.
- Sexually explicit or NSFW content, including nonconsensual intimate imagery,
  and child sexual abuse material (CSAM) in any form. The prohibition includes
  real, altered, and AI-generated material.
- Credible threats, material that incites violence, graphic violence,
  targeted harassment, or hate content.
- Credentials, private transcripts, personal or confidential information
  disclosed without authorization, and material that violates applicable law
  or third-party rights. Contributors must have the necessary rights and
  permissions for material they submit or reference.

These requirements apply to commits (including earlier commits in a pull
request), code, READMEs, prompts, resources, test cases, PR descriptions and
reviews, issues, comments, screenshots, attachments, and external links.
They apply to text, images, audio, video, and any embedded, encoded (including
Base64), obfuscated, encrypted, or archived form. Testing, research,
demonstrations, and multimodal evaluation are subject to the same restrictions.

### Examples of policy violations

- A detection test that embeds a prohibited image as Base64, or a ZIP
  attachment containing prohibited samples, violates this policy.
- A README that links to a dataset, release, or repository containing
  prohibited material violates this policy, including when the destination
  is gated or the link is labeled for research.
- A test report that reproduces a working attack payload, secret, or private
  conversation violates this policy. Describe the behavior in plain language
  and use non-executable indicators or aggregate evaluation results.

### Enforcement and reporting

Pull requests containing prohibited material will be rejected. Maintainers
may hide or remove material, close or lock discussions, revoke commit access,
or impose a temporary or permanent ban in accordance with the project's
[Code of Conduct](../CODE_OF_CONDUCT.md). Maintainers may escalate violations
to the hosting platform and refer matters to the appropriate authorities where
required by applicable law.

Report prohibited material to the platform hosting it. For content on GitHub,
use its [abuse reporting process](https://docs.github.com/en/communities/maintaining-your-safety-on-github/reporting-abuse-or-spam).
Report suspected CSAM immediately and do not download, copy, forward, or
attach it. For security vulnerabilities in Quarry, follow the
[security policy](../SECURITY.md).

## Before adding a rule

Search all community publishers, core rules, and open contribution pull requests
for related detections. Reference similar rules and explain the coverage gap.
Prefer extending an existing rule when it already addresses the behavior.
Changing the publisher or rule name does not justify duplicate coverage;
redundant additions without a demonstrated gap will be rejected.

Publisher names identify the contributor or organization. Pack and rule names
describe their detection scope or observable behavior. Inappropriate,
misleading, or irrelevant names and content will be rejected even if technical
checks pass.

## Publisher and rule layout

Use lowercase kebab-case publisher and pack directories:

```text
rules/rule-packs/community/
  example-labs/
    coding-agent-security/
      <rule-stem>/
        <rule-stem>.rule
        README.md
        <detection_purpose>_judge_prompt.md  # optional
        resources/                         # optional other runtime files
```

Pack IDs derive from their paths, and pack versions live in Git tags. A
publisher namespace identifies the contributor or organization.

We recommend lowercase snake_case rule filenames using
`<threat_name>_<observable_behavior>.rule`, for example
`indirect_injection_config_write_after_tool_output.rule`. Where applicable,
consider a threat name from
[Cisco's AI Security Framework](https://learn-cloudsecurity.cisco.com/ai-security-framework)
and explain the mapping in the contribution. Match the rule folder name to the
filename stem. Describe only behavior the rule can establish. Event order alone
does not prove causation.

## Required rule metadata

Every rule must include `description`, `author`, `revision`, and
`requires_quarry` as nonempty strings. Use `revision` to identify the current
revision of the rule, for example `revision = "0"` for a new rule. See the
[rule pack contract](rule-packs/README.md#rule-identity-and-metadata) for field
definitions and compatibility syntax.

## Rule README and dataset references

Add a `README.md` alongside every `.rule` file. Submit the rule, its README,
and any plain-text prompt or resource required at runtime. A deterministic rule
with no runtime resources needs only the rule file and its README. The README
should give reviewers the following information.

- The type of threat the rule covers and the behavior or signals it checks.
- Related rules, distinct coverage, likely false positives, and important limits.
- A Configuration section at the top with runtime dependencies, parameter
  bindings, and compatible Quarry versions. For a model-backed rule, include
  the model alias, prompt file, text resource key, and threshold bindings.
- A summary of any evaluation, including matching and clean scenarios and
  limitations.
- Source and license provenance, preserving original attribution and notices.

Dataset references are optional. Link to a dataset page on Hugging Face or
GitHub if you have permission to share it and the dataset complies with this
policy and the host's content rules. Include the dataset's name, license, and
what it was used to evaluate. Describe behavior in plain language and keep
samples, raw outputs, and media out of the contribution.

## Automated checks

The [trusted rule pack workflow](../.github/workflows/trusted-rule-packs.yml)
checks proposed core and community packs before merge. It runs the validator,
compiler, and locked dependencies from the pull request's base commit on
`main` and reads the proposed merge's pack files as data from Git.
It uses the metadata schema from `main`. The regular CI workflow validates the
published packs again on pushes to `main` across Python 3.11, 3.12, and 3.13.
Land compiler or schema changes before submitting rules that depend on them.

The validator checks:

- Rule folder and file layout, one README per rule, and globally unique names.
- Required string metadata and compilation against the trusted Quarry runtime,
  including `requires_quarry` compatibility.
- Supported text file formats and bounded, contained file paths.

CI compiles rules offline. It does not run models or probers, fetch linked
datasets, or determine whether text contains prohibited content. Maintainers
review contributions for compliance with the content policy.

To enforce this check before merge, configure the organization ruleset to
require the workflow file from `refs/heads/main` for the `main` branch of
`cisco-ai-defense/project-quarry`. Select the workflow itself so the requirement
is tied to the protected source.

## Maintainer review

Maintainers review overlap with existing rules, content compliance, source and
dataset provenance, licensing, attribution, detection quality, and coverage
limits. Compilation establishes compatibility. Detection quality requires
separate evaluation.

From the repository root, run `uv run --frozen python rules/validate.py` and attach
the results. Follow the pack contract's deprecation and removal process if your
change retires an existing rule.
