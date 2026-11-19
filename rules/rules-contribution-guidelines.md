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

Core and community rules follow the pack contract (`rules/rule-packs/README.md`) and the
corpus validation workflow (`rules/README.md`). Repository maintainers review all
contributions. Anyone may edit any publisher's rules; preserve original author
attribution and add contributor credit. Publisher maintainers are contacts for
their content, and repository maintainers approve changes.

## Before adding a rule

Search all community publishers, core rules, and open contribution pull requests
for related detections. Reference similar rules and explain the coverage gap.
Prefer extending an existing rule when it already addresses the behavior.
Changing the publisher or rule name does not justify duplicate coverage;
redundant additions without a demonstrated gap will be rejected.

Publisher names identify the contributor or organization. Pack and rule names
describe their detection scope or observable behavior. Inappropriate,
misleading, or irrelevant names and content will be rejected even if technical
checks pass. Relevant illustrative attack scenarios are valid fixture content.

## Publisher and rule layout

Use lowercase kebab-case publisher and pack directories:

```text
rules/rule-packs/community/
  example-labs/
    coding-agent-security/
      <rule-stem>/
        <rule-stem>.rule
        README.md                    # optional
        resources/                   # optional
```

Pack IDs derive from their paths, and pack versions live in Git tags. A
publisher namespace identifies the contributor or organization.

We recommend lowercase snake_case rule filenames using
`<threat_name>_<observable_behavior>.rule`, for example
`indirect_injection_config_write_after_tool_output.rule`. Where applicable,
consider a threat name from
[Cisco's AI Security Framework](https://learn-cloudsecurity.cisco.com/ai-security-framework)
and explain the mapping in the contribution. Keep the rule and case folders
consistent with the filename stem. Describe only behavior the rule can establish;
event order alone does not prove causation.

## Submission evidence

Supply the rule, required runtime resources, and cases under
`rules/test-cases/<pack-id>/<rule-stem>/`. Include
`positive/`, `negative/`, and `expectations.yaml`, and update
`rules/test-cases/corpus-manifest.json`. Use only fictitious conversations and
reserved example domains; keep private transcripts, credentials, private URLs,
and proprietary test material out of submissions.

Pack and rule READMEs are optional. The pull request must still explain:

- Detected behavior, inspected surfaces, and important match boundaries.
- Related rules and the distinct coverage added.
- Positive and negative scenarios, false positives, and unsupported forms.
- Runtime dependencies, parameter bindings, and compatible Quarry versions.
- Evaluation performed, its limitations, and all compile-only or unevaluated cases.
- Source and license provenance, preserving original attribution and notices.

Compilable model and prober rules are accepted. Provide any completion response
schemas needed for compilation as described in the pack contract. CI evaluates
local cases and reports other cases as unevaluated; it does not send untrusted
fixture content to an LLM or external provider. Submit runtime evaluation
evidence separately for model or prober behavior.

## Automated checks

The CI workflow (`.github/workflows/ci.yml`)
runs the rule-pack validator (`rules/validate.py`)
on every pull request and on pushes to `main`, using Python 3.11, 3.12, and 3.13.
It checks all core and community packs across every publisher:

- Rule folder/file layout and globally unique declared rule names.
- Required string metadata, version/status formats, and compilation against the
  checkout's Quarry runtime, including `requires_quarry` compatibility.
- Reviewed inventory coverage for rules, resources, and cases, including missing
  files, unlisted additions, and orphaned cases.
- Matching positive/negative JSON cases and expectations, valid conversation
  inputs, and safe contained file paths and YAML loading.
- Complete scans with expected verdicts for local rules. An `evidence: true`
  expectation also requires match evidence to be present.

Model, prober, named-text, and parameter-dependent rules compile without runtime
execution. Their fixture inputs are validated and their verdicts are reported as
unevaluated. Fixture validation does not use external providers or credentials.
Recommended naming conventions produce warnings; duplicate declared rule names
and structural or validation errors fail the check.

## Maintainer review

CI does not establish semantic uniqueness, content relevance, fixture-data
provenance, attribution, licensing, detection quality, or appropriate release
changes. Maintainers review these alongside related-rule comparisons, bypasses,
false positives, coverage limits, version updates, and deprecation or
removal reasons.

From the repository root, run `uv run --frozen python rules/validate.py` and attach
the results. Maintainers review correctness, bypasses, false positives,
provenance, compatibility, and limitations before accepting a change. A
replacement rule is not required when deprecating an unsupported rule; follow the
pack contract's deprecation and removal process.
