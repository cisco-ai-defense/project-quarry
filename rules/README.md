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

# Detection rule packs

This directory contains maintained detection rules and their example validation
cases. The examples (`examples/README.md`) teach Quarry's language and integrations.

```text
rules/
  rules-contribution-guidelines.md
  rule-packs/
    metadata-schema.yaml
    core/
      coding-agents-runtime-behavior-rules/
        <rule-stem>/
          <rule-stem>.rule
          README.md                  # optional
          prompt.md                  # optional
          resources/                 # optional
    community/
      <publisher>/
        <pack>/                      # same contents as a core pack
  test-cases/
    corpus-manifest.json
    core/<pack>/<rule-stem>/
      positive/<case>.json
      negative/<case>.json
      expectations.yaml
    community/<publisher>/<pack>/<rule-stem>/
      positive/<case>.json
      negative/<case>.json
      expectations.yaml
  validate.py
```

Core packs are maintained and tested by the Quarry team. Community packs are
maintained by their contributors, with repository maintainers reviewing changes.

## Contribution guidelines

Follow the [rule contribution guidelines](rules-contribution-guidelines.md)
for accepted names, existing-rule comparisons, publisher namespaces, example
cases, attribution, and submission evidence. The
repository contribution guide (`CONTRIBUTING.md`)
covers the general pull request process. Anyone may improve an existing rule;
preserve its original attribution and credit additional contributors.

The [automated checks](rules-contribution-guidelines.md#automated-checks)
apply to every core and community pack. Maintainers review detection quality,
redundancy, relevance, and provenance before accepting a contribution.

## Included example packs

The core pack contains four model-judged example rules. The community pack
contains one local rule:

- Core: coding agents runtime behavior (`rules/rule-packs/core/coding-agents-runtime-behavior-rules/README.md`)
  covers instruction control, sensitive data access, data transfer, and
  suspicious text written to agent configuration.
- Community: Example Labs shell review (`rules/rule-packs/community/example-labs/shell-review/README.md`)
  checks download-to-shell commands submitted to shell tools. Example Labs is a
  placeholder publisher illustrating the community layout.

Each example has matching `positive/`, `negative/`, and `expectations.yaml` files
under `rules/test-cases/`, separate from the actual rules.

## Validate the corpus

From the repository root, run:

```console
uv run --frozen python rules/validate.py
uv run --frozen python rules/validate.py --compile-only
```

The default runner checks the inventory and metadata, compiles active rules, and
executes cases for local rules. Rules requiring model calls, probers, named text
resources, or parameter bindings are reported as **compile-only**, with their
cases explicitly unevaluated. Completion rules can supply response schemas for
compilation in `expectations.yaml`. `--compile-only` skips all case execution.
Compilation establishes compatibility, not evaluated detection performance.

CI does not send fixture content to an LLM or an external provider. Contributions
using models or probers are welcome when they compile; provide separate reviewed
evaluation evidence for their runtime behavior. A failed or incomplete scan
cannot pass as a negative case.

The reviewed `rules/test-cases/corpus-manifest.json` records repository-relative paths
in `rules`, `resources`, and `test_cases`. Add or remove rules, resources, and
their cases in the same inventory change.

## Select packs explicitly

Quarry loads configured `.rule` files. A directory loads only its immediate rule
files unless recursion is enabled. Select individual files or recurse within a
chosen pack, such as
`rules/rule-packs/core/coding-agents-runtime-behavior-rules/`. Configure named text,
model aliases, and probers in the application; use `compile_policy()` for
parameterized rules.

See the pack contract (`rules/rule-packs/README.md`) for identity, metadata, versions, and
deprecation, and the rule contribution guidelines (`rules/rules-contribution-guidelines.md`)
for naming, attribution, and review.
