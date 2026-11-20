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

This directory contains maintained detection rules. Each rule has a README
summarizing the threats it covers. The repository's language and integration
examples are under `examples/`.

```text
rules/
  rules-contribution-guidelines.md
  rule-packs/
    metadata-schema.yaml
    core/
      coding-agents-runtime-behavior-rules/
        <rule-stem>/
          <rule-stem>.rule
          README.md
          <detection_purpose>_judge_prompt.md  # optional
          resources/                         # optional for other runtime files
    community/
      <publisher>/
        <pack>/                      # same contents as a core pack
  validate.py
```

Core packs are maintained and tested by the Quarry team. Community packs are
maintained by their contributors, with repository maintainers reviewing changes.

## Contribution guidelines

Follow the [rule contribution guidelines](rules-contribution-guidelines.md)
for naming, existing-rule comparisons, publisher namespaces, attribution,
required README content, and dataset references. The
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

## Validate rule packs

From the repository root, run:

```console
uv run --frozen python rules/validate.py
```

The validator checks pack layout, rule metadata, and README presence, and
compiles the rules offline without provider configuration. Submitted rules must
compile without external compiler inputs, such as model response schemas.
Detection performance requires separate evaluation.
For rules that use models or probers, describe the runtime requirements and any
evaluation limitations in the rule README. Reviewers assess contribution
content under the [rules contribution guidelines](rules-contribution-guidelines.md).

## Select packs explicitly

Use the rule packs from a Git checkout or the Quarry source distribution.
Select the `.rule` files your application needs. To load a directory of
non-parameterized rules, enable recursion if the files are in rule
subdirectories. For a parameterized rule, compile each rule separately with
`compile_policy()`. Register any text resources, model aliases, and probers the
rule references, and use the rule README's Configuration section to bind policy
parameters and supply scan parameters.

See the pack contract (`rules/rule-packs/README.md`) for identity, metadata, versions, and
deprecation, and the rule contribution guidelines (`rules/rules-contribution-guidelines.md`)
for naming, attribution, and review.
