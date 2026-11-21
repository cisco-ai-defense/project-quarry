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

# Rule pack contract

Core packs live at `rules/rule-packs/core/<pack>/`; community packs live at
`rules/rule-packs/community/<publisher>/<pack>/`. A pack ID is this path relative to
`rules/rule-packs/`. Each rule has its own folder containing one `.rule` file
and a `README.md` describing the threats the rule covers and its limits.
Only Markdown or plain text files needed by the rule at runtime may accompany
these files. Put judge prompts beside the `.rule` file and use
`<detection_purpose>_judge_prompt.md` for their names. A pack-level README is
optional.

The Quarry team owns core pack testing and maintenance. Repository maintainers
review changes to both core and community packs. Any contributor may improve an
existing rule; preserve its original attribution and add contributors.

## Rule identity and metadata

Use stable, unique declared rule names. We recommend
`<pack-id>/<rule-stem>` as the declared name and
`<threat_name>_<observable_behavior>.rule` as the filename. These naming
conventions aid discovery; a valid existing rule is not rejected solely because
its declared name lacks a pack namespace. Quarry rejects duplicate declared
names across selected files.

Rule metadata values are strings. The
metadata schema (`rules/rule-packs/metadata-schema.yaml`) requires:

- `description`: detected behavior and its important boundaries.
- `author`: original author attribution.
- `revision`: the current rule revision as a nonempty string, such as `"0"` for a new rule.
- `requires_quarry`: tested compatibility using `>=VERSION` or
  `>=VERSION,<VERSION`.

### Optional metadata

- `title` gives the rule a reader-facing name.
- `severity` provides an advisory triage label based on potential impact and
  the evidence available at match time. For model-backed rules, set this
  label for your triage policy after evaluating your chosen thresholds.
- `threat_type` describes the behavior, for example `persistence`.
- `signal_channel` identifies the evidence the rule inspects, for example
  `tool_calls` or `tool_responses`.

Choose metadata values that describe the rule's coverage. Other string metadata
is allowed. For updates by additional contributors, keep `author` and record
their attribution in a `contributors` string or the contribution history.

## Rule README

Put a Configuration section at the top of the rule README for any model,
prober, or parameter requirements. For model-backed rules, document the model
alias, prompt file, text resource key, and threshold bindings. Describe the
observable behavior, inspected surfaces, example scenarios, and limits. Keep
descriptions at the level of behavior.
You may link to a permitted dataset on GitHub or Hugging Face. The
[rules contribution guidelines](../rules-contribution-guidelines.md) define
acceptable references and prohibited material.

## Versions and releases

Pack versions live in immutable Git tags. There is no per-pack manifest. For example, tag
`rule-packs/core/coding-agents-runtime-behavior-rules/v0.1.0` identifies that
pack's release. Consumers pin release commits and select their packs explicitly.

- Use `0.x` versions or prereleases for experimental content.
- Patch releases fix behavior within documented intent or correct documentation.
- Minor releases add compatible coverage or rules. Breaking changes need an
  explicit migration note; stable releases use a major version change.
- Record changes to shared prompts or resources and their affected rules in the
  contribution history.

## Deprecation and removal

For a supported rule that should be retired, record the reason in the
contribution history. A replacement is optional. Rules that remain in the pack
can still compile and run when explicitly selected.

If a rule no longer compiles or is unsupported, remove its rule file, README,
and any resources unique to that rule together in a reviewed pull request.
Record its ID, reason, last supported tag or commit, and any replacement in the
contribution history.
Broken or unsafe rules may be removed immediately while Quarry is in alpha.
