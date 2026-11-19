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
`rules/rule-packs/`. Each rule has its own folder containing one `.rule` file and
any runtime prompts or resources. Pack and rule READMEs are optional. Keep
synthetic cases in `rules/test-cases/<pack-id>/<rule-stem>/`.

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
- `version`: independent rule version in Semantic Versioning format.
- `status`: `experimental`, `stable`, or `deprecated`.
- `requires_quarry`: tested compatibility using `>=VERSION` or
  `>=VERSION,<VERSION`.

`title`, `threat_type`, and `signal_channel` are optional. Other string metadata
is allowed. For updates by additional contributors, keep `author` and record
their attribution in a `contributors` string or the contribution history.

## Cases and compilation inputs

Each active rule needs synthetic matching and clean cases, including useful
near misses. Store them separately in `positive/` and `negative/`, and declare
their expected results in `expectations.yaml`:

```yaml
cases:
  positive/example.json:
    match: true
  negative/example.json:
    match: false
```

When a completion expression needs a response shape at compile time, add the
model alias and its JSON Schema under `compile.response_schemas`:

```yaml
compile:
  response_schemas:
    judge:
      type: object
      properties:
        verdict:
          type: boolean
      required: [verdict]
      additionalProperties: false
```

This describes the compiler input; it does not configure a provider or run a
model. The validation runner (`rules/README.md#validate-the-corpus`) reports cases
requiring models, probers, named text, or parameters as unevaluated.

## Versions and releases

Rule versions live in metadata; pack versions live in immutable Git tags and
`rules/rule-packs/<pack-id>/CHANGELOG.md`. There is no per-pack manifest. For example, tag
`rule-packs/core/coding-agents-runtime-behavior-rules/v0.1.0` identifies that
pack's release. Consumers pin release commits and select their packs explicitly.

- Use `0.x` versions or prereleases for experimental content.
- Patch releases fix behavior within documented intent, documentation, or cases.
- Minor releases add compatible coverage or rules. Breaking changes need an
  explicit migration note; stable releases use a major version change.
- Version affected rules when shared prompts or resources change. Unchanged
  rules keep their versions, and the pack changelog identifies each change.

## Deprecation and removal

For a supported rule that should be retired, set `status = "deprecated"`, add a
reason in the pack changelog, and preferably a `deprecation_reason` string. A
replacement is optional. Deprecated rules can still compile and run when
explicitly selected.

If a rule no longer compiles or is unsupported, remove its active rule,
associated resources and cases, and inventory entries together in a reviewed
pull request. Record its ID, reason, last supported tag or commit, and any
replacement in the changelog. Git preserves the previous rule and cases.
Broken or unsafe rules may be removed immediately while Quarry is in alpha.
