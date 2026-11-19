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

# Coding agents runtime behavior rules

This core pack contains four rules illustrating model-assisted
detection of:

- Changes to agent configuration containing download-and-execute commands.
- Attempts to override agent instructions or approval controls.
- Sensitive data sent through tool calls.
- Access to credential stores or sensitive files through tool calls.

Each rule uses readable word and phrase prefilters to select tool-call content
for a model judge. The rules and their policy texts describe the inspected
surfaces and their limits. Configure each model alias and policy text, and bind
the required threshold before running a rule. Thresholds and detection accuracy
have not been calibrated for these examples.

The matching and nonmatching examples are under
`rules/test-cases/core/coding-agents-runtime-behavior-rules/`, separate from the
rules. From the repository root, run `uv run --frozen python rules/validate.py`
to validate the corpus. Offline validation compiles model-dependent rules and
checks the case inputs, but does not evaluate their verdicts. Test configured
models against representative data before relying on these rules.
