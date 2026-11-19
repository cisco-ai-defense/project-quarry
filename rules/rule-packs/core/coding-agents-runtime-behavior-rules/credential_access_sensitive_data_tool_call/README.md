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

# Sensitive data access in a tool call

The taxonomy mapping identifies related risks for review; a matching access
may be authorized and does not by itself establish credential theft.

This rule groups readable file paths, credential-store commands, environment
indicators, and sensitive-data terms as a prefilter for tool-call arguments.
Each selected call and its name are sent individually to a model judge that distinguishes
access to credential-bearing data from benign names, examples, and status calls.
The same indicator may be selected in many legitimate contexts.

Configure the score-capable model alias `credential_access_judge` and register
`resources/credential_access_judge_v1.md` as `credential_access_judge_v1`.
Supply a numeric `threshold` for each scan, and select and calibrate it on
representative data. The offline validator compiles the rule but does not run
its model-dependent cases. No detection accuracy is established by the examples.

Only the current tool-call arguments are judged; the rule does not see the
full task, a prior call, or a tool response. A match does not establish that
access succeeded or was unauthorized. Incomplete scans and provider errors are
not clean non-matches.
