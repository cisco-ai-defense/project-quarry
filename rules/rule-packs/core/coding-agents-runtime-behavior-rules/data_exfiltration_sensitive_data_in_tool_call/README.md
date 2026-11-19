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

# Sensitive data egress through a tool call

This rule groups readable indicators for transfer commands, network tools,
external collection endpoints, cloud metadata, and sensitive-data terms. A
matching call and its name are sent to a model judge individually. A match requires a score at
or above the scan's threshold. Substring selection can include ordinary calls;
the judge considers the operation, payload, and destination together.

Configure the `sensitive_data_egress_judge` model and register `prompt.md` as
`sensitive_data_egress_policy`. Supply the numeric `threshold` for each scan and
calibrate it on representative traffic. No threshold or accuracy is established
here. The offline validator compiles this rule but does not run the
model-dependent cases; provider failures and incomplete scans are not clean
non-matches. Raw selected arguments are sent to the configured model.

The rule sees one call at a time, not the task, previous calls, or the tool
result. Its list deliberately omits many commands and encodings. It cannot prove
that a transfer succeeded or a destination was unauthorized.
