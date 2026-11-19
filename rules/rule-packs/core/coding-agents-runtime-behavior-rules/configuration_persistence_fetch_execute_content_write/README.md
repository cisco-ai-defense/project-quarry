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

# Remote execution in agent configuration

This example selects tool calls containing both an agent-file indicator and an
execution-related word in the same argument string, then scores each selected
call separately. The model sees the tool name and raw arguments and evaluates
whether a configuration change would cause remote content to be fetched and run.
The prefilter uses readable, case-insensitive terms rather than a complete parser;
the judge must distinguish actual changes from reads, examples, and removals.

Configure a scoring model alias `configuration_persistence_judge`, register
`resources/configuration_persistence_policy.md` as the text resource
`configuration_persistence_policy`, and bind policy parameter `threshold` to a
number chosen for the deployment. No threshold or model accuracy has been
calibrated. Offline corpus validation compiles the rule but does not evaluate
its model-dependent cases.

Coverage depends on the listed file and execution terms and is limited to tool
calls; other file names, encodings, and tool results can be missed. Generic names
such as `settings.json` may select unrelated calls for scoring. The model has
only the selected call, not the full task or authorization context; a match is a
review signal rather than proof that a change succeeded or was unauthorized.
Selected tool arguments are sent to the configured model. Treat incomplete scans
and provider errors separately from clean non-matches.
