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

# Shell review example pack

Community pack with one example rule:
download-to-shell command submitted to a shell tool (`rules/rule-packs/community/example-labs/shell-review/code_execution_download_to_shell_call/code_execution_download_to_shell_call.rule`).
Example Labs is a placeholder publisher illustrating a contributor namespace.

The rule checks the `command` argument of `execute_shell`, `run_shell`, and
`shell` calls for a `curl`/`wget` pipeline to `sh`/`bash`. It detects submitted
command text; it does not prove execution or missing approval. Coverage is
limited to these named tools and the literal command pipeline.

Example validation cases stay outside the pack:

- Positive example (`rules/test-cases/community/example-labs/shell-review/code_execution_download_to_shell_call/positive/curl_shell_command.json`)
- Negative example (`rules/test-cases/community/example-labs/shell-review/code_execution_download_to_shell_call/negative/shell_download_for_review.json`)
- All expected results (`rules/test-cases/community/example-labs/shell-review/code_execution_download_to_shell_call/expectations.yaml`): 2 positive and 4 negative cases.

From the repository root, run `uv run --frozen python rules/validate.py` to validate the corpus.
