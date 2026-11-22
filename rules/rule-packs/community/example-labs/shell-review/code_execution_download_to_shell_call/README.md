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

# Downloaded content passed to a shell

## Configuration

This deterministic rule runs without a model, prompt, or threshold.

**Advised severity `MEDIUM`.** A match identifies submitted command text.
Execution and authorization need separate evidence.

## Detection scope

This rule checks the `command` argument of supported shell tools, including
`execute_shell`, `run_shell`, `shell`, `Bash`, `exec_command`, and
`run_terminal_cmd`.

| Command family | Detection pattern |
| --- | --- |
| POSIX shell | A recognized download command feeds directly into a supported shell interpreter. |
| PowerShell | A web request command feeds directly into an expression evaluator. |

The rule looks for both stages in the same command argument. The downloader
must start at a command boundary, and the next pipeline stage must start
with the interpreter or evaluator.

## Example scenarios

| Scenario | Expected rule outcome |
| --- | --- |
| A supported shell tool receives a command that sends a recognized download directly to a recognized shell interpreter. | Match. |
| A supported shell tool downloads content to a file for inspection. | No match. |
| A user message describes the behavior without a shell tool call. | No match. |

## Limits

- Coverage depends on the recognized tool names, command names, and paths.
  The patterns inspect command text and can miss long arguments, quoted
  separators, shell wrappers, or multiline continuations. Command text alone
  cannot establish whether the download stage writes to standard output.
