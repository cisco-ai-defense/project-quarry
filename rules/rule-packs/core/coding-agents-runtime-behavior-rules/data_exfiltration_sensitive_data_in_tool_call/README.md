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

## Configuration

Register the model alias under `models` for the `score` operation and the prompt
file under `texts`.

| Model alias | Prompt file | `text_ref` key | Threshold binding |
| --- | --- | --- | --- |
| `sensitive_data_egress_judge` | [sensitive_data_egress_judge_prompt.md](sensitive_data_egress_judge_prompt.md) | `sensitive_data_egress_policy` | Supply `threshold` in `parameters` with each scan. |

The bundled prompt is for a binary LLM judge. GPT and Claude Sonnet or Haiku are
model families to evaluate. For System One, such as TypeSafe Jev, register
equivalent yes/no criteria under the same text resource key. See
[judge backends](../README.md#judge-backends) for adapter setup and cutoff guidance.

**Advised severity `HIGH`.** A matching call can disclose credentials or
private data outside their intended environment. You can adjust the `severity`
metadata for your triage policy after evaluating precision and recall at your
chosen threshold.

## Detection scope

This rule looks for calls that send sensitive data outside the local context.
The prefilter checks the raw arguments for transfer operations, network tools,
external collection endpoints, cloud metadata, and sensitive-data terms. The
configured judge receives each selected call's name and raw arguments and
considers the operation, data, and destination together. A match requires a
score at or above the scan's threshold.

## Example scenarios

The table describes the intended judge decisions for calls that reach it.

| Scenario | Intended judge score |
| --- | --- |
| A call sends private account records to an external recipient. | High. The call discloses private data to another destination. |
| A call sends a local credentials file to an external service. | High. The call transfers credential-bearing material. |
| A call downloads a public package. | Low. The download sends no sensitive data. |
| A call performs a routine service lookup with a public identifier. | Low. The identifier is public. |

## Limits

- Transfer and destination terms can select ordinary calls. The prefilter covers
  a subset of network operations and encodings.
- The rule sees one call at a time. A match indicates a possible transfer.
  Confirming that the transfer succeeded or lacked authorization requires
  evidence beyond the selected call.
- Treat provider errors and incomplete scans separately from non-matches.
