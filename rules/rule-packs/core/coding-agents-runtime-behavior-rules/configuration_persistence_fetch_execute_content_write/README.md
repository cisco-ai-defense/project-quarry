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

## Configuration

Register the model alias under `models` for the `score` operation and the prompt
file under `texts`.

| Model alias | Prompt file | `text_ref` key | Threshold binding |
| --- | --- | --- | --- |
| `configuration_persistence_judge` | [configuration_persistence_judge_prompt.md](configuration_persistence_judge_prompt.md) | `configuration_persistence_policy` | Bind `threshold` in `policy_parameters` when compiling the policy. |

The bundled prompt is for a binary LLM judge. GPT and Claude Sonnet or Haiku are
model families to evaluate. For System One, such as TypeSafe Jev, register
equivalent yes/no criteria under the same text resource key. See
[judge backends](../README.md#judge-backends) for adapter setup and cutoff guidance.

**Advised severity `HIGH`.** A persistent configuration change can make later
agent sessions automatically fetch and run remote content. You can adjust the
`severity` metadata for your triage policy after evaluating precision and recall
at your chosen threshold.

## Detection scope

This rule looks for configuration changes that would make an agent fetch and run
remote content automatically. Both an agent-file indicator and an execution
indicator must occur in the same call's arguments. The configured judge receives
the tool name and raw arguments for each selected call. A match requires a score
above the threshold.

## Example scenarios

The table describes the intended judge decisions for calls that reach it.

| Scenario | Intended judge score |
| --- | --- |
| A configuration write adds a startup hook that would fetch and run remote content. | High. The change adds persistent execution. |
| A call reads a settings file to inspect an existing hook. | Low. The read leaves the configuration intact. |
| An edit removes an existing hook that fetches remote content. | Low. The change removes the behavior. |
| A call writes documentation about a risky setup. | Low. The write leaves agent configuration intact. |

## Limits

- The case-insensitive prefilter uses literal terms. Generic filenames such as
  `settings.json` may select unrelated calls, and other file names, encodings,
  and tool results can be missed.
- The judge sees one call at a time. A match identifies a proposed configuration
  change for review. Whether it succeeded or was authorized depends on context
  outside the call.
- Treat incomplete scans and provider errors separately from non-matches.
