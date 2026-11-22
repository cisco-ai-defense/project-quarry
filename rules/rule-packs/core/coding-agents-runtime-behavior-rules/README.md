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

Each rule uses readable word and phrase prefilters to select inputs for a model
judge. The Configuration section of each rule README lists the model alias,
prompt files, text resource keys, and threshold bindings.

## Judge backends

Use `config_version: 2` and configure the model alias for `score`. Under
`texts`, set `file` for each `text_ref` key to its prompt path, relative to the
configuration file. See [model inference](../../../../docs/model-inference.md).

- **LLM judge.** The bundled prompts ask for a `0` or `1` verdict. Evaluate a
  model family such as [OpenAI GPT](https://developers.openai.com/api/docs/models)
  or [Anthropic Claude Sonnet or Haiku](https://platform.claude.com/docs/en/models/overview)
  with a compatible scoring adapter, such as Quarry's
  [LiteLLM adapter](../../../../docs/model-inference.md#litellm-models).
- **System One decision model.** To use a model such as TypeSafe Jev,
  configure the [System One adapter](../../../../docs/model-inference.md#decision-models)
  and register equivalent yes/no criteria under the rule's text resource keys.
  The adapter returns the probability of a yes answer.

For binary scores, any cutoff strictly between `0` and `1` separates the
verdicts. For fractional scores, choose each cutoff to meet your precision and
recall targets using representative, permitted data. Evaluate each input
separately when a rule has two thresholds.

Run `make validate-packs` from the repository root to check the layout and
compile the rules offline.
