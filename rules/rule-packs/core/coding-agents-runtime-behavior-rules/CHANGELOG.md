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

# Changelog

## 0.1.0 — Unreleased

- Add an experimental review signature for download-to-shell text written to agent configuration, with synthetic matching and near-miss cases.
- The signature checks named tool calls and their path/content fields. It does not prove that configuration was consumed, that code executed, or that user authorization was missing.
- Coverage is limited to `write_file`/`create_file`, `AGENTS.md`, `.claude/settings.json`, and a literal `curl`/`wget` pipeline to `sh`/`bash`. Encoded payloads, patch formats, other tools, and other configuration formats need separate evaluation.
- Cautionary examples embedded in agent configuration can also match; the signature does not interpret instruction intent or parse the configuration format.
