# Changelog

## 0.1.0 — Unreleased

- Add an experimental review signature for download-to-shell text written to agent configuration, with synthetic matching and near-miss cases.
- The signature checks named tool calls and their path/content fields. It does not prove that configuration was consumed, that code executed, or that user authorization was missing.
- Coverage is limited to `write_file`/`create_file`, `AGENTS.md`, `.claude/settings.json`, and a literal `curl`/`wget` pipeline to `sh`/`bash`. Encoded payloads, patch formats, other tools, and other configuration formats need separate evaluation.
- Cautionary examples embedded in agent configuration can also match; the signature does not interpret instruction intent or parse the configuration format.
