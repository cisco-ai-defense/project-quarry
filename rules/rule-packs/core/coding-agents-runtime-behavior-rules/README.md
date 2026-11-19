# Coding agents runtime behavior rules

Experimental core pack with one example rule:
download-to-shell text written to agent configuration (`rules/rule-packs/core/coding-agents-runtime-behavior-rules/configuration_persistence_fetch_execute_content_write/configuration_persistence_fetch_execute_content_write.rule`).

The rule checks `write_file` and `create_file` calls targeting `AGENTS.md` or
`.claude/settings.json` for a `curl`/`wget` pipeline to `sh`/`bash`. A match is a
review lead; it does not establish execution, persistence, or missing approval.
See the changelog (`rules/rule-packs/core/coding-agents-runtime-behavior-rules/CHANGELOG.md`) for coverage limits.

Synthetic validation cases stay outside the pack:

- Positive example (`rules/test-cases/core/coding-agents-runtime-behavior-rules/configuration_persistence_fetch_execute_content_write/positive/agents_download_shell.json`)
- Negative example (`rules/test-cases/core/coding-agents-runtime-behavior-rules/configuration_persistence_fetch_execute_content_write/negative/download_without_execution.json`)
- All expected results (`rules/test-cases/core/coding-agents-runtime-behavior-rules/configuration_persistence_fetch_execute_content_write/expectations.yaml`): 3 positive and 5 negative cases.

From the repository root, run `uv run --frozen python rules/validate.py` to validate the corpus.
