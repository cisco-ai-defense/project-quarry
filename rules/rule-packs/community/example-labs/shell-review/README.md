# Shell review example pack

Synthetic community pack with one experimental example rule:
download-to-shell command submitted to a shell tool (`rules/rule-packs/community/example-labs/shell-review/code_execution_download_to_shell_call/code_execution_download_to_shell_call.rule`).
The publisher metadata (`rules/rule-packs/community/example-labs/publisher.yaml`) illustrates a contributor namespace;
Example Labs is a placeholder publisher.

The rule checks the `command` argument of `execute_shell`, `run_shell`, and
`shell` calls for a `curl`/`wget` pipeline to `sh`/`bash`. It detects submitted
command text; it does not prove execution or missing approval. See the
changelog (`rules/rule-packs/community/example-labs/shell-review/CHANGELOG.md`) for coverage limits.

Synthetic validation cases stay outside the pack:

- Positive example (`rules/test-cases/community/example-labs/shell-review/code_execution_download_to_shell_call/positive/curl_shell_command.json`)
- Negative example (`rules/test-cases/community/example-labs/shell-review/code_execution_download_to_shell_call/negative/shell_download_for_review.json`)
- All expected results (`rules/test-cases/community/example-labs/shell-review/code_execution_download_to_shell_call/expectations.yaml`): 2 positive and 4 negative cases.

From the repository root, run `uv run --frozen python rules/validate.py` to validate the corpus.
