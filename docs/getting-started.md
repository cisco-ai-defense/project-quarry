# Project Quarry: Getting started

Quarry supports Python 3.11, 3.12, and 3.13. Clone the repository and install the reviewed dependencies with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/cisco-ai-defense/project-quarry.git
cd project-quarry
uv sync --frozen
```

## Run a sample scan

```bash
uv run --frozen quarry scan --config examples/config.yaml --target examples/data/benign.json
```

`examples/config.yaml` loads fourteen local rules that need no model credentials.
Its four semantic rules use deterministic keyword stubs through an explicit opt-in.
These stubs are for examples and tests, not production decisions.
Run the same command with `examples/data/tool_calls.json` to see a match,
or with `examples/data/` to scan the full conversation directory. Every JSON file
there is valid CLI input. The complete example rule collection, including model
and agent-drift rules, runs through
`uv run --frozen python scripts/e2e_signature_test.py` with deterministic in-process
backends. `benign.json` and `benign_multiturn.json` are clean controls.

The command exits `0` for a complete scan without matches, `1` for a complete scan with matches, and `2` for invalid input or an incomplete scan. Run `quarry scan --help` for directory, batch, explain, and output options.

## Write a rule

Save this as `rules/credential-request.rule`:

```text
rule "Credential request"
metadata:
    severity = "high"
strings:
    $terms = ["api key", "password", "access token"] nocase
condition:
    any msg in user_messages where msg.content contains_any $terms
```

Create `config.yaml`:

```yaml
config_version: 2
rules:
  - rules/
```

Then run `uv run --frozen quarry scan --config config.yaml --target conversation.json`. Quarry parses, validates, and compiles the rule when the scanner loads. Fix a rejected rule before serving scans.

Continue with the [DSL guide](dsl.md), [configuration](configuration.md), and the [generated language reference](../quarry/language/reference.md).
