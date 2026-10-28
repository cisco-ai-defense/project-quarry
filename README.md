# Project Quarry

Project Quarry is a Python framework for detecting threats and behavior patterns in agent conversations. Write rules over messages, tool calls, tool results, and event order. You can also call a configured model adapter from a rule when local checks are not enough.

Quarry parses and validates rules when you load them, then compiles each rule to a symbolic finite-state automaton. A scan runs that compiled program over the conversation. Rules that cannot compile fail during loading.

Quarry is in alpha, and its DSL syntax may change before v1.0.0 in ways that require updates to existing rules.

## Features

- Text, regular expressions (RE2), structured-field, and tool use checks
- Multi-turn flow rules, captures, counts, and typed policy parameters
- Optional configured model scoring, classification, completion, token counting, and spans
- Witness evidence, explain output, and typed collections for supported rules
- Sync and async Python APIs, plus human, JSON, and CSV CLI output
- Explicit input, compile, time, and model-call limits

## Install

Quarry supports Python 3.11 through 3.13.

**Setup:**

```bash
git clone https://github.com/cisco-ai-defense/project-quarry.git
cd project-quarry
uv sync --frozen
```

Run `uv sync --frozen --extra inference` if your application uses a built-in model provider.

## Quick start

Scan an example conversation from the repository root:

```bash
uv run --frozen quarry scan --config examples/config.yaml --target examples/data/benign.json
```

`examples/config.yaml` selects fourteen local rules that run without model credentials.
Four semantic rules use deterministic keyword stubs via an explicit config opt-in; these stubs are for examples and tests, not production decisions.

To see a match, scan `examples/data/tool_calls.json` with the same config. You can also target `examples/data/` to scan every conversation file; a successful scan with matches exits `1`.

The full example rule collection is exercised by `uv run --frozen python scripts/e2e_signature_test.py`, which supplies deterministic model and prober implementations in process.

A YAML configuration file loads your own rules:

```yaml
config_version: 2
rules:
  - rules/
```

For example, save this rule as `rules/credential-request.rule`:

```text
rule "Credential request"
strings:
    $terms = ["api key", "access token"] nocase
condition:
    any msg in user_messages where msg.content contains_any $terms
```

Pass an OpenAI-shaped conversation as JSON with a `messages` array. The CLI exits `0` for a complete clean scan, `1` for a complete match, and `2` when the scan is invalid or incomplete.

## Rule Structure

A `.rule` file contains one named rule. Quarry parses, validates, links, and compiles it when the scanner loads. The [generated language reference](quarry/language/reference.md) lists current functions, fields, operators, and runnable examples.

```text
rule "Suspicious request"
metadata:
    severity = "medium"
strings:
    $blocked = ["reveal a secret", "ignore the policy"] nocase
variables:
    has_blocked_text = any msg in user_messages where msg.content contains_any $blocked
condition:
    has_blocked_text
```

Optional blocks appear in this order: `metadata`, `strings`, `variables`, `parameters`. Every rule needs `condition:`. A model span rule can also have a `collect:` block after `condition:`. Use `//` for line comments; `#name` is the occurrence-count operator.

Use `=` to define values in `metadata:`, `strings:`, and `variables:`. Use `==` to compare values in `condition:` expressions, including predicates inside `matches { ... }`. Other comparisons remain `!=`, `<`, `<=`, `>`, and `>=`.

```text
rule "Shell call"
strings:
    $needle = "shell"
condition:
    any call in tool_calls where call.name == "shell"
```

## Python API

```python
from quarry import QuarryGuard

guard = QuarryGuard("config.yaml")
matches = guard.scan({"messages": [{"role": "user", "content": "Plan a picnic."}]})
for match in matches:
    print(match["rule"], match["metadata"])
```

`QuarryGuard` raises on incomplete scans. `DSLScanner` returns a `ScanResult` so you can inspect completion status, evidence, and typed outputs. `compile_policy` prepares one reusable parameterized rule.

## Documentation

- [Getting started](docs/getting-started.md)
- [How Quarry works](docs/how-quarry-works.md)
- [DSL guide](docs/dsl.md) and [generated language reference](quarry/language/reference.md)
- [Rule writing](docs/rule-writing.md) and [string matching](docs/string-matching.md)
- [Configuration](docs/configuration.md) and [message input](docs/messages.md)
- [Python API](docs/python-api.md) and [parameters](docs/parameters.md)
- [Model inference](docs/model-inference.md) and [application integration](docs/integration.md)
- [Agent plugin](docs/PLUGIN.md) for Codex, Claude Code, and MCP hosts
- [Architecture](docs/architecture.md) and [testing](docs/testing.md)
- [Synthetic examples](examples/README.md)

See [CONTRIBUTING.md](CONTRIBUTING.md) to work on Quarry. Report security issues through [SECURITY.md](SECURITY.md). Keep conversation data and credentials out of public issues.

## License

Distributed under the `Apache 2.0` License. See [LICENSE](LICENSE) for more
information.