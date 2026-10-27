# Project Quarry: Testing

Create a development environment and run the public checks:

```bash
uv sync --frozen --group dev --extra inference
make ci
```

Focused commands include:

```bash
uv run --frozen --extra inference pytest tests/engine
uv run --frozen --extra inference pytest tests/language
uv run --frozen --extra inference python scripts/e2e_signature_test.py
uv run --frozen --extra inference python examples/model-inference/run.py
uv run --frozen --extra inference python examples/parameters/run.py
```

To check rules without reading a target conversation, run
`quarry scan -c config.yaml -t ignored.json --validate-only`. The command prints
performance and detection-quality warnings. It exits `0` with no warnings or
only medium detection-quality advisories, `1` for a performance warning or a
high detection-quality warning, and `2` if rules cannot load.

The engine tests compare compiled verdicts with the frozen legacy corpus and direct semantic properties. Language tests regenerate the machine-readable catalog, Markdown reference, and packaged examples. Add a matching and a clean synthetic conversation for each new public example. Add parser, compiler, safety, and invalid-input checks for a language change.

Before a release, run `make build`, install the wheel and source distribution outside the checkout, run `pip check`, import `quarry`, invoke `quarry --help`, and scan a local example. CI runs quality checks and the test suite across the advertised Python versions. Provider tests use local adapters or a loopback HTTP server; they do not call external services.

The [generated language reference](../quarry/language/reference.md) is built from runtime annotations and must stay in sync with the code. Run `make language-generate` after changing its source metadata, then review the resulting diff.
