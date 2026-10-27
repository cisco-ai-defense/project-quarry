# Project Quarry examples

The `rules/` directory contains version 2 rules that can be loaded together by a
general scanner. `policy-only/` contains parameterized rules that require
`CompiledPolicy`; `legacy-v1/` contains the version 1 split compatibility example.
Each rule has a matching dataset in `data/`.

The corpus currently has 77 rule and positive-dataset pairs, 59 negative
datasets, and two benign controls. Each rule exercises a specific public DSL
feature. The two Quarry-only examples remain alongside the 75 examples ported
from the full engine.

## Example contract

Every rule in these three directories has exactly one matching conversation at
`data/<name>.json`. `scripts/e2e_signature_test.py` loads and executes every rule,
requires it to match its paired dataset, and requires it to reject any optional
`data/<name>.negative.json` and both benign control datasets. The pairing check
also rejects orphan rules and datasets. The gate checks that split examples
score multiple chunks and that the span example collects the expected email.
It also loads the shared rules through `all-rules.yaml` and completes a normal scan
with deterministic model resources.

Each full-engine vocabulary rule's `metadata.feature` identifies the public DSL item it primarily
exercises. The examples cover deterministic core constructs, agent-drift probers, model
classification/scoring/completion/token/split/span operations, compatibility inference
forms, and policy/scan parameter binding.

The example gate uses a deterministic in-process model adapter and explicit prober
stubs. It performs no network calls and needs no credentials. `examples/config.yaml`
loads fourteen local rules for credential-free CLI scans, explicitly opting into
deterministic keyword stubs for four semantic rules. These stubs are for examples and
tests, not production decisions. `examples/all-rules.yaml` selects
the full `rules/` directory for the gate; a CLI scan using it also needs configured
model aliases, the `policy` text resource, and prober backends. Runner-only version
and parameter settings live in `runner-options.yaml`, so every file in `data/` is a
valid CLI conversation.

The five rules in `drift/` have their own positive and negative
datasets. The separate `gemma4/` example has a model-provider configuration
for an optional live run. The gate runs all six with in-process fake resources;
it does not contact the configured model endpoint.

Run the complete example gate with:

```console
uv run --frozen python scripts/e2e_signature_test.py
```

`corpus-manifest.json` is the reviewed inventory of every file under
`examples/`. The gate compares that fixed list with the
checkout before running the rules. This catches a rule and its dataset being
removed together, which pairing checks alone cannot catch. When adding,
removing, or renaming an example intentionally, update the sorted path lists in
the manifest in the same change and review the inventory diff.

For a credential-free CLI scan of fourteen local examples:

```console
uv run --frozen quarry scan -c examples/config.yaml -t examples/data/benign.json
uv run --frozen quarry scan -c examples/config.yaml -t examples/data/tool_calls.json
```

The first command exits `0` (clean); the second exits `1` (match). A directory
scan with `-t examples/data/` also completes and exits `1` because it finds matches.
`examples/smoke.yaml` remains a minimal one-rule option.

## More examples

[`parameters/`](parameters/) shows policy-bound and scan-bound values.
[`model-inference/`](model-inference/) shows how to register a model adapter.
Both use a local adapter and require no credentials. The generated
[language examples](../quarry/language/examples/) cover more compiler forms and
link to the [language reference](../quarry/language/reference.md).
