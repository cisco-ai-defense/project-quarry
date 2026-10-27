# Agent drift examples

These five rules illustrate agent behavior checks that use tool capabilities,
task intent, topic drift, or model scores. Each rule has a positive and negative
conversation under `data/`. The deterministic example gate loads every rule and
checks both conversations without external services:

```console
python scripts/e2e_signature_test.py
```

The gate opts into Quarry's simple prober stubs and a fake model adapter. For
real detections, configure the appropriate prober and model backends before
loading these rules into a scanner.
