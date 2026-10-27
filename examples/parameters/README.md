# Project Quarry parameterized policy example

`policy.rule` declares a policy-bound instruction and a scan-bound threshold. `run.py` passes the
instruction to `compile_policy`, then provides the threshold for one scan.

Run the example from the repository root:

```bash
uv run --frozen python examples/parameters/run.py
```

The example adapter returns a fixed score and makes no network request.
