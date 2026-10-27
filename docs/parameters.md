# Project Quarry: Parameters

A parameterized rule declares `number` or `text` values. Add `policy` to bind a value when you compile the policy; pass other values at scan time.

```text
rule "Policy threshold"
parameters:
    instructions: text policy
    threshold: number
condition:
    model("guard").score(messages, params.instructions) >= params.threshold
```

```python
from pathlib import Path
from quarry import ModelInferenceBackend, compile_policy
from quarry.config import QuarryConfig

source = Path("examples/parameters/policy.rule").read_text(encoding="utf-8")
config = QuarryConfig.from_yaml(Path("examples/parameters/config.yaml"))
backend = ModelInferenceBackend(config, adapters={"example": example_adapter})
policy = compile_policy(
    source,
    policy_parameters={"instructions": "Score credential requests as unsafe."},
    runtime_config=config,
    model_inference_backend=backend,
)
result = policy.scan(conversation, parameters={"threshold": 0.8})
result.raise_for_status()
```

`compile_policy` checks parameter names, types, bindings, and use. It compiles the rule once, and repeated `scan` or `scan_async` calls reuse the prepared policy. `scan_many` and `scan_many_async` preserve input order under a batch deadline.

Results include policy and parameter digests. Numeric parameter evidence can include its value; text evidence stays redacted and uses a keyed digest. Pass an `evidence_key` of at least 16 bytes to `compile_policy` when you need stable digests across prepared instances. Protect the key as an application secret. Keep sensitive text out of logs. Serialization rejects policies with bound text parameters or `text_ref` resources; compile those policies from source in the host process.

The [parameter example](../examples/parameters/) uses a local adapter and makes no network request.
