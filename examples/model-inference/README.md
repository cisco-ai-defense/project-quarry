# Project Quarry model inference example

`classify.rule` calls the model alias from `config.example.yaml`. `run.py` registers a local adapter
that returns a fixed classifier response.

Run the example from the repository root:

```bash
uv run --frozen python examples/model-inference/run.py
```

Replace the example adapter with your provider adapter and update the model configuration for your
application. Keep provider keys out of configuration files and logs.
