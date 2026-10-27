# Optional Gemma 4 example

`prompt_injection.rule` combines a regex prefilter with a model score against
`conversation.json`. The deterministic example gate executes the rule with an
in-process fake model; it requires no credentials or network access.

`config.yaml` is an optional live provider example. A live scan requires the
credentials and endpoint access named in that file:

```console
quarry scan -c examples/gemma4/config.yaml -t examples/gemma4/conversation.json
```

The live provider result may differ from the deterministic gate result.
