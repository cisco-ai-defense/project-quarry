# Registered extensions

Quarry can evaluate facts that do not come from an agent conversation. An application registers an `ExtensionSpec` with Pydantic schemas when it creates a scanner. Its adapter then supplies JSON events and, if needed, JSON state to each scan. The same interface can describe memory writes, sandbox decisions, approvals, retrieval results, or another domain. The names `memory` and `sandbox` below are examples, not built-in Quarry objects.

Rule text cannot load Python packages. The application chooses which schemas and adapters it trusts. Quarry checks registered names and fields when loading rules, and validates each event and state object before evaluating them. The DSL uses its existing dotted field access and `any ... in ... where ...` syntax; extensions do not need new grammar or primitive types.

This change advances the compiler version from `compiler-7` to `compiler-8`. Saved compiled policies from `compiler-7` cannot be loaded by this version; recompile their original rule source before scanning. There is no migration of the saved artifact itself.

## Register a domain

```python
from pydantic import BaseModel, ConfigDict
from quarry import DSLScanner, ExtensionSpec
from quarry.config import QuarryConfig


class WriteCommitted(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    key: str
    source_tool: str | None = None


memory = ExtensionSpec("memory", {"write_committed": WriteCommitted})
scanner = DSLScanner(
    QuarryConfig(
        config_version=2,
        quarry_rule_texts=(
            '''rule "Tool wrote memory"
condition:
    any write in memory.write_committed where write.payload.source_tool == "tool_a"''',
        ),
    ),
    extensions=[memory],
)
```

The event payload schema says what fields a rule may read. Use `extra="forbid"` on every nested model too, so misspelled input fields fail validation. A nullable field is still required unless it has a default: `str | None = None` lets the adapter omit `source_tool`, and a rule sees `null`. Set `validate_default=True` on every model or field with defaults, including factories, so field validators check omitted values when used. Quarry also checks static default types when registering a schema. Incoming JSON may use a Pydantic field alias; rules use the Python field name.

An event has a unique nonempty `id`, a registered `type` in the form `namespace.event_name`, a required object `payload`, and an optional `run_id`:

```python
event = {
    "id": "write-1",
    "type": "memory.write_committed",
    "run_id": "run-1",
    "payload": {"key": "preference", "source_tool": "tool_a"},
}
result = scanner.scan_data(None, events=[event])
result.raise_for_status()
```

The adapter should emit an event only after the underlying action has reached the state its event name promises. `ExtensionSpec` validates shape; adapter owners define and version the meaning of an event. The host selects facts for the current scan and keeps different runs apart. Quarry exposes `run_id` to rules but does not match it to a conversation automatically.

## Supply state

An extension can also register one state schema. A sandbox adapter, for example, can supply the runtime's own access decision and policy mode:

```python
from typing import Literal


class FileRead(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    allowed: bool


class SandboxPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    mode: Literal["observe", "enforce"] = "enforce"


sandbox = ExtensionSpec("sandbox", {"file_read": FileRead}, state=SandboxPolicy)
```

```text
rule "Sandbox denied file read"
condition:
    any read in sandbox.file_read where
        read.payload.allowed == false and sandbox.state.mode == "enforce"
```

The host passes `events=[...]` and `extension_state={"sandbox": {}}`; `mode` then uses the schema default. Omitted state is unknown, even if all its fields have defaults. State is static during one scan and takes no position in an event flow. The adapter supplies the authoritative `allowed` decision; a rule should not infer access from a path prefix.

State fields support direct scalar checks. `len`, `count`, and model or external functions over a state field are rejected when the rule loads; those inputs need an explicit domain definition before they can have reliable semantics.

## Order conversation and extension events

For `before`, `after`, `right_before`, `right_after`, and `within` across sources, pass an explicit `timeline`. It must list every conversation message, tool call, and extension event once, and preserve the conversation's original order:

```python
timeline = [
    {"message_index": 0},
    {"tool_call_id": "call-7"},
    {"event_id": "write-1"},
]
result = scanner.scan_data(conversation, events=[event], timeline=timeline)
result.raise_for_status()
```

An omitted manifest makes a possible cross-source flow incomplete; a malformed manifest makes the scan invalid. A conversation-only flow uses the conversation clock, even if unrelated extension events are present. A cross-source flow uses the shared timeline, where every intervening item counts toward `within`. An unbounded extension-only flow can use event list order. A bounded flow such as `right_after` needs a supplied conversation, even if it is explicitly empty, because conversation items could fall between its two events. It also needs complete event feeds for every registered namespace, since an unobserved namespace could have an intervening event. A mixed `until` that checks a static condition at each step has the same requirement. If the supplied conversation has items, pass a manifest to establish their order. A supplied manifest takes precedence.

The host can pass `event_feeds` to name the registered namespaces whose event feeds it observed completely, including those with zero events. It must supply all events for each named namespace. For a scanner registered with both `memory` and `sandbox` (separate from the memory-only scanner above):

```python
combined_scanner = DSLScanner(
    QuarryConfig(
        config_version=2,
        quarry_rule_texts=('rule "No sandbox reads"\ncondition: count(sandbox.file_read) == 0',),
    ),
    extensions=[memory, sandbox],
)
result = combined_scanner.scan_data(
    None,
    events=[event],  # The memory event defined above.
    event_feeds={"memory", "sandbox"},  # Sandbox was observed but had no events.
)
```

Without `event_feeds`, `events=None` means no feed was observed; `events=[]` keeps the legacy meaning that all registered feeds were observed and empty; and a nonempty `events` list marks only the namespaces represented by those events as observed. Thus `events=[event]` alone says nothing about an empty sandbox feed. Likewise, `conversation=None` means the host did not supply a conversation. Rules that need an unobserved source become incomplete. When scanning just one domain, select its rules with `rules_subset`. Check `ScanResult.raise_for_status()` before acting on matches. Invalid event types, duplicate IDs, malformed payloads, or bad state fail the scan.
