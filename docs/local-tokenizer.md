# Optional local tokenizer

Quarry can count an explicitly rendered request with a pinned local tokenizer,
without making a provider request. Install `cisco-project-quarry[local-tokenizer]`
to use this adapter. The base installation and existing model identities remain
unchanged when local configuration is absent.

This count describes the configured tokenizer, template and framing. It is not
an attestation of a remote provider's private template or billable token count.
Choose and verify the artifact and template for the model revision you use.

## Configuration

For a count-only model, set both adapters to `local_tokenizer`. A model that also
performs inference keeps its normal inference adapter and uses `local_tokenizer`
in its `tokens` block. That model still needs its usual inference dependencies.

```yaml
config_version: 2
models:
  counter:
    adapter: local_tokenizer
    model: synthetic-counter
    revision: model-r1
    tokens:
      adapter: local_tokenizer
      extraction:
        count: count
      local:
        artifact_path: tokenizer.json
        # Replace with the actual SHA256 of these artifact bytes.
        artifact_sha256: "0000000000000000000000000000000000000000000000000000000000000000"
        model_revision: model-r1
        tokenizer_revision: tokenizer-r1
        template: >-
          {% for m in messages %}{{ m.role }}: {{ m.content }}
          {% endfor %}{% if tools %}{{ tools|tojson }}{% endif %}
        framing:
          per_request: 0
          per_message: 0
```

The local model revision must equal the containing model's `revision`.
`tokens.endpoint` and a separate `tokens.model` override are invalid for this
adapter. A relative artifact path is relative to the configuration directory.
Configuration parsing opens no artifact and imports neither optional library.
Only configured models needed by loaded rules receive a native adapter owner.

`model("counter").input_tokens(messages, tools)` uses this configuration. The
existing trim and split operations also use whole-request local counts. Local
counts use no provider rate-limit tickets or physical model calls. Their results
use a separate versioned response-cache namespace with the full count settings.

## Exact text encoding and framing

For SDK text encoding, construct `LocalTokenizerConfig` and use
`LocalTokenizerAdapter.encode_text(config, text)`. It returns immutable
`LocalTokenizerEncoding` with actual uint32 token IDs, original Unicode
code-point offsets and native special-token masks. Offsets are Python string
indices, not UTF8 byte indices. Byte-level tokens can share an original code
point, and normalization can leave source gaps. No complete source coverage is
invented.

`count_request(config, messages, tools)` renders the entire request once, encodes
that rendered text, then adds `per_request + per_message * len(messages)` once.
These finite nonnegative margins are scalar count additions; they do not become
fabricated token IDs or source offsets. `encode_text` does not add the margins.
Artifact padding, truncation, post-processors and decoders are disabled and are
never instantiated. No special endpoints are added implicitly. Put explicit
begin/end markers in the template or its declared variables when required.

Existing model-window hosts retain their explicit rendering, endpoint, context
and last-turn geometry contracts. Raw encoding does not invent a window layout
or make assumptions about a provider's token IDs.

## Supported artifacts and templates

The version-pinned native backend supports bounded JSON WordLevel, WordPiece,
BPE and Unigram artifacts. Unknown or unmodeled components fail unavailable;
there is no remote fallback. BPE dropout, precompiled normalizer programs and
user-supplied regex patterns are unsupported. Conservative admission bounds
cover normalizer/pretokenizer expansion, native token allocation and search
work before native encoding. A valid artifact or input can exceed these bounds;
that produces an explicit limit failure rather than a partial count.

Supported normalizers include bounded sequences of lowercase, Unicode
normalization, accent/whitespace stripping, Bert normalization, literal replace
and prepend. Supported pretokenizers include bounded sequences of whitespace,
Bert, Unicode scripts, digits, punctuation, character delimiters, literal split,
byte-level and metaspace operations. Artifact bytes, JSON nodes/nesting, stage
counts, vocabularies, added tokens and token content also have absolute caps.

Jinja parses template syntax only. Quarry interprets a restricted syntax tree;
it never compiles or executes template Python. Supported operations include
finite loops/conditionals, assignment, native dictionary/list access, limited
string/integer concatenation and explicit safe filters such as `tojson`,
`default`, `length`, `trim`, `lower`, `upper`, `replace`, `join` and `string`.
There are no template imports, includes, macros, arbitrary calls, private
attributes, mutation, multiplication or exponentiation. Unsupported syntax
fails unavailable. Node/depth/step/output limits apply to actual execution.

Declare up to 32 bounded string `template_variables`, plus explicit boolean
`add_generation_prompt`, `trim_blocks` and `lstrip_blocks` options as needed.
Templates receive only detached `messages`, `tools`, those variables and the
generation-prompt flag. A nonempty offered tool collection must be fully
serialized into actual output. Computing JSON in an unused variable or printing
only tool names does not satisfy this check. Structured tool-call/result/name
message fields likewise need full serialization. Supported message content is
native text or a structured call's null content. Unsupported media/content
parts and opaque objects fail input validation without coercion or omission.

## Ownership, limits and offline behavior

Snapshotting, whole-request rendering, artifact loading and native encoding run
off the caller's event loop. Each native adapter has at most 2 worker threads and
8 accepted jobs, including queued or cancelled jobs still owned by the executor.
Cancellation ends the wait; it does not declare native work finished or free its
slot. Closing rejects new jobs and lets accepted jobs reach actual termination.
Inherited fork owners reject access before touching parent mutexes.

Each worker retains at most 2 validated pinned artifacts. Eviction and worker
exit release them outside owner locks. Native BPE/Unigram word caches are
disabled; this artifact cache retains no request text, rendering, encoding or
count. A cached validated artifact remains pinned to its original bytes; use a
new digest/revision when changing the artifact. Reopening an evicted artifact
verifies its bytes again. The separate normal response cache follows the
configured token-count retention policy.

Loading uses only the explicitly configured regular local file. It performs no
network access, model discovery, cache-directory search or implicit download.
Supply a materialized local artifact for offline use. Symlinks are rejected
where the platform supplies no-follow open semantics. This feature changes no
global TLS or certificate behavior.

The configured wall allowance is checked before and after owned work and at
interpreter/input checkpoints. A thread cannot preempt an ongoing native call
or filesystem read; its resources stay owned until it returns. These finite
admission bounds are not an operating-system memory quota.

Errors retain native unavailable, invalid-input, invalid-output and limit
types. Scan failure categories are respectively `model_provider`,
`invalid_input`, `model_invalid_output` and `evaluation_limit`; an earlier scan
deadline keeps timeout precedence. Native exception messages/tracebacks do not
escape the worker. Public model identity omits artifact paths and hashes
template/variable text while retaining revision, digest and count semantics.
