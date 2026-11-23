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

## Opt-in capped count proofs

An exact token count always encodes the complete rendered request. For a caller
that only needs to know whether a cap is exceeded, a verified source-width bound
can sometimes avoid encoding. This is an explicit opt-in, and a width supplied
by a caller is a claim requiring runtime attestation.

Construct a `TokenizerWidthClaim` with a positive native integer
`max_source_characters_per_token`, the artifact SHA256, model/tokenizer revisions
and `LocalTokenizerConfig.count_settings_sha256`. The supported method is
`bytelevel-bpe-complete-alphabet-v1`. The settings digest includes the template,
framing, limits, flags and tokenizer semantics; it omits the artifact's path.
Recompute the claim when any of these settings changes. Configuration can supply
the same fields in `tokens.source_width`, alongside `tokens.local`; remote
adapters, incomplete claims and mismatched settings are rejected during parsing.
Omitting `source_width` preserves prior configuration/cache/policy identities.

`LocalTokenizerAdapter.attest_width(config, claim)` loads the pinned artifact off
the caller loop and returns detached verified metadata, or `None` when the
claimed width cannot be proved. The verifier checks the actual pinned native
version and state. Its supported pipeline has no normalizer, one byte-level
pretokenizer, complete model coverage of all 256 byte symbols, unique dense
vocabulary IDs and concatenating BPE merges. Density is checked before native
serialization, whose work otherwise depends on the largest ID. Added literal tokens contribute their own
source width and must not consume surrounding whitespace. Unknown-token loss,
ID aliasing, destructive splitting, normalization, truncation, padding and
unmodeled settings cannot authorize a proof. Artifact padding/truncation and
post-processors remain disabled by the canonical local loader.

`count_request_capped(config, claim, messages, tools, cap=...)` returns an
immutable `CappedTokenCount`. `exact_count` contains the canonical count when
encoding was required. Otherwise it is `None`, and `lower_bound` is a proved
strict lower bound above `cap`, with an attestation fingerprint. A lower bound
must never be used as an exact count, subtracted from a budget, or used to pick
token coordinates. Unsupported or insufficient claims use canonical encoding.
The cap is a native integer from zero through `2**63 - 1`.

The capped path snapshots and renders every message and offered tool before
proving overflow. It preserves original ordering, structured field coverage,
input validation, work admission and framing. A proof also needs conservative
admission to guarantee the configured output-token limit; otherwise encoding
runs and preserves that limit failure. Each worker retains at most two bounded
attestation records, with no request text, encodings or native artifact references.

When configured, native trim fit comparisons may use capped counts. The whole
selected result still receives a canonical exact count before a cut is returned.
Implicit score requests with custom message builders, ordinary `input_tokens`,
split arithmetic and downstream model validation keep canonical counts. Cached
capped replies have their own method namespace and bind the actual adapter
lifetime, alias, revisions, full settings and cap. Complete native reply shape
and binding are checked before retention and again on reads. Token-count cache
retention policy still applies; fingerprints contain no request text or digest.

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
