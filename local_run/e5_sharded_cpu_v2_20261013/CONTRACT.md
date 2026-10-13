# Prospective CPU E5 shard execution contract, version 1

Frozen Phase 3–10 files remain unchanged. The measured single CPU pass is about
61.3 hours, exceeding the available 24-hour allocation. This new orchestration
version changes process boundaries and durable writes, not the numerical
document encoder. It does not claim that the frozen full-population cache
writer or its monolithic replay ran.

## Fixed inputs and numerical target

The exact passed source replay, tokenizer census, CPU budget result, original
ASCII source view, assets, runtime and frozen encoder bytes must match their
recorded hashes. All 140,250 prepared records retain their original objects,
text, source IDs, partition and order. All content is encoded. Each worker uses
the unchanged `LocalTransformer` and `encode_document`: one CPU thread, one
interop thread, FP32, eager attention, MKLDNN disabled, deterministic algorithms,
seed zero, 256 content tokens per chunk, padding width 261, and batches of at
most eight chunks within one document. There is no GPU or tolerance fallback.

Thirty-two contiguous, nonempty whole-document ranges are planned before full
encoding. Boundary weights use the passed budget's observed maximum forward
time per batch size, and each document's exact census batches. A document is
never split. All ranges must cover indices 0 through 140,249 exactly once,
without overlap, omission, reordering or additional records. The shape-weighted
estimates are planning scenarios, not guaranteed completion limits. A full
shard allocation is capped at eight hours; concurrency and launch require a
separate resource decision after qualification. No bulk jobs are auto-submitted.

## Qualification before full work

Use 19 unchanged natural calibration records chosen before forwards: the two
lowest fixed-salt record-ID hashes in each exact chunk-count bucket 1–8, then
one each at 9, 17 and the maximum calibration chunk count (currently 51).
Keep their original source order. A new private, explicitly engineering-only
subset receipt binds its derivation from the passed complete source frame and
inherits the already passed fixed guards; it is not a replacement scientific
population or a claim that original preparation was rerun on the subset.

In a fresh process, run the unmodified `encode_local` on that qualified
engineering subset to obtain the golden vectors and complete chunk log. In
separate fresh worker processes, encode four planned qualification ranges using
the exact document primitives. Assemble in original subset order and require
every FP32 byte and every row/token/chunk detail to equal the golden output.
Replay each shard in another fresh process and require exact vector bytes and
the complete ledger again. This gate uses 19 documents / 149 chunks per pass
and three logical passes, totaling 57 document encodings / 447 chunks /
84 forward batches. Check
the frozen Phase 6 acceptance import in the pinned CPU interpreter before any
forwards so missing optional import dependencies cannot surface after long work.

The qualification report must pass before a full worker accepts a full plan.
The full plan and its first accepted qualification bind the exact source,
code/package, assets and numerical runtime. Modified pins, plans, code, sources,
weights or runtime require a new qualification and new output directories.

## Completion, resumption and replay

A shard is reusable only as a completed immutable unit. Write vectors and the
row log in a newly created private directory, flush and fsync both, verify all
counts, token details and original inputs, bind output hashes, then atomically
publish a completion receipt last. No row-level resume exists. A crash,
timeout, missing receipt, mismatched receipt, changed file or partial directory
is not a completed shard; preserve it and use a new attempt directory. A replay
receipt also binds the exact completed shard and is written only after checking
every row from a fresh process. Existing passing shards may be reused only
under byte-identical inputs, plan, code, assets and numerical runtime, with
their entire output inventory rechecked. No zero-work reuse after pin changes.

Assembly requires all 32 completed shards and all 32 independent replay
receipts, exact ordered full-frame coverage, and exact schema/dtype/dimensions,
IDs, source text hashes, content-token hashes, chunk counts and costs. It writes
a new canonical full matrix and full ordered row ledger with a versioned
manifest. The output never uses the frozen cache schema to disguise the new
writer. Full-population replay under this new execution contract is recorded
separately from the original monolithic acceptance route. Downstream use needs
an explicit versioned adapter; a successful engineering cache does not satisfy
source authenticity, human calibration, or study-readiness gates.

## Accounting and failures

Record complete worker wall time, cold loading, forwards/encoding, validation,
durable output and final hashing costs, and peak Linux RSS. The full matrix
payload is 430,848,000 bytes; shard files and assembly coexist, and text,
tokenizer/model, activations, logs and page cache add memory/storage. Full
encoding plus full replay is 280,500 document encodings, 624,318 chunks and
295,650 forward batches. Failed attempts and uncertainty remain visible.
All external-source, original-route acceptance and confirmatory-readiness flags
remain false. No paid compute, download, source mutation or silent fallback is
part of this package.
