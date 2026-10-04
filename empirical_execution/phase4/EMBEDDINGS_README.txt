OFFLINE SEMANTIC ENCODING — IMPLEMENTATION READY; REAL MODEL RUN STILL BLOCKED
===========================================================================

Purpose
-------
embeddings.py encodes an unchanged phase3 prepared population with either
intfloat/multilingual-e5-base or sentence-transformers/all-mpnet-base-v2.
The output is a complete ordered N x 768 FP32 .npy memory map, an aligned ID
list, per-document chunk logs, execution summary, and sealed cache manifest.
The manifest binds the full input rows (including source/partition fields),
normalized text/ID sequence, preparation audit, local model/tokenizer bytes,
code, runtime versions and resulting cache bytes.

Current status
--------------
No real model forward has been run here. torch, transformers, tokenizers and
safetensors are absent; original pinned model/tokenizer assets are absent.
check_embeddings.py exercises explicit nonlinguistic software fixtures only.
Its numerical fixture arrays and fake tokenizer are not empirical datasets or
semantic embeddings. Production encoding does not accept this fixture backend.
No runtime install, network download, remote inference or GPU job occurs.

Representation lock (prospective, before confirmatory outcomes)
-------------------------------------------------------------
The original protocol specifies all chunks of at most 256 content tokens,
E5 query prefix, content-weighted pooling, document normalization and FP32
storage. This implementation removes remaining operational ambiguity:

1. Input texts must already be normalized by the fixed corpus parser and
   phase3 preparation. Title/body concatenation is a parser responsibility;
   this module never changes text, labels, source IDs or partition membership.
2. Tokenize the complete text once without prefix or model special tokens and
   without truncation. Split the resulting IDs into consecutive chunks of at
   most 256 content tokens. No decode/re-tokenize operation is introduced.
3. Tokenize E5's literal query: prefix separately and prepend those IDs to each
   content chunk. MPNet has no prefix. Add the tokenizer's model special tokens.
   This exact boundary convention is a declared study choice; separately
   tokenized prefix/content is not asserted to equal concatenated-string
   tokenization for every possible tokenizer. Prefix/special counts are logged.
   Refuse a model/tokenizer capacity that cannot fit 256 content tokens plus
   the overhead. No later text is dropped and chunk size is never silently cut.
4. On CPU FP32, obtain the attention-mask mean of the final hidden states,
   including attended prefix and special tokens and excluding padding. This
   follows the pooling convention in the official model cards. Normalize each
   chunk vector with a coordinate-ordered FP64 L2 calculation.
5. Form the FP64 content-token-count-weighted mean in chunk order; prefixes and
   special tokens do not increase these document weights. Normalize the document
   using the same FP64 rule, then cast once to FP32 for storage.
6. Batches include chunks of one document only. Pad each chunk to the fixed
   256-content-plus-prefix-and-special length. Deleting another document cannot
   change this document's token IDs, chunk grouping, batch shape or padding.
   Pin the intradocument batch size (default 8), thread count (default 1), seed,
   eager attention, disabled MKLDNN and deterministic torch algorithms.
7. Learners use the stored FP32 vector promoted exactly to FP64, with no second
   normalization. Graph scoring uses phase3's separately specified FP64
   normalization and coordinate-ordered cosine reference. Never conflate them.

Model/library/CPU changes may alter output bits. The computed cache hash is the
frozen experiment object; this module does not claim universal cross-machine
bit identity or real-arithmetic score certification. It also does not certify
publisher provenance merely because an arbitrary revision string was supplied.

Local asset manifest
--------------------
Use actual independently sourced immutable model/tokenizer revisions. Materialize
regular local files (not symlinks), and provide exact installed library versions
as a JSON object with numpy, torch, transformers, tokenizers and safetensors keys.
Do not use invented version/revision values. Output the manifest outside both
asset directories, because the inventory binds every file inside those directories.

python empirical_execution/phase4/embeddings.py inventory \
  --model-dir /local/e5-model --tokenizer-dir /local/e5-tokenizer \
  --encoder-id intfloat/multilingual-e5-base \
  --encoder-revision ACTUAL_FULL_COMMIT_SHA \
  --tokenizer-revision ACTUAL_FULL_COMMIT_SHA \
  --versions /local/exact-runtime-versions.json \
  --out /local/e5-assets.json

The inventory and actual runtime must match at execution. The model needs
config.json plus safetensors weights. Custom .py/.pyc files, auto_map custom
implementations, external shard/config paths, changed or unlisted files, and
pickle fallback are refused. trust_remote_code=False and local_files_only=True
are set on both loaders; offline environment flags are set before imports.
Byte pinning proves internal reproducibility, not a claimed public origin.

Encoding
--------
Run in a fresh process so CPU thread configuration can be pinned. The output
directory must not exist. Explicitly declare the evidence role; the safe default
is engineering_nonconfirmatory. Even confirmatory_calibration is a declared
scope, not a passed scientific or provenance gate.

python empirical_execution/phase4/embeddings.py encode \
  --records /local/prepared/records.jsonl \
  --preparation-audit /local/prepared/audit.json \
  --asset-manifest /local/e5-assets.json \
  --model-dir /local/e5-model --tokenizer-dir /local/e5-tokenizer \
  --batch-size 8 --threads 1 \
  --evidence-role confirmatory_calibration --out /local/e5-cache

Python API:
encode_local(records_path, preparation_audit_path, asset_manifest_path,
             model_dir, tokenizer_dir, out_dir, batch_size=8, threads=1,
             evidence_role="engineering_nonconfirmatory")

Returns a sealed cache manifest only after all rows complete and input/model
hashes are rechecked. An interrupted/failed partial cache has no successful
manifest. Failures during encoding write FAILED.json and are re-raised.

For ordinary phase3 prepared populations, the E5 manifest is compatible with
phase3/run_preparation.py cache loading. Versioned News preparation has its own
ccu-news-preparation-1 audit: 2017 calibration-domain and 2018–19 training-domain
cohorts preserve the original source hash assignment, and test domains remain
reserved. The encoder verifies this distinct audit, dates, cohort roles, source
derivations and exact-text guard without pretending phase3 produced its cohort.
Use the phase4 calibration and guard CLI for this News preparation.
MPNet is also encoded over the identical complete prepared rows and later uses
the same E5-guarded study population. Its provenance encoder_role is robustness.
MPNet requires its own calibration threshold for its curation arm; do not feed
the E5 threshold to MPNet or apply a second MPNet population filter. The frozen
phase3 primary CLI deliberately accepts E5 only for confirmatory operations.
The new phase4/calibrate_cache.py CLI supports both encoders through the same
unchanged phase3 calibration core, with separate encoder-specific frames.

Per-encoder calibration commands
--------------------------------
Run selection independently for E5 and MPNet, using each encoder's full cache
over the same complete prepared population. Only calibration-partition vectors
are gathered; the source/full-row bindings remain part of the sampling frame.

python empirical_execution/phase4/calibrate_cache.py selection \
  --records /local/prepared/records.jsonl \
  --preparation-audit /local/prepared/audit.json \
  --vectors /local/e5-cache/vectors.npy \
  --cache-manifest /local/e5-cache/cache_manifest.json \
  --asset-manifest /local/e5-assets.json --out /local/e5-selection

After genuine independent blinded human collection (the tool never supplies it):

python empirical_execution/phase4/calibrate_cache.py threshold \
  --manifest /local/e5-selection/private_sampling_manifest.json \
  --responses /local/e5-selection/completed_responses.json \
  --out /local/e5-threshold-lock.json

python empirical_execution/phase4/calibrate_cache.py validation \
  --records /local/prepared/records.jsonl \
  --preparation-audit /local/prepared/audit.json \
  --vectors /local/e5-cache/vectors.npy \
  --cache-manifest /local/e5-cache/cache_manifest.json \
  --asset-manifest /local/e5-assets.json \
  --lock /local/e5-threshold-lock.json --out /local/e5-validation

After fresh independent blinded validation collection:

python empirical_execution/phase4/calibrate_cache.py quality \
  --manifest /local/e5-validation/private_sampling_manifest.json \
  --responses /local/e5-validation/completed_responses.json \
  --lock /local/e5-threshold-lock.json --out /local/e5-quality.json

Use an independent MPNet cache/selection/lock/validation/output path for MPNet.
Cross-encoder locks and quality frames are refused. Validation reuses the locked
scoring configuration; its block size is not a new tuning parameter. Existing
output paths are never overwritten. Blank responses fail before a threshold or
quality report is created. A diagnostic selection failure remains a failure;
the core's statistical report always leaves confirmatory_study_ready false.

After E5 quality passes, apply the population's single semantic guard:

python empirical_execution/phase4/calibrate_cache.py guard \
  --records /local/prepared/records.jsonl \
  --preparation-audit /local/prepared/audit.json \
  --vectors /local/e5-cache/vectors.npy \
  --cache-manifest /local/e5-cache/cache_manifest.json \
  --asset-manifest /local/e5-assets.json \
  --lock /local/e5-threshold-lock.json --quality /local/e5-quality.json \
  --out /local/guarded-population

The guard validates the exact calibration frame and threshold/quality bindings,
then uses the existing shared E5 semantic_guard. MPNet is refused here. Output
contains records.jsonl, a sealed semantic_guard_done_external_review_pending
audit, and retained_original_row_indices.json. The sealed audit's
retained_original_row_indices_sha256 is cal.digest(parsed_integer_index_list),
not the raw index-file checksum; verify it before gathering. The corresponding
retained_record_ids_sha256 uses cal.digest(the ordered retained record IDs).
Apply the same original row-index subset to both encoders' caches and verify
that the original IDs at those indices equal the retained IDs. Whole-source panel selection remains a later
step; News calendar selection uses only the final guarded analysis records and
its independent declared extraction-coverage evidence. This guard preserves
calibration/test records and only removes E5-linked training records. Its passed
statistical input does not authenticate human authorship or corpus provenance.

Cache reuse needs NumPy, not the original transformer runtime or model bytes.
It checks the sealed asset inventory declarations and matching revisions/library
versions, encoding logs, complete IDs/source/text/file bytes, prepared audit,
and current frozen derivation code. This verifies internal consistency only;
asset origin and actual human work still require independent provenance review.

Digest convention: source_cache_manifest_sha256 is cache_manifest["sha256"],
the SHA256 of canonical UTF-8 JSON with the top-level sha256 field removed,
sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False.
It is not the raw JSON file checksum. Other *_file_sha256 fields hash raw bytes.

Run the per-encoder integration/rejection checks:
  python empirical_execution/phase4/check_calibrate_cache.py
These use explicitly labeled temporary software fixtures, including a diagnostic
numerical lock to exercise validation. They provide no real model embeddings,
completed human judgments, chosen semantic threshold or passed semantic gate.

Accounting and outstanding checks
---------------------------------
The output cache is memory-mapped, but total memory is not constant. Full input
records, tokenizer/model, complete token IDs and chunk lists for the longest
document, one intradocument batch and transformer activations remain resident.
Resident mapped pages and OS caches also count. The summary records model bytes,
output bytes, process high-water RSS, elapsed time, per-document content/chunk
lengths, and both the 256-content first-chunk and model-capacity first-window
truncation fractions. RSS is a process high-water mark, not isolated stage RSS.

Run software checks:
  python empirical_execution/phase4/check_embeddings.py

After real local assets/runtime arrive, run a small actual-model acceptance test
first: repeated encoding, interleaved-document replay, full row/cache checks,
padding checks, and a manually inspected long-document token/chunk ledger. These
checks remain outstanding; the fixture audit cannot substitute for them.

Primary sources consulted 2026-10-04
-----------------------------------
https://huggingface.co/intfloat/multilingual-e5-base
  768 dimensions, query prefix for symmetric/nonretrieval tasks, attended-token
  mean pooling then normalization; library-version-dependent numeric outputs.
https://huggingface.co/sentence-transformers/all-mpnet-base-v2
  768 dimensions, attention-mask mean pooling and normalization.
The all-chunk document rule above is this study's declared design, not a claim
that either card's default inference already implements full-document pooling.
