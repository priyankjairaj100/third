THIRD LLM-only selection requests v2 — 2026-10-14

Correction and preserved predecessor
V1 was sealed but never staged or executed. Prelaunch review caught its use of raw
UTF-8 template SHA instead of the qualified pilot canonical-JSON-string SHA. V2
corrects that comparison only, adds a compact hash-bound actual Qwen and Mistral template
regression fixture, and retains all request/result schemas, source/sample/seed
identities, limits and runtime policy. V1 remains byte-identical. No model or
scientific result is replaced. Template identity now calls
pilot.digest_bytes(pilot.canonical(tokenizer.chat_template)) exactly.

Purpose
This package prepares the actual selected pairs for the two fixed judges. It reads
the completed full selection result and its independently supplied review, exact
sampling lock, current frame/source and qualified tokenizer assets. It produces
full-text references and a CPU token census. It does not sample again, draw entropy,
load weights, call a model, create an activation lock, select a threshold or satisfy
a semantic/human/scientific gate. Human annotation is not an active dependency.

The compact tokenizer_templates.fixture.json contains only exact template strings
extracted from already-local source configurations, their source hashes and expected
canonical/raw template hashes. Full tokenizer configurations stay private.

The package does not accept a bounded engineering result as full selection. The
actual completed result and independent review must be given with externally
reviewed RAW SHA256 values at execution time. No production result/review hashes
are invented in this package. Sampling lock raw bytes are already fixed as
52823fc3cbfcc358f58dab29a21b5d01746bd0dbc8cd90c232113cae4c8d39b7 (7535bytes).

Checked inputs and source scope
The pin file binds the sealed v3 protocol, historical39-file snapshot, current
20996-row calibration frame, sampler, sampling-lock helper, unchanged original
scorer/panels and both qualified pilot renderers. Historical evidence remains
separate from the executing checkout. The actual executing frozen modules and
renderer source files are hashed before import and again before publication.

The full 140250-row source file, source audit and frame payloads are freshly hashed
in Slurm. Every calibration record is checked against its exact original full-row
position, record ID, source unit/kind, partition and text SHA. Physical LF delimits
JSONL; Unicode U+2028/U+2029/U+0085 inside strings are preserved. Only selected unique
texts are retained in output. Missing, duplicated, reordered or altered mappings
fail the whole preparation. Same-source calibration pairs remain eligible.

Every retained sampled score is independently recalculated using the frozen
coordinate-ordered pairwise reference on actual frame vectors. Exact decorated
pair identities, seeds, inclusion probabilities/weights, all20 bin sample counts,
the full220405510 weighted denominator, and committed swap reservoir are checked.
This is not a second full220M score/bin or selection-reservoir replay. The passed
full-selection metadata review has that explicit limitation; its optional retained
payload replay may be absent because this preparer performs that check itself.

Current byte/identity evidence is assessed-development provenance. It does not
authenticate the publisher, establish deleted/private completeness, pass original
phase6 source/cache acceptance, or establish semantic relevance. Those false claim
flags are explicit in the result.

Exact rendering and CPU environment
Use the existing qualified judge environment, not encoder-cpu:
  WORKSPACE/envs/judge-qwen-v3-20261011/bin/python
Python3.13.7 and the full qualified package versions in expectations.json are
required. Linux Slurm on csis.cn1 is required BEFORE heavy input hashing/imports.
Required environment:
  CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
The CUDA-enabled torch distribution is imported solely to make CPU token tensors,
matching the exact qualified tokenize_call's return_tensors='pt'. CUDA stays
uninitialized, no model object is constructed and no weight shard is opened.
No new downloads, dependency installation, tokenizer fixes or environment changes
are performed. The driver's source contains no AutoModel/load_model/generate call.

Both unchanged pilot tokenize_call functions are imported under unique names and
called directly. Only the system prompt, fixed task and full A/B texts enter the
rendered prompt. Scores, IDs, tags, selection strata, historical judgments, sibling
model responses and desired answers are never rendered. json.dumps(...,
ensure_ascii=False) keeps the original renderer's spaces/key order, followed by
the exact native chat template and untruncated tokenizer. The returned input row
SHA uses the qualified pilot's canonical JSON (ensure_ascii=True, sorted keys,
compact separators). The result records that convention explicitly.

Tokenizer/config files are fully rehashed and backend JSON, exact class, template,
vocabulary size/length and special IDs must match the qualified pilot. Complete
model snapshot file inventory is checked, but weight-byte hashes are deliberately
not recomputed. Each existing bound weight manifest is hashed. Standard HF snapshot
symlinks are permitted for tokenizer files only through their fully resolved,
hash-bound targets, whose resolution and bytes are rechecked at finalization.
No weight loading or weight-byte verification claim is made.

Request and output contract
Success contains exactly8 files:
  texts.private.jsonl — each selected text stored once, with calibration/full-row
    indices, record_id, text_sha256 and exact text.
  requests.private.jsonl — shared request metadata: request_index, selection_index,
    stage selection|selection_swap, orientation canonical|swapped_diagnostic,
    diagnostic_only, opaque/natural pair IDs, shown calibration indices, shown
    record IDs/text hashes, and model_seeds keyed by exact model ID.
  sample.private.jsonl — unchanged canonical sampled rows and their full weights.
  qwen_tokens.private.jsonl and mistral_tokens.private.jsonl — shared request fields
    plus model_id, model_revision, model_seed, input_row_sha256 with explicit hash
    convention, rendered_prompt_sha256, exact input_token_ids/count/hash, mask
    policy (one per unpadded token),32768total/512output controls, context-fit flag,
    preparation_status, full-text/truncation flags, activation_lock_sha256:null
    and call_id:null. These are requests, not response envelopes or model labels.
  census.json — compact per-model counts, tokens, overflows, tokenizer identity,
    timing and exact future scheduling order.
  input_bindings.json — raw input/code/tokenizer/runtime hashes and symlink targets.
  result.json — internally sealed completion, artifact bindings, external raw
    selection/review/lock/frame bindings, costs and explicit scope flags.

The list above groups the two token files on one line: its count is8.
Canonical requests appear first in frozen sample order; selected swaps then appear
in the committed sorted sample-index order. Every model's schedule sorts descending
input tokens, ascending selection_index, then canonical before diagnostic when the
same sample has an exact token-count tie. The future runner consumes the explicit
scheduled_request_indices; it must not re-sort by a different tie rule.

Text reconstruction for future inference uses shown_calibration_row_indices into
texts.private.jsonl, verifies shown record IDs/text hashes, then builds exactly
{pair_id:opaque_pair_id, orientation, text_a, text_b} for the qualified renderer.
The model-specific seed is copied/verified from protocol identities. No final call
ID exists until a separate new activation lock is sealed: call IDs depend on that
lock's raw hash. This avoids a request/activation self-reference cycle.

All canonical pairs and exactly ceil(n/10) committed diagnostic swaps are retained
for BOTH tokenizers. A request with input_tokens+512>32768 remains present with
preparation_status=context_overflow. It will be unavailable/y=0 with its complete
sampling weight under the later v3 statistics, never filtered, shortened or
replaced. Fitting requests are ready_for_future_activation, not inference-ready by
this result alone. Only selected requests are censused; full-frame context
feasibility remains explicitly unchecked.

Bounded publication and failure
Fresh private output directory under WORKSPACE/qualification, mode0700, is required.
No overwrite/resume is supported. Output is limited to64MiB total, reserving1MiB
for small final/failure metadata. Token ledgers stream one request at a time; only
compact census metadata remains in RAM. Texts are not duplicated per model and
rendered prompts are represented by hashes, not an extra full-text copy.
Unexpectedly long texts/tokens or any changed input fail the entire preparation;
partial artifacts and failure.json remain, and no result.json is published. The
64MiB cap is not permission to truncate/drop rows. A separate stage controller
maintains a128MiB total allowance and the unchanged8GiB free-space reserve.

Finalization hashes all watched inputs/code/tokenizer/runtime files and published
payloads, rechecks source/frame/snapshot/model/lock/selection inventories, and only
then writes result.json. Input preparation, tokenization and final recheck timing
are separate; result records elapsed time BEFORE its own publication. The external
controller must additionally record child start/exit elapsed time and result/output
byte total, including final serialization, filesystem sync and process shutdown.
These timing layers overlap; do not add them. No tokenization-speed projection of
GPU inference walltime is supplied.

CLI (all paths are explicit, no remote work by this package)
  PYTHON -B prepare_requests.py
    --repo REPO --workspace WORKSPACE --package-sha256 OWN_MANIFEST_SHA
    --sampler-package SAMPLER_DIR --lock-package SAMPLING_HELPER_DIR
    --protocol-package PROTOCOL_DIR --protocol-evidence HISTORICAL_SNAPSHOT_DIR
    --frame-package FRAME_ADAPTER_DIR --frame-dir ACTUAL_FRAME_DIR
    --selection-result ACTUAL_RESULT --selection-result-sha256 RAW_RESULT_SHA
    --selection-review ACTUAL_REVIEW --selection-review-sha256 RAW_REVIEW_SHA
    --sampling-lock ACTUAL_LOCK --sampling-lock-sha256 RAW_LOCK_SHA
    --qwen-model QWEN_SNAPSHOT --qwen-assets-manifest QUALIFIED_QWEN_MANIFEST
    --mistral-model MISTRAL_SNAPSHOT --mistral-assets-manifest QUALIFIED_MISTRAL_MANIFEST
    --out FRESH_PRIVATE_OUTPUT
Other package/snapshot/frame/source and asset manifest hashes are pinned inside
expectations.json; there are no caller-overridable policy or threshold options.

The root-owned llm_annotation_request_control_v2_20261014 package stages/submits
this CPU job only after actual full selection passes independent review. This
package does not submit jobs or add a second Slurm wrapper.

Software checks (no production assets/models/entropy)
  PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover
    -s local_run/llm_annotation_requests_v2_20261014 -p test_requests.py -v
For a separately staged package, set THIRD_REQUEST_REPO to the actual executing
repository, THIRD_REQUEST_PROTOCOL to the staged sealed protocol directory and
THIRD_REQUEST_SAMPLER_PACKAGE to the staged sealed sampler directory. Tests use
synthetic texts/vectors and a clearly fake tokenizer; renderer code itself is the
actual unchanged qualified code. This verifies software contracts, not real-model
inference, real-asset tokenization parity, activation, or scientific results.
