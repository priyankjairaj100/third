# Full AskUbuntu E5 token/chunk census v1

A read-only resource-planning wrapper over the unchanged frozen `chunk_inputs`.
It streams all 140,250 prepared rows through the existing fast E5 tokenizer.
No model constructor, forward, embedding, download, threshold, or data filtering
is invoked. This is a complete tokenization census, not cache or study acceptance.

The exact passed source replay result, three ASCII-view files, existing asset
manifest/seal, immutable E5 revision, published weight bytes, pinned CPU library
versions, Python version, and frozen chunker/dependencies must match. All used
source/code/receipt/package files and complete model/tokenizer asset inventories
are checked initially and finally. Weight files are read for hashes only, never
loaded into a model. Complete row/text/ordered-ID digests and partition counts
must match the source frame. Legal Unicode separators use physical-LF JSONL.

The tokenizer uses `local_files_only=True`, `trust_remote_code=False`, and
`use_fast=True`. The qualified capacity is 512, fixed padding is 261, content
chunks are 256, and future batches contain at most eight chunks of one record.
The census preserves the original full-text chunk contract, including prefix,
specials, all chunks, masks, and token-ID digest. It makes no GPU substitution.

Private outputs are a row ledger with ID/text hash, complete frozen token/chunk
details and per-record batch/padding costs, plus compact aggregate histograms,
maxima, totals, source bindings, result, and process-completion receipt. Raw text
and token IDs are not copied into the ledger. Aggregate payload sizes distinguish
the future FP32 matrix and calibration gather from metadata/working memory.
Complete encoding still requires model forwards and a second independent full
cache replay; tokenization speed is never projected into encoding walltime.

The driver measures validation, tokenizer loading, complete scanning, ledger
flush/fsync/hash, final asset/input/code checks, and durable aggregate/binding
writes. Some inner timers are nested and must not be summed. The Slurm envelope
measures the entire child through its final result write and interpreter exit;
it explicitly excludes only its own small completion-receipt write. Scheduler
elapsed time includes the package check, tests and envelope. Preserve all three
levels rather than reporting a favorable partial timing.

## Launch

Six positional arguments: repository, workspace, explicit existing AskUbuntu
ASCII view directory, exact passed source-replay result, new private run root,
and expected census package-manifest SHA256. The script always uses
`$workspace/envs/encoder-cpu/bin/python`; it does not use the source-replay base
Python, install dependencies, reseal assets, or alter their runtime pins.

```bash
umask 077
mkdir -p "$workspace/qualification"
mkdir "$run_root"
sbatch --output="$run_root/slurm.out" --error="$run_root/slurm.err" \
  "$repo/local_run/e5_token_census_v1_20261011/census.sbatch" \
  "$repo" "$workspace" "$view" "$replay_result" "$run_root" "$package_sha256"
```

Resources: CPU only, two CPUs, 48 GiB, one hour, excludes `csis.mn1`, uses
`--propagate=NONE`, and checks effective memory limits. A missing success result,
timeout, nonzero exit, changed binding, or incomplete ledger is not completion.
Preserve failed attempts; no automatic retries/resume or partial-population pass.

## Software tests

Tests use fake tokenizers and tiny labeled software fixtures only; they perform
no model loads, downloads, or scientific inference.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s local_run/e5_token_census_v1_20261011 -p test_census.py
bash -n local_run/e5_token_census_v1_20261011/census.sbatch
```
