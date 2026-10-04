# Phase 6 isolated methods and measurements

This implementation preserves the Phase 5 source files and results.
The workers use the unchanged numerical gate from `phase5/decoders.py`.
The gate requires a normalized residual of at most `1e-10`.
A failed gate stops that worker.
The worker does not change its tolerance or fetch missing payload.

## Methods

The primary methods remain O-G, O-T, B-E, B-A, P-I, P-S, P-R, and B-F.
The existing joint-span and compact-payload variants remain available.

B-F stores only the payload of the initial selection.
It removes requested records or source units from that selection.
It cannot admit previously excluded records.
Its target therefore remains the declared wrong target.
The evaluator must compare B-F with that target for implementation correctness.
It must compare B-F with counterfactual retraining for scientific effects.

O-G-cached is a named sensitivity with the O-T selection kernel.
It stores the complete fixed graph and all learner payload.
It rebuilds the learner moments from the surviving selection.
It does not repeat scalar pair scoring after each request.
The shared bundle records graph construction cost.
This tier does not supply a distinct graph algorithm.
Its state permits greater access than a summary state.

## FP32 variants

Append `-FP32` to any primary method name.
These eight variants retain the same curator contract.
O-G-FP32 still uses the fixed FP64 scalar curator scorer.
Only learner state and learner arithmetic change.
Every variant promotes its stored moments to the common FP64 decoder.
The decoder gate tests the promoted stored system.
It does not establish agreement with fresh FP64 training.
Report the actual head difference as an accuracy result.

| Variant | Persistent learner values | Update or reconstruction arithmetic |
|---|---|---|
| B-E-FP32, B-A-FP32 | FP32 features and moments; FP64 original targets | FP32 signed products; targets cast before each product |
| P-I-FP32, P-S-FP32 | FP32 polynomial coefficients | FP32 products, coefficient additions, and cancellations |
| P-R-FP32 | FP32 rank coordinates | FP32 products, coordinate additions, and omitted-coordinate reconstruction |
| O-G-FP32, O-T-FP32 | FP32 learner features and targets | Fresh FP32 moment products |
| B-F-FP32 | FP32 features and targets from the initial selection | Fresh FP32 moment products on its own target |

Construction rejects populations with at least `2**24` records.
This guard protects exact integer counts in FP32 coefficient state.
It does not protect model statistics from rounding error.
The FP32 snapshot schema identifies each precision variant.
The summary loader temporarily promotes coefficients for the existing structural validator.
That temporary allocation contributes to load time and resident memory.
The loader restores FP32 coefficient storage before restriction.
Empty coefficient matrices also use FP32 storage.

## Service panels

Use `phase6.run_isolated.run_service` with the existing Phase 5 arguments.
The new `panel` argument accepts three values.

| Panel | Behavior |
|---|---|
| `cold_process` | Fresh construction process and fresh repair process; no extra numerical warmup |
| `warm_service` | The same fresh processes, with fixed library warmup before construction and before repair |
| `every_release` | Fresh processes, with a snapshot and `fsync` after every release |

The warmup uses a fixed identity Cholesky solve and matrix product.
Those arrays are a software warmup, not empirical observations.
The worker records warmup time.
Full lifecycle cost includes that time.
Each service starts from a newly constructed state.
No repetition reuses a state changed by an earlier trajectory.
The repair process remains alive across the requests within one trajectory.
The warm panel does not claim a cold or warm filesystem cache.

The primary persistence policy writes one final state snapshot.
Every head release is still written.
The sensitivity writes and synchronizes each state snapshot.
Serialization buffers, writes, synchronization, and construction transfer remain charged.
The process cap and thread count apply to every method.
The caller must lock their values from genuine development evidence.

## Memory and byte definitions

The service records several different quantities.
These quantities must not be substituted for each other.

| Field | Meaning |
|---|---|
| `unique_ndarray_root_buffer_bytes` | Sum of distinct array buffer extents reachable from state |
| `ndarray_views_logical_bytes` | Sum of reachable array view extents, including aliases |
| `canonical_metadata_utf8_bytes` | Exact length of the defined JSON metadata representation |
| `full_serialization_buffer_bytes` | Size of complete output buffers produced by serialization |
| `compaction_copy_output_bytes` | Known advanced-index and copy materializations in the stated compaction branch |
| `payload_input_value_bytes` | Feature and target value bytes for counted signed-update rows |
| `gather_and_precision_copy_output_bytes` | Bytes written into the stated payload gathers and precision conversions |
| `returned_binary_read_bytes_by_category` | Actual Python binary read results from the worker's own NPZ load |
| `read_call_bytes` | Process read-byte counter change, with the initial meter read removed |
| `kernel_storage_read_bytes` | Kernel storage-read counter change |
| `output_write_call_bytes_before_report` | Sum of successful explicit output writes before the report |
| `sampled_process_tree_peak_rss_sum_bytes` | Largest sampled sum of RSS across the observed worker process tree |
| `peak_rss_bytes` | The operating system's process high-water RSS from `wait4` |

The metadata representation includes type, array, and alias descriptions.
Its size is not Python heap size.
Its size also does not measure metadata access traffic.
The array inventory does not measure allocator slack.
The compaction counter includes known intermediate copies.
It excludes other temporary arrays and uninstrumented metadata operations.
The full lifecycle still includes inventory and reporting overhead.

The NPZ read meter splits disjoint byte ranges into four categories.
These categories cover payload values, statistic values, metadata values, and serialization format bytes.
Repeated reads count repeatedly.
The meter excludes hash reads, text sidecars, and unobserved C-level reads.
It supports uncompressed snapshot members.
It does not claim to measure storage traffic or memory traffic.

The parent samples `/proc` at intervals of at least 20 milliseconds.
It resolves process identifiers across the observed PID namespace boundary.
Each sample includes the root worker and its observed descendants.
The scan is not atomic.
It can miss short processes and peaks between samples.
Shared mappings count once per process.
The sum therefore does not measure unique physical pages.
The coordinator's memory remains outside the worker tree.
Per-process high-water values are never summed into a purported simultaneous peak.

No field measures hardware cache traffic or exact allocator ownership.
Those quantities remain unclaimed.

## Dependency scope and isolation

The worker hashes imported module files and mapped ELF files.
The report states the time and scope of this observation.
This observation does not prove complete build or distribution dependency closure.
A library loaded after that boundary can be absent from the manifest.
The repair restriction denies new file opens after state loading.
It also denies process inspection, sockets, IPC, and relevant descriptor-transfer routes.
The existing seccomp TSYNC checks cover all current threads.
A real forbidden-file probe must fail.
The restriction does not prove physical erasure or security against arbitrary malicious native code.

## Forecasts and observed failures

An optional forecast contains `forecast_peak_bytes`, `development_evidence_sha256`, and `scope`.
The caller must authenticate that evidence through its accepted resource dossier.
The worker checks the forecast schema and records it before execution.
A forecast above the cap produces `forecast_infeasible` before any child starts.
It does not produce an observed OOM label.

A Python allocation failure produces `observed_allocation_failure`.
A wall timeout produces a censored lower bound.
Other exits or signals retain their own status.
A bare kill signal does not establish an OOM.
No failure silently changes the model, dimension, access contract, or decoder.

## Verification and preserved history

Run:

```sh
OPENBLAS_NUM_THREADS=1 python empirical_execution/phase6/check_workers.py \
  --destination empirical_execution/phase6/results/new_worker_check
```

The check uses 20 existing Civil Comments records and their original toxicity labels.
It tests lexical dimensions 16 and 768.
A lexical dimension of 768 is not a semantic encoder.
The check uses all method variants, three service panels, and both common solvers.
An independent dual solve checks the FP64 heads.
FP32 head differences remain reported accuracy values.
Every released head must pass the unchanged stored-system gate.

The first run exposed a PID namespace error in the RSS sampler.
Its 58 services passed their head checks.
The sampler returned empty process sets and failed the measurement check.
That run remains in `results/worker_engineering`.
The second run passed 98 services before the final empty-snapshot and read-meter additions.
It remains in `results/worker_engineering_release_v2` as development history.
The final source-bound result belongs in `results/worker_engineering_final/verification.json`.
No result starts the primary semantic study.

## Final observed result

The final integration passed 1,708 checks across 98 isolated services.
Those services produced 196 restricted checkpoint releases.
The matrix includes every primary FP32 variant at lexical dimension 768.
The largest FP64 head difference from the independent dual solve was `1.4576717199913652e-13`.
The largest FP32 head difference was `9.50639713690056e-7`.
The FP32 value describes approximation error.
It does not pass a relaxed exactness test.
The unchanged residual gate accepted each released stored-system solution.

Eight additional full-deletion checks verified snapshot restoration for every primary FP32 variant.
Each check compared the count, Gram matrix, and cross matrix after restoration.
The explicit forecast-refusal check started no worker.
It did not report an observed OOM.
That forecast was a marked software fixture, not measured development evidence.

Verbose service files are archived with complete file manifests.
The final summary remains available at `results/worker_final_summary.json`.
Restore the final 1,571 files with:

```sh
python3 tools/phase6_result_archive.py restore worker_engineering_final
```

The restoration tool verifies each file.
Use `worker_engineering` or `worker_engineering_release_v2` to restore the historical directories.
The final directory contains 29,496,461 uncompressed bytes before archive packaging.
The historical directories contain 46,540,559 uncompressed bytes across 2,502 files.

Each retained service record includes head files, configuration, counters, hashes, and failure logs.
Large transient state snapshots were removed after their accounting and hashes were recorded.
Replay the frozen check to reconstruct those transient states.
The archived file manifest describes the retained files exactly.
The process measurements came from shared engineering infrastructure.
They do not establish quiet-host primary timing results.

## Corrected wrong-target evaluation

Independent review found one evaluator branch error after the final run.
The original evaluator treated B-F-FP32 as a counterfactual learner.
Its service always used the correct B-F baseline target.
The corrected evaluator now uses retained original selection for both B-F variants.

`results/worker_head_target_recheck.json` records the correction.
The recheck passed 857 checks across 294 saved heads.
Those heads include 98 initial heads and 196 checkpoint heads.
The maximum head differences stated above did not change.
The recheck records both evaluator hashes and the unchanged worker source hashes.
The archived final directory remains unchanged.
The old evaluator bytes remain in `results/worker_evaluator_before_target_fix.txt`.

After restoration, reproduce the recheck with a new destination:

```sh
OPENBLAS_NUM_THREADS=1 python empirical_execution/phase6/recheck_worker_heads.py \
  --directory empirical_execution/phase6/results/worker_engineering_final \
  --destination empirical_execution/phase6/results/new_worker_head_recheck.json
```

The recheck does not rerun any service.
It does not replace the original measurements or create an independent empirical replication.
