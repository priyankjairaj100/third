CPU postprocessing control, prospective 14 October 2026.

This package runs the already sealed integration audit.py and ingest.py without
editing those sources, changing the activation/resource plan, loading weights,
generating labels, or retrying calls. It is not an annotation activation writer.
All family artifacts are read-only. Reports and logs have separate fixed roots:
  qualification/llm-only-v3-output-audit-qwen-a01/run/report.json
  qualification/llm-only-v3-output-audit-mistral-a01/run/report.json
  qualification/llm-only-v3-selection-statistics-a01/run/report.json
These paths are beneath the existing cluster local_workspace. The original
inference output inventories are never extended with postprocessing reports.

Run control.py with --mode qwen, mistral, or statistics and the external raw
SHA256 arguments --package-sha256, --resource-plan-sha256, --admission-sha256,
and --activation-sha256. Family mode also requires --result-sha256; the literal
absent is allowed only when the terminal family's result.json does not exist.
Statistics requires --qwen-result-sha256, --mistral-result-sha256,
--qwen-audit-sha256 and --mistral-audit-sha256 instead. No default result pins or
mutable latest pointers are accepted. There is no overwrite, resume or retry.

The login preflight verifies the sealed operational control/helper chain and
all admitted packages, request/activation/resource metadata, exact family child
commands, launch/process receipts, raw result pins, and fresh terminal Slurm
accounting. Existing stages or active Third jobs are refused. A fresh strict
quota/mount check admits 32 MiB above the unchanged 8 GiB reserve per one-shot
postprocessing job. This is a snapshot, not a filesystem reservation. A durable
exclusive intent precedes one sbatch; ambiguous submission is retained for
manual reconciliation without automatic retry. No remote operation was run to
develop or test this package.

Slurm requests csis.cn1, CPU-only 4 CPUs, 8 GiB and 20 minutes, with CUDA hidden,
offline tokenizers and one numerical thread. Twelve software tests run first.
The exact sealed integration CLI then runs in a fresh child. Combined console
capture is capped at14 MiB, tests at1 MiB, accepted report at8 MiB, and a failed
partial report at16 MiB through inherited RLIMIT_FSIZE. The 32 MiB allowance
also covers compact receipts and quiet Slurm logs. An unexpected write/cap
failure is preserved; it does not become an accepted result. Child wall time
is measured through exit before log flushing; CPU/RSS timings include auditing,
not model inference. Initial input/result/package checks are repeated afterward.

COMPLETED0 or FAILED2 family runs with the matching process code and no console
cap failure may enter the unchanged artifact auditor. Ordinary runtime_error
can therefore retain all planned not_attempted rows and still pass provenance.
All terminal failures can be inspected, but timeout/integrity/missing-output
runs are quarantined: a failed postprocess result/log is retained and cannot
feed statistics. A zero child exit cannot override an ineligible terminal run.

Statistics additionally binds the two successful postprocessing jobs, their
exact commands/receipts and externally pinned sealed audit reports. It joins
both complete90-pair model panels; invalid, overflow, runtime_error and
not_attempted outcomes retain their original statuses and denominator. The9
swaps per model remain diagnostics. A passed statistics computation can report
a failed threshold criterion. It does not assert semantic truth, independent
logit replay, publisher authenticity, human validity or confirmatory readiness.
There are zero human annotation calls and no future-validation activation.
