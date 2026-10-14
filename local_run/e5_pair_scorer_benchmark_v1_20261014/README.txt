Frozen pair scorer microbenchmark — prospective engineering package

This package does not submit a job. It measures existing frozen scoring code;
it performs no encoder/LLM forward, sampling, threshold selection or human work.
It neither admits the full cache nor alters Phase 3–10 or the v2 annotation plan.
The package may be sealed after independent review; execution needs a separate
root launch decision, fresh capacity check and the resulting manifest hash.

INPUTS AND SCOPE
Expectations bind v2 CPU core 0eb7309e…, dispatch d68d07ca…, qualification 13495
result bb84fc18…, its assembled 19-row FP32 vectors, repair result fc1c5431…,
and actual frozen scorer/protocol files. Full bytes/hashes are in expectations.
The production CLI invokes unchanged common.context/require_qualification and
verify_final. Actual golden/assembled/shard/replay bytes, full source frame,
census, assets and runtime are checked again. It never loads model weights into
an encoder. Existing validators read/hash model files without model forwards.
The separately pinned SciPy repair and current scipy==1.17.1 are checked; this
is not a new audit of every installed environment file.

The 19 natural records were selected by the earlier engineering qualification.
Their 171 unordered pairs are a numerical correctness fixture, not a probability
sample or semantic validation. Sizes 256,260,512 explicitly repeat vector index
[i modulo19], retaining exact index maps and source cache bindings. They are
software shape fixtures, not additional natural records. No body text, natural
pair IDs or output vectors are written by this package.

NUMERICAL CONTRACT
Use unchanged phase3.panels graph_normalize/reference_cosines and calibration
iter_pair_scores. FP32 -> FP64, ordered square sums, square root normalization,
ordered dot products, unclipped raw scores and original boundary audit. Scorer
source hashes are checked. No BLAS dot, alternate reduction, GPU or centering.
A 19-by-19 direct pair-at-a-time table supplies bitwise reference values. Genuine
pair traversal checks B=1,7,19,256. Shape cases use B=256; each has two modes:
score_only, and boundary_and_bins (the original boundary audit plus _bin counts).
Each mode has one warmup and three measured repetitions: 28 cases total.
Every emitted pair is checked against independently enumerated tile traversal
and its exact reference bytes. The SHA256 stream packs <II pair indices then
<d raw score. Validation/collector/hash overhead is INCLUDED in each timing.
Warmups and correctness checks are separately labeled and not benchmark repeats.
There is no reservoir, RNG draw, pair pack or threshold search.

The numerical budget is 300 seconds, cooperatively checked before/after each case
and every 4096 yielded pairs. A single bounded tile may finish before checking;
Slurm's 20-minute process cap is separate. Expiration records an incomplete case
and a failed result; no continuation is allowed. Cases flush/fsync independently.
A complete result requires all 28 cases and final immutable input/code checks.

OUTPUTS AND LIMITS
input_bindings.json records pins/maps before numerical execution; cases.jsonl
records numerical aggregate evidence; result.json is the exclusive final report.
Failed runs preserve available files/counters and never publish passed=true.
A stage-level process_completion.json measures child walltime through exit;
software_validation.json records16 cluster tests with 0 skips. The result timer
ends before its own publication; do not substitute it for process walltime.
Peak RSS is Linux ru_maxrss for the benchmark child, including imports/preflight;
source verification and final hashing times are separate. Tests use a separate
wrapper process and their memory is not attributed to the numerical child.

Full calibration has 20,996 rows / 220,405,510 unordered pairs. At B=256 the frozen
implementation computes 223,102,992 tile entries (diagonal redundancy included),
or 171,343,097,856 coordinate products, across 3,486 tiles. No full-pair work occurs
here. Record measured shape costs; any later scaling is a CONDITIONAL scenario,
not a walltime guarantee. Repeated vectors cannot establish the natural boundary
frequency, bin distribution, reservoir cost or end-to-end calibration cost.
Concurrent load is uncontrolled and recorded as such; no scheduler exclusivity
or reproducible throughput across machines is claimed.

LOCAL CHECKS
Set THIRD_BENCHMARK_REPO to the repo and THIRD_BENCHMARK_QUALIFICATION to the
copied qualification directory containing result.json, assembled/ and golden/.
Run python3 -m unittest discover -s PACKAGE -p test_benchmark.py -v.
These tests need only NumPy and repository Python sources. No installations or
model inference are needed. Actual qualification source/vector byte checks and
pair scoring run locally; full core admission/runtime preflight requires the
cluster's source/model files and pinned encoder-cpu interpreter and is unrun
by this software validation.

PROSPECTIVE ROOT-ONLY LAUNCH INTERFACE
benchmark.sbatch takes exactly 11 positional arguments:
  PACKAGE_FOLDER REPO WORKSPACE VIEW REPLAY_RESULT CENSUS_DIR BUDGET_DIR
  QUALIFICATION_RESULT REPAIR_RESULT NEW_STAGE_ROOT PACKAGE_MANIFEST_SHA256
It creates NEW_STAGE_ROOT, then NEW_STAGE_ROOT/benchmark, exclusively beneath
WORKSPACE/qualification. Root prepares a separate parent for Slurm stdout/err.
Wrapper verifies package bytes before tests/code import, runs exactly 16 tests
with 0 skips, and invokes benchmark.py in a fresh child. It uses the existing
WORKSPACE/envs/encoder-cpu/bin/python, 2 CPUs, 48 GiB, 20 min, cpu-short, no GPU,
exclude csis.mn1,propagateNONE,offline environment and numerical threads 1.
Root should target qualified csis.cn1 for the inherited exact fingerprint;
otherwise preflight fails rather than claiming cross-machine qualification.
The 48 GiB is conservative reuse of existing source-validation capacity; scoring
working arrays are small. No submit/retry/resume/poll code is included.
