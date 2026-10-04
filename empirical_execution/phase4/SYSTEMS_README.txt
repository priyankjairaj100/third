FRESH-PROCESS SYSTEMS MEASUREMENT HARNESS
=======================================

Status
------
Executable and software-tested. No empirical method benchmark or paper systems
claim is produced by these command tests. In particular the existing combined
multi-method execution runner must NOT be relabeled a fair per-method timing run.
Actual comparative benchmarking still requires separate method entry points,
fixed matched inputs, trajectory manifests, lifecycle boundaries and a study lock.

Run from repository root:
  python3 -m empirical_execution.phase4.systems SPEC.json \
    --workspace ABSOLUTE_WORKSPACE --destination NEW_DIRECTORY_INSIDE_WORKSPACE

API: run_benchmark(spec, workspace, destination) -> sealed benchmark report.
Existing destinations are refused. Default repetitions=5; only integer 1..5 allowed.
Every repetition uses a fresh process and output directory. At each repetition,
method IDs are sorted then shuffled with Python random.Random(order_seed). The full
sequential order is checkpointed before any timed command starts. Methods never run
concurrently within this harness. This does not exclude other workspace jobs.

Spec shape (required fields):
  evidence_role: software_harness_validation OR development_systems
  order_seed: integer
  timeout_seconds: positive finite per-child elapsed limit
  methods: [
    {
      method_id: unique ASCII alphanumeric/_/- identifier,
      argv: ["/absolute/python", "method_entrypoint.py", "--output", "{output_dir}"],
      input_files: [{path: "method_entrypoint.py", sha256: "64 lowercase hex"}, ...]
    }, ...
  ]
Optional: repetitions (default 5, maximum 5), threads (default 1),
termination_grace_seconds (default 0.5, positive and at most 5).

Commands are structured argv arrays, executed shell=False with stdin closed.
Literal {output_dir} in each argument expands to that run's new artifact directory.
cwd is the declared workspace. Relative declared inputs resolve inside workspace.
Method code must reconstruct initial state for every repeat; a fresh process alone
does not prevent reading a previously mutated state file. Fixed input manifests and
code review remain necessary. This harness cannot establish fair access-policy or
mathematical-contract parity between arbitrary commands.

Pin ALL method code and inputs
------------------------------
Every method entry-point script, imported project-code artifact, data/cache file,
request manifest, solver/configuration file and environment lock that determines
the measurement must appear in input_files with its actual hash. Script filenames
inside argv are only strings; they do not automatically pin their file bytes.
The harness automatically hashes the resolved executable (for Python this is the
interpreter), full expanded canonical argv bytes and its own code. It does NOT
discover every import, library, executable dependency or file opened by a method.
An empty input_files array is useful for tiny python -c software checks and does
not establish a pinned real benchmark. method_code_pinning_completeness_verified
and per_method_comparative_isolation_verified always remain false in this module.

Thread/environment policy
-------------------------
Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS,
VECLIB_MAXIMUM_THREADS and BLIS_NUM_THREADS to the declared thread count;
OMP_DYNAMIC=FALSE, MKL_DYNAMIC=FALSE and PYTHONHASHSEED=0. Other environment values
are inherited but not dumped (they may contain secrets). Libraries that ignore
these variables or custom worker pools need separately pinned command settings.
The report records Python/platform and available CPU affinity without changing it.
It does not modify system privileges, caches, governors or machine configuration.

What is charged and retained
----------------------------
  * spawn_to_reap_elapsed_seconds: process startup through wait4 reporting exit.
  * charged_end_to_end_elapsed_seconds: before input verification/directory setup
    through child lifecycle, post-run input rehashing, output fsync and output
    inspection/hashing. This conservatively includes measurement overhead.
  * child user/system CPU seconds and POSIX wait4 ru_maxrss, converted to bytes
    using the OS convention (Linux KiB; macOS bytes). This is a child high-water
    statistic, not a sum or simultaneous peak of all descendants/parent state.
    Reaped descendant accounting follows OS wait4 semantics. It cannot establish
    whole-deployment peak memory; designated-state bytes are separately measured.
  * Every regular file in the run directory: logical and allocated bytes, hashes,
    including stdout/stderr; method-artifact bytes are also separately reported.
  * Files are fsynced, as are visited directories. Unsupported persistence calls
    make measurement incomplete. This is an OS persistence operation, not a
    guarantee about remote storage hardware or catastrophe recovery.
  * The separate persistence receipt charges benchmark.json bytes/hash and total
    elapsed through its final checkpoint. Receipt self-bytes are explicitly
    excluded to avoid a self-referential count. Run files are separate from the
    harness ledger so self-accounting cannot silently inflate method state.

Wall time must be interpreted with its declared scope. End-to-end charged time
includes input/output hashing and forced persistence; the process-only field does
not. Both are reported, neither is silently substituted for the other. Total
benchmark elapsed includes per-run checkpoints. No page-cache flush is performed,
so a fresh process is not a cold-storage run or a guarantee of unbiased order
effects. Do not claim warm/cold comparisons without an additional declared policy.

Failures stay in the run registry
--------------------------------
Nonzero exits, termination signals, timeout flags, failed executable launches,
incorrect/changed input hashes and incomplete output inspection are all retained.
Timeout sends SIGTERM to the newly created process group, then SIGKILL after the
declared grace. Kernel-uninterruptible processes cannot be promised a hard wall
deadline. A surviving process group after leader exit is killed and marks the run
incomplete; descendants that create a different session cannot be fully monitored
by this harness. No failed run is automatically retried or dropped from counts.
Output symlinks/nonregular files are not followed and make accounting incomplete.

Scope and limits
----------------
This is NOT a filesystem, network or privilege security sandbox. Trusted local
commands retain normal caller permissions and can write undeclared locations.
Reported output bytes cover the declared run directory, not unknown external
writes; code/access review must rule those out for a paper measurement. No memory
or disk ceiling is enforced. Use the prespecified method limits before beginning
large runs. Nothing launches remote compute or changes user accounts. The two
supported roles stay nonconfirmatory: paper_systems_result=false and
confirmatory_study_ready=false in every report.

Verification
------------
  python3 -m empirical_execution.phase4.check_systems
The check uses only short local software commands: successful file writes,
runtime/environment inspection, nonzero exit, signal, sleep timeout, missing
executable, bad input hash, symlink output and input mutation. No natural corpus
timing, synthetic empirical dataset or competing method comparison is performed.
Raw software-command report records and a check ledger are retained in
results/systems_checks.json; temporary output directories are removed afterward.
