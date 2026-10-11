# AskUbuntu frozen source replay, prospective wrapper v1

This stage reruns the unchanged frozen AskUbuntu XML adapter and population
preparation. It requires an explicitly supplied successful source-lineage result,
the expected SHA256 of that result and its package manifest, and an existing
sealed ASCII prepared view with its transport receipt. It refuses missing inputs,
changed bindings, failed or partial lineage results, and existing output paths.
It does not download, encode models, run a token census, edit frozen files, or
claim publisher authenticity, full-cache acceptance, human evidence, or study
readiness.

## Exact checks

The wrapper verifies its own package and frozen source files, the complete
lineage package, every recorded original lineage input, and every lineage stage
receipt. The earlier lineage result binds its executed audit driver and
expectations; it does not record every package file. Those two recorded bindings
must match the expected package. Current whole-package verification is separate.
The lineage result's SHA is supplied from the independently checked result.

The prepared view must bind original and derived files/seals, converter bytes,
and unchanged original audit semantics. Every original prepared JSONL object and
its order are checked against the exact ASCII transport again. No new transport
view or fabricated receipt is created.

The unchanged `phase4.adapters.adapt_stack` writes only new private scratch.
Every rebuilt adapter output, audit setting, count, and metadata field must match
the original. This wrapper also requires exact SQLite identity-index bytes;
a runtime-dependent SQLite difference is a refusal requiring investigation,
never an automatic weaker fallback or proof that corpus rows differ. The
unchanged adapter verifier then runs.

The rebuilt rows are read using physical LF, preserving legal Unicode separators
inside JSON strings. This versioned caller avoids `str.splitlines` in the frozen
acceptance caller; the frozen adapter and preparation themselves remain unchanged.
The unchanged `phase3.panels.prepare_population` uses the full rebuilt population,
all native duplicate links, pinned design, target 10000, and no preview sources.
Every prepared row/order, panel ID, and every generated preparation-audit field
must equal the original. The existing view must additionally pass the unchanged
frozen prepared-record loader and equal the complete replayed population.
All watched input, source, receipt, package, and code bytes are checked again at
the end. A successful final result is required; partial stage receipts never count
as completion. Successful replay proves derivation from the supplied local bytes,
not external source origin, historical execution, completeness, or attribution.

## Launch parameters

After the preceding lineage job has completed successfully, supply these **nine
positional arguments** to `replay.sbatch`:

1. Repository absolute path.
2. Private workspace absolute path.
3. Existing AskUbuntu ASCII view directory (contains records.jsonl, audit.json,
   receipt.json); no default and no reconstruction if absent.
4. Completed source-lineage `result.json` absolute path.
5. SHA256 of that exact result JSON, from the verified preceding job.
6. Expected source-lineage `package_manifest.json` SHA256, recorded for its launch.
7. Newly created private run root under workspace/qualification.
8. Existing Python executable with NumPy and idna; no automatic installation.
   Prefer the original source/preparation runtime (`/nfs_home/software/miniconda/bin/python3`,
   currently reported as Python 3.13), after dependency availability is checked.
   This is separate from the later pinned CPU embedding runtime.
9. Expected replay `package_manifest.json` SHA256, recorded for this launch.

```bash
umask 077
mkdir -p "$workspace/qualification"
mkdir "$run_root"
sbatch --output="$run_root/slurm.out" --error="$run_root/slurm.err" \
  "$repo/local_run/source_replay_v1_20261011/replay.sbatch" \
  "$repo" "$workspace" "$view" "$lineage_result" "$lineage_result_sha256" \
  "$lineage_package_sha256" "$run_root" "$replay_python" "$replay_package_sha256"
```

The bounded job requests two CPUs, 48 GiB, two hours, excludes the known login
node, and disables inherited memory limits. It checks effective address space
before work. Several complete source-file passes, full record lists, canonical
digests, and private regenerated adapter scratch are charged; no constant-memory
claim is made. The corpus text stays only in private scratch and input files.
Normal completion/failure removes temporary adapter scratch; an externally killed
job may retain private partial scratch, which must not be treated as acceptance.
No automatic resubmission is provided.

## Local software tests

Only tiny labeled software fixtures are generated in temporary directories.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s local_run/source_replay_v1_20261011 -p test_replay.py
bash -n local_run/source_replay_v1_20261011/replay.sbatch
```

All result flags for external source authenticity, primary input acceptance,
full-population cache acceptance, and confirmatory readiness remain false.
CPU E5 encoding and independent full-cache replay are later, separate stages.
