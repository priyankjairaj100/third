# AskUbuntu prospective local source-lineage audit v1

This is a new, CPU-only, read-only audit of existing source files. It never
downloads, changes original/derived inputs, edits frozen code, infers external
authenticity from checksums, or grants study/source/cache acceptance. Outputs are
new private metadata only. Frozen Phases 3–10 and prior engineering runs stay intact.

The audit checks the observed archive SHA/size, hashes streamed `Posts.xml` and
`PostLinks.xml` archive-member bytes against the existing extracted files, proves
the entire existing `Posts.angle.xml` equals the historical Tags-only operation,
checks every declared adapter output hash and input/code binding, verifies the
complete physical-LF JSONL to ASCII serialization without changing any object or
row order, verifies the complete sealed prepared population/text/ID frame, and
independently reconstructs native owner groups and split hashes. Every watched
input is hashed again at the end. Passing means these local byte and logical
bindings were observed now, not that a historical collection event or publisher
authenticity was proved. Expected hashes come from previously inspected local
acquisition/adapter/preparation metadata, not a trusted publisher checksum.

## Tests and execution

Tests use only tiny labeled software fixtures and stdlib Python 3.10+:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s local_run/source_lineage_audit_v1_20261011 -p test_audit.py
bash -n local_run/source_lineage_audit_v1_20261011/audit.sbatch
```

Remote launch is **not performed or currently available**: the existing SSH
master failed, so there are no automatic retries, probes, or authentication
attempts following that failure. After the user restores an authenticated
connection and this folder is transferred, the smallest execution is one CPU
Slurm allocation. The existing work authorization remains valid. `7z` (or explicit
path to compatible `7zz`) must already exist; no installation is attempted.
The CPU partition name must be checked against current cluster configuration.
Run the following on the cluster only after those prerequisites are satisfied:

```bash
repo=/nfs_home/users/poonam/ccu-acl2027/repo
workspace=/nfs_home/users/poonam/ccu-acl2027/local_workspace
run_root="$workspace/qualification/source-lineage-v1-$(date -u +%Y%m%dT%H%M%SZ)"
umask 077
mkdir -p "$workspace/qualification"
mkdir "$run_root"
sbatch --output="$run_root/slurm.out" --error="$run_root/slurm.err" \
  "$repo/local_run/source_lineage_audit_v1_20261011/audit.sbatch" \
  "$repo" "$workspace" "$run_root" python3 7z
```

The script uses two CPUs, 16 GiB, a two-hour bound, no GPU, no model/network
access, and a new output directory. Archive-member extraction is stdout-only
with a 30-minute deadline per member. It requires a Slurm allocation and refuses
the known login hostname. Temporary stderr stays under the private run directory.
Source data is streamed; source IDs for uniqueness/splits and the sealed audit
remain resident. The report records elapsed time and peak RSS. It prints no row
text, IDs, or arbitrary parser exception contents. SHA checks require several
sequential passes over existing multi-GB files; this must run on a compute node.

## Explicit gaps

- Full frozen adapter regeneration and fixed-guard/panel preparation replay are
  deferred. Bound output hashes do not themselves prove the transformation ran.
- The earlier prepared ASCII view and receipt are not required or validated by
  this stage; it reads original prepared JSONL safely using physical LF.
- The archive URL/release is historically declared. Publisher authenticity,
  archival completeness, native-link completeness beyond the local member, and
  attribution review remain unresolved. No acquisition receipt is invented.
- Original `Posts.xml` has no independently recorded expected hash in the
  inspected metadata. This run establishes its observed equality to the local
  archive member, not an externally authenticated historical digest.
- Full frozen CPU E5 cache, semantic guard, two-model scientific annotations,
  blinded author audit, and original human-calibration gates remain separate.

`result.json` and immutable per-stage JSON report successes completed before a
failure. Final success requires end-of-run equality of all watched file bytes.
An externally killed job may have stage receipts but no final result; that is
incomplete, never success. Do not rerun into the same output directory.
