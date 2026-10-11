Additive SciPy import repair, version 2, 2026-10-11

Version 1 and failed repair job 13370 remain unchanged. That attempt stopped
before installation (zero pip attempts) because setuptools lists its own
RECORD plus 16 vendored RECORD files. Version 2 reads the owning distribution's
metadata directory using CPython 3.13.7 PathDistribution._path, validates its
own METADATA Name and Version, then hashes only its own raw RECORD bytes.
This deliberate private API use is confined to the fixed Python runtime.
An absent owner RECORD stays explicit null; there is no vendored fallback.
The new regression fixture covers multiple vendored RECORDs, owner changes,
metadata mismatch and absent-owner handling. It runs before repair in Slurm.

Purpose: CPU shard qualification job13366 stopped BEFORE any model forwards
because importing frozen phase6.acceptance required missing SciPy. Preserve
that failed attempt. This package installs only scipy==1.17.1 into the existing
encoder-cpu environment, then checks that import. It does not rerun models,
source processing, acceptance routines, qualification, or dispatch jobs.

Official pinned source:
https://pypi.org/pypi/scipy/1.17.1/json
Published requirement: Python>=3.11 and numpy>=1.26.4,<2.7. The existing target
is CPython3.13.7/x86_64 and NumPy2.5.3. The exact35,203,952-byte wheel is:
scipy-1.17.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
SHA256:581b2264fc0aa555f3f435a5944da7504ea3a065d7029ad60e7c3d1ae09c5464
The saved official metadata and expectations bind that URL, filename, size,
hash and dependency declaration. Compatible runtime wheel tags are checked.

The before inventory must exactly match the38 observed distribution versions
in expectations.json. SciPy must be absent. Every distribution's version and
RECORD path/hash/size (or explicit absence) are saved before and after. All
preexisting entries must remain identical; only SciPy1.17.1 may be added.
This checks RECORD metadata identity and versions, not a hash of every existing
installed package payload. Existing idna3.20 and all five frozen encoder
versions remain unchanged; no numerical asset inventory is regenerated.

One HTTPS download attempt is made for the exact official wheel, retained in
the private output. Exact size and published SHA are checked BEFORE pip.
One pip install is permitted, using the existing environment's interpreter:
--isolated install --only-binary=:all: --no-deps --no-index --no-cache-dir
--disable-pip-version-check --retries0 --require-hashes --find-links<privatewheel>
--requirement<sealedrequirements> --report<privatepipreport>.
Pip reads only the downloaded local wheel. Its JSON report must show exactly
the pinned SciPy installation with the same wheel SHA. No upgrades, reinstall,
resolver-based dependency changes, fallback versions, source builds or retries
are attempted. A failure preserves the wheel/partial file, logs, inventory and
result. There is no automatic rollback or cleanup of installation evidence.

The unchanged frozen e.installed_versions() is checked before and after.
phase6.acceptance must import from this repository, SciPy from encoder-cpu,
and the pinned repository source bindings must remain unchanged. The receipt
also inventories repository module files actually imported. Import success
is not execution or passage of the original acceptance route.

The sealed shard core's numerical fingerprint does not cover every installed
distribution. The parent must explicitly bind this repair result in subsequent
launch provenance and use a NEW qualification attempt. No existing core,
dispatch, asset manifest, frozen source, or failed job13366 evidence is edited.
Scientific readiness and full-population cache acceptance remain false.

Four positional Slurm arguments: repo workspace NEW_output_directory packageSHA.
Output must be new and below workspace/qualification. Example after review:
  sbatch --output="$run_root/repair-%j.out" --error="$run_root/repair-%j.err" \
    "$repo/local_run/e5_scipy_import_repair_v2_20261011/repair.sbatch" \
    "$repo" "$workspace" "$run_root/repair" "$reviewed_package_sha"
Create run_root first, but do not create its repair child directory.
The job requests cpu-short,1CPU,4GiB,10minutes, excludescsis.mn1 and uses
--propagate=NONE. It checks LinuxSlurm context and exactencoder-cpu interpreter.
All created evidence uses private permissions. No SSH or scheduler submission
is performed by the package itself.

Validation before execution is Python syntax, two small RECORD ownership
fixtures, shell syntax and independent review. No installer test or local
environment mutation is performed. Only an actual passing result.json can
establish completion of this repair.
