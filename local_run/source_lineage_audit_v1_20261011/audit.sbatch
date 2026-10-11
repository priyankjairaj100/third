#!/usr/bin/env bash
#SBATCH --job-name=third-source-lineage
#SBATCH --partition=cpu-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --exclude=csis.mn1
#SBATCH --propagate=NONE
set -euo pipefail
umask 077
repo=$1
workspace=$2
run_root=$3
audit_python=${4:-python3}
seven_zip=${5:-7z}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TMPDIR="$run_root"
"$audit_python" - "$repo" <<'PY'
import hashlib, json, os, resource, socket, sys
from pathlib import Path
if not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('Slurm allocation required')
if socket.gethostname().split('.')[:2] == ['csis', 'mn1']:
    raise SystemExit('Login node prohibited')
limit = resource.getrlimit(resource.RLIMIT_AS)[0]
if limit != resource.RLIM_INFINITY and limit < 12 * 1024**3:
    raise SystemExit('Inherited address-space limit below 12 GiB')
package = Path(sys.argv[1]) / 'local_run/source_lineage_audit_v1_20261011'
manifest = json.loads((package / 'package_manifest.json').read_bytes())
for name, expected in manifest['files'].items():
    if Path(name).name != name:
        raise SystemExit('Unsafe package member')
    data = (package / name).read_bytes()
    if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
        raise SystemExit('Package code binding mismatch')
PY
"$audit_python" -m unittest discover -s "$repo/local_run/source_lineage_audit_v1_20261011" -p test_audit.py
"$audit_python" "$repo/local_run/source_lineage_audit_v1_20261011/audit_source.py" \
  --repo "$repo" --workspace "$workspace" --out "$run_root/audit" --seven-zip "$seven_zip"
