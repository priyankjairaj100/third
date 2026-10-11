THIRD — LOCAL MODEL ANNOTATION TECHNICAL PILOT

This runner implements the bounded technical pilot from the separately frozen
model_annotation_v2_20261011 plan. It does not implement threshold selection,
human collection, model-panel validation, or confirmatory study execution.

Preparation selects 40 unique unordered native pairs by fixed SHA256 ranking
from the existing 420-pair AskUbuntu mean-removed engineering selection pack.
Four independently hash-selected pairs also receive reversed-order diagnostics.
These are 44 scheduled calls on 40 pairs, not 44 independent observations.
The engineering pack and earlier subset judgments are exposed development
history. No population semantic precision or accepted CPU-cache claim follows.

Texts are reconstructed in full from the original calibration records. The
source pack, records, selected identities, orientations, input bytes, plan,
prompt, driver and cached model assets are bound before the first inference.
Official pinned Hugging Face metadata supplies expected LFS SHA256 values and
ordinary Git blob identities. All required files must match before loading.

Use pilot.sbatch under Slurm on one A100 80GB with 48GiB host memory. The script
requests --propagate=NONE and verifies the address-space limit on the compute
node. Pass repository root, private workspace, pinned model snapshot, new run
root, and dedicated judge Python as its five positional arguments. Set Slurm
stdout/stderr under that private run root when submitting. Never reuse an
existing preparation or inference output directory.

The dedicated judge venv can reference unchanged encoder-cuda dependency
packages through a read-only .pth file. This avoids duplicating large CUDA
packages while keeping separate interpreter configuration and a sealed judge
runtime manifest. It is not an independently installed dependency tree. No
packages in either encoder environment are installed or modified by this run.

Keep source texts, mappings, raw model responses, runtime logs and full manifests
inside ignored local_workspace/. Only code, public model hash metadata and a
compact source-bound result checkpoint belong in the repository. Outputs have
model provenance; no genuine human responses are produced by this program.

This folder is a new version outside frozen Phases 3–10. Original human gates
and all scientific study readiness checks remain unchanged and unmet.
