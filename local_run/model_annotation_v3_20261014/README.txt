THIRD LLM-ONLY PROTOCOL V3 — PROSPECTIVE

PLAN.txt is the readable contract; PLAN.json holds exact rules and local bindings.
This separate version follows the user's 14 October instruction: no human
annotation. Original v1/v2 files and locks stay unchanged. Historical human gates
remain unmet; they do not block this explicitly model-only branch.

protocol.py provides only planning checks and small reference helpers for v3
identities/labels. It cannot sample, activate, submit jobs, infer model labels,
accept a cache, or pass scientific gates. Software fixtures are not annotations.
Use the existing environment; no installation or network is needed:
  python3 -B -m unittest discover -s local_run/model_annotation_v3_20261014 -p 'test_*.py'
  python3 -B local_run/model_annotation_v3_20261014/protocol.py --repo .

SOFTWARE_VALIDATION.json records only local protocol checks. package_manifest.json
binds the package members by raw bytes/SHA256 (the manifest excludes itself).
Bound runtime plans/results remain engineering evidence. Activation requires a
new separately reviewed v3 source/cache/frame and inference/statistics pipeline.
No human forms or human-response schema are supplied by this version.
