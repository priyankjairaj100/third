Finite v4 selection and conditional fresh validation controller, execution v2.

The actual running private controller hash is recorded in SOURCE_BINDINGS.json.
This portable backup replaces only local root bindings and helper-source paths
and hashes. Set THIRD_TASK_ROOT and THIRD_LOCAL_REPO to the restored paths.
It uses work/resume-20261014/node_pool_v2_pipeline_a01 as an exclusive run root.
Restore remote_existing_session.py to work/resume-20261014 only after reconciling
the existing controller, stage roots, intents, scheduler jobs and outputs.
The transport requires an explicitly authenticated SSH master saved privately in
agent_control_path.txt alongside that helper. No credentials are included.

Do not start this backup while the original controller is active, or delete an
occupied root to restart it. Failure stops without retry; uncertain submissions
must be reconciled. Existing stage outputs and failed judgments are immutable.
Local advancement requires the Mac and its authenticated connection to stay up.
This process covers calibration only, not the full repair/utility/cost study.

All annotation is by Qwen and Mistral. No human calls or human accuracy claims.
Selection failure skips validation. Validation retains the committed fresh seed,
maximum 440 calls, exact finite-population bound, and overlap stability checks.
