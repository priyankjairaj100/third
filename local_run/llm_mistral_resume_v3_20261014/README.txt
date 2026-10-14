Bounded continuation after Qwen audit13623 and explicit GPU-worker launch repair.

The stopped v2 controller and empty Mistral launch receipt are retained unchanged.
Mistral had no remote intent/job when the original CPU-worker predecessor check
rejected successful Qwen13574 on GPU worker csis.cn3. A separately sealed explicit
adapter fixes only this job-role check and additionally requires exact passed
Qwen audit artifacts. See ../llm_mistral_launch_adapter_v2_20261014/README.txt.
All original remote inference, audit and statistics packages remain unchanged.

The active local controller is run_mistral_resume_v3.py in the original task's
work/resume-20261014/sampling_v1. It owns state mistral_resume_v3_a01 and uses a
new Mistral launch-receipt filename. The v2 family reviewer only adds an explicit
launch-receipt argument so the original zero-byte failure is not overwritten.
It never repeats Qwen or its audit. It launches Mistral once, exports/reviews it,
executes its CPU audit, then computes and independently recomputes frozen exact
selection statistics. A failed selection is preserved without automatic validation.
No human annotations, model retries, new seeds, new activation or scientific
threshold changes are allowed. The original one-shot coordinator is imported;
its120-second local polling, ten-hour bound and stop-on-transport-failure apply.
This is not a recurring automation. No additional GPU is used for orchestration.

Read intent.json, events.jsonl and completion.json/failure.json before acting.
Never restart occupied v1/v2/v3 controller folders or duplicate a stage. Local
resumption checks accepted the cleared state, rejected each of three occupied
remaining slots, and rejected a Qwen repeat. Adapter seven-case tests passed
both locally and on the cluster before launch. These are engineering checks.
